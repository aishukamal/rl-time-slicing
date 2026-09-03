# elastic-vd — fork of verl's fully_async_main.py driver @ 983cb0f2 for the
# 32-GPU DAPO elastic showcase (successor of m1/fully_async_main_elastic.py).
#
# What this fork changes vs upstream (everything else copied verbatim):
#  1. ElasticVDTrainer replaces FullyAsyncTrainer (manual FSDP offload lever +
#     resident guard, vd_elastic_trainer.py). ElasticVDRollouter replaces
#     FullyAsyncRollouter (exposes the LB handle + the concurrency lever —
#     same M1 subclass, unchanged rationale).
#  2. SPARE BOOTSTRAP between trainer init and rollouter init:
#       trainer.init_workers  ->  elastic_offload (frees the 16 trainer GPUs;
#       first live exercise of the yield lever)  ->  discover trainer
#       nodes/GPUs  ->  launch 4 x TP4 spares out-of-tree on those GPUs
#       (vd_spares.launch_spare), warm-probe (STANDALONE forces
#       load_format=auto -> real version-0 weights from disk)  ->  pause +
#       sleep L2  ->  elastic_onload  ->  continue stock init.
#     Rationale: vLLM KV budgeting happens at engine INIT, so spares must
#     init while the trainer GPUs are empty; which nodes are "trainer nodes"
#     is only known after Ray places the trainer's resource pool.
#  3. Named detached handles actor "elastic_controller_handles" (namespace
#     "elastic") for the external controller — same as M1.
#  4. VDNodeAgent actors pinned per node (fleet dump/serve + trainer-node
#     prefetch for the wcache pipeline) — registered in the handles actor.
#
# Spare-only config deltas (REGISTERED; the fleet + trainer configs are the
# stock shell's): +enable_sleep_mode=True, gpu_memory_utilization
# VD_SPARE_GPU_UTIL (default 0.70).
#
# Set VD_ELASTIC=0 to run this driver with ZERO elastic machinery (it then
# defers to upstream classes end-to-end) — but the baseline arm should use
# the stock module instead: python3 -m verl.experimental.fully_async_policy.fully_async_main.

import copy
import os
import socket
import sys
import time
from pprint import pprint

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import hydra
import ray
from omegaconf import OmegaConf, open_dict

import verl.experimental.fully_async_policy as _fap_pkg
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local

from vd_elastic_trainer import ElasticVDTrainer
from vd_spares import VDNodeAgent, launch_spare

HANDLES_ACTOR_NAME = "elastic_controller_handles"
HANDLES_NAMESPACE = "elastic"

_FAP_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(_fap_pkg.__file__)), "config")


# ----------------------------------------------------------------------------
# Rollouter subclass (verbatim M1 lineage: LB handle + concurrency lever)
# ----------------------------------------------------------------------------

_ROLLOUTER_BASE = FullyAsyncRollouter.__ray_actor_class__


@ray.remote(num_cpus=10, max_concurrency=100)
class ElasticVDRollouter(_ROLLOUTER_BASE):
    def get_load_balancer(self):
        return self.llm_server_manager.global_load_balancer

    def get_standalone_server_addresses(self):
        return list(self.llm_server_manager.server_addresses)

    def get_fleet_server_handles(self):
        """(address, server_handle) for each in-tree standalone replica —
        wcache dump target selection + invariant cross-checks."""
        out = []
        for rep in self.llm_server_manager.get_standalone_replicas():
            out.append((rep._server_address, rep._server_handle))
        return out

    def elastic_get_max_concurrent_samples(self) -> int:
        return int(self.max_concurrent_samples)

    def elastic_set_max_concurrent_samples(self, value: int) -> dict:
        old = self.max_concurrent_samples
        value = int(value)
        if self.max_required_samples is not None:
            value = min(value, self.max_required_samples)
        self.max_concurrent_samples = value
        self._record_active_count()
        print(f"[ElasticVDRollouter] max_concurrent_samples: {old} -> {value} (elastic override)")
        return {"old": old, "new": value}


@ray.remote(num_cpus=0)
class ElasticControllerHandles:
    def __init__(self):
        self._store = {}

    def put(self, key, value):
        self._store[key] = value
        return True

    def get(self, key, default=None):
        return self._store.get(key, default)

    def keys(self):
        return sorted(self._store.keys())


# ----------------------------------------------------------------------------
# TaskRunner fork
# ----------------------------------------------------------------------------

_TASK_RUNNER_BASE = FullyAsyncTaskRunner.__ray_actor_class__


@ray.remote(num_cpus=1)
class ElasticVDTaskRunner(_TASK_RUNNER_BASE):
    def _initialize_components(self, config) -> None:
        print(f"[VD MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        assert not config.async_training.use_trainer_do_validate, "elastic-vd requires stock use_trainer_do_validate=False"
        assert not config.async_training.get("use_dynamic_resource_scheduling", False), (
            "elastic-vd is OUR elastic; verl-native dynamic scheduling must stay off (stretch arm runs it separately)"
        )

        print("[VD MAIN] Initializing model and tokenizer...")
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        print("[VD MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        handles = ElasticControllerHandles.options(
            name=HANDLES_ACTOR_NAME,
            namespace=HANDLES_NAMESPACE,
            lifetime="detached",
            get_if_exists=True,
        ).remote()
        self.components["handles"] = handles

        # ==== upstream order: trainer FIRST ====
        print("[VD MAIN] Creating ElasticVDTrainer...")
        self._create_trainer(config)
        trainer = self.components["trainer"]

        # ==== spare bootstrap on the (briefly empty) trainer GPUs ====
        if os.environ.get("VD_ELASTIC", "1") == "1":
            self._bootstrap_spares(config, trainer, handles, tokenizer)
        else:
            print("[VD MAIN] VD_ELASTIC=0: skipping spare bootstrap")

        # ==== stock tail (fully_async_main.py:83-115 minus hybrid injection,
        #      which the asserts above rule out) ====
        print("[VD MAIN] Creating ElasticVDRollouter...")
        self._create_rollouter(config)
        rollouter = self.components["rollouter"]

        print("[VD MAIN] Setting up rollouter reference on trainer")
        ray.get(trainer.set_rollouter.remote(rollouter))

        total_train_steps = ray.get(rollouter.get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(trainer.set_total_train_steps.remote(total_train_steps))

        max_queue_size = ray.get(rollouter.get_max_queue_size.remote())
        print(f"[VD MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(rollouter.set_message_queue_client.remote(message_queue_client))
        ray.get(trainer.set_message_queue_client.remote(message_queue_client))

        ray.get(trainer.load_checkpoint.remote())
        ray.get(rollouter.load_checkpoint.remote())

        print("[VD MAIN] Param sync before fit..")
        ray.get(trainer._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(trainer._fit_validate.remote(True))

        # ---- handle registration for the external controller ----
        load_balancer = ray.get(rollouter.get_load_balancer.remote())
        max_concurrent_base = ray.get(rollouter.elastic_get_max_concurrent_samples.remote())
        fleet_servers = ray.get(rollouter.get_fleet_server_handles.remote())
        ray.get(handles.put.remote("rollouter", rollouter))
        ray.get(handles.put.remote("trainer", trainer))
        ray.get(handles.put.remote("message_queue", message_queue))
        ray.get(handles.put.remote("load_balancer", load_balancer))
        ray.get(handles.put.remote("max_concurrent_base", max_concurrent_base))
        ray.get(handles.put.remote("fleet_servers", fleet_servers))
        ray.get(handles.put.remote("init_complete", True))
        print(
            f"[VD MAIN] elastic_controller_handles ready (namespace={HANDLES_NAMESPACE}): "
            f"keys={ray.get(handles.keys.remote())}"
        )
        print("[VD MAIN] All components initialized successfully")

    # ------------------------------------------------------------------

    def _bootstrap_spares(self, config, trainer, handles, tokenizer) -> None:
        t0 = time.time()
        print("[VD MAIN] SPARE BOOTSTRAP: offloading trainer to free its GPUs...")
        off = ray.get(trainer.elastic_offload.remote())
        print(f"[VD MAIN] initial offload: {off['seconds']:.1f}s")

        info = ray.get(trainer.get_trainer_worker_info.remote())
        # info: (pid, node_id, node_ip, cuda_visible_devices, accel_ids) per rank
        by_node = {}
        for pid, node_id, node_ip, cvd, accel in info:
            by_node.setdefault(node_id, {"ip": node_ip, "gpus": set()})
            for a in (accel or ([cvd] if cvd else [])):
                by_node[node_id]["gpus"].add(str(a))
        print(f"[VD MAIN] trainer nodes: { {k[:12]: (v['ip'], sorted(v['gpus'])) for k, v in by_node.items()} }")
        assert len(by_node) == config.trainer.nnodes, f"expected {config.trainer.nnodes} trainer nodes, got {len(by_node)}"

        # Spare-only config deltas (registered): sleep mode + KV budget.
        park_mode = os.environ.get("VD_SPARE_PARK_MODE", "sleepL1")  # sleepL1 | sleep | paused
        spare_cfg = copy.deepcopy(config.actor_rollout_ref.rollout)
        with open_dict(spare_cfg):
            spare_cfg.enable_sleep_mode = park_mode in ("sleep", "sleepL1")
            spare_cfg.free_cache_engine = park_mode in ("sleep", "sleepL1")
            spare_cfg.gpu_memory_utilization = float(os.environ.get("VD_SPARE_GPU_UTIL", "0.60"))

        spares = []
        replica_rank = 100  # far from fleet ranks 0..3; only used for actor naming
        for node_id, entry in sorted(by_node.items(), key=lambda kv: kv[1]["ip"]):
            gpus = sorted(entry["gpus"], key=lambda s: int(s) if s.isdigit() else s)
            assert len(gpus) == 8, f"trainer node {node_id[:12]} exposes {gpus}"
            for half in (gpus[:4], gpus[4:]):
                cvd = ",".join(half)
                print(f"[VD MAIN] launching spare replica_rank={replica_rank} on node {node_id[:12]} GPUs {cvd}...")
                rep = launch_spare(spare_cfg, config.actor_rollout_ref.model, replica_rank, node_id, cvd)
                spares.append({
                    "replica_rank": replica_rank,
                    "node_id": node_id,
                    "node_ip": entry["ip"],
                    "gpus": cvd,
                    "address": rep._server_address,
                })
                self.components.setdefault("spare_replicas", []).append(rep)
                replica_rank += 1

        # Warm probe (version-0 weights from disk), then park.
        prompt_ids = tokenizer.encode("elastic-vd warmup probe")
        for rep, meta in zip(self.components["spare_replicas"], spares):
            server = rep._server_handle
            ray.get(server.set_global_steps.remote(0))
            n = ray.get(server.spare_probe_generate.remote(prompt_ids, 8))
            print(f"[VD MAIN] spare {meta['replica_rank']} warm probe ok ({n} tokens) at {meta['address']}")
            ray.get(server.abort_all_requests.remote())          # leaves engine paused
            ray.get(server.wait_for_requests_to_drain.remote())
            if park_mode == "sleep":
                slept = ray.get(server.spare_sleep.remote(2))
                print(f"[VD MAIN] spare {meta['replica_rank']} parked (sleep L2) in {slept['seconds']}s")
            elif park_mode == "sleepL1":
                slept = ray.get(server.spare_sleep.remote(1))
                print(f"[VD MAIN] spare {meta['replica_rank']} parked (sleep L1: weights->CPU, "
                      f"KV freed) in {slept['seconds']}s")
            else:
                print(f"[VD MAIN] spare {meta['replica_rank']} parked PAUSED-RESIDENT "
                      f"(weights v0 + KV held, zero SM usage)")

        print("[VD MAIN] SPARE BOOTSTRAP: onloading trainer...")
        on = ray.get(trainer.elastic_onload.remote())
        print(f"[VD MAIN] initial onload: {on['seconds']:.1f}s")

        # Node agents: one per trainer node (prefetch) + fleet agents attach
        # later (controller creates them once fleet placement is known).
        agents = {}
        for node_id, entry in by_node.items():
            agent = VDNodeAgent.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                name=f"vd_node_agent_{entry['ip'].replace('.', '_')}",
                namespace=HANDLES_NAMESPACE,
                lifetime="detached",
                get_if_exists=True,
            ).remote(role="trainer")
            agents[entry["ip"]] = agent

        ray.get(handles.put.remote("spares", spares))
        ray.get(handles.put.remote(
            "spare_servers", [rep._server_handle for rep in self.components["spare_replicas"]]
        ))
        ray.get(handles.put.remote("spare_addresses", [m["address"] for m in spares]))
        ray.get(handles.put.remote("trainer_node_agents", agents))
        ray.get(handles.put.remote("trainer_nodes", {k: v["ip"] for k, v in by_node.items()}))
        ray.get(handles.put.remote("spares_state", "parked"))
        ray.get(handles.put.remote("spares_park_mode", park_mode))
        print(f"[VD MAIN] SPARE BOOTSTRAP complete in {time.time() - t0:.1f}s: {len(spares)} x TP4 spares parked")

    # ------------------------------------------------------------------
    # Component factories (upstream bodies with elastic classes swapped in)
    # ------------------------------------------------------------------

    def _create_rollouter(self, config) -> None:
        rollouter = ElasticVDRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter
        print("[VD MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }
        trainer = ElasticVDTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[VD MAIN] ElasticVDTrainer created and initialized successfully")


# ----------------------------------------------------------------------------
# Entry point (mirrors fully_async_main.py:222-243 + M1 PYTHONPATH injection)
# ----------------------------------------------------------------------------


def _inject_pythonpath(config) -> None:
    key = "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH"
    try:
        with open_dict(config):
            existing = OmegaConf.select(config, key)
            merged = _THIS_DIR if not existing else f"{_THIS_DIR}:{existing}"
            OmegaConf.update(config, key, merged, force_add=True)
        print(f"[VD MAIN] injected PYTHONPATH={merged} into ray runtime_env")
    except Exception as e:
        print(f"[VD MAIN] WARNING: could not inject PYTHONPATH ({e}); export it before launch instead")


@hydra.main(config_path=_FAP_CONFIG_DIR, config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    start_time = time.time()
    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    _inject_pythonpath(config)
    run_ppo(config, task_runner_class=ElasticVDTaskRunner)
    print(f"total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
