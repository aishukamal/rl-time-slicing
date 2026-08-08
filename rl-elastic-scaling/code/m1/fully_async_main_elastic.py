# elastic-rl-poc M1 — fork of verl's fully_async_main.py driver
# (verl/experimental/fully_async_policy/fully_async_main.py @ 983cb0f2).
#
# What this fork changes vs upstream (everything else is copied verbatim):
#
#  1. INIT ORDERING (topology): rollouter (R1) is created FIRST, then the
#     out-of-tree R2 vLLM server is launched and WARMED on the GPU the trainer
#     will later use, then R2 is PARKED (drained + cuda-checkpointed to host
#     RAM via the snapshot-agent), and only THEN is the trainer created — so
#     the trainer's FSDP allocation finds its GPU empty. Upstream creates the
#     trainer first (only needed for hybrid worker-group injection, which M1
#     asserts off).
#
#  2. TOPOLOGY DECISION (deliverable D) — R2 runs OUT-OF-TREE, outside Ray's
#     GPU accounting. Logical GPU oversubscription was investigated and
#     rejected against the pinned source:
#       - Every 1-GPU verl workload creates a placement group whose bundle
#         demands a whole logical GPU ({"GPU": 1}, RayResourcePool.
#         get_placement_groups, single_controller/ray/base.py:146-151), so
#         trainer + R1 + R2 need 3 logical GPUs on a 2-GPU node -> Ray must be
#         started with num_gpus>=3.
#       - With 3 logical ids on 2 physical GPUs, exactly one workload receives
#         Ray accelerator id "2". Both device-selection paths then break on a
#         nonexistent device: (a) Ray-set CUDA_VISIBLE_DEVICES="2"; (b) the
#         NOSET path, where verl's Worker does
#         set_device(int(get_accelerator_ids()[..][0])) with the same logical
#         id (single_controller/base/worker.py:273-281).
#       - Which workload gets id "2" is decided by the raylet's free-id pool at
#         actor-schedule time; RayWorkerGroup._create_worker
#         (base.py:623-683) exposes no seam to pin ids per workload.
#     The fallback anticipated in the task is therefore used: R2 is a
#     vLLMHttpServer actor with NO Ray GPU, created exactly the way verl itself
#     creates rollout server actors — NodeAffinity + the platform NOSET env
#     vars + an explicit cuda_visible_devices argument (the in-tree pattern at
#     vllm_async_server.py:1197-1236; server actors there also hold no Ray
#     GPU). Its heavy GPU processes are the vLLM mp-executor subprocesses.
#     Consequences (documented, accepted for M1):
#       - R2 is invisible to rollouter.get_replicas() -> it can never leak into
#         the frozen NCCL weight-sync group (landmine #1 solved by
#         construction; elastic_trainer.py keeps the filter as belt-and-braces).
#       - R2 bypasses rollouter concurrency accounting
#         (fully_async_rollouter.py:503-506): max_concurrent_samples would stay
#         at 1 replica x concurrent_samples_per_replica. The rollouter subclass
#         below adds elastic_set_max_concurrent_samples() so the switch
#         controller can raise/lower it at activation/deactivation.
#       - GPU pairing is computed, not assumed: Ray's 2 placement groups
#         (R1's, then the trainer's) each reserve one whole physical GPU, so
#         the trainer deterministically lands on the GPU R1 did NOT take. The
#         driver reads R1's CUDA_VISIBLE_DEVICES after rollouter init and puts
#         R2 on the OTHER GPU — i.e. R2 shares with the trainer by construction.
#
#  3. A small named, detached Ray actor "elastic_controller_handles"
#     (namespace "elastic") holding the rollouter, GlobalRequestLoadBalancer,
#     MessageQueue, trainer and R2 handles + metadata, so the external
#     controller (r2_lifecycle.py, later the M2 policy loop) can reach them.
#     Needed because verl's LB/rollouter/MQ actors are unnamed (landmine #5).
#
#  4. ElasticFullyAsyncTrainer (see elastic_trainer.py) replaces
#     FullyAsyncTrainer; ElasticFullyAsyncRollouter (below) replaces
#     FullyAsyncRollouter to expose the LB handle and the concurrency lever.
#
# Config expectations (see k8s-job-m1.yaml): rollout.nnodes=1,
# rollout.n_gpus_per_node=1 (R1 only — R2 is NOT sized in-tree, see above),
# trainer.nnodes=1, trainer.n_gpus_per_node=1, TP=1,
# async_training.use_trainer_do_validate=False,
# async_training.use_dynamic_resource_scheduling=False.

import os
import socket
import sys
import time
from pprint import pprint
from uuid import uuid4

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import hydra
import ray
from omegaconf import OmegaConf, open_dict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import verl.experimental.fully_async_policy as _fap_pkg
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.plugin.platform import get_platform
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from elastic_trainer import ElasticFullyAsyncTrainer

HANDLES_ACTOR_NAME = "elastic_controller_handles"
HANDLES_NAMESPACE = "elastic"

# hydra config lives inside the verl package; our fork is out-of-tree, so
# resolve the packaged config dir to an absolute path.
# UNVERIFIED (hydra behavior, not verl): @hydra.main accepts absolute
# config_path directories; verified working pattern in hydra>=1.3 docs but not
# executed against the image's hydra version.
_FAP_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(_fap_pkg.__file__)), "config")


# ----------------------------------------------------------------------------
# Rollouter subclass: expose LB handle + concurrency lever + R1 GPU discovery.
# Same actor options as upstream (@ray.remote(num_cpus=10, max_concurrency=100),
# fully_async_rollouter.py:329). Subclass via __ray_actor_class__ (the verl
# unwrap pattern, single_controller/ray/base.py:968-971).
# ----------------------------------------------------------------------------

_ROLLOUTER_BASE = FullyAsyncRollouter.__ray_actor_class__


@ray.remote(num_cpus=10, max_concurrency=100)
class ElasticFullyAsyncRollouter(_ROLLOUTER_BASE):
    def get_load_balancer(self):
        """Handle of the GlobalRequestLoadBalancer actor (created unnamed in
        FullyAsyncLLMServerManager._init_global_load_balancer,
        llm_server.py:554-559). Actor handles are serializable RPC returns."""
        return self.llm_server_manager.global_load_balancer

    def get_standalone_server_addresses(self):
        """LB server_ids of the in-tree (standalone) replicas — R1."""
        return list(self.llm_server_manager.server_addresses)

    async def elastic_get_r1_cuda_visible_devices(self):
        """Physical GPU index string Ray assigned to R1's engine worker.

        CheckpointEngineWorker extends verl Worker, whose
        get_cuda_visible_devices() returns the process CUDA_VISIBLE_DEVICES
        (single_controller/base/worker.py:303-308)."""
        replica = self.llm_server_manager.get_standalone_replicas()[0]
        return await replica.workers[0].get_cuda_visible_devices.remote()

    def elastic_get_max_concurrent_samples(self) -> int:
        """Base concurrency directly from the attribute. get_statistics() is
        NOT usable pre-MQ: it awaits self.message_queue_client.get_statistics()
        unconditionally (fully_async_rollouter.py:1167) and the MQ client is
        only set later in driver init."""
        return int(self.max_concurrent_samples)

    def elastic_set_max_concurrent_samples(self, value: int) -> dict:
        """Out-of-tree replicas bypass _update_max_concurrent_samples
        (fully_async_rollouter.py:1274-1294); this mirrors it with an explicit
        value. The processor loop reads max_concurrent_samples on every
        submission (fully_async_rollouter.py:970), so the change takes effect
        immediately."""
        old = self.max_concurrent_samples
        value = int(value)
        if self.max_required_samples is not None:
            value = min(value, self.max_required_samples)
        self.max_concurrent_samples = value
        self._record_active_count()
        print(f"[ElasticFullyAsyncRollouter] max_concurrent_samples: {old} -> {value} (elastic override)")
        return {"old": old, "new": value}


# ----------------------------------------------------------------------------
# Named handle registry for the external switch controller.
# ----------------------------------------------------------------------------


@ray.remote(num_cpus=0)
class ElasticControllerHandles:
    """Tiny key/value actor. Detached + namespaced so r2_lifecycle.py (a
    separate Ray driver in the same pod) can look it up by name. Holds actor
    HANDLES (rollouter/LB/MQ/trainer/R2 server) plus plain metadata."""

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


@ray.remote(num_cpus=1)  # same options as upstream (fully_async_main.py:35)
class ElasticFullyAsyncTaskRunner(_TASK_RUNNER_BASE):
    def _initialize_components(self, config) -> None:
        print(f"[ELASTIC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # M1 preconditions: no hybrid replicas, no in-tree dynamic scheduling.
        assert not config.async_training.use_trainer_do_validate, (
            "elastic M1 requires use_trainer_do_validate=False (no hybrid worker group)"
        )
        assert not config.async_training.get("use_dynamic_resource_scheduling", False), (
            "elastic M1 requires use_dynamic_resource_scheduling=False"
        )
        assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1, "M1 layout is TP=1"

        print("[ELASTIC MAIN] Initializing model and tokenizer...")
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

        print("[ELASTIC MAIN] Creating worker mapping and resource pools...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # ---- handle registry (created early so metadata lands as we go) ----
        handles = ElasticControllerHandles.options(
            name=HANDLES_ACTOR_NAME,
            namespace=HANDLES_NAMESPACE,
            lifetime="detached",
            get_if_exists=True,
        ).remote()
        self.components["handles"] = handles
        ray.get(handles.put.remote("r2_parked", False))

        # ==== ELASTIC ORDERING STEP 1: rollouter (R1) BEFORE the trainer ====
        # Upstream creates the trainer first only for hybrid worker-group
        # injection (fully_async_main.py:77-84), which is asserted off above;
        # FullyAsyncRollouter.init_workers has no trainer dependency.
        print("[ELASTIC MAIN] Creating FullyAsyncRollouter (R1) FIRST...")
        self._create_rollouter(config)
        rollouter = self.components["rollouter"]

        load_balancer = ray.get(rollouter.get_load_balancer.remote())
        max_concurrent_base = ray.get(rollouter.elastic_get_max_concurrent_samples.remote())
        ray.get(handles.put.remote("rollouter", rollouter))
        ray.get(handles.put.remote("load_balancer", load_balancer))
        ray.get(handles.put.remote("max_concurrent_base", max_concurrent_base))

        # ==== ELASTIC ORDERING STEP 2: launch + warm R2 on the trainer GPU ====
        r1_gpu = str(ray.get(rollouter.elastic_get_r1_cuda_visible_devices.remote())).strip()
        r2_gpu = os.environ.get("ELASTIC_R2_GPU") or ("0" if r1_gpu == "1" else "1")
        print(f"[ELASTIC MAIN] R1 is on physical GPU {r1_gpu}; R2 (and later the trainer) -> GPU {r2_gpu}")
        r2_replica = self._launch_r2(config, r2_gpu)
        r2_server = r2_replica._server_handle
        r2_address = r2_replica._server_address

        r2_server_pid = ray.get(
            r2_server.__ray_call__.remote(lambda self: __import__("os").getpid())
        )
        # Landmine #4 insurance: never serve with global_steps=None.
        ray.get(r2_server.set_global_steps.remote(0))

        if os.environ.get("ELASTIC_R2_WARMUP", "1") == "1":
            print("[ELASTIC MAIN] R2 warmup generation...")
            try:
                prompt_ids = tokenizer.encode("elastic warmup probe")
                out = ray.get(
                    r2_server.generate.remote(
                        prompt_ids=prompt_ids,
                        sampling_params={"max_tokens": 8},
                        request_id=uuid4().hex,
                    )
                )
                print(f"[ELASTIC MAIN] R2 warmup ok ({len(out.token_ids)} tokens)")
            except Exception as e:
                print(f"[ELASTIC MAIN] WARNING: R2 warmup generation failed (non-fatal): {e}")

        ray.get(handles.put.remote("r2_server", r2_server))
        ray.get(handles.put.remote("r2_address", r2_address))
        ray.get(handles.put.remote("r2_server_pid", r2_server_pid))
        ray.get(handles.put.remote("r2_gpu", r2_gpu))
        ray.get(handles.put.remote("r2_replica_rank", r2_replica.replica_rank))
        ray.get(handles.put.remote("r2_state", "warm"))
        print(
            f"[ELASTIC MAIN] R2 up: address={r2_address} actor=vllm_server_"
            f"{r2_replica.replica_rank}_0 server_pid={r2_server_pid} gpu={r2_gpu}"
        )

        # ==== ELASTIC ORDERING STEP 3: PARK R2 before trainer allocation ====
        # R2 currently holds gpu_memory_utilization x 80GB on the trainer's
        # GPU; the trainer cannot init until that memory is checkpointed away.
        self._park_r2_barrier(handles, r2_server, r2_server_pid)

        # ==== ELASTIC ORDERING STEP 4: trainer on the now-free GPU ====
        print("[ELASTIC MAIN] Creating ElasticFullyAsyncTrainer...")
        self._create_trainer(config)
        trainer = self.components["trainer"]

        print("[ELASTIC MAIN] Setting up rollouter reference on trainer")
        # Builds the (R2-excluding) checkpoint manager — elastic_trainer.py.
        ray.get(trainer.set_rollouter.remote(rollouter))

        # ---- from here on: verbatim upstream tail (fully_async_main.py:89-115) ----
        total_train_steps = ray.get(rollouter.get_total_train_steps.remote())
        print(f"total_train_steps {total_train_steps}")
        ray.get(trainer.set_total_train_steps.remote(total_train_steps))

        max_queue_size = ray.get(rollouter.get_max_queue_size.remote())
        print(f"[ELASTIC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)
        self.components["message_queue"] = message_queue
        self.components["message_queue_client"] = message_queue_client

        ray.get(rollouter.set_message_queue_client.remote(message_queue_client))
        ray.get(trainer.set_message_queue_client.remote(message_queue_client))

        ray.get(trainer.load_checkpoint.remote())
        ray.get(rollouter.load_checkpoint.remote())

        print("[ELASTIC MAIN] Param sync before fit.. (NCCL group: trainer + R1 only; frozen hereafter)")
        ray.get(trainer._fit_update_weights.remote())

        if config.trainer.get("val_before_train", True):
            ray.get(trainer._fit_validate.remote(True))

        # ---- final handle registration for the switch controller ----
        trainer_pids = ray.get(trainer.get_trainer_worker_pids.remote())
        ray.get(handles.put.remote("trainer", trainer))
        ray.get(handles.put.remote("message_queue", message_queue))
        ray.get(handles.put.remote("trainer_worker_pids", trainer_pids))
        ray.get(handles.put.remote("trainer_state", "active"))
        print(
            f"[ELASTIC MAIN] elastic_controller_handles ready "
            f"(namespace={HANDLES_NAMESPACE}): keys={ray.get(handles.keys.remote())}; "
            f"trainer worker pids={trainer_pids}"
        )

        self._maybe_register_trainer_app_channel(trainer_pids)

        print("[ELASTIC MAIN] All components initialized successfully")

    # ------------------------------------------------------------------
    # Component factories (upstream bodies with elastic classes swapped in)
    # ------------------------------------------------------------------

    def _create_rollouter(self, config) -> None:
        """Copy of fully_async_main.py:117-136 minus hybrid injection (asserted
        off), with ElasticFullyAsyncRollouter."""
        print("[ELASTIC MAIN] Starting create rollouter...")
        rollouter = ElasticFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter
        print("[ELASTIC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        """Copy of fully_async_main.py:138-157 with ElasticFullyAsyncTrainer."""
        print("[ELASTIC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = ElasticFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ELASTIC MAIN] ElasticFullyAsyncTrainer created and initialized successfully")

    # ------------------------------------------------------------------
    # R2 launch (out-of-tree standalone vLLM server)
    # ------------------------------------------------------------------

    def _launch_r2(self, config, r2_gpu: str):
        """Launch R2 exactly the way vLLMReplica.launch_servers launches server
        actors (vllm_async_server.py:1174-1256), except:
          - no CheckpointEngineWorker worker group (workers=[]) — R2 never
            receives NCCL weight sync, and self.workers is only consulted for
            DP>1 asserts (vllm_async_server.py:357-373), which M1 (DP=1) never
            hits;
          - no Ray GPU request: the actor gets the platform NOSET env vars and
            an explicit cuda_visible_devices, the same mechanism as in-tree
            server actors (vllm_async_server.py:1212-1235).
        STANDALONE mode forces load_format=auto (vllm_async_server.py:175-177),
        so R2 loads real version-0 weights from disk.
        """
        replica_rank = int(os.environ.get("ELASTIC_R2_REPLICA_RANK", "1"))
        replica = vLLMReplica(
            replica_rank=replica_rank,
            config=config.actor_rollout_ref.rollout,
            model_config=config.actor_rollout_ref.model,
            gpus_per_node=1,
        )
        replica.rollout_mode = RolloutMode.STANDALONE

        env_vars = {
            **{var: "1" for var in get_platform().ray_noset_envvars()},
            **get_platform().rollout_env_vars(),
        }
        # Same naming convention as in-tree ("vllm_" prefix + server_{replica}_{node},
        # vllm_async_server.py:1205-1211, 1324-1326): replica_rank=1 -> vllm_server_1_0,
        # no collision with R1's vllm_server_0_0.
        name = f"vllm_server_{replica_rank}_0"
        node_id = ray.get_runtime_context().get_node_id()

        server = replica.server_class.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            runtime_env={"env_vars": env_vars},
            name=name,
            max_concurrency=replica.max_concurrency,
        ).remote(
            config=replica.config,
            model_config=replica.model_config,
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=replica_rank,
            node_rank=0,
            gpus_per_node=1,
            nnodes=1,
            cuda_visible_devices=str(r2_gpu),
        )

        print(f"[ELASTIC MAIN] Launching R2 vLLM server actor '{name}' on GPU {r2_gpu}...")
        ray.get(server.launch_server.remote())
        server_address, server_port = ray.get(server.get_server_address.remote())

        replica.servers = [server]
        replica._server_handle = server
        replica._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
        return replica

    # ------------------------------------------------------------------
    # Park barrier
    # ------------------------------------------------------------------

    def _park_r2_barrier(self, handles, r2_server, r2_server_pid) -> None:
        """Suspend R2 (freeing its GPU) before the trainer allocates.

        Auto mode (default, ELASTIC_AUTO_PARK=1 + AGENT_ENDPOINT set): the
        driver drains and cuda-checkpoints R2 itself via the timeslice client.
        Manual mode: block until the operator runs `r2_lifecycle.py park-r2`,
        which sets the r2_parked flag on the handles actor.
        """
        agent = os.environ.get("AGENT_ENDPOINT", "")
        auto = os.environ.get("ELASTIC_AUTO_PARK", "1") == "1" and bool(agent)

        if auto:
            print(f"[ELASTIC MAIN] Auto-parking R2 via snapshot-agent at {agent}...")
            import r2_lifecycle as rlc  # shared helpers; same ConfigMap dir

            timer = rlc.PhaseTimer("initial-park(auto)", timings_file=rlc.DEFAULT_TIMINGS_FILE)
            # R2 is not in the LB yet and carries no traffic; abort anyway so
            # the engine is in the paused state every resume path expects.
            with timer.phase("abort_all_requests"):
                ray.get(r2_server.abort_all_requests.remote())
            with timer.phase("wait_for_requests_to_drain"):
                ray.get(r2_server.wait_for_requests_to_drain.remote())
            with timer.phase("discover_gpu_pids"):
                pids = rlc.discover_gpu_pids(r2_server_pid)
                ray.get(handles.put.remote("r2_gpu_pids", pids))
            with timer.phase("cuda_snapshot_r2"):
                rlc.cuda_snapshot(agent, rlc.JOB_R2, pids)
            timer.finish()
            ray.get(handles.put.remote("r2_state", "suspended"))
            ray.get(handles.put.remote("r2_parked", True))
            print("[ELASTIC MAIN] R2 parked (auto). GPU freed for trainer init.")
            return

        print(
            "[ELASTIC MAIN] Waiting for R2 park. Run inside the pod:\n"
            "    python3 /workspace/m1/r2_lifecycle.py park-r2\n"
            "(sets the r2_parked flag on elastic_controller_handles)"
        )
        while not ray.get(handles.get.remote("r2_parked")):
            time.sleep(5)
        print("[ELASTIC MAIN] R2 park confirmed; continuing with trainer init.")

    # ------------------------------------------------------------------
    # Optional M2 hook: app_channel registration for the trainer
    # ------------------------------------------------------------------

    def _maybe_register_trainer_app_channel(self, trainer_pids) -> None:
        """CallbackAdapter path (timeslice.snapshot_agent.register_workload):
        makes the trainer addressable by job_id alone via the agent's
        app_channel backend; the callbacks delegate to the same explicit-PID
        cuda C/R the CLI uses. Default OFF for M1 (the CLI drives the cuda
        backend directly, keeping the C2 timing breakdown free of the extra
        agent->channel->agent hop).
        UNVERIFIED: agent-side handling of a nested Snapshot RPC issued from
        inside a WorkloadChannel command callback — verify before enabling."""
        agent = os.environ.get("AGENT_ENDPOINT", "")
        if os.environ.get("ELASTIC_TRAINER_APP_CHANNEL", "0") != "1" or not agent:
            return
        import r2_lifecycle as rlc
        from timeslice.snapshot_agent import register_workload

        pids = list(trainer_pids)
        self._trainer_channel = register_workload(
            agent,
            job_id="elastic-trainer-channel",
            group="elastic",
            on_snapshot=lambda mode, tags: rlc.cuda_snapshot(agent, rlc.JOB_TRAINER, pids),
            on_restore=lambda tags: rlc.cuda_restore(agent, rlc.JOB_TRAINER, pids),
            supported_modes=["offload"],
        )
        print("[ELASTIC MAIN] trainer registered on app_channel as 'elastic-trainer-channel'")


# ----------------------------------------------------------------------------
# Entry point (mirrors fully_async_main.py:222-243)
# ----------------------------------------------------------------------------


def _inject_pythonpath(config) -> None:
    """Make this directory importable in every Ray worker so classes pickled
    by reference (elastic_trainer.ElasticFullyAsyncTrainer) resolve."""
    key = "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH"
    try:
        with open_dict(config):
            existing = OmegaConf.select(config, key)
            merged = _THIS_DIR if not existing else f"{_THIS_DIR}:{existing}"
            OmegaConf.update(config, key, merged, force_add=True)
        print(f"[ELASTIC MAIN] injected PYTHONPATH={merged} into ray runtime_env")
    except Exception as e:
        print(
            f"[ELASTIC MAIN] WARNING: could not inject PYTHONPATH into ray runtime_env ({e}); "
            f"export PYTHONPATH={_THIS_DIR} before launch instead"
        )


@hydra.main(config_path=_FAP_CONFIG_DIR, config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    start_time = time.time()
    auto_set_device(config)
    # TODO(upstream): unify rollout config with actor_rollout_ref (copied)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)
    _inject_pythonpath(config)
    run_ppo(config, task_runner_class=ElasticFullyAsyncTaskRunner)
    print(f"total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
