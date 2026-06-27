"""rl_loop.py — GRPO RL loop driver for TPU time-slicing PoC

Matches the GPU reference implementation's structure:
  acquire sampler → generate → yield sampler →
  acquire trainer → train → yield trainer → repeat

Env vars:
  TRAINER_URL        http://trainer-a-svc:8200
  SAMPLER_URL        http://sampler-a-svc:8300
  ORCHESTRATOR_URL   http://tpu-orchestrator-svc:9000
  JOB_ID             job-a | job-b
  N_RL_STEPS         number of RL iterations (default: 10)
  PROMPTS_PER_STEP   prompts sampled per step (default: 16)
  GROUP_SIZE         completions per prompt (default: 8)
  MAX_NEW_TOKENS     max generation length (default: 256)
  WEIGHT_SYNC_INTERVAL  sync every N steps (0=off, default: 0)
  CHECKPOINTED       true if workloads are pre-checkpointed (job-b)
"""

import json
import logging
import os
import random
import sys
import time
import urllib.request

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [rl-loop] %(levelname)s %(message)s"
)
log = logging.getLogger("rl-loop")

TRAINER_URL = os.environ.get("TRAINER_URL", "http://trainer-a-svc:8200")
SAMPLER_URL = os.environ.get("SAMPLER_URL", "http://sampler-a-svc:8300")
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://tpu-orchestrator-svc:9000")
JOB_ID = os.environ.get("JOB_ID", "job-a")
LOG_DIR = os.environ.get("LOG_DIR", "/data/rl_logs")
N_RL_STEPS = int(os.environ.get("N_RL_STEPS", "10"))
PROMPTS_PER_STEP = int(os.environ.get("PROMPTS_PER_STEP", "16"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
WEIGHT_SYNC_INTERVAL = int(os.environ.get("WEIGHT_SYNC_INTERVAL", "0"))
GEN_BATCH_SIZE = int(os.environ.get("GEN_BATCH_SIZE", "3"))
TRAINER_POD_NAME = os.environ.get("TRAINER_POD_NAME", "tpu-trainers")
SAMPLER_POD_NAME = os.environ.get("SAMPLER_POD_NAME", "tpu-samplers")

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
DATASET_PATH = os.environ.get("DATASET_PATH", "/tmp/gsm8k_prompts.json")

# -- Import reward module (bundled via ConfigMap) ------------------------------

sys.path.insert(0, "/app")
from reward import compute_rewards, compute_advantages, extract_answer


# -- Dataset -------------------------------------------------------------------

def download_dataset():
    if os.path.exists(DATASET_PATH):
        log.info(f"Dataset already present at {DATASET_PATH}")
        return
    log.info("Downloading GSM8K dataset...")
    data = urllib.request.urlopen(GSM8K_URL, timeout=60).read().decode("utf-8")
    records = [json.loads(line) for line in data.strip().split("\n")]
    with open(DATASET_PATH, "w") as f:
        json.dump(records, f)
    log.info(f"Saved {len(records)} prompts to {DATASET_PATH}")


def load_dataset():
    with open(DATASET_PATH) as f:
        return json.load(f)


def sample_batch(dataset, n):
    batch = random.sample(dataset, min(n, len(dataset)))
    prompts, answers = [], []
    for item in batch:
        question = item.get("question", item.get("prompt", ""))
        answer = item.get("answer", item.get("ground_truth", ""))
        gt = extract_answer(str(answer)) or str(answer)
        prompt = (
            "<|im_start|>system\n"
            "Respond in the following format:\n\n"
            "<reasoning>\n...\n</reasoning>\n"
            "<answer>\n...\n</answer>"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "What is the largest single-digit prime number?"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<reasoning>\n"
            "Single-digit primes are 2, 3, 5, 7. The largest is 7.\n"
            "</reasoning>\n"
            "<answer>\n7\n</answer>"
            "<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<reasoning>\n"
        )
        prompts.append(prompt)
        answers.append(gt)
    return prompts, answers


# -- HTTP helpers --------------------------------------------------------------

def http_post(url, data, timeout=600):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode())


def http_get(url, timeout=30):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read().decode())


def wait_for_service(name, url, max_wait=1800):
    log.info(f"Waiting for {name} at {url}...")
    for i in range(max_wait // 5):
        try:
            r = http_get(f"{url}/health")
            if r.get("status") == "ok" or r.get("model_loaded") is not None:
                log.info(f"{name} is ready: {r}")
                return True
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"{name} not ready after {max_wait}s")


# -- Orchestrator helpers ------------------------------------------------------

def get_pod_node(pod_name):
    try:
        import ssl
        token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        ca_path = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        k8s_host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        k8s_port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        ns = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()
        with open(token_path) as f:
            token = f.read().strip()
        ctx = ssl.create_default_context(cafile=ca_path)
        url = f"https://{k8s_host}:{k8s_port}/api/v1/namespaces/{ns}/pods/{pod_name}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        resp = urllib.request.urlopen(req, timeout=10, context=ctx)
        data = json.loads(resp.read().decode())
        return data.get("spec", {}).get("nodeName", "")
    except Exception as e:
        log.warning(f"Could not get node for pod {pod_name}: {e}")
        return ""


def orch_register(workload_id, pool, pids, url=""):
    pod_map = {"trainer": TRAINER_POD_NAME, "sampler": SAMPLER_POD_NAME}
    node = get_pod_node(pod_map.get(pool, ""))
    log.info(f"Registering {workload_id} pool={pool} pids={pids} node={node}")
    return http_post(f"{ORCHESTRATOR_URL}/register", {
        "workload_id": workload_id,
        "pool": pool,
        "pids": pids,
        "node": node,
        "url": url,
    })


def orch_acquire(workload_id):
    t0 = time.perf_counter()
    result = http_post(f"{ORCHESTRATOR_URL}/acquire", {"workload_id": workload_id}, timeout=1800)
    ms = round((time.perf_counter() - t0) * 1000)
    log.info(f"Acquired {workload_id}: restore_ms={result.get('restore_ms', 0)} wait_ms={result.get('wait_ms', 0)} total={ms}ms")
    return result, ms


def orch_yield(workload_id):
    t0 = time.perf_counter()
    result = http_post(f"{ORCHESTRATOR_URL}/yield", {"workload_id": workload_id}, timeout=300)
    ms = round((time.perf_counter() - t0) * 1000)
    log.info(f"Yielded {workload_id}: checkpoint_ms={result.get('checkpoint_ms', 0)} total={ms}ms")
    return result, ms


# -- Main loop -----------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info(f"TPU GRPO RL Loop | {JOB_ID}")
    log.info(f"  Trainer:      {TRAINER_URL}")
    log.info(f"  Sampler:      {SAMPLER_URL}")
    log.info(f"  Orchestrator: {ORCHESTRATOR_URL}")
    log.info(f"  Steps: {N_RL_STEPS}, Prompts/step: {PROMPTS_PER_STEP}, "
             f"G={GROUP_SIZE}, max_tokens={MAX_NEW_TOKENS}, gen_batch={GEN_BATCH_SIZE}")
    log.info(f"  Weight sync: {'every ' + str(WEIGHT_SYNC_INTERVAL) + ' steps' if WEIGHT_SYNC_INTERVAL > 0 else 'off'}")
    log.info(f"  Total completions/step: {PROMPTS_PER_STEP * GROUP_SIZE}")
    log.info("=" * 60)

    download_dataset()
    dataset = load_dataset()
    log.info(f"Loaded {len(dataset)} GSM8K prompts")

    wait_for_service("Orchestrator", ORCHESTRATOR_URL)
    wait_for_service("Sampler", SAMPLER_URL)
    wait_for_service("Trainer", TRAINER_URL)

    trainer_wl = f"{JOB_ID}-trainer"
    sampler_wl = f"{JOB_ID}-sampler"
    trainer_pids = http_get(f"{TRAINER_URL}/get_pids")["pids"]
    sampler_pids = http_get(f"{SAMPLER_URL}/get_pids")["pids"]
    orch_register(trainer_wl, "trainer", trainer_pids, url=TRAINER_URL)
    orch_register(sampler_wl, "sampler", sampler_pids, url=SAMPLER_URL)

    all_metrics = []
    holding_sampler = False

    for step in range(1, N_RL_STEPS + 1):
        t0 = time.perf_counter()
        log.info(f"\n--- Step {step}/{N_RL_STEPS} ---")

        # -- ACQUIRE SAMPLER --
        acquire_sampler_ms = 0
        if not holding_sampler:
            _, acquire_sampler_ms = orch_acquire(sampler_wl)

        # -- ENSURE SAMPLER IS READY (lazy init under lock) --
        if step == 1:
            health = http_get(f"{SAMPLER_URL}/health")
            if not health.get("vllm_started"):
                log.info("Sampler vLLM not started — triggering /start_vllm (under lock)...")
                http_post(f"{SAMPLER_URL}/start_vllm", {})
                for _ in range(360):
                    h = http_get(f"{SAMPLER_URL}/health")
                    if h.get("vllm_started"):
                        log.info("Sampler vLLM ready")
                        break
                    time.sleep(5)
                else:
                    raise RuntimeError("Sampler vLLM failed to start under lock")

        # -- GENERATE (batched to control phase duration) --
        prompts, answers = sample_batch(dataset, PROMPTS_PER_STEP)
        n_batches = (len(prompts) + GEN_BATCH_SIZE - 1) // GEN_BATCH_SIZE
        log.info(f"Generating {len(prompts)}x{GROUP_SIZE} rollouts in {n_batches} batches of {GEN_BATCH_SIZE}...")
        t_gen = time.perf_counter()
        completions = []
        for b in range(n_batches):
            batch_prompts = prompts[b * GEN_BATCH_SIZE : (b + 1) * GEN_BATCH_SIZE]
            gen_result = http_post(f"{SAMPLER_URL}/generate", {
                "prompts": batch_prompts,
                "group_size": GROUP_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
            })
            completions.extend(gen_result["completions"])
            if (b + 1) % 10 == 0:
                log.info(f"    batch {b+1}/{n_batches}: {len(completions)} completions so far")
        gen_ms = round((time.perf_counter() - t_gen) * 1000)
        log.info(f"  {len(completions)} completions in {gen_ms}ms ({n_batches} batches)")

        # -- YIELD SAMPLER --
        _, yield_sampler_ms = orch_yield(sampler_wl)
        holding_sampler = False

        # -- REWARDS (CPU, no TPU needed) --
        expanded_gt = [a for a in answers for _ in range(GROUP_SIZE)]
        rewards, reward_stats = compute_rewards(completions, expanded_gt)
        advantages, _ = compute_advantages(rewards, GROUP_SIZE)
        mean_reward = sum(rewards) / max(len(rewards), 1)
        log.info(f"  reward={mean_reward:.3f}  "
                 f"correct={reward_stats['correct_rate']:.2%}  "
                 f"format={reward_stats['format_rate']:.2%}  "
                 f"truncated={reward_stats['truncated_rate']:.2%}")

        # -- ACQUIRE TRAINER --
        _, acquire_trainer_ms = orch_acquire(trainer_wl)

        # -- TRAIN --
        expanded_prompts = [p for p in prompts for _ in range(GROUP_SIZE)]
        log.info(f"Training on {len(completions)} samples...")
        t_train = time.perf_counter()
        train_result = http_post(f"{TRAINER_URL}/train", {
            "prompts": expanded_prompts,
            "completions": completions,
            "advantages": advantages,
            "group_size": 1,
        })
        train_ms = round((time.perf_counter() - t_train) * 1000)
        log.info(f"  loss={train_result.get('loss', 0):.4f}  "
                 f"kl={train_result.get('kl_loss', 0):.4f}  {train_ms}ms")

        # -- WEIGHT SYNC (optional) --
        sync_ms = 0
        if WEIGHT_SYNC_INTERVAL > 0 and step % WEIGHT_SYNC_INTERVAL == 0:
            _, reacquire_ms = orch_acquire(sampler_wl)
            log.info("Syncing weights (both TPUs held)...")
            t_sync = time.perf_counter()
            try:
                sync_result = http_post(f"{SAMPLER_URL}/reload_weights", {}, timeout=300)
                sync_ms = round((time.perf_counter() - t_sync) * 1000)
                log.info(f"  weights synced in {sync_ms}ms")
            except Exception as e:
                sync_ms = round((time.perf_counter() - t_sync) * 1000)
                log.warning(f"  weight sync failed ({sync_ms}ms): {e}")
            holding_sampler = True

        # -- YIELD TRAINER --
        _, yield_trainer_ms = orch_yield(trainer_wl)

        # -- STEP SUMMARY --
        step_ms = round((time.perf_counter() - t0) * 1000)
        rec = {
            "type": "step",
            "job_id": JOB_ID,
            "step": step,
            "gen_ms": gen_ms,
            "train_ms": train_ms,
            "sync_ms": sync_ms,
            "acquire_sampler_ms": acquire_sampler_ms,
            "yield_sampler_ms": yield_sampler_ms,
            "acquire_trainer_ms": acquire_trainer_ms,
            "yield_trainer_ms": yield_trainer_ms,
            "step_ms": step_ms,
            "mean_reward": round(mean_reward, 4),
            "correct_rate": reward_stats["correct_rate"],
            "format_rate": reward_stats["format_rate"],
            "truncated_rate": reward_stats["truncated_rate"],
            "loss": train_result.get("loss"),
            "kl_loss": train_result.get("kl_loss"),
            "grad_norm": train_result.get("grad_norm"),
        }
        all_metrics.append(rec)
        log.info(f"  Step {step} done in {step_ms}ms")

        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, f"metrics_{JOB_ID}.jsonl"), "a") as f:
            f.write(json.dumps({**rec, "ts": time.time()}) + "\n")

    if holding_sampler:
        orch_yield(sampler_wl)

    log.info("=" * 60)
    log.info(f"{JOB_ID} COMPLETE — {len(all_metrics)} steps")
    avg = lambda k: sum(m.get(k, 0) for m in all_metrics) / max(len(all_metrics), 1)
    log.info(f"  avg reward:  {avg('mean_reward'):.3f}")
    log.info(f"  avg correct: {avg('correct_rate'):.2%}")
    log.info(f"  avg format:  {avg('format_rate'):.2%}")
    log.info(f"  avg loss:    {avg('loss'):.4f}")
    log.info(f"  avg gen:     {avg('gen_ms'):.0f}ms")
    log.info(f"  avg train:   {avg('train_ms'):.0f}ms")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
    log.info("=== [RL_JOB_COMPLETED] ===")
    log.info("Sleeping until metrics are collected...")
    while True:
        time.sleep(3600)
