/*
 * Universal GPU C/R Shim
 *
 * LD_PRELOAD library that enables cuda-checkpoint freeze/restore for
 * multi-GPU (TP) workloads. Works with any PyTorch+NCCL framework
 * (tested: vLLM, SGLang).
 *
 * What it does:
 *   1. Intercepts ncclCommInitRank to track communicator handles
 *   2. On SIGRTMIN+1: calls ncclCommSuspend on all tracked comms
 *   3. On SIGRTMIN+2: calls ncclCommResume on all tracked comms
 *
 * Required environment variables (set by snapshot agent):
 *   NCCL_P2P_DISABLE=1   — no P2P/NVLink transport
 *   NCCL_SHM_DISABLE=1   — no shared memory transport
 *   NCCL_NVLS_ENABLE=0   — no NVLS channels
 *
 * These force NCCL to use NET/Socket (TCP loopback) which creates no
 * cross-process GPU state. cuda-checkpoint can then freeze/restore
 * each process independently.
 *
 * C/R sequence (driven by snapshot agent):
 *   1. kill -SIGRTMIN+1 <all_gpu_pids>     → ncclCommSuspend
 *   2. cuda-checkpoint --toggle --pid <pid>  → freeze (sequential)
 *   3. cuda-checkpoint --toggle --pid <pid>  → restore (sequential)
 *   4. kill -SIGRTMIN+2 <all_gpu_pids>     → ncclCommResume
 *
 * Build:
 *   gcc -shared -fPIC -o libcr-shim.so universal_cr_shim.c -ldl
 *
 * Usage:
 *   LD_PRELOAD=/path/to/libcr-shim.so \
 *   NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0 \
 *   vllm serve ... --tensor-parallel-size 2
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

typedef void* ncclComm_t;
typedef int ncclResult_t;
typedef struct { char internal[128]; } ncclUniqueId;

typedef ncclResult_t (*ncclCommInitRank_fn)(ncclComm_t*, int, ncclUniqueId, int);
typedef ncclResult_t (*ncclCommInitRankConfig_fn)(ncclComm_t*, int, ncclUniqueId, int, void*);
typedef ncclResult_t (*ncclCommDestroy_fn)(ncclComm_t);
typedef ncclResult_t (*ncclCommSuspend_fn)(ncclComm_t, int);
typedef ncclResult_t (*ncclCommResume_fn)(ncclComm_t);

static ncclCommInitRank_fn real_ncclCommInitRank;
static ncclCommInitRankConfig_fn real_ncclCommInitRankConfig;
static ncclCommDestroy_fn real_ncclCommDestroy;
static ncclCommSuspend_fn real_ncclCommSuspend;
static ncclCommResume_fn real_ncclCommResume;

#define MAX_COMMS 64
static ncclComm_t tracked_comms[MAX_COMMS];
static int num_comms = 0;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#define RESOLVE(var, type, name, mode) if (!(var)) (var) = (type)dlsym(mode, name)

static void resolve_nccl(void) {
    RESOLVE(real_ncclCommInitRank, ncclCommInitRank_fn, "ncclCommInitRank", RTLD_NEXT);
    RESOLVE(real_ncclCommInitRankConfig, ncclCommInitRankConfig_fn, "ncclCommInitRankConfig", RTLD_NEXT);
    RESOLVE(real_ncclCommDestroy, ncclCommDestroy_fn, "ncclCommDestroy", RTLD_NEXT);
    RESOLVE(real_ncclCommSuspend, ncclCommSuspend_fn, "ncclCommSuspend", RTLD_DEFAULT);
    RESOLVE(real_ncclCommResume, ncclCommResume_fn, "ncclCommResume", RTLD_DEFAULT);
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId id, int rank) {
    resolve_nccl();
    ncclResult_t r = real_ncclCommInitRank(comm, nranks, id, rank);
    if (r == 0 && num_comms < MAX_COMMS) {
        tracked_comms[num_comms++] = *comm;
        fprintf(stderr, "[cr-shim] PID %d: tracked comm %p (#%d, rank=%d/%d)\n",
                getpid(), (void*)*comm, num_comms, rank, nranks);
    }
    return r;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId id, int rank, void* config) {
    resolve_nccl();
    ncclResult_t r = real_ncclCommInitRankConfig(comm, nranks, id, rank, config);
    if (r == 0 && num_comms < MAX_COMMS) {
        tracked_comms[num_comms++] = *comm;
        fprintf(stderr, "[cr-shim] PID %d: tracked comm %p (#%d, rank=%d/%d)\n",
                getpid(), (void*)*comm, num_comms, rank, nranks);
    }
    return r;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    resolve_nccl();
    for (int i = 0; i < num_comms; i++) {
        if (tracked_comms[i] == comm) {
            tracked_comms[i] = tracked_comms[--num_comms];
            break;
        }
    }
    return real_ncclCommDestroy(comm);
}

static void suspend_handler(int sig) {
    double t0 = now_ms();
    resolve_nccl();
    fprintf(stderr, "[cr-shim] PID %d: SUSPEND — %d comms\n", getpid(), num_comms);
    if (real_ncclCommSuspend) {
        for (int i = 0; i < num_comms; i++) {
            int r = real_ncclCommSuspend(tracked_comms[i], 0x01);
            fprintf(stderr, "[cr-shim] PID %d:   comm %p suspend rc=%d\n",
                    getpid(), (void*)tracked_comms[i], r);
        }
    } else {
        fprintf(stderr, "[cr-shim] PID %d:   WARNING: ncclCommSuspend not found\n", getpid());
    }
    fprintf(stderr, "[cr-shim] PID %d: suspend done %.1fms\n", getpid(), now_ms() - t0);
}

static void resume_handler(int sig) {
    double t0 = now_ms();
    resolve_nccl();
    fprintf(stderr, "[cr-shim] PID %d: RESUME — %d comms\n", getpid(), num_comms);
    if (real_ncclCommResume) {
        for (int i = 0; i < num_comms; i++) {
            int r = real_ncclCommResume(tracked_comms[i]);
            fprintf(stderr, "[cr-shim] PID %d:   comm %p resume rc=%d\n",
                    getpid(), (void*)tracked_comms[i], r);
        }
    } else {
        fprintf(stderr, "[cr-shim] PID %d:   WARNING: ncclCommResume not found\n", getpid());
    }
    fprintf(stderr, "[cr-shim] PID %d: resume done %.1fms\n", getpid(), now_ms() - t0);
}

__attribute__((constructor))
static void init_cr_shim(void) {
    struct sigaction sa = {0};

    sa.sa_handler = suspend_handler;
    sigaction(SIGRTMIN + 1, &sa, NULL);

    sa.sa_handler = resume_handler;
    sigaction(SIGRTMIN + 2, &sa, NULL);

    fprintf(stderr, "[cr-shim] PID %d: ready (suspend=sig%d, resume=sig%d)\n",
            getpid(), SIGRTMIN + 1, SIGRTMIN + 2);
}
