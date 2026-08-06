/*
 * Universal GPU C/R Shim v2 — NVLink-compatible (destroy/recreate)
 *
 * v1 (universal_cr_shim.c) uses ncclCommSuspend/Resume, which requires NCCL
 * to run on TCP transport (NCCL_P2P_DISABLE=1 etc.) because suspend does not
 * tear down P2P/SHM transport state that cuda-checkpoint cannot restore.
 * That costs steady-state performance: no NVLink.
 *
 * v2 removes the transport restriction. Instead of suspending comms across
 * the freeze, it DESTROYS them before freeze and RECREATES them after
 * restore:
 *
 *   Steady state:  NVLink P2P — full speed. (NVLS must be off: multicast
 *                  objects cannot be re-created in a restored CUDA context,
 *                  so launch with NCCL_NVLS_ENABLE=0.)
 *   C/R window:    quiesce workload
 *                  SIGRTMIN+1 -> ncclCommDestroy all comms
 *                      (all cross-process GPU state torn down by NCCL itself)
 *                  cuda-checkpoint freeze   (sees only process-private state)
 *                  ... GPU free for other workloads ...
 *                  cuda-checkpoint restore
 *                  SIGRTMIN+2 -> arms lazy recreate (flag only)
 *                  workload resumes; the FIRST collective call on each rank
 *                  performs the rendezvous + ncclCommInitRank on the app's
 *                  own thread (async-signal-safe context), then proceeds.
 *
 * The application (PyTorch etc.) still holds the ORIGINAL ncclComm_t handles.
 * The shim maintains an app_handle -> current_handle table and translates on
 * every intercepted NCCL call, so the recreate is invisible to the framework.
 * While comms are destroyed, query calls (ncclCommGetAsyncError etc. — e.g.
 * PyTorch's watchdog) are answered from cached values instead of touching the
 * dead comm.
 *
 * Fresh ncclUniqueId rendezvous: the original uniqueId is stale after restore
 * (bootstrap sockets are dead), so rank 0 of each comm generates a new one at
 * recreate time and publishes it via CR_RENDEZVOUS_DIR (default
 * /dev/shm/cr-rendezvous). Same pod: automatic. Multi-node: use a shared
 * volume, or extend the rendezvous to TCP.
 *
 * Requirements:
 *   - Workload quiesced during the C/R window (no collectives in flight).
 *   - All ranks get SIGRTMIN+1 together (destroy is collective), and all
 *     ranks resume together after SIGRTMIN+2 (init is collective, performed
 *     lazily at the first collective per rank).
 *   - NCCL_NVLS_ENABLE=0 at launch (see above). NVLink P2P stays on.
 *
 * Build:
 *   gcc -shared -fPIC -o libcr-shim-v2.so universal_cr_shim_v2.c -ldl -lpthread
 *
 * Usage:
 *   NCCL_NVLS_ENABLE=0 LD_PRELOAD=/path/to/libcr-shim-v2.so <workload>
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef void* ncclComm_t;
typedef int   ncclResult_t;
typedef struct { char internal[128]; } ncclUniqueId;

/* ---- real-function pointer types (enums passed as int: ABI-compatible) -- */
typedef ncclResult_t (*fnInitRank)(ncclComm_t*, int, ncclUniqueId, int);
typedef ncclResult_t (*fnInitRankConfig)(ncclComm_t*, int, ncclUniqueId, int, void*);
typedef ncclResult_t (*fnGetUniqueId)(ncclUniqueId*);
typedef ncclResult_t (*fnCommDestroy)(ncclComm_t);
typedef ncclResult_t (*fnCommAbort)(ncclComm_t);
typedef ncclResult_t (*fnCommFinalize)(ncclComm_t);
typedef ncclResult_t (*fnAllReduce)(const void*, void*, size_t, int, int, ncclComm_t, void*);
typedef ncclResult_t (*fnBroadcast)(const void*, void*, size_t, int, int, ncclComm_t, void*);
typedef ncclResult_t (*fnBcast)(void*, size_t, int, int, ncclComm_t, void*);
typedef ncclResult_t (*fnReduce)(const void*, void*, size_t, int, int, int, ncclComm_t, void*);
typedef ncclResult_t (*fnAllGather)(const void*, void*, size_t, int, ncclComm_t, void*);
typedef ncclResult_t (*fnReduceScatter)(const void*, void*, size_t, int, int, ncclComm_t, void*);
typedef ncclResult_t (*fnSend)(const void*, size_t, int, int, ncclComm_t, void*);
typedef ncclResult_t (*fnRecv)(void*, size_t, int, int, ncclComm_t, void*);
typedef ncclResult_t (*fnCommCount)(const ncclComm_t, int*);
typedef ncclResult_t (*fnCommCuDevice)(const ncclComm_t, int*);
typedef ncclResult_t (*fnCommUserRank)(const ncclComm_t, int*);
typedef ncclResult_t (*fnCommGetAsyncError)(ncclComm_t, ncclResult_t*);
typedef ncclResult_t (*fnCommRegister)(const ncclComm_t, void*, size_t, void**);
typedef ncclResult_t (*fnCommDeregister)(const ncclComm_t, void*);

static fnInitRank          real_InitRank;
static fnInitRankConfig    real_InitRankConfig;
static fnGetUniqueId       real_GetUniqueId;
static fnCommDestroy       real_CommDestroy;
static fnCommAbort         real_CommAbort;
static fnCommFinalize      real_CommFinalize;
static fnAllReduce         real_AllReduce;
static fnBroadcast         real_Broadcast;
static fnBcast             real_Bcast;
static fnReduce            real_Reduce;
static fnAllGather         real_AllGather;
static fnReduceScatter     real_ReduceScatter;
static fnSend              real_Send;
static fnRecv              real_Recv;
static fnCommCount         real_CommCount;
static fnCommCuDevice      real_CommCuDevice;
static fnCommUserRank      real_CommUserRank;
static fnCommGetAsyncError real_CommGetAsyncError;
static fnCommRegister      real_CommRegister;
static fnCommDeregister    real_CommDeregister;

/* Resolve real NCCL symbols from an explicit handle to the real library.
 * We must NOT fall back to RTLD_DEFAULT: this shim can be dlopen'd directly
 * (vLLM's PyNCCL via VLLM_NCCL_SO_PATH), and RTLD_DEFAULT would find our own
 * exports -> infinite recursion. CR_NCCL_LIB points at the real libnccl. */
static void* real_nccl_handle(void) {
    static void* h;
    if (h) return h;
    const char* p = getenv("CR_NCCL_LIB");
    if (p && p[0]) h = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libnccl.so.2", RTLD_NOLOAD | RTLD_NOW);
    if (!h) h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
    return h;
}

#define RESOLVE(var, type, name) \
    do { \
        if (!(var)) { void* _h = real_nccl_handle(); if (_h) (var) = (type)dlsym(_h, name); } \
        if (!(var)) (var) = (type)dlsym(RTLD_NEXT, name); \
    } while (0)

static void resolve_all(void) {
    RESOLVE(real_InitRank,          fnInitRank,          "ncclCommInitRank");
    RESOLVE(real_InitRankConfig,    fnInitRankConfig,    "ncclCommInitRankConfig");
    RESOLVE(real_GetUniqueId,       fnGetUniqueId,       "ncclGetUniqueId");
    RESOLVE(real_CommDestroy,       fnCommDestroy,       "ncclCommDestroy");
    RESOLVE(real_CommAbort,         fnCommAbort,         "ncclCommAbort");
    RESOLVE(real_CommFinalize,      fnCommFinalize,      "ncclCommFinalize");
    RESOLVE(real_AllReduce,         fnAllReduce,         "ncclAllReduce");
    RESOLVE(real_Broadcast,         fnBroadcast,         "ncclBroadcast");
    RESOLVE(real_Bcast,             fnBcast,             "ncclBcast");
    RESOLVE(real_Reduce,            fnReduce,            "ncclReduce");
    RESOLVE(real_AllGather,         fnAllGather,         "ncclAllGather");
    RESOLVE(real_ReduceScatter,     fnReduceScatter,     "ncclReduceScatter");
    RESOLVE(real_Send,              fnSend,              "ncclSend");
    RESOLVE(real_Recv,              fnRecv,              "ncclRecv");
    RESOLVE(real_CommCount,         fnCommCount,         "ncclCommCount");
    RESOLVE(real_CommCuDevice,      fnCommCuDevice,      "ncclCommCuDevice");
    RESOLVE(real_CommUserRank,      fnCommUserRank,      "ncclCommUserRank");
    RESOLVE(real_CommGetAsyncError, fnCommGetAsyncError, "ncclCommGetAsyncError");
    RESOLVE(real_CommRegister,      fnCommRegister,      "ncclCommRegister");
    RESOLVE(real_CommDeregister,    fnCommDeregister,    "ncclCommDeregister");
}

/* ---- CUDA graph tracking -------------------------------------------------
 * Intercept cudaGraphInstantiate to track live graph executables.
 * Before destroying NCCL comms, we destroy all tracked graphs so
 * cuda-checkpoint can freeze the multi-device process.
 * Frameworks auto-recapture on the next forward pass after restore. */
typedef void* cudaGraphExec_t;
typedef void* cudaGraph_t;
typedef int cudaError_t;
typedef cudaError_t (*fnGraphInstantiate)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
typedef cudaError_t (*fnGraphExecDestroy)(cudaGraphExec_t);

static fnGraphInstantiate real_GraphInstantiate;
static fnGraphExecDestroy real_GraphExecDestroy;

#define MAX_GRAPHS 4096
static cudaGraphExec_t tracked_graphs[MAX_GRAPHS];
static int n_graphs = 0;

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* out, cudaGraph_t graph,
                                 unsigned long long flags) {
    if (!real_GraphInstantiate) {
        real_GraphInstantiate = (fnGraphInstantiate)dlsym(RTLD_NEXT, "cudaGraphInstantiate");
        if (!real_GraphInstantiate)
            real_GraphInstantiate = (fnGraphInstantiate)dlsym(RTLD_NEXT, "cudaGraphInstantiate_v2");
    }
    if (!real_GraphInstantiate) return 999;
    cudaError_t r = real_GraphInstantiate(out, graph, flags);
    if (r == 0 && out && *out && n_graphs < MAX_GRAPHS) {
        tracked_graphs[n_graphs++] = *out;
    }
    return r;
}

/* Also intercept the older 5-arg overload (CUDA <12.x compat) */
cudaError_t cudaGraphInstantiate_v2(cudaGraphExec_t* out, cudaGraph_t graph,
                                     void* errNode, char* errLog, size_t bufSize) {
    typedef cudaError_t (*fn5)(cudaGraphExec_t*, cudaGraph_t, void*, char*, size_t);
    static fn5 real_fn;
    if (!real_fn) real_fn = (fn5)dlsym(RTLD_NEXT, "cudaGraphInstantiate_v2");
    if (!real_fn) return 999;
    cudaError_t r = real_fn(out, graph, errNode, errLog, bufSize);
    if (r == 0 && out && *out && n_graphs < MAX_GRAPHS) {
        tracked_graphs[n_graphs++] = *out;
    }
    return r;
}

/* PyTorch/torch.compile uses this variant (libtorch_cuda.so links
 * cudaGraphInstantiateWithFlags from libcudart.so) */
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* out, cudaGraph_t graph,
                                          unsigned long long flags) {
    typedef cudaError_t (*fnFlags)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
    static fnFlags real_fn;
    if (!real_fn) real_fn = (fnFlags)dlsym(RTLD_NEXT, "cudaGraphInstantiateWithFlags");
    if (!real_fn) return 999;
    cudaError_t r = real_fn(out, graph, flags);
    if (r == 0 && out && *out && n_graphs < MAX_GRAPHS) {
        tracked_graphs[n_graphs++] = *out;
        fprintf(stderr, "[cr-shim2] PID %d: tracked graph exec %p (#%d)\n",
                getpid(), *out, n_graphs);
    }
    return r;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t ge) {
    if (!real_GraphExecDestroy)
        real_GraphExecDestroy = (fnGraphExecDestroy)dlsym(RTLD_NEXT, "cudaGraphExecDestroy");
    /* remove from tracking */
    for (int i = 0; i < n_graphs; i++) {
        if (tracked_graphs[i] == ge) {
            tracked_graphs[i] = tracked_graphs[--n_graphs];
            break;
        }
    }
    return real_GraphExecDestroy ? real_GraphExecDestroy(ge) : 0;
}

/* Also track graph templates (cudaGraph_t), not just executables */
typedef cudaError_t (*fnGraphDestroy)(cudaGraph_t);
static fnGraphDestroy real_GraphDestroy;

#define MAX_GRAPH_TEMPLATES 4096
static cudaGraph_t tracked_templates[MAX_GRAPH_TEMPLATES];
static int n_templates = 0;

/* Intercept cudaGraphCreate to track templates */
cudaError_t cudaGraphCreate(cudaGraph_t* out, unsigned int flags) {
    typedef cudaError_t (*fn)(cudaGraph_t*, unsigned int);
    static fn real_fn;
    if (!real_fn) real_fn = (fn)dlsym(RTLD_NEXT, "cudaGraphCreate");
    if (!real_fn) return 999;
    cudaError_t r = real_fn(out, flags);
    if (r == 0 && out && *out && n_templates < MAX_GRAPH_TEMPLATES)
        tracked_templates[n_templates++] = *out;
    return r;
}

cudaError_t cudaGraphDestroy(cudaGraph_t g) {
    if (!real_GraphDestroy)
        real_GraphDestroy = (fnGraphDestroy)dlsym(RTLD_NEXT, "cudaGraphDestroy");
    for (int i = 0; i < n_templates; i++) {
        if (tracked_templates[i] == g) {
            tracked_templates[i] = tracked_templates[--n_templates];
            break;
        }
    }
    return real_GraphDestroy ? real_GraphDestroy(g) : 0;
}

/* Also intercept cudaStreamEndCapture which produces a graph template */
typedef cudaError_t (*fnStreamEndCapture)(void*, cudaGraph_t*);
cudaError_t cudaStreamEndCapture(void* stream, cudaGraph_t* out) {
    static fnStreamEndCapture real_fn;
    if (!real_fn) real_fn = (fnStreamEndCapture)dlsym(RTLD_NEXT, "cudaStreamEndCapture");
    if (!real_fn) return 999;
    cudaError_t r = real_fn(stream, out);
    if (r == 0 && out && *out && n_templates < MAX_GRAPH_TEMPLATES)
        tracked_templates[n_templates++] = *out;
    return r;
}

static void reset_all_graphs(void) {
    if (!real_GraphExecDestroy)
        real_GraphExecDestroy = (fnGraphExecDestroy)dlsym(RTLD_NEXT, "cudaGraphExecDestroy");
    if (!real_GraphDestroy)
        real_GraphDestroy = (fnGraphDestroy)dlsym(RTLD_NEXT, "cudaGraphDestroy");

    fprintf(stderr, "[cr-shim2] PID %d: resetting %d graph execs + %d graph templates\n",
            getpid(), n_graphs, n_templates);

    /* Destroy executables first (they reference templates) */
    for (int i = 0; i < n_graphs; i++) {
        if (real_GraphExecDestroy && tracked_graphs[i])
            real_GraphExecDestroy(tracked_graphs[i]);
    }
    int exec_count = n_graphs;
    n_graphs = 0;

    /* Then destroy templates */
    for (int i = 0; i < n_templates; i++) {
        if (real_GraphDestroy && tracked_templates[i])
            real_GraphDestroy(tracked_templates[i]);
    }
    int tmpl_count = n_templates;
    n_templates = 0;

    /* Synchronize all devices to flush driver-side cleanup */
    typedef int (*fnDevSync)(void);
    fnDevSync ds = (fnDevSync)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    if (ds) ds();

    fprintf(stderr, "[cr-shim2] PID %d: reset %d execs + %d templates done\n",
            getpid(), exec_count, tmpl_count);
}

/* ---- comm table: app_handle (stable, what the framework holds) ->
 *      cur_handle (live comm, changes across destroy/recreate) ------------ */
#define MAX_COMMS 64
typedef struct {
    ncclComm_t app_handle;
    ncclComm_t cur_handle;
    int nranks;
    int rank;
    int cudev;       /* cached for query calls while destroyed */
    int destroyed;
} comm_rec_t;

static comm_rec_t comms[MAX_COMMS];
static int n_comms = 0;
static int generation = 0;              /* bumped per recreate cycle */
static volatile sig_atomic_t need_recreate = 0;
static pthread_mutex_t recreate_mtx = PTHREAD_MUTEX_INITIALIZER;

static comm_rec_t* find_rec(ncclComm_t app) {
    for (int i = 0; i < n_comms; i++)
        if (comms[i].app_handle == app) return &comms[i];
    return NULL;
}

static ncclComm_t xlate(ncclComm_t c) {
    comm_rec_t* r = find_rec(c);
    return r ? r->cur_handle : c;
}

static const char* rdir(void) {
    const char* d = getenv("CR_RENDEZVOUS_DIR");
    return d && d[0] ? d : "/dev/shm/cr-rendezvous";
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- init interception: record creation params -------------------------- */

static void track(ncclComm_t comm, int nranks, int rank) {
    if (n_comms >= MAX_COMMS) {
        fprintf(stderr, "[cr-shim2] PID %d: comm table full!\n", getpid());
        return;
    }
    comm_rec_t* r = &comms[n_comms++];
    r->app_handle = comm;
    r->cur_handle = comm;
    r->nranks = nranks;
    r->rank = rank;
    r->cudev = -1;
    r->destroyed = 0;
    if (real_CommCuDevice) real_CommCuDevice(comm, &r->cudev);
    fprintf(stderr, "[cr-shim2] PID %d: tracked comm %p (#%d, rank=%d/%d, dev=%d)\n",
            getpid(), comm, n_comms, rank, nranks, r->cudev);
}

ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId id, int rank) {
    resolve_all();
    ncclResult_t r = real_InitRank(comm, nranks, id, rank);
    if (r == 0 && comm && *comm) track(*comm, nranks, rank);
    return r;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId id,
                                    int rank, void* config) {
    resolve_all();
    ncclResult_t r = real_InitRankConfig(comm, nranks, id, rank, config);
    /* nonblocking configs return ncclInProgress (7) with a valid comm */
    if ((r == 0 || r == 7) && comm && *comm) track(*comm, nranks, rank);
    return r;
}

/* ---- destroy: SIGRTMIN+1 -------------------------------------------------
 * ncclCommDestroy tears down ALL transport state — P2P imports/exports,
 * SHM segments, proxy threads. After this the process has no cross-process
 * GPU state and cuda-checkpoint can freeze it, even for NVLink workloads.
 * Runs in signal-handler context: the workload is quiesced (main thread
 * parked in sleep, no NCCL/CUDA calls in flight), which makes this safe in
 * practice. */
static void destroy_handler(int sig) {
    (void)sig;
    double t0 = now_ms();
    resolve_all();
    /* When using cuMemUnmap-based sleep (not cuda-checkpoint), CUDA graphs
     * can survive — they reference virtual addresses which stay reserved.
     * Only reset graphs when CR_RESET_GRAPHS=1 (cuda-checkpoint path).
     * For the sleep/wake path, skip graph reset to keep them alive. */
    if (getenv("CR_RESET_GRAPHS") && getenv("CR_RESET_GRAPHS")[0] == '1')
        reset_all_graphs();
    fprintf(stderr, "[cr-shim2] PID %d: DESTROY — %d comms\n", getpid(), n_comms);
    for (int i = 0; i < n_comms; i++) {
        if (comms[i].destroyed) continue;
        int rc = real_CommDestroy(comms[i].cur_handle);
        fprintf(stderr, "[cr-shim2] PID %d:   comm %p destroy rc=%d\n",
                getpid(), comms[i].cur_handle, rc);
        comms[i].destroyed = 1;
    }

    /* Synchronize all devices to ensure destroy is fully flushed */
    typedef int (*fnDevSync)(void);
    fnDevSync dev_sync = (fnDevSync)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    if (dev_sync) dev_sync();

    fprintf(stderr, "[cr-shim2] PID %d: destroy done %.1fms\n", getpid(), now_ms() - t0);
}

/* ---- recreate arm: SIGRTMIN+2 --------------------------------------------
 * Async-signal-safe: only sets a flag. The heavy lifting (rendezvous +
 * collective ncclCommInitRank) happens lazily on the app's own thread at
 * the first intercepted collective call — a safe context for the thread
 * spawns / mallocs / socket work that NCCL init does. */
static void recreate_arm_handler(int sig) {
    (void)sig;
    need_recreate = 1;
}

/* ---- lazy recreate: runs on the app thread ------------------------------- */
static void do_recreate_locked(void) {
    double t0 = now_ms();
    resolve_all();
    generation++;
    mkdir(rdir(), 0777);
    fprintf(stderr, "[cr-shim2] PID %d: RECREATE (lazy, on app thread) — %d comms (gen=%d)\n",
            getpid(), n_comms, generation);

    for (int i = 0; i < n_comms; i++) {
        if (!comms[i].destroyed) continue;

        char path[480], tmp[512];
        snprintf(path, sizeof(path), "%s/uid_%d_%d", rdir(), i, generation);
        ncclUniqueId uid;

        if (comms[i].rank == 0) {
            int rc = real_GetUniqueId(&uid);
            if (rc != 0) {
                fprintf(stderr, "[cr-shim2] PID %d:   GetUniqueId failed rc=%d\n", getpid(), rc);
                continue;
            }
            snprintf(tmp, sizeof(tmp), "%s.tmp", path);
            int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0666);
            if (fd < 0) {
                fprintf(stderr, "[cr-shim2] PID %d:   cannot write %s: %s\n",
                        getpid(), tmp, strerror(errno));
                continue;
            }
            ssize_t w = write(fd, &uid, sizeof(uid));
            close(fd);
            if (w != sizeof(uid)) { fprintf(stderr, "[cr-shim2] short write\n"); continue; }
            rename(tmp, path);   /* atomic publish */
            fprintf(stderr, "[cr-shim2] PID %d:   comm #%d: published fresh uid\n", getpid(), i);
        } else {
            int got = 0;
            for (int tries = 0; tries < 6000; tries++) {   /* up to 60s */
                int fd = open(path, O_RDONLY);
                if (fd >= 0) {
                    ssize_t rd = read(fd, &uid, sizeof(uid));
                    close(fd);
                    if (rd == sizeof(uid)) { got = 1; break; }
                }
                usleep(10000);
            }
            if (!got) {
                fprintf(stderr, "[cr-shim2] PID %d:   comm #%d: rendezvous TIMEOUT (%s)\n",
                        getpid(), i, path);
                continue;
            }
            fprintf(stderr, "[cr-shim2] PID %d:   comm #%d: got fresh uid\n", getpid(), i);
        }

        /* Synchronize all CUDA devices before re-init. After
         * cuda-checkpoint restore, the driver-level CUDA state may
         * not be fully flushed. Without this barrier, NCCL's CommCheck
         * sees partially-restored state and fails with rc=6
         * ("corrupted comm object"). */
        typedef int (*fnDevSync)(void);
        static fnDevSync dev_sync;
        if (!dev_sync) dev_sync = (fnDevSync)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
        if (dev_sync) dev_sync();

        /* collective re-init — every rank reaches here from its first
         * post-resume collective */
        ncclComm_t nc = NULL;
        int rc = real_InitRank(&nc, comms[i].nranks, uid, comms[i].rank);
        if (rc != 0 || !nc) {
            fprintf(stderr, "[cr-shim2] PID %d:   comm #%d: re-init FAILED rc=%d\n",
                    getpid(), i, rc);
            continue;
        }
        fprintf(stderr, "[cr-shim2] PID %d:   comm #%d: recreated %p -> %p (rank=%d/%d)\n",
                getpid(), i, comms[i].app_handle, nc, comms[i].rank, comms[i].nranks);
        comms[i].cur_handle = nc;
        comms[i].destroyed = 0;
    }
    fprintf(stderr, "[cr-shim2] PID %d: recreate done %.1fms\n", getpid(), now_ms() - t0);
}

static void ensure_live(void) {
    if (!need_recreate) return;
    pthread_mutex_lock(&recreate_mtx);
    if (need_recreate) {
        do_recreate_locked();
        need_recreate = 0;
    }
    pthread_mutex_unlock(&recreate_mtx);
}

/* ---- translated call surface --------------------------------------------
 * Collectives trigger the lazy recreate; query calls are answered from
 * cache while the comm is destroyed (keeps PyTorch's watchdog off a dead
 * comm during the C/R window). */

ncclResult_t ncclAllReduce(const void* s, void* r, size_t n, int dt, int op,
                           ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_AllReduce(s, r, n, dt, op, xlate(c), st);
}
ncclResult_t ncclBroadcast(const void* s, void* r, size_t n, int dt, int root,
                           ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_Broadcast(s, r, n, dt, root, xlate(c), st);
}
ncclResult_t ncclBcast(void* b, size_t n, int dt, int root, ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_Bcast(b, n, dt, root, xlate(c), st);
}
ncclResult_t ncclReduce(const void* s, void* r, size_t n, int dt, int op, int root,
                        ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_Reduce(s, r, n, dt, op, root, xlate(c), st);
}
ncclResult_t ncclAllGather(const void* s, void* r, size_t n, int dt,
                           ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_AllGather(s, r, n, dt, xlate(c), st);
}
ncclResult_t ncclReduceScatter(const void* s, void* r, size_t n, int dt, int op,
                               ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_ReduceScatter(s, r, n, dt, op, xlate(c), st);
}
ncclResult_t ncclSend(const void* b, size_t n, int dt, int peer, ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_Send(b, n, dt, peer, xlate(c), st);
}
ncclResult_t ncclRecv(void* b, size_t n, int dt, int peer, ncclComm_t c, void* st) {
    resolve_all(); ensure_live();
    return real_Recv(b, n, dt, peer, xlate(c), st);
}

/* query calls: serve from cache while destroyed */
ncclResult_t ncclCommCount(const ncclComm_t c, int* out) {
    resolve_all();
    comm_rec_t* rec = find_rec((ncclComm_t)c);
    if (rec && rec->destroyed) { if (out) *out = rec->nranks; return 0; }
    return real_CommCount(rec ? rec->cur_handle : (ncclComm_t)c, out);
}
ncclResult_t ncclCommCuDevice(const ncclComm_t c, int* out) {
    resolve_all();
    comm_rec_t* rec = find_rec((ncclComm_t)c);
    if (rec && rec->destroyed) { if (out) *out = rec->cudev; return 0; }
    return real_CommCuDevice(rec ? rec->cur_handle : (ncclComm_t)c, out);
}
ncclResult_t ncclCommUserRank(const ncclComm_t c, int* out) {
    resolve_all();
    comm_rec_t* rec = find_rec((ncclComm_t)c);
    if (rec && rec->destroyed) { if (out) *out = rec->rank; return 0; }
    return real_CommUserRank(rec ? rec->cur_handle : (ncclComm_t)c, out);
}
ncclResult_t ncclCommGetAsyncError(ncclComm_t c, ncclResult_t* out) {
    resolve_all();
    comm_rec_t* rec = find_rec(c);
    if (rec && rec->destroyed) { if (out) *out = 0; return 0; }   /* report healthy */
    return real_CommGetAsyncError(rec ? rec->cur_handle : c, out);
}
ncclResult_t ncclCommRegister(const ncclComm_t c, void* buf, size_t n, void** h) {
    resolve_all(); ensure_live();
    return real_CommRegister(xlate((ncclComm_t)c), buf, n, h);
}
ncclResult_t ncclCommDeregister(const ncclComm_t c, void* h) {
    resolve_all();
    comm_rec_t* rec = find_rec((ncclComm_t)c);
    if (rec && rec->destroyed) return 0;   /* registration died with the comm */
    return real_CommDeregister(rec ? rec->cur_handle : (ncclComm_t)c, h);
}

/* app-driven lifecycle: translate and drop our record */
static void untrack(ncclComm_t app) {
    for (int i = 0; i < n_comms; i++) {
        if (comms[i].app_handle == app) {
            comms[i] = comms[--n_comms];
            return;
        }
    }
}
ncclResult_t ncclCommDestroy(ncclComm_t c) {
    resolve_all();
    comm_rec_t* rec = find_rec(c);
    if (rec && rec->destroyed) { untrack(c); return 0; }   /* already gone */
    ncclComm_t live = xlate(c);
    untrack(c);
    return real_CommDestroy(live);
}
ncclResult_t ncclCommAbort(ncclComm_t c) {
    resolve_all();
    comm_rec_t* rec = find_rec(c);
    if (rec && rec->destroyed) { untrack(c); return 0; }
    ncclComm_t live = xlate(c);
    untrack(c);
    return real_CommAbort(live);
}
ncclResult_t ncclCommFinalize(ncclComm_t c) {
    resolve_all();
    comm_rec_t* rec = find_rec(c);
    if (rec && rec->destroyed) return 0;
    return real_CommFinalize(rec ? rec->cur_handle : c);
}

/* ---- PyNCCL pass-through surface ------------------------------------------
 * vLLM's PyNCCL loads one .so via ctypes and dlsym's every NCCL function from
 * that handle — LD_PRELOAD interposition does not apply. Setting
 * VLLM_NCCL_SO_PATH to THIS shim routes PyNCCL through us instead, so its
 * comms are tracked/translated like everything else. These exports complete
 * the surface PyNCCL needs; they forward to the real NCCL. */
typedef ncclResult_t (*fnGetVersion)(int*);
typedef const char*  (*fnGetErrorString)(int);
typedef const char*  (*fnGetLastError)(ncclComm_t);
typedef ncclResult_t (*fnGroupStart)(void);
typedef ncclResult_t (*fnGroupEnd)(void);
typedef ncclResult_t (*fnMemAlloc)(void**, size_t);
typedef ncclResult_t (*fnMemFree)(void*);
typedef ncclResult_t (*fnCommWindowRegister)(ncclComm_t, void*, size_t, void**, int);
typedef ncclResult_t (*fnCommWindowDeregister)(ncclComm_t, void*);

static fnGetVersion            real_GetVersion;
static fnGetErrorString        real_GetErrorString;
static fnGetLastError          real_GetLastError;
static fnGroupStart            real_GroupStart;
static fnGroupEnd              real_GroupEnd;
static fnMemAlloc              real_MemAlloc;
static fnMemFree               real_MemFree;
static fnCommWindowRegister    real_CommWindowRegister;
static fnCommWindowDeregister  real_CommWindowDeregister;

ncclResult_t ncclGetVersion(int* v) {
    RESOLVE(real_GetVersion, fnGetVersion, "ncclGetVersion");
    return real_GetVersion ? real_GetVersion(v) : 3;
}
const char* ncclGetErrorString(int rc) {
    RESOLVE(real_GetErrorString, fnGetErrorString, "ncclGetErrorString");
    return real_GetErrorString ? real_GetErrorString(rc) : "unknown";
}
const char* ncclGetLastError(ncclComm_t c) {
    RESOLVE(real_GetLastError, fnGetLastError, "ncclGetLastError");
    return real_GetLastError ? real_GetLastError(xlate(c)) : "";
}
ncclResult_t ncclGroupStart(void) {
    RESOLVE(real_GroupStart, fnGroupStart, "ncclGroupStart");
    ensure_live();   /* recreate must not happen inside a group */
    return real_GroupStart();
}
ncclResult_t ncclGroupEnd(void) {
    RESOLVE(real_GroupEnd, fnGroupEnd, "ncclGroupEnd");
    return real_GroupEnd();
}
ncclResult_t ncclGetUniqueId(ncclUniqueId* id) {
    resolve_all();
    return real_GetUniqueId(id);
}
ncclResult_t ncclMemAlloc(void** p, size_t n) {
    RESOLVE(real_MemAlloc, fnMemAlloc, "ncclMemAlloc");
    return real_MemAlloc(p, n);
}
ncclResult_t ncclMemFree(void* p) {
    RESOLVE(real_MemFree, fnMemFree, "ncclMemFree");
    return real_MemFree(p);
}
ncclResult_t ncclCommWindowRegister(ncclComm_t c, void* buf, size_t n, void** win, int flags) {
    RESOLVE(real_CommWindowRegister, fnCommWindowRegister, "ncclCommWindowRegister");
    ensure_live();
    return real_CommWindowRegister(xlate(c), buf, n, win, flags);
}
ncclResult_t ncclCommWindowDeregister(ncclComm_t c, void* win) {
    RESOLVE(real_CommWindowDeregister, fnCommWindowDeregister, "ncclCommWindowDeregister");
    comm_rec_t* rec = find_rec(c);
    if (rec && rec->destroyed) return 0;
    return real_CommWindowDeregister(rec ? rec->cur_handle : c, win);
}

/* ---- constructor --------------------------------------------------------- */
__attribute__((constructor))
static void init_cr_shim2(void) {
    struct sigaction sa = {0};
    sa.sa_handler = destroy_handler;
    sigaction(SIGRTMIN + 1, &sa, NULL);
    sa.sa_handler = recreate_arm_handler;
    sigaction(SIGRTMIN + 2, &sa, NULL);
    fprintf(stderr, "[cr-shim2] PID %d: ready (destroy=sig%d, recreate-arm=sig%d, rendezvous=%s)\n",
            getpid(), SIGRTMIN + 1, SIGRTMIN + 2, rdir());
}
