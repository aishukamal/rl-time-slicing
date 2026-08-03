/*
 * mc_test.c — Isolate whether cuMulticastCreate works after cuda-checkpoint
 * restore, independent of NCCL.
 *
 * Sequence:
 *   1. Init CUDA, create context on dev 0
 *   2. cuMulticastCreate ("PRE")  — expected to succeed (NVLS-capable node)
 *   3. Print PID, wait for /tmp/mc_go (external driver freezes+restores us here)
 *   4. cuMulticastCreate ("POST") — the verdict:
 *        succeeds -> NOT a driver limitation; NCCL-internal state is the blocker
 *        fails    -> confirmed driver limitation, minimal repro for NVIDIA
 *
 * Build: gcc -o mc_test mc_test.c -I/usr/local/cuda/include \
 *            -L/usr/local/nvidia/lib64 -lcuda
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void try_multicast(const char* tag) {
    CUmulticastObjectProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.numDevices = 2;
    prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.flags = 0;

    size_t gran = 0;
    CUresult r = cuMulticastGetGranularity(&gran, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED);
    const char* es = "?";
    if (r != CUDA_SUCCESS) {
        cuGetErrorString(r, &es);
        printf("[%s] cuMulticastGetGranularity rc=%d (%s)\n", tag, r, es);
        fflush(stdout);
        return;
    }
    prop.size = gran;

    CUmemGenericAllocationHandle mc = 0;
    r = cuMulticastCreate(&mc, &prop);
    cuGetErrorString(r, &es);
    printf("[%s] cuMulticastCreate(size=%zu, ndev=2) rc=%d (%s)\n", tag, gran, r, es);
    fflush(stdout);

    if (r == CUDA_SUCCESS) {
        /* also exercise AddDevice — nvlsAllocateMem does this next */
        CUdevice dev0, dev1;
        cuDeviceGet(&dev0, 0);
        cuDeviceGet(&dev1, 1);
        CUresult r2 = cuMulticastAddDevice(mc, dev0);
        cuGetErrorString(r2, &es);
        printf("[%s] cuMulticastAddDevice(dev0) rc=%d (%s)\n", tag, r2, es);
        r2 = cuMulticastAddDevice(mc, dev1);
        cuGetErrorString(r2, &es);
        printf("[%s] cuMulticastAddDevice(dev1) rc=%d (%s)\n", tag, r2, es);
        fflush(stdout);
        cuMemRelease(mc);
    }
}

#include <string.h>

int main(void) {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { printf("cuInit failed rc=%d\n", r); return 1; }

    CUdevice dev;
    CUcontext ctx;
    cuDeviceGet(&dev, 0);
    r = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (r != CUDA_SUCCESS) { printf("ctx retain failed rc=%d\n", r); return 1; }
    cuCtxSetCurrent(ctx);

    /* touch the GPU so cuda-checkpoint has real state to save */
    CUdeviceptr p;
    cuMemAlloc(&p, 1 << 20);

    /* SKIP_PRE=1: never touch multicast before the checkpoint. Disambiguates
     * "driver cannot bind multicast devices in any restored process" from
     * "prior multicast state poisons the checkpoint image" (NCCL #2117
     * describes silent cuMulticastUnbind failures leaving stale bindings). */
    if (!getenv("SKIP_PRE"))
        try_multicast("PRE");
    else
        printf("[PRE] skipped (SKIP_PRE=1)\n");

    printf("PID %d waiting for /tmp/mc_go (freeze+restore me now)\n", getpid());
    fflush(stdout);
    while (access("/tmp/mc_go", F_OK) != 0) usleep(200000);

    try_multicast("POST");

    /* Is the staleness context-scoped or process-wide? Try a brand-new
     * standalone context (not the primary one cuda-checkpoint restored). */
    /* CUDA 13 headers remap cuCtxCreate to a 4-arg _v4; use the stable v2 ABI */
    CUresult cuCtxCreate_v2(CUcontext*, unsigned int, CUdevice);
    CUcontext fresh;
    r = cuCtxCreate_v2(&fresh, 0, dev);
    if (r == CUDA_SUCCESS) {
        cuCtxSetCurrent(fresh);
        try_multicast("POST-FRESH-CTX");
        cuCtxDestroy(fresh);
    } else {
        printf("[POST-FRESH-CTX] cuCtxCreate failed rc=%d\n", r);
    }

    printf("done\n");
    fflush(stdout);
    return 0;
}
