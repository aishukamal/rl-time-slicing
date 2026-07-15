// Standalone selective C/R validation for GPU-CR.
//
// Allocates three GPU buffers with distinct patterns via (hooked) cudaMalloc,
// then waits for an external driver to run:
//   cr_client -i -p <pid>
//   cr_client -c -s <ptrA>:<size> -p <pid>     (evict buffer A only)
//   cr_client -r -s <ptrA>:<size> -p <pid>     (restore buffer A)
// Between checkpoint and restore this program does NOT touch buffer A
// (that's the eviction contract), but it DOES verify B and C remain
// readable and correct while A is evicted.
//
// Protocol via files in $TEST_DIR (default /tmp/seltest):
//   writes  ready        "<pid> <ptrA> <ptrB> <ptrC> <size>"
//   waits   ckpt_done    (driver creates after selective checkpoint)
//   writes  midcheck     "OK" or "FAIL ..." (B/C integrity during eviction)
//   waits   restore_done (driver creates after selective restore)
//   writes  result       "PASS" or "FAIL ..."

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <sys/stat.h>

#define BUF_ELEMS (2 * 1024 * 1024)  // 2M floats = 8MB
#define BUF_BYTES (BUF_ELEMS * sizeof(float))

static const char* test_dir() {
    const char* d = getenv("TEST_DIR");
    return d ? d : "/tmp/seltest";
}

static void write_file(const char* name, const std::string& content) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", test_dir(), name);
    FILE* f = fopen(path, "w");
    if (!f) { perror("fopen"); exit(1); }
    fputs(content.c_str(), f);
    fclose(f);
}

static void wait_file(const char* name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", test_dir(), name);
    fprintf(stderr, "[test] waiting for %s ...\n", path);
    struct stat st;
    while (stat(path, &st) != 0) usleep(200 * 1000);
    fprintf(stderr, "[test] %s appeared\n", path);
}

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "[test] CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

static void fill_pattern(float* host, float base) {
    for (int i = 0; i < BUF_ELEMS; i++) host[i] = base + (float)(i % 65536);
}

static std::string check_buf(void* dev, float base, const char* label) {
    static float* host = nullptr;
    if (!host) host = (float*)malloc(BUF_BYTES);
    cudaError_t e = cudaMemcpy(host, dev, BUF_BYTES, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        return std::string("FAIL ") + label + " memcpy: " + cudaGetErrorString(e);
    }
    for (int i = 0; i < BUF_ELEMS; i++) {
        float expect = base + (float)(i % 65536);
        if (host[i] != expect) {
            char msg[256];
            snprintf(msg, sizeof(msg), "FAIL %s mismatch at %d: got %f expect %f",
                     label, i, host[i], expect);
            return std::string(msg);
        }
    }
    return "";
}

int main() {
    mkdir(test_dir(), 0777);

    // Establish the primary CUDA context on this thread BEFORE the first
    // (LD_PRELOAD-hooked) cudaMalloc — the hook's cuMem* VMM calls need a
    // current context, which PyTorch normally provides but a raw program
    // must set up itself. cudaSetDevice is not intercepted.
    CK(cudaSetDevice(0));
    CK(cudaFree(0));

    float* host = (float*)malloc(BUF_BYTES);

    void *A, *B, *C;
    CK(cudaMalloc(&A, BUF_BYTES));
    CK(cudaMalloc(&B, BUF_BYTES));
    CK(cudaMalloc(&C, BUF_BYTES));

    fill_pattern(host, 1000.0f); CK(cudaMemcpy(A, host, BUF_BYTES, cudaMemcpyHostToDevice));
    fill_pattern(host, 2000.0f); CK(cudaMemcpy(B, host, BUF_BYTES, cudaMemcpyHostToDevice));
    fill_pattern(host, 3000.0f); CK(cudaMemcpy(C, host, BUF_BYTES, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());

    size_t free0, total;
    CK(cudaMemGetInfo(&free0, &total));
    fprintf(stderr, "[test] buffers ready. A=%p B=%p C=%p size=%zu free=%zu MB\n",
            A, B, C, (size_t)BUF_BYTES, free0 >> 20);

    char ready[512];
    snprintf(ready, sizeof(ready), "%d %p %p %p %zu\n", getpid(), A, B, C, (size_t)BUF_BYTES);
    write_file("ready", ready);

    // --- Driver: init + selective checkpoint of A happens here ---
    wait_file("ckpt_done");

    // While A is evicted: B and C must still be fully intact and usable.
    std::string err_b = check_buf(B, 2000.0f, "B(during-evict)");
    std::string err_c = check_buf(C, 3000.0f, "C(during-evict)");
    size_t free1;
    CK(cudaMemGetInfo(&free1, &total));
    fprintf(stderr, "[test] during evict: free=%zu MB (was %zu MB)\n", free1 >> 20, free0 >> 20);
    if (!err_b.empty() || !err_c.empty()) {
        write_file("midcheck", err_b + " " + err_c);
        write_file("result", "FAIL during-evict integrity");
        return 1;
    }
    write_file("midcheck", "OK\n");

    // --- Driver: selective restore of A happens here ---
    wait_file("restore_done");

    // All three buffers must now be intact, at the SAME pointers.
    std::string ea = check_buf(A, 1000.0f, "A(after-restore)");
    std::string eb = check_buf(B, 2000.0f, "B(after-restore)");
    std::string ec = check_buf(C, 3000.0f, "C(after-restore)");

    // Also verify A is writable/usable after restore (fresh physical pages).
    if (ea.empty()) {
        fill_pattern(host, 5000.0f);
        CK(cudaMemcpy(A, host, BUF_BYTES, cudaMemcpyHostToDevice));
        ea = check_buf(A, 5000.0f, "A(rewrite)");
    }

    if (ea.empty() && eb.empty() && ec.empty()) {
        fprintf(stderr, "[test] PASS\n");
        write_file("result", "PASS\n");
        return 0;
    }
    fprintf(stderr, "[test] %s %s %s\n", ea.c_str(), eb.c_str(), ec.c_str());
    write_file("result", ea + " " + eb + " " + ec + "\n");
    return 1;
}
