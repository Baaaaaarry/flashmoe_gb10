#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void die(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "cuda_router_readback_smoke: %s: %s\n", what, cudaGetErrorString(err));
    exit(1);
}

__global__ static void write_value_kernel(int32_t *dst, int32_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dst[0] = value;
    }
}

static void run_device_copy_mode(int iters) {
    int32_t *device_ptr = NULL;
    int32_t *host_pinned = NULL;
    die(cudaMalloc(&device_ptr, sizeof(int32_t)), "cudaMalloc(device_ptr)");
    die(cudaHostAlloc(&host_pinned, sizeof(int32_t), cudaHostAllocPortable), "cudaHostAlloc(host_pinned)");

    double total_wait_ms = 0.0;
    double total_copy_ms = 0.0;
    double max_wait_ms = 0.0;
    double max_copy_ms = 0.0;
    int32_t sink = 0;

    for (int i = 0; i < iters; i++) {
        write_value_kernel<<<1, 1>>>(device_ptr, i);
        die(cudaGetLastError(), "write_value_kernel launch");

        const double t_wait0 = now_ms();
        die(cudaDeviceSynchronize(), "cudaDeviceSynchronize(device)");
        const double wait_ms = now_ms() - t_wait0;

        const double t_copy0 = now_ms();
        die(cudaMemcpy(host_pinned, device_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        const double copy_ms = now_ms() - t_copy0;

        sink ^= host_pinned[0];
        total_wait_ms += wait_ms;
        total_copy_ms += copy_ms;
        if (wait_ms > max_wait_ms) max_wait_ms = wait_ms;
        if (copy_ms > max_copy_ms) max_copy_ms = copy_ms;
    }

    printf("mode=device_sync_copy iterations=%d avg_wait_ms=%.6f avg_copy_ms=%.6f max_wait_ms=%.6f max_copy_ms=%.6f sink=%d\n",
           iters,
           total_wait_ms / (double)iters,
           total_copy_ms / (double)iters,
           max_wait_ms,
           max_copy_ms,
           sink);

    die(cudaFree(device_ptr), "cudaFree(device_ptr)");
    die(cudaFreeHost(host_pinned), "cudaFreeHost(host_pinned)");
}

static void run_mapped_host_mode(int iters) {
    int32_t *host_ptr = NULL;
    int32_t *device_alias = NULL;
    die(cudaHostAlloc(&host_ptr, sizeof(int32_t), cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc(mapped)");
    die(cudaHostGetDevicePointer((void **)&device_alias, host_ptr, 0), "cudaHostGetDevicePointer");

    double total_wait_ms = 0.0;
    double total_read_ms = 0.0;
    double max_wait_ms = 0.0;
    double max_read_ms = 0.0;
    int32_t sink = 0;

    for (int i = 0; i < iters; i++) {
        write_value_kernel<<<1, 1>>>(device_alias, i);
        die(cudaGetLastError(), "write_value_kernel(mapped) launch");

        const double t_wait0 = now_ms();
        die(cudaDeviceSynchronize(), "cudaDeviceSynchronize(mapped)");
        const double wait_ms = now_ms() - t_wait0;

        const double t_read0 = now_ms();
        sink ^= host_ptr[0];
        const double read_ms = now_ms() - t_read0;

        total_wait_ms += wait_ms;
        total_read_ms += read_ms;
        if (wait_ms > max_wait_ms) max_wait_ms = wait_ms;
        if (read_ms > max_read_ms) max_read_ms = read_ms;
    }

    printf("mode=mapped_sync_cpu_read iterations=%d avg_wait_ms=%.6f avg_read_ms=%.6f max_wait_ms=%.6f max_read_ms=%.6f sink=%d\n",
           iters,
           total_wait_ms / (double)iters,
           total_read_ms / (double)iters,
           max_wait_ms,
           max_read_ms,
           sink);

    die(cudaFreeHost(host_ptr), "cudaFreeHost(mapped)");
}

int main(int argc, char **argv) {
    int iters = 10000;
    if (argc >= 2) {
        iters = atoi(argv[1]);
        if (iters <= 0) iters = 10000;
    }
    run_device_copy_mode(iters);
    run_mapped_host_mode(iters);
    return 0;
}
