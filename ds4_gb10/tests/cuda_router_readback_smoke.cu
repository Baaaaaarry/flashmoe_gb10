#include <cuda_runtime.h>

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    ROUTER_IN_DIM = 4096,
    ROUTER_N_EXPERT = 256,
    ROUTER_TOP_K = 6,
};

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

__global__ static void router_logits_kernel(
        float       *logits,
        const float *x,
        const float *w,
        uint32_t     in_dim,
        uint32_t     n_expert) {
    const uint32_t e = (uint32_t)threadIdx.x + (uint32_t)blockIdx.x * (uint32_t)blockDim.x;
    if (e >= n_expert) return;
    float acc = 0.0f;
    const float *row = w + (uint64_t)e * in_dim;
    for (uint32_t i = 0; i < in_dim; i++) {
        acc += row[i] * x[i];
    }
    logits[e] = acc;
}

__global__ static void router_topk6_kernel(
        int32_t       *selected,
        float         *weights,
        const float   *logits,
        uint32_t       n_expert) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float best_v[ROUTER_TOP_K];
    int32_t best_i[ROUTER_TOP_K];
    for (int k = 0; k < ROUTER_TOP_K; k++) {
        best_v[k] = -INFINITY;
        best_i[k] = -1;
    }
    for (uint32_t i = 0; i < n_expert; i++) {
        const float v = logits[i];
        if (v <= best_v[ROUTER_TOP_K - 1]) continue;
        int pos = ROUTER_TOP_K - 1;
        while (pos > 0 && v > best_v[pos - 1]) {
            best_v[pos] = best_v[pos - 1];
            best_i[pos] = best_i[pos - 1];
            pos--;
        }
        best_v[pos] = v;
        best_i[pos] = (int32_t)i;
    }
    float sum = 0.0f;
    for (int k = 0; k < ROUTER_TOP_K; k++) {
        selected[k] = best_i[k];
        weights[k] = expf(best_v[k]);
        sum += weights[k];
    }
    if (sum > 0.0f) {
        for (int k = 0; k < ROUTER_TOP_K; k++) {
            weights[k] /= sum;
        }
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

static void init_host_buffer(float *buf, uint64_t n, float scale) {
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t v = (uint32_t)(i % 251u);
        buf[i] = ((float)v / 251.0f - 0.5f) * scale;
    }
}

static void run_router_like_mode(int iters) {
    float *x_h = NULL, *w_h = NULL;
    float *x_d = NULL, *w_d = NULL, *logits_d = NULL, *weights_d = NULL;
    int32_t *selected_d = NULL, *selected_h = NULL;
    const uint64_t x_bytes = (uint64_t)ROUTER_IN_DIM * sizeof(float);
    const uint64_t w_bytes = (uint64_t)ROUTER_IN_DIM * ROUTER_N_EXPERT * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)ROUTER_N_EXPERT * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)ROUTER_TOP_K * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)ROUTER_TOP_K * sizeof(float);

    x_h = (float *)malloc((size_t)x_bytes);
    w_h = (float *)malloc((size_t)w_bytes);
    if (!x_h || !w_h) {
        fprintf(stderr, "cuda_router_readback_smoke: host alloc failed\n");
        exit(1);
    }
    init_host_buffer(x_h, ROUTER_IN_DIM, 1.0f);
    init_host_buffer(w_h, (uint64_t)ROUTER_IN_DIM * ROUTER_N_EXPERT, 0.25f);

    die(cudaMalloc(&x_d, x_bytes), "cudaMalloc(x_d)");
    die(cudaMalloc(&w_d, w_bytes), "cudaMalloc(w_d)");
    die(cudaMalloc(&logits_d, logits_bytes), "cudaMalloc(logits_d)");
    die(cudaMalloc(&selected_d, selected_bytes), "cudaMalloc(selected_d)");
    die(cudaMalloc(&weights_d, weights_bytes), "cudaMalloc(weights_d)");
    die(cudaHostAlloc(&selected_h, selected_bytes, cudaHostAllocPortable), "cudaHostAlloc(selected_h)");
    die(cudaMemcpy(x_d, x_h, x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(x)");
    die(cudaMemcpy(w_d, w_h, w_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(w)");

    double total_kernel_ms = 0.0;
    double total_wait_ms = 0.0;
    double total_copy_ms = 0.0;
    double max_kernel_ms = 0.0;
    double max_wait_ms = 0.0;
    double max_copy_ms = 0.0;
    int32_t sink = 0;

    for (int i = 0; i < iters; i++) {
        const double t_kernel0 = now_ms();
        router_logits_kernel<<<1, ROUTER_N_EXPERT>>>(logits_d, x_d, w_d, ROUTER_IN_DIM, ROUTER_N_EXPERT);
        die(cudaGetLastError(), "router_logits_kernel launch");
        router_topk6_kernel<<<1, 1>>>(selected_d, weights_d, logits_d, ROUTER_N_EXPERT);
        die(cudaGetLastError(), "router_topk6_kernel launch");
        const double kernel_ms = now_ms() - t_kernel0;

        const double t_wait0 = now_ms();
        die(cudaDeviceSynchronize(), "cudaDeviceSynchronize(router_like)");
        const double wait_ms = now_ms() - t_wait0;

        const double t_copy0 = now_ms();
        die(cudaMemcpy(selected_h, selected_d, selected_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy router selected");
        const double copy_ms = now_ms() - t_copy0;

        for (int k = 0; k < ROUTER_TOP_K; k++) sink ^= selected_h[k];
        total_kernel_ms += kernel_ms;
        total_wait_ms += wait_ms;
        total_copy_ms += copy_ms;
        if (kernel_ms > max_kernel_ms) max_kernel_ms = kernel_ms;
        if (wait_ms > max_wait_ms) max_wait_ms = wait_ms;
        if (copy_ms > max_copy_ms) max_copy_ms = copy_ms;
    }

    printf("mode=router_like_sync_copy iterations=%d avg_kernel_ms=%.6f avg_wait_ms=%.6f avg_copy_ms=%.6f max_kernel_ms=%.6f max_wait_ms=%.6f max_copy_ms=%.6f sink=%d\n",
           iters,
           total_kernel_ms / (double)iters,
           total_wait_ms / (double)iters,
           total_copy_ms / (double)iters,
           max_kernel_ms,
           max_wait_ms,
           max_copy_ms,
           sink);

    die(cudaFree(x_d), "cudaFree(x_d)");
    die(cudaFree(w_d), "cudaFree(w_d)");
    die(cudaFree(logits_d), "cudaFree(logits_d)");
    die(cudaFree(selected_d), "cudaFree(selected_d)");
    die(cudaFree(weights_d), "cudaFree(weights_d)");
    die(cudaFreeHost(selected_h), "cudaFreeHost(selected_h)");
    free(x_h);
    free(w_h);
}

int main(int argc, char **argv) {
    int iters = 10000;
    if (argc >= 2) {
        iters = atoi(argv[1]);
        if (iters <= 0) iters = 10000;
    }
    run_device_copy_mode(iters);
    run_mapped_host_mode(iters);
    run_router_like_mode(iters);
    return 0;
}
