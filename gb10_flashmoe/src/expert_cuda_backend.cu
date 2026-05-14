#include "flashmoe/expert_cuda_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace flashmoe {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

__device__ float device_silu(float x) {
    if (!isfinite(x)) {
        return 0.0f;
    }
    if (x >= 0.0f) {
        const float z = expf(-x);
        return x / (1.0f + z);
    }
    const float z = expf(x);
    return x * z / (1.0f + z);
}

__device__ float finite_or_zero_device(float x) {
    return isfinite(x) ? x : 0.0f;
}

__global__ void matvec_rows_kernel(const float* weights,
                                   const float* input,
                                   float* output,
                                   std::size_t rows,
                                   std::size_t cols) {
    const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float accum = 0.0f;
    const std::size_t row_offset = row * cols;
    for (std::size_t col = 0; col < cols; ++col) {
        accum += finite_or_zero_device(weights[row_offset + col]) * finite_or_zero_device(input[col]);
    }
    output[row] = finite_or_zero_device(accum);
}

__global__ void silu_mul_kernel(float* gate, const float* up, std::size_t size) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    gate[idx] = finite_or_zero_device(device_silu(gate[idx]) * up[idx]);
}

__global__ void add_scaled_kernel(float* hidden,
                                  const float* update,
                                  std::size_t size,
                                  float scale) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }
    hidden[idx] = finite_or_zero_device(hidden[idx] + finite_or_zero_device(update[idx]) * scale);
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

bool cuda_expert_backend_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

double cuda_unpack_and_execute_expert(const ExpertRecord& record,
                                      DenseOperatorState& dense_state,
                                      const DenseRuntimeArtifact*,
                                      MaterializedExpertMap& materialized,
                                      double* unpack_ms,
                                      double* compute_ms,
                                      double* combine_ms,
                                      float route_weight) {
    auto& slot = materialized[record.id];
    if (slot.bytes.empty() && !slot.device_ready) {
        throw std::runtime_error("materialized expert bytes missing for CUDA execution");
    }
    if (!slot.host_ready) {
        throw std::runtime_error("expert weights must be host-unpacked before CUDA execution");
    }
    if (slot.hidden_dim != dense_state.hidden.size()) {
        throw std::runtime_error("expert hidden dim does not match runtime hidden dim");
    }

    const auto unpack_start = Clock::now();
    if (!slot.device_ready) {
        float* d_gate = nullptr;
        float* d_up = nullptr;
        float* d_down = nullptr;
        check_cuda(cudaMalloc(&d_gate, slot.gate_weights.size() * sizeof(float)), "cudaMalloc gate");
        check_cuda(cudaMalloc(&d_up, slot.up_weights.size() * sizeof(float)), "cudaMalloc up");
        check_cuda(cudaMalloc(&d_down, slot.down_weights.size() * sizeof(float)), "cudaMalloc down");
        check_cuda(cudaMemcpy(
            d_gate, slot.gate_weights.data(), slot.gate_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy gate");
        check_cuda(cudaMemcpy(
            d_up, slot.up_weights.data(), slot.up_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy up");
        check_cuda(cudaMemcpy(
            d_down, slot.down_weights.data(), slot.down_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy down");
        slot.device_gate = reinterpret_cast<std::uintptr_t>(d_gate);
        slot.device_up = reinterpret_cast<std::uintptr_t>(d_up);
        slot.device_down = reinterpret_cast<std::uintptr_t>(d_down);
        slot.device_gate_len = slot.gate_weights.size();
        slot.device_up_len = slot.up_weights.size();
        slot.device_down_len = slot.down_weights.size();
        slot.device_ready = true;
        slot.bytes.clear();
        slot.bytes.shrink_to_fit();
        slot.gate_weights.clear();
        slot.gate_weights.shrink_to_fit();
        slot.up_weights.clear();
        slot.up_weights.shrink_to_fit();
        slot.down_weights.clear();
        slot.down_weights.shrink_to_fit();
    }
    const auto unpack_end = Clock::now();

    float* d_input = nullptr;
    float* d_gate_out = nullptr;
    float* d_up_out = nullptr;
    float* d_out = nullptr;
    const std::size_t hidden_size = slot.hidden_dim;
    const std::size_t inter_size = slot.intermediate_dim;
    check_cuda(cudaMalloc(&d_input, hidden_size * sizeof(float)), "cudaMalloc input");
    check_cuda(cudaMalloc(&d_gate_out, inter_size * sizeof(float)), "cudaMalloc gate_out");
    check_cuda(cudaMalloc(&d_up_out, inter_size * sizeof(float)), "cudaMalloc up_out");
    check_cuda(cudaMalloc(&d_out, hidden_size * sizeof(float)), "cudaMalloc out");

    // FFN reads from moe_input (post-attn-norm); accumulate result into hidden (residual) on host.
    const auto& ffn_input = (dense_state.moe_input.size() == hidden_size)
        ? dense_state.moe_input : dense_state.hidden;
    check_cuda(cudaMemcpy(
        d_input, ffn_input.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy moe_input");

    const int threads = 256;
    const int inter_blocks = static_cast<int>((inter_size + threads - 1) / threads);
    const int hidden_blocks = static_cast<int>((hidden_size + threads - 1) / threads);

    const auto compute_start = Clock::now();
    matvec_rows_kernel<<<inter_blocks, threads>>>(
        reinterpret_cast<const float*>(slot.device_gate), d_input, d_gate_out, inter_size, hidden_size);
    matvec_rows_kernel<<<inter_blocks, threads>>>(
        reinterpret_cast<const float*>(slot.device_up), d_input, d_up_out, inter_size, hidden_size);
    silu_mul_kernel<<<inter_blocks, threads>>>(d_gate_out, d_up_out, inter_size);
    matvec_rows_kernel<<<hidden_blocks, threads>>>(
        reinterpret_cast<const float*>(slot.device_down), d_gate_out, d_out, hidden_size, inter_size);
    check_cuda(cudaGetLastError(), "expert ffns kernels");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize compute");
    const auto compute_end = Clock::now();

    const auto combine_start = Clock::now();
    std::vector<float> expert_out(hidden_size);
    check_cuda(cudaMemcpy(expert_out.data(), d_out, hidden_size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy expert out");
    const float safe_route_weight = std::isfinite(route_weight) ? route_weight : 0.0f;
    for (std::size_t i = 0; i < hidden_size; ++i) {
        const float upd = std::isfinite(expert_out[i]) ? expert_out[i] : 0.0f;
        dense_state.hidden[i] = std::isfinite(dense_state.hidden[i] + safe_route_weight * upd)
            ? (dense_state.hidden[i] + safe_route_weight * upd) : dense_state.hidden[i];
    }
    const auto combine_end = Clock::now();

    cudaFree(d_input);
    cudaFree(d_gate_out);
    cudaFree(d_up_out);
    cudaFree(d_out);

    if (unpack_ms != nullptr) {
        *unpack_ms += elapsed_ms(unpack_start, unpack_end);
    }
    if (compute_ms != nullptr) {
        *compute_ms += elapsed_ms(compute_start, compute_end);
    }
    if (combine_ms != nullptr) {
        *combine_ms += elapsed_ms(combine_start, combine_end);
    }
    return (unpack_ms != nullptr ? *unpack_ms : 0.0) + (compute_ms != nullptr ? *compute_ms : 0.0)
        + (combine_ms != nullptr ? *combine_ms : 0.0);
}

void cuda_release_expert_buffers(MaterializedExpert& slot) {
    if (slot.device_gate != 0) {
        cudaFree(reinterpret_cast<void*>(slot.device_gate));
    }
    if (slot.device_up != 0) {
        cudaFree(reinterpret_cast<void*>(slot.device_up));
    }
    if (slot.device_down != 0) {
        cudaFree(reinterpret_cast<void*>(slot.device_down));
    }
    slot.device_gate = 0;
    slot.device_up = 0;
    slot.device_down = 0;
    slot.device_gate_len = 0;
    slot.device_up_len = 0;
    slot.device_down_len = 0;
    slot.device_ready = false;
}

}  // namespace flashmoe
