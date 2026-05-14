#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>
#include <map>
#include <cstdint>
#include <vector>

namespace {

void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cuBLAS status ", static_cast<int>(status));
}

cublasOperation_t kNoTrans = CUBLAS_OP_N;
cublasOperation_t kTrans = CUBLAS_OP_T;

struct ExpertSpan {
  int64_t expert_idx;
  int64_t offset;
  int64_t count;
};

cublasHandle_t get_cublas_handle() {
  static thread_local cublasHandle_t handle = []() {
    cublasHandle_t created = nullptr;
    check_cublas(cublasCreate(&created), "cublasCreate");
    return created;
  }();
  return handle;
}

cudaDataType_t get_cuda_dtype(const at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
      return CUDA_R_16F;
    case at::kBFloat16:
      return CUDA_R_16BF;
    case at::kFloat:
      return CUDA_R_32F;
    default:
      TORCH_CHECK(false, "Unsupported dtype for cuBLASLt GEMM");
  }
}

cublasComputeType_t get_compute_type(const at::ScalarType dtype) {
  switch (dtype) {
    case at::kHalf:
    case at::kBFloat16:
      return CUBLAS_COMPUTE_32F;
    case at::kFloat:
      return CUBLAS_COMPUTE_32F;
    default:
      TORCH_CHECK(false, "Unsupported dtype for cuBLASLt compute type");
  }
}

const void* offset_const_ptr(const torch::Tensor& tensor, int64_t row_offset) {
  return static_cast<const char*>(tensor.const_data_ptr()) +
      row_offset * tensor.stride(0) * tensor.element_size();
}

void* offset_mutable_ptr(torch::Tensor& tensor, int64_t row_offset) {
  return static_cast<char*>(tensor.mutable_data_ptr()) +
      row_offset * tensor.stride(0) * tensor.element_size();
}

template <typename PtrType>
torch::Tensor pointer_array_to_device_tensor(
    const std::vector<PtrType>& pointers,
    const at::Device& device) {
  std::vector<int64_t> pointer_values;
  pointer_values.reserve(pointers.size());
  for (PtrType ptr : pointers) {
    pointer_values.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr)));
  }
  auto cpu_tensor = torch::tensor(pointer_values, torch::TensorOptions().dtype(torch::kInt64));
  return cpu_tensor.to(device, /*non_blocking=*/false, /*copy=*/true);
}

void grouped_row_major_gemm(
    cublasHandle_t handle,
    cudaStream_t stream,
    const torch::Tensor& packed_input,
    const torch::Tensor& expert_weights,
    torch::Tensor& packed_output,
    const std::vector<ExpertSpan>& active_spans,
    int64_t out_dim,
    int64_t in_dim) {
  TORCH_CHECK(packed_input.scalar_type() == expert_weights.scalar_type(), "GEMM dtype mismatch");
  TORCH_CHECK(packed_input.scalar_type() == packed_output.scalar_type(), "Output dtype mismatch");
  TORCH_CHECK(packed_input.is_contiguous(), "packed_input must be contiguous");
  TORCH_CHECK(expert_weights.is_contiguous(), "expert_weights must be contiguous");
  TORCH_CHECK(packed_output.is_contiguous(), "packed_output must be contiguous");

  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");

  std::map<int64_t, std::vector<const ExpertSpan*>> spans_by_count;
  for (const auto& span : active_spans) {
    spans_by_count[span.count].push_back(&span);
  }

  for (const auto& [count, spans] : spans_by_count) {
    TORCH_CHECK(count <= std::numeric_limits<int>::max(), "Grouped GEMM count exceeds int32");
    TORCH_CHECK(out_dim <= std::numeric_limits<int>::max(), "Grouped GEMM output dim exceeds int32");
    TORCH_CHECK(in_dim <= std::numeric_limits<int>::max(), "Grouped GEMM input dim exceeds int32");
    TORCH_CHECK(spans.size() <= static_cast<size_t>(std::numeric_limits<int>::max()), "Grouped GEMM group too large");

    std::vector<const void*> weight_ptrs;
    std::vector<const void*> input_ptrs;
    std::vector<void*> output_ptrs;
    weight_ptrs.reserve(spans.size());
    input_ptrs.reserve(spans.size());
    output_ptrs.reserve(spans.size());

    for (const auto* span : spans) {
      weight_ptrs.push_back(offset_const_ptr(expert_weights, span->expert_idx));
      input_ptrs.push_back(offset_const_ptr(packed_input, span->offset));
      output_ptrs.push_back(offset_mutable_ptr(packed_output, span->offset));
    }

    auto weight_ptrs_dev = pointer_array_to_device_tensor(weight_ptrs, packed_input.device());
    auto input_ptrs_dev = pointer_array_to_device_tensor(input_ptrs, packed_input.device());
    auto output_ptrs_dev = pointer_array_to_device_tensor(output_ptrs, packed_input.device());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasGemmBatchedEx(
            handle,
            kTrans,
            kNoTrans,
            static_cast<int>(out_dim),
            static_cast<int>(count),
            static_cast<int>(in_dim),
            &alpha,
            reinterpret_cast<const void* const*>(weight_ptrs_dev.data_ptr<int64_t>()),
            get_cuda_dtype(expert_weights.scalar_type()),
            static_cast<int>(in_dim),
            reinterpret_cast<const void* const*>(input_ptrs_dev.data_ptr<int64_t>()),
            get_cuda_dtype(packed_input.scalar_type()),
            static_cast<int>(in_dim),
            &beta,
            reinterpret_cast<void* const*>(output_ptrs_dev.data_ptr<int64_t>()),
            get_cuda_dtype(packed_output.scalar_type()),
            static_cast<int>(out_dim),
            static_cast<int>(spans.size()),
            get_compute_type(packed_input.scalar_type()),
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmBatchedEx");
  }
}

}  // namespace

torch::Tensor flashmoe_moe_forward_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_proj_weights,
    torch::Tensor up_proj_weights,
    torch::Tensor down_proj_weights,
    torch::Tensor route_ids,
    torch::Tensor route_weights) {
  const at::cuda::CUDAGuard device_guard(hidden_states.device());
  auto hidden = hidden_states.contiguous();
  auto gate_w = gate_proj_weights.contiguous();
  auto up_w = up_proj_weights.contiguous();
  auto down_w = down_proj_weights.contiguous();
  auto routes = route_ids.contiguous();
  auto probs = route_weights.contiguous();

  TORCH_CHECK(hidden.dim() == 2, "hidden_states must be [tokens, hidden]");
  TORCH_CHECK(gate_w.dim() == 3, "gate_proj_weights must be [experts, inter, hidden]");
  TORCH_CHECK(up_w.dim() == 3, "up_proj_weights must be [experts, inter, hidden]");
  TORCH_CHECK(down_w.dim() == 3, "down_proj_weights must be [experts, hidden, inter]");
  TORCH_CHECK(routes.dim() == 2, "route_ids must be [tokens, topk]");
  TORCH_CHECK(probs.dim() == 2, "route_weights must be [tokens, topk]");

  const int64_t num_tokens = hidden.size(0);
  const int64_t hidden_dim = hidden.size(1);
  const int64_t inter_dim = gate_w.size(1);
  const int64_t num_experts = gate_w.size(0);
  const int64_t topk = routes.size(1);
  const int64_t total_routes = num_tokens * topk;

  auto output = torch::zeros({num_tokens, hidden_dim}, hidden.options());

  auto token_ids = torch::arange(
      num_tokens,
      hidden.options().dtype(torch::kInt64)).unsqueeze(1).expand({num_tokens, topk}).reshape({-1});
  auto flat_expert_ids = routes.to(torch::kInt64).reshape({-1});
  auto flat_route_weights = probs.reshape({-1});

  auto sorted = torch::sort(flat_expert_ids);
  auto sorted_expert_ids = std::get<0>(sorted).contiguous();
  auto sort_indices = std::get<1>(sorted).contiguous();
  auto sorted_token_ids = token_ids.index_select(0, sort_indices).contiguous();
  auto sorted_route_weights = flat_route_weights.index_select(0, sort_indices).contiguous();
  auto per_expert_counts_cpu = torch::bincount(sorted_expert_ids, {}, num_experts).to(torch::kCPU);
  auto packed_hidden = hidden.index_select(0, sorted_token_ids).contiguous();
  auto packed_gate = torch::empty({total_routes, inter_dim}, hidden.options());
  auto packed_up = torch::empty({total_routes, inter_dim}, hidden.options());
  auto packed_act = torch::empty({total_routes, inter_dim}, hidden.options());
  auto packed_output = torch::empty({total_routes, hidden_dim}, hidden.options());

  cublasHandle_t cublas_handle = get_cublas_handle();
  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t cuda_stream = stream.stream();

  const auto* counts_ptr = per_expert_counts_cpu.data_ptr<int64_t>();
  int64_t offset = 0;
  std::vector<ExpertSpan> active_spans;
  active_spans.reserve(num_experts);
  for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
    const int64_t count = counts_ptr[expert_idx];
    if (count == 0) {
      continue;
    }

    active_spans.push_back({expert_idx, offset, count});
    offset += count;
  }

  grouped_row_major_gemm(
      cublas_handle,
      cuda_stream,
      packed_hidden,
      gate_w,
      packed_gate,
      active_spans,
      inter_dim,
      hidden_dim);
  grouped_row_major_gemm(
      cublas_handle,
      cuda_stream,
      packed_hidden,
      up_w,
      packed_up,
      active_spans,
      inter_dim,
      hidden_dim);

  packed_act.copy_(torch::silu(packed_gate));
  packed_act.mul_(packed_up);

  grouped_row_major_gemm(
      cublas_handle,
      cuda_stream,
      packed_act,
      down_w,
      packed_output,
      active_spans,
      hidden_dim,
      inter_dim);

  packed_output.mul_(sorted_route_weights.unsqueeze(1));
  output.index_add_(0, sorted_token_ids, packed_output);
  return output;
}
