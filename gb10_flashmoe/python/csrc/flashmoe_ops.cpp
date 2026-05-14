#include <torch/extension.h>

torch::Tensor flashmoe_moe_forward_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_proj_weights,
    torch::Tensor up_proj_weights,
    torch::Tensor down_proj_weights,
    torch::Tensor route_ids,
    torch::Tensor route_weights);

torch::Tensor moe_forward(
    torch::Tensor hidden_states,
    torch::Tensor gate_proj_weights,
    torch::Tensor up_proj_weights,
    torch::Tensor down_proj_weights,
    torch::Tensor route_ids,
    torch::Tensor route_weights) {
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
  TORCH_CHECK(gate_proj_weights.is_cuda(), "gate_proj_weights must be CUDA");
  TORCH_CHECK(up_proj_weights.is_cuda(), "up_proj_weights must be CUDA");
  TORCH_CHECK(down_proj_weights.is_cuda(), "down_proj_weights must be CUDA");
  TORCH_CHECK(route_ids.is_cuda(), "route_ids must be CUDA");
  TORCH_CHECK(route_weights.is_cuda(), "route_weights must be CUDA");
  return flashmoe_moe_forward_cuda(
      hidden_states,
      gate_proj_weights,
      up_proj_weights,
      down_proj_weights,
      route_ids,
      route_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_forward", &moe_forward, "FlashMoE fused routed experts forward (CUDA)");
}
