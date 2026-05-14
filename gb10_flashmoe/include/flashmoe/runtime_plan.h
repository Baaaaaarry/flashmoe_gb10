#pragma once

#include <string_view>
#include <vector>

#include "flashmoe/model_spec.h"

namespace flashmoe {

struct HardwareProfile {
    std::string_view name;
    double unified_memory_gb = 0.0;
    double memory_bandwidth_gbps = 0.0;
    double nvme_read_gbps = 0.0;
    bool supports_cuda_fp8 = false;
    bool supports_cuda_fp4 = false;
};

struct RuntimePlan {
    std::string_view runtime_family;
    ExpertStorageFormat cold_storage_format = ExpertStorageFormat::kDense;
    RuntimeComputeFormat hot_cache_compute_format = RuntimeComputeFormat::kFp16;
    bool use_framework_parameter_tree_for_routed_experts = true;
    bool use_layer_major_bundle_layout = false;
    bool use_async_pread = false;
    bool use_predictor_in_hot_path = false;
    bool use_gpu_side_unpack = false;
    double dense_resident_budget_gb = 0.0;
    double hot_expert_cache_budget_gb = 0.0;
    double kv_cache_budget_gb = 0.0;
    double workspace_budget_gb = 0.0;
    double safety_margin_gb = 0.0;
    double total_runtime_budget_gb = 0.0;
    double estimated_cold_tail_gb = 0.0;
    std::vector<std::string_view> reasons;
    std::vector<std::string_view> milestones;
    std::vector<std::string_view> risks;
};

HardwareProfile gb10_hardware_profile();
RuntimePlan recommend_runtime_plan(const ModelSpec& spec, const HardwareProfile& hw);

}  // namespace flashmoe
