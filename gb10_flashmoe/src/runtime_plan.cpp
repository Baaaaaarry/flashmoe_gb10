#include "flashmoe/runtime_plan.h"

#include <algorithm>

namespace flashmoe {

HardwareProfile gb10_hardware_profile() {
    return HardwareProfile{
        .name = "dgx-spark-gb10",
        .unified_memory_gb = 128.0,
        .memory_bandwidth_gbps = 273.0,
        .nvme_read_gbps = 14.5,
        .supports_cuda_fp8 = true,
        .supports_cuda_fp4 = true,
    };
}

RuntimePlan recommend_runtime_plan(const ModelSpec& spec, const HardwareProfile& hw) {
    RuntimePlan plan;
    plan.runtime_family = "dedicated-streamed-moe-runtime";
    plan.use_framework_parameter_tree_for_routed_experts = false;
    plan.use_layer_major_bundle_layout = true;
    plan.use_async_pread = true;
    plan.use_predictor_in_hot_path = false;
    plan.use_gpu_side_unpack = false;
    plan.kv_cache_budget_gb = 32.0;
    plan.workspace_budget_gb = 12.0;
    plan.safety_margin_gb = 12.0;

    const double dense_share = std::max(0.0, 1.0 - spec.routed_expert_share);
    const double dense_resident_budget = spec.source_weight_gb * dense_share;
    plan.dense_resident_budget_gb = dense_resident_budget;
    plan.hot_expert_cache_budget_gb = 24.0;
    plan.total_runtime_budget_gb = plan.dense_resident_budget_gb
        + plan.hot_expert_cache_budget_gb
        + plan.kv_cache_budget_gb
        + plan.workspace_budget_gb
        + plan.safety_margin_gb;

    if (spec.family == ModelFamily::kQwen35Moe) {
        plan.cold_storage_format = ExpertStorageFormat::kQ3Like;
        plan.hot_cache_compute_format = hw.supports_cuda_fp8 ? RuntimeComputeFormat::kFp8
                                                             : RuntimeComputeFormat::kBf16;
        plan.estimated_cold_tail_gb = spec.source_weight_gb - dense_resident_budget - plan.hot_expert_cache_budget_gb;
        plan.reasons = {
            "Current vLLM + PyTorch wrapper path was validated as non-deployable.",
            "Q3-like replay reduced routed-expert read traffic from 211.7 GB to 38.1 GB on the same trace.",
            "Qwen3.5 MoE needs routed experts fully outside the framework parameter tree to avoid 112 GB UMA residency.",
        };
        plan.milestones = {
            "Keep export/manifest/replay toolchain as-is.",
            "Build a text-only runtime that instantiates dense/router/shared layers only.",
            "Store routed experts on SSD as Q3-like bundles and promote misses into FP8 hot-cache slots.",
            "Replace per-step host-side restack with persistent layer-major cache slots.",
        };
        plan.risks = {
            "Q3-like currently saves bandwidth only; compute still requires unpack before GEMV/GEMM.",
            "Without GPU-side unpack, decode throughput will remain below the Apple-paper targets.",
            "The next limiting factor is grouped compute / dispatch cost rather than expert-cache capacity.",
        };
        return plan;
    }

    if (spec.family == ModelFamily::kDeepSeekV4Flash) {
        plan.cold_storage_format = ExpertStorageFormat::kMxfp4;
        plan.hot_cache_compute_format = hw.supports_cuda_fp8 ? RuntimeComputeFormat::kFp8
                                                             : RuntimeComputeFormat::kBf16;
        plan.estimated_cold_tail_gb = spec.source_weight_gb - dense_resident_budget - plan.hot_expert_cache_budget_gb;
        plan.reasons = {
            "DeepSeek-V4-Flash already centers the deployment problem on expert residency rather than KV cache.",
            "The ds4 family proved that a narrow runtime with explicit tensor budgeting is a better fit than a general wrapper path.",
            "MXFP4 or Q3-like routed experts can both satisfy the cold-storage requirement; runtime must stay model-spec driven.",
        };
        plan.milestones = {
            "Abstract routing, layer count, top-k, and cache policy from the runtime core.",
            "Preserve CSA/HCA KV path resident while making experts purely streamed.",
            "Validate with a text-only runtime before adding OpenAI-compatible serving.",
            "Add predictor only after basic decode throughput and residency are stable.",
        };
        plan.risks = {
            "DeepSeek hash-router prefill path adds model-specific control logic that Qwen3.5 does not require.",
            "Switching from Q3-like to MXFP4 storage may reduce tooling reuse unless manifest and unpack interfaces stay generic.",
        };
        return plan;
    }

    plan.cold_storage_format = ExpertStorageFormat::kQ3Like;
    plan.hot_cache_compute_format = RuntimeComputeFormat::kBf16;
    plan.estimated_cold_tail_gb = 0.0;
    plan.reasons = {"Unknown model family; falling back to conservative streamed-MoE plan."};
    plan.milestones = {"Define a concrete model spec before implementing runtime kernels."};
    plan.risks = {"Unknown expert ratio can invalidate the cache budget estimate."};
    return plan;
}

}  // namespace flashmoe
