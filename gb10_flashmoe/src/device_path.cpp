#include "flashmoe/device_path.h"

namespace flashmoe {

std::string_view to_string(UnpackBackend backend) {
    switch (backend) {
    case UnpackBackend::kHost:
        return "host";
    case UnpackBackend::kGpu:
        return "gpu";
    }
    return "host";
}

std::string_view to_string(ComputeBackend backend) {
    switch (backend) {
    case ComputeBackend::kEstimated:
        return "estimated";
    case ComputeBackend::kGroupedGemm:
        return "grouped_gemm";
    case ComputeBackend::kGroupedGemv:
        return "grouped_gemv";
    }
    return "estimated";
}

DevicePathProfile host_unpack_profile(const ModelSpec& spec,
                                      const HardwareProfile& hw,
                                      const RuntimePlan& plan) {
    const double compute_per_expert = spec.family == ModelFamily::kDeepSeekV4Flash ? 0.085 : 0.070;
    return DevicePathProfile{
        .name = "host_unpack",
        .unpack_backend = UnpackBackend::kHost,
        .compute_backend = ComputeBackend::kGroupedGemv,
        .route_ms = spec.family == ModelFamily::kDeepSeekV4Flash ? 0.045 : 0.035,
        .load_fixed_ms = 0.040,
        .load_scale = 1.00,
        .unpack_fixed_ms = 0.030,
        .unpack_scale = 1.00,
        .compute_per_expert_ms = compute_per_expert,
        .compute_graph_overhead_ms = 0.0,
        .combine_ms = 0.012 + 0.004 * static_cast<double>(spec.top_k),
    };
}

DevicePathProfile gpu_unpack_profile(const ModelSpec& spec,
                                     const HardwareProfile& hw,
                                     const RuntimePlan& plan) {
    const double compute_per_expert = spec.family == ModelFamily::kDeepSeekV4Flash ? 0.078 : 0.062;
    return DevicePathProfile{
        .name = "gpu_unpack",
        .unpack_backend = UnpackBackend::kGpu,
        .compute_backend = ComputeBackend::kGroupedGemv,
        .route_ms = spec.family == ModelFamily::kDeepSeekV4Flash ? 0.043 : 0.033,
        .load_fixed_ms = 0.036,
        .load_scale = 0.96,
        .unpack_fixed_ms = 0.010,
        .unpack_scale = 0.45,
        .compute_per_expert_ms = compute_per_expert,
        .compute_graph_overhead_ms = 0.008,
        .combine_ms = 0.012 + 0.003 * static_cast<double>(spec.top_k),
    };
}

}  // namespace flashmoe
