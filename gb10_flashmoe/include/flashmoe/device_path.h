#pragma once

#include <optional>
#include <string_view>

#include "flashmoe/compute_profile.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/runtime_plan.h"

namespace flashmoe {

enum class UnpackBackend {
    kHost = 0,
    kGpu,
};

enum class ComputeBackend {
    kEstimated = 0,
    kGroupedGemm,
    kGroupedGemv,
};

struct DevicePathProfile {
    std::string_view name;
    UnpackBackend unpack_backend = UnpackBackend::kHost;
    ComputeBackend compute_backend = ComputeBackend::kEstimated;
    double route_ms = 0.0;
    double load_fixed_ms = 0.0;
    double load_scale = 1.0;
    double unpack_fixed_ms = 0.0;
    double unpack_scale = 1.0;
    double compute_per_expert_ms = 0.0;
    double compute_graph_overhead_ms = 0.0;
    double combine_ms = 0.0;
    std::optional<ComputeProfile> measured_compute;
};

std::string_view to_string(UnpackBackend backend);
std::string_view to_string(ComputeBackend backend);

DevicePathProfile host_unpack_profile(const ModelSpec& spec, const HardwareProfile& hw, const RuntimePlan& plan);
DevicePathProfile gpu_unpack_profile(const ModelSpec& spec, const HardwareProfile& hw, const RuntimePlan& plan);

}  // namespace flashmoe
