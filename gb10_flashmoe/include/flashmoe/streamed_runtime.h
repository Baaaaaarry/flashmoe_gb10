#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "flashmoe/compute_profile.h"
#include "flashmoe/decode_harness.h"
#include "flashmoe/device_path.h"
#include "flashmoe/expert_store.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/online_runtime.h"
#include "flashmoe/runtime_plan.h"

namespace flashmoe {

struct StreamedRuntimeReport {
    std::uint64_t steps = 0;
    std::uint64_t cold_steps = 0;
    std::uint64_t warm_steps = 0;
    std::uint64_t routed_requests = 0;
    std::uint64_t unique_experts_seen = 0;
    std::uint64_t active_expert_peak = 0;
    std::uint64_t missing_manifest_entries = 0;
    double dense_resident_gb = 0.0;
    double hot_cache_capacity_gb = 0.0;
    double kv_cache_budget_gb = 0.0;
    double workspace_budget_gb = 0.0;
    double safety_margin_gb = 0.0;
    double total_runtime_budget_gb = 0.0;
    double hot_cache_used_gb = 0.0;
    double promoted_gb = 0.0;
    double avg_route_ms = 0.0;
    double avg_load_ms = 0.0;
    double avg_unpack_ms = 0.0;
    double avg_compute_ms = 0.0;
    double avg_combine_ms = 0.0;
    double avg_total_ms = 0.0;
    double cold_avg_total_ms = 0.0;
    double warm_avg_total_ms = 0.0;
    double estimated_tok_per_s = 0.0;
    double steady_state_tok_per_s = 0.0;
    SlotCacheStats slot_cache{};
    std::vector<DecodeStepEstimate> samples;
};

class StreamedMoERuntime {
public:
    StreamedMoERuntime(ModelSpec spec,
                       HardwareProfile hw,
                       RuntimePlan plan,
                       ExpertManifestStore store,
                       DevicePathProfile profile);

    StreamedRuntimeReport run(const std::vector<RouteStep>& trace, std::size_t max_steps = 0) const;

private:
    ModelSpec spec_;
    HardwareProfile hw_;
    RuntimePlan plan_;
    ExpertManifestStore store_;
    DevicePathProfile profile_;
};

}  // namespace flashmoe
