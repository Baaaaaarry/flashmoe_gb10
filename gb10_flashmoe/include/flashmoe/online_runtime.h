#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flashmoe/dense_resident_loader.h"
#include "flashmoe/expert_cache.h"
#include "flashmoe/expert_store.h"
#include "flashmoe/runtime_plan.h"

namespace flashmoe {

struct RouteStep {
    std::uint64_t step = 0;
    std::uint16_t layer = 0;
    std::vector<std::uint16_t> experts;
};

struct OnlineRuntimeStats {
    std::uint64_t route_steps = 0;
    std::uint64_t routed_requests = 0;
    std::uint64_t unique_experts_seen = 0;
    std::uint64_t active_expert_peak = 0;
    std::uint64_t missing_manifest_entries = 0;
    double dense_resident_gb = 0.0;
    double hot_cache_capacity_gb = 0.0;
    double hot_cache_used_gb = 0.0;
    double cold_bytes_touched_gb = 0.0;
    CacheStats cache_stats{};
};

std::vector<RouteStep> load_route_trace_file(const std::string& path);

class OnlineDecodeScheduler {
public:
    OnlineDecodeScheduler(
        DenseResidentPlan resident_plan,
        RuntimePlan runtime_plan,
        ExpertManifestStore store,
        ExpertCache cache);

    void process(const RouteStep& step);
    [[nodiscard]] const OnlineRuntimeStats& stats() const noexcept { return stats_; }

private:
    DenseResidentPlan resident_plan_;
    RuntimePlan runtime_plan_;
    ExpertManifestStore store_;
    ExpertCache cache_;
    OnlineRuntimeStats stats_{};
    std::vector<ExpertId> seen_;
};

}  // namespace flashmoe
