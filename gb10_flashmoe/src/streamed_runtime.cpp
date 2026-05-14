#include "flashmoe/streamed_runtime.h"

#include <algorithm>
#include <unordered_set>

namespace flashmoe {
namespace {

std::uint64_t make_seen_key(ExpertId id) {
    return (static_cast<std::uint64_t>(id.layer) << 32U) | static_cast<std::uint64_t>(id.expert);
}

double unpack_ms_for(const ExpertRecord& record, const HardwareProfile& hw, const DevicePathProfile& profile) {
    const double bytes_gb = static_cast<double>(record.bytes) / 1e9;
    const double format_scale = record.format == ExpertStorageFormat::kQ3Like ? 8.0
        : record.format == ExpertStorageFormat::kMxfp4 ? 5.0
                                                       : 1.0;
    return profile.unpack_fixed_ms
        + profile.unpack_scale * (bytes_gb / std::max(1.0, hw.memory_bandwidth_gbps)) * 1000.0 * format_scale;
}

double load_ms_for(const ExpertRecord& record, const HardwareProfile& hw, const DevicePathProfile& profile) {
    const double bytes_gb = static_cast<double>(record.bytes) / 1e9;
    return profile.load_fixed_ms + profile.load_scale * (bytes_gb / std::max(1.0, hw.nvme_read_gbps)) * 1000.0;
}

double compute_ms_for(const RuntimePlan& plan,
                      const DevicePathProfile& profile,
                      std::size_t active_experts) {
    const double fallback = profile.compute_graph_overhead_ms
        + profile.compute_per_expert_ms * static_cast<double>(active_experts)
            * (plan.hot_cache_compute_format == RuntimeComputeFormat::kFp8 ? 1.0 : 1.35);
    if (profile.measured_compute.has_value() && !profile.measured_compute->empty()) {
        return profile.measured_compute->estimate_ms(active_experts, fallback);
    }
    return fallback;
}

double combine_ms_for(const DevicePathProfile& profile, std::size_t active_experts) {
    return profile.combine_ms - (0.004 * 4.0) + 0.004 * static_cast<double>(active_experts);
}

double token_per_s_from_ms(double ms) {
    return ms > 0.0 ? 1000.0 / ms : 0.0;
}

bool is_cold_step(const DecodeStepEstimate& step) {
    return step.slot_promotions > 0;
}

}  // namespace

StreamedMoERuntime::StreamedMoERuntime(ModelSpec spec,
                                       HardwareProfile hw,
                                       RuntimePlan plan,
                                       ExpertManifestStore store,
                                       DevicePathProfile profile)
    : spec_(spec),
      hw_(hw),
      plan_(std::move(plan)),
      store_(std::move(store)),
      profile_(std::move(profile)) {}

StreamedRuntimeReport StreamedMoERuntime::run(const std::vector<RouteStep>& trace, std::size_t max_steps) const {
    StreamedRuntimeReport report;
    report.dense_resident_gb = plan_.dense_resident_budget_gb;
    report.hot_cache_capacity_gb = plan_.hot_expert_cache_budget_gb;
    report.kv_cache_budget_gb = plan_.kv_cache_budget_gb;
    report.workspace_budget_gb = plan_.workspace_budget_gb;
    report.safety_margin_gb = plan_.safety_margin_gb;
    report.total_runtime_budget_gb = plan_.total_runtime_budget_gb;

    ComputeReadySlotCache slot_cache(plan_.hot_expert_cache_budget_gb, plan_.hot_cache_compute_format);
    std::unordered_set<std::uint64_t> seen;

    const std::size_t limit = max_steps > 0 ? std::min(max_steps, trace.size()) : trace.size();
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& route = trace[i];
        DecodeStepEstimate step;
        step.step = route.step;
        step.layer = route.layer;
        step.active_experts = route.experts.size();
        step.route_ms = profile_.route_ms;

        report.steps += 1;
        report.routed_requests += route.experts.size();
        report.active_expert_peak = std::max<std::uint64_t>(report.active_expert_peak, route.experts.size());

        for (const auto expert : route.experts) {
            const ExpertId id{route.layer, expert};
            const auto* record = store_.find(id);
            if (record == nullptr) {
                report.missing_manifest_entries += 1;
                continue;
            }

            seen.insert(make_seen_key(id));
            if (slot_cache.contains(id)) {
                SlotCacheStats local = report.slot_cache;
                slot_cache.touch_or_promote(*record, local);
                step.slot_hits += 1;
                report.slot_cache = local;
                continue;
            }

            step.slot_promotions += 1;
            step.load_ms += load_ms_for(*record, hw_, profile_);
            step.unpack_ms += unpack_ms_for(*record, hw_, profile_);
            slot_cache.touch_or_promote(*record, report.slot_cache);
        }

        step.compute_ms = compute_ms_for(plan_, profile_, route.experts.size());
        step.combine_ms = combine_ms_for(profile_, route.experts.size());
        step.total_ms = step.route_ms + step.load_ms + step.unpack_ms + step.compute_ms + step.combine_ms;

        report.avg_route_ms += step.route_ms;
        report.avg_load_ms += step.load_ms;
        report.avg_unpack_ms += step.unpack_ms;
        report.avg_compute_ms += step.compute_ms;
        report.avg_combine_ms += step.combine_ms;
        report.avg_total_ms += step.total_ms;

        if (is_cold_step(step)) {
            report.cold_steps += 1;
            report.cold_avg_total_ms += step.total_ms;
        } else {
            report.warm_steps += 1;
            report.warm_avg_total_ms += step.total_ms;
        }

        if (report.samples.size() < 16) {
            report.samples.push_back(step);
        }
    }

    report.unique_experts_seen = seen.size();
    report.hot_cache_used_gb = slot_cache.bytes_used_gb();
    report.promoted_gb = report.slot_cache.promoted_gb;

    if (report.steps > 0) {
        const double denom = static_cast<double>(report.steps);
        report.avg_route_ms /= denom;
        report.avg_load_ms /= denom;
        report.avg_unpack_ms /= denom;
        report.avg_compute_ms /= denom;
        report.avg_combine_ms /= denom;
        report.avg_total_ms /= denom;
        report.estimated_tok_per_s = token_per_s_from_ms(report.avg_total_ms);
    }
    if (report.cold_steps > 0) {
        report.cold_avg_total_ms /= static_cast<double>(report.cold_steps);
    }
    if (report.warm_steps > 0) {
        report.warm_avg_total_ms /= static_cast<double>(report.warm_steps);
        report.steady_state_tok_per_s = token_per_s_from_ms(report.warm_avg_total_ms);
    }

    return report;
}

}  // namespace flashmoe
