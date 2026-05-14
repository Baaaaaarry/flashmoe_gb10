#include "flashmoe/decode_harness.h"

#include <algorithm>

namespace flashmoe {
namespace {

double compute_ready_bytes_factor(RuntimeComputeFormat format) {
    switch (format) {
    case RuntimeComputeFormat::kFp16:
    case RuntimeComputeFormat::kBf16:
        return 2.0;
    case RuntimeComputeFormat::kFp8:
        return 1.0;
    case RuntimeComputeFormat::kMxfp4:
        return 0.6;
    }
    return 2.0;
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
    const double format_scale = plan.hot_cache_compute_format == RuntimeComputeFormat::kFp8 ? 1.0 : 1.35;
    return profile.compute_graph_overhead_ms
        + profile.compute_per_expert_ms * static_cast<double>(active_experts) * format_scale;
}

double combine_ms_for(const DevicePathProfile& profile, std::size_t active_experts) {
    return profile.combine_ms - (0.004 * 4.0) + 0.004 * static_cast<double>(active_experts);
}

bool is_cold_step(const DecodeStepEstimate& step) {
    return step.slot_promotions > 0;
}

double token_per_s_from_ms(double ms) {
    return ms > 0.0 ? 1000.0 / ms : 0.0;
}

std::size_t compute_ready_bytes(const ExpertRecord& record, RuntimeComputeFormat format) {
    const double factor = compute_ready_bytes_factor(format);
    return static_cast<std::size_t>(static_cast<double>(record.bytes) * factor);
}

}  // namespace

ComputeReadySlotCache::ComputeReadySlotCache(double capacity_gb, RuntimeComputeFormat compute_format)
    : capacity_bytes_(static_cast<std::size_t>(capacity_gb * 1e9)),
      compute_format_(compute_format) {}

bool ComputeReadySlotCache::contains(ExpertId id) const {
    return std::any_of(slots_.begin(), slots_.end(), [id](const Slot& slot) { return slot.id == id; });
}

void ComputeReadySlotCache::touch_or_promote(const ExpertRecord& record, SlotCacheStats& stats, std::vector<ExpertId>* evicted) {
    clock_ += 1;
    for (auto& slot : slots_) {
        if (slot.id == record.id) {
            slot.last_touch = clock_;
            stats.slot_hits += 1;
            return;
        }
    }

    stats.promotions += 1;
    const std::size_t ready_bytes = compute_ready_bytes(record, compute_format_);
    while (bytes_used_ + ready_bytes > capacity_bytes_ && !slots_.empty()) {
        auto victim = std::min_element(
            slots_.begin(),
            slots_.end(),
            [](const Slot& lhs, const Slot& rhs) { return lhs.last_touch < rhs.last_touch; });
        bytes_used_ -= victim->bytes;
        if (evicted != nullptr) {
            evicted->push_back(victim->id);
        }
        slots_.erase(victim);
        stats.slot_evictions += 1;
    }

    slots_.push_back(Slot{
        .id = record.id,
        .bytes = ready_bytes,
        .last_touch = clock_,
    });
    bytes_used_ += ready_bytes;
    stats.promoted_gb += static_cast<double>(ready_bytes) / 1e9;
    stats.resident_slot_gb = static_cast<double>(bytes_used_) / 1e9;
}

double ComputeReadySlotCache::bytes_used_gb() const noexcept {
    return static_cast<double>(bytes_used_) / 1e9;
}

DecodeHarnessReport run_decode_harness(
    const ModelSpec& spec,
    const RuntimePlan& plan,
    const HardwareProfile& hw,
    const DevicePathProfile& profile,
    const ExpertManifestStore& store,
    const std::vector<RouteStep>& trace,
    std::size_t max_steps) {
    ComputeReadySlotCache slot_cache(plan.hot_expert_cache_budget_gb, plan.hot_cache_compute_format);
    DecodeHarnessReport report;

    const std::size_t limit = max_steps > 0 ? std::min(max_steps, trace.size()) : trace.size();
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& route = trace[i];
        DecodeStepEstimate step;
        step.step = route.step;
        step.layer = route.layer;
        step.active_experts = route.experts.size();
        step.route_ms = profile.route_ms;
        step.compute_ms = compute_ms_for(plan, profile, route.experts.size());
        step.combine_ms = combine_ms_for(profile, route.experts.size());

        for (const auto expert : route.experts) {
            const ExpertId id{route.layer, expert};
            const auto* record = store.find(id);
            if (record == nullptr) {
                continue;
            }
            if (slot_cache.contains(id)) {
                SlotCacheStats local = report.slot_cache;
                slot_cache.touch_or_promote(*record, local);
                step.slot_hits += 1;
                report.slot_cache = local;
                continue;
            }

            step.slot_promotions += 1;
            step.load_ms += load_ms_for(*record, hw, profile);
            step.unpack_ms += unpack_ms_for(*record, hw, profile);
            slot_cache.touch_or_promote(*record, report.slot_cache);
        }

        step.total_ms = step.route_ms + step.load_ms + step.unpack_ms + step.compute_ms + step.combine_ms;
        report.steps += 1;
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

    report.slot_cache.resident_slot_gb = slot_cache.bytes_used_gb();
    return report;
}

}  // namespace flashmoe
