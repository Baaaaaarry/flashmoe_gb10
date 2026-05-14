#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "flashmoe/device_path.h"
#include "flashmoe/expert_store.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/online_runtime.h"
#include "flashmoe/runtime_plan.h"

namespace flashmoe {

struct SlotCacheStats {
    std::uint64_t promotions = 0;
    std::uint64_t slot_hits = 0;
    std::uint64_t slot_evictions = 0;
    double promoted_gb = 0.0;
    double resident_slot_gb = 0.0;
};

struct DecodeStepEstimate {
    std::uint64_t step = 0;
    std::uint16_t layer = 0;
    std::size_t active_experts = 0;
    double route_ms = 0.0;
    double load_ms = 0.0;
    double unpack_ms = 0.0;
    double compute_ms = 0.0;
    double combine_ms = 0.0;
    double total_ms = 0.0;
    std::uint64_t slot_promotions = 0;
    std::uint64_t slot_hits = 0;
};

struct DecodeHarnessReport {
    std::uint64_t steps = 0;
    std::uint64_t cold_steps = 0;
    std::uint64_t warm_steps = 0;
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

class ComputeReadySlotCache {
public:
    ComputeReadySlotCache(double capacity_gb, RuntimeComputeFormat compute_format);

    bool contains(ExpertId id) const;
    void touch_or_promote(const ExpertRecord& record, SlotCacheStats& stats, std::vector<ExpertId>* evicted = nullptr);
    [[nodiscard]] double bytes_used_gb() const noexcept;

private:
    struct Slot {
        ExpertId id{};
        std::size_t bytes = 0;
        std::uint64_t last_touch = 0;
    };

    std::size_t capacity_bytes_ = 0;
    std::size_t bytes_used_ = 0;
    std::uint64_t clock_ = 0;
    RuntimeComputeFormat compute_format_ = RuntimeComputeFormat::kFp16;
    std::vector<Slot> slots_;
};

DecodeHarnessReport run_decode_harness(
    const ModelSpec& spec,
    const RuntimePlan& plan,
    const HardwareProfile& hw,
    const DevicePathProfile& profile,
    const ExpertManifestStore& store,
    const std::vector<RouteStep>& trace,
    std::size_t max_steps = 0);

}  // namespace flashmoe
