#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashmoe/expert_types.h"

namespace flashmoe {

enum class PolicyKind {
    kLru,
    kLfu,
    kRecencyFrequency,
    kOracle,
};

struct PolicyTuning {
    double recency_weight = 0.70;
    double frequency_weight = 0.30;
    double layer_balance_weight = 0.10;
    std::uint64_t recency_window = 256;
};

struct EntryView {
    ExpertId id{};
    std::size_t bytes = 0;
    std::uint64_t last_touch = 0;
    std::uint64_t access_count = 0;
    std::uint32_t resident_epoch = 0;
};

class FutureTraceOracle {
public:
    using Timeline = std::vector<std::uint64_t>;

    void add_occurrence(ExpertId id, std::uint64_t step) {
        positions_[id].push_back(step);
    }

    void finalize() {
        for (auto& [_, steps] : positions_) {
            std::sort(steps.begin(), steps.end());
        }
    }

    std::uint64_t next_use_after(ExpertId id, std::uint64_t step) const {
        auto it = positions_.find(id);
        if (it == positions_.end()) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        auto pos = std::upper_bound(it->second.begin(), it->second.end(), step);
        if (pos == it->second.end()) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return *pos;
    }

private:
    std::unordered_map<ExpertId, Timeline, ExpertIdHash> positions_;
};

class EvictionPolicy {
public:
    EvictionPolicy(PolicyKind kind, PolicyTuning tuning = {}, const FutureTraceOracle* oracle = nullptr)
        : kind_(kind), tuning_(tuning), oracle_(oracle) {}

    [[nodiscard]] PolicyKind kind() const noexcept { return kind_; }

    [[nodiscard]] double eviction_score(const EntryView& entry,
                                        std::uint64_t step,
                                        std::uint64_t max_access_count,
                                        std::uint32_t busiest_layer_entries) const {
        switch (kind_) {
            case PolicyKind::kLru:
                return static_cast<double>(age(entry, step));
            case PolicyKind::kLfu:
                return static_cast<double>(std::numeric_limits<std::uint64_t>::max() - entry.access_count);
            case PolicyKind::kRecencyFrequency:
                return recency_frequency_score(entry, step, max_access_count, busiest_layer_entries);
            case PolicyKind::kOracle:
                if (oracle_ == nullptr) {
                    return static_cast<double>(age(entry, step));
                }
                return static_cast<double>(oracle_->next_use_after(entry.id, step));
        }
        return 0.0;
    }

    [[nodiscard]] std::string name() const {
        switch (kind_) {
            case PolicyKind::kLru: return "lru";
            case PolicyKind::kLfu: return "lfu";
            case PolicyKind::kRecencyFrequency: return "recency_frequency";
            case PolicyKind::kOracle: return "oracle";
        }
        return "unknown";
    }

private:
    [[nodiscard]] std::uint64_t age(const EntryView& entry, std::uint64_t step) const {
        return step >= entry.last_touch ? (step - entry.last_touch) : 0;
    }

    [[nodiscard]] double recency_frequency_score(const EntryView& entry,
                                                 std::uint64_t step,
                                                 std::uint64_t max_access_count,
                                                 std::uint32_t busiest_layer_entries) const {
        const double recency = std::min(
            1.0,
            static_cast<double>(age(entry, step)) / static_cast<double>(std::max<std::uint64_t>(1, tuning_.recency_window)));
        const double frequency = static_cast<double>(entry.access_count) /
                                 static_cast<double>(std::max<std::uint64_t>(1, max_access_count));
        const double layer_pressure = static_cast<double>(entry.resident_epoch) /
                                      static_cast<double>(std::max<std::uint32_t>(1U, busiest_layer_entries));
        return tuning_.recency_weight * recency
             - tuning_.frequency_weight * frequency
             + tuning_.layer_balance_weight * layer_pressure;
    }

    PolicyKind kind_;
    PolicyTuning tuning_;
    const FutureTraceOracle* oracle_;
};

}  // namespace flashmoe
