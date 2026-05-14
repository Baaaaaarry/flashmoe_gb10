#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "flashmoe/cache_policy.h"

namespace flashmoe {

struct CacheEntry {
    ExpertId id{};
    std::size_t bytes = 0;
    std::uint64_t last_touch = 0;
    std::uint64_t access_count = 0;
    std::uint32_t layer_resident_count_snapshot = 0;
};

class ExpertCache {
public:
    ExpertCache(std::size_t capacity_bytes, EvictionPolicy policy);

    [[nodiscard]] bool contains(ExpertId id) const;
    [[nodiscard]] std::size_t bytes_used() const noexcept { return bytes_used_; }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }
    [[nodiscard]] const CacheStats& stats() const noexcept { return stats_; }

    bool touch_or_insert(const ExpertRequest& request, std::vector<ExpertId>* evicted = nullptr);

private:
    bool ensure_space(std::size_t needed_bytes, std::uint64_t step, std::vector<ExpertId>* evicted);
    std::optional<ExpertId> select_victim(std::uint64_t step) const;

    std::size_t capacity_bytes_ = 0;
    std::size_t bytes_used_ = 0;
    std::uint64_t global_max_access_count_ = 1;
    CacheStats stats_{};
    EvictionPolicy policy_;
    std::unordered_map<ExpertId, CacheEntry, ExpertIdHash> entries_;
    std::unordered_map<std::uint16_t, std::uint32_t> layer_resident_counts_;
};

}  // namespace flashmoe
