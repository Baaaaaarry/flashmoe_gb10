#include "flashmoe/expert_cache.h"

#include <algorithm>
#include <limits>

namespace flashmoe {

ExpertCache::ExpertCache(std::size_t capacity_bytes, EvictionPolicy policy)
    : capacity_bytes_(capacity_bytes), policy_(std::move(policy)) {}

bool ExpertCache::contains(ExpertId id) const {
    return entries_.find(id) != entries_.end();
}

bool ExpertCache::touch_or_insert(const ExpertRequest& request, std::vector<ExpertId>* evicted) {
    auto it = entries_.find(request.id);
    if (it != entries_.end()) {
        it->second.last_touch = request.step;
        it->second.access_count += 1;
        global_max_access_count_ = std::max(global_max_access_count_, it->second.access_count);
        stats_.hits += 1;
        return true;
    }

    stats_.misses += 1;
    if (!ensure_space(request.bytes, request.step, evicted)) {
        return false;
    }

    auto& entry = entries_[request.id];
    entry.id = request.id;
    entry.bytes = request.bytes;
    entry.last_touch = request.step;
    entry.access_count = 1;
    auto& layer_count = layer_resident_counts_[request.id.layer];
    layer_count += 1;
    entry.layer_resident_count_snapshot = layer_count;

    bytes_used_ += request.bytes;
    stats_.bytes_loaded += request.bytes;
    return false;
}

bool ExpertCache::ensure_space(std::size_t needed_bytes, std::uint64_t step, std::vector<ExpertId>* evicted) {
    if (needed_bytes > capacity_bytes_) {
        return false;
    }
    while (bytes_used_ + needed_bytes > capacity_bytes_) {
        auto victim = select_victim(step);
        if (!victim.has_value()) {
            return false;
        }
        auto it = entries_.find(*victim);
        if (it == entries_.end()) {
            return false;
        }
        bytes_used_ -= it->second.bytes;
        stats_.evictions += 1;
        stats_.bytes_evicted += it->second.bytes;
        auto layer_it = layer_resident_counts_.find(it->second.id.layer);
        if (layer_it != layer_resident_counts_.end() && layer_it->second > 0) {
            layer_it->second -= 1;
        }
        if (evicted != nullptr) {
            evicted->push_back(it->second.id);
        }
        entries_.erase(it);
    }
    return true;
}

std::optional<ExpertId> ExpertCache::select_victim(std::uint64_t step) const {
    if (entries_.empty()) {
        return std::nullopt;
    }
    std::uint32_t busiest_layer_entries = 1;
    for (const auto& [_, count] : layer_resident_counts_) {
        busiest_layer_entries = std::max(busiest_layer_entries, count);
    }

    double best_score = -std::numeric_limits<double>::infinity();
    std::optional<ExpertId> victim;
    for (const auto& [id, entry] : entries_) {
        EntryView view{
            .id = id,
            .bytes = entry.bytes,
            .last_touch = entry.last_touch,
            .access_count = entry.access_count,
            .resident_epoch = entry.layer_resident_count_snapshot,
        };
        const double score = policy_.eviction_score(view, step, global_max_access_count_, busiest_layer_entries);
        if (!victim.has_value() || score > best_score) {
            best_score = score;
            victim = id;
        }
    }
    return victim;
}

}  // namespace flashmoe
