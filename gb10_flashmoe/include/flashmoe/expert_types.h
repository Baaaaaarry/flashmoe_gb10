#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace flashmoe {

struct ExpertId {
    std::uint16_t layer = 0;
    std::uint16_t expert = 0;

    bool operator==(const ExpertId& other) const noexcept {
        return layer == other.layer && expert == other.expert;
    }

    bool operator!=(const ExpertId& other) const noexcept {
        return !(*this == other);
    }

    bool operator<(const ExpertId& other) const noexcept {
        if (layer != other.layer) {
            return layer < other.layer;
        }
        return expert < other.expert;
    }
};

struct ExpertIdHash {
    std::size_t operator()(const ExpertId& id) const noexcept {
        return (static_cast<std::size_t>(id.layer) << 16U) ^ static_cast<std::size_t>(id.expert);
    }
};

struct ExpertRequest {
    ExpertId id{};
    std::size_t bytes = 0;
    std::uint64_t step = 0;
};

struct CacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t bytes_loaded = 0;
    std::uint64_t bytes_evicted = 0;
};

}  // namespace flashmoe
