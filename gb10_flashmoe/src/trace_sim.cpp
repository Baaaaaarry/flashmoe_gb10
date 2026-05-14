#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "flashmoe/expert_cache.h"

using flashmoe::CacheStats;
using flashmoe::EntryView;
using flashmoe::EvictionPolicy;
using flashmoe::ExpertCache;
using flashmoe::ExpertId;
using flashmoe::ExpertRequest;
using flashmoe::FutureTraceOracle;
using flashmoe::PolicyKind;
using flashmoe::PolicyTuning;

namespace {

struct TraceLine {
    std::uint64_t token = 0;
    std::uint16_t layer = 0;
    std::vector<std::uint16_t> experts;
};

std::vector<TraceLine> load_trace(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open trace file: " + path);
    }

    std::vector<TraceLine> lines;
    std::string raw;
    while (std::getline(input, raw)) {
        if (raw.empty() || raw[0] == '#') {
            continue;
        }
        std::istringstream row(raw);
        TraceLine line;
        row >> line.token >> line.layer;
        std::uint16_t expert = 0;
        while (row >> expert) {
            line.experts.push_back(expert);
        }
        if (!line.experts.empty()) {
            lines.push_back(std::move(line));
        }
    }
    return lines;
}

void run_policy(const std::string& label,
                PolicyKind kind,
                std::size_t cache_bytes,
                std::size_t expert_bytes,
                const std::vector<TraceLine>& trace,
                const FutureTraceOracle* oracle = nullptr) {
    ExpertCache cache(cache_bytes, EvictionPolicy(kind, PolicyTuning{}, oracle));
    std::vector<ExpertId> evicted;
    std::uint64_t step = 0;
    for (const auto& line : trace) {
        for (const auto expert : line.experts) {
            cache.touch_or_insert(ExpertRequest{
                .id = ExpertId{line.layer, expert},
                .bytes = expert_bytes,
                .step = step++,
            }, &evicted);
        }
    }

    const CacheStats& stats = cache.stats();
    const double total = static_cast<double>(stats.hits + stats.misses);
    const double hit_rate = total > 0.0 ? (100.0 * static_cast<double>(stats.hits) / total) : 0.0;
    std::cout << std::left << std::setw(20) << label
              << " hit_rate=" << std::setw(7) << std::fixed << std::setprecision(2) << hit_rate << "%"
              << " evictions=" << std::setw(8) << stats.evictions
              << " cache_GB=" << std::setw(6) << std::setprecision(1)
              << (static_cast<double>(cache_bytes) / 1e9)
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: flashmoe_trace_sim TRACE.txt [cache_gb] [expert_mb]\n";
        std::cerr << "Trace format: token layer expert0 expert1 expert2 expert3\n";
        return EXIT_FAILURE;
    }

    const std::string trace_path = argv[1];
    const double cache_gb = argc >= 3 ? std::stod(argv[2]) : 96.0;
    const double expert_mb = argc >= 4 ? std::stod(argv[3]) : 5.44;
    const std::size_t cache_bytes = static_cast<std::size_t>(cache_gb * 1e9);
    const std::size_t expert_bytes = static_cast<std::size_t>(expert_mb * 1024.0 * 1024.0);

    const auto trace = load_trace(trace_path);
    FutureTraceOracle oracle;
    std::uint64_t step = 0;
    for (const auto& line : trace) {
        for (const auto expert : line.experts) {
            oracle.add_occurrence(ExpertId{line.layer, expert}, step++);
        }
    }
    oracle.finalize();

    run_policy("lru", PolicyKind::kLru, cache_bytes, expert_bytes, trace);
    run_policy("lfu", PolicyKind::kLfu, cache_bytes, expert_bytes, trace);
    run_policy("recency_frequency", PolicyKind::kRecencyFrequency, cache_bytes, expert_bytes, trace);
    run_policy("oracle", PolicyKind::kOracle, cache_bytes, expert_bytes, trace, &oracle);
    return EXIT_SUCCESS;
}
