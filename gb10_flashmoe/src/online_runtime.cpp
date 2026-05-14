#include "flashmoe/online_runtime.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "flashmoe/cache_policy.h"

namespace flashmoe {
namespace {

std::uint64_t make_seen_key(ExpertId id) {
    return (static_cast<std::uint64_t>(id.layer) << 32U) | static_cast<std::uint64_t>(id.expert);
}

}  // namespace

std::vector<RouteStep> load_route_trace_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open route trace: " + path);
    }

    std::vector<RouteStep> out;
    std::string raw;
    std::uint64_t step = 0;
    while (std::getline(input, raw)) {
        if (raw.empty() || raw[0] == '#') {
            continue;
        }
        const auto colon = raw.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        RouteStep item;
        item.step = step++;
        item.layer = static_cast<std::uint16_t>(std::stoi(raw.substr(0, colon)));
        std::stringstream experts(raw.substr(colon + 1));
        std::string token;
        while (std::getline(experts, token, ',')) {
            if (!token.empty()) {
                item.experts.push_back(static_cast<std::uint16_t>(std::stoi(token)));
            }
        }
        if (!item.experts.empty()) {
            out.push_back(std::move(item));
        }
    }
    return out;
}

OnlineDecodeScheduler::OnlineDecodeScheduler(
    DenseResidentPlan resident_plan,
    RuntimePlan runtime_plan,
    ExpertManifestStore store,
    ExpertCache cache)
    : resident_plan_(std::move(resident_plan)),
      runtime_plan_(std::move(runtime_plan)),
      store_(std::move(store)),
      cache_(std::move(cache)) {
    stats_.dense_resident_gb = resident_plan_.total_resident_gb;
    stats_.hot_cache_capacity_gb = runtime_plan_.hot_expert_cache_budget_gb;
}

void OnlineDecodeScheduler::process(const RouteStep& step) {
    stats_.route_steps += 1;
    stats_.active_expert_peak = std::max<std::uint64_t>(stats_.active_expert_peak, step.experts.size());
    std::vector<ExpertId> evicted;

    for (const auto expert : step.experts) {
        const ExpertId id{step.layer, expert};
        stats_.routed_requests += 1;
        const auto* record = store_.find(id);
        if (record == nullptr) {
            stats_.missing_manifest_entries += 1;
            continue;
        }
        const std::uint64_t seen_key = make_seen_key(id);
        bool already_seen = false;
        for (const auto prior : seen_) {
            if (make_seen_key(prior) == seen_key) {
                already_seen = true;
                break;
            }
        }
        if (!already_seen) {
            seen_.push_back(id);
            stats_.unique_experts_seen += 1;
        }

        const bool hit = cache_.touch_or_insert(ExpertRequest{
            .id = id,
            .bytes = record->bytes,
            .step = step.step,
        }, &evicted);
        if (!hit) {
            stats_.cold_bytes_touched_gb += static_cast<double>(record->bytes) / 1e9;
        }
    }

    stats_.cache_stats = cache_.stats();
    stats_.hot_cache_used_gb = static_cast<double>(cache_.bytes_used()) / 1e9;
}

}  // namespace flashmoe
