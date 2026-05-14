#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "flashmoe/cache_policy.h"
#include "flashmoe/dense_resident_loader.h"
#include "flashmoe/expert_cache.h"
#include "flashmoe/expert_store.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/online_runtime.h"
#include "flashmoe/runtime_plan.h"

using flashmoe::DenseResidentPlan;
using flashmoe::EvictionPolicy;
using flashmoe::ExpertCache;
using flashmoe::ExpertManifestStore;
using flashmoe::ModelFamily;
using flashmoe::OnlineDecodeScheduler;
using flashmoe::PolicyKind;
using flashmoe::PolicyTuning;
using flashmoe::RuntimePlan;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: flashmoe_online_runtime_demo MODEL_FAMILY expert_manifest.json routing_trace.txt [cache_gb] [max_steps]\n";
        return EXIT_FAILURE;
    }

    const ModelFamily family = flashmoe::parse_model_family(argv[1]);
    if (family == ModelFamily::kUnknown) {
        std::cerr << "unknown model family: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    const std::string manifest_path = argv[2];
    const std::string trace_path = argv[3];
    const double cache_gb_override = argc >= 5 ? std::stod(argv[4]) : -1.0;
    const std::size_t max_steps = argc >= 6 ? static_cast<std::size_t>(std::stoull(argv[5])) : 0;

    const auto& spec = flashmoe::builtin_model_spec(family);
    const auto hw = flashmoe::gb10_hardware_profile();
    RuntimePlan plan = flashmoe::recommend_runtime_plan(spec, hw);
    if (cache_gb_override > 0.0) {
        plan.hot_expert_cache_budget_gb = cache_gb_override;
    }
    const DenseResidentPlan resident = flashmoe::build_dense_resident_plan(spec, plan);
    const ExpertManifestStore store = ExpertManifestStore::from_json_file(manifest_path);
    auto route = flashmoe::load_route_trace_file(trace_path);
    if (max_steps > 0 && route.size() > max_steps) {
        route.resize(max_steps);
    }

    const std::size_t cache_bytes = static_cast<std::size_t>(plan.hot_expert_cache_budget_gb * 1e9);
    ExpertCache cache(cache_bytes, EvictionPolicy(PolicyKind::kRecencyFrequency, PolicyTuning{}, nullptr));
    OnlineDecodeScheduler scheduler(resident, plan, store, std::move(cache));
    for (const auto& step : route) {
        scheduler.process(step);
    }

    const auto& stats = scheduler.stats();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "model=" << spec.short_name << '\n';
    std::cout << "manifest_entries=" << store.size() << '\n';
    std::cout << "route_steps=" << stats.route_steps << '\n';
    std::cout << "routed_requests=" << stats.routed_requests << '\n';
    std::cout << "unique_experts_seen=" << stats.unique_experts_seen << '\n';
    std::cout << "active_expert_peak=" << stats.active_expert_peak << '\n';
    std::cout << "missing_manifest_entries=" << stats.missing_manifest_entries << '\n';
    std::cout << "dense_resident_gb=" << stats.dense_resident_gb << '\n';
    std::cout << "hot_cache_capacity_gb=" << stats.hot_cache_capacity_gb << '\n';
    std::cout << "hot_cache_used_gb=" << stats.hot_cache_used_gb << '\n';
    std::cout << "cold_bytes_touched_gb=" << stats.cold_bytes_touched_gb << '\n';
    std::cout << "cache_hits=" << stats.cache_stats.hits << '\n';
    std::cout << "cache_misses=" << stats.cache_stats.misses << '\n';
    std::cout << "cache_evictions=" << stats.cache_stats.evictions << '\n';
    std::cout << "cache_bytes_loaded_gb=" << static_cast<double>(stats.cache_stats.bytes_loaded) / 1e9 << '\n';
    std::cout << "cache_bytes_evicted_gb=" << static_cast<double>(stats.cache_stats.bytes_evicted) / 1e9 << '\n';
    return EXIT_SUCCESS;
}
