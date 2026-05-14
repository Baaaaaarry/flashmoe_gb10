#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>

#include "flashmoe/compute_profile.h"
#include "flashmoe/device_path.h"
#include "flashmoe/expert_store.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/online_runtime.h"
#include "flashmoe/runtime_plan.h"
#include "flashmoe/streamed_runtime.h"

namespace {

void print_report(const flashmoe::DevicePathProfile& profile,
                  const flashmoe::StreamedRuntimeReport& report) {
    std::cout << "profile=" << profile.name << '\n';
    std::cout << "unpack_backend=" << flashmoe::to_string(profile.unpack_backend) << '\n';
    std::cout << "compute_backend=" << flashmoe::to_string(profile.compute_backend) << '\n';
    if (profile.measured_compute.has_value()) {
        std::cout << "measured_compute_backend=" << profile.measured_compute->backend_name() << '\n';
    }
    std::cout << "steps=" << report.steps << '\n';
    std::cout << "cold_steps=" << report.cold_steps << '\n';
    std::cout << "warm_steps=" << report.warm_steps << '\n';
    std::cout << "routed_requests=" << report.routed_requests << '\n';
    std::cout << "unique_experts_seen=" << report.unique_experts_seen << '\n';
    std::cout << "active_expert_peak=" << report.active_expert_peak << '\n';
    std::cout << "missing_manifest_entries=" << report.missing_manifest_entries << '\n';
    std::cout << "dense_resident_gb=" << report.dense_resident_gb << '\n';
    std::cout << "hot_cache_capacity_gb=" << report.hot_cache_capacity_gb << '\n';
    std::cout << "kv_cache_budget_gb=" << report.kv_cache_budget_gb << '\n';
    std::cout << "workspace_budget_gb=" << report.workspace_budget_gb << '\n';
    std::cout << "safety_margin_gb=" << report.safety_margin_gb << '\n';
    std::cout << "total_runtime_budget_gb=" << report.total_runtime_budget_gb << '\n';
    std::cout << "hot_cache_used_gb=" << report.hot_cache_used_gb << '\n';
    std::cout << "promoted_gb=" << report.promoted_gb << '\n';
    std::cout << "avg_route_ms=" << report.avg_route_ms << '\n';
    std::cout << "avg_load_ms=" << report.avg_load_ms << '\n';
    std::cout << "avg_unpack_ms=" << report.avg_unpack_ms << '\n';
    std::cout << "avg_compute_ms=" << report.avg_compute_ms << '\n';
    std::cout << "avg_combine_ms=" << report.avg_combine_ms << '\n';
    std::cout << "avg_total_ms=" << report.avg_total_ms << '\n';
    std::cout << "cold_avg_total_ms=" << report.cold_avg_total_ms << '\n';
    std::cout << "warm_avg_total_ms=" << report.warm_avg_total_ms << '\n';
    std::cout << "estimated_tok_per_s=" << report.estimated_tok_per_s << '\n';
    std::cout << "steady_state_tok_per_s=" << report.steady_state_tok_per_s << '\n';
    std::cout << "slot_promotions=" << report.slot_cache.promotions << '\n';
    std::cout << "slot_hits=" << report.slot_cache.slot_hits << '\n';
    std::cout << "slot_evictions=" << report.slot_cache.slot_evictions << '\n';
    std::cout << "resident_slot_gb=" << report.slot_cache.resident_slot_gb << '\n';
    for (const auto& sample : report.samples) {
        std::cout << "sample step=" << sample.step
                  << " layer=" << sample.layer
                  << " active=" << sample.active_experts
                  << " route_ms=" << sample.route_ms
                  << " load_ms=" << sample.load_ms
                  << " unpack_ms=" << sample.unpack_ms
                  << " compute_ms=" << sample.compute_ms
                  << " combine_ms=" << sample.combine_ms
                  << " total_ms=" << sample.total_ms
                  << " promotions=" << sample.slot_promotions
                  << " hits=" << sample.slot_hits
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: flashmoe_streamed_runtime MODEL_FAMILY expert_manifest.json routing_trace.txt [max_steps] [compute_profile.json] [cache_budget_gb]\n";
        return EXIT_FAILURE;
    }

    const auto family = flashmoe::parse_model_family(argv[1]);
    if (family == flashmoe::ModelFamily::kUnknown) {
        std::cerr << "unknown model family: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    const auto& spec = flashmoe::builtin_model_spec(family);
    const auto hw = flashmoe::gb10_hardware_profile();
    auto plan = flashmoe::recommend_runtime_plan(spec, hw);
    plan.use_gpu_side_unpack = true;
    if (argc >= 7) {
        plan.hot_expert_cache_budget_gb = std::stod(argv[6]);
    }

    const auto store = flashmoe::ExpertManifestStore::from_json_file(argv[2]);
    const auto trace = flashmoe::load_route_trace_file(argv[3]);
    const std::size_t max_steps = argc >= 5 ? static_cast<std::size_t>(std::stoull(argv[4])) : 0;

    auto profile = flashmoe::gpu_unpack_profile(spec, hw, plan);
    if (argc >= 6) {
        profile.measured_compute = flashmoe::ComputeProfile::from_json_file(argv[5]);
    }

    const flashmoe::StreamedMoERuntime runtime(spec, hw, plan, store, profile);
    const auto report = runtime.run(trace, max_steps);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "model=" << spec.short_name << '\n';
    std::cout << "runtime_family=" << plan.runtime_family << '\n';
    print_report(profile, report);
    return EXIT_SUCCESS;
}
