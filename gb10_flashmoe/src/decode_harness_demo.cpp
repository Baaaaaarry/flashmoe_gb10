#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>

#include "flashmoe/compute_profile.h"
#include "flashmoe/decode_harness.h"
#include "flashmoe/device_path.h"
#include "flashmoe/expert_store.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/online_runtime.h"
#include "flashmoe/runtime_plan.h"

using flashmoe::ModelFamily;

namespace {

void print_report(const flashmoe::DevicePathProfile& profile,
                  const flashmoe::DecodeHarnessReport& report) {
    std::cout << "profile=" << profile.name << '\n';
    std::cout << "unpack_backend=" << flashmoe::to_string(profile.unpack_backend) << '\n';
    std::cout << "compute_backend=" << flashmoe::to_string(profile.compute_backend) << '\n';
    if (profile.measured_compute.has_value()) {
        std::cout << "measured_compute_backend=" << profile.measured_compute->backend_name() << '\n';
    }
    std::cout << "steps=" << report.steps << '\n';
    std::cout << "cold_steps=" << report.cold_steps << '\n';
    std::cout << "warm_steps=" << report.warm_steps << '\n';
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
    std::cout << "promoted_gb=" << report.slot_cache.promoted_gb << '\n';
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
        std::cerr << "Usage: flashmoe_decode_harness MODEL_FAMILY expert_manifest.json routing_trace.txt [max_steps] [compute_profile.json]\n";
        return EXIT_FAILURE;
    }

    const ModelFamily family = flashmoe::parse_model_family(argv[1]);
    if (family == ModelFamily::kUnknown) {
        std::cerr << "unknown model family: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    const auto& spec = flashmoe::builtin_model_spec(family);
    const auto hw = flashmoe::gb10_hardware_profile();
    const auto plan = flashmoe::recommend_runtime_plan(spec, hw);
    const auto store = flashmoe::ExpertManifestStore::from_json_file(argv[2]);
    const auto trace = flashmoe::load_route_trace_file(argv[3]);
    const std::size_t max_steps = argc >= 5 ? static_cast<std::size_t>(std::stoull(argv[4])) : 0;
    std::optional<flashmoe::ComputeProfile> compute_profile;
    if (argc >= 6) {
        compute_profile = flashmoe::ComputeProfile::from_json_file(argv[5]);
    }

    auto host_profile = flashmoe::host_unpack_profile(spec, hw, plan);
    auto gpu_profile = flashmoe::gpu_unpack_profile(spec, hw, plan);
    if (compute_profile.has_value()) {
        host_profile.measured_compute = compute_profile;
        gpu_profile.measured_compute = compute_profile;
        host_profile.compute_backend = flashmoe::ComputeBackend::kGroupedGemv;
        gpu_profile.compute_backend = flashmoe::ComputeBackend::kGroupedGemv;
    }
    const auto host_report = flashmoe::run_decode_harness(spec, plan, hw, host_profile, store, trace, max_steps);
    const auto gpu_report = flashmoe::run_decode_harness(spec, plan, hw, gpu_profile, store, trace, max_steps);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "model=" << spec.short_name << '\n';
    print_report(host_profile, host_report);
    print_report(gpu_profile, gpu_report);
    return EXIT_SUCCESS;
}
