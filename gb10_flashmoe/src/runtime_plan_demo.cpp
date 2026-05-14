#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "flashmoe/model_spec.h"
#include "flashmoe/runtime_plan.h"

using flashmoe::HardwareProfile;
using flashmoe::ModelFamily;
using flashmoe::RuntimePlan;

namespace {

void print_lines(std::string_view title, const std::vector<std::string_view>& lines) {
    std::cout << title << ":\n";
    for (const auto line : lines) {
        std::cout << "  - " << line << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: flashmoe_runtime_plan MODEL_FAMILY\n";
        std::cerr << "  examples: qwen3.5-397b-a17b | deepseek-v4-flash\n";
        return EXIT_FAILURE;
    }

    const ModelFamily family = flashmoe::parse_model_family(argv[1]);
    if (family == ModelFamily::kUnknown) {
        std::cerr << "unknown model family: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    const auto& spec = flashmoe::builtin_model_spec(family);
    const HardwareProfile hw = flashmoe::gb10_hardware_profile();
    const RuntimePlan plan = flashmoe::recommend_runtime_plan(spec, hw);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "model=" << spec.short_name << '\n';
    std::cout << "hardware=" << hw.name << '\n';
    std::cout << "runtime_family=" << plan.runtime_family << '\n';
    std::cout << "cold_storage_format=" << flashmoe::to_string(plan.cold_storage_format) << '\n';
    std::cout << "hot_cache_compute_format=" << flashmoe::to_string(plan.hot_cache_compute_format) << '\n';
    std::cout << "dense_resident_budget_gb=" << plan.dense_resident_budget_gb << '\n';
    std::cout << "hot_expert_cache_budget_gb=" << plan.hot_expert_cache_budget_gb << '\n';
    std::cout << "kv_cache_budget_gb=" << plan.kv_cache_budget_gb << '\n';
    std::cout << "workspace_budget_gb=" << plan.workspace_budget_gb << '\n';
    std::cout << "safety_margin_gb=" << plan.safety_margin_gb << '\n';
    std::cout << "total_runtime_budget_gb=" << plan.total_runtime_budget_gb << '\n';
    std::cout << "estimated_cold_tail_gb=" << plan.estimated_cold_tail_gb << '\n';
    std::cout << "use_framework_parameter_tree_for_routed_experts="
              << (plan.use_framework_parameter_tree_for_routed_experts ? "true" : "false") << '\n';
    std::cout << "use_layer_major_bundle_layout=" << (plan.use_layer_major_bundle_layout ? "true" : "false") << '\n';
    std::cout << "use_async_pread=" << (plan.use_async_pread ? "true" : "false") << '\n';
    std::cout << "use_gpu_side_unpack=" << (plan.use_gpu_side_unpack ? "true" : "false") << '\n';

    print_lines("reasons", plan.reasons);
    print_lines("milestones", plan.milestones);
    print_lines("risks", plan.risks);

    const auto compatibility = flashmoe::compatibility_notes(spec);
    std::cout << "compatibility_notes:\n";
    for (const auto& note : compatibility) {
        std::cout << "  - " << note.component << ": " << note.guidance << '\n';
    }
    return EXIT_SUCCESS;
}
