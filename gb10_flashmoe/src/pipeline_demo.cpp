#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "flashmoe/pipeline_model.h"

using flashmoe::PipelineConfig;
using flashmoe::estimate_layer_latency;

int main() {
    PipelineConfig apple_like{
        .cmd1_ms = 1.22,
        .cmd2_ms = 0.55,
        .expert_io_ms = 2.41,
        .expert_compute_ms = 0.16,
        .overlap_factor = 0.05,
        .allow_gpu_io_overlap = true,
    };
    PipelineConfig gb10_target{
        .cmd1_ms = 0.80,
        .cmd2_ms = 0.45,
        .expert_io_ms = 1.55,
        .expert_compute_ms = 0.20,
        .overlap_factor = 0.55,
        .allow_gpu_io_overlap = true,
    };

    const auto apple = estimate_layer_latency(apple_like);
    const auto gb10 = estimate_layer_latency(gb10_target);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Apple-like serial layer: " << apple.serial_layer_ms
              << " ms, steady state: " << apple.steady_state_layer_ms << " ms\n";
    std::cout << "GB10 target serial layer: " << gb10.serial_layer_ms
              << " ms, steady state: " << gb10.steady_state_layer_ms << " ms\n";
    std::cout << "Estimated decode throughput at 60 layers: "
              << (1000.0 / (gb10.steady_state_layer_ms * 60.0)) << " tok/s\n";
    return EXIT_SUCCESS;
}
