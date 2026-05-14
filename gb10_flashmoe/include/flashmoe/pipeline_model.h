#pragma once

#include <cstddef>

namespace flashmoe {

struct PipelineConfig {
    double cmd1_ms = 0.90;
    double cmd2_ms = 0.55;
    double expert_io_ms = 2.40;
    double expert_compute_ms = 0.25;
    double overlap_factor = 0.60;
    bool allow_gpu_io_overlap = true;
};

struct PipelineEstimate {
    double steady_state_layer_ms = 0.0;
    double serial_layer_ms = 0.0;
    double overlapped_ms = 0.0;
};

PipelineEstimate estimate_layer_latency(const PipelineConfig& config);

}  // namespace flashmoe
