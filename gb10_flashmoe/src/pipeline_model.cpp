#include "flashmoe/pipeline_model.h"

#include <algorithm>

namespace flashmoe {

PipelineEstimate estimate_layer_latency(const PipelineConfig& config) {
    PipelineEstimate estimate{};
    estimate.serial_layer_ms = config.cmd1_ms + config.cmd2_ms + config.expert_io_ms + config.expert_compute_ms;

    const double overlappable = config.allow_gpu_io_overlap
        ? std::min(config.expert_io_ms, config.cmd1_ms + config.cmd2_ms + config.expert_compute_ms)
        : 0.0;
    estimate.overlapped_ms = overlappable * config.overlap_factor;
    estimate.steady_state_layer_ms = estimate.serial_layer_ms - estimate.overlapped_ms;
    return estimate;
}

}  // namespace flashmoe
