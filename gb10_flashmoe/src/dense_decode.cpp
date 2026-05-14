#include "flashmoe/dense_decode.h"

#include <algorithm>

namespace flashmoe {

DenseDecodeStep estimate_dense_decode_step(const ModelSpec& spec,
                                           const RuntimePlan& plan,
                                           const DenseResidentPlan& resident,
                                           std::size_t context_tokens,
                                           std::size_t generated_tokens) {
    const double memory_scale = spec.supports_runtime_fp8_compute ? 0.55 : 1.0;
    const double context_scale = 1.0 + (static_cast<double>(std::min<std::size_t>(context_tokens, 32768)) / 32768.0) * 0.65;
    const double generation_scale = 1.0 + std::min<double>(static_cast<double>(generated_tokens) / 512.0, 0.25);

    DenseDecodeStep step;
    step.embed_ms = 0.018 + 0.002 * memory_scale;
    step.attention_ms = (0.095 + 0.020 * memory_scale) * context_scale;
    step.norm_router_ms = 0.026 + 0.004 * generation_scale;
    step.lm_head_ms = 0.032 + 0.005 * memory_scale;

    const double resident_pressure = resident.total_resident_gb / std::max(1.0, plan.dense_resident_budget_gb);
    step.attention_ms *= resident_pressure;
    step.total_ms = step.embed_ms + step.attention_ms + step.norm_router_ms + step.lm_head_ms;
    return step;
}

}  // namespace flashmoe
