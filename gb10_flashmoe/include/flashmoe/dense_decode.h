#pragma once

#include <cstddef>

#include "flashmoe/dense_resident_loader.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/runtime_plan.h"

namespace flashmoe {

struct DenseDecodeStep {
    double embed_ms = 0.0;
    double attention_ms = 0.0;
    double norm_router_ms = 0.0;
    double lm_head_ms = 0.0;
    double total_ms = 0.0;
};

DenseDecodeStep estimate_dense_decode_step(const ModelSpec& spec,
                                           const RuntimePlan& plan,
                                           const DenseResidentPlan& resident,
                                           std::size_t context_tokens,
                                           std::size_t generated_tokens);

}  // namespace flashmoe
