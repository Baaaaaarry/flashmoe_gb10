#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "flashmoe/decode_state.h"
#include "flashmoe/dense_operator_chain.h"
#include "flashmoe/model_spec.h"

namespace flashmoe {

struct RuntimeRouteStep {
    std::uint16_t layer = 0;
    std::vector<std::uint16_t> experts;
    std::vector<float> weights;
};

RuntimeRouteStep runtime_route_for_layer(const ModelSpec& spec,
                                         const DenseOperatorState& dense_state,
                                         std::uint16_t layer,
                                         const DenseRuntimeArtifact* artifact = nullptr);

}  // namespace flashmoe
