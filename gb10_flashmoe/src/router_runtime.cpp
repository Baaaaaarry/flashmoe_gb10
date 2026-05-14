#include "flashmoe/router_runtime.h"

#include <algorithm>

namespace flashmoe {
RuntimeRouteStep runtime_route_for_layer(const ModelSpec& spec,
                                         const DenseOperatorState& dense_state,
                                         std::uint16_t layer,
                                         const DenseRuntimeArtifact* artifact) {
    RuntimeRouteStep step;
    step.layer = layer;
    const auto selection = router_selection_from_hidden(dense_state, artifact, spec, layer);
    step.experts = selection.experts;
    step.weights = selection.weights;
    return step;
}

}  // namespace flashmoe
