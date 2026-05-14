#pragma once

#include <cstddef>
#include <cstdint>

#include "flashmoe/dense_operator_chain.h"
#include "flashmoe/expert_operator_chain.h"
#include "flashmoe/expert_store.h"

namespace flashmoe {

bool cuda_expert_backend_available();
double cuda_unpack_and_execute_expert(const ExpertRecord& record,
                                      DenseOperatorState& dense_state,
                                      const DenseRuntimeArtifact* artifact,
                                      MaterializedExpertMap& materialized,
                                      double* unpack_ms,
                                      double* compute_ms,
                                      double* combine_ms,
                                      float route_weight = 1.0f);
void cuda_release_expert_buffers(MaterializedExpert& slot);

}  // namespace flashmoe
