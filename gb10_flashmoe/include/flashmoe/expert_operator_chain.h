#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "flashmoe/decode_harness.h"
#include "flashmoe/dense_operator_chain.h"
#include "flashmoe/expert_store.h"

namespace flashmoe {

struct MaterializedExpert {
    std::vector<std::uint8_t> bytes;
    std::size_t hidden_dim = 0;
    std::size_t intermediate_dim = 0;
    std::vector<float> gate_weights;
    std::vector<float> up_weights;
    std::vector<float> down_weights;
    bool host_ready = false;
    std::uintptr_t device_gate = 0;
    std::uintptr_t device_up = 0;
    std::uintptr_t device_down = 0;
    std::size_t device_gate_len = 0;
    std::size_t device_up_len = 0;
    std::size_t device_down_len = 0;
    bool device_ready = false;
};

using MaterializedExpertMap = std::unordered_map<ExpertId, MaterializedExpert, ExpertIdHash>;

double materialize_expert_record(const ExpertRecord& record,
                                 MaterializedExpertMap& materialized);
double unpack_materialized_expert(const ExpertRecord& record,
                                  MaterializedExpertMap& materialized);
double execute_materialized_expert_fused(const ExpertRecord& record,
                                         DenseOperatorState& dense_state,
                                         const DenseRuntimeArtifact* artifact,
                                         MaterializedExpertMap& materialized,
                                         float route_weight = 1.0f);
void release_materialized_expert(MaterializedExpert& slot);

}  // namespace flashmoe
