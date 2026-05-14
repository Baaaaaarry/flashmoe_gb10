#include "flashmoe/dense_resident_loader.h"

namespace flashmoe {

DenseResidentPlan build_dense_resident_plan(const ModelSpec& spec, const RuntimePlan& plan) {
    DenseResidentPlan resident;
    resident.total_resident_gb = plan.dense_resident_budget_gb;
    resident.excluded_routed_groups = {
        "routed expert gate_proj",
        "routed expert up_proj",
        "routed expert down_proj",
    };

    const double embeddings_gb = resident.total_resident_gb * 0.18;
    const double attention_gb = resident.total_resident_gb * 0.38;
    const double norms_and_router_gb = resident.total_resident_gb * 0.12;
    const double shared_expert_gb = resident.total_resident_gb * 0.20;
    const double lm_head_gb = resident.total_resident_gb
        - embeddings_gb - attention_gb - norms_and_router_gb - shared_expert_gb;

    resident.resident_groups = {
        {"token_embedding", embeddings_gb, true},
        {"attention_and_projection", attention_gb, true},
        {"norms_and_router", norms_and_router_gb, true},
        {"shared_experts", shared_expert_gb, spec.shared_experts_per_layer > 0},
        {"lm_head", lm_head_gb, true},
    };
    return resident;
}

}  // namespace flashmoe
