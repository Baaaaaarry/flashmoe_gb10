#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "flashmoe/dense_artifact.h"
#include "flashmoe/kv_cache.h"
#include "flashmoe/model_spec.h"

namespace flashmoe {

struct DenseOperatorState {
    std::vector<float> hidden;     // residual-connected hidden state (unnormed)
    std::vector<float> moe_input;  // post-attn-norm hidden fed into MoE experts
    std::vector<float> logits;
    std::uint32_t last_token_id = 0;
};

struct DenseOperatorMetrics {
    double embed_ms = 0.0;
    double attention_ms = 0.0;
    double norm_router_ms = 0.0;
    double lm_head_ms = 0.0;
};

struct RouterSelection {
    std::vector<std::uint16_t> experts;
    std::vector<float> weights;
};

DenseOperatorState make_dense_operator_state(const ModelSpec& spec);
std::uint32_t prompt_token_id(std::size_t index, std::size_t total, std::uint64_t seed);

// Embed a single token into state.hidden (direct model mode).
void embed_token_direct(DenseOperatorState& state,
                        const DenseRuntimeArtifact& artifact,
                        std::uint32_t token_id);

// Run one transformer layer's attention for direct model mode.
// Updates kv_cache[layer_idx][position], adds attn output to state.hidden (residual),
// then computes state.moe_input = post_attn_norm(state.hidden).
void run_layer_attention_direct(DenseOperatorState& state,
                                KvCacheState& kv_cache,
                                const DenseRuntimeArtifact& artifact,
                                std::size_t layer_idx,
                                std::size_t position);

DenseOperatorMetrics dense_prefill_step(DenseOperatorState& state,
                                        KvCacheState& kv_cache,
                                        const DenseRuntimeArtifact* artifact,
                                        const ModelSpec& spec,
                                        std::uint32_t token_id,
                                        std::size_t position);
DenseOperatorMetrics dense_decode_step(DenseOperatorState& state,
                                       KvCacheState& kv_cache,
                                       const DenseRuntimeArtifact* artifact,
                                       const ModelSpec& spec,
                                       std::uint32_t token_id,
                                       std::size_t position);
void normalize_dense_hidden(DenseOperatorState& state);
void refresh_logits_from_hidden(DenseOperatorState& state, const DenseRuntimeArtifact* artifact = nullptr);
RouterSelection router_selection_from_hidden(const DenseOperatorState& state,
                                             const DenseRuntimeArtifact* artifact,
                                             const ModelSpec& spec,
                                             std::uint16_t layer);
std::vector<std::uint16_t> router_topk_from_hidden(const DenseOperatorState& state,
                                                   const DenseRuntimeArtifact* artifact,
                                                   const ModelSpec& spec,
                                                   std::uint16_t layer);
std::uint32_t sample_token_from_logits(const DenseOperatorState& state);

}  // namespace flashmoe
