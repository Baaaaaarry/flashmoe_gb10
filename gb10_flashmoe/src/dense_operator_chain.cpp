#include "flashmoe/dense_operator_chain.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace flashmoe {
namespace {

constexpr std::size_t kRuntimeVocabSize = 32;

float seeded_weight(std::uint64_t a, std::uint64_t b) {
    std::uint64_t x = a * 0x9e3779b97f4a7c15ULL ^ (b + 0xbf58476d1ce4e5b9ULL);
    x ^= x >> 30U;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27U;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31U;
    return static_cast<float>((x % 20001ULL) / 10000.0 - 1.0);
}

float finite_or_zero(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

void sanitize(std::vector<float>& values) {
    for (auto& value : values) {
        if (!std::isfinite(value)) {
            value = 0.0f;
        }
    }
}

void normalize(std::vector<float>& values) {
    double sum_sq = 0.0;
    for (auto& value : values) {
        if (!std::isfinite(value)) {
            value = 0.0f;
        }
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(std::max(1e-6, sum_sq / std::max<std::size_t>(1, values.size()))));
    for (auto& value : values) {
        value *= inv;
    }
}

float rms_inv(const std::vector<float>& values, const std::vector<std::uint32_t>& indices) {
    double sum_sq = 0.0;
    for (const auto idx : indices) {
        if (idx < values.size()) {
            const auto value = finite_or_zero(values[idx]);
            sum_sq += static_cast<double>(value) * static_cast<double>(value);
        }
    }
    return static_cast<float>(1.0 / std::sqrt(std::max(1e-6, sum_sq / std::max<std::size_t>(1, indices.size()))));
}

std::vector<float> sampled_hidden(const DenseOperatorState& state, const DenseRuntimeArtifact& artifact) {
    std::vector<float> sampled(artifact.sampled_dims(), 0.0f);
    for (std::size_t d = 0; d < artifact.sampled_dims(); ++d) {
        const auto idx = artifact.sampled_indices[d];
        if (idx < state.hidden.size()) {
            sampled[d] = state.hidden[idx];
        }
    }
    return sampled;
}

std::vector<float> matvec(const std::vector<float>& matrix, std::size_t dim, std::size_t row_offset, const std::vector<float>& input) {
    std::vector<float> output(dim, 0.0f);
    const std::size_t base = row_offset * dim * dim;
    for (std::size_t r = 0; r < dim; ++r) {
        float accum = 0.0f;
        const std::size_t row = base + r * dim;
        for (std::size_t c = 0; c < dim; ++c) {
            accum += finite_or_zero(matrix[row + c]) * finite_or_zero(input[c]);
        }
        output[r] = finite_or_zero(accum);
    }
    return output;
}

// Apply RMS norm in-place: values[i] = values[i] * inv_rms * norm_weights[i]
void apply_rms_norm(std::vector<float>& values, const std::vector<float>& norm_weights) {
    double sum_sq = 0.0;
    for (const auto v : values) {
        const auto s = finite_or_zero(v);
        sum_sq += static_cast<double>(s) * static_cast<double>(s);
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(std::max(1e-6, sum_sq / std::max<std::size_t>(1, values.size()))));
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float w = i < norm_weights.size() ? finite_or_zero(norm_weights[i]) : 1.0f;
        values[i] = finite_or_zero(values[i]) * inv * w;
    }
}

// Run attention over the per-layer KV cache for the given layer and current position.
// Uses a sliding window of up to 32 past tokens.
std::vector<float> run_attention(const std::vector<float>& query,
                                 const KvCacheState& kv_cache,
                                 std::size_t layer_idx,
                                 std::size_t position,
                                 std::size_t attn_dim) {
    const std::size_t window = std::min<std::size_t>(position + 1U, 32U);
    double max_score = -std::numeric_limits<double>::infinity();
    std::array<float, 32> raw_scores{};
    for (std::size_t back = 0; back < window; ++back) {
        const std::size_t token_index = position - back;
        const auto* key = kv_cache_key_ptr(kv_cache, token_index, layer_idx);
        float score = 0.0f;
        for (std::size_t i = 0; i < attn_dim; ++i) {
            score += finite_or_zero(query[i]) * finite_or_zero(key[i]);
        }
        raw_scores[back] = finite_or_zero(score / std::sqrt(static_cast<float>(attn_dim + 1U)));
        max_score = std::max<double>(max_score, raw_scores[back]);
    }
    std::array<float, 32> scores{};
    double score_sum = 0.0;
    for (std::size_t back = 0; back < window; ++back) {
        const float s = std::exp(static_cast<float>(raw_scores[back] - max_score));
        scores[back] = s;
        score_sum += s;
    }
    const double inv_score_sum = score_sum > 0.0 ? 1.0 / score_sum : 1.0;
    std::vector<float> context(kv_cache.kv_width, 0.0f);
    for (std::size_t back = 0; back < window; ++back) {
        const std::size_t token_index = position - back;
        const auto* value = kv_cache_value_ptr(kv_cache, token_index, layer_idx);
        const float weight = static_cast<float>(scores[back] * inv_score_sum);
        for (std::size_t i = 0; i < attn_dim; ++i) {
            context[i] += finite_or_zero(weight) * finite_or_zero(value[i]);
        }
    }
    sanitize(context);
    return context;
}

DenseOperatorMetrics run_dense_pass(DenseOperatorState& state,
                                    KvCacheState& kv_cache,
                                    const DenseRuntimeArtifact* artifact,
                                    const ModelSpec& spec,
                                    std::uint32_t token_id,
                                    std::size_t position,
                                    float decode_scale) {
    const std::size_t hidden = state.hidden.size();
    std::vector<float> mixed(hidden, 0.0f);
    std::vector<float> attended(hidden, 0.0f);

    const bool use_direct_model = artifact != nullptr && artifact->uses_direct_model();
    if (use_direct_model) {
        // Direct model mode: embedding and per-layer attention are handled separately
        // (embed_token_direct + run_layer_attention_direct). This path is unused for
        // direct_model_mode; run_dense_pass is kept for the non-direct fallback only.
        const auto embed = lookup_embedding_from_model_dir(*artifact, token_id);
        for (std::size_t i = 0; i < hidden; ++i) {
            mixed[i] = i < embed.size() ? finite_or_zero(embed[i]) : 0.0f;
        }
        // No attention applied here for direct_model_mode; set attended = mixed.
        for (std::size_t i = 0; i < hidden; ++i) {
            attended[i] = mixed[i];
        }
    } else {
        for (std::size_t i = 0; i < hidden; ++i) {
            const float embed = seeded_weight(token_id + 17U, i + 11U) * 0.35f;
            mixed[i] = state.hidden[i] * (0.82f + 0.03f * decode_scale) + embed;
        }
        if (artifact != nullptr && artifact->has_embeddings()) {
            const auto dims = artifact->sampled_dims();
            const auto vocab_row = token_id % artifact->runtime_vocab_size;
            for (std::size_t d = 0; d < dims; ++d) {
                const auto hidden_idx = artifact->sampled_indices[d];
                if (hidden_idx < mixed.size()) {
                    mixed[hidden_idx] += artifact->embedding_weights[vocab_row * dims + d];
                }
            }
        }
    }

    const bool use_artifact_attention = artifact != nullptr && artifact->has_attention() && artifact->has_norms() && !use_direct_model;
    const std::size_t artifact_attn_dim = use_artifact_attention ? artifact->sampled_dims() : 0;
    const std::size_t attn_dim = use_artifact_attention
        ? std::min<std::size_t>(artifact_attn_dim, kv_cache.kv_width)
        : kv_cache.kv_width;
    std::vector<float> query(attn_dim, 0.0f);
    std::vector<float> projected_q;
    std::vector<float> projected_k;
    std::vector<float> projected_v;

    if (!use_direct_model) {
        if (use_artifact_attention) {
            const auto layer_index = std::min<std::size_t>(position % std::max<std::size_t>(1, artifact->num_layers), artifact->num_layers - 1);
            auto sampled = sampled_hidden(state, *artifact);
            const float pre_inv = rms_inv(state.hidden, artifact->sampled_indices);
            for (std::size_t d = 0; d < sampled.size(); ++d) {
                sampled[d] *= pre_inv * artifact->input_norm_weights[layer_index * sampled.size() + d];
            }
            projected_q = matvec(artifact->q_proj_weights, sampled.size(), layer_index, sampled);
            projected_k = matvec(artifact->k_proj_weights, sampled.size(), layer_index, sampled);
            projected_v = matvec(artifact->v_proj_weights, sampled.size(), layer_index, sampled);
            for (std::size_t i = 0; i < std::min<std::size_t>(query.size(), projected_q.size()); ++i) {
                query[i] = projected_q[i];
            }
        } else {
            for (std::size_t i = 0; i < kv_cache.kv_width; ++i) {
                query[i] = mixed[(i * 13U + position * 7U) % hidden] * seeded_weight(token_id + 31U, i + 5U);
            }
        }

        if (position < kv_cache.tokens && kv_cache.kv_width > 0) {
            float* current_key = kv_cache_key_ptr(kv_cache, position, 0);
            float* current_value = kv_cache_value_ptr(kv_cache, position, 0);
            for (std::size_t i = 0; i < kv_cache.kv_width; ++i) {
                if (use_artifact_attention && i < attn_dim && i < projected_k.size() && i < projected_v.size()) {
                    current_key[i] = finite_or_zero(projected_k[i]);
                    current_value[i] = finite_or_zero(projected_v[i]);
                } else {
                    current_key[i] = finite_or_zero(mixed[(i * 11U + 3U) % hidden] * seeded_weight(token_id + 101U, i + 17U));
                    current_value[i] = finite_or_zero(mixed[(i * 19U + 9U) % hidden] * seeded_weight(token_id + 211U, i + 23U));
                }
            }
        }

        const auto context = run_attention(query, kv_cache, 0, position, attn_dim);

        std::vector<float> projected_o;
        if (use_artifact_attention) {
            const auto layer_index = std::min<std::size_t>(position % std::max<std::size_t>(1, artifact->num_layers), artifact->num_layers - 1);
            std::vector<float> o_input(artifact->sampled_dims(), 0.0f);
            for (std::size_t i = 0; i < std::min<std::size_t>(attn_dim, o_input.size()); ++i) {
                o_input[i] = context[i];
            }
            projected_o = matvec(artifact->o_proj_weights, artifact->sampled_dims(), layer_index, o_input);
        }

        for (std::size_t i = 0; i < hidden; ++i) {
            const std::size_t j = (i * 17U + position * 13U + token_id) % hidden;
            const std::size_t k = (i * 29U + position * 7U + 3U) % hidden;
            const float ctx = context[(i * 7U + position) % kv_cache.kv_width];
            attended[i] = 0.54f * mixed[i] + 0.18f * mixed[j] + 0.12f * mixed[k] + 0.16f * ctx;
        }
        if (use_artifact_attention) {
            const auto layer_index = std::min<std::size_t>(position % std::max<std::size_t>(1, artifact->num_layers), artifact->num_layers - 1);
            for (std::size_t d = 0; d < artifact->sampled_dims(); ++d) {
                const auto hidden_idx = artifact->sampled_indices[d];
                if (hidden_idx < attended.size() && d < projected_o.size()) {
                    attended[hidden_idx] += 0.25f * projected_o[d];
                }
            }
            const float post_inv = rms_inv(attended, artifact->sampled_indices);
            for (std::size_t d = 0; d < artifact->sampled_dims(); ++d) {
                const auto hidden_idx = artifact->sampled_indices[d];
                if (hidden_idx < attended.size()) {
                    attended[hidden_idx] *= post_inv * artifact->post_norm_weights[layer_index * artifact->sampled_dims() + d];
                }
            }
        }
    }

    sanitize(attended);
    normalize(attended);
    state.hidden.swap(attended);
    state.moe_input = state.hidden;
    state.last_token_id = token_id;

    refresh_logits_from_hidden(state, artifact);

    const float base_scale = static_cast<float>(hidden) / std::max(1.0f, static_cast<float>(spec.hidden_size));
    return DenseOperatorMetrics{
        .embed_ms = 0.010 * base_scale,
        .attention_ms = (0.080 + 0.0015 * static_cast<double>(std::min<std::size_t>(position, 4096U))) * decode_scale,
        .norm_router_ms = 0.020 * base_scale,
        .lm_head_ms = 0.030 * base_scale,
    };
}

}  // namespace

DenseOperatorState make_dense_operator_state(const ModelSpec& spec) {
    DenseOperatorState state;
    state.hidden.assign(spec.hidden_size, 0.0f);
    state.moe_input.assign(spec.hidden_size, 0.0f);
    state.logits.assign(kRuntimeVocabSize, 0.0f);
    return state;
}

std::uint32_t prompt_token_id(std::size_t index, std::size_t total, std::uint64_t seed) {
    return static_cast<std::uint32_t>((seed + index * 131U + total * 17U) % kRuntimeVocabSize);
}

void embed_token_direct(DenseOperatorState& state,
                        const DenseRuntimeArtifact& artifact,
                        std::uint32_t token_id) {
    const auto embed = lookup_embedding_from_model_dir(artifact, token_id);
    const std::size_t hidden = state.hidden.size();
    for (std::size_t i = 0; i < hidden; ++i) {
        state.hidden[i] = i < embed.size() ? finite_or_zero(embed[i]) : 0.0f;
    }
    state.moe_input.assign(hidden, 0.0f);
    state.last_token_id = token_id;
}

void run_layer_attention_direct(DenseOperatorState& state,
                                KvCacheState& kv_cache,
                                const DenseRuntimeArtifact& artifact,
                                std::size_t layer_idx,
                                std::size_t position) {
    const std::size_t hidden = state.hidden.size();
    const std::size_t attn_dim = kv_cache.kv_width;

    // Pre-attention RMS norm (does not modify state.hidden).
    // Always normalize: apply_rms_norm uses w=1.0 fallback when norm_weights is empty.
    auto normed = state.hidden;
    const auto input_norm = lookup_norm_from_model_dir(artifact, layer_idx, false);
    apply_rms_norm(normed, input_norm);

    // Q/K/V projections
    auto projected_q = attention_projection_from_model_dir(artifact, layer_idx, "q", normed, attn_dim);
    auto projected_k = attention_projection_from_model_dir(artifact, layer_idx, "k", normed, attn_dim);
    auto projected_v = attention_projection_from_model_dir(artifact, layer_idx, "v", normed, attn_dim);

    // Write K/V into per-layer KV cache
    if (position < kv_cache.tokens && kv_cache.kv_width > 0) {
        float* key_ptr = kv_cache_key_ptr(kv_cache, position, layer_idx);
        float* val_ptr = kv_cache_value_ptr(kv_cache, position, layer_idx);
        for (std::size_t i = 0; i < kv_cache.kv_width; ++i) {
            key_ptr[i] = i < projected_k.size() ? finite_or_zero(projected_k[i]) : 0.0f;
            val_ptr[i] = i < projected_v.size() ? finite_or_zero(projected_v[i]) : 0.0f;
        }
    }

    // Attention over past positions for this layer
    std::vector<float> query(attn_dim, 0.0f);
    for (std::size_t i = 0; i < std::min(attn_dim, projected_q.size()); ++i) {
        query[i] = projected_q[i];
    }
    const auto context = run_attention(query, kv_cache, layer_idx, position, attn_dim);

    // O projection: context → hidden dim
    std::vector<float> o_input(attn_dim, 0.0f);
    for (std::size_t i = 0; i < std::min(attn_dim, context.size()); ++i) {
        o_input[i] = context[i];
    }
    const auto projected_o = attention_projection_from_model_dir(artifact, layer_idx, "o", o_input, hidden);

    // Post-attention residual: hidden += attn_out
    for (std::size_t i = 0; i < std::min(hidden, projected_o.size()); ++i) {
        state.hidden[i] = finite_or_zero(state.hidden[i] + projected_o[i]);
    }

    // Pre-MoE norm → state.moe_input (post_attention_layernorm on residual hidden).
    // Always normalize: apply_rms_norm uses w=1.0 fallback when norm_weights is empty.
    state.moe_input = state.hidden;
    const auto post_norm = lookup_norm_from_model_dir(artifact, layer_idx, true);
    apply_rms_norm(state.moe_input, post_norm);
}

DenseOperatorMetrics dense_prefill_step(DenseOperatorState& state,
                                        KvCacheState& kv_cache,
                                        const DenseRuntimeArtifact* artifact,
                                        const ModelSpec& spec,
                                        std::uint32_t token_id,
                                        std::size_t position) {
    return run_dense_pass(state, kv_cache, artifact, spec, token_id, position, 1.15f);
}

DenseOperatorMetrics dense_decode_step(DenseOperatorState& state,
                                       KvCacheState& kv_cache,
                                       const DenseRuntimeArtifact* artifact,
                                       const ModelSpec& spec,
                                       std::uint32_t token_id,
                                       std::size_t position) {
    return run_dense_pass(state, kv_cache, artifact, spec, token_id, position, 1.0f);
}

void normalize_dense_hidden(DenseOperatorState& state) {
    normalize(state.hidden);
}

void refresh_logits_from_hidden(DenseOperatorState& state, const DenseRuntimeArtifact* artifact) {
    if (artifact != nullptr && artifact->uses_direct_model()) {
        auto normalized = state.hidden;
        const auto final_norm = final_norm_from_model_dir(*artifact);
        if (!final_norm.empty()) {
            apply_rms_norm(normalized, final_norm);
        }
        state.logits = lm_head_logits_from_model_dir(*artifact, normalized);
        return;
    }
    if (artifact != nullptr && artifact->has_lm_head()) {
        const auto dims = artifact->sampled_dims();
        state.logits.assign(artifact->runtime_vocab_size, 0.0f);
        for (std::size_t token = 0; token < artifact->runtime_vocab_size; ++token) {
            float accum = 0.0f;
            for (std::size_t d = 0; d < dims; ++d) {
                const auto hidden_idx = artifact->sampled_indices[d];
                if (hidden_idx < state.hidden.size()) {
                    accum += state.hidden[hidden_idx] * artifact->lm_head_weights[token * dims + d];
                }
            }
            state.logits[token] = accum;
        }
        return;
    }

    state.logits.assign(kRuntimeVocabSize, 0.0f);
    for (std::size_t token = 0; token < kRuntimeVocabSize; ++token) {
        float accum = 0.0f;
        for (std::size_t i = 0; i < state.hidden.size(); i += 32U) {
            accum += state.hidden[i] * seeded_weight(token + 101U, i + 41U);
        }
        state.logits[token] = accum;
    }
}

RouterSelection router_selection_from_hidden(const DenseOperatorState& state,
                                             const DenseRuntimeArtifact* artifact,
                                             const ModelSpec& spec,
                                             std::uint16_t layer) {
    std::vector<std::pair<float, std::uint16_t>> scored;
    scored.reserve(spec.routed_experts_per_layer);
    if (artifact != nullptr && artifact->uses_direct_model() && layer < artifact->num_layers) {
        // Use moe_input (post-attn-norm'd hidden) for routing, as the Qwen3 router gate
        // operates on the normalized residual. Fall back to hidden if moe_input is empty.
        const auto& router_input = (!state.moe_input.empty()) ? state.moe_input : state.hidden;
        const auto full_scores = router_scores_from_model_dir(*artifact, layer, router_input);
        const auto limit = std::min<std::size_t>(full_scores.size(), spec.routed_experts_per_layer);
        for (std::size_t expert = 0; expert < limit; ++expert) {
            scored.emplace_back(full_scores[expert], static_cast<std::uint16_t>(expert));
        }
    } else if (artifact != nullptr && artifact->has_router() && layer < artifact->num_layers) {
        const auto dims = artifact->sampled_dims();
        const std::size_t layer_offset = static_cast<std::size_t>(layer) * artifact->routed_experts_per_layer * dims;
        for (std::uint16_t expert = 0; expert < spec.routed_experts_per_layer; ++expert) {
            float score = 0.0f;
            const std::size_t row = layer_offset + static_cast<std::size_t>(expert) * dims;
            for (std::size_t d = 0; d < dims; ++d) {
                const auto hidden_idx = artifact->sampled_indices[d];
                if (hidden_idx < state.hidden.size()) {
                    score += state.hidden[hidden_idx] * artifact->router_weights[row + d];
                }
            }
            scored.emplace_back(score, expert);
        }
    } else {
        for (std::uint16_t expert = 0; expert < spec.routed_experts_per_layer; ++expert) {
            float score = 0.0f;
            for (std::size_t i = 0; i < state.hidden.size(); i += 64U) {
                score += state.hidden[i] * seeded_weight(layer + 1009U, expert * 17U + i + 1U);
            }
            scored.emplace_back(score, expert);
        }
    }
    const std::size_t keep = std::min<std::size_t>(spec.top_k, scored.size());
    std::partial_sort(
        scored.begin(), scored.begin() + keep, scored.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

    RouterSelection selection;
    selection.experts.reserve(keep);
    selection.weights.reserve(keep);
    double max_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < keep; ++i) {
        max_score = std::max<double>(max_score, scored[i].first);
    }
    std::vector<double> exp_scores(keep, 0.0);
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < keep; ++i) {
        const double e = std::exp(static_cast<double>(scored[i].first) - max_score);
        exp_scores[i] = e;
        sum_exp += e;
    }
    const double inv_sum = sum_exp > 0.0 ? 1.0 / sum_exp : 1.0 / std::max<std::size_t>(1, keep);
    for (std::size_t i = 0; i < keep; ++i) {
        selection.experts.push_back(scored[i].second);
        selection.weights.push_back(static_cast<float>(exp_scores[i] * inv_sum));
    }
    return selection;
}

std::vector<std::uint16_t> router_topk_from_hidden(const DenseOperatorState& state,
                                                   const DenseRuntimeArtifact* artifact,
                                                   const ModelSpec& spec,
                                                   std::uint16_t layer) {
    return router_selection_from_hidden(state, artifact, spec, layer).experts;
}

std::uint32_t sample_token_from_logits(const DenseOperatorState& state) {
    const auto it = std::max_element(state.logits.begin(), state.logits.end());
    return it == state.logits.end() ? 0U : static_cast<std::uint32_t>(std::distance(state.logits.begin(), it));
}

}  // namespace flashmoe
