#include "flashmoe/kv_cache.h"

namespace flashmoe {
namespace {

std::size_t kv_width_for_spec(const ModelSpec& spec) {
    if (spec.has_mla_kv_compression) {
        return 96;
    }
    return std::min<std::size_t>(256, std::max<std::size_t>(64, spec.hidden_size / 16));
}

}  // namespace

KvCacheState make_kv_cache_state(const ModelSpec& spec, const RuntimePlan& plan) {
    KvCacheState state;
    state.budget_gb = plan.kv_cache_budget_gb;
    state.kv_width = kv_width_for_spec(spec);
    state.num_layers = std::max<std::size_t>(1, spec.num_layers);
    state.bytes_per_token = static_cast<double>(state.num_layers) * static_cast<double>(state.kv_width) * 2.0 * sizeof(float);
    return state;
}

bool kv_cache_append(KvCacheState& state, std::size_t token_count) {
    const double next_bytes = state.bytes_per_token * static_cast<double>(state.tokens + token_count);
    if (next_bytes > state.budget_gb * 1e9) {
        return false;
    }
    state.tokens += token_count;
    state.keys.resize(state.tokens * state.num_layers * state.kv_width, 0.0f);
    state.values.resize(state.tokens * state.num_layers * state.kv_width, 0.0f);
    state.used_gb = next_bytes / 1e9;
    return true;
}

float* kv_cache_key_ptr(KvCacheState& state, std::size_t token_index, std::size_t layer) {
    return state.keys.data() + (token_index * state.num_layers + layer) * state.kv_width;
}

float* kv_cache_value_ptr(KvCacheState& state, std::size_t token_index, std::size_t layer) {
    return state.values.data() + (token_index * state.num_layers + layer) * state.kv_width;
}

const float* kv_cache_key_ptr(const KvCacheState& state, std::size_t token_index, std::size_t layer) {
    return state.keys.data() + (token_index * state.num_layers + layer) * state.kv_width;
}

const float* kv_cache_value_ptr(const KvCacheState& state, std::size_t token_index, std::size_t layer) {
    return state.values.data() + (token_index * state.num_layers + layer) * state.kv_width;
}

}  // namespace flashmoe
