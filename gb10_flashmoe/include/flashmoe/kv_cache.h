#pragma once

#include <cstddef>
#include <vector>

#include "flashmoe/model_spec.h"
#include "flashmoe/runtime_plan.h"

namespace flashmoe {

struct KvCacheState {
    double budget_gb = 0.0;
    double bytes_per_token = 0.0;
    std::size_t kv_width = 0;
    std::size_t num_layers = 0;
    std::size_t tokens = 0;
    double used_gb = 0.0;
    std::vector<float> keys;
    std::vector<float> values;
};

KvCacheState make_kv_cache_state(const ModelSpec& spec, const RuntimePlan& plan);
bool kv_cache_append(KvCacheState& state, std::size_t token_count);
float* kv_cache_key_ptr(KvCacheState& state, std::size_t token_index, std::size_t layer);
float* kv_cache_value_ptr(KvCacheState& state, std::size_t token_index, std::size_t layer);
const float* kv_cache_key_ptr(const KvCacheState& state, std::size_t token_index, std::size_t layer);
const float* kv_cache_value_ptr(const KvCacheState& state, std::size_t token_index, std::size_t layer);

}  // namespace flashmoe
