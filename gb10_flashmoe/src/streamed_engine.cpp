#include "flashmoe/streamed_engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "flashmoe/expert_cuda_backend.h"
#include "flashmoe/router_runtime.h"

namespace flashmoe {
namespace {

std::vector<ExpertId> release_evicted_materialized(MaterializedExpertMap& materialized,
                                                   const std::vector<ExpertId>& evicted_ids) {
    std::vector<ExpertId> released;
    for (const auto& id : evicted_ids) {
        auto it = materialized.find(id);
        if (it == materialized.end()) {
            continue;
        }
        cuda_release_expert_buffers(it->second);
        release_materialized_expert(it->second);
        materialized.erase(it);
        released.push_back(id);
    }
    return released;
}

struct StepExecutionResult {
    SessionStepStats stats;
    std::size_t trace_steps = 0;
    VectorStats after_dense_hidden;
    VectorStats after_experts_hidden;
    VectorStats after_shared_hidden;
    VectorStats after_norm_hidden;
};

std::uint64_t next_session_id() {
    static std::uint64_t counter = 1;
    return counter++;
}

std::size_t count_tokens_from_ids(const std::vector<std::uint32_t>& token_ids) {
    return std::max<std::size_t>(1, token_ids.size());
}

std::uint64_t make_seen_key(ExpertId id) {
    return (static_cast<std::uint64_t>(id.layer) << 32U) | static_cast<std::uint64_t>(id.expert);
}

bool already_seen(const std::vector<ExpertId>& seen, ExpertId id) {
    const auto key = make_seen_key(id);
    for (const auto prior : seen) {
        if (make_seen_key(prior) == key) {
            return true;
        }
    }
    return false;
}

double token_per_s_from_ms(double ms) {
    return ms > 0.0 ? 1000.0 / ms : 0.0;
}

bool progress_enabled() {
    const char* env = std::getenv("FLASHMOE_PROGRESS");
    if (env == nullptr) {
        return false;
    }
    const std::string value(env);
    return !(value.empty() || value == "0" || value == "false" || value == "FALSE");
}

bool disable_layer_normalize() {
    const char* env = std::getenv("FLASHMOE_DISABLE_LAYER_NORMALIZE");
    if (env == nullptr) {
        return false;
    }
    const std::string value(env);
    return !(value.empty() || value == "0" || value == "false" || value == "FALSE");
}

std::size_t progress_every() {
    const char* env = std::getenv("FLASHMOE_PROGRESS_EVERY");
    if (env == nullptr) {
        return 1;
    }
    try {
        return std::max<std::size_t>(1, static_cast<std::size_t>(std::stoull(env)));
    } catch (...) {
        return 1;
    }
}

VectorStats summarize_vector(const std::vector<float>& values) {
    VectorStats s;
    if (values.empty()) {
        return s;
    }
    s.min = std::numeric_limits<double>::infinity();
    s.max = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const auto v : values) {
        if (std::isnan(v)) {
            s.nan_count += 1;
            continue;
        }
        if (!std::isfinite(v)) {
            s.inf_count += 1;
            continue;
        }
        s.min = std::min<double>(s.min, v);
        s.max = std::max<double>(s.max, v);
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    const auto finite = values.size() - s.nan_count - s.inf_count;
    if (finite == 0) {
        s.min = 0.0;
        s.max = 0.0;
        return s;
    }
    s.mean = sum / static_cast<double>(finite);
    s.l2 = std::sqrt(sum_sq);
    return s;
}

std::string top_logits_summary(const std::vector<float>& logits, std::size_t k = 5) {
    if (logits.empty()) {
        return "[]";
    }
    std::vector<std::pair<float, std::size_t>> scored;
    scored.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
        scored.emplace_back(logits[i], i);
    }
    const auto keep = std::min(k, scored.size());
    std::partial_sort(
        scored.begin(),
        scored.begin() + keep,
        scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < keep; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << scored[i].second << ":" << scored[i].first;
    }
    out << "]";
    return out.str();
}

void print_progress_line(const char* phase,
                         std::size_t index,
                         std::size_t total,
                         std::size_t position,
                         const SessionStepStats& stats,
                         std::size_t trace_steps) {
    std::cerr << "[" << phase << "] "
              << index << "/" << total
              << " position=" << position
              << " trace_steps=" << trace_steps
              << " total_ms=" << stats.total_ms
              << " embed_ms=" << stats.embed_ms
              << " attention_ms=" << stats.attention_ms
              << " norm_router_ms=" << stats.norm_router_ms
              << " lm_head_ms=" << stats.lm_head_ms
              << " route_ms=" << stats.route_ms
              << " load_ms=" << stats.load_ms
              << " unpack_ms=" << stats.unpack_ms
              << " compute_ms=" << stats.compute_ms
              << " combine_ms=" << stats.combine_ms
              << " promotions=" << stats.promotions
              << " hits=" << stats.hits
              << '\n';
}

// Execute one transformer layer: attention (direct_model_mode only) + routed experts + shared expert.
// For direct_model_mode, run_layer_attention_direct sets state.moe_input before experts run.
// Experts read from state.moe_input and accumulate into state.hidden.
StepExecutionResult execute_step(const StreamedEngine& engine,
                                 std::size_t& trace_cursor,
                                 ComputeReadySlotCache& slot_cache,
                                 SlotCacheStats& slot_stats,
                                 std::vector<ExpertId>& seen,
                                 DenseOperatorState& mutable_dense_state,
                                 MaterializedExpertMap& materialized_experts,
                                 KvCacheState& kv_cache,
                                 std::size_t layer_idx,
                                 std::size_t position,
                                 const DenseOperatorMetrics& dense_metrics) {
    StepExecutionResult result;
    result.stats.embed_ms = dense_metrics.embed_ms;
    result.stats.attention_ms = dense_metrics.attention_ms;
    result.stats.norm_router_ms = dense_metrics.norm_router_ms;
    result.stats.lm_head_ms = dense_metrics.lm_head_ms;

    // Per-layer attention for direct model mode.
    if (engine.dense_artifact() != nullptr && engine.dense_artifact()->uses_direct_model()) {
        const auto attn_t0 = std::chrono::steady_clock::now();
        run_layer_attention_direct(mutable_dense_state, kv_cache, *engine.dense_artifact(), layer_idx, position);
        const auto attn_t1 = std::chrono::steady_clock::now();
        result.stats.attention_ms += std::chrono::duration<double, std::milli>(attn_t1 - attn_t0).count();
    }

    result.after_dense_hidden = summarize_vector(mutable_dense_state.hidden);

    RuntimeRouteStep runtime_route;
    const RouteStep* route_ptr = nullptr;
    if (engine.use_runtime_router()) {
        runtime_route = runtime_route_for_layer(
            engine.spec(), mutable_dense_state, static_cast<std::uint16_t>(layer_idx), engine.dense_artifact());
    } else {
        if (trace_cursor >= engine.trace().size()) {
            if (engine.rewind_trace()) {
                trace_cursor = 0;
            } else {
                return result;
            }
        }
        route_ptr = &engine.trace()[trace_cursor++];
    }

    result.trace_steps += 1;
    result.stats.route_ms += engine.profile().route_ms;
    const auto route_layer = route_ptr != nullptr ? route_ptr->layer : runtime_route.layer;
    const auto& route_experts = route_ptr != nullptr ? route_ptr->experts : runtime_route.experts;
    bool layer_touched = false;
    for (std::size_t route_index = 0; route_index < route_experts.size(); ++route_index) {
        const auto expert = route_experts[route_index];
        const ExpertId id{route_layer, expert};
        const auto* record = engine.store().find(id);
        if (record == nullptr) {
            continue;
        }
        layer_touched = true;
        if (!already_seen(seen, id)) {
            seen.push_back(id);
        }
        if (slot_cache.contains(id)) {
            SlotCacheStats local = slot_stats;
            std::vector<ExpertId> evicted_ids;
            slot_cache.touch_or_promote(*record, local, &evicted_ids);
            result.stats.hits += 1;
            slot_stats = local;
            release_evicted_materialized(materialized_experts, evicted_ids);
        } else {
            result.stats.promotions += 1;
            result.stats.load_ms += materialize_expert_record(*record, materialized_experts);
            std::vector<ExpertId> evicted_ids;
            slot_cache.touch_or_promote(*record, slot_stats, &evicted_ids);
            release_evicted_materialized(materialized_experts, evicted_ids);
        }
        const float route_weight = route_ptr != nullptr
            ? (route_experts.empty() ? 1.0f : 1.0f / static_cast<float>(route_experts.size()))
            : (route_index < runtime_route.weights.size() ? runtime_route.weights[route_index] : 1.0f);
        if (engine.use_cuda_expert_backend()) {
            auto& slot = materialized_experts[id];
            if (!slot.host_ready) {
                result.stats.unpack_ms += unpack_materialized_expert(*record, materialized_experts);
            }
            cuda_unpack_and_execute_expert(
                *record,
                mutable_dense_state,
                engine.dense_artifact(),
                materialized_experts,
                &result.stats.unpack_ms,
                &result.stats.compute_ms,
                &result.stats.combine_ms,
                route_weight);
        } else {
            result.stats.unpack_ms += unpack_materialized_expert(*record, materialized_experts);
            result.stats.compute_ms += execute_materialized_expert_fused(
                *record, mutable_dense_state, engine.dense_artifact(), materialized_experts, route_weight);
        }
    }
    if (layer_touched) {
        result.after_experts_hidden = summarize_vector(mutable_dense_state.hidden);
        if (engine.dense_artifact() != nullptr && engine.dense_artifact()->uses_direct_model()) {
            apply_shared_expert_from_model_dir(
                *engine.dense_artifact(), layer_idx,
                mutable_dense_state.moe_input, mutable_dense_state.hidden);
        }
        result.after_shared_hidden = summarize_vector(mutable_dense_state.hidden);
        // No global normalize: residual connections are maintained explicitly per layer.
        result.after_norm_hidden = result.after_shared_hidden;
    } else if (!engine.dense_artifact() || !engine.dense_artifact()->uses_direct_model()) {
        // Non-direct mode: keep old normalize behavior.
        if (!disable_layer_normalize()) {
            normalize_dense_hidden(mutable_dense_state);
        }
        result.after_norm_hidden = summarize_vector(mutable_dense_state.hidden);
    }

    result.stats.total_ms = result.stats.embed_ms + result.stats.attention_ms + result.stats.norm_router_ms
        + result.stats.lm_head_ms + result.stats.route_ms + result.stats.load_ms + result.stats.unpack_ms
        + result.stats.compute_ms + result.stats.combine_ms;
    return result;
}

DecodeOpTrace to_decode_trace(const SessionStepStats& stats) {
    return DecodeOpTrace{
        .embed_ms = stats.embed_ms,
        .attention_ms = stats.attention_ms,
        .norm_router_ms = stats.norm_router_ms,
        .lm_head_ms = stats.lm_head_ms,
        .route_ms = stats.route_ms,
        .load_ms = stats.load_ms,
        .unpack_ms = stats.unpack_ms,
        .compute_ms = stats.compute_ms,
        .combine_ms = stats.combine_ms,
        .promotions = stats.promotions,
        .hits = stats.hits,
    };
}

}  // namespace

StreamedEngine::StreamedEngine(ModelSpec spec,
                   HardwareProfile hw,
                   RuntimePlan plan,
                   DevicePathProfile profile,
                   DenseResidentPlan dense_resident_plan,
                   std::optional<DenseRuntimeArtifact> dense_artifact,
                   std::optional<TokenizerArtifact> tokenizer_artifact,
                   ExpertManifestStore store,
                               std::vector<RouteStep> trace,
                               bool use_runtime_router,
                               bool use_cuda_expert_backend,
                               bool rewind_trace)
    : spec_(spec),
      hw_(hw),
      plan_(std::move(plan)),
      profile_(std::move(profile)),
      dense_resident_plan_(std::move(dense_resident_plan)),
      dense_artifact_(std::move(dense_artifact)),
      tokenizer_artifact_(std::move(tokenizer_artifact)),
      store_(std::move(store)),
      trace_(std::move(trace)),
      use_runtime_router_(use_runtime_router),
      use_cuda_expert_backend_(use_cuda_expert_backend),
      rewind_trace_(rewind_trace) {}

StreamedEngine StreamedEngine::load(const StreamedEngineConfig& config) {
    if (config.family == ModelFamily::kUnknown) {
        throw std::runtime_error("unknown model family");
    }
    const auto& spec = builtin_model_spec(config.family);
    const auto hw = gb10_hardware_profile();
    auto plan = recommend_runtime_plan(spec, hw);
    plan.use_gpu_side_unpack = true;
    if (config.hot_cache_budget_gb.has_value()) {
        plan.hot_expert_cache_budget_gb = *config.hot_cache_budget_gb;
        plan.total_runtime_budget_gb = plan.dense_resident_budget_gb
            + plan.hot_expert_cache_budget_gb
            + plan.kv_cache_budget_gb
            + plan.workspace_budget_gb
            + plan.safety_margin_gb;
    }

    auto profile = gpu_unpack_profile(spec, hw, plan);
    if (config.compute_profile_path.has_value()) {
        profile.measured_compute = ComputeProfile::from_json_file(*config.compute_profile_path);
    }

    auto trace = std::vector<RouteStep>{};
    if (!config.use_runtime_router) {
        trace = load_route_trace_file(config.trace_path);
    }

    auto dense_artifact = std::optional<DenseRuntimeArtifact>{};
    if (config.dense_artifact_path.has_value()) {
        dense_artifact = load_dense_runtime_artifact(*config.dense_artifact_path);
    } else if (config.model_path.has_value()) {
        dense_artifact = load_dense_runtime_artifact_from_model_dir(*config.model_path, spec);
    }
    auto tokenizer_artifact = std::optional<TokenizerArtifact>{};
    if (config.tokenizer_artifact_path.has_value()) {
        tokenizer_artifact = load_tokenizer_artifact(*config.tokenizer_artifact_path);
    } else if (config.model_path.has_value()) {
        tokenizer_artifact = load_tokenizer_artifact_from_model_dir(*config.model_path);
    }

    return StreamedEngine(
        spec,
        hw,
        plan,
        profile,
        build_dense_resident_plan(spec, plan),
        std::move(dense_artifact),
        std::move(tokenizer_artifact),
        ExpertManifestStore::from_json_file(config.manifest_path),
        std::move(trace),
        config.use_runtime_router,
        config.prefer_cuda_expert_backend && cuda_expert_backend_available(),
        config.rewind_trace);
}

StreamedSession StreamedEngine::create_session(std::string prompt) const {
    return StreamedSession(*this, std::move(prompt));
}

StreamedSession::StreamedSession(const StreamedEngine& engine, std::string prompt)
    : engine_(engine),
      session_id_(next_session_id()),
      prompt_(std::move(prompt)),
      prompt_tokens_(0),
      decode_state_(make_decode_state(prompt_)),
      dense_state_(make_dense_operator_state(engine.spec())),
      kv_cache_(make_kv_cache_state(engine.spec(), engine.plan())),
      slot_cache_(engine.plan().hot_expert_cache_budget_gb, engine.plan().hot_cache_compute_format) {
    auto prompt_token_ids = encode_prompt_runtime_vocab(prompt_, engine.tokenizer_artifact());
    prompt_tokens_ = count_tokens_from_ids(prompt_token_ids);
    prefill_tokens(prompt_token_ids);
}

void StreamedSession::append_user_turn(std::string text) {
    auto token_ids = encode_prompt_runtime_vocab(text, engine_.tokenizer_artifact());
    const std::size_t added_tokens = count_tokens_from_ids(token_ids);
    if (!prompt_.empty()) {
        prompt_.append("\n");
    }
    prompt_.append(std::move(text));
    prompt_tokens_ += added_tokens;
    prefill_tokens(token_ids);
}

void StreamedSession::prefill_tokens(const std::vector<std::uint32_t>& token_ids) {
    const bool show_progress = progress_enabled();
    const std::size_t every = progress_every();
    const bool use_direct = engine_.dense_artifact() != nullptr && engine_.dense_artifact()->uses_direct_model();
    const std::size_t num_layers = std::max<std::size_t>(1, engine_.spec().num_layers);

    for (std::size_t token_idx = 0; token_idx < token_ids.size(); ++token_idx) {
        if (!kv_cache_append(kv_cache_, 1)) {
            break;
        }
        const auto position_before = decode_state_.position;
        const auto prompt_token = token_ids[token_idx];

        StepExecutionResult combined_step;
        if (use_direct) {
            // Proper transformer: embed once, then run all layers sequentially.
            const auto embed_t0 = std::chrono::steady_clock::now();
            embed_token_direct(dense_state_, *engine_.dense_artifact(), prompt_token);
            const auto embed_t1 = std::chrono::steady_clock::now();
            combined_step.stats.embed_ms = std::chrono::duration<double, std::milli>(embed_t1 - embed_t0).count();
            for (std::size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
                const auto layer_step = execute_step(
                    engine_, trace_cursor_, slot_cache_, slot_stats_, seen_,
                    dense_state_, materialized_experts_,
                    kv_cache_, layer_idx, decode_state_.position,
                    DenseOperatorMetrics{});
                combined_step.trace_steps += layer_step.trace_steps;
                combined_step.stats.attention_ms += layer_step.stats.attention_ms;
                combined_step.stats.route_ms += layer_step.stats.route_ms;
                combined_step.stats.load_ms += layer_step.stats.load_ms;
                combined_step.stats.unpack_ms += layer_step.stats.unpack_ms;
                combined_step.stats.compute_ms += layer_step.stats.compute_ms;
                combined_step.stats.combine_ms += layer_step.stats.combine_ms;
                combined_step.stats.promotions += layer_step.stats.promotions;
                combined_step.stats.hits += layer_step.stats.hits;
                combined_step.after_experts_hidden = layer_step.after_experts_hidden;
                combined_step.after_shared_hidden = layer_step.after_shared_hidden;
                combined_step.after_norm_hidden = layer_step.after_norm_hidden;
            }
            refresh_logits_from_hidden(dense_state_, engine_.dense_artifact());
            // After all 60 layers are warm for the first token, the raw shard bytes in
            // blob_cache are no longer needed — tensor_cache holds BF16 copies.  Free
            // them to recover ~35 GB of host memory before subsequent tokens run.
            if (token_idx == 0) {
                flush_model_dir_blob_cache(engine_.dense_artifact()->model_dir);
            }
        } else {
            const auto dense_metrics = dense_prefill_step(
                dense_state_, kv_cache_, engine_.dense_artifact(), engine_.spec(), prompt_token, decode_state_.position);
            for (std::size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
                const auto layer_step = execute_step(
                    engine_, trace_cursor_, slot_cache_, slot_stats_, seen_,
                    dense_state_, materialized_experts_,
                    kv_cache_, layer_idx, decode_state_.position,
                    layer_idx == 0 ? dense_metrics : DenseOperatorMetrics{});
                combined_step.trace_steps += layer_step.trace_steps;
                combined_step.stats.route_ms += layer_step.stats.route_ms;
                combined_step.stats.load_ms += layer_step.stats.load_ms;
                combined_step.stats.unpack_ms += layer_step.stats.unpack_ms;
                combined_step.stats.compute_ms += layer_step.stats.compute_ms;
                combined_step.stats.combine_ms += layer_step.stats.combine_ms;
                combined_step.stats.promotions += layer_step.stats.promotions;
                combined_step.stats.hits += layer_step.stats.hits;
                if (layer_idx == 0) {
                    combined_step.stats.embed_ms = layer_step.stats.embed_ms;
                    combined_step.stats.attention_ms = layer_step.stats.attention_ms;
                    combined_step.stats.norm_router_ms = layer_step.stats.norm_router_ms;
                    combined_step.stats.lm_head_ms = layer_step.stats.lm_head_ms;
                }
            }
        }
        combined_step.stats.total_ms = combined_step.stats.embed_ms + combined_step.stats.attention_ms
            + combined_step.stats.norm_router_ms + combined_step.stats.lm_head_ms
            + combined_step.stats.route_ms + combined_step.stats.load_ms + combined_step.stats.unpack_ms
            + combined_step.stats.compute_ms + combined_step.stats.combine_ms;
        combined_step.after_dense_hidden = summarize_vector(dense_state_.hidden);

        prefill_tokens_ += 1;
        prefill_trace_steps_ += combined_step.trace_steps;
        prefill_total_ms_ += combined_step.stats.total_ms;
        decode_state_.token_ids.push_back(prompt_token);
        apply_prefill_step(decode_state_, to_decode_trace(combined_step.stats), prefill_tokens_);
        if (show_progress && (((token_idx + 1) % every) == 0 || (token_idx + 1) == token_ids.size())) {
            print_progress_line("prefill", token_idx + 1, token_ids.size(), position_before, combined_step.stats, combined_step.trace_steps);
        }
    }
}

SessionGenerateResult StreamedSession::generate(std::size_t max_tokens) {
    SessionGenerateResult result;
    result.prompt_tokens = prompt_tokens_;
    result.prefill_tokens = prefill_tokens_;
    result.prefill_trace_steps = prefill_trace_steps_;
    result.prefill_total_ms = prefill_total_ms_;
    result.route_source = engine_.use_runtime_router() ? "runtime_router" : "trace";
    result.completion_tokens = 0;
    const bool show_progress = progress_enabled();
    const std::size_t every = progress_every();

    if (engine_.trace().empty() && !engine_.use_runtime_router()) {
        throw std::runtime_error("runtime trace is empty");
    }

    const bool use_direct = engine_.dense_artifact() != nullptr && engine_.dense_artifact()->uses_direct_model();
    const std::size_t num_layers = std::max<std::size_t>(1, engine_.spec().num_layers);

    for (std::size_t token_idx = 0; token_idx < max_tokens; ++token_idx) {
        if (!kv_cache_append(kv_cache_, 1)) {
            result.finish_reason = "kv_budget";
            break;
        }
        const auto position_before = decode_state_.position;

        StepExecutionResult step;
        if (use_direct) {
            const auto embed_t0 = std::chrono::steady_clock::now();
            embed_token_direct(dense_state_, *engine_.dense_artifact(), dense_state_.last_token_id);
            const auto embed_t1 = std::chrono::steady_clock::now();
            step.stats.embed_ms = std::chrono::duration<double, std::milli>(embed_t1 - embed_t0).count();
            for (std::size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
                const auto layer_step = execute_step(
                    engine_, trace_cursor_, slot_cache_, slot_stats_, seen_,
                    dense_state_, materialized_experts_,
                    kv_cache_, layer_idx, decode_state_.position,
                    DenseOperatorMetrics{});
                step.trace_steps += layer_step.trace_steps;
                step.stats.attention_ms += layer_step.stats.attention_ms;
                step.stats.route_ms += layer_step.stats.route_ms;
                step.stats.load_ms += layer_step.stats.load_ms;
                step.stats.unpack_ms += layer_step.stats.unpack_ms;
                step.stats.compute_ms += layer_step.stats.compute_ms;
                step.stats.combine_ms += layer_step.stats.combine_ms;
                step.stats.promotions += layer_step.stats.promotions;
                step.stats.hits += layer_step.stats.hits;
                step.after_experts_hidden = layer_step.after_experts_hidden;
                step.after_shared_hidden = layer_step.after_shared_hidden;
                step.after_norm_hidden = layer_step.after_norm_hidden;
            }
            refresh_logits_from_hidden(dense_state_, engine_.dense_artifact());
        } else {
            const auto dense_metrics = dense_decode_step(
                dense_state_, kv_cache_, engine_.dense_artifact(), engine_.spec(), dense_state_.last_token_id, decode_state_.position);
            for (std::size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
                const auto layer_step = execute_step(
                    engine_, trace_cursor_, slot_cache_, slot_stats_, seen_,
                    dense_state_, materialized_experts_,
                    kv_cache_, layer_idx, decode_state_.position,
                    layer_idx == 0 ? dense_metrics : DenseOperatorMetrics{});
                step.trace_steps += layer_step.trace_steps;
                step.stats.route_ms += layer_step.stats.route_ms;
                step.stats.load_ms += layer_step.stats.load_ms;
                step.stats.unpack_ms += layer_step.stats.unpack_ms;
                step.stats.compute_ms += layer_step.stats.compute_ms;
                step.stats.combine_ms += layer_step.stats.combine_ms;
                step.stats.promotions += layer_step.stats.promotions;
                step.stats.hits += layer_step.stats.hits;
                if (layer_idx == 0) {
                    step.stats.embed_ms = layer_step.stats.embed_ms;
                    step.stats.attention_ms = layer_step.stats.attention_ms;
                    step.stats.norm_router_ms = layer_step.stats.norm_router_ms;
                    step.stats.lm_head_ms = layer_step.stats.lm_head_ms;
                }
            }
        }
        step.stats.total_ms = step.stats.embed_ms + step.stats.attention_ms
            + step.stats.norm_router_ms + step.stats.lm_head_ms
            + step.stats.route_ms + step.stats.load_ms + step.stats.unpack_ms
            + step.stats.compute_ms + step.stats.combine_ms;
        step.after_dense_hidden = summarize_vector(dense_state_.hidden);

        result.trace_steps_consumed += step.trace_steps;
        result.total_ms += step.stats.total_ms;
        result.token_stats.push_back(step.stats);

        apply_decode_step(decode_state_, to_decode_trace(step.stats), generated_tokens_);
        const auto token_id = sample_token_from_logits(dense_state_);
        auto text = token_text_from_runtime_vocab(token_id, engine_.tokenizer_artifact());
        const bool artifact_decode = !text.empty();
        if (text.empty()) {
            text = token_text_from_id(token_id);
        }
        const auto sampled = SampledToken{.token_id = token_id, .text = std::move(text)};
        result.sampled_token_ids.push_back(sampled.token_id);
        decode_state_.token_ids.push_back(sampled.token_id);
        dense_state_.last_token_id = sampled.token_id;
        if (!result.text.empty() && !artifact_decode) {
            result.text.push_back(' ');
        }
        result.text += sampled.text;
        generated_tokens_ += 1;
        result.completion_tokens += 1;
        if (show_progress && (((token_idx + 1) % every) == 0 || (token_idx + 1) == max_tokens)) {
            print_progress_line("decode", token_idx + 1, max_tokens, position_before, step.stats, step.trace_steps);
            const auto hidden_summary = summarize_vector(dense_state_.hidden);
            const auto logits_summary = summarize_vector(dense_state_.logits);
            std::cerr << "[decode] token_id=" << sampled.token_id
                      << " text=" << sampled.text
                      << " cumulative_ms=" << result.total_ms
                      << " avg_token_ms="
                      << (result.completion_tokens > 0 ? result.total_ms / static_cast<double>(result.completion_tokens) : 0.0)
                      << " hidden[min=" << hidden_summary.min
                      << ",max=" << hidden_summary.max
                      << ",mean=" << hidden_summary.mean
                      << ",l2=" << hidden_summary.l2
                      << ",nan=" << hidden_summary.nan_count
                      << ",inf=" << hidden_summary.inf_count
                      << "]"
                      << " logits[min=" << logits_summary.min
                      << ",max=" << logits_summary.max
                      << ",mean=" << logits_summary.mean
                      << ",l2=" << logits_summary.l2
                      << ",nan=" << logits_summary.nan_count
                      << ",inf=" << logits_summary.inf_count
                      << "]"
                      << " after_dense[min=" << step.after_dense_hidden.min
                      << ",max=" << step.after_dense_hidden.max
                      << ",mean=" << step.after_dense_hidden.mean
                      << ",l2=" << step.after_dense_hidden.l2
                      << ",nan=" << step.after_dense_hidden.nan_count
                      << ",inf=" << step.after_dense_hidden.inf_count
                      << "]"
                      << " after_experts[min=" << step.after_experts_hidden.min
                      << ",max=" << step.after_experts_hidden.max
                      << ",mean=" << step.after_experts_hidden.mean
                      << ",l2=" << step.after_experts_hidden.l2
                      << ",nan=" << step.after_experts_hidden.nan_count
                      << ",inf=" << step.after_experts_hidden.inf_count
                      << "]"
                      << " after_shared[min=" << step.after_shared_hidden.min
                      << ",max=" << step.after_shared_hidden.max
                      << ",mean=" << step.after_shared_hidden.mean
                      << ",l2=" << step.after_shared_hidden.l2
                      << ",nan=" << step.after_shared_hidden.nan_count
                      << ",inf=" << step.after_shared_hidden.inf_count
                      << "]"
                      << " after_norm[min=" << step.after_norm_hidden.min
                      << ",max=" << step.after_norm_hidden.max
                      << ",mean=" << step.after_norm_hidden.mean
                      << ",l2=" << step.after_norm_hidden.l2
                      << ",nan=" << step.after_norm_hidden.nan_count
                      << ",inf=" << step.after_norm_hidden.inf_count
                      << "]"
                      << " top_logits=" << top_logits_summary(dense_state_.logits)
                      << '\n';
        }
    }

    result.total_tokens = prompt_tokens_ + generated_tokens_;
    result.session_generated_tokens = generated_tokens_;
    result.avg_token_ms = result.completion_tokens > 0
        ? result.total_ms / static_cast<double>(result.completion_tokens)
        : 0.0;
    result.tok_per_s = token_per_s_from_ms(result.avg_token_ms);
    result.slot_promotions = slot_stats_.promotions;
    result.slot_hits = slot_stats_.slot_hits;
    result.slot_evictions = slot_stats_.slot_evictions;
    result.resident_slot_gb = slot_cache_.bytes_used_gb();
    result.kv_cache_used_gb = kv_cache_.used_gb;
    return result;
}

}  // namespace flashmoe
