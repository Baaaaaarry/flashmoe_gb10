#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "flashmoe/compute_profile.h"
#include "flashmoe/decode_state.h"
#include "flashmoe/dense_artifact.h"
#include "flashmoe/dense_decode.h"
#include "flashmoe/dense_operator_chain.h"
#include "flashmoe/dense_resident_loader.h"
#include "flashmoe/decode_harness.h"
#include "flashmoe/device_path.h"
#include "flashmoe/expert_store.h"
#include "flashmoe/expert_operator_chain.h"
#include "flashmoe/kv_cache.h"
#include "flashmoe/model_spec.h"
#include "flashmoe/online_runtime.h"
#include "flashmoe/runtime_plan.h"
#include "flashmoe/tokenizer_artifact.h"

namespace flashmoe {

struct SessionStepStats {
    double embed_ms = 0.0;
    double attention_ms = 0.0;
    double norm_router_ms = 0.0;
    double lm_head_ms = 0.0;
    double route_ms = 0.0;
    double load_ms = 0.0;
    double unpack_ms = 0.0;
    double compute_ms = 0.0;
    double combine_ms = 0.0;
    double total_ms = 0.0;
    std::uint64_t promotions = 0;
    std::uint64_t hits = 0;
};

struct VectorStats {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double l2 = 0.0;
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
};

struct SessionGenerateResult {
    std::string text;
    std::string finish_reason = "length";
    std::string route_source = "trace";
    std::size_t prompt_tokens = 0;
    std::size_t prefill_tokens = 0;
    std::size_t completion_tokens = 0;
    std::size_t total_tokens = 0;
    std::size_t session_generated_tokens = 0;
    std::size_t prefill_trace_steps = 0;
    std::size_t trace_steps_consumed = 0;
    double prefill_total_ms = 0.0;
    double total_ms = 0.0;
    double avg_token_ms = 0.0;
    double tok_per_s = 0.0;
    std::vector<std::uint32_t> sampled_token_ids;
    std::uint64_t slot_promotions = 0;
    std::uint64_t slot_hits = 0;
    std::uint64_t slot_evictions = 0;
    double resident_slot_gb = 0.0;
    double kv_cache_used_gb = 0.0;
    std::vector<SessionStepStats> token_stats;
};

struct StreamedEngineConfig {
    ModelFamily family = ModelFamily::kUnknown;
    std::optional<std::string> model_path;
    std::string manifest_path;
    std::string trace_path;
    std::optional<std::string> compute_profile_path;
    std::optional<std::string> dense_artifact_path;
    std::optional<std::string> tokenizer_artifact_path;
    std::optional<double> hot_cache_budget_gb;
    bool use_runtime_router = false;
    bool prefer_cuda_expert_backend = true;
    bool rewind_trace = true;
};

class StreamedEngine;

class StreamedSession {
public:
    StreamedSession(const StreamedEngine& engine, std::string prompt);

    SessionGenerateResult generate(std::size_t max_tokens);
    [[nodiscard]] std::uint64_t session_id() const noexcept { return session_id_; }
    [[nodiscard]] std::size_t generated_tokens() const noexcept { return generated_tokens_; }
    [[nodiscard]] std::size_t prompt_tokens() const noexcept { return prompt_tokens_; }
    [[nodiscard]] const std::string& prompt() const noexcept { return prompt_; }
    [[nodiscard]] std::size_t prefill_tokens() const noexcept { return prefill_tokens_; }
    [[nodiscard]] std::size_t prefill_trace_steps() const noexcept { return prefill_trace_steps_; }
    [[nodiscard]] double prefill_total_ms() const noexcept { return prefill_total_ms_; }
    [[nodiscard]] double kv_cache_used_gb() const noexcept { return kv_cache_.used_gb; }
    [[nodiscard]] double resident_slot_gb() const noexcept { return slot_cache_.bytes_used_gb(); }
    void append_user_turn(std::string text);

private:
    void prefill_tokens(const std::vector<std::uint32_t>& token_ids);

    const StreamedEngine& engine_;
    std::uint64_t session_id_ = 0;
    std::string prompt_;
    std::size_t prompt_tokens_ = 0;
    std::size_t prefill_tokens_ = 0;
    std::size_t prefill_trace_steps_ = 0;
    std::size_t generated_tokens_ = 0;
    std::size_t trace_cursor_ = 0;
    double prefill_total_ms_ = 0.0;
    DecodeState decode_state_;
    DenseOperatorState dense_state_;
    MaterializedExpertMap materialized_experts_;
    KvCacheState kv_cache_;
    ComputeReadySlotCache slot_cache_;
    SlotCacheStats slot_stats_{};
    std::vector<ExpertId> seen_;
};

class StreamedEngine {
public:
    static StreamedEngine load(const StreamedEngineConfig& config);

    StreamedSession create_session(std::string prompt) const;

    [[nodiscard]] const ModelSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] const HardwareProfile& hw() const noexcept { return hw_; }
    [[nodiscard]] const RuntimePlan& plan() const noexcept { return plan_; }
    [[nodiscard]] const DevicePathProfile& profile() const noexcept { return profile_; }
    [[nodiscard]] const ExpertManifestStore& store() const noexcept { return store_; }
    [[nodiscard]] const std::vector<RouteStep>& trace() const noexcept { return trace_; }
    [[nodiscard]] bool use_runtime_router() const noexcept { return use_runtime_router_; }
    [[nodiscard]] bool use_cuda_expert_backend() const noexcept { return use_cuda_expert_backend_; }
    [[nodiscard]] bool rewind_trace() const noexcept { return rewind_trace_; }
    [[nodiscard]] const DenseResidentPlan& dense_resident_plan() const noexcept { return dense_resident_plan_; }
    [[nodiscard]] const DenseRuntimeArtifact* dense_artifact() const noexcept {
        return dense_artifact_.has_value() ? &*dense_artifact_ : nullptr;
    }
    [[nodiscard]] const TokenizerArtifact* tokenizer_artifact() const noexcept {
        return tokenizer_artifact_.has_value() ? &*tokenizer_artifact_ : nullptr;
    }

private:
    friend class StreamedSession;

    StreamedEngine(ModelSpec spec,
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
                   bool rewind_trace);

    ModelSpec spec_;
    HardwareProfile hw_;
    RuntimePlan plan_;
    DevicePathProfile profile_;
    DenseResidentPlan dense_resident_plan_;
    std::optional<DenseRuntimeArtifact> dense_artifact_;
    std::optional<TokenizerArtifact> tokenizer_artifact_;
    ExpertManifestStore store_;
    std::vector<RouteStep> trace_;
    bool use_runtime_router_ = false;
    bool use_cuda_expert_backend_ = false;
    bool rewind_trace_ = true;
};

}  // namespace flashmoe
