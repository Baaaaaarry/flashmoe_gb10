#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace flashmoe {

struct ModelSpec;

struct DenseRuntimeArtifact {
    std::string family;
    std::string model_dir;
    bool direct_model_mode = false;
    std::size_t hidden_size = 0;
    std::vector<std::uint32_t> sampled_indices;
    std::size_t runtime_vocab_size = 0;
    std::size_t num_layers = 0;
    std::size_t routed_experts_per_layer = 0;
    std::vector<float> embedding_weights;
    std::vector<float> q_proj_weights;
    std::vector<float> k_proj_weights;
    std::vector<float> v_proj_weights;
    std::vector<float> o_proj_weights;
    std::vector<float> input_norm_weights;
    std::vector<float> post_norm_weights;
    std::vector<float> router_weights;
    std::vector<float> lm_head_weights;

    [[nodiscard]] std::size_t sampled_dims() const noexcept { return sampled_indices.size(); }
    [[nodiscard]] bool has_embeddings() const noexcept { return !embedding_weights.empty(); }
    [[nodiscard]] bool has_attention() const noexcept {
        return !q_proj_weights.empty() && !k_proj_weights.empty() && !v_proj_weights.empty() && !o_proj_weights.empty();
    }
    [[nodiscard]] bool has_norms() const noexcept { return !input_norm_weights.empty() && !post_norm_weights.empty(); }
    [[nodiscard]] bool has_router() const noexcept { return !router_weights.empty(); }
    [[nodiscard]] bool has_lm_head() const noexcept { return !lm_head_weights.empty(); }
    [[nodiscard]] bool uses_direct_model() const noexcept { return direct_model_mode && !model_dir.empty(); }
};

// Release raw shard bytes kept in the blob cache.  Call this once all needed
// tensors are warm in the BF16 tensor_cache to reclaim ~35 GB of host memory.
void flush_model_dir_blob_cache(const std::string& model_dir);

std::optional<DenseRuntimeArtifact> load_dense_runtime_artifact(const std::string& manifest_json_path);
std::optional<DenseRuntimeArtifact> load_dense_runtime_artifact_from_model_dir(
    const std::string& model_dir,
    const ModelSpec& spec,
    std::size_t sample_stride = 1,
    std::size_t max_sampled_dims = 4096,
    std::size_t runtime_vocab_size = 32768);

std::vector<float> lookup_embedding_from_model_dir(const DenseRuntimeArtifact& artifact, std::uint32_t token_id);
std::vector<float> lookup_norm_from_model_dir(const DenseRuntimeArtifact& artifact, std::size_t layer, bool post_attention);
std::vector<float> router_scores_from_model_dir(const DenseRuntimeArtifact& artifact, std::size_t layer, const std::vector<float>& hidden);
std::vector<float> lm_head_logits_from_model_dir(const DenseRuntimeArtifact& artifact, const std::vector<float>& hidden);
std::vector<float> final_norm_from_model_dir(const DenseRuntimeArtifact& artifact);
std::vector<float> attention_projection_from_model_dir(
    const DenseRuntimeArtifact& artifact,
    std::size_t layer,
    std::string_view projection,
    const std::vector<float>& input,
    std::size_t target_dim);
void apply_shared_expert_from_model_dir(
    const DenseRuntimeArtifact& artifact,
    std::size_t layer,
    const std::vector<float>& moe_input,
    std::vector<float>& hidden);

}  // namespace flashmoe
