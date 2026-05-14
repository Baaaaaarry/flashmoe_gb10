#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace flashmoe {

enum class ModelFamily {
    kUnknown = 0,
    kQwen35Moe,
    kDeepSeekV4Flash,
};

enum class ExpertStorageFormat {
    kDense = 0,
    kQ3Like,
    kMxfp4,
    kIq2xxs,
};

enum class RuntimeComputeFormat {
    kFp16 = 0,
    kBf16,
    kFp8,
    kMxfp4,
};

struct ModelSpec {
    ModelFamily family = ModelFamily::kUnknown;
    std::string_view short_name;
    std::string_view source_precision;
    std::uint16_t num_layers = 0;
    std::uint16_t routed_experts_per_layer = 0;
    std::uint16_t shared_experts_per_layer = 0;
    std::uint16_t top_k = 0;
    std::uint32_t hidden_size = 0;
    std::uint32_t expert_intermediate_size = 0;
    double source_weight_gb = 0.0;
    double routed_expert_share = 0.0;
    bool has_mla_kv_compression = false;
    bool supports_runtime_fp8_compute = false;
    bool supports_hash_prefill_router = false;
};

struct ModelCompatibilityNote {
    std::string_view component;
    std::string_view guidance;
};

const ModelSpec& builtin_model_spec(ModelFamily family);
ModelFamily parse_model_family(std::string_view value);
std::string_view to_string(ModelFamily family);
std::string_view to_string(ExpertStorageFormat format);
std::string_view to_string(RuntimeComputeFormat format);
std::vector<ModelCompatibilityNote> compatibility_notes(const ModelSpec& spec);

}  // namespace flashmoe
