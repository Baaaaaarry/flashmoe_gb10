#include "flashmoe/model_spec.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace flashmoe {
namespace {

constexpr ModelSpec kQwen35Spec{
    .family = ModelFamily::kQwen35Moe,
    .short_name = "qwen3.5-397b-a17b",
    .source_precision = "fp8",
    .num_layers = 60,
    .routed_experts_per_layer = 512,
    .shared_experts_per_layer = 1,
    .top_k = 4,
    .hidden_size = 4096,
    .expert_intermediate_size = 1536,
    .source_weight_gb = 152.0,
    .routed_expert_share = 0.93,
    .has_mla_kv_compression = false,
    .supports_runtime_fp8_compute = true,
    .supports_hash_prefill_router = false,
};

constexpr ModelSpec kDeepSeekV4FlashSpec{
    .family = ModelFamily::kDeepSeekV4Flash,
    .short_name = "deepseek-v4-flash",
    .source_precision = "fp4+fp8",
    .num_layers = 43,
    .routed_experts_per_layer = 256,
    .shared_experts_per_layer = 1,
    .top_k = 6,
    .hidden_size = 4096,
    .expert_intermediate_size = 2048,
    .source_weight_gb = 152.1,
    .routed_expert_share = 0.94,
    .has_mla_kv_compression = true,
    .supports_runtime_fp8_compute = true,
    .supports_hash_prefill_router = true,
};

std::string normalize(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return out;
}

}  // namespace

const ModelSpec& builtin_model_spec(ModelFamily family) {
    switch (family) {
    case ModelFamily::kQwen35Moe:
        return kQwen35Spec;
    case ModelFamily::kDeepSeekV4Flash:
        return kDeepSeekV4FlashSpec;
    case ModelFamily::kUnknown:
        break;
    }
    throw std::invalid_argument("unknown model family");
}

ModelFamily parse_model_family(std::string_view value) {
    const std::string key = normalize(value);
    if (key == "qwen35" || key == "qwen35397ba17b" || key == "qwen35moe" || key == "qwen397b") {
        return ModelFamily::kQwen35Moe;
    }
    if (key == "deepseekv4flash" || key == "dsv4flash" || key == "deepseekflash") {
        return ModelFamily::kDeepSeekV4Flash;
    }
    return ModelFamily::kUnknown;
}

std::string_view to_string(ModelFamily family) {
    switch (family) {
    case ModelFamily::kQwen35Moe:
        return "qwen3.5-moe";
    case ModelFamily::kDeepSeekV4Flash:
        return "deepseek-v4-flash";
    case ModelFamily::kUnknown:
        return "unknown";
    }
    return "unknown";
}

std::string_view to_string(ExpertStorageFormat format) {
    switch (format) {
    case ExpertStorageFormat::kDense:
        return "dense";
    case ExpertStorageFormat::kQ3Like:
        return "q3like";
    case ExpertStorageFormat::kMxfp4:
        return "mxfp4";
    case ExpertStorageFormat::kIq2xxs:
        return "iq2xxs";
    }
    return "dense";
}

std::string_view to_string(RuntimeComputeFormat format) {
    switch (format) {
    case RuntimeComputeFormat::kFp16:
        return "fp16";
    case RuntimeComputeFormat::kBf16:
        return "bf16";
    case RuntimeComputeFormat::kFp8:
        return "fp8";
    case RuntimeComputeFormat::kMxfp4:
        return "mxfp4";
    }
    return "fp16";
}

std::vector<ModelCompatibilityNote> compatibility_notes(const ModelSpec& spec) {
    std::vector<ModelCompatibilityNote> notes;
    notes.push_back({
        .component = "router",
        .guidance = spec.supports_hash_prefill_router
            ? "Preserve both hash-router prefill path and learned-router decode path."
            : "Single learned-router path is sufficient; no hash-router prefill split required.",
    });
    notes.push_back({
        .component = "kv-cache",
        .guidance = spec.has_mla_kv_compression
            ? "Keep MLA/CSA/HCA logic resident; KV cache is not the primary bottleneck."
            : "KV cache is conventional; optimize expert residency before KV cache policy.",
    });
    notes.push_back({
        .component = "compute-format",
        .guidance = spec.supports_runtime_fp8_compute
            ? "Prefer FP8 hot-cache compute to avoid per-step dense BF16 restaging."
            : "Use BF16/FP16 hot-cache compute until native FP8 kernels are available.",
    });
    return notes;
}

}  // namespace flashmoe
