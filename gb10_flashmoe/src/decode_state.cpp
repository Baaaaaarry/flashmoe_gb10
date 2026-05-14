#include "flashmoe/decode_state.h"

#include <array>
#include <functional>

namespace flashmoe {
namespace {

constexpr std::array<const char*, 32> kRuntimeVocab = {
    "the",      "model",   "returns", "streamed", "cache",   "expert", "token",  "decode",
    "runtime",  "layer",   "router",  "slot",     "prompt",  "response", "kernel", "budget",
    "cuda",     "prefill", "warm",    "cold",     "kv",      "dense",  "moe",    "trace",
    "session",  "hidden",  "logits",  "sample",   "latency", "memory", "route",  "compute",
};

std::uint64_t mix_component(std::uint64_t seed, std::uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed;
}

std::uint64_t quantize_ms(double ms) {
    return static_cast<std::uint64_t>(ms * 1000.0);
}

std::uint64_t step_seed(const DecodeOpTrace& step, std::size_t token_index) {
    std::uint64_t seed = 1469598103934665603ULL ^ static_cast<std::uint64_t>(token_index);
    seed = mix_component(seed, quantize_ms(step.embed_ms));
    seed = mix_component(seed, quantize_ms(step.attention_ms));
    seed = mix_component(seed, quantize_ms(step.norm_router_ms));
    seed = mix_component(seed, quantize_ms(step.lm_head_ms));
    seed = mix_component(seed, quantize_ms(step.route_ms));
    seed = mix_component(seed, quantize_ms(step.load_ms));
    seed = mix_component(seed, quantize_ms(step.unpack_ms));
    seed = mix_component(seed, quantize_ms(step.compute_ms));
    seed = mix_component(seed, quantize_ms(step.combine_ms));
    seed = mix_component(seed, step.promotions);
    seed = mix_component(seed, step.hits);
    return seed;
}

std::uint64_t prompt_seed(std::string_view prompt) {
    return std::hash<std::string_view>{}(prompt) ^ 0x517cc1b727220a95ULL;
}

}  // namespace

DecodeState make_decode_state(std::string_view prompt) {
    DecodeState state;
    state.hidden_seed = prompt_seed(prompt);
    return state;
}

void apply_prefill_step(DecodeState& state, const DecodeOpTrace& step, std::size_t token_index) {
    state.hidden_seed = mix_component(state.hidden_seed, step_seed(step, token_index));
    state.position += 1;
}

void apply_decode_step(DecodeState& state, const DecodeOpTrace& step, std::size_t token_index) {
    state.hidden_seed = mix_component(state.hidden_seed ^ 0x3141592653589793ULL, step_seed(step, token_index));
    state.position += 1;
}

SampledToken sample_next_token(const DecodeState& state, std::string_view prompt) {
    const std::uint64_t prompt_mix = std::hash<std::string_view>{}(prompt);
    const std::uint64_t mixed = mix_component(state.hidden_seed, prompt_mix ^ state.position);
    const std::uint32_t token_id = static_cast<std::uint32_t>(mixed % kRuntimeVocab.size());
    return SampledToken{
        .token_id = token_id,
        .text = std::string(kRuntimeVocab[token_id]),
    };
}

std::string token_text_from_id(std::uint32_t token_id) {
    return std::string(kRuntimeVocab[token_id % kRuntimeVocab.size()]);
}

}  // namespace flashmoe
