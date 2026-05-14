#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace flashmoe {

struct DecodeOpTrace {
    double embed_ms = 0.0;
    double attention_ms = 0.0;
    double norm_router_ms = 0.0;
    double lm_head_ms = 0.0;
    double route_ms = 0.0;
    double load_ms = 0.0;
    double unpack_ms = 0.0;
    double compute_ms = 0.0;
    double combine_ms = 0.0;
    std::uint64_t promotions = 0;
    std::uint64_t hits = 0;
};

struct DecodeState {
    std::size_t position = 0;
    std::uint64_t hidden_seed = 0;
    std::vector<std::uint32_t> token_ids;
};

struct SampledToken {
    std::uint32_t token_id = 0;
    std::string text;
};

DecodeState make_decode_state(std::string_view prompt);
void apply_prefill_step(DecodeState& state, const DecodeOpTrace& step, std::size_t token_index);
void apply_decode_step(DecodeState& state, const DecodeOpTrace& step, std::size_t token_index);
SampledToken sample_next_token(const DecodeState& state, std::string_view prompt);
std::string token_text_from_id(std::uint32_t token_id);

}  // namespace flashmoe
