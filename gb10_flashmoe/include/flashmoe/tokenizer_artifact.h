#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace flashmoe {

struct TokenizerArtifact {
    std::string model_dir;
    bool direct_model_mode = false;
    std::vector<std::string> tokens;
    std::optional<std::uint32_t> bos_token_id;
    std::optional<std::uint32_t> eos_token_id;
    std::optional<std::uint32_t> unk_token_id;
};

std::optional<TokenizerArtifact> load_tokenizer_artifact(const std::string& path);
std::optional<TokenizerArtifact> load_tokenizer_artifact_from_model_dir(const std::string& model_dir);
std::string token_text_from_runtime_vocab(std::uint32_t token_id, const TokenizerArtifact* artifact);
std::vector<std::uint32_t> encode_prompt_runtime_vocab(std::string_view prompt, const TokenizerArtifact* artifact);

}  // namespace flashmoe
