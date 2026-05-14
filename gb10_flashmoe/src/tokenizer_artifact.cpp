#include "flashmoe/tokenizer_artifact.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <array>
#include <cstdio>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace flashmoe {
namespace {

std::string load_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open tokenizer artifact: " + path);
    }
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

std::string json_unescape(std::string text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\\' && i + 1 < text.size()) {
            const char next = text[i + 1];
            switch (next) {
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case '\\': out.push_back('\\'); break;
            case '"': out.push_back('"'); break;
            default: out.push_back(next); break;
            }
            i += 1;
        } else {
            out.push_back(text[i]);
        }
    }
    return out;
}

std::optional<std::uint32_t> extract_optional_u32(const std::string& text, const char* key) {
    const std::regex pattern(std::string("\"") + key + R"("\s*:\s*([0-9]+))");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(std::stoul(match[1].str()));
}

std::string normalize_runtime_token(std::string token) {
    if (token.rfind("Ġ", 0) == 0 || token.rfind("▁", 0) == 0) {
        token.erase(0, 1);
        return " " + token;
    }
    return token;
}

std::string shell_escape_single(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    out.push_back('\'');
    for (char ch : text) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

std::string run_command_capture(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string out;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return {};
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        out.append(buffer.data());
    }
    const int rc = pclose(pipe);
    if (rc != 0) {
        return {};
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

std::vector<std::uint32_t> parse_json_u32_list(const std::string& text) {
    std::vector<std::uint32_t> values;
    const std::regex number_pattern(R"(([0-9]+))");
    auto begin = text.cbegin();
    auto end = text.cend();
    std::match_results<std::string::const_iterator> m;
    while (std::regex_search(begin, end, m, number_pattern)) {
        values.push_back(static_cast<std::uint32_t>(std::stoul(m[1].str())));
        begin = m.suffix().first;
    }
    return values;
}

std::vector<std::uint32_t> encode_with_python_tokenizer(std::string_view prompt, const std::string& model_dir) {
    const auto bridge = std::filesystem::path(__FILE__).parent_path().parent_path() / "python" / "flashmoe_vllm_plugin" / "tokenizer_bridge.py";
    const auto cmd = std::string("python3 ") + shell_escape_single(bridge.string()) +
        " --model " + shell_escape_single(model_dir) +
        " --encode " + shell_escape_single(std::string(prompt));
    const auto output = run_command_capture(cmd);
    return output.empty() ? std::vector<std::uint32_t>{} : parse_json_u32_list(output);
}

std::string decode_with_python_tokenizer(std::uint32_t token_id, const std::string& model_dir) {
    const auto bridge = std::filesystem::path(__FILE__).parent_path().parent_path() / "python" / "flashmoe_vllm_plugin" / "tokenizer_bridge.py";
    const auto cmd = std::string("python3 ") + shell_escape_single(bridge.string()) +
        " --model " + shell_escape_single(model_dir) +
        " --decode-ids " + shell_escape_single("[" + std::to_string(token_id) + "]");
    return run_command_capture(cmd);
}

std::string extract_json_object(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
        return {};
    }
    const auto brace_pos = text.find('{', key_pos);
    if (brace_pos == std::string::npos) {
        return {};
    }
    int depth = 0;
    bool in_string = false;
    for (std::size_t i = brace_pos; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '"' && (i == 0 || text[i - 1] != '\\')) {
            in_string = !in_string;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            depth += 1;
        } else if (ch == '}') {
            depth -= 1;
            if (depth == 0) {
                return text.substr(brace_pos, i - brace_pos + 1);
            }
        }
    }
    return {};
}

}  // namespace

std::optional<TokenizerArtifact> load_tokenizer_artifact(const std::string& path) {
    if (path.empty()) {
        return std::nullopt;
    }
    const auto text = load_text(path);
    const std::regex pattern(R"("tokens"\s*:\s*\[([^\]]*)\])");
    std::smatch outer;
    if (!std::regex_search(text, outer, pattern)) {
        throw std::runtime_error("tokenizer artifact missing tokens array: " + path);
    }

    TokenizerArtifact artifact;
    artifact.model_dir.clear();
    artifact.direct_model_mode = false;
    const std::regex item_pattern("\"((?:[^\"\\\\]|\\\\.)*)\"");
    auto begin = outer[1].first;
    auto end = outer[1].second;
    std::match_results<std::string::const_iterator> item;
    while (std::regex_search(begin, end, item, item_pattern)) {
        artifact.tokens.push_back(json_unescape(item[1].str()));
        begin = item.suffix().first;
    }
    if (artifact.tokens.empty()) {
        throw std::runtime_error("tokenizer artifact tokens array is empty: " + path);
    }
    artifact.bos_token_id = extract_optional_u32(text, "bos_token_id");
    artifact.eos_token_id = extract_optional_u32(text, "eos_token_id");
    artifact.unk_token_id = extract_optional_u32(text, "unk_token_id");
    return artifact;
}

std::optional<TokenizerArtifact> load_tokenizer_artifact_from_model_dir(const std::string& model_dir_path) {
    if (model_dir_path.empty()) {
        return std::nullopt;
    }
    const std::filesystem::path model_dir(model_dir_path);
    const auto tokenizer_path = model_dir / "tokenizer.json";
    const auto tokenizer_text = load_text(tokenizer_path.string());
    const auto vocab_text = extract_json_object(tokenizer_text, "vocab");
    if (vocab_text.empty()) {
        throw std::runtime_error("tokenizer.json is missing vocab object: " + tokenizer_path.string());
    }

    const std::regex item_pattern("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*([0-9]+)");
    std::size_t max_id = 0;
    std::vector<std::pair<std::size_t, std::string>> items;
    auto begin = vocab_text.cbegin();
    auto end = vocab_text.cend();
    std::match_results<std::string::const_iterator> item;
    while (std::regex_search(begin, end, item, item_pattern)) {
        const auto token = json_unescape(item[1].str());
        const auto id = static_cast<std::size_t>(std::stoull(item[2].str()));
        max_id = std::max(max_id, id);
        items.emplace_back(id, token);
        begin = item.suffix().first;
    }
    if (items.empty()) {
        throw std::runtime_error("tokenizer.json vocab is empty: " + tokenizer_path.string());
    }

    TokenizerArtifact artifact;
    artifact.model_dir = model_dir_path;
    artifact.direct_model_mode = true;
    artifact.tokens.assign(max_id + 1, std::string{});
    for (const auto& [id, token] : items) {
        artifact.tokens[id] = token;
    }
    for (std::size_t i = 0; i < artifact.tokens.size(); ++i) {
        if (artifact.tokens[i].empty()) {
            artifact.tokens[i] = "<tok_" + std::to_string(i) + ">";
        }
    }

    const auto tokenizer_config_path = model_dir / "tokenizer_config.json";
    if (std::filesystem::exists(tokenizer_config_path)) {
        const auto config_text = load_text(tokenizer_config_path.string());
        artifact.bos_token_id = extract_optional_u32(config_text, "bos_token_id");
        artifact.eos_token_id = extract_optional_u32(config_text, "eos_token_id");
        artifact.unk_token_id = extract_optional_u32(config_text, "unk_token_id");
    }
    return artifact;
}

std::string token_text_from_runtime_vocab(std::uint32_t token_id, const TokenizerArtifact* artifact) {
    if (artifact == nullptr || artifact->tokens.empty()) {
        return {};
    }
    if (artifact->direct_model_mode && !artifact->model_dir.empty()) {
        const auto decoded = decode_with_python_tokenizer(token_id, artifact->model_dir);
        if (!decoded.empty()) {
            return decoded;
        }
    }
    return normalize_runtime_token(artifact->tokens[token_id % artifact->tokens.size()]);
}

std::vector<std::uint32_t> encode_prompt_runtime_vocab(std::string_view prompt, const TokenizerArtifact* artifact) {
    if (artifact != nullptr && artifact->direct_model_mode && !artifact->model_dir.empty()) {
        const auto ids = encode_with_python_tokenizer(prompt, artifact->model_dir);
        if (!ids.empty()) {
            return ids;
        }
    }
    std::vector<std::uint32_t> token_ids;
    if (artifact != nullptr && artifact->bos_token_id.has_value()) {
        token_ids.push_back(*artifact->bos_token_id);
    }
    const std::string prompt_text(prompt);
    std::istringstream input{prompt_text};
    std::string piece;
    while (input >> piece) {
        std::uint32_t token_id = 0;
        bool found = false;
        if (artifact != nullptr) {
            for (std::size_t i = 0; i < artifact->tokens.size(); ++i) {
                if (artifact->tokens[i] == piece || artifact->tokens[i] == ("Ġ" + piece) || artifact->tokens[i] == ("▁" + piece)) {
                    token_id = static_cast<std::uint32_t>(i);
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            const auto hashed = std::hash<std::string>{}(piece);
            const auto vocab = artifact != nullptr && !artifact->tokens.empty() ? artifact->tokens.size() : 32U;
            token_id = static_cast<std::uint32_t>(hashed % vocab);
            if (artifact != nullptr && artifact->unk_token_id.has_value()) {
                token_id = *artifact->unk_token_id;
            }
        }
        token_ids.push_back(token_id);
    }
    if (token_ids.empty()) {
        token_ids.push_back(0);
    }
    return token_ids;
}

}  // namespace flashmoe
