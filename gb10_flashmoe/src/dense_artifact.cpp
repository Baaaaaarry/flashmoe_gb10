#include "flashmoe/dense_artifact.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "flashmoe/model_spec.h"

namespace flashmoe {
namespace {

struct TensorView {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t data_begin = 0;
    std::size_t data_end = 0;
};

using WeightMap = std::unordered_map<std::string, std::string>;
using BlobCache = std::unordered_map<std::string, std::vector<std::uint8_t>>;

struct ModelDirRuntimeCache {
    WeightMap weight_map;
    BlobCache blob_cache;
    std::unordered_map<std::string, std::vector<std::uint16_t>> tensor_cache;  // BF16 storage
    std::unordered_map<std::string, std::vector<std::size_t>> shape_cache;
};

std::string load_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

std::vector<std::uint8_t> load_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size, 0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    return bytes;
}

std::string extract_string(const std::string& text, const char* key) {
    const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return {};
    }
    return match[1].str();
}

std::size_t extract_u64(const std::string& text, const char* key) {
    const std::regex pattern(std::string("\"") + key + R"("\s*:\s*([0-9]+))");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return 0;
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::vector<std::uint32_t> extract_u32_array(const std::string& text, const char* key) {
    const std::regex pattern(std::string("\"") + key + R"("\s*:\s*\[([^\]]*)\])");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return {};
    }
    std::vector<std::uint32_t> values;
    std::stringstream input(match[1].str());
    while (input.good()) {
        std::string item;
        std::getline(input, item, ',');
        if (item.empty()) {
            continue;
        }
        std::size_t begin = 0;
        while (begin < item.size() && std::isspace(static_cast<unsigned char>(item[begin]))) {
            begin += 1;
        }
        if (begin >= item.size()) {
            continue;
        }
        values.push_back(static_cast<std::uint32_t>(std::stoul(item.substr(begin))));
    }
    return values;
}

std::vector<float> load_float_blob(const std::string& path, std::size_t expected_count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open dense artifact blob: " + path);
    }
    std::vector<float> values(expected_count, 0.0f);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));
    const auto read_bytes = static_cast<std::size_t>(input.gcount());
    if (read_bytes != expected_count * sizeof(float)) {
        throw std::runtime_error("dense artifact blob size mismatch: " + path);
    }
    return values;
}

float finite_or_zero(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

float stable_silu(float x) {
    if (!std::isfinite(x)) {
        return 0.0f;
    }
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return x / (1.0f + z);
    }
    const float z = std::exp(x);
    return x * z / (1.0f + z);
}

std::string escape_regex(std::string_view text) {
    std::string out;
    out.reserve(text.size() * 2);
    for (const char ch : text) {
        switch (ch) {
        case '.': case '^': case '$': case '|': case '(': case ')':
        case '[': case ']': case '{': case '}': case '*': case '+':
        case '?': case '\\':
            out.push_back('\\');
            out.push_back(ch);
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
}

std::uint64_t read_u64_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000U) << 16U;
    std::uint32_t exponent = (bits >> 10U) & 0x1FU;
    std::uint32_t mantissa = bits & 0x03FFU;
    std::uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                exponent -= 1U;
            }
            mantissa &= 0x03FFU;
            out = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        out = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        out = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t out = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16U);
}

float fp8_e4m3_to_float(std::uint8_t bits) {
    if (bits == 0) {
        return 0.0f;
    }
    const int sign = (bits & 0x80U) ? -1 : 1;
    const int exponent = static_cast<int>((bits >> 3U) & 0x0FU);
    const int mantissa = static_cast<int>(bits & 0x07U);
    if (exponent == 0) {
        const float frac = static_cast<float>(mantissa) / 8.0f;
        return static_cast<float>(sign) * std::ldexp(frac, -6);
    }
    if (exponent == 0x0F) {
        if (mantissa == 0) {
            return sign > 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float frac = 1.0f + static_cast<float>(mantissa) / 8.0f;
    return static_cast<float>(sign) * std::ldexp(frac, exponent - 7);
}

float fp8_e5m2_to_float(std::uint8_t bits) {
    if (bits == 0) {
        return 0.0f;
    }
    const int sign = (bits & 0x80U) ? -1 : 1;
    const int exponent = static_cast<int>((bits >> 2U) & 0x1FU);
    const int mantissa = static_cast<int>(bits & 0x03U);
    if (exponent == 0) {
        const float frac = static_cast<float>(mantissa) / 4.0f;
        return static_cast<float>(sign) * std::ldexp(frac, -14);
    }
    if (exponent == 0x1F) {
        if (mantissa == 0) {
            return sign > 0 ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
        }
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float frac = 1.0f + static_cast<float>(mantissa) / 4.0f;
    return static_cast<float>(sign) * std::ldexp(frac, exponent - 15);
}

TensorView parse_tensor(const std::vector<std::uint8_t>& bytes, const std::string& name) {
    if (bytes.size() < sizeof(std::uint64_t)) {
        throw std::runtime_error("safetensors file too small");
    }
    const auto header_len = static_cast<std::size_t>(read_u64_le(bytes, 0));
    const auto header_begin = sizeof(std::uint64_t);
    const auto header_end = header_begin + header_len;
    if (header_end > bytes.size()) {
        throw std::runtime_error("safetensors header exceeds file size");
    }
    const std::string header(
        reinterpret_cast<const char*>(bytes.data() + header_begin),
        reinterpret_cast<const char*>(bytes.data() + header_end));
    const std::regex pattern(
        "\\\"" + escape_regex(name) +
        "\\\"\\s*:\\s*\\{\\s*\\\"dtype\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"\\s*,\\s*\\\"shape\\\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\\\"data_offsets\\\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    std::smatch match;
    if (!std::regex_search(header, match, pattern)) {
        throw std::runtime_error("tensor not found: " + name);
    }

    TensorView tensor;
    tensor.dtype = match[1].str();
    std::stringstream shape_input(match[2].str());
    while (shape_input.good()) {
        std::string item;
        std::getline(shape_input, item, ',');
        if (item.empty()) {
            continue;
        }
        std::size_t begin = 0;
        while (begin < item.size() && std::isspace(static_cast<unsigned char>(item[begin]))) {
            begin += 1;
        }
        if (begin < item.size()) {
            tensor.shape.push_back(static_cast<std::size_t>(std::stoull(item.substr(begin))));
        }
    }
    tensor.data_begin = header_end + static_cast<std::size_t>(std::stoull(match[3].str()));
    tensor.data_end = header_end + static_cast<std::size_t>(std::stoull(match[4].str()));
    if (tensor.data_end > bytes.size() || tensor.data_end < tensor.data_begin) {
        throw std::runtime_error("tensor offsets out of range: " + name);
    }
    return tensor;
}

std::vector<float> decode_float_tensor(const std::vector<std::uint8_t>& bytes, const TensorView& tensor) {
    const std::size_t count = [&]() {
        std::size_t n = 1;
        for (const auto dim : tensor.shape) {
            n *= dim;
        }
        return n;
    }();
    std::vector<float> out(count, 0.0f);
    const auto* data = bytes.data() + tensor.data_begin;
    if (tensor.dtype == "F32") {
        std::memcpy(out.data(), data, count * sizeof(float));
        return out;
    }
    if (tensor.dtype == "F16") {
        for (std::size_t i = 0; i < count; ++i) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, data + i * sizeof(std::uint16_t), sizeof(bits));
            out[i] = half_to_float(bits);
        }
        return out;
    }
    if (tensor.dtype == "BF16") {
        for (std::size_t i = 0; i < count; ++i) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, data + i * sizeof(std::uint16_t), sizeof(bits));
            out[i] = bf16_to_float(bits);
        }
        return out;
    }
    if (tensor.dtype == "F8_E4M3" || tensor.dtype == "F8E4M3") {
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = fp8_e4m3_to_float(data[i]);
        }
        return out;
    }
    if (tensor.dtype == "F8_E5M2" || tensor.dtype == "F8E5M2") {
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = fp8_e5m2_to_float(data[i]);
        }
        return out;
    }
    throw std::runtime_error("unsupported tensor dtype: " + tensor.dtype);
}

const std::vector<std::uint8_t>& get_blob(const std::filesystem::path& model_dir,
                                          const std::string& relative_path,
                                          BlobCache& cache) {
    const auto absolute = (model_dir / relative_path).string();
    auto it = cache.find(absolute);
    if (it == cache.end()) {
        it = cache.emplace(absolute, load_bytes(absolute)).first;
    }
    return it->second;
}

WeightMap load_weight_map(const std::filesystem::path& model_dir) {
    const auto index_path = model_dir / "model.safetensors.index.json";
    const auto text = load_text(index_path.string());
    const auto weight_map_pos = text.find("\"weight_map\"");
    if (weight_map_pos == std::string::npos) {
        throw std::runtime_error("weight_map not found in " + index_path.string());
    }
    const auto brace_begin = text.find('{', weight_map_pos);
    if (brace_begin == std::string::npos) {
        throw std::runtime_error("weight_map opening brace not found in " + index_path.string());
    }
    std::size_t brace_end = std::string::npos;
    int depth = 0;
    bool in_string = false;
    for (std::size_t i = brace_begin; i < text.size(); ++i) {
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
                brace_end = i;
                break;
            }
        }
    }
    if (brace_end == std::string::npos || brace_end <= brace_begin) {
        throw std::runtime_error("weight_map closing brace not found in " + index_path.string());
    }
    const auto body = text.substr(brace_begin + 1, brace_end - brace_begin - 1);

    const std::regex entry_pattern("\"((?:[^\"\\\\]|\\\\.)+)\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)+)\"");
    WeightMap map;
    auto begin = body.cbegin();
    auto end = body.cend();
    std::match_results<std::string::const_iterator> item;
    while (std::regex_search(begin, end, item, entry_pattern)) {
        map.emplace(item[1].str(), item[2].str());
        begin = item.suffix().first;
    }
    if (map.empty()) {
        throw std::runtime_error("weight_map entries not found in " + index_path.string());
    }
    return map;
}

std::string find_first_key(const WeightMap& map, const std::vector<std::string>& candidates) {
    for (const auto& key : candidates) {
        if (map.find(key) != map.end()) {
            return key;
        }
    }
    throw std::runtime_error("none of the expected keys were found");
}

std::string maybe_find_first_key(const WeightMap& map, const std::vector<std::string>& candidates) {
    for (const auto& key : candidates) {
        if (map.find(key) != map.end()) {
            return key;
        }
    }
    return {};
}

std::string find_router_key(const WeightMap& map, int layer) {
    return find_first_key(map, {
        "model.language_model.layers." + std::to_string(layer) + ".mlp.gate.weight",
        "language_model.layers." + std::to_string(layer) + ".mlp.gate.weight",
        "model.layers." + std::to_string(layer) + ".mlp.gate.weight",
        "layers." + std::to_string(layer) + ".mlp.gate.weight",
    });
}

std::string find_norm_key_optional(const WeightMap& map, int layer, const std::string& name) {
    return maybe_find_first_key(map, {
        "model.language_model.layers." + std::to_string(layer) + "." + name + ".weight",
        "language_model.layers." + std::to_string(layer) + "." + name + ".weight",
        "model.layers." + std::to_string(layer) + "." + name + ".weight",
        "layers." + std::to_string(layer) + "." + name + ".weight",
    });
}

std::string find_attention_key_optional(const WeightMap& map, int layer, const std::string& proj) {
    std::vector<std::string> aliases{proj};
    if (proj == "o_proj") {
        aliases.push_back("out_proj");
    }
    std::vector<std::string> candidates;
    for (const auto& alias : aliases) {
        candidates.push_back("model.language_model.layers." + std::to_string(layer) + ".self_attn." + alias + ".weight");
        candidates.push_back("language_model.layers." + std::to_string(layer) + ".self_attn." + alias + ".weight");
        candidates.push_back("model.layers." + std::to_string(layer) + ".self_attn." + alias + ".weight");
        candidates.push_back("layers." + std::to_string(layer) + ".self_attn." + alias + ".weight");
    }
    return maybe_find_first_key(map, candidates);
}

std::string find_linear_attn_key_optional(const WeightMap& map, int layer, const std::string& name) {
    return maybe_find_first_key(map, {
        "model.language_model.layers." + std::to_string(layer) + ".linear_attn." + name + ".weight",
        "language_model.layers." + std::to_string(layer) + ".linear_attn." + name + ".weight",
        "model.layers." + std::to_string(layer) + ".linear_attn." + name + ".weight",
        "layers." + std::to_string(layer) + ".linear_attn." + name + ".weight",
    });
}

std::string find_shared_expert_key_optional(const WeightMap& map, int layer, const std::string& proj) {
    return maybe_find_first_key(map, {
        "model.language_model.layers." + std::to_string(layer) + ".mlp.shared_expert." + proj + ".weight",
        "language_model.layers." + std::to_string(layer) + ".mlp.shared_expert." + proj + ".weight",
        "model.layers." + std::to_string(layer) + ".mlp.shared_expert." + proj + ".weight",
        "layers." + std::to_string(layer) + ".mlp.shared_expert." + proj + ".weight",
    });
}

std::vector<std::uint32_t> sampled_indices(std::size_t hidden_size, std::size_t sample_stride, std::size_t max_dims) {
    std::vector<std::uint32_t> out;
    for (std::size_t i = 0; i < hidden_size; i += std::max<std::size_t>(1, sample_stride)) {
        out.push_back(static_cast<std::uint32_t>(i));
        if (out.size() >= max_dims) {
            break;
        }
    }
    if (out.empty()) {
        out.push_back(0);
    }
    return out;
}

std::vector<float> sample_norm_weight(const std::vector<float>& tensor, const std::vector<std::size_t>& shape,
                                      const std::vector<std::uint32_t>& sampled) {
    if (shape.size() != 1) {
        throw std::runtime_error("expected 1D norm tensor");
    }
    std::vector<float> out(sampled.size(), 0.0f);
    for (std::size_t i = 0; i < sampled.size(); ++i) {
        out[i] = tensor[std::min<std::size_t>(sampled[i], shape[0] - 1)];
    }
    return out;
}

std::vector<float> sample_direct_projection(const std::vector<float>& tensor, const std::vector<std::size_t>& shape,
                                            const std::vector<std::uint32_t>& sampled, std::size_t hidden_size) {
    if (shape.size() != 2) {
        throw std::runtime_error("expected 2D projection tensor");
    }
    const std::size_t rows = shape[0];
    const std::size_t cols = shape[1];
    const std::size_t dims = sampled.size();
    std::vector<float> out(dims * dims, 0.0f);
    if (cols == hidden_size) {
        for (std::size_t r = 0; r < dims; ++r) {
            const auto src_r = std::min<std::size_t>(sampled[r], rows - 1);
            for (std::size_t c = 0; c < dims; ++c) {
                const auto src_c = std::min<std::size_t>(sampled[c], cols - 1);
                out[r * dims + c] = tensor[src_r * cols + src_c];
            }
        }
        return out;
    }
    if (rows == hidden_size) {
        for (std::size_t r = 0; r < dims; ++r) {
            const auto src_r = std::min<std::size_t>(sampled[r], rows - 1);
            for (std::size_t c = 0; c < dims; ++c) {
                const auto src_c = std::min<std::size_t>(sampled[c], cols - 1);
                out[r * dims + c] = tensor[src_r * cols + src_c];
            }
        }
        return out;
    }
    throw std::runtime_error("direct projection tensor does not consume hidden states");
}

std::vector<float> sample_rectangular_projection(const std::vector<float>& tensor, const std::vector<std::size_t>& shape,
                                                 const std::vector<std::uint32_t>& sampled) {
    if (shape.size() != 2) {
        throw std::runtime_error("expected 2D projection tensor");
    }
    const std::size_t dims = sampled.size();
    const std::size_t rows = shape[0];
    const std::size_t cols = shape[1];
    std::vector<float> out(dims * dims, 0.0f);
    for (std::size_t r = 0; r < dims; ++r) {
        const auto src_r = std::min<std::size_t>(r, rows - 1);
        for (std::size_t c = 0; c < dims; ++c) {
            const auto src_c = std::min<std::size_t>(sampled[c], cols - 1);
            out[r * dims + c] = tensor[src_r * cols + src_c];
        }
    }
    return out;
}

std::vector<float> matmul(const std::vector<float>& left, std::size_t left_rows, std::size_t left_cols,
                          const std::vector<float>& right, std::size_t right_cols) {
    std::vector<float> out(left_rows * right_cols, 0.0f);
    for (std::size_t r = 0; r < left_rows; ++r) {
        for (std::size_t k = 0; k < left_cols; ++k) {
            const auto lhs = left[r * left_cols + k];
            for (std::size_t c = 0; c < right_cols; ++c) {
                out[r * right_cols + c] += lhs * right[k * right_cols + c];
            }
        }
    }
    return out;
}

std::vector<float> materialize_low_rank_projection(const std::vector<float>& left, const std::vector<std::size_t>& left_shape,
                                                   const std::vector<float>& right, const std::vector<std::size_t>& right_shape,
                                                   const std::vector<std::uint32_t>& sampled, std::size_t hidden_size) {
    if (left_shape.size() != 2 || right_shape.size() != 2) {
        throw std::runtime_error("expected 2D low-rank tensors");
    }
    if (left_shape[1] != hidden_size || right_shape[1] != left_shape[0] || right_shape[0] != hidden_size) {
        throw std::runtime_error("low-rank projection dimensions do not align");
    }
    const std::size_t rank = left_shape[0];
    const std::size_t dims = sampled.size();
    std::vector<float> sampled_left(rank * dims, 0.0f);
    for (std::size_t r = 0; r < rank; ++r) {
        for (std::size_t c = 0; c < dims; ++c) {
            sampled_left[r * dims + c] = left[r * left_shape[1] + sampled[c]];
        }
    }
    std::vector<float> sampled_right(dims * rank, 0.0f);
    for (std::size_t r = 0; r < dims; ++r) {
        const auto src_r = sampled[r];
        for (std::size_t c = 0; c < rank; ++c) {
            sampled_right[r * rank + c] = right[src_r * right_shape[1] + c];
        }
    }
    return matmul(sampled_right, dims, rank, sampled_left, dims);
}

std::vector<float> sample_router_weight(const std::vector<float>& tensor, const std::vector<std::size_t>& shape,
                                        const std::vector<std::uint32_t>& sampled, std::size_t hidden_size) {
    if (shape.size() != 2) {
        throw std::runtime_error("expected 2D router tensor");
    }
    if (shape[1] == hidden_size) {
        std::vector<float> out(shape[0] * sampled.size(), 0.0f);
        for (std::size_t r = 0; r < shape[0]; ++r) {
            for (std::size_t c = 0; c < sampled.size(); ++c) {
                out[r * sampled.size() + c] = tensor[r * shape[1] + sampled[c]];
            }
        }
        return out;
    }
    if (shape[0] == hidden_size) {
        std::vector<float> out(shape[1] * sampled.size(), 0.0f);
        for (std::size_t expert = 0; expert < shape[1]; ++expert) {
            for (std::size_t d = 0; d < sampled.size(); ++d) {
                out[expert * sampled.size() + d] = tensor[sampled[d] * shape[1] + expert];
            }
        }
        return out;
    }
    throw std::runtime_error("router tensor shape does not match hidden size");
}

std::vector<float> load_named_tensor(const std::filesystem::path& model_dir, const WeightMap& weight_map,
                                     BlobCache& cache, const std::string& key, std::vector<std::size_t>* shape_out = nullptr) {
    const auto weight_it = weight_map.find(key);
    if (weight_it == weight_map.end()) {
        throw std::runtime_error("missing weight key: " + key);
    }
    const auto& blob = get_blob(model_dir, weight_it->second, cache);
    const auto view = parse_tensor(blob, key);
    if (shape_out != nullptr) {
        *shape_out = view.shape;
    }
    return decode_float_tensor(blob, view);
}

ModelDirRuntimeCache& runtime_cache_for_model_dir(const std::string& model_dir) {
    static std::unordered_map<std::string, ModelDirRuntimeCache> caches;
    auto it = caches.find(model_dir);
    if (it == caches.end()) {
        ModelDirRuntimeCache cache;
        cache.weight_map = load_weight_map(std::filesystem::path(model_dir));
        it = caches.emplace(model_dir, std::move(cache)).first;
    }
    return it->second;
}

}  // namespace

void flush_model_dir_blob_cache(const std::string& model_dir) {
    auto& cache = runtime_cache_for_model_dir(model_dir);
    cache.blob_cache.clear();
}

namespace {

// Returns tensor as float32 (decoded from BF16 cache on hit, or freshly decoded on miss).
// Stores BF16 in tensor_cache to halve persistent memory usage (~27 GB → ~13.5 GB).
std::vector<float> load_named_tensor_cached(const DenseRuntimeArtifact& artifact,
                                            const std::string& key,
                                            std::vector<std::size_t>* shape_out = nullptr) {
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    auto tensor_it = cache.tensor_cache.find(key);
    if (tensor_it == cache.tensor_cache.end()) {
        const auto model_dir = std::filesystem::path(artifact.model_dir);
        const auto weight_it = cache.weight_map.find(key);
        if (weight_it == cache.weight_map.end()) {
            throw std::runtime_error("missing weight key: " + key);
        }
        const auto& blob = get_blob(model_dir, weight_it->second, cache.blob_cache);
        const auto view = parse_tensor(blob, key);
        std::vector<std::size_t> shape = view.shape;
        const auto tensor_f32 = decode_float_tensor(blob, view);
        cache.shape_cache[key] = shape;
        std::vector<std::uint16_t> tensor_bf16(tensor_f32.size());
        for (std::size_t i = 0; i < tensor_f32.size(); ++i) {
            tensor_bf16[i] = float_to_bf16(tensor_f32[i]);
        }
        tensor_it = cache.tensor_cache.emplace(key, std::move(tensor_bf16)).first;
    }
    if (shape_out != nullptr) {
        *shape_out = cache.shape_cache[key];
    }
    const auto& bf16 = tensor_it->second;
    std::vector<float> out(bf16.size());
    for (std::size_t i = 0; i < bf16.size(); ++i) {
        out[i] = bf16_to_float(bf16[i]);
    }
    return out;
}

std::string find_embed_key(const WeightMap& weight_map) {
    return find_first_key(weight_map, {
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
    });
}

std::string find_lm_head_key(const WeightMap& weight_map) {
    return find_first_key(weight_map, {
        "lm_head.weight",
        "model.lm_head.weight",
        "language_model.lm_head.weight",
    });
}

std::string find_final_norm_key(const WeightMap& weight_map) {
    return find_first_key(weight_map, {
        "model.norm.weight",
        "model.language_model.norm.weight",
        "language_model.norm.weight",
        "norm.weight",
    });
}

}  // namespace

std::optional<DenseRuntimeArtifact> load_dense_runtime_artifact(const std::string& manifest_json_path) {
    if (manifest_json_path.empty()) {
        return std::nullopt;
    }
    const auto text = load_text(manifest_json_path);
    DenseRuntimeArtifact artifact;
    artifact.family = extract_string(text, "family");
    artifact.model_dir.clear();
    artifact.direct_model_mode = false;
    artifact.sampled_indices = extract_u32_array(text, "sampled_indices");
    artifact.runtime_vocab_size = extract_u64(text, "runtime_vocab_size");
    artifact.num_layers = extract_u64(text, "num_layers");
    artifact.routed_experts_per_layer = extract_u64(text, "routed_experts_per_layer");

    const auto sampled_dims = artifact.sampled_indices.size();
    artifact.hidden_size = sampled_dims;
    if (sampled_dims == 0 || artifact.runtime_vocab_size == 0 || artifact.num_layers == 0 || artifact.routed_experts_per_layer == 0) {
        throw std::runtime_error("dense artifact metadata is incomplete: " + manifest_json_path);
    }

    const auto embed_path = extract_string(text, "embedding_path");
    const auto q_proj_path = extract_string(text, "q_proj_path");
    const auto k_proj_path = extract_string(text, "k_proj_path");
    const auto v_proj_path = extract_string(text, "v_proj_path");
    const auto o_proj_path = extract_string(text, "o_proj_path");
    const auto input_norm_path = extract_string(text, "input_norm_path");
    const auto post_norm_path = extract_string(text, "post_norm_path");
    const auto router_path = extract_string(text, "router_path");
    const auto lm_head_path = extract_string(text, "lm_head_path");
    artifact.embedding_weights = load_float_blob(embed_path, artifact.runtime_vocab_size * sampled_dims);
    artifact.q_proj_weights = load_float_blob(q_proj_path, artifact.num_layers * sampled_dims * sampled_dims);
    artifact.k_proj_weights = load_float_blob(k_proj_path, artifact.num_layers * sampled_dims * sampled_dims);
    artifact.v_proj_weights = load_float_blob(v_proj_path, artifact.num_layers * sampled_dims * sampled_dims);
    artifact.o_proj_weights = load_float_blob(o_proj_path, artifact.num_layers * sampled_dims * sampled_dims);
    artifact.input_norm_weights = load_float_blob(input_norm_path, artifact.num_layers * sampled_dims);
    artifact.post_norm_weights = load_float_blob(post_norm_path, artifact.num_layers * sampled_dims);
    artifact.router_weights = load_float_blob(router_path, artifact.num_layers * artifact.routed_experts_per_layer * sampled_dims);
    artifact.lm_head_weights = load_float_blob(lm_head_path, artifact.runtime_vocab_size * sampled_dims);
    return artifact;
}

std::optional<DenseRuntimeArtifact> load_dense_runtime_artifact_from_model_dir(
    const std::string& model_dir_path,
    const ModelSpec& spec,
    std::size_t sample_stride,
    std::size_t max_sampled_dims,
    std::size_t runtime_vocab_size) {
    if (model_dir_path.empty()) {
        return std::nullopt;
    }
    const std::filesystem::path model_dir(model_dir_path);
    const auto weight_map = load_weight_map(model_dir);
    BlobCache cache;

    const auto embed_key = find_first_key(weight_map, {
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
    });
    const auto lm_head_key = find_first_key(weight_map, {
        "lm_head.weight",
        "model.lm_head.weight",
        "language_model.lm_head.weight",
    });

    std::vector<std::size_t> embed_shape;
    const auto embed = load_named_tensor(model_dir, weight_map, cache, embed_key, &embed_shape);
    const auto hidden_size = embed_shape.at(1);
    const auto sampled = sampled_indices(hidden_size, sample_stride, max_sampled_dims);
    const auto vocab_size = std::min<std::size_t>(runtime_vocab_size, embed_shape.at(0));

    DenseRuntimeArtifact artifact;
    artifact.family = std::string(spec.short_name);
    artifact.model_dir = model_dir_path;
    artifact.direct_model_mode = true;
    artifact.hidden_size = hidden_size;
    artifact.sampled_indices = sampled;
    artifact.runtime_vocab_size = vocab_size;
    artifact.num_layers = spec.num_layers;
    artifact.routed_experts_per_layer = spec.routed_experts_per_layer;
    // Direct model mode loads all tensors from disk on demand during inference
    // via load_named_tensor_cached. The sampled weight fields (embedding_weights,
    // lm_head_weights, q_proj_weights, etc.) are never consulted when
    // uses_direct_model() is true, so skip the expensive per-layer loading loop
    // that would otherwise hold multiple large shard files in memory at startup.
    return artifact;

    // The code below is unreachable in direct_model_mode and exists only for the
    // sampled-artifact path that is currently unused. It is kept so the sampled
    // path can be re-enabled if needed by a future caller that sets
    // direct_model_mode = false after this function.
    artifact.embedding_weights.assign(vocab_size * sampled.size(), 0.0f);
    for (std::size_t row = 0; row < vocab_size; ++row) {
        for (std::size_t d = 0; d < sampled.size(); ++d) {
            artifact.embedding_weights[row * sampled.size() + d] = embed[row * hidden_size + sampled[d]];
        }
    }

    std::vector<std::size_t> lm_head_shape;
    const auto lm_head = load_named_tensor(model_dir, weight_map, cache, lm_head_key, &lm_head_shape);
    artifact.lm_head_weights.assign(vocab_size * sampled.size(), 0.0f);
    for (std::size_t row = 0; row < vocab_size; ++row) {
        for (std::size_t d = 0; d < sampled.size(); ++d) {
            artifact.lm_head_weights[row * sampled.size() + d] = lm_head[row * lm_head_shape.at(1) + sampled[d]];
        }
    }

    for (std::size_t layer = 0; layer < spec.num_layers; ++layer) {
        const auto router_key = find_router_key(weight_map, static_cast<int>(layer));
        std::vector<std::size_t> router_shape;
        const auto router = load_named_tensor(model_dir, weight_map, cache, router_key, &router_shape);
        const auto router_sampled = sample_router_weight(router, router_shape, sampled, hidden_size);
        artifact.router_weights.insert(artifact.router_weights.end(), router_sampled.begin(), router_sampled.end());

        const auto q_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "q_proj");
        const auto k_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "k_proj");
        const auto v_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "v_proj");
        const auto o_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "o_proj");

        std::vector<float> q_sampled;
        std::vector<float> k_sampled;
        std::vector<float> v_sampled;
        std::vector<float> o_sampled;

        if (!q_key.empty() && !k_key.empty() && !v_key.empty() && !o_key.empty()) {
            std::vector<std::size_t> shape;
            q_sampled = sample_direct_projection(load_named_tensor(model_dir, weight_map, cache, q_key, &shape), shape, sampled, hidden_size);
            k_sampled = sample_direct_projection(load_named_tensor(model_dir, weight_map, cache, k_key, &shape), shape, sampled, hidden_size);
            v_sampled = sample_direct_projection(load_named_tensor(model_dir, weight_map, cache, v_key, &shape), shape, sampled, hidden_size);
            o_sampled = sample_direct_projection(load_named_tensor(model_dir, weight_map, cache, o_key, &shape), shape, sampled, hidden_size);
        } else {
            const auto q_a_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "q_a_proj");
            const auto q_b_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "q_b_proj");
            const auto kv_a_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "kv_a_proj_with_mqa");
            const auto kv_b_key = find_attention_key_optional(weight_map, static_cast<int>(layer), "kv_b_proj");
            if (!q_a_key.empty() && !q_b_key.empty() && !kv_a_key.empty() && !kv_b_key.empty()) {
                std::vector<std::size_t> left_shape;
                std::vector<std::size_t> right_shape;
                const auto q_left = load_named_tensor(model_dir, weight_map, cache, q_a_key, &left_shape);
                const auto q_right = load_named_tensor(model_dir, weight_map, cache, q_b_key, &right_shape);
                q_sampled = materialize_low_rank_projection(q_left, left_shape, q_right, right_shape, sampled, hidden_size);
                const auto kv_left = load_named_tensor(model_dir, weight_map, cache, kv_a_key, &left_shape);
                const auto kv_right = load_named_tensor(model_dir, weight_map, cache, kv_b_key, &right_shape);
                const auto kv_sampled = materialize_low_rank_projection(kv_left, left_shape, kv_right, right_shape, sampled, hidden_size);
                k_sampled = kv_sampled;
                v_sampled = kv_sampled;
                std::vector<std::size_t> o_shape;
                o_sampled = sample_direct_projection(load_named_tensor(model_dir, weight_map, cache, find_first_key(weight_map, {
                    "model.language_model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight",
                    "language_model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight",
                    "model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight",
                    "layers." + std::to_string(layer) + ".self_attn.o_proj.weight",
                    "model.language_model.layers." + std::to_string(layer) + ".self_attn.out_proj.weight",
                    "language_model.layers." + std::to_string(layer) + ".self_attn.out_proj.weight",
                    "model.layers." + std::to_string(layer) + ".self_attn.out_proj.weight",
                    "layers." + std::to_string(layer) + ".self_attn.out_proj.weight",
                }), &o_shape), o_shape, sampled, hidden_size);
            } else {
                const auto linear_qkv_key = find_linear_attn_key_optional(weight_map, static_cast<int>(layer), "in_proj_qkv");
                const auto linear_out_key = find_linear_attn_key_optional(weight_map, static_cast<int>(layer), "out_proj");
                const auto linear_a_key = find_linear_attn_key_optional(weight_map, static_cast<int>(layer), "in_proj_a");
                const auto linear_b_key = find_linear_attn_key_optional(weight_map, static_cast<int>(layer), "in_proj_b");
                if (!linear_qkv_key.empty() && !linear_out_key.empty()) {
                    std::vector<std::size_t> qkv_shape;
                    const auto qkv = load_named_tensor(model_dir, weight_map, cache, linear_qkv_key, &qkv_shape);
                    const auto rows = qkv_shape.at(0);
                    const auto cols = qkv_shape.at(1);
                    const auto chunk = std::max<std::size_t>(1, rows / 3);
                    const auto take_rect = [&](std::size_t start, std::size_t end) {
                        const auto clipped_end = std::min(rows, end);
                        const auto clipped_start = std::min(start, clipped_end);
                        std::vector<float> sub((clipped_end - clipped_start) * cols, 0.0f);
                        std::copy_n(qkv.data() + clipped_start * cols, sub.size(), sub.data());
                        return sample_rectangular_projection(sub, {clipped_end - clipped_start, cols}, sampled);
                    };
                    q_sampled = take_rect(0, chunk);
                    k_sampled = take_rect(chunk, std::min(rows, 2 * chunk));
                    v_sampled = take_rect(std::min(rows, 2 * chunk), rows);
                    std::vector<std::size_t> out_shape;
                    o_sampled = sample_rectangular_projection(load_named_tensor(model_dir, weight_map, cache, linear_out_key, &out_shape), out_shape, sampled);
                } else if (!linear_a_key.empty() && !linear_b_key.empty() && !linear_out_key.empty()) {
                    std::vector<std::size_t> left_shape;
                    std::vector<std::size_t> right_shape;
                    const auto left = load_named_tensor(model_dir, weight_map, cache, linear_a_key, &left_shape);
                    const auto right = load_named_tensor(model_dir, weight_map, cache, linear_b_key, &right_shape);
                    const auto effective = materialize_low_rank_projection(left, left_shape, right, right_shape, sampled, hidden_size);
                    q_sampled = effective;
                    k_sampled = effective;
                    v_sampled = effective;
                    std::vector<std::size_t> out_shape;
                    o_sampled = sample_rectangular_projection(load_named_tensor(model_dir, weight_map, cache, linear_out_key, &out_shape), out_shape, sampled);
                } else {
                    throw std::runtime_error("missing supported attention projection weights at layer " + std::to_string(layer));
                }
            }
        }
        artifact.q_proj_weights.insert(artifact.q_proj_weights.end(), q_sampled.begin(), q_sampled.end());
        artifact.k_proj_weights.insert(artifact.k_proj_weights.end(), k_sampled.begin(), k_sampled.end());
        artifact.v_proj_weights.insert(artifact.v_proj_weights.end(), v_sampled.begin(), v_sampled.end());
        artifact.o_proj_weights.insert(artifact.o_proj_weights.end(), o_sampled.begin(), o_sampled.end());

        auto input_norm_key = find_norm_key_optional(weight_map, static_cast<int>(layer), "input_layernorm");
        auto post_norm_key = find_norm_key_optional(weight_map, static_cast<int>(layer), "post_attention_layernorm");
        if (input_norm_key.empty()) {
            input_norm_key = find_norm_key_optional(weight_map, static_cast<int>(layer), "self_attn.q_a_layernorm");
        }
        if (post_norm_key.empty()) {
            post_norm_key = find_norm_key_optional(weight_map, static_cast<int>(layer), "self_attn.kv_a_layernorm");
        }
        if (input_norm_key.empty()) {
            input_norm_key = find_norm_key_optional(weight_map, static_cast<int>(layer), "linear_attn.in_norm");
        }
        if (post_norm_key.empty()) {
            post_norm_key = find_norm_key_optional(weight_map, static_cast<int>(layer), "linear_attn.out_norm");
        }
        if (input_norm_key.empty() || post_norm_key.empty()) {
            throw std::runtime_error("missing norm weights at layer " + std::to_string(layer));
        }
        std::vector<std::size_t> norm_shape;
        const auto input_norm = load_named_tensor(model_dir, weight_map, cache, input_norm_key, &norm_shape);
        const auto post_norm = load_named_tensor(model_dir, weight_map, cache, post_norm_key, &norm_shape);
        const auto input_sampled = sample_norm_weight(input_norm, norm_shape, sampled);
        const auto post_sampled = sample_norm_weight(post_norm, norm_shape, sampled);
        artifact.input_norm_weights.insert(artifact.input_norm_weights.end(), input_sampled.begin(), input_sampled.end());
        artifact.post_norm_weights.insert(artifact.post_norm_weights.end(), post_sampled.begin(), post_sampled.end());
    }

    return artifact;
}

std::vector<float> lookup_embedding_from_model_dir(const DenseRuntimeArtifact& artifact, std::uint32_t token_id) {
    if (!artifact.uses_direct_model()) {
        return {};
    }
    // Ensure the embedding tensor is in the BF16 cache (first call loads from disk).
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    const auto key = find_embed_key(cache.weight_map);
    std::vector<std::size_t> shape;
    // Prime the tensor_cache (may trigger disk I/O on first call).
    load_named_tensor_cached(artifact, key, &shape);
    if (shape.size() != 2) {
        throw std::runtime_error("embedding tensor is not rank-2");
    }
    const auto vocab = shape[0];
    const auto hidden = shape[1];
    const auto row = static_cast<std::size_t>(token_id) % std::max<std::size_t>(1, vocab);

    // Decode only the needed row from the BF16 cache rather than the full ~2 GB tensor.
    const auto& bf16 = cache.tensor_cache.at(key);
    std::vector<float> out(hidden, 0.0f);
    const std::size_t row_offset = row * hidden;
    for (std::size_t i = 0; i < hidden; ++i) {
        out[i] = bf16_to_float(bf16[row_offset + i]);
    }
    return out;
}

std::vector<float> lookup_norm_from_model_dir(const DenseRuntimeArtifact& artifact, std::size_t layer, bool post_attention) {
    if (!artifact.uses_direct_model()) {
        return {};
    }
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    const auto layer_i = static_cast<int>(layer);
    auto key = post_attention
        ? find_norm_key_optional(cache.weight_map, layer_i, "post_attention_layernorm")
        : find_norm_key_optional(cache.weight_map, layer_i, "input_layernorm");
    if (key.empty() && !post_attention) {
        key = find_norm_key_optional(cache.weight_map, layer_i, "self_attn.q_a_layernorm");
    }
    if (key.empty() && post_attention) {
        key = find_norm_key_optional(cache.weight_map, layer_i, "self_attn.kv_a_layernorm");
    }
    if (key.empty() && !post_attention) {
        key = find_norm_key_optional(cache.weight_map, layer_i, "linear_attn.in_norm");
    }
    if (key.empty() && post_attention) {
        key = find_norm_key_optional(cache.weight_map, layer_i, "linear_attn.out_norm");
    }
    if (key.empty()) {
        return {};
    }
    std::vector<std::size_t> shape;
    auto tensor = load_named_tensor_cached(artifact, key, &shape);
    if (shape.size() != 1) {
        throw std::runtime_error("norm tensor is not rank-1");
    }
    return tensor;
}

std::vector<float> router_scores_from_model_dir(const DenseRuntimeArtifact& artifact, std::size_t layer, const std::vector<float>& hidden) {
    if (!artifact.uses_direct_model()) {
        return {};
    }
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    const auto key = find_router_key(cache.weight_map, static_cast<int>(layer));
    std::vector<std::size_t> shape;
    load_named_tensor_cached(artifact, key, &shape);
    if (shape.size() != 2) {
        throw std::runtime_error("router tensor is not rank-2");
    }
    const std::size_t rows = shape[0];
    const std::size_t cols = shape[1];
    const auto& bf16 = cache.tensor_cache.at(key);
    std::vector<float> out(rows, 0.0f);
    if (cols == hidden.size()) {
        for (std::size_t r = 0; r < rows; ++r) {
            float acc = 0.0f;
            const std::size_t row_off = r * cols;
            for (std::size_t c = 0; c < cols; ++c) {
                acc += bf16_to_float(bf16[row_off + c]) * hidden[c];
            }
            out[r] = acc;
        }
    } else if (rows == hidden.size()) {
        for (std::size_t c = 0; c < cols; ++c) {
            float acc = 0.0f;
            for (std::size_t r = 0; r < rows; ++r) {
                acc += bf16_to_float(bf16[r * cols + c]) * hidden[r];
            }
            out[c] = acc;
        }
    } else {
        throw std::runtime_error("router tensor shape does not match hidden size");
    }
    return out;
}

std::vector<float> lm_head_logits_from_model_dir(const DenseRuntimeArtifact& artifact, const std::vector<float>& hidden) {
    if (!artifact.uses_direct_model()) {
        return {};
    }
    // Prime the BF16 cache on first call; subsequent calls skip disk I/O.
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    const auto key = find_lm_head_key(cache.weight_map);
    std::vector<std::size_t> shape;
    load_named_tensor_cached(artifact, key, &shape);
    if (shape.size() != 2) {
        throw std::runtime_error("lm_head tensor is not rank-2");
    }
    const std::size_t rows = shape[0];
    const std::size_t cols = shape[1];
    const std::size_t vocab = artifact.runtime_vocab_size > 0 ? std::min<std::size_t>(artifact.runtime_vocab_size, rows) : rows;
    if (cols != hidden.size()) {
        throw std::runtime_error("lm_head width does not match hidden size");
    }
    // Matvec directly on the BF16 cache — avoids allocating the ~4 GB float32 intermediate.
    const auto& bf16 = cache.tensor_cache.at(key);
    std::vector<float> logits(vocab, 0.0f);
    for (std::size_t token = 0; token < vocab; ++token) {
        float acc = 0.0f;
        const std::size_t row_offset = token * cols;
        for (std::size_t d = 0; d < cols; ++d) {
            acc += bf16_to_float(bf16[row_offset + d]) * hidden[d];
        }
        logits[token] = acc;
    }
    return logits;
}

std::vector<float> final_norm_from_model_dir(const DenseRuntimeArtifact& artifact) {
    if (!artifact.uses_direct_model()) {
        return {};
    }
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    const auto key = find_final_norm_key(cache.weight_map);
    std::vector<std::size_t> shape;
    auto tensor = load_named_tensor_cached(artifact, key, &shape);
    if (shape.size() != 1) {
        throw std::runtime_error("final norm tensor is not rank-1");
    }
    return tensor;
}

std::vector<float> attention_projection_from_model_dir(
    const DenseRuntimeArtifact& artifact,
    std::size_t layer,
    std::string_view projection,
    const std::vector<float>& input,
    std::size_t target_dim) {
    if (!artifact.uses_direct_model()) {
        return {};
    }
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    const auto layer_i = static_cast<int>(layer);

    // Matvec directly on BF16-cached weights to avoid a full float32 decode (saves ~234–468 MB per call).
    const auto matvec_bf16 = [&](const std::string& key, std::size_t out_dim) -> std::vector<float> {
        // Prime tensor_cache on first call; hits BF16 cache afterward.
        std::vector<std::size_t> shape;
        load_named_tensor_cached(artifact, key, &shape);
        if (shape.size() != 2) {
            throw std::runtime_error("attention projection tensor is not rank-2");
        }
        const std::size_t rows = shape[0];
        const std::size_t cols = shape[1];
        const auto& bf16 = cache.tensor_cache.at(key);
        const std::size_t in_size = input.size();

        if (cols == in_size) {
            std::vector<float> out(std::min(out_dim, rows), 0.0f);
            for (std::size_t r = 0; r < out.size(); ++r) {
                float acc = 0.0f;
                const std::size_t row_off = r * cols;
                for (std::size_t c = 0; c < cols; ++c) {
                    acc += bf16_to_float(bf16[row_off + c]) * input[c];
                }
                out[r] = acc;
            }
            return out;
        }
        if (rows == in_size) {
            std::vector<float> out(std::min(out_dim, cols), 0.0f);
            for (std::size_t c = 0; c < out.size(); ++c) {
                float acc = 0.0f;
                for (std::size_t r = 0; r < rows; ++r) {
                    acc += bf16_to_float(bf16[r * cols + c]) * input[r];
                }
                out[c] = acc;
            }
            return out;
        }
        const std::size_t overlap_cols = std::min(cols, in_size);
        if (overlap_cols > 0) {
            std::vector<float> out(std::min(out_dim, rows), 0.0f);
            for (std::size_t r = 0; r < out.size(); ++r) {
                float acc = 0.0f;
                const std::size_t row_off = r * cols;
                for (std::size_t c = 0; c < overlap_cols; ++c) {
                    acc += bf16_to_float(bf16[row_off + c]) * input[c];
                }
                out[r] = acc;
            }
            return out;
        }
        const std::size_t overlap_rows = std::min(rows, in_size);
        if (overlap_rows > 0) {
            std::vector<float> out(std::min(out_dim, cols), 0.0f);
            for (std::size_t c = 0; c < out.size(); ++c) {
                float acc = 0.0f;
                for (std::size_t r = 0; r < overlap_rows; ++r) {
                    acc += bf16_to_float(bf16[r * cols + c]) * input[r];
                }
                out[c] = acc;
            }
            return out;
        }
        throw std::runtime_error("attention projection shape does not overlap input width");
    };

    const auto direct_q = find_attention_key_optional(cache.weight_map, layer_i, projection == "o" ? "o_proj" : std::string(projection) + "_proj");
    if (!direct_q.empty()) {
        return matvec_bf16(direct_q, target_dim);
    }

    // Helper: low-rank two-stage matvec on BF16-cached tensors (input → latent → output).
    const auto matvec_lowrank_bf16 = [&](const std::string& a_key, const std::string& b_key) -> std::vector<float> {
        std::vector<std::size_t> a_shape;
        std::vector<std::size_t> b_shape;
        load_named_tensor_cached(artifact, a_key, &a_shape);
        load_named_tensor_cached(artifact, b_key, &b_shape);
        const auto& a_bf16 = cache.tensor_cache.at(a_key);
        const auto& b_bf16 = cache.tensor_cache.at(b_key);
        const std::size_t rank = a_shape.at(0);
        const std::size_t a_cols = a_shape.at(1);
        const std::size_t b_rows = b_shape.at(0);
        const std::size_t b_cols = b_shape.at(1);
        std::vector<float> tmp(rank, 0.0f);
        for (std::size_t r = 0; r < rank; ++r) {
            float acc = 0.0f;
            const std::size_t row_off = r * a_cols;
            for (std::size_t c = 0; c < a_cols && c < input.size(); ++c) {
                acc += bf16_to_float(a_bf16[row_off + c]) * input[c];
            }
            tmp[r] = acc;
        }
        std::vector<float> out(std::min(target_dim, b_rows), 0.0f);
        for (std::size_t r = 0; r < out.size(); ++r) {
            float acc = 0.0f;
            const std::size_t row_off = r * b_cols;
            for (std::size_t c = 0; c < std::min(b_cols, rank); ++c) {
                acc += bf16_to_float(b_bf16[row_off + c]) * tmp[c];
            }
            out[r] = acc;
        }
        return out;
    };

    if (projection == "q" || projection == "k" || projection == "v") {
        const auto q_a_key = find_attention_key_optional(cache.weight_map, layer_i, "q_a_proj");
        const auto q_b_key = find_attention_key_optional(cache.weight_map, layer_i, "q_b_proj");
        const auto kv_a_key = find_attention_key_optional(cache.weight_map, layer_i, "kv_a_proj_with_mqa");
        const auto kv_b_key = find_attention_key_optional(cache.weight_map, layer_i, "kv_b_proj");
        if (projection == "q" && !q_a_key.empty() && !q_b_key.empty()) {
            return matvec_lowrank_bf16(q_a_key, q_b_key);
        }
        if ((projection == "k" || projection == "v") && !kv_a_key.empty() && !kv_b_key.empty()) {
            return matvec_lowrank_bf16(kv_a_key, kv_b_key);
        }

        const auto linear_qkv_key = find_linear_attn_key_optional(cache.weight_map, layer_i, "in_proj_qkv");
        if (!linear_qkv_key.empty()) {
            std::vector<std::size_t> shape;
            load_named_tensor_cached(artifact, linear_qkv_key, &shape);
            const auto& qkv_bf16 = cache.tensor_cache.at(linear_qkv_key);
            const std::size_t rows = shape.at(0);
            const std::size_t cols = shape.at(1);
            const std::size_t chunk = std::max<std::size_t>(1, rows / 3);
            std::size_t start = 0;
            if (projection == "k") {
                start = chunk;
            } else if (projection == "v") {
                start = std::min(rows, 2 * chunk);
            }
            const std::size_t end = projection == "q" ? std::min(rows, chunk)
                : (projection == "k" ? std::min(rows, 2 * chunk) : rows);
            std::vector<float> out(std::min(target_dim, end - start), 0.0f);
            for (std::size_t r = 0; r < out.size(); ++r) {
                float acc = 0.0f;
                const std::size_t row_off = (start + r) * cols;
                for (std::size_t c = 0; c < cols && c < input.size(); ++c) {
                    acc += bf16_to_float(qkv_bf16[row_off + c]) * input[c];
                }
                out[r] = acc;
            }
            return out;
        }
    }

    if (projection == "o") {
        const auto linear_out_key = find_linear_attn_key_optional(cache.weight_map, layer_i, "out_proj");
        if (!linear_out_key.empty()) {
            return matvec_bf16(linear_out_key, target_dim);
        }
    }

    throw std::runtime_error("missing supported attention projection for layer " + std::to_string(layer) + " projection " + std::string(projection));
}

void apply_shared_expert_from_model_dir(
    const DenseRuntimeArtifact& artifact,
    std::size_t layer,
    const std::vector<float>& moe_input,
    std::vector<float>& hidden) {
    if (!artifact.uses_direct_model()) {
        return;
    }
    auto& cache = runtime_cache_for_model_dir(artifact.model_dir);
    const auto layer_i = static_cast<int>(layer);
    const auto gate_key = find_shared_expert_key_optional(cache.weight_map, layer_i, "gate_proj");
    const auto up_key = find_shared_expert_key_optional(cache.weight_map, layer_i, "up_proj");
    const auto down_key = find_shared_expert_key_optional(cache.weight_map, layer_i, "down_proj");
    if (gate_key.empty() || up_key.empty() || down_key.empty()) {
        return;
    }

    std::vector<std::size_t> gate_shape;
    std::vector<std::size_t> up_shape;
    std::vector<std::size_t> down_shape;
    // Prime the BF16 cache; avoid decoding full float32 matrices (saves ~3× memory bandwidth).
    load_named_tensor_cached(artifact, gate_key, &gate_shape);
    load_named_tensor_cached(artifact, up_key, &up_shape);
    load_named_tensor_cached(artifact, down_key, &down_shape);
    if (gate_shape.size() != 2 || up_shape.size() != 2 || down_shape.size() != 2) {
        throw std::runtime_error("shared expert tensors must be rank-2");
    }
    const std::size_t inter = gate_shape[0];
    const std::size_t in_dim = gate_shape[1];
    if (up_shape[0] != inter || up_shape[1] != in_dim || down_shape[1] != inter) {
        throw std::runtime_error("shared expert tensor shapes do not align");
    }
    const auto& gate_bf16 = cache.tensor_cache.at(gate_key);
    const auto& up_bf16 = cache.tensor_cache.at(up_key);
    const auto& down_bf16 = cache.tensor_cache.at(down_key);
    // Shared expert reads from moe_input (post-attn-norm), accumulates into hidden (residual).
    const std::size_t input_len = std::min(in_dim, moe_input.size());
    std::vector<float> gate_out(inter, 0.0f);
    for (std::size_t r = 0; r < inter; ++r) {
        float g = 0.0f;
        float u = 0.0f;
        const std::size_t row_off = r * in_dim;
        for (std::size_t c = 0; c < input_len; ++c) {
            const float inp = finite_or_zero(moe_input[c]);
            g += finite_or_zero(bf16_to_float(gate_bf16[row_off + c])) * inp;
            u += finite_or_zero(bf16_to_float(up_bf16[row_off + c])) * inp;
        }
        gate_out[r] = finite_or_zero(stable_silu(g) * u);
    }
    const std::size_t down_rows = std::min(hidden.size(), down_shape[0]);
    for (std::size_t r = 0; r < down_rows; ++r) {
        float acc = 0.0f;
        const std::size_t row_off = r * down_shape[1];
        for (std::size_t c = 0; c < inter; ++c) {
            acc += finite_or_zero(bf16_to_float(down_bf16[row_off + c])) * finite_or_zero(gate_out[c]);
        }
        hidden[r] = finite_or_zero(hidden[r] + acc);
    }
}

}  // namespace flashmoe
