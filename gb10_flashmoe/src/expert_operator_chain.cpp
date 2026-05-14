#include "flashmoe/expert_operator_chain.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace flashmoe {
namespace {

using Clock = std::chrono::steady_clock;

struct TensorView {
    std::string dtype;
    std::vector<std::size_t> shape;
    std::size_t data_begin = 0;
    std::size_t data_end = 0;
};

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

MaterializedExpert& ensure_slot(const ExpertRecord& record, MaterializedExpertMap& materialized) {
    return materialized.try_emplace(record.id).first->second;
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

std::vector<float> decode_float_tensor(const std::vector<std::uint8_t>& bytes, const TensorView& tensor) {
    const std::size_t count = std::accumulate(
        tensor.shape.begin(), tensor.shape.end(), static_cast<std::size_t>(1), std::multiplies<>{});
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
    throw std::runtime_error("unsupported dense tensor dtype: " + tensor.dtype);
}

std::vector<std::int32_t> decode_i32_tensor(const std::vector<std::uint8_t>& bytes, const TensorView& tensor) {
    const std::size_t count = std::accumulate(
        tensor.shape.begin(), tensor.shape.end(), static_cast<std::size_t>(1), std::multiplies<>{});
    if (tensor.dtype != "I32") {
        throw std::runtime_error("expected I32 tensor, got: " + tensor.dtype);
    }
    std::vector<std::int32_t> out(count, 0);
    std::memcpy(out.data(), bytes.data() + tensor.data_begin, count * sizeof(std::int32_t));
    return out;
}

std::vector<std::uint8_t> decode_u8_tensor(const std::vector<std::uint8_t>& bytes, const TensorView& tensor) {
    const std::size_t count = std::accumulate(
        tensor.shape.begin(), tensor.shape.end(), static_cast<std::size_t>(1), std::multiplies<>{});
    if (tensor.dtype != "U8") {
        throw std::runtime_error("expected U8 tensor, got: " + tensor.dtype);
    }
    std::vector<std::uint8_t> out(count, 0);
    std::memcpy(out.data(), bytes.data() + tensor.data_begin, count);
    return out;
}

TensorView parse_tensor(const std::vector<std::uint8_t>& bytes, const std::string& name) {
    if (bytes.size() < sizeof(std::uint64_t)) {
        throw std::runtime_error("expert bundle too small");
    }
    const auto header_len = static_cast<std::size_t>(read_u64_le(bytes, 0));
    const auto header_begin = sizeof(std::uint64_t);
    const auto header_end = header_begin + header_len;
    if (header_end > bytes.size()) {
        throw std::runtime_error("expert bundle header exceeds file size");
    }
    const std::string header(
        reinterpret_cast<const char*>(bytes.data() + header_begin),
        reinterpret_cast<const char*>(bytes.data() + header_end));
    const std::string escaped_name = std::regex_replace(name, std::regex(R"([.^$|()\\[*+?{\]])"), R"(\$&)");
    const std::regex pattern(
        "\\\"" + escaped_name +
        "\\\"\\s*:\\s*\\{\\s*\\\"dtype\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"\\s*,\\s*\\\"shape\\\"\\s*:\\s*\\[([^\\]]*)\\]\\s*,\\s*\\\"data_offsets\\\"\\s*:\\s*\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    std::smatch match;
    if (!std::regex_search(header, match, pattern)) {
        throw std::runtime_error("tensor not found in expert bundle: " + name);
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
        tensor.shape.push_back(static_cast<std::size_t>(std::stoull(item)));
    }
    const auto relative_begin = static_cast<std::size_t>(std::stoull(match[3].str()));
    const auto relative_end = static_cast<std::size_t>(std::stoull(match[4].str()));
    tensor.data_begin = header_end + relative_begin;
    tensor.data_end = header_end + relative_end;
    if (tensor.data_end > bytes.size() || tensor.data_end < tensor.data_begin) {
        throw std::runtime_error("tensor data offsets out of range: " + name);
    }
    return tensor;
}

std::vector<float> unpack_q3_weight(const std::vector<std::uint8_t>& qweight,
                                    const std::vector<float>& scales,
                                    const std::vector<std::int32_t>& shape) {
    if (shape.size() < 2) {
        throw std::runtime_error("q3like tensor shape must be at least 2D");
    }
    const std::size_t rows = static_cast<std::size_t>(shape[0]);
    std::size_t cols = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        cols *= static_cast<std::size_t>(shape[i]);
    }
    if (rows != scales.size()) {
        throw std::runtime_error("q3like scale rows mismatch");
    }

    const std::size_t count = rows * cols;
    std::vector<float> out(count, 0.0f);
    std::size_t out_index = 0;
    for (std::size_t byte_index = 0; byte_index + 2 < qweight.size() && out_index < count; byte_index += 3) {
        const std::uint32_t packed =
            static_cast<std::uint32_t>(qweight[byte_index]) |
            (static_cast<std::uint32_t>(qweight[byte_index + 1]) << 8U) |
            (static_cast<std::uint32_t>(qweight[byte_index + 2]) << 16U);
        for (std::size_t group = 0; group < 8 && out_index < count; ++group) {
            const std::uint32_t value = (packed >> (group * 3U)) & 0x7U;
            const std::int32_t quant = static_cast<std::int32_t>(value) - 4;
            const std::size_t row = out_index / cols;
            out[out_index++] = static_cast<float>(quant) * scales[row];
        }
    }
    return out;
}

std::vector<float> load_q3_tensor(const std::vector<std::uint8_t>& bytes, const std::string& prefix) {
    const auto qweight = decode_u8_tensor(bytes, parse_tensor(bytes, prefix + ".qweight"));
    const auto scale = decode_float_tensor(bytes, parse_tensor(bytes, prefix + ".scale"));
    const auto shape = decode_i32_tensor(bytes, parse_tensor(bytes, prefix + ".shape"));
    return unpack_q3_weight(qweight, scale, shape);
}

std::vector<float> load_dense_tensor(const std::vector<std::uint8_t>& bytes, const std::string& name) {
    return decode_float_tensor(bytes, parse_tensor(bytes, name));
}

void unpack_real_expert(const ExpertRecord& record, MaterializedExpert& slot) {
    if (record.format == ExpertStorageFormat::kQ3Like) {
        slot.gate_weights = load_q3_tensor(slot.bytes, "gate_proj");
        slot.up_weights = load_q3_tensor(slot.bytes, "up_proj");
        slot.down_weights = load_q3_tensor(slot.bytes, "down_proj");
        const auto gate_shape = decode_i32_tensor(slot.bytes, parse_tensor(slot.bytes, "gate_proj.shape"));
        const auto down_shape = decode_i32_tensor(slot.bytes, parse_tensor(slot.bytes, "down_proj.shape"));
        slot.intermediate_dim = static_cast<std::size_t>(gate_shape[0]);
        slot.hidden_dim = static_cast<std::size_t>(down_shape[0]);
        return;
    }
    if (record.format == ExpertStorageFormat::kDense) {
        const auto gate_view = parse_tensor(slot.bytes, "gate_proj.weight");
        const auto down_view = parse_tensor(slot.bytes, "down_proj.weight");
        slot.gate_weights = load_dense_tensor(slot.bytes, "gate_proj.weight");
        slot.up_weights = load_dense_tensor(slot.bytes, "up_proj.weight");
        slot.down_weights = load_dense_tensor(slot.bytes, "down_proj.weight");
        slot.intermediate_dim = gate_view.shape.at(0);
        slot.hidden_dim = down_view.shape.at(0);
        return;
    }
    throw std::runtime_error("unsupported expert storage format");
}

float silu(float x) {
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

float finite_or_zero(float value) {
    return std::isfinite(value) ? value : 0.0f;
}

}  // namespace

double materialize_expert_record(const ExpertRecord& record,
                                 MaterializedExpertMap& materialized) {
    auto& slot = ensure_slot(record, materialized);
    if (!slot.bytes.empty()) {
        return 0.0;
    }

    const auto start = Clock::now();
    std::ifstream input(record.path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open expert file: " + record.path);
    }
    input.seekg(static_cast<std::streamoff>(record.offset), std::ios::beg);
    slot.bytes.resize(record.bytes);
    input.read(reinterpret_cast<char*>(slot.bytes.data()), static_cast<std::streamsize>(slot.bytes.size()));
    const auto read_bytes = static_cast<std::size_t>(input.gcount());
    slot.bytes.resize(read_bytes);
    const auto end = Clock::now();
    return elapsed_ms(start, end);
}

double unpack_materialized_expert(const ExpertRecord& record,
                                  MaterializedExpertMap& materialized) {
    auto& slot = ensure_slot(record, materialized);
    if (slot.host_ready) {
        return 0.0;
    }

    const auto start = Clock::now();
    unpack_real_expert(record, slot);
    slot.host_ready = true;
    const auto end = Clock::now();
    return elapsed_ms(start, end);
}

double execute_materialized_expert_fused(const ExpertRecord& record,
                                         DenseOperatorState& dense_state,
                                         const DenseRuntimeArtifact*,
                                         MaterializedExpertMap& materialized,
                                         float route_weight) {
    auto& slot = ensure_slot(record, materialized);
    if (!slot.host_ready) {
        throw std::runtime_error("expert weights must be unpacked before execution");
    }
    if (slot.hidden_dim != dense_state.hidden.size()) {
        throw std::runtime_error("expert hidden dim does not match runtime hidden dim");
    }

    // FFN reads from moe_input (post-attn-norm), accumulates into hidden (residual).
    const auto& ffn_input = (dense_state.moe_input.size() == slot.hidden_dim)
        ? dense_state.moe_input : dense_state.hidden;

    const auto start = Clock::now();
    std::vector<float> gate(slot.intermediate_dim, 0.0f);
    std::vector<float> up(slot.intermediate_dim, 0.0f);
    std::vector<float> activated(slot.intermediate_dim, 0.0f);
    std::vector<float> out(slot.hidden_dim, 0.0f);

    for (std::size_t row = 0; row < slot.intermediate_dim; ++row) {
        float gate_accum = 0.0f;
        float up_accum = 0.0f;
        const std::size_t row_offset = row * slot.hidden_dim;
        for (std::size_t col = 0; col < slot.hidden_dim; ++col) {
            const float inp = finite_or_zero(ffn_input[col]);
            gate_accum += finite_or_zero(slot.gate_weights[row_offset + col]) * inp;
            up_accum += finite_or_zero(slot.up_weights[row_offset + col]) * inp;
        }
        gate[row] = finite_or_zero(gate_accum);
        up[row] = finite_or_zero(up_accum);
        activated[row] = finite_or_zero(silu(gate[row]) * up[row]);
    }

    for (std::size_t row = 0; row < slot.hidden_dim; ++row) {
        float accum = 0.0f;
        const std::size_t row_offset = row * slot.intermediate_dim;
        for (std::size_t col = 0; col < slot.intermediate_dim; ++col) {
            accum += finite_or_zero(slot.down_weights[row_offset + col]) * finite_or_zero(activated[col]);
        }
        out[row] = finite_or_zero(accum);
    }

    const float safe_weight = std::isfinite(route_weight) ? route_weight : 0.0f;
    for (std::size_t i = 0; i < dense_state.hidden.size(); ++i) {
        dense_state.hidden[i] = finite_or_zero(dense_state.hidden[i] + safe_weight * finite_or_zero(out[i]));
    }

    const auto end = Clock::now();
    return elapsed_ms(start, end);
}

void release_materialized_expert(MaterializedExpert& slot) {
    slot.bytes.clear();
    slot.bytes.shrink_to_fit();
    slot.gate_weights.clear();
    slot.gate_weights.shrink_to_fit();
    slot.up_weights.clear();
    slot.up_weights.shrink_to_fit();
    slot.down_weights.clear();
    slot.down_weights.shrink_to_fit();
    slot.hidden_dim = 0;
    slot.intermediate_dim = 0;
    slot.host_ready = false;
    slot.device_ready = false;
    slot.device_gate = 0;
    slot.device_up = 0;
    slot.device_down = 0;
    slot.device_gate_len = 0;
    slot.device_up_len = 0;
    slot.device_down_len = 0;
}

}  // namespace flashmoe
