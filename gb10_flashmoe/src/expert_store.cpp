#include "flashmoe/expert_store.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace flashmoe {
namespace {

ExpertStorageFormat parse_format(std::string_view value) {
    if (value == "dense") {
        return ExpertStorageFormat::kDense;
    }
    if (value == "q3like") {
        return ExpertStorageFormat::kQ3Like;
    }
    if (value == "mxfp4") {
        return ExpertStorageFormat::kMxfp4;
    }
    if (value == "iq2xxs") {
        return ExpertStorageFormat::kIq2xxs;
    }
    return ExpertStorageFormat::kDense;
}

std::string read_all(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open manifest: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

ExpertManifestStore ExpertManifestStore::from_json_file(const std::string& path) {
    const std::string raw = read_all(path);
    const std::regex object_pattern(R"json(\{[^{}]*\})json");
    const std::regex layer_pattern(R"json("layer_id"\s*:\s*([0-9]+))json");
    const std::regex expert_pattern(R"json("expert_id"\s*:\s*([0-9]+))json");
    const std::regex path_pattern(R"json("path"\s*:\s*"([^"]+)")json");
    const std::regex offset_pattern(R"json("offset"\s*:\s*([0-9]+))json");
    const std::regex size_pattern(R"json("size_bytes"\s*:\s*([0-9]+))json");
    const std::regex format_pattern(R"json("format"\s*:\s*"([^"]+)")json");

    ExpertManifestStore store;
    std::smatch match;
    auto begin = raw.cbegin();
    bool found = false;
    while (std::regex_search(begin, raw.cend(), match, object_pattern)) {
        const std::string object = match.str();
        std::smatch field;
        ExpertRecord record;
        if (!std::regex_search(object, field, layer_pattern)) {
            begin = match.suffix().first;
            continue;
        }
        record.id.layer = static_cast<std::uint16_t>(std::stoi(field[1].str()));
        if (!std::regex_search(object, field, expert_pattern)) {
            begin = match.suffix().first;
            continue;
        }
        record.id.expert = static_cast<std::uint16_t>(std::stoi(field[1].str()));
        if (!std::regex_search(object, field, path_pattern)) {
            begin = match.suffix().first;
            continue;
        }
        record.path = field[1].str();
        if (!std::regex_search(object, field, offset_pattern)) {
            begin = match.suffix().first;
            continue;
        }
        record.offset = static_cast<std::size_t>(std::stoull(field[1].str()));
        if (!std::regex_search(object, field, size_pattern)) {
            begin = match.suffix().first;
            continue;
        }
        record.bytes = static_cast<std::size_t>(std::stoull(field[1].str()));
        if (std::regex_search(object, field, format_pattern)) {
            record.format = parse_format(field[1].str());
        }
        found = true;
        store.records_.push_back(std::move(record));
        begin = match.suffix().first;
    }

    if (!found) {
        throw std::runtime_error("no manifest entries parsed from: " + path);
    }
    return store;
}

bool ExpertManifestStore::contains(ExpertId id) const {
    return find(id) != nullptr;
}

const ExpertRecord* ExpertManifestStore::find(ExpertId id) const {
    for (const auto& record : records_) {
        if (record.id == id) {
            return &record;
        }
    }
    return nullptr;
}

std::vector<ExpertRecord> ExpertManifestStore::layer_records(std::uint16_t layer) const {
    std::vector<ExpertRecord> out;
    for (const auto& record : records_) {
        if (record.id.layer == layer) {
            out.push_back(record);
        }
    }
    return out;
}

}  // namespace flashmoe
