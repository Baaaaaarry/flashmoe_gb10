#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "flashmoe/expert_types.h"
#include "flashmoe/model_spec.h"

namespace flashmoe {

struct ExpertRecord {
    ExpertId id{};
    std::string path;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    ExpertStorageFormat format = ExpertStorageFormat::kDense;
};

class ExpertManifestStore {
public:
    static ExpertManifestStore from_json_file(const std::string& path);

    [[nodiscard]] bool contains(ExpertId id) const;
    [[nodiscard]] const ExpertRecord* find(ExpertId id) const;
    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] std::vector<ExpertRecord> layer_records(std::uint16_t layer) const;

private:
    std::vector<ExpertRecord> records_;
};

}  // namespace flashmoe
