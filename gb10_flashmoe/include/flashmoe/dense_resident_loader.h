#pragma once

#include <string_view>
#include <vector>

#include "flashmoe/model_spec.h"
#include "flashmoe/runtime_plan.h"

namespace flashmoe {

struct ResidentTensorGroup {
    std::string_view name;
    double estimated_gb = 0.0;
    bool always_resident = true;
};

struct DenseResidentPlan {
    double total_resident_gb = 0.0;
    std::vector<ResidentTensorGroup> resident_groups;
    std::vector<std::string_view> excluded_routed_groups;
};

DenseResidentPlan build_dense_resident_plan(const ModelSpec& spec, const RuntimePlan& plan);

}  // namespace flashmoe
