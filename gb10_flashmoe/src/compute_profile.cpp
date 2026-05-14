#include "flashmoe/compute_profile.h"

#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace flashmoe {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open compute profile: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

ComputeProfile ComputeProfile::from_json_file(const std::string& path) {
    const std::string text = read_file(path);
    ComputeProfile profile;

    {
        const std::regex backend_re(R"json("backend"\s*:\s*"([^"]+)")json");
        std::smatch match;
        if (std::regex_search(text, match, backend_re)) {
            profile.backend_name_ = match[1].str();
        }
    }

    const std::regex point_re(
        R"(\{[^{}]*"active_experts"\s*:\s*([0-9]+)[^{}]*"avg_ms"\s*:\s*([0-9]+(?:\.[0-9]+)?)[^{}]*\})");
    for (std::sregex_iterator it(text.begin(), text.end(), point_re), end; it != end; ++it) {
        profile.points_.push_back(ComputeProfilePoint{
            .active_experts = static_cast<std::size_t>(std::stoull((*it)[1].str())),
            .avg_ms = std::stod((*it)[2].str()),
        });
    }

    std::sort(
        profile.points_.begin(),
        profile.points_.end(),
        [](const ComputeProfilePoint& lhs, const ComputeProfilePoint& rhs) {
            return lhs.active_experts < rhs.active_experts;
        });
    return profile;
}

bool ComputeProfile::empty() const noexcept {
    return points_.empty();
}

double ComputeProfile::estimate_ms(std::size_t active_experts, double fallback_ms) const {
    if (points_.empty()) {
        return fallback_ms;
    }
    auto it = std::lower_bound(
        points_.begin(),
        points_.end(),
        active_experts,
        [](const ComputeProfilePoint& point, std::size_t value) {
            return point.active_experts < value;
        });
    if (it == points_.begin()) {
        return it->avg_ms;
    }
    if (it == points_.end()) {
        return points_.back().avg_ms;
    }
    if (it->active_experts == active_experts) {
        return it->avg_ms;
    }

    const auto& upper = *it;
    const auto& lower = *(it - 1);
    const double span = static_cast<double>(upper.active_experts - lower.active_experts);
    if (span <= 0.0) {
        return upper.avg_ms;
    }
    const double ratio = static_cast<double>(active_experts - lower.active_experts) / span;
    return lower.avg_ms + ratio * (upper.avg_ms - lower.avg_ms);
}

std::string_view ComputeProfile::backend_name() const noexcept {
    return backend_name_;
}

}  // namespace flashmoe
