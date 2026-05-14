#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace flashmoe {

struct ComputeProfilePoint {
    std::size_t active_experts = 0;
    double avg_ms = 0.0;
};

class ComputeProfile {
public:
    static ComputeProfile from_json_file(const std::string& path);

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] double estimate_ms(std::size_t active_experts, double fallback_ms) const;
    [[nodiscard]] std::string_view backend_name() const noexcept;

private:
    std::string backend_name_;
    std::vector<ComputeProfilePoint> points_;
};

}  // namespace flashmoe
