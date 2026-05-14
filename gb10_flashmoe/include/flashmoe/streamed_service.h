#pragma once

#include <mutex>
#include <string_view>
#include <unordered_map>
#include <cstdint>
#include <string>
#include <memory>

#include "flashmoe/streamed_engine.h"

namespace flashmoe {

struct ServiceConfig {
    StreamedEngineConfig engine;
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
};

class StreamedService {
public:
    explicit StreamedService(ServiceConfig config);

    void run();

private:
    std::string handle_request(std::string_view request);
    std::string handle_healthz() const;
    std::string handle_models() const;
    std::string handle_create_session(std::string_view body);
    std::string handle_get_session(std::string_view path) const;
    std::string handle_chat_completions(std::string_view body);

    ServiceConfig config_;
    StreamedEngine engine_;
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, std::unique_ptr<StreamedSession>> sessions_;
};

}  // namespace flashmoe
