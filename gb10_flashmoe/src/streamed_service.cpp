#include "flashmoe/streamed_service.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <sys/socket.h>
#include <unistd.h>

namespace flashmoe {
namespace {

std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
}

std::string body_from_request(std::string_view request) {
    const auto pos = request.find("\r\n\r\n");
    if (pos == std::string_view::npos) {
        return {};
    }
    return std::string(request.substr(pos + 4));
}

std::string request_path(std::string_view request) {
    const auto line_end = request.find("\r\n");
    const auto line = request.substr(0, line_end);
    const auto first_sp = line.find(' ');
    if (first_sp == std::string_view::npos) {
        return {};
    }
    const auto second_sp = line.find(' ', first_sp + 1);
    if (second_sp == std::string_view::npos) {
        return {};
    }
    return std::string(line.substr(first_sp + 1, second_sp - first_sp - 1));
}

std::string request_method(std::string_view request) {
    const auto first_sp = request.find(' ');
    if (first_sp == std::string_view::npos) {
        return {};
    }
    return std::string(request.substr(0, first_sp));
}

std::size_t parse_max_tokens(std::string_view body) {
    const std::regex pattern(R"json("max_tokens"\s*:\s*([0-9]+))json");
    std::match_results<std::string_view::const_iterator> match;
    if (std::regex_search(body.begin(), body.end(), match, pattern)) {
        return static_cast<std::size_t>(std::stoull(match[1].str()));
    }
    return 16;
}

std::string parse_last_content(std::string_view body) {
    const std::regex pattern(R"json("content"\s*:\s*"([^"]*)")json");
    std::match_results<std::string_view::const_iterator> match;
    std::string last;
    auto begin = body.begin();
    while (std::regex_search(begin, body.end(), match, pattern)) {
        last = match[1].str();
        begin = match.suffix().first;
    }
    return last;
}

using ChatMessage = std::pair<std::string, std::string>;

std::vector<ChatMessage> parse_messages(std::string_view body) {
    const std::regex pattern(
        R"json(\{\s*"role"\s*:\s*"([^"]+)"\s*,\s*"content"\s*:\s*"([^"]*)"\s*\})json");
    std::vector<ChatMessage> messages;
    std::match_results<std::string_view::const_iterator> match;
    auto begin = body.begin();
    while (std::regex_search(begin, body.end(), match, pattern)) {
        messages.emplace_back(match[1].str(), match[2].str());
        begin = match.suffix().first;
    }
    return messages;
}

std::string render_chat_prompt(const std::vector<ChatMessage>& messages) {
    std::ostringstream out;
    for (const auto& [role, content] : messages) {
        if (content.empty()) {
            continue;
        }
        if (out.tellp() > 0) {
            out << "\n";
        }
        out << "<|" << role << "|>\n" << content;
    }
    return out.str();
}

std::string http_ok_json(std::string_view body) {
    std::ostringstream out;
    out << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return out.str();
}

std::string http_not_found() {
    static constexpr std::string_view body = R"({"error":"not_found"})";
    std::ostringstream out;
    out << "HTTP/1.1 404 Not Found\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return out.str();
}

bool parse_stream(std::string_view body) {
    const std::regex pattern(R"json("stream"\s*:\s*(true|false))json");
    std::match_results<std::string_view::const_iterator> match;
    if (std::regex_search(body.begin(), body.end(), match, pattern)) {
        return match[1].str() == "true";
    }
    return false;
}

std::optional<std::uint64_t> parse_session_id(std::string_view body) {
    const std::regex pattern(R"json("session_id"\s*:\s*([0-9]+))json");
    std::match_results<std::string_view::const_iterator> match;
    if (std::regex_search(body.begin(), body.end(), match, pattern)) {
        return static_cast<std::uint64_t>(std::stoull(match[1].str()));
    }
    return std::nullopt;
}

std::string sse_wrap(std::string_view payload) {
    std::ostringstream out;
    out << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: text/event-stream\r\n"
        << "Cache-Control: no-cache\r\n"
        << "Connection: close\r\n\r\n"
        << "data: " << payload << "\n\n"
        << "data: [DONE]\n\n";
    return out.str();
}

struct StepBreakdown {
    double embed_ms = 0.0;
    double attention_ms = 0.0;
    double norm_router_ms = 0.0;
    double lm_head_ms = 0.0;
    double route_ms = 0.0;
    double load_ms = 0.0;
    double unpack_ms = 0.0;
    double compute_ms = 0.0;
    double combine_ms = 0.0;
};

StepBreakdown aggregate_breakdown(const std::vector<SessionStepStats>& token_stats) {
    StepBreakdown total;
    for (const auto& step : token_stats) {
        total.embed_ms += step.embed_ms;
        total.attention_ms += step.attention_ms;
        total.norm_router_ms += step.norm_router_ms;
        total.lm_head_ms += step.lm_head_ms;
        total.route_ms += step.route_ms;
        total.load_ms += step.load_ms;
        total.unpack_ms += step.unpack_ms;
        total.compute_ms += step.compute_ms;
        total.combine_ms += step.combine_ms;
    }
    return total;
}

std::string json_array_of_token_ids(const std::vector<std::uint32_t>& token_ids) {
    std::ostringstream out;
    out << "[";
    for (std::size_t i = 0; i < token_ids.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << token_ids[i];
    }
    out << "]";
    return out.str();
}

}  // namespace

StreamedService::StreamedService(ServiceConfig config)
    : config_(std::move(config)),
      engine_(StreamedEngine::load(config_.engine)) {}

void StreamedService::run() {
    const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        throw std::runtime_error("failed to create socket");
    }

    int enable = 1;
    ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(config_.port);
    if (::inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) != 1) {
        ::close(server_fd);
        throw std::runtime_error("invalid bind address");
    }

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(server_fd);
        throw std::runtime_error("failed to bind service socket");
    }
    if (::listen(server_fd, 16) != 0) {
        ::close(server_fd);
        throw std::runtime_error("failed to listen on service socket");
    }

    while (true) {
        const int client_fd = ::accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            continue;
        }

        char buffer[65536];
        const ssize_t received = ::read(client_fd, buffer, sizeof(buffer));
        if (received > 0) {
            const std::string response = handle_request(std::string_view(buffer, static_cast<std::size_t>(received)));
            const char* data = response.data();
            std::size_t remaining = response.size();
            while (remaining > 0) {
                const ssize_t written = ::write(client_fd, data, remaining);
                if (written <= 0) {
                    break;
                }
                data += written;
                remaining -= static_cast<std::size_t>(written);
            }
        }
        ::close(client_fd);
    }
}

std::string StreamedService::handle_request(std::string_view request) {
    const std::string method = request_method(request);
    const std::string path = request_path(request);
    if (method == "GET" && path == "/healthz") {
        return http_ok_json(handle_healthz());
    }
    if (method == "GET" && path == "/v1/models") {
        return http_ok_json(handle_models());
    }
    if (method == "POST" && path == "/v1/sessions") {
        return http_ok_json(handle_create_session(body_from_request(request)));
    }
    if (method == "GET" && path.rfind("/v1/sessions/", 0) == 0) {
        return http_ok_json(handle_get_session(path));
    }
    if (method == "POST" && path == "/v1/chat/completions") {
        return handle_chat_completions(body_from_request(request));
    }
    return http_not_found();
}

std::string StreamedService::handle_healthz() const {
    std::ostringstream out;
    out << "{"
        << "\"ok\":true,"
        << "\"model\":\"" << engine_.spec().short_name << "\","
        << "\"runtime_family\":\"" << engine_.plan().runtime_family << "\","
        << "\"route_source\":\"" << (engine_.use_runtime_router() ? "runtime_router" : "trace") << "\","
        << "\"hot_cache_budget_gb\":" << engine_.plan().hot_expert_cache_budget_gb
        << "}";
    return out.str();
}

std::string StreamedService::handle_models() const {
    std::ostringstream out;
    out << "{"
        << "\"object\":\"list\","
        << "\"data\":[{"
        << "\"id\":\"" << engine_.spec().short_name << "\","
        << "\"object\":\"model\","
        << "\"owned_by\":\"flashmoe\""
        << "}]"
        << "}";
    return out.str();
}

std::string StreamedService::handle_create_session(std::string_view body) {
    auto messages = parse_messages(body);
    std::string prompt = render_chat_prompt(messages);
    if (prompt.empty()) {
        prompt = parse_last_content(body);
    }
    auto session = std::make_unique<StreamedSession>(engine_.create_session(prompt));
    const auto id = session->session_id();
    const auto prompt_tokens = session->prompt_tokens();
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_[id] = std::move(session);

    std::ostringstream out;
    out << "{"
        << "\"session_id\":" << id << ","
        << "\"model\":\"" << engine_.spec().short_name << "\","
        << "\"prompt_tokens\":" << prompt_tokens
        << "}";
    return out.str();
}

std::string StreamedService::handle_get_session(std::string_view path) const {
    const auto slash = path.find_last_of('/');
    if (slash == std::string_view::npos || slash + 1 >= path.size()) {
        return R"({"error":"invalid_session_path"})";
    }
    const auto id = static_cast<std::uint64_t>(std::stoull(std::string(path.substr(slash + 1))));
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(id);
    if (it == sessions_.end()) {
        return R"({"error":"session_not_found"})";
    }

    std::ostringstream out;
    out << "{"
        << "\"session_id\":" << it->second->session_id() << ","
        << "\"generated_tokens\":" << it->second->generated_tokens() << ","
        << "\"prompt_tokens\":" << it->second->prompt_tokens() << ","
        << "\"prefill_tokens\":" << it->second->prefill_tokens() << ","
        << "\"prefill_trace_steps\":" << it->second->prefill_trace_steps() << ","
        << "\"prefill_total_ms\":" << it->second->prefill_total_ms() << ","
        << "\"kv_cache_used_gb\":" << it->second->kv_cache_used_gb() << ","
        << "\"resident_slot_gb\":" << it->second->resident_slot_gb() << ","
        << "\"model\":\"" << engine_.spec().short_name << "\""
        << "}";
    return out.str();
}

std::string StreamedService::handle_chat_completions(std::string_view body) {
    const std::size_t max_tokens = parse_max_tokens(body);
    auto messages = parse_messages(body);
    std::string prompt = render_chat_prompt(messages);
    if (prompt.empty()) {
        prompt = parse_last_content(body);
    }
    const bool stream = parse_stream(body);
    const auto requested_session = parse_session_id(body);

    SessionGenerateResult result;
    std::uint64_t session_id = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        StreamedSession* session_ptr = nullptr;
        if (requested_session.has_value()) {
            auto it = sessions_.find(*requested_session);
            if (it == sessions_.end()) {
                return http_ok_json(R"({"error":"session_not_found"})");
            }
            if (!prompt.empty()) {
                it->second->append_user_turn(prompt);
            }
            session_ptr = it->second.get();
        } else {
            auto session = std::make_unique<StreamedSession>(engine_.create_session(prompt));
            session_id = session->session_id();
            session_ptr = session.get();
            sessions_[session_id] = std::move(session);
        }

        if (session_id == 0) {
            session_id = session_ptr->session_id();
        }
        result = session_ptr->generate(max_tokens);
    }
    const auto breakdown = aggregate_breakdown(result.token_stats);

    std::ostringstream out;
    out << "{"
        << "\"id\":\"chatcmpl-streamed-" << session_id << "\","
        << "\"object\":\"chat.completion\","
        << "\"model\":\"" << engine_.spec().short_name << "\","
        << "\"route_source\":\"" << result.route_source << "\","
        << "\"session_id\":" << session_id << ","
        << "\"choices\":[{"
        << "\"index\":0,"
        << "\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(result.text) << "\","
        << "\"token_ids\":" << json_array_of_token_ids(result.sampled_token_ids) << "},"
        << "\"finish_reason\":\"" << result.finish_reason << "\""
        << "}],"
        << "\"usage\":{"
        << "\"prompt_tokens\":" << result.prompt_tokens << ","
        << "\"prefill_tokens\":" << result.prefill_tokens << ","
        << "\"completion_tokens\":" << result.completion_tokens << ","
        << "\"total_tokens\":" << result.total_tokens
        << "},"
        << "\"runtime\":{"
        << "\"prefill_trace_steps\":" << result.prefill_trace_steps << ","
        << "\"trace_steps_consumed\":" << result.trace_steps_consumed << ","
        << "\"prefill_total_ms\":" << result.prefill_total_ms << ","
        << "\"total_ms\":" << result.total_ms << ","
        << "\"avg_token_ms\":" << result.avg_token_ms << ","
        << "\"tok_per_s\":" << result.tok_per_s << ","
        << "\"session_generated_tokens\":" << result.session_generated_tokens << ","
        << "\"slot_promotions\":" << result.slot_promotions << ","
        << "\"slot_hits\":" << result.slot_hits << ","
        << "\"slot_evictions\":" << result.slot_evictions << ","
        << "\"resident_slot_gb\":" << result.resident_slot_gb << ","
        << "\"kv_cache_used_gb\":" << result.kv_cache_used_gb << ","
        << "\"embed_ms\":" << breakdown.embed_ms << ","
        << "\"attention_ms\":" << breakdown.attention_ms << ","
        << "\"norm_router_ms\":" << breakdown.norm_router_ms << ","
        << "\"lm_head_ms\":" << breakdown.lm_head_ms << ","
        << "\"route_ms\":" << breakdown.route_ms << ","
        << "\"load_ms\":" << breakdown.load_ms << ","
        << "\"unpack_ms\":" << breakdown.unpack_ms << ","
        << "\"compute_ms\":" << breakdown.compute_ms << ","
        << "\"combine_ms\":" << breakdown.combine_ms
        << "}"
        << "}";
    if (stream) {
        return sse_wrap(out.str());
    }
    return http_ok_json(out.str());
}

}  // namespace flashmoe
