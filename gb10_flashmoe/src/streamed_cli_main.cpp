#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "flashmoe/model_spec.h"
#include "flashmoe/streamed_engine.h"

namespace {

std::string slurp_stdin() {
    std::ostringstream out;
    out << std::cin.rdbuf();
    return out.str();
}

void print_usage() {
    std::cerr
        << "Usage: flashmoe_streamed_cli MODEL_FAMILY expert_manifest.json [routing_trace.txt] [compute_profile.json] [cache_budget_gb]\n"
        << "       [--runtime-router] [--cpu-expert-backend] [--model-path PATH] [--dense-artifact PATH] [--tokenizer-artifact PATH]\n"
        << "       [--prompt TEXT | --prompt-stdin] [--max-tokens N] [--interactive]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return EXIT_FAILURE;
    }

    const auto family = flashmoe::parse_model_family(argv[1]);
    if (family == flashmoe::ModelFamily::kUnknown) {
        std::cerr << "unknown model family: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    flashmoe::StreamedEngineConfig config;
    config.family = family;
    config.manifest_path = argv[2];
    config.trace_path = argc >= 4 ? std::string(argv[3]) : std::string{};
    if (argc >= 5 && std::string_view(argv[4]).size() > 0 && std::string_view(argv[4]) != "--runtime-router") {
        config.compute_profile_path = std::string(argv[4]);
    }
    if (argc >= 6 && std::string_view(argv[5]).size() > 0 && std::string_view(argv[5]) != "--runtime-router") {
        config.hot_cache_budget_gb = std::stod(argv[5]);
    }

    bool interactive = false;
    bool prompt_stdin = false;
    std::size_t max_tokens = 32;
    std::string prompt;

    for (int i = 3; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--runtime-router") {
            config.use_runtime_router = true;
        } else if (arg == "--cpu-expert-backend") {
            config.prefer_cuda_expert_backend = false;
        } else if (arg == "--model-path" && i + 1 < argc) {
            config.model_path = std::string(argv[++i]);
        } else if (arg == "--dense-artifact" && i + 1 < argc) {
            config.dense_artifact_path = std::string(argv[++i]);
        } else if (arg == "--tokenizer-artifact" && i + 1 < argc) {
            config.tokenizer_artifact_path = std::string(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--prompt-stdin") {
            prompt_stdin = true;
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--interactive") {
            interactive = true;
        }
    }

    if (!config.use_runtime_router && config.trace_path.empty()) {
        std::cerr << "routing_trace.txt is required unless --runtime-router is set\n";
        return EXIT_FAILURE;
    }

    try {
        auto engine = flashmoe::StreamedEngine::load(config);

        if (interactive) {
            auto session = engine.create_session(prompt_stdin ? slurp_stdin() : prompt);
            if (!prompt.empty() || prompt_stdin) {
                const auto result = session.generate(max_tokens);
                std::cout << result.text << "\n";
            }
            std::string line;
            while (true) {
                std::cout << "> " << std::flush;
                if (!std::getline(std::cin, line)) {
                    break;
                }
                if (line == "/exit" || line == "/quit") {
                    break;
                }
                session.append_user_turn(line);
                const auto result = session.generate(max_tokens);
                std::cout << result.text << "\n";
            }
            return EXIT_SUCCESS;
        }

        if (prompt_stdin) {
            prompt = slurp_stdin();
        }
        auto session = engine.create_session(prompt);
        const auto result = session.generate(max_tokens);
        std::cout << result.text << "\n";
        std::cerr << "prompt_tokens=" << result.prompt_tokens
                  << " prefill_tokens=" << result.prefill_tokens
                  << " completion_tokens=" << result.completion_tokens
                  << " tok_per_s=" << result.tok_per_s
                  << " resident_slot_gb=" << result.resident_slot_gb
                  << "\n";
    } catch (const std::exception& exc) {
        std::cerr << "streamed cli failed: " << exc.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
