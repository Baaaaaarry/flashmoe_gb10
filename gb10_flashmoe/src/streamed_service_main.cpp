#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

#include "flashmoe/model_spec.h"
#include "flashmoe/streamed_service.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: flashmoe_streamed_service MODEL_FAMILY expert_manifest.json [routing_trace.txt] [compute_profile.json] [port] [cache_budget_gb] [--runtime-router] [--model-path PATH] [--dense-artifact PATH] [--tokenizer-artifact PATH]\n";
        return EXIT_FAILURE;
    }

    const auto family = flashmoe::parse_model_family(argv[1]);
    if (family == flashmoe::ModelFamily::kUnknown) {
        std::cerr << "unknown model family: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    flashmoe::ServiceConfig config;
    config.engine.family = family;
    config.engine.manifest_path = argv[2];
    config.engine.trace_path = argc >= 4 ? std::string(argv[3]) : std::string{};
    if (config.engine.trace_path == "--runtime-router") {
        config.engine.trace_path.clear();
        config.engine.use_runtime_router = true;
    }
    if (argc >= 5 && std::string_view(argv[4]).size() > 0 && std::string_view(argv[4]) != "--runtime-router") {
        config.engine.compute_profile_path = std::string(argv[4]);
    }
    if (argc >= 6 && std::string_view(argv[5]).size() > 0 && std::string_view(argv[5]) != "--runtime-router") {
        config.port = static_cast<std::uint16_t>(std::stoi(argv[5]));
    }
    if (argc >= 7 && std::string_view(argv[6]).size() > 0 && std::string_view(argv[6]) != "--runtime-router") {
        config.engine.hot_cache_budget_gb = std::stod(argv[6]);
    }
    for (int i = 3; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--runtime-router") {
            config.engine.use_runtime_router = true;
        }
        if (std::string_view(argv[i]) == "--cpu-expert-backend") {
            config.engine.prefer_cuda_expert_backend = false;
        }
        if (std::string_view(argv[i]) == "--model-path" && i + 1 < argc) {
            config.engine.model_path = std::string(argv[i + 1]);
        }
        if (std::string_view(argv[i]) == "--dense-artifact" && i + 1 < argc) {
            config.engine.dense_artifact_path = std::string(argv[i + 1]);
        }
        if (std::string_view(argv[i]) == "--tokenizer-artifact" && i + 1 < argc) {
            config.engine.tokenizer_artifact_path = std::string(argv[i + 1]);
        }
    }
    if (!config.engine.use_runtime_router && config.engine.trace_path.empty()) {
        std::cerr << "routing_trace.txt is required unless --runtime-router is set\n";
        return EXIT_FAILURE;
    }

    try {
        flashmoe::StreamedService service(std::move(config));
        service.run();
    } catch (const std::exception& exc) {
        std::cerr << "streamed service failed: " << exc.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
