cmake_minimum_required(VERSION 3.20)
project(gb10_flashmoe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(flashmoe_core
    src/compute_profile.cpp
    src/decode_state.cpp
    src/dense_decode.cpp
    src/dense_artifact.cpp
    src/dense_operator_chain.cpp
    src/decode_harness.cpp
    src/dense_resident_loader.cpp
    src/device_path.cpp
    src/expert_cache.cpp
    src/expert_operator_chain.cpp
    src/expert_store.cpp
    src/io_pool.cpp
    src/kv_cache.cpp
    src/model_spec.cpp
    src/online_runtime.cpp
    src/pipeline_model.cpp
    src/router_runtime.cpp
    src/runtime_plan.cpp
    src/streamed_engine.cpp
    src/streamed_runtime.cpp
    src/streamed_service.cpp
    src/tokenizer_artifact.cpp
)

target_include_directories(flashmoe_core PUBLIC include)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    target_sources(flashmoe_core PRIVATE src/expert_cuda_backend.cu)
    target_compile_definitions(flashmoe_core PRIVATE FLASHMOE_WITH_CUDA_EXPERT_BACKEND=1)
    target_link_libraries(flashmoe_core PUBLIC CUDA::cudart)
    set_target_properties(flashmoe_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
else()
    target_sources(flashmoe_core PRIVATE src/expert_cuda_backend_stub.cpp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(flashmoe_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(Threads REQUIRED)
target_link_libraries(flashmoe_core PUBLIC Threads::Threads)

add_executable(flashmoe_trace_sim src/trace_sim.cpp)
target_link_libraries(flashmoe_trace_sim PRIVATE flashmoe_core)

add_executable(flashmoe_pipeline_demo src/pipeline_demo.cpp)
target_link_libraries(flashmoe_pipeline_demo PRIVATE flashmoe_core)

add_executable(flashmoe_runtime_plan src/runtime_plan_demo.cpp)
target_link_libraries(flashmoe_runtime_plan PRIVATE flashmoe_core)

add_executable(flashmoe_online_runtime_demo src/online_runtime_demo.cpp)
target_link_libraries(flashmoe_online_runtime_demo PRIVATE flashmoe_core)

add_executable(flashmoe_decode_harness src/decode_harness_demo.cpp)
target_link_libraries(flashmoe_decode_harness PRIVATE flashmoe_core)

add_executable(flashmoe_streamed_runtime src/streamed_runtime_demo.cpp)
target_link_libraries(flashmoe_streamed_runtime PRIVATE flashmoe_core)

add_executable(flashmoe_streamed_service src/streamed_service_main.cpp)
target_link_libraries(flashmoe_streamed_service PRIVATE flashmoe_core)

add_executable(flashmoe_streamed_cli src/streamed_cli_main.cpp)
target_link_libraries(flashmoe_streamed_cli PRIVATE flashmoe_core)
