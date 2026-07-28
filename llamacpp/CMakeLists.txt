cmake_minimum_required(VERSION 3.20)
project(llamacpp_bench LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── 选项 ──────────────────────────────────────────────────────
option(GGML_CUDA "Enable CUDA acceleration" ON)
option(GGML_BLAS "Enable BLAS acceleration" OFF)
option(GGML_METAL "Enable Metal (macOS only)" OFF)

# ── 子项目: llama.cpp ─────────────────────────────────────────
# 如果已 clone llama.cpp:
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/CMakeLists.txt)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "")

    if(GGML_CUDA)
        set(GGML_CUDA ON CACHE BOOL "")
    endif()

    add_subdirectory(llama.cpp)
endif()

# ── 主程序 ────────────────────────────────────────────────────
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include
)

add_library(bench_utils STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/src/benchmark_utils.cpp
)

add_executable(llama_benchmark
    src/inference.cpp
    src/benchmark.cpp
)

target_link_libraries(llama_benchmark PRIVATE
    bench_utils
    llama
    ggml
)

if(WIN32)
    target_link_libraries(llama_benchmark PRIVATE psapi)
endif()
