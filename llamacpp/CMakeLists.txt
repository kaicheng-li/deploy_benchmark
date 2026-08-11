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
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "")
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "")
    set(LLAMA_BUILD_SERVER ON CACHE BOOL "")

    if(GGML_CUDA)
        set(GGML_CUDA ON CACHE BOOL "")
    endif()

    add_subdirectory(llama.cpp)
endif()

# ── 本项目工具 ───────────────────────────────────────────────
# This target is the native C++ client for an externally running llama-server.
add_executable(llama_api_benchmark
    src/api_benchmark.cpp
)

find_package(Threads REQUIRED)
target_link_libraries(llama_api_benchmark PRIVATE Threads::Threads)
