cmake_minimum_required(VERSION 3.20)
project(tensorrt_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 查找 CUDA ─────────────────────────────────────────────────
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ── 查找 TensorRT ─────────────────────────────────────────────
# 尝试多个常见路径
set(TRT_SEARCH_PATHS
    /usr/src/tensorrt
    /usr/local/tensorrt
    /opt/tensorrt
    ${CMAKE_SOURCE_DIR}/../../TensorRT-10.*
    ${CMAKE_SOURCE_DIR}/../../TensorRT-8.*
    $ENV{TENSORRT_DIR}
    $ENV{TensorRT_ROOT}
)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TRT_SEARCH_PATHS}
    PATH_SUFFIXES include
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer nvinfer10
    PATHS ${TRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(WARNING "未找到 TensorRT SDK，请设置 -DTensorRT_ROOT=<path>")
    message(WARNING "  TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(WARNING "  TensorRT library: ${TENSORRT_LIBRARY}")
else()
    message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
endif()

# ── 编译 ──────────────────────────────────────────────────────
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(trt_benchmark
    src/benchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/src/benchmark_utils.cpp
)

target_link_libraries(trt_benchmark PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

if(TENSORRT_LIBRARY)
    target_link_libraries(trt_benchmark PRIVATE ${TENSORRT_LIBRARY})
endif()

if(WIN32)
    target_link_libraries(trt_benchmark PRIVATE psapi)
endif()

# 如果找不到 TensorRT，至少用 CUDA 编译成功（运行时会提示）
if(NOT TENSORRT_LIBRARY)
    message(WARNING "=== TensorRT library 未找到 ===")
    message(WARNING "  构建可能失败。请通过 -DTensorRT_ROOT 指定路径:")
    message(WARNING "  cmake .. -DTensorRT_ROOT=/path/to/TensorRT-10.x.x")
endif()
