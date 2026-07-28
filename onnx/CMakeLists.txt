cmake_minimum_required(VERSION 3.20)
project(onnx_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 查找 ONNX Runtime ──────────────────────────────────────────
# 尝试多个常见路径 / 环境变量
set(ORT_SEARCH_PATHS
    /usr/local/onnxruntime
    /opt/onnxruntime
    $ENV{ONNXRUNTIME_DIR}
    $ENV{ORT_ROOT}
)

find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ORT_SEARCH_PATHS}
    PATH_SUFFIXES include
)

find_library(ORT_LIBRARY
    NAMES onnxruntime
    PATHS ${ORT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(WARNING "未找到 ONNX Runtime SDK")
    message(WARNING "  请通过 -DONNXRUNTIME_DIR=<path> 指定路径")
    message(WARNING "  ORT include: ${ORT_INCLUDE_DIR}")
    message(WARNING "  ORT library: ${ORT_LIBRARY}")
endif()

# ── 可选 OpenCV（CV 任务预处理）────────────────────────────────
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    add_definitions(-DHAS_OPENCV)
    message(STATUS "OpenCV found: ${OpenCV_LIBS}")
else()
    message(STATUS "OpenCV not found — CV benchmark 将使用内置预处理")
endif()

# ── 公共库 ──────────────────────────────────────────────────────
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/include
    ${ORT_INCLUDE_DIR}
)

# ── 推理库 ──────────────────────────────────────────────────────
add_library(onnx_inference STATIC
    src/inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/src/benchmark_utils.cpp
)

target_include_directories(onnx_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/include
    ${ORT_INCLUDE_DIR}
)

if(ORT_LIBRARY)
    target_link_libraries(onnx_inference PUBLIC ${ORT_LIBRARY})
endif()

if(OpenCV_FOUND)
    target_include_directories(onnx_inference PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()

if(WIN32)
    target_link_libraries(onnx_inference PUBLIC psapi)
endif()

# ── 基准测试可执行文件 ──────────────────────────────────────────
add_executable(onnx_benchmark src/benchmark.cpp)

target_link_libraries(onnx_benchmark PRIVATE onnx_inference)

if(OpenCV_FOUND)
    target_link_libraries(onnx_benchmark PRIVATE ${OpenCV_LIBS})
endif()

if(WIN32)
    target_link_libraries(onnx_benchmark PRIVATE psapi)
endif()
