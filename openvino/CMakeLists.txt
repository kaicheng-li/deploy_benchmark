cmake_minimum_required(VERSION 3.20)
project(openvino_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 查找 OpenVINO ─────────────────────────────────────────────
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# ── 可选 OpenCV（CV 任务预处理）────────────────────────────────
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    add_definitions(-DHAS_OPENCV)
    message(STATUS "OpenCV found: ${OpenCV_LIBS}")
else()
    message(STATUS "OpenCV not found — CV benchmark 将使用内置预处理")
endif()

# ── 公共头文件 ────────────────────────────────────────────────
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/include
    ${OpenVINO_INCLUDE_DIRS}
)

# ── 推理库 ────────────────────────────────────────────────────
add_library(ov_inference STATIC
    src/inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/src/benchmark_utils.cpp
)

target_include_directories(ov_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/include
    ${OpenVINO_INCLUDE_DIRS}
)

target_link_libraries(ov_inference PUBLIC openvino::runtime)

if(OpenCV_FOUND)
    target_include_directories(ov_inference PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()

if(WIN32)
    target_link_libraries(ov_inference PUBLIC psapi)
endif()

# ── 基准测试可执行文件 ────────────────────────────────────────
add_executable(ov_benchmark src/benchmark.cpp)

target_link_libraries(ov_benchmark PRIVATE
    ov_inference
    openvino::runtime
)

if(OpenCV_FOUND)
    target_link_libraries(ov_benchmark PRIVATE ${OpenCV_LIBS})
endif()

if(WIN32)
    target_link_libraries(ov_benchmark PRIVATE psapi)
endif()
