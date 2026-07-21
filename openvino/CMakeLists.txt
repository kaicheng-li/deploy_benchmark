cmake_minimum_required(VERSION 3.20)
project(openvino_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenVINO
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# 可选 OpenCV（图像预处理）
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    add_definitions(-DHAS_OPENCV)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/include
    ${OpenVINO_INCLUDE_DIRS}
)

add_executable(ov_benchmark src/benchmark.cpp)

target_link_libraries(ov_benchmark PRIVATE
    openvino::runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp_common/src/benchmark_utils.cpp
)

if(OpenCV_FOUND)
    target_link_libraries(ov_benchmark PRIVATE ${OpenCV_LIBS})
endif()

if(WIN32)
    target_link_libraries(ov_benchmark PRIVATE psapi)
endif()
