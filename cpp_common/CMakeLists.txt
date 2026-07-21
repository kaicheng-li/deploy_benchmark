cmake_minimum_required(VERSION 3.20)
project(deploy_bench_cpp_common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 可选 CUDA 支持
find_package(CUDA QUIET)
if(CUDA_FOUND)
    add_definitions(-DHAS_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

add_library(benchmark_utils STATIC
    src/benchmark_utils.cpp
)

target_include_directories(benchmark_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(CUDA_FOUND)
    target_link_libraries(benchmark_utils PUBLIC ${CUDA_LIBRARIES})
endif()

# Windows 平台链接 psapi
if(WIN32)
    target_link_libraries(benchmark_utils PUBLIC psapi)
endif()
