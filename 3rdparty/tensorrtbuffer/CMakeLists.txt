cmake_minimum_required(VERSION 3.6)
project(tensorrtbuffer)

set(CMAKE_CXX_STANDARD 14)

set(TensorRT_ROOT "/home/wk/3rd/TensorRT-8.6.1.6" CACHE PATH "Path to TensorRT root directory")
set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT}/include" CACHE PATH "Path to TensorRT include directory")

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${TensorRT_INCLUDE_DIR}
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(tensorrtbuffer SHARED src/logger.cpp src/sample_options.cpp src/sample_utils.cpp)



