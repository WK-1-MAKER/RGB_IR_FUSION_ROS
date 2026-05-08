cmake_minimum_required(VERSION 3.5)
project(two_stream)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "/home/wk/3rd/onnxruntime-linux-x64-gpu-1.10.0" CACHE PATH "Path to ONNX Runtime root directory" FORCE)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
set(TENSORRT_ROOT "/home/wk/3rd/TensorRT-8.6.1.6" CACHE PATH "Path to TensorRT root directory")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")
set(TENSORRT_TARGET_LIB_DIR "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
set(TENSORRT_LIBRARY_PATHS ${TENSORRT_LIB_DIR})
if(EXISTS "${TENSORRT_TARGET_LIB_DIR}")
  list(APPEND TENSORRT_LIBRARY_PATHS ${TENSORRT_TARGET_LIB_DIR})
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  message_filters
)

find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime libonnxruntime.so libonnxruntime.so.1.10.0
  PATHS ${ONNXRUNTIME_DIR}/lib
  NO_DEFAULT_PATH
)

find_library(TENSORRT_NVINFER_LIB NAMES nvinfer PATHS ${TENSORRT_LIBRARY_PATHS} NO_DEFAULT_PATH)
find_library(TENSORRT_NVONNXPARSER_LIB NAMES nvonnxparser PATHS ${TENSORRT_LIBRARY_PATHS} NO_DEFAULT_PATH)
find_library(TENSORRT_NVINFER_PLUGIN_LIB NAMES nvinfer_plugin PATHS ${TENSORRT_LIBRARY_PATHS} NO_DEFAULT_PATH)
find_library(TENSORRT_NVPARSERS_LIB NAMES nvparsers PATHS ${TENSORRT_LIBRARY_PATHS} NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "Cannot find ONNX Runtime library under ${ONNXRUNTIME_DIR}/lib")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Cannot find ONNX Runtime headers under ${ONNXRUNTIME_DIR}/include")
endif()

if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
  message(FATAL_ERROR "Cannot find TensorRT headers under ${TENSORRT_INCLUDE_DIR}")
endif()

if(NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_NVONNXPARSER_LIB)
  message(FATAL_ERROR "Cannot find TensorRT core libraries under ${TENSORRT_LIB_DIR}")
endif()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp sensor_msgs std_msgs cv_bridge image_transport message_filters
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_DIR}/include
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer/include
  /usr/include/eigen3
)

add_executable(ros_yolo_node
  src/main.cpp
  src/yolo_node.cpp
  src/detector.cpp
  src/utils.cpp
  src/super_point.cpp
  src/light_glue.cpp
  3rdparty/tensorrtbuffer/src/logger.cpp
  3rdparty/tensorrtbuffer/src/sample_options.cpp
  3rdparty/tensorrtbuffer/src/sample_utils.cpp
)

target_link_libraries(ros_yolo_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIB}
  ${CUDA_LIBRARIES}
  ${TENSORRT_NVINFER_LIB}
  ${TENSORRT_NVONNXPARSER_LIB}
  -Wl,--no-as-needed
  ${TENSORRT_NVINFER_PLUGIN_LIB}
  -Wl,--as-needed
  ${TENSORRT_NVPARSERS_LIB}
  yaml-cpp
)

set_target_properties(ros_yolo_node PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_DIR}/lib;${TENSORRT_LIB_DIR};${TENSORRT_TARGET_LIB_DIR}"
  INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib;${TENSORRT_LIB_DIR};${TENSORRT_TARGET_LIB_DIR}"
)

install(TARGETS ros_yolo_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
