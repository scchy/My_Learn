cmake_minimum_required(VERSION 3.10)

project(trt_demo LANGUAGES CXX CUDA)
# 添加cmake_MODULE_PATH 否则找不到Find
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(TensorRT REQUIRED)
if(TensorRT_FOUND)
    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
    message(STATUS "TensorRT include files: ${TensorRT_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Cannot find TensorRT")
endif()

# 添加可执行文件
add_executable(build src/build.cpp)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS})
target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES})

add_executable(runtime src/runtime.cu)
target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS})
target_link_libraries(runtime PRIVATE ${TensorRT_LIBRARIES})
