cmake_minimum_required(VERSION 3.11)

project(TensorRT_YoloV5 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)
message("OPENCV INCLUDE DIRS: ${OpenCV_INCLUDE_DIRS}")
message("OPENCV LINK LIBRARIES: ${OpenCV_LIBS}")

# 添加公共头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
include_directories(${TensorRT_SAMPLE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib/)
# 添加动态库
add_library(
    engine SHARED
    src/engine.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
)
# 链接库
target_link_libraries(engine PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(engine PROPERTIES CUDA_ARCHITECTURES "61")

add_library(
    utils SHARED 
    src/utils/preprocess.cpp
    src/utils/postprocess.cpp
    src/utils/draw.cpp
)

target_link_libraries(utils PUBLIC ${OpenCV_LIBS})

add_library(algo SHARED src/facedet.cpp)
target_link_libraries(algo engine utils cuda cudart)
add_executable(facedet_test facedet_test.cpp)
target_link_libraries(facedet_test algo gflags)
