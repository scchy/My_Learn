cmake_minimum_required(VERSION 3.11)
project(TensorRT_NANO_APP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/local/cuda/include)
include_directories(${TensorRT_SAMPLE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib/)

add_library(engine SHARED
    src/engine.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
)
target_link_libraries(engine PUBLIC ${TensorRT_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(engine PROPERTIES CUDA_ARCHITECTURES "61")

add_library(utils SHARED
    src/utils/preprocess.cpp
    src/utils/postprocess.cpp
    src/utils/draw.cpp
)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})

add_library(algo SHARED
    src/facenet.cpp
    src/facedet.cpp
    src/attribute.cpp

)
target_link_libraries(algo engine utils cuda cudart)

add_executable(build
    build.cu
    ${TensorRT_SAMPLE_DIR}/common/logger.cpp
    ${TensorRT_SAMPLE_DIR}/common/sampleUtils.cpp
)
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${TensorRT_SAMPLE_DIR}/common)
target_link_libraries(build
    PRIVATE
    -Wl,--no-as-needed,--gc-sections,--unresolved-symbols=ignore-in-shared-libs
    gflags
    utils
    ${TensorRT_LIBRARIES}
    )
set_target_properties(build PROPERTIES CUDA_ARCHITECTURES "61")

add_subdirectory(streamer)

add_executable(stream
    stream.cpp
)
target_link_libraries(stream algo gflags streamer)
