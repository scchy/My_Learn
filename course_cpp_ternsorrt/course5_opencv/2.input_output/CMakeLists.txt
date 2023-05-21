cmake_minimum_required(VERSION 3.10)

project(opencv_demo)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not find OpenCV")
endif ()



# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS})
# 链接库
link_libraries(${OpenCV_LIBS} ${gflags_LIBRARIES})

# 添加可执行文件
add_executable(demo_1.img src/1.img.cpp)
add_executable(demo_2.video src/2.video.cpp)
add_executable(demo_3.camera src/3.camera.cpp)

