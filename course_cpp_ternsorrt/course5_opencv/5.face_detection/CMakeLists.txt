# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(face_detect)


# 添加opencv库
find_package(OpenCV REQUIRED)

# 添加gflags库
find_package(gflags REQUIRED)

# 添加头文件（两个target都需要）
include_directories(${OpenCV_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS} ${gflags_LIBRARIES})

add_executable(face_detect src/face_detect.cpp)



#[[
cmake -S . -B build 
cmake --build build 
]]