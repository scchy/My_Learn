cmake_minimum_required(VERSION 3.10)

project(opencv_basic_input_output)

# 添加opencv库
find_package(OpenCV REQUIRED)
# 添加gflags库
find_package(gflags REQUIRED)

# 添加头文件（两个target都需要）
include_directories(${OpenCV_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS} ${gflags_LIBRARIES})


# 将src目录下的所有cpp文件编译成对应的可执行文件
# 例如： src/1_read_image.cpp -> 1_read_image
# 列出src目录下的所有cpp文件，存入SRC_FILES变量，是一个list， GLOB_RECURSE 表示递归查找
file(GLOB_RECURSE SRC_FILES src/*.cpp)
message(STATUS "SRC FILES: ${SRC_FILES}")

# 遍历SRC_FILES列表，对每个文件编译成一个可执行文件
foreach(SRC_FILE ${SRC_FILES})
    # 获取文件名，不包含后缀，
    # 存入TARGET_NAME变量，
    # NAME_WLE：File name with neither the directory nor the last extension
    get_filename_component(TARGET_NAME ${SRC_FILE} NAME_WLE)
    # 添加可执行文件
    add_executable(${TARGET_NAME} ${SRC_FILE})
endforeach()

#[[
cmake -S . -B build 
cmake --build build 
]]