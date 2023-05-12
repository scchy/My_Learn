cmake_minimum_required(VERSION 3.10)

project(find_demo)
# 添加可执行文件
add_executable(find_demo main.cpp)

# 寻找gflags
find_package(gflags REQUIRED)

if(gflags_FOUND)
    message(STATUS "gflags found")
    message(STATUS "gflags include dir: ${gflags_INCLUDE_DIR}")
    message(STATUS "gflags lib dir: ${gflags_LIBRARIES}")

    # 将可执行文件添加到头文件目录和库文件目录
    target_include_directories(find_demo PRIVATE ${gflags_INCLUDE_DIR})
    target_link_libraries(find_demo ${gflags_LIBRARIES})
else()
    message(FATAL_ERROR "gflags not found")
endif()
