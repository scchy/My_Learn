cmake_minimum_required(VERSION 3.10)

project(main)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS} ${gflags_LIBRARIES})

add_executable(main src/main.cpp)
#[[
cmake -S . -B build
cmake --build build
]]