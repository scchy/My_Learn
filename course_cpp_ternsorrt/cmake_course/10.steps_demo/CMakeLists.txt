cmake_minimum_required(VERSION 3.10)

project(steps_demo)

add_executable(${PROJECT_NAME} main.cpp)

#[[
cmake -S . -B build
cd build
cmake --build . --target help
cmake --build . --target main.i
cmake --build . --target main.s
cmake --build . --target main.o
cmake --build . 
]]