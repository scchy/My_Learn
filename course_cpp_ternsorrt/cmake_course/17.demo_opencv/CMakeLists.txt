cmake_minimum_required(VERSION 3.10)
project(demo_opencv)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not find OpenCV")
endif ()


add_executable(demo_opencv main.cpp)
target_include_directories(demo_opencv PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo_opencv ${OpenCV_LIBS})
