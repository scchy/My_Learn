# 最低版本要求
cmake_minimum_required(VERSION 3.10)
# 项目名称
project(custom_install_demo)

#[[
首先会在CMAKE_MODULE_PATH中查找Finddlib.cmake文件
如果找不到，会在CMAKE_PREFIX_PATH中查找dlibConfig.cmake配置文件
set(CMAKE_PREFIX_PATH "./" CACHE PATH "dlib的安转路径")
]]
set(CMAKE_PREFIX_PATH "/root/myGithub/My_Learn/course_cpp_ternsorrt/cmake_course/16.custom_install_demo/custom_mod/install/config")

find_package(dlib REQUIRED)

if (dlib_FOUND)
    message(STATUS "dlib_FOUND: ${dlib_FOUND}")
    message(STATUS "dlib_INCLUDE_DIR: ${dlib_INCLUDE_DIR}")
    message(STATUS "dlib_LIBRARIES: ${dlib_LIBRARIES}")
    message(STATUS "dlib_AUTHOR: ${dlib_AUTHOR}")
    # 获取${dlib_INCLUDE_DIR}的父目录
    get_filename_component(dlib_INCLUDE_DIR_PARENT ${dlib_INCLUDE_DIR} DIRECTORY)
    message(STATUS "dlib_INCLUDE_DIR_PARENT: ${dlib_INCLUDE_DIR_PARENT}")
else()
    message(FATAL_ERROR "dlib not found")
endif()

# 设置RPATH，否则install后，运行时找不到动态库
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
SET(CMAKE_INSTALL_RPATH "${dlib_INCLUDE_DIR_PARENT}/lib")

# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp)
# 为目标添加链接库
target_link_libraries(${PROJECT_NAME} ${dlib_LIBRARIES})

# 安转
install(
    TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

#[[
cmake -S . -B build  -DCMAKE_INSTALL_PREFIX=./installed 
cmake --build build
cmake --install build
]]