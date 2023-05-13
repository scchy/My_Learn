cmake_minimum_required(VERSION 3.10)

# 项目文件
project(main)

# 设置CMAKE_MODULE_PATH,以便find_package查找，cmake下需要有Finddlib.cmake文件
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/)
message(STATUS "cmake_module_path: ${CMAKE_MODULE_PATH}")

# # 设置一个缓存变量，用于命令行设置dlib的安装路径，给Finddlib.cmake使用
set(DLIB_INSTALL_PATH "./" CACHE PATH "dlib的安转路径")
message(STATUS "dlib的安装路径为：${DLIB_INSTALL_PATH}")

#使用find_package查找dlib(这里会从CMAKE_MODULE_PATH设置的路径中查找)find_package(dlib REQUERED)
find_package(dlib REQUIRED)

if(dlib_FOUND)
    message("dlib find")
    message("dlib includedir: ${dlib_INCLUDE_DIR}")
    message("dlib lib: ${dlib_LIBRARY}")
    message("dlib version: ${dlib_VERSION}")
    message("dlib author: ${dlib_AUTHOR}")
    message("dlib lib dir: ${dlib_LIBRARY_DIR}")
else()
    message("dlib not found")
endif()

# 设置RPATH, 否则install后，运行是会找不到动态库
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "${dlib_LIBRARY_DIR}")

# 添加可执行文件
add_executable(main main.cpp)

# 添加头文件
target_include_directories(main PUBLIC ${dlib_INCLUDE_DIR})
# 链接动态库
target_link_libraries(main ${dlib_LIBRARY})


# 设置安转
install(
    TARGETS main
    RUNTIME DESTINATION bin 
    # LIBRARY DESTINATION lib  # 静态库
    # ARCHIVE DESTINATION lib  # 动态库
    # PUBLIC_HEADER DESTINATION include
)


#[[
CMAKE_INSTALL_PREFIX为安装路径（系统内置），DLIB_INSTALL_PATH为dlib安装路径

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./installed -DDLIB_INSTALL_PATH=/root/myGithub/My_Learn/course_cpp_ternsorrt/cmake_course/15.custom_find/custom_mod/install
cmake --build build
cmake --install build

]]