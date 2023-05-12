cmake_minimum_required(VERSION 3.10)

project(install_demo)

# 添加公共头文件目录
include_directories(include)

# 添加静态库
add_library(slib STATIC src/slib.cpp include/slib.h)
# target_include_directories(slib PUBLIC include) # 不添加公共头文件需要这样

# 添加动态库
add_library(dlib SHARED src/dlib.cpp include/dlib.h)
# target_include_directories(dlib PUBLIC include)
# 设置RPATH解决install后无法找打动态库的问题
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 添加可执行文件
add_executable(${PROJECT_NAME} main.cpp)

# 链接库
target_link_libraries(${PROJECT_NAME} slib dlib)

# 安装头文件目录
# 方法一：使用install将文件夹安转到指定目录
install(DIRECTORY include/ DESTINATION include)
# 方法二： 设置公共头文件
# set_target_properties(slib PROPERTIES PUBLIC_HEADER include/slib.h)
# set_target_properties(dlib PROPERTIES PUBLIC_HEADER include/dlib.h)

# 安装
message(STATUS "CMAKE_INSTALL_PREFIX默认值: ${CMAKE_INSTALL_PREFIX}")
install(
    TARGETS ${PROJECT_NAME} slib dlib
    RUNTIME DESTINATION bin # 可执行文件
    LIBRARY DESTINATION lib # 动态库
    ARCHIVE DESTINATION lib # 静态库
    # PUBLIC_HEADER DESTINATION include # 公共头文件 # 方法二需要
)

#[[
运行方式
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./installed
cmake --build build
cmake --install build


ldd ./bin/install_demo 
]]