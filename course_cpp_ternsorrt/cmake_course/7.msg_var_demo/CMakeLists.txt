
cmake_minimum_required(VERSION 3.10)
project(msg_var_demo)

# 输出消息
message("输出消息")
message("输出1" "输出2" 输出3)

# 设置变量
# set(VAR1 "变量1")
# message("VAR1=" ${VAR1})
# message("\${VAR1}=${VAR1}")

# 设置变量缓存，可以在命令行中修改（-D）
# set(CACHE_VAR_TEST "new value" CACHE STRING "变量缓存的描述")
# message("变量缓存的值：${CACHE_VAR_TEST}")
##  cmake -S . -B build -DCACHE_VAR_TEST=abc

# 常见的内置的变量，更多访问：https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html#variables-that-provide-information


# 第一类：提供信息的变量
message("${PROJECT_NAME}") # 项目名称
message("${CMAKE_SOURCE_DIR}") # 源码目录
message("${CMAKE_BINARY_DIR}") # 编译目录
message("${CMAKE_CURRENT_LIST_FILE}") # 当前CMakeLists.txt文件路径

# 第二类：控制CMake运行的变量，更多：https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html#variables-that-change-behavior
set(BUILD_SHARED_LIBS ON) # 设置是否构建动态库，默认为OFF，即构建静态库，设置为ON后，构建动态库


# 第三类：描述系统的变量，更多：https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html#variables-that-describe-the-system
message("是否是Windows系统：${WIN32}")
message("是否是Linux系统：${UNIX}")
message("系统名称：${CMAKE_SYSTEM_NAME}")
