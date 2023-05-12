#account_dir/CMakeLists.txt

# 版本
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(Account)

#[[
指定生成目标：动态库
三个参数：库名称 库类型 源文件
]]
add_library(Account SHARED Account.cpp)