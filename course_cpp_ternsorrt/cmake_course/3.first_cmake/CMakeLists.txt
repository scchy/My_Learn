# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# first_cmake是项目名称
# VERSION是版本号
# DESCRIPTION是项目描述
# LANGUGES是项目语音
project(first_cmake
	VERSION 1.0.0
	DESCRIPTION "项目描述"
	LANGUAGES CXX)

# 添加可执行程序，first_cmake是可执行程序名称  main.cpp是源文件
add_executable(first_cmake main.cpp)

