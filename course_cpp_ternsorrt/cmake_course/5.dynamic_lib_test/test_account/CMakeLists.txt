# test_account/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(test_account)

# 添加可执行文件目标
add_executable(test_account main.cpp)

# 为目标添加头文件目录
target_include_directories(test_account PUBLIC "../account_dir")

# 
target_link_directories(test_account PUBLIC "../account_dir/build")
# 为目标添加库
target_link_libraries(test_account Account)
