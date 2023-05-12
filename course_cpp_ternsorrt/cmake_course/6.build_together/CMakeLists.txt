# 6.build_together/CMakeLists.txt

# 最低版本
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(test_account)

# 添加目标库： 动态库
add_library(Account SHARED "account_dir/Account.cpp" "account_dir/Account.h")

# 添加目标；可执行文件
add_executable(test_account "test_account/main.cpp")
# 添加头文件目录
target_include_directories(test_account PUBLIC "account_dir")

# 连接库
target_link_libraries(test_account Account)
