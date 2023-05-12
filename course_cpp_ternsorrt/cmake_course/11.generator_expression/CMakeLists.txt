cmake_minimum_required(VERSION 3.10)

project(generator_expression)

#[[
==================================================== 
1.条件表达式：$<condition:true_string>，当condition为真时，返回true_string，否则返回空字符串

示例：
$<0:TEST>  
$<1:TEST>  
$<$<BOOL:TRUE>:TEST>
]]

# 需要注意的是，生成表达式被展开是在生成构建系统的时候，
# 所以不能通过解析配置`CMakeLists.txt`阶段的`message`命令打印，
# 可以用类似`file(GENERATE OUTPUT "./generator_test.txt" CONTENT "$<$<BOOL:TRUE>:TEST>")`
# 生成文件的方式间接测试。

# 用来测试
# file(GENERATE OUTPUT "./generator_test.txt" CONTENT "$<$<BOOL:TRUE>:TEST>")
# file(GENERATE OUTPUT "./gg.txt" CONTENT "$<$<BOOL:TRUE>:TEST>")

#[[
====================================================
2.变量查询（Variable-Query）

$<TARGET_EXISTS:target>：判断目标是否存在
$<CONFIG:Debug>：判断当前构建类型是否为Debug
]]

# 添加一个可执行文件
# add_executable(generator_expression main.cpp)
# file(GENERATE OUTPUT "./generator_test.txt" CONTENT "$<TARGET_EXISTS:generator_expression>")
# file(GENERATE OUTPUT "./generator_test.txt" CONTENT "$<$<TARGET_EXISTS:generator_expression>:目标已存在>") # 嵌套

# 设置构建类型，Debug/Release/...
# SET(CMAKE_BUILD_TYPE "Debug") 
# file(GENERATE OUTPUT "./generator_test.txt" CONTENT "$<$<CONFIG:Debug>:--coverage>") # 嵌套


#[[
====================================================
3.目标查询（Target-Query）
$<TARGET_FILE:target>：获取编译目标的文件路径
$<TARGET_FILE_NAME:target>：获取编译目标的文件名
]]

# 添加一个可执行文件
add_executable(generator_expression main.cpp)
file(GENERATE OUTPUT "./generator_test.txt" CONTENT "$<TARGET_FILE:generator_expression>")

