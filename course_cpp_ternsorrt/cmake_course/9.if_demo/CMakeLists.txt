cmake_minimum_required(VERSION 3.10)

project(syntax_test)

message("====== if ======")
#[[
if 测试，语法如下：

if (<condition>)
    <commands>
elseif (<condition>) 
    <commands>
else()
    <commands>
endif()

<condition>：可以是常量、变量、字符串
为true的常量：ON、YES、TRUE、Y、1、非0数字
为false的常量：OFF、NO、FALSE、N、0、空字符串、NOTFOUND

]]
# 常量 ===============================
if(1)
    message("1 是 true")
endif()

if(0)
    message("0 是 true")
else()
    message("0 是 false")
endif()


# 未定义的变量 ===============================
if(UNDEFINED)
    message("UNDEFINED 是 true")
else()
    message("UNDEFINED 是 false")
endif()

# 定义了的变量 ===============================
set(DEFINED 1)
if(DEFINED)
    message("DEFINED 是 true")
else()
    message("DEFINED 是 false")
endif()

# 字符串 ===============================
# 把字符串当成常量

if("Y") 
    message("字符串是 true")
else()
    message("字符串是 false")
endif()


message("====== if 配合逻辑运算符 ======")
#[[
if 配合逻辑运算符（AND、OR、NOT、括号）使用

if (<condition1> AND <condition2>) # 两个条件都为true
    <commands>
endif()

if (<condition1> OR <condition2>) # 两个条件有一个为true
    <commands>
endif()

if (NOT <condition>) # 条件为false
    <commands>
endif()

if (<condition1> AND (<condition2> OR <condition3>)) # 先算括号里的
    <commands>
endif()

]]

# AND ===============================
if(1 AND YES)
    message("1 AND YES 是 true")
else()
    message("1 AND YES 是 false")
endif()

# OR ===============================
if(1 OR NO)
    message("1 OR NO 是 true")
else()
    message("1 OR NO 是 false")
endif()

# NOT ===============================
if(NOT 0)
    message("NOT 0 是 true")
else()
    message("NOT 0 是 false")
endif()

# 括号 ===============================
if(1 AND (0 OR 1))
    message("1 AND (0 OR 1) 是 true")
else()
    message("1 AND (0 OR 1) 是 false")
endif()