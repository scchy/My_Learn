
cmake_minimum_required(VERSION 3.10)

project(function_macro_test)

# ====================================
# 定义一个宏，宏名为my_macro，没有参数
macro(my_macro)
    message("宏内部的信息")
    set(macro_var "宏内部变量test 111")
endmacro(my_macro)

# 调用宏
my_macro()
my_macro()
# 输出宏内部的信息，也能访问到变量，理解为代码替换
message(${macro_var})


# ====================================
# 定义一个宏，宏名为second_macro，有两个参数
message(=============================================)
macro(second_macro arg1 arg2)
    message("第一个参数：${arg1}, 第二个参数：${arg2}")    
endmacro(second_macro)

# 调用宏
second_macro("hello" "world")

# ====================================
# 定义一个函数，函数名为my_func，没有参数
message(=============================================)
function(my_func)
    message("函数内部的信息")
    set(func_var "变量test")
endfunction(my_func)

# 调用函数
my_func()
my_func()
# 访问不了函数内部的变量，因为函数是一个独立的作用域
# message(${func_var})


# ====================================
# 定义一个函数，函数名为second_func，有两个参数
message(=============================================)
function(second_func arg1 arg2)
    message("第一个参数：${arg1}, 第二个参数：${arg2}")
endfunction(second_func)

# 调用函数
second_func("hello" "world")


