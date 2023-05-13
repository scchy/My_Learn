# 最低版本
cmake_minimum_required(VERSION 3.10)
# 项目名称
project(custom_mod)

# 添加动态库
add_library(dlib SHARED src/dlib.cpp)

#[[
路径被两次引用
1- 编译dlib
2- install export 写入configs时
所以会报错
]]
# target_include_directories(dlib PUBLIC include)

# 使用生成器表达式，解决路径被两次引用问题
target_include_directories(dlib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>  # 生成时的头文件目录
    $<INSTALL_INTERFACE:include>                    # 安装时的头文件目录
)

# 设置公共头文件
set_target_properties(dlib PROPERTIES PUBLIC_HEADER include/dlib.h)

# 添加安转
install(TARGETS dlib
    EXPORT dlib      # 导出目标
    RUNTIME DESTINATION bin # 可执行文件
    LIBRARY DESTINATION lib # 动态库
    ARCHIVE DESTINATION lib # 静态库
    PUBLIC_HEADER DESTINATION include # 公共头文件，只有在设置了 PUBLIC_HEADER 时才会安装，或者使用下面一句
    )

# 通过模板生成 <Package>Config.cmake. 用于find_package()寻找包
# 可以包赚一些信息，比如版本号等
set(TARGET_NAME dlib)

install(EXPORT dlib
    FILE ${TARGET_NAME}Target.cmake
    DESTINATION config
)

include(CMakePackageConfigHelpers)
# configure_package_config_file 参数：模板文件，生成文件，安装路径
configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/${TARGET_NAME}Config.cmake.in
    ${TARGET_NAME}Config.cmake
    INSTALL_DESTINATION config
)

# 将生成的<Package>Config.cmake 安转到${CMAKE_INSTALL_PREFIX}/config目录下
# CNAKE_CURRENT_BIANRY_DIR 为当前编译目录
install( FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}Config.cmake
    DESTINATION config
)

#[[
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./install
cmake --build build
cmake --install build
]]

