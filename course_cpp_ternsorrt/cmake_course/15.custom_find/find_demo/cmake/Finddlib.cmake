#寻找 dlib.h
find_path(dlib_INCLUDE_DIR dlib.h PATHS ${DLIB_INSTALL_PATH}/include)

#寻找 libdlib.so
#[[
注意HINTS与PATHS区别：
- HINTS是在搜索系统路径之前先搜索HINTS指定的路径。
- PATHS是先搜索系统路径，然后再搜索PATHS指定的路径
]]
find_library(dlib_LIBRARY dlib  HINTS ${DLIB_INSTALL_PATH}/lib)

# 如果dlib_INCLUDE_DIR和dlib_LIBRARY都找到了，那么就设置dlib_FOUND为TRUE
if(dlib_INCLUDE_DIR AND dlib_LIBRARY)
    set(dlib_FOUND TRUE) 
    set(dlib_VERSION 1.0.0) # dlib的版本号
    set(dlib_AUTHOR "scc") # dlib的作者
    message(STATUS "FIND: dlib_LIBRARY=${dlib_LIBRARY}")
    # lib文件所在目录
    get_filename_component(dlib_LIBRARY_DIR ${dlib_LIBRARY} DIRECTORY)
endif()