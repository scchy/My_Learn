#include <iostream>
#include <gflags/gflags.h> // 引入gflags头文件

DEFINE_string(name, "", "姓名");
DEFINE_int32(age, 0, "年龄");

int main(int argc, char **argv)
{                                                               
    gflags::ParseCommandLineFlags(&argc, &argv, true);          // 解析命令行参数
    std::cout << "name: " << FLAGS_name << std::endl;          // 输出name参数
    std::cout << "age: " << FLAGS_age << std::endl;            // 输出age参数`
    return 0;
}
