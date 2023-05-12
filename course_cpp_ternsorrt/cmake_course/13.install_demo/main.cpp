#include "slib.h"
#include "dlib.h"
#include <iostream>

int main()
{
    std::cout << "main函数被调用" << std::endl;
    slib_test();
    dlib_test();
    return 0; 
}
