#include "Account.h"
#include <iostream>

Account::Account(/* args */)
{
    std::cout << "Account的构造函数被调用" << std::endl;
}

Account::~Account()
{
    std::cout << "Account的析构函数被调用" << std::endl;
}