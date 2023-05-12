#include "Account.h"
#include <iostream>

Account::Account(/* args */)
{
    std::cout << "构造函数Account::Account()" << std::endl;
}
Account::~Account()
{
    std::cout << "析构函数Account::~Account()" << std::endl;
}