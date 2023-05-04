#ifndef ACCOUNT_H
#define ACCOUNT_H
#include <string>


class Account
{
private:
    // 属性
    std::string name {"None"};
    double balance {0.0};
public:
    // 构造函数
    Account();
    Account(std::string name, double v);
    // 方法
    bool deposit(double amt);
    bool withdraw(double amt);
    void set_name(std::string name);
    std::string get_name();
    void set_balance(double v);
    double get_balance();
    // 拷贝函数声明
    Account(const Account &src);
};
#endif