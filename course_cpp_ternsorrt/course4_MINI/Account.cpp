#include "Account.h"
#include <iostream>


// Account::Account(): name("None"), balance(0.0) {
//     std::cout << "没有参数的构造函数" << std::endl;
// };
// Account::Account(std::string name, double v){
//     this->name = name;
//     this->balance = v;
// };
// 代理构造函数
Account::Account(): Account("None", 0.0) {
    std::cout << "没有参数的构造函数" << std::endl;
};

Account::Account(std::string name, double v): name(name), balance(v){};

// 拷贝构造函数:根据已存在对象的属性更新新对象
Account::Account(const Account &src)
:name(src.name), balance(src.balance){
    std::cout << "拷贝构造函数被调用， 是" << src.name << "的拷贝。" << std::endl;
}

void Account::set_name(std::string name){
    this->name = name;
}

void Account::set_balance(double v){
    this->balance = v;
}


std::string Account::get_name(){
    return name;
}

double Account::get_balance(){
    return balance;
}

bool Account::deposit(double amt){
    balance += amt;
    std::cout << name << "刚存入" << amt << "元，现在余额是";
    std::cout << balance << "元" << std::endl;
    return true;
};
bool Account::withdraw(double amt){
    if(balance >= amt){
        balance -= amt;
        std::cout << name << "刚取出" << amt << "元，现在余额是";
        std::cout << balance << "元" << std::endl;
        return true;
    }else{
        std::cout << name << "余额不足，取款失败"<< std::endl;
        return false;
    }
};