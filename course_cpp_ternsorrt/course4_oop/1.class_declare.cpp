#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Account
{
    // 属性
    string name {"None"};
    double balance {0.0};
    // 方法
    bool deposit(double amt);
    bool withdraw(double amt);
};

int main(){
    Account j_a;
    Account a_a;

    Account acs [] {j_a, a_a};
    vector<Account> acc_vec {j_a};
    acc_vec.push_back(a_a);

    Account *p_acc = new Account();
    delete p_acc;
    return 0;
}