#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Account
{
public:
    // 属性
    string name {"None"};
    double balance {0.0};
    // 方法
    bool deposit(double amt){
        balance += amt;
        cout << name << "刚存入" << amt << "元，现在余额是";
        cout << balance << "元" << endl;
        return true;
    };
    bool withdraw(double amt){
        if(balance >= amt){
            balance -= amt;
            cout << name << "刚取出" << amt << "元，现在余额是";
            cout << balance << "元" << endl;
            return true;
        }else{
            cout << name << "余额不足，取款失败"<< endl;
            return false;
        }
    };
};

int main(){
    Account j_acc;
    j_acc.name = "jobs";
    j_acc.balance = 1000.0;

    j_acc.deposit(400.0);
    Account *alice_acc = new Account();
    (*alice_acc).name = "Alice";
    alice_acc -> balance = 2000.0;

    (*alice_acc).deposit(2000.0);
    alice_acc -> withdraw(500.0);


    return 0;
}