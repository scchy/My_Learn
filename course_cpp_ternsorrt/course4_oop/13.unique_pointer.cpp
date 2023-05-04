#include <iostream>
#include <string>
#include <vector>
#include <memory>
using namespace std;


class Account
{
private:
    string name {"account"};
    double balance {0.0};
public:
    Account(string name = "none", double balance = 0.0);
    ~Account();
    bool deposit(double amount);
    void printInfo() const;
    double getBalance();
};

Account::Account(string name, double balance)
    :name {name}, balance {balance}
{
    cout << "构造函数，name: " << name << endl;
}

Account::~Account()
{
    cout << "析构函数，name: " << name << endl;
}
bool Account::deposit(double amount)
{
    balance += amount;
    return true;
}

void Account::printInfo() const
{
    cout << "name: " << name << ", balance: " << balance << endl;
}
double Account::getBalance()
{
    return balance;
}


int main()
{
    // 直接调用指针不会释放
    // Account *bob_acc = new Account("Bob", 2000.0);
    // delete bob_acc; // 调用析构函数

    // // 会自动释放
    // unique_ptr<Account> p1 {new Account("Bob", 2000.0)};

    // // make_unique
    // auto p2 = make_unique<Account>("mike", 200.1);
    // unique_ptr<Account> p3; // unique_ptr 不允许拷贝只允许移动
    // p3 = move(p2); // p2变为null

    // auto p4 = make_unique<Account>("Helen", 200.1);
    // p4->deposit(100.0);
    // p4->printInfo();

    vector<unique_ptr<Account>> accs;
    accs.push_back(make_unique<Account>("alice", 1000));
    accs.push_back(make_unique<Account>("bob", 400));
    accs.push_back(make_unique<Account>("mike", 5000));
    
    for(const auto &acc: accs){
        cout << acc->getBalance() << endl;
    }

    return 0;
}