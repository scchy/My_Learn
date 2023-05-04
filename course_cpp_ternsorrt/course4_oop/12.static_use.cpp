#include <iostream>
#include <string>
#include <vector>
using namespace std;


class Account
{
private:
    string name;
    double balance;
    static int num_acc;

public:
    static int get_num_accounts();
    // const 函数不对属性进行修改
    string get_name() const{ // 获取名字
        return name;
    }
    // 构造函数
    Account();
    Account(string name = "none", double balance = 0.0);
    ~Account();
    void set_name(std::string name);
    void set_balance(double v);
    double get_balance();
};

int Account::num_acc {0};

int Account::get_num_accounts() {
    return num_acc;
};

void Account::set_name(std::string name){
    this->name = name;
}

void Account::set_balance(double v){
    this->balance = v;
}


double Account::get_balance(){
    return balance;
}
Account::Account(): Account("None", 0.0){};

Account::Account(string name, double balance)
    : balance{balance} ,name{name}{
        cout << "构造函数" << endl;
        num_acc++;
}

Account::~Account()
{
    cout << "析构函数" << endl;
    num_acc--;
}


int main()
{
    cout << Account::get_num_accounts() << endl;
    Account alice_acc {"Alice", 1000.0};
    cout << alice_acc.get_num_accounts() << endl;
    Account alice_acc1 {"Alice1", 1000.0};
    cout << alice_acc1.get_num_accounts() << endl;
    cout << Account::get_num_accounts() << endl;
    return 0;
}
