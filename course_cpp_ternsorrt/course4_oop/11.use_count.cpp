#include <iostream>
#include <string>
#include <vector>
using namespace std;


class Account
{
private:
    
    double balance {0.0};

public:
    string name {"account"};

    void set_new_name(string new_name) const{ // 修改名字
        // name = new_name;
    }
    // const 函数不对属性进行修改
    string get_name() const{ // 获取名字
        return name;
    }
    // 构造函数
    Account(string name = "none", double balance = 0.0);
    ~Account();
};

Account::Account(string name, double balance)
    : balance{balance} ,name{name}{
        cout << "构造函数" << endl;
}

Account::~Account()
{
    cout << "析构函数" << endl;
}


int main()
{
    // 常对象只能调用常函数
    const Account alice_acc {"Alice", 1000.0};
    alice_acc.set_new_name("Alice2");
    cout << alice_acc.get_name() << endl;
    return 0;
}