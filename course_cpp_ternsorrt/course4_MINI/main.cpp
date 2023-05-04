#include <iostream>
#include <vector>
#include <string>
#include "Account.h"
using namespace std;

void printAccount(Account acc);

int main(){
    Account j_acc;
    j_acc.set_name("jobs");
    j_acc.set_balance(1000.0);
    j_acc.deposit(400.0);
    j_acc.withdraw(900.0);

    Account alice_acc = Account("Alice", 121212.0);
    alice_acc.deposit(400.0);
    alice_acc.withdraw(900.0);
    printAccount(alice_acc);
    return 0;
}

void printAccount(Account acc){
    cout << acc.get_name() << "的余额是：" <<  acc.get_balance() << endl;
}