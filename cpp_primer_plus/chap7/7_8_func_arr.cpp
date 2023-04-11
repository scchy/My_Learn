// array 对象的函数
#include <iostream>
#include <array>
#include <string>

using namespace std;
const int Season=4;
const array<string, Season> Sname = {"春", "夏", "秋", "冬"};

// modify array object
void fill(array<double, Season> *pa);

// 查看array
void show(array<double, Season> pa);

int main(){
    array<double, Season> exp;
    fill(&exp);
    show(exp);
    return 0;
}

void fill(array<double, Season> * pa){
    for(int i=0; i<Season; i++){
        cout << "Enter" << Sname[i] << " expense: ";
        cin >> (*pa)[i];
    }
}


void show(array<double, Season> pa){
    double tt = 0.0;
    for(int i=0; i<Season; i++){
        cout << Sname[i] << ": $" << pa[i] << endl;
        tt += pa[i];
    }
    cout << "Total Expense: $" << tt << endl;
}