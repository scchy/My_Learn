#include <iostream>
#include <vector>
#include <string>
using namespace std;


int main(){
    int my_num {10};
    cout << "my_num变量的值是：" << my_num << endl;
    cout << "my_num变量的sizeof是：" << sizeof(my_num) << endl;
    cout << "my_num变量的地址是：" << &my_num << endl;

    cout << "=============================" << endl;
    int *my_ptr;
    cout << "my_ptr变量的值是：" << my_ptr << endl;
    cout << "my_ptr变量的sizeof是：" << sizeof(my_ptr) << endl;
    cout << "my_ptr变量的地址是：" << &my_ptr << endl;

    my_ptr = nullptr;
    cout << "my_ptr变量的值是：" << my_ptr << endl;

    cout << "=============================" << endl;
    int *p1 {nullptr};
    double *p2 {nullptr};
    long long *p3 {nullptr};
    string *p4 {nullptr};
    vector<string> *p5 {nullptr};
    
    cout << "int 指针的sizeof是：" << sizeof(p1) << endl;
    cout << "double 指针的sizeof是：" << sizeof(p2) << endl;
    cout << "long long 指针的sizeof是：" << sizeof(p3) << endl;
    cout << "string 指针的sizeof是：" << sizeof(p4) << endl;
    cout << "vector<string> 指针的sizeof是：" << sizeof(p5) << endl;

    cout << "=============================" << endl;
    int s_score {100};
    double h_temp {41.5};
    int *score_ptr {nullptr};
    score_ptr = &s_score;

    cout << "s_score的值是：" << s_score << endl;
    cout << "s_score的地址是：" << &s_score << endl;
    cout << "score_ptr的值是：" << score_ptr << endl;

    // score_ptr = &h_temp;
    return 0;
}

