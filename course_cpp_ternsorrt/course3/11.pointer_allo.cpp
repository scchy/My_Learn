#include <iostream>
#include <vector>
#include <string>
using namespace std;


int main(){
    int *int_ptr {nullptr};
    cout << "int_ptr = " << int_ptr << endl; // 0x0
    int_ptr= new int;
    cout << "int_ptr = " << int_ptr << endl;
    cout << "解引用int_ptr = " << *int_ptr << endl;

    *int_ptr = 200;
    cout << "赋值后解引用int_ptr = " << *int_ptr << endl;

    // 赋值一段连续的存储空间
    double *double_ptr {nullptr};
    size_t size {0};
    cout << "请输入要分配的内存大小: ";
    cin >> size;
    double_ptr = new double[size];
    cout << "double_ptr = " << double_ptr << endl;
    delete [] double_ptr; // 释放内存
    return 0;
}