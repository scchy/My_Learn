#include <iostream>
#include <vector>
#include <string>
using namespace std;


int main(){
    int st_score [] {100, 98, 90};
    cout << "st_score的值是" <<  st_score << endl;
    int *score_ptr {st_score}; // score_ptr指向st_score数组的第一个元素
    cout << "score_ptr的值是" <<  score_ptr << endl;

    cout << "============= 数组名称，下标方式访问元素 ================" << endl;
    cout << st_score[0] << endl;
    cout << st_score[1] << endl;
    cout << st_score[2] << endl;
    cout << "============= 指针名称，下标方式访问元素 ================" << endl;
    cout << score_ptr[0] << endl;
    cout << score_ptr[1] << endl;
    cout << score_ptr[2] << endl;

    cout << "============= 数组名称，指针运算符方式访问元素 ================" << endl;
    cout << *st_score << endl;
    cout << *(st_score + 1) << endl;
    cout << *(st_score + 2) << endl;

    cout << "============= 指针名称，指针运算符方式访问元素 ================" << endl;
    cout << *score_ptr << endl;
    cout << *(score_ptr + 1) << endl;
    cout << *(score_ptr + 2) << endl;

    return 0;
}