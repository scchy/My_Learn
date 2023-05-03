#include <iostream>
#include <vector>
#include <string>
using namespace std;


int main(){
    int s_score {100};
    int * score_ptr {&s_score};

    cout << "s_score的值是：" << s_score << endl;
    cout << "通过指针score_ptr访问s_score的值是：" << *score_ptr << endl;

    // 重新赋值
    *score_ptr = 150;
    cout << "Updated, s_score的值是：" << s_score << endl;
    cout << "Updated, 通过指针score_ptr访问s_score的值是：" << *score_ptr << endl;

    cout << "===================================" << endl;
    double h {41.5};
    double l {37.5};
    double *tmp_ptr {&h};
    cout << "通过指针tmp_ptr访问 h 的值是：" << *tmp_ptr << endl;
    tmp_ptr = &l;
    cout << "通过指针tmp_ptr访问 l 的值是：" << *tmp_ptr << endl;

    cout << "===================================" << endl;
    string str {"Hello"};
    string *str_ptr {&str};
    cout << "通过指针str_ptr访问 str 的值是：" << *str_ptr << endl;
    str = "World";
    cout << "Ypdated, 通过指针str_ptr访问 str 的值是：" << *str_ptr << endl;

    cout << "===================================" << endl;
    vector <string> my_str_vec {"Hello", "World", "computer", "version"};
    vector <string> *vec_ptr {&my_str_vec};
    cout << "my_str_vec的第一个元素是：" << my_str_vec.at(0) << endl; 
    cout << "通过指针vec_ptr访问 my_str_vec的第一个元素是：" << (*vec_ptr).at(0) << endl;
    return 0;
}