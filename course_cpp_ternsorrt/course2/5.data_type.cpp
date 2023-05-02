#include <iostream>
using namespace std;

int main(){
    // 字符型
    cout << "====================" << endl;
    char my_char {'j'}; // 注意是单引号，双引号是string类型
    cout << "my char: " << my_char << endl;


    // 整型
    cout << "====================" << endl;
    short my_score {59}; 
    cout << "my score: " << my_score << endl;

    // = 与 {} 的区别
    // short 范围-32768（-2^15） to 32767（2^15-1），overflow溢出案例
    short overflow_num_1  = 32768; // 不会报错，但是值会变成-32768
    cout << overflow_num_1 << endl; // -32768

    // short overflow_num_2 {32768}; // 编译器会做校验，会报错，所以推荐使用{}
    // cout << overflow_num_2 << endl; // -32768


    int my_height {178};
    cout << "my height: " << my_height << endl;

    long people_in_hangzhou {10360000};
    cout << "people in hangzhou: " << people_in_hangzhou << endl;

    long long people_on_earth {80'0000'0000}; // 为方便阅读c++ 14标准后支持'数字分割
    cout << "people on earth: " << people_on_earth << endl;


    // 浮点型
    cout << "====================" << endl;
    float book_price {32.23f}; // 加上f表示float类型，因为默认是double类型
    cout << "book price: " << book_price << endl;

    double pi {3.14149};
    cout << "pi: " << pi << endl;



    // 布尔型
    cout << "====================" << endl;
    bool add_to_cart {false};
    // cout << boolalpha; // 以bool的形式输出
    cout << "add to cart: " << add_to_cart << endl; // 0表示false

    return 0;
}