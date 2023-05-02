#include <iostream>
#include <string>
using namespace std;

int main()
{
    string s1;
    string s2 {"hello"};
    string s3 {"hello", 3};
    string s4 {s2, 1, 3};
    string s5 (5, 'x');

    cout << "s1: " << s1 << endl;
    cout << "s2: " << s2 << endl;
    cout << "s3: " << s3 << endl;
    cout << "s4: " << s4 << endl;
    cout << "s5: " << s5 << endl;

    // 拼接
    string p1 {"C++"};
    string p2 {"强大的语言"};
    cout << p1 + p2 << endl;
    // "C++" + "强大的语言" 报错

    p1[2] = 'p';
    cout << p1 << endl;
    return 0;
}
