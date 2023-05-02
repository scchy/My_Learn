#include <iostream>
using namespace std;

int main()
{
    char ip_char {};
    cout << "请输入你的成绩：";
    cin >> ip_char;
    switch (ip_char)
    {
    case 'a':
    case 'A':
        cout << "优秀" << endl;
        break;
    case 'b':
    case 'B':
        cout << "良好" << endl;
        break;
    case 'c':
    case 'C':
        cout << "中等" << endl;
        break;
    case 'd':
    case 'D':
        cout << "不及格" << endl;
        break;
    default:
        cout << "输入错误" << endl;
    }

    return 0;
}