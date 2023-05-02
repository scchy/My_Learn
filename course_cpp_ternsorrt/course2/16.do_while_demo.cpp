#include <iostream>
using namespace std;

int main()
{
    char ip_char {};
    do{
        cout << endl;
        cout << "菜单选项: " << endl;
        cout << "1. 查看记录" << endl;
        cout << "2. 修改记录" << endl;
        cout << "3. 增加记录" << endl;
        cout << "q或Q. 退出" << endl;

        cout << "输入你的选择" << endl;
        cin >> ip_char;
        switch (ip_char)
        {
        case '1':
            cout << "查看记录" << endl;
            break;
        case '2':
            cout << "修改记录" << endl;
            break;
        case '3':
            cout << "增加记录" << endl;
            break;
        case 'q':
        case 'Q':
            cout << "退出" << endl;
            break;
        default:
            cout << "输入错误" << endl;
            break;
        }
    }while(ip_char != 'q' && ip_char != 'Q');

    return 0;
}