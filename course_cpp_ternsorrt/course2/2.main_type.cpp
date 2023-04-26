#include <iostream>
using namespace std;

// argc 代表argument count，参数数量
// argv 代表argumen vector，参数列表  

int main(int argc, char** argv)
{
    cout << "参数数量：" << argc << endl;
    cout << "==========================" << endl;
    for (int i = 0;i < argc;i++)
    {
        cout << argv[i] << endl;
    }
    return 0;
}

// 调用方式
// ./program arg1 arg2