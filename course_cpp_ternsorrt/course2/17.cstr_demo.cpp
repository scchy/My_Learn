#include <iostream>
#include <cstring>
using namespace std;

int main()
{
    char f_name[20] {};
    char l_name[20] {};
    char full_name[50] {};
    cout << "输入你的姓：" << endl;
    cin >> f_name;
    cout << "输入你的名：" << endl;
    cin >> l_name;
    cout << "=====================" << endl;
    cout << "您的姓: " <<  f_name << ", 一共有" << strlen(f_name) << "个字符" << endl;
    cout << "您的名: " <<  l_name << ", 一共有" << strlen(l_name) << "个字符" << endl;
    
    strcpy(full_name, f_name);
    strcat(full_name, " ");
    strcat(full_name, l_name);
    cout << "您的全名: " <<  full_name << ", 一共有" << strlen(full_name) << "个字符" << endl;
    // cin.getlin(full_name, 50); // 读取一行，最多50个字符
    // ctrcmp 
    for(size_t i {0}; i < strlen(full_name); i++){
        if(isalpha(full_name[i])){
            full_name[i] = toupper(full_name[i]);
        }else{
            full_name[i] = '#';
        }
    }
    cout << "您的全名: " <<  full_name << ", 一共有" << strlen(full_name) << "个字符" << endl;
    return 0;
}

