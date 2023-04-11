# include <iostream>

using namespace std;

// 多少个ch 在 str中
unsigned int c_in_str(const char* str, char ch);

// 返回C风格字符串: n个c的字符串
char * build_str(char c, int n);


int main(){
    char mmm[15] = "minimum";
    const char *wail = "ululate"; // 注意需要const
    unsigned int ms = c_in_str(mmm, 'm');
    unsigned int us = c_in_str(wail, 'u');
    cout << ms << " m char in" << mmm << endl;
    cout << us << " u char in" << wail << endl;

    char * bd = build_str('+', 6);
    cout << bd << "-DONE-" << bd << endl;
    delete [] bd; // 清除数据
    return 0;
}


unsigned int c_in_str(const char * str, char ch){
    unsigned int out = 0;
    while(*str){
        if(*str == ch){ out ++ ;}
        str++; // 移动指针
    }
    return out;
}


char * build_str(char c, int n){
    char * ptr = new char[n + 1];
    ptr[n] = '\0';
    while (n-- > 0){
        ptr[n] = c;
    }
    return ptr;
}
