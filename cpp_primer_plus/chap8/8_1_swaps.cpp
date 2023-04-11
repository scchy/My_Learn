#include <iostream>

using namespace std;

// 按引用传递
void swapr(int & a, int & b);

// 按指针传递
void swapp(int* p, int* q);


void swapv(int a, int b);

int main(){
    int aa =300;
    int bb = 350;
    cout << "aa = " << aa << ",bb = " << bb << endl;
    cout << "Using swapr: \n";
    swapr(aa, bb);
    cout << "aa = " << aa << ",bb = " << bb << endl;

    cout << "Using swapp: \n";
    swapp(&aa, &bb);
    cout << "aa = " << aa << ",bb = " << bb << endl;

    cout << "Using swapv: \n";
    swapv(aa, bb);
    cout << "aa = " << aa << ",bb = " << bb << endl;

    return 0;
}


void swapr(int & a, int & b){
    int tmp;
    tmp = a; a = b; b = tmp;
}

void swapp(int * a, int * b){
    int tmp;
    tmp = *a; *a = *b; *b = tmp;
}

void swapv(int a, int b){
    int tmp;
    tmp = a; a = b; b = tmp;
}



