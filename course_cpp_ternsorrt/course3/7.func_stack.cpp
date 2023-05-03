#include <iostream>
#include <vector>
#include <string>
using namespace std;


void func_2(int &x, int y, int z);
int func_1(int a, int b);

int main(){
    int x {20};
    int y {30};
    int z {};
    z = func_1(x, y);
    cout << "Z=" << z << endl;
    return 0;
}

int func_1(int a, int b){
    int res {};
    res = a + b;
    func_2(res, a, b);
    return res;
}

void func_2(int &x, int y, int z){
    x += y + z;
}

