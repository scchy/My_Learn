#include<iostream>
using namespace std;

int main(){
    int room_width {0};
    int room_len {0};
    cout << "Input room width:";
    cin >> room_width;
    cout << "Input room len:";
    cin >> room_len;
    
    cout << "=======计算结果======" << endl;
    cout << room_len * room_width << endl;
    return 0;
}