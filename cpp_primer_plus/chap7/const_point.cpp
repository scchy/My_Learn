// const 和 指针的关系
#include <iostream>

using namespace std;

int main(){
    int age = 20;
    const int* pt = &age;
    cout << "Age: " << age << endl;
    cout << "*pt: " << *pt << endl;
    age = 30;
    cout << "Age: " << age << endl;
    cout << "*pt: " << *pt << endl; // pt也随着age更改
    return 0;
}