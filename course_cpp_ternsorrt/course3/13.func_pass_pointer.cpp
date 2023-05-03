#include <iostream>
#include <vector>
#include <string>
using namespace std;

void double_data(int *int_ptr);


int main(){
    int v {20};
    cout << "V = " << v << endl;
    double_data(&v);
    cout << "V = " << v << endl;

    int *int_ptr {nullptr};
    int_ptr = &v;
    double_data(int_ptr);
    cout << "V = " << v << endl;
    return 0;
}

void double_data(int *int_ptr){
    *int_ptr *= 2;
}