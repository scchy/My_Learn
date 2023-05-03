#include <iostream>
#include <vector>
#include <string>
using namespace std;

// type &name == type *const name
int *creat_array(size_t size, int init = 0);
void display(int *array, size_t size); 

int main(){
    size_t size {};
    int v {};
    int *new_arr {nullptr};
    cout << "请输入数组的大小: " ;
    cin >> size;

    cout << "请输入数组的初始值: " ;
    cin >> v;

    new_arr = creat_array(size, v);
    display(new_arr, size);
    return 0;
}

int *creat_array(size_t size, int init){
    int *new_st {nullptr};
    new_st = new int[size];
    for(size_t i{0}; i< size; i++){
        *(new_st + i) = init;
    }
    return new_st;
}

void display(int *array, size_t size){
    for(size_t i{0}; i < size; i++){
        cout << *array++ << " ";
    }
    cout << endl;
}