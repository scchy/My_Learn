#include <iostream>
using namespace std;


void print_arr(const int arr[], size_t size);
void change_arr(int arr[], size_t size);

int main(){
    int s_scoer[] {100, 98, 99, 98};
    print_arr(s_scoer, 4);
    change_arr(s_scoer, 4);
    print_arr(s_scoer, 4);
    return 0;
}


void print_arr(const int arr[], size_t size){
    cout << arr << endl;
    for(size_t i {0}; i < size; i++)
        cout << arr[i] << " ";
    cout << endl;
}

void change_arr(int arr[], size_t size){
    cout << arr << endl;
    for(size_t i {0}; i < size; i++)
        arr[i] = 69;
}

