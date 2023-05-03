#include <iostream>
#include <vector>
#include <string>
using namespace std;

void display(const vector<string> *const v); 
void display(int *array, int sentinel); 
 

int main(){
    vector<string> my_str {"apple", "banana", "orange"};
    display(&my_str);
    int s_score [] {100, 98, 90, 86, 84, -1};
    display(s_score, -1);
    return 0;
}

void display(const vector<string> *const v){
    for(auto str: *v){
        cout << str << " ";
    }
    cout << endl;
}

void display(int *array, int sentinel){
    while(*array != sentinel)
        cout << *array++ << endl;
    cout << endl;
}
 