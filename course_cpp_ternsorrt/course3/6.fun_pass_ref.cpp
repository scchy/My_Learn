#include <iostream>
#include <vector>
#include <string>
using namespace std;

void pass_by_ref_3(vector<string> &v){
    v.clear(); // 清空数组
}

void print_vector(const vector<string> &v){
    for(auto s: v)
        cout << s << " ";
    cout << endl;
}


int main(){
    vector<string> lg {"C++", "Python", "Java"};
    print_vector(lg);
    pass_by_ref_3(lg);
    print_vector(lg);
    return 0;
}


