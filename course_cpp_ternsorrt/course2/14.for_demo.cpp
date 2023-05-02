#include <iostream>
#include <vector>
using namespace std;

int main()
{
    for(int i {0}; i < 10; i += 2){
        cout << i << endl;
    }
    for(int i {10}; i < 10; i-- ){
        cout << i << endl;
    }
    // 两个参数
    for(int i {1}, j{5}; i <= 5; i++, j++ ){
        cout << i << " * " << j << " = "<< i*j << endl;
    }
    // for 遍历vector
    vector<int> nums {1, 2, 4, 5, 6, 7, 8, 9};
    for(unsigned i {0}; i< nums.size(); ++i){
        cout << nums.at(i) << endl;
    }
    return 0;
}