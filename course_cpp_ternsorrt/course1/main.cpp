#include <iostream>

using namespace std;

int main(){
    int sum = 0;
    for(int i = 0; i< 5;i++){
        sum += i;
    }
    cout << "Result=" << sum << endl;
    return 0;
}