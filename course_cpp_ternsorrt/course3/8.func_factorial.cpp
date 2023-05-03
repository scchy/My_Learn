#include <iostream>
#include <vector>
#include <string>
using namespace std;


unsigned long long factrial(unsigned long long n);

int main(){
    cout << factrial(3) << endl;
    cout << factrial(8) << endl;
    cout << factrial(12) << endl;
    return 0;
}

unsigned long long factrial(unsigned long long n){
    if (n == 0)
        return 1;
    return n * factrial(n -1);
}

