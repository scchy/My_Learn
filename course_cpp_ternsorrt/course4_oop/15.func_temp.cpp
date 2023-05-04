#include <iostream>
#include <string>

using namespace std;

template<typename T>
T min_func(T a, T b){
    return a < b ? a: b;
}

template<class T1, class T2>
void display(T1 a, T2 b){
    cout << a << " " << b << endl;
}

int main(){
    cout << min_func<int>(1, 2) << endl;
    cout << min_func('B', 'A') << endl;
    cout << min_func(3.2, 5.3) << endl;

    display<char, double>('A', 1.2);
    display(1, 1.2);
    display("Hello", 1.2);
    return 0;
}
