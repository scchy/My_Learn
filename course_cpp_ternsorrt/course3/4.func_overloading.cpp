#include <iostream>
#include <cmath>
#include <string>
#include <vector>
using namespace std;

void demo_print(int);
void demo_print(string);
void demo_print(double);
void demo_print(string, string);
void demo_print(vector<string>);


int main(){
    demo_print(100);
    demo_print(100.01);
    demo_print("AAA");
    demo_print("AAA", "BB");
    vector <string> lg {"C++", "Python", "Java"};
    demo_print(lg);
    return 0;
}

void demo_print(int a){
    cout << "Input int(" << a << ")" << endl;
};
void demo_print(string a){
    cout << "Input string(" << a << ")" << endl;
};
void demo_print(double a ){
    cout << "Input double(" << a << ")" << endl;
};
void demo_print(string a, string  b){
    cout << "Input double(" << a << ", " << b << ")" << endl;
};
void demo_print(vector<string> v){
    cout << "Input vector(";
    for(string e: v){
        cout << e << " ";
    }
    cout << ")" <<endl;
};