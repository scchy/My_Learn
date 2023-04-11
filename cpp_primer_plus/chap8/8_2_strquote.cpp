#include <iostream>
#include <string>

using namespace std;

string v1(const string & s1, const string & s2);
const string & v2(string & s1, const string & s2);
const string & v3(string & s1, const string & s2);


int main(){
    string input;
    string copy;
    string res;
    cout << "Enter a string: ";
    getline(cin, input);
    copy = input;

    cout << "Your string as Entered: "<< input << endl;
    res = v1(input, "***");
    cout << "v1 -> " << res << endl;
    cout << "org string -> " << input << endl;

    res = v2(input, "###");
    cout << "v2 -> " << res << endl;
    cout << "org string -> " << input << endl;

    input = copy;
    res = v3(input, "@@@");
    cout << "v3 -> " << res << endl;
    cout << "org string -> " << input << endl;
    return 0;
}