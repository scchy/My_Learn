# include<iostream>
# include<climits>
using namespace std;

int main()
{
    cout<< "size of data types: "<< endl;
    cout << "==================" << endl;
    cout << "char: "<< sizeof(char) << "bytes" << endl; 
    cout << "short: "<< sizeof(short) << "bytes" << endl; 
    cout << "int: "<< sizeof(int) << "bytes" << endl; 
    cout << "long: "<< sizeof(long) << "bytes" << endl; 
    cout << "long long: "<< sizeof(long long) << "bytes" << endl; 
    
    cout << "==================" << endl;
    cout << "float: "<< sizeof(float) << "bytes" << endl; 
    cout << "double:  "<< sizeof(double) << "bytes" << endl; 
    
    cout << "==================" << endl;
    cout<< "min and max values: "<< endl;
    cout << "char: "<< CHAR_MIN << "-" << CHAR_MAX << endl; 
    cout << "short: "<< SHRT_MIN << "-" << SHRT_MAX << endl; 
    cout << "int: "<< INT_MIN << "-" << INT_MAX << endl; 
    cout << "long: "<< LONG_MIN << "-" << LONG_MAX << endl; 
    cout << "long long: "<< LLONG_MIN << "-" << LLONG_MAX << endl; 
    
    cout << "==================" << endl;
    cout<< "size of using variable name: "<< endl;
    int age {3};
    cout << "age is " << sizeof(age) << " bytes" << endl;
    cout << endl;
    return 0;
}