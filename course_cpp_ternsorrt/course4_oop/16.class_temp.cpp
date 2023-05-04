#include <iostream>
#include <string>
#include <vector>

using namespace std;

template<typename T>
class Item
{
private:
    string name;
    T value;
public:
    Item(string name, T value)
        : name(name), value(value){};
    string get_name() const {return name;}
    T get_value() const {return value;}
};

template<typename T1, typename T2>
struct My_pair
{
    T1 first;
    T2 second;
};

int main(){
    // Item<int> item1 {"aa", 100};
    // cout << item1.get_name() << " " << item1.get_value() << endl;

    // Item<string> item2 {"bob", "C++"};
    // cout << item2.get_name() << " " << item2.get_value() << endl;
    
    // vector<Item<double>> vec;
    // vec.push_back(Item<double>("Frank", 100.0));
    // vec.push_back(Item<double>("George", 200.0));
    // vec.push_back(Item<double>("Harry", 400.0));

    // for(const auto &item: vec){
    //     cout << item.get_name() << " " << item.get_value() << endl;
    // }

    My_pair<string, int> p1{"hello", 100};
    My_pair<int, double> p2{100, 3.11};
    cout << p1.first << " " << p1.second << endl;
    cout << p2.first << " " << p2.second << endl;
    return 0;
}
    
