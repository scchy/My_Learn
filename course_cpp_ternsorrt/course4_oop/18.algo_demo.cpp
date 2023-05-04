/*
* 算法示例
* STL算法基于迭代器生成的序列
* STL提供了很多算法（例如查找、排序、计数、操作），可以对序列进行操作
* 更多请查看：https://zh.cppreference.com/w/cpp/algorithm
* 多数算法要求提供额外参数，例如：排序算法需要提供排序规则，一般使用函数指针、lambda表达式或仿函数（函数对象）
*/

#include <iostream>
#include <vector>
#include <algorithm> // 算法头文件
#include <list>

using namespace std;

void test1()
{
    cout << "======================================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5};
    vector<int>::iterator loc = find(v1.begin(), v1.end(), 3); // 查找3
    if(loc != v1.end())
        cout << "找到了 3" << endl;
    else
        cout << "未找到 3" << endl;
}

void test2()
{
    cout << "======================================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5, 1, 1};
    int cnt = count(v1.begin(), v1.end(), 1);
    cout << "1的个数=" << cnt << endl;
}

bool isEven(int x){
    return x % 2 == 0;
}

void test3()
{
    cout << "======================================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5, 1, 1};
    int cnt = count_if(v1.begin(), v1.end(), isEven);
    // 使用lambda表达式
    cnt = count_if(v1.begin(), v1.end(), [](int x){return x % 2 == 0;});
    cout << "偶数个数=" << cnt << endl;

    cnt = count_if(v1.begin(), v1.end(), [](int x){return x > 3;});
    cout << "大于3的个数=" << cnt << endl;
}

void test4()
{
    cout << "======================================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5, 1, 1};
    for(const auto &e:v1)
        cout << e << " ";
    cout << endl;

    replace(v1.begin(), v1.end(), 1, 100); // 1 -> 100;
    for(const auto &e:v1)
        cout << e << " ";
    cout << endl;

}

void test5()
{
    cout << "======================================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5, 1, 1};
    if(all_of(v1.begin(), v1.end(), [](int x){return x > 3;}))
        cout << "所有元素大于3" << endl;
    else
        cout << "不是所有元素大于3" << endl; 
    
    if(any_of(v1.begin(), v1.end(), [](int x){return x > 3;}))
        cout << "存在元素大于3" << endl;
    else
        cout << "不存在元素大于3" << endl; 

    if(none_of(v1.begin(), v1.end(), [](int x){return x < 0;}))
        cout << "没有元素小于0" << endl;
    else
        cout << "有元素小于于0" << endl; 
}

void test6()
{
    cout << "======================================" << endl;
    string s1 {"hello world"};
    cout << s1 << endl;
    transform(s1.begin(), s1.end(), s1.begin(), ::toupper);
    cout << s1 << endl;

}

int main(){
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    return 0;
}