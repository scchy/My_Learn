/*
* set示例
* set是一种关联式容器
* 根据元素的值，自动排序，重复元素会被自动去重
* 不支持随机访问，不支持下标运算符
* 支持各种迭代器和算法

*/

#include <iostream>
#include <set>

using namespace std;

// 用于显示set的函数模板
template <typename T>
void printSet(const std::set<T> &s)
{
    std::cout << "[";
    for (const auto &e : s)
        std::cout << e << " ";
    std::cout << "]" << std::endl;
}



void test1()
{
    std::cout << "test1 ======================" << std::endl;
    std::set<int> s1{1, 2, 3, 4, 5};
    printSet(s1);

    s1 = {1,1,1,2,2,2,3,3,3}; // 重复元素会被自动去重
    printSet(s1);

    s1.insert(10); 
    s1.insert(0);
    printSet(s1);

    if (s1.count(10)) // count返回1表示找到，返回0表示未找到
        std::cout << "找到10" << std::endl;
    else
        std::cout << "未找到10" << std::endl;

    auto it = s1.find(10); // find返回迭代器，如果找到，返回迭代器指向该元素，否则返回end()
    if (it != s1.end())
        std::cout << "找到" << *it << std::endl;
    
    s1.clear();
    printSet(s1);
}


void test2()
{
    std::cout << "test2 ======================" << std::endl;
    set<std::string> s1 {"A", "B", "C", "D", "E"};
    printSet(s1);

    auto result = s1.insert("F"); // insert返回一个pair，第一个元素是迭代器，指向插入的元素，第二个元素是bool，表示是否插入成功
    printSet(s1);
    std::cout <<  boolalpha; // boolalpha表示输出true/false
    std::cout << "first: " << *(result.first) << std::endl;
    std::cout << "second: " << result.second << std::endl; // 插入成功，返回true


    result = s1.insert("A"); // A 已经存在，插入失败，但是返回的迭代器指向A
    printSet(s1);
    std::cout << std::boolalpha; // boolalpha表示输出true/false
    std::cout << "first: " << *(result.first) << std::endl;
    std::cout << "second: " << result.second << std::endl; // 插入失败，返回false，表示有重复元素


}



int main()
{
    // test1();
    test2();
    return 0;
}
