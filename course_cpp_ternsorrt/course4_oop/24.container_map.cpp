/*
* map示例
* map是一种关联式容器，它的元素是key-value对（std::pair），key是唯一的，value可以重复
* map中的元素是按key自动排序的
* 使用key访问元素

*/

#include <iostream>
#include <map>
#include <set>

using namespace std;

template <typename T1, typename T2>
void printMap(const std::map<T1, T2> &m)
{
    std::cout << "[";
    for (const auto &e : m)
        std::cout << e.first << ":" << e.second << " ";
    std::cout << "]" << std::endl;
}

// map的value是set
void printMap(const std::map<std::string, std::set<int>> &m)
{
    std::cout << "[";
    for (const auto &e : m)
    {
        std::cout << e.first << ":[ ";
        for (const auto &s : e.second)
            std::cout << s << " ";
        std::cout << "] ";
    }
    std::cout << "]" << std::endl;
}

void test1()
{
    std::cout << "test1 ======================" << std::endl;
    map<string, int> m1 {
        {"mike", 10},
        {"jane", 20},
        {"tom", 30}
    };
    printMap(m1);
    m1.insert(pair<string, int>("ana", 100));
    printMap(m1);
    m1.insert(std::make_pair("bob", 200)); // 插入一个pair
    printMap(m1);

    m1["jim"] = 300; // 如果key不存在，会自动插入一个pair，如果key存在，会更新value
    printMap(m1);
    m1["jim"] += 100; // 更新value
    printMap(m1);

    std::cout << "mike的计次：" << m1.count("mike") << std::endl; // count返回1表示找到，返回0表示未找到
    std::cout << "alice的计次：" << m1.count("alice") << std::endl; 
    auto it = m1.find("jim"); // find返回迭代器，如果未找到，返回end()
    if (it != m1.end())
        std::cout << "找到" << it->first << "，value为" << it->second << std::endl;
    else
        std::cout << "未找到jim" << std::endl;

    m1.clear(); // 清空map
    printMap(m1);
}


void test2()
{
    std::cout << "test2 ======================" << std::endl;
    std::map<std::string, std::set<int>> student_grades { // string是key，set是value
        {"mike", {100, 90}},
        {"jane", {99, 88, 77}},
        {"tom", {98, 87, 76}},
    };

    printMap(student_grades);

    student_grades["mike"].insert(80); // 插入80分

    printMap(student_grades);
    auto it = student_grades.find("jane");
    if (it != student_grades.end())
    {
        it->second.erase(88); // 删除88分
        printMap(student_grades);
    }
   
}



int main(){
    // test1();
    test2();
    return 0;
}