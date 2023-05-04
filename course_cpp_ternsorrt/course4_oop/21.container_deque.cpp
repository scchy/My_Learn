/*
* deque（double ended queue，双端数组）示例
* 动态数组，和vector类似，但是deque是双端的，可以在头部和尾部进行插入和删除操作
* 与vector不同，deque在内存中是分段连续的，每段内存都是连续的，所以在头部和尾部插入和删除元素都很快
* 获取元素的复杂度是常数
* 在头部和尾部插入和删除元素的复杂度是常数
* 在中间插入和删除元素的复杂度是线性的
* 支持迭代器和算法
*/
#include <iostream>
#include <deque>
#include <vector>
#include <algorithm>
using namespace std;

// 用于显示deque的函数模板
template <typename T>
void display(const std::deque<T> &d)
{
    std::cout << "[ ";
    for (const auto &item:d )
        std::cout << item << " ";
    std::cout << "]";
    std::cout << std::endl;
}

void test1()
{
    std::cout << "test1 ======================" << std::endl;
    std::deque<int> d1{1, 2, 3, 4, 5};
    display(d1);

    std::deque<int> d2 (10,100);
    display(d2);
    d2[0] = 99;
    d2.at(1) = 88;
    display(d2);
}

void test2()
{
    std::cout << "test2 ======================" << std::endl;
    std::deque<int> d1 {0,0,0,0};
    display(d1);

    d1.push_back(10);
    d1.push_back(20);
    display(d1);

    d1.push_front(100);
    d1.push_front(200);
    display(d1);

    std::cout << "第一个元素: " << d1.front() << std::endl;
    std::cout << "最后一个元素: " << d1.back() << std::endl;
    std::cout << "大小: " << d1.size() << std::endl;

    d1.pop_back();
    d1.pop_front();
    display(d1);
}

void test3()
{
    std::cout << "test3 ======================" << std::endl;
    std::vector<int> v1 {1,2,3,4,5,6,7,8,9,10};
    std::deque<int> d2;

    // 将vector中的偶数放入deque后，奇数放入deque前
    for (const auto &item:v1)
    {
        if (item % 2 == 0)
            d2.push_back(item);
        else
            d2.push_front(item);
    }
    display(d2);
}

void test4()
{
    std::cout << "test4 ======================" << std::endl;
    std::vector<int> v1 {1,2,3,4,5,6,7,8,9,10};
    std::deque<int> d2;

    // 将vector中的元素放到d2后
    for (const auto &item:v1)
        d2.push_back(item);
    display(d2);

    d2.clear(); // 清空deque
    // 将vector中的元素放到d2前
    for (const auto &item:v1)
        d2.push_front(item);
    display(d2);

}

void test5()
{
    std::cout << "test5 ======================" << std::endl;
    std::vector<int> v1 {1,2,3,4,5,6,7,8,9,10};
    std::deque<int> d2;

    copy(v1.begin(), v1.end(), back_inserter(d2));
    display(d2);

    d2.clear();
    copy(v1.begin(), v1.end(), front_inserter(d2));
    display(d2);
}


int main()
{
    test1();
    test2();
    test3();
    test4();
    test5();
    return 0;
}
