/*
* vector示例
* vector是一个动态数组，可以随意增加元素
* 与array一样，vector在内存中是连续的，对应的内存空间会随着元素的增加而增加
* 获取元素的复杂度是常数，与vector的大小无关
* 在vector末尾增加、删除元素的复杂度是常数，与vector的大小无关
* 在vector中间增加、删除元素的复杂度是线性的，与vector的大小有关
* 可以使用迭代器和算法
*/

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;


// 打印vector的函数模板
template <typename T>
void printVector(const std::vector<T> &v)
{
    std::cout << "[";
    for (const auto &e : v)
        std::cout << e << " ";
    std::cout << "]" << std::endl;
}

void test1()
{
    std::cout << "test1 ======================" << std::endl;
    std::vector<int> v1{1, 2, 3, 4, 5};
    printVector(v1);

    v1 = {10, 20, 30, 40, 50}; // 可以直接赋值
    printVector(v1);

    std::vector<int> v2(10, 88); // 10个88
    printVector(v2);
}

void test2()
{
    std::cout << "test2 ======================" << std::endl;
    std::vector<int> v1{1, 2, 3, 4, 5};

    printVector(v1);
    std::cout << "size: " << v1.size() << std::endl;         // 大小
    std::cout << "capacity: " << v1.capacity() << std::endl; // 容量
    std::cout << "max_size: " << v1.max_size() << std::endl; // 最大容量

    // 以double的方式增加 capacity
    v1.push_back(6);
    printVector(v1);
    std::cout << "size: " << v1.size() << std::endl;         // 大小
    std::cout << "capacity: " << v1.capacity() << std::endl; // 容量
    std::cout << "max_size: " << v1.max_size() << std::endl; // 最大容量

    v1.shrink_to_fit(); // 释放多余的内存
    printVector(v1);
    std::cout << "size: " << v1.size() << std::endl;         // 大小
    std::cout << "capacity: " << v1.capacity() << std::endl; // 容量
    std::cout << "max_size: " << v1.max_size() << std::endl; // 最大容量

    v1.reserve(100); // 预留100个元素的空间
    printVector(v1);
    std::cout << "size: " << v1.size() << std::endl;         // 大小
    std::cout << "capacity: " << v1.capacity() << std::endl; // 容量，预留100个元素的空间后，容量恢复到100，直到超出100
    std::cout << "max_size: " << v1.max_size() << std::endl; // 最大容量
}


void test3()
{
    std::cout << "test3 ======================" << std::endl;
    std::vector<int> v1{1, 2, 3, 4, 5};
    printVector(v1);
    v1[0] = 100;
    v1.at(1) = 200;
    printVector(v1);

    std::cout << "v1的第一个元素: " << v1.front() << std::endl;
    std::cout << "v1的最后一个元素: " << v1.back() << std::endl;

    v1.pop_back(); // 删除最后一个元素
    printVector(v1);
}

void test4()
{
    vector<int> v1 {1, 2, 3, 4, 5};
    printVector(v1);

    v1.clear(); // 清空容器
    printVector(v1);

    v1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    v1.erase(v1.begin(), v1.begin() + 3); // 删除前三个元素
    printVector(v1);
    // 删除所有偶数
    v1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    printVector(v1);
    vector<int>::iterator it = v1.begin();
    while (it != v1.end())
    {
        if (*it % 2 == 0)
            v1.erase(it);
        else
            it++;
    }
    printVector(v1);
}

// 判断是否为偶数
int getEven(int x)
{
    return x % 2 == 0;
}


void test7()
{
    std::cout << "test7 ======================" << std::endl;
    std::vector<int> v1{1, 2, 3, 4, 5};
    std::vector<int> v2{10, 20};

    printVector(v1);
    printVector(v2);

    // 插入到v2的后面
    std::copy(v1.begin(), v1.end(), back_inserter(v2)); // 拷贝
    printVector(v1);
    printVector(v2);

    v1 = {1, 2, 3, 4, 5};
    v2 = {10, 20};
    // std::copy_if(v1.begin(),v1.end(),std::back_inserter(v2),getEven); // 拷贝偶数
    std::copy_if(v1.begin(), v1.end(), back_inserter(v2), 
    [](int x){ return x % 2 == 0; }); // 使用lambda表达式
    printVector(v1);
    printVector(v2);
}

void test8()
{
    std::cout << "test8 ======================" << std::endl;
    std::vector<int> v1{1, 2, 3, 4, 5};
    std::vector<int> v2{10, 20, 30, 40, 50};
    std::vector<int> v3;
    // 第一个序列的起始 - 终点， 第二个序列起始，返回到哪里， lambda
    std::transform(v1.begin(), v1.end(), v2.begin(), 
                    std::back_inserter(v3),
                   [](int x, int y){ return x + y; }); // 加法
    // std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(v3), std::plus<int>()); // 使用内置的加法函数
    std::cout << "v1 + v2 = " << std::endl;
    printVector(v3);

    v3.clear(); // 清空容器
    std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(v3),
                   [](int x, int y){ return x * y; }); // 乘法
    // std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(v3), std::multiplies<int>()); // 使用内置的乘法函数
    std::cout << "v1 * v2 = " << std::endl;
    printVector(v3);
}
void test9()
{
    std::cout << "test9 ======================" << std::endl;
    std::vector<int> v1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> v2{100, 200, 300, 400};
    printVector(v1);
    printVector(v2);

    auto it = std::find(v1.begin(), v1.end(), 5); // 查找5第一次出现的位置
    if (it != v1.end())
    {
        std::cout << "找到了:5 " << std::endl;
        v1.insert(it, v2.begin(), v2.end()); // 插入
    }
    else
    {
        std::cout << "没有找到" << std::endl;
    }

    printVector(v1);
}

int main()
{
    // test1();
    // test2();
    // test3();
    // test4();
    // test7();
    test8();
    // test9();
    return 0;
}