/*
* priority_queue （优先级队列）示例
* 允许按照优先级来插入和删除元素
* 优先级最高的元素总是位于队首（最大值在队首）
* 本身没有提供迭代器

*/
#include <iostream>
#include <queue>

using namespace std;


// 显示priority_queue的函数模板
template <typename T>
void display(std::priority_queue<T> pq)
{
    std::cout << "[";
    while (!pq.empty())
    {
        T elem = pq.top(); // 读取优先级最高元素
        std::cout << elem << " ";
        pq.pop(); // 弹出优先级最高元素
    }
    std::cout << "]" << std::endl;
}


void test1()
{
    std::cout << "test1 ======================" << std::endl;
    std::priority_queue<int> pq;
    for (auto i : {3,5,8,1,2,9,4,7,6})
        pq.push(i);
    display(pq);

    std::cout << "大小: " << pq.size() << std::endl;
    std::cout << "最大值: " << pq.top() << std::endl;

    pq.pop(); // 弹出最大值
    display(pq);
}

int main()
{
    test1();
    return 0;
}