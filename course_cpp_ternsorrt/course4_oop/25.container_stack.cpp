/*
* stack 示例
* stack 是一种容器适配器，遵循后进先出（LIFO）的原则
* stack 本身不是容器，它是基于容器实现的（如vector、list、deque等）
* 所有的操作都在栈顶进行（top）
* stack 本身没有迭代器
*/

#include <iostream>
#include <stack>
#include <vector>
#include <list>
#include <deque>

using namespace std;


// 显示stack的函数模板
template <typename T>
void display(std::stack<T> s)
{
    std::cout << "[";
    while (!s.empty())
    {
        T elem = s.top(); // 读取栈顶元素
        std::cout << elem << " ";
        s.pop(); // 弹出栈顶元素
    }
    std::cout << "]" << std::endl;
}


int main()
{
    stack<int> s;
    for (auto i: {1,2,3,4,5})
        s.push(i);
    display(s);
    s.push(100); // 压入元素
    display(s);

    s.pop(); // 弹出元素
    s.pop();

    display(s);    
    while (!s.empty())
        s.pop(); // 弹出所有元素
    display(s);

    s.push(10);
    display(s);

    s.top() = 100; // 修改栈顶元素
    display(s);
    
    return 0;
}

