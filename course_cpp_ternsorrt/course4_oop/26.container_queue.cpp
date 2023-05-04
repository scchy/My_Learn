/*
* queue示例
* queue是一种容器适配器，遵循先进先出（FIFO）的原则
* queue的底层容器可以是deque、list
* 元素只能从队尾压入，从队首弹出（排队）
* queue本身没有提供迭代器
*/

#include <iostream>
#include <queue>
using namespace std;


template<typename T>
void display(queue<T> q)
{
    cout << "[ ";
    while(!q.empty())
    {
        T ele = q.front();
        cout << ele << " ";
        q.pop();
    }
    cout << "]" << endl;
}


int main()
{
    std::queue<int> q;
    for (auto i : {1, 2, 3, 4, 5})
        q.push(i);

    display(q);

    std::cout << "队首元素: " << q.front() << std::endl;
    std::cout << "队尾元素: " << q.back() << std::endl;


    q.push(100); // 压入元素
    display(q);

    q.pop(); // 弹出元素
    q.pop();

    display(q);
    // q.clear(); // 并没有clear()方法
    while (!q.empty())
        q.pop(); // 弹出所有元素
    display(q);

    std::cout << "size: " << q.size() << std::endl;

    q.push(10);
    q.push(20);
    q.push(30);
    display(q);

    std::cout << "第一个元素: " << q.front() << std::endl;
    std::cout << "最后一个元素: " << q.back() << std::endl;

    q.front() = 100; // 修改队首元素
    q.back() = 200; // 修改队尾元素
    display(q);
    return 0;
}