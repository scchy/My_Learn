/*
* array容器示例
* array大小固定，不可改变
* 在内存中是连续的
* 获取元素的复杂度是常数，与array元素个数无关
* 是对原始数组的封装，也可以获取原始数组的指针
* 如果数组大小固定，尽量使用array，而不是使用C++原生数组，因为array可以使用标准库的算法
*/

#include <iostream>
#include <array> // 使用array容器
#include <algorithm> 
#include <numeric> 

using namespace std;


// 打印数组
void display(const std::array<int,5> &arr)
{
    std::cout << "[ ";
    for (const auto &a: arr)
        std::cout << a << " ";
    std::cout << "]" << std::endl;
}



void test1()
{
    std::cout << "test1 ======================" << std::endl;
    array<int, 5> arr1 {1, 2, 3, 4, 5};
    std::array<int, 5> arr2;

    display(arr1);
    display(arr2); // 未初始化，值为随机值

    arr2 = {10, 20, 30, 40, 50}; // 可以直接赋值

    display(arr1);
    display(arr2); 

    std::cout << "arr1的大小：" << arr1.size() << std::endl;
    std::cout << "arr2的大小：" << arr2.size() << std::endl;

    arr1[0] = 1000;
    arr1.at(1) = 2000;
    display(arr1);

    std::cout << "arr1的第一个元素：" << arr1.front() << std::endl;
    std::cout << "arr1的最后一个元素：" << arr1.back() << std::endl;
}

void test2()
{
    std::cout << "test2 ======================" << std::endl;
    std::array<int, 5> arr1 {1, 2, 3, 4, 5};
    std::array<int, 5> arr2 {10, 20, 30, 40, 50};

    display(arr1);
    display(arr2);

    arr1.fill(0);
    display(arr1);
    display(arr2);

    // 交换
    arr1.swap(arr2);
    display(arr1);
    display(arr2);
}

void test3()
{
    std::cout << "test3 ======================" << std::endl;
    array<int, 5> arr1 {1, 2, 3, 4, 5};
    int *ptr = arr1.data(); // 返回数组的首地址
    cout << ptr << endl;
    cout << *ptr << endl;
    *ptr = 1000;
    display(arr1);
}

void test4()
{
    std::cout << "test4-sort\n======================" << std::endl;
    std::array<int, 5> arr1 {3,1,4,2,5};
    display(arr1);
    sort(arr1.begin(), arr1.end());
    display(arr1);
}

void test5()
{
    std::cout << "test5-max min\n======================" << std::endl;
    std::array<int, 5> arr1 {3,6,4,2,5};
    std::array<int, 5>::iterator min_val = min_element(arr1.begin(), arr1.end());
    auto max_val = max_element(arr1.begin(), arr1.end());
    std::cout << "min: " << *min_val << std::endl;
    std::cout << "max: " << *max_val << std::endl;
}


void test6()
{
    std::cout << "test6 adjacent_find\n======================" << std::endl;
    array<int, 5> arr1 {3,6,2,2,5};
    auto adj = adjacent_find(arr1.begin(), arr1.end()); // 查找相邻的两个相同的元素
    if(adj != arr1.end())
        cout << "adjacent: " << *adj << std::endl;
    else
        cout << "没有找到相邻的两个相同的元素" << std::endl;

}


void test7()
{
    std::cout << "test7-accumulate\n======================" << std::endl;
    std::array<int, 5> arr1 {1,2,3,4,5};
    int sum = accumulate(arr1.begin(), arr1.end(), 0); // 求和
    std::cout << "sum: " << sum << std::endl;
}


int main(){
    // test1();
    // test2();
    // test3();
    // test4();
    test5();
    test6();
    test7();
    return 0;
}