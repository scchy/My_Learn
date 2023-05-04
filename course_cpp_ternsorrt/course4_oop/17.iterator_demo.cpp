/*
* 迭代器示例
* 迭代器可以将任意的容器抽象成一个序列，可以使用迭代器变量容器中的元素
* 迭代器设计的目的是为了解决容器与算法之间的耦合问题，与指针类似，可以通过带齐访问容器中的元素

* 迭代器的声明方式为： 容器类型::iterator ,比如：

std::vector<int>::iterator it;
std::list<int>::iterator it;
std::map<int>::iterator it;
std::set<int>::iterator it;
*/
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <list>

using namespace std;


// 打印vector元素
void printVec(const vector<int> &v)
{
    cout << "[";
    for (const auto &e : v)
        cout << e << " ";
    cout << "]" << endl;
}

void test1(){
    cout << "===============================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5};
    vector<int>::iterator it = v1.begin();
    cout << *it << endl;

    it++; // 指向第二个元素
    cout << *it << endl;
    it += 2; // 指向第四个元素
    cout << *it << endl;

    it = v1.end() -1; // 指向最后一个元素； end()指向最后的下一个位置
    cout << *it << endl;
}

void test2(){
    cout << "test2\n===============================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5};
    vector<int>::iterator it = v1.begin();
    while(it != v1.end()){
        cout << *it << endl;
        it++;
    }
    it = v1.begin();
    while(it != v1.end()){
        *it = 100;
        it++;
    }
    printVec(v1);

}

void test3(){
    cout << "test3\n===============================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5};
    // 等价于  vector<int>::const_iterator it = v1.cbegin();
    // auto it = v1.cbegin();
    vector<int>::const_iterator it = v1.begin();
    while(it != v1.end()){
        cout << *it << endl;
        it++;
    }
    // it = v1.begin();
    // while(it != v1.end()){
    //     *it = 100;
    //     it++;
    // }
    printVec(v1);
}

void test4(){
    cout << "test4\n===============================" << endl;
    vector<int> v1 {1, 2, 3, 4, 5};
    auto it = v1.rbegin(); // 返回反向迭代器，指向最后一个元素
    while(it != v1.rend()){
        cout << *it << endl;
        it++;
    }
    list<string> l1 {"hello", "world", "c++"};
    auto it2 = l1.rbegin();
    cout << *it2 << endl;
    it2++;
    cout << *it2 << endl;

    map<string, string> m1 {
        {"hello", "你好"},
        {"world", "世界"},
        {"Computer", "计算机"}
    };
    auto it3 = m1.begin();
    while(it3 != m1.end()){
        cout << it3->first << " : " << it3->second << endl;
        it3++;
    }
}

int main(){
    // test1();
    // test2();
    // test3();
    test4();
    return 0;
}

