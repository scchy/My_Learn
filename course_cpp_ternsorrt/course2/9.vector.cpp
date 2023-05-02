#include <iostream>
#include <vector>
using namespace std;



int main()
{
    // vector<int> st_scores(3, 100); // 3个元素每个都是100
    vector<int> st_scores {100, 98, 97};
    cout << st_scores.size() << endl;
    // [0] 或者.at访问
    st_scores.push_back(200);
    cout << st_scores.size() << endl;

    // 获取不存在元素
    // cout << st_scores[5] << endl;
    // at会进行越界检测
    // cout << st_scores.at(5) << endl;
    
    return 0;
}