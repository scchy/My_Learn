# include<iostream>

using namespace std;

// 计算圆的么面积
int main()
{
    const double pi {3.1415926};
    cout << "输入半径： ";
    double radius;
    cin >> radius;
    cout << "面积是： " << radius * radius * pi << endl;
    return 0;
}