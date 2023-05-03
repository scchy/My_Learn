#include <iostream>
#include <cmath>
using namespace std;

const double PI {3.1415926};
void circle_area();
double calculate_area(double);

int main()
{
    circle_area();
    return 0;
}

void circle_area(){
    cout << "输入圆的半径：";
    double r {};
    cin >> r;

    cout << "圆的面积是：" << calculate_area(r) << endl;

}

double calculate_area(double r){
    return PI * pow(r, 2.0);
}