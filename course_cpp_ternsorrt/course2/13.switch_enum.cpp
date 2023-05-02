#include <iostream>
using namespace std;

int main()
{
    enum TF_light {red, yellow, green};
    TF_light l_color {yellow};
    switch (l_color)
    {
    case red:
        cout << "红灯" << endl;
        break;
    case yellow:
        cout << "黄灯" << endl;
        break;
    case green:
        cout << "绿灯" << endl;
        break;
    default:
        break;
    }
    return 0;
}
