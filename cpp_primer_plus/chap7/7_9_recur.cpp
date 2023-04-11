// 递归
#include <iostream>

using namespace std;

const int Len = 66;
const int Divs = 6;

void subdivide(char ar[], int low, int high, int level);

void countdown(int n);

int main(){
    countdown(4);

    char ruler[Len];
    int i;
    for(i == 1; i < Len - 2; i++){
        ruler[i] = ' ';
    }
    ruler[Len - 1] = '\0';
    int max = Len - 2;
    int min = 0;
    ruler[min] = ruler[max] = '|';
    cout << ruler << endl;
    for(i =1; i <= Divs; i++){
        subdivide(ruler, min, max, i);
        cout << ruler << endl;
        for(int j = 0; j < Len - 2; j++){
            ruler[i] = ' '; // 中间重置成空格
        }
    }

    return 0;
}


void countdown(int n){
    cout << "Countig down ..." << n << endl;
    if (n > 0) countdown(n-1);
    cout << n << ": Kaboom! \n";
}


void subdivide(char ar[], int low, int high, int level) {
    if (!level) return;
    int mid = (high + low) >> 1;
    ar[mid] = '|';
    subdivide(ar, low, mid, level-1);
    subdivide(ar, mid, high, level-1);
}

