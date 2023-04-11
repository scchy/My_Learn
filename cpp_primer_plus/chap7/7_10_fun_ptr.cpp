#include <iostream>

using namespace std;

double betsy(int lns);
double pam(int lns);

void estimate(int lines, double (*func)(int));


int main(){
    int code;
    cout << "你需要写多少行的代码？ ";
    cin >> code;
    cout << "Besty's预估: \n";
    estimate(code, betsy);
    cout << "pam's预估: \n";
    estimate(code, pam);
    return 0;
}

double betsy(int lns){
    return 0.05 * lns;
}

double pam(int lns){
    return 0.03 * lns + 0.0004 * lns * lns;
}

void estimate(int lines, double (*func)(int)){
    cout << lines << "会花费: ";
    cout << (*func)(lines) << " hour(s)\n";
}



