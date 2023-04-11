// lotto.cpp -- 赢的概率
# include <iostream>

using namespace std;

long double probability(unsigned, unsigned);

int main(){
    double total, choices;
    cout << "Enter the total number of choices on the game card and\n"
            "the number of picks allowed:\n";
    while( (cin >> total >> choices) && choices <= total ){
        cout << "you have  one chance in ";
        cout << probability(total, choices);
        cout << " of wining.\n";
        cout << "Next two numbers (q to quit): ";
    }
    cout << "bye\n";
    return 0;
}

long double probability(unsigned numbers, unsigned picks){
    long double res = 1.0;
    long double n;
    unsigned p;
    for (n=numbers, p = picks; p > 0; n--, p--){
        res = res * n / p;
    }
    return res;
}
