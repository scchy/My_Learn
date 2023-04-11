# include <iostream>

using namespace std;
const int MAX=5;

// 填充数组
int fill_array(double arr[], int limit);

// 防止无意破坏数组需要在变量前加一个 const
void show_arr(const double arr[], int n);

// 修改数组 每个数乘以 r
void revalue(double r, double arr[], int n);


int main(){
    double properties[MAX];
    int size = fill_array(properties, MAX);
    show_arr(properties, size);
    if (size> 0){
        cout << "Enter revaluation factor: ";
        double f;
        while (! (cin >> f) ){
            cin.clear();
            while (cin.get() != '\n') continue;
            cout << "Bad input; Please enter a number: ";
        }
        revalue(f, properties, size);
        show_arr(properties, size);
    }
    cout << "Done.\n";
    cin.get();
    cin.get();
    return 0;
}


int fill_array(double arr[], int limit){
    double tmp;
    int i;
    for (i=0; i < limit; i++){
        cout << "Enter value #" << (i + 1) << ": ";
        cin >> tmp;
        if (!cin) // bad input
        {
            cin.clear();
            while(cin.get() != '\n')
                continue;
            cout << "Bad input; input process terminated.\n";
            break;
        }
        else if (tmp < 0){
            break;
        }
        arr[i] = tmp;
    }
    return i;
}


void show_arr(const double arr[], int n){
    for(int i =0 ; i<n; i++){
        cout << "Property #" << (i+1) << ": $";
        cout << arr[i] << endl;
    }
}


void revalue(double r, double arr[], int n){
    for(int i=0; i<n; i++){
        arr[i] *= r;
    }
}

