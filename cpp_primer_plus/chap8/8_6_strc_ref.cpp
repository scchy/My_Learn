#include <iostream>
#include <string>

using namespace std;

struct free_throws
{
    string name;
    int made;
    int attempts;
    float percent; 
};


void display(const free_throws & ft);
void set_pc(free_throws & ft);

free_throws & accumulate(free_throws & target, const free_throws & source);

int main(){
    // init
    free_throws one = {"Ifelsa Branch", 13, 14};
    free_throws two = {"Andor Knott", 10, 16};
    free_throws three = {"Minnie Max", 7, 9};
    free_throws four = {"Whily looper", 6, 14};

    free_throws tt = {"Throwgoods", 0, 0};
    free_throws dup;
    set_pc(one);
    display(one);
    accumulate(tt, one);
    display(tt);

    display(accumulate(tt, two));
    accumulate(accumulate(tt, three), four);
    display(tt);
    return 0;
}

void display(const free_throws & ft){
    cout << "Name: " << ft.name << endl;
    cout << " Made: " << ft.made << '\t';
    cout << "Attempts: " << ft.attempts << '\t';
    cout << "Percent: " << ft.percent << endl; 
}

void set_pc(free_throws & ft){
    if (ft.attempts){
        ft.percent = 100.0f * float(ft.made) / float(ft.attempts);
    }else{
        ft.percent = 0;
    }
}

free_throws & accumulate(free_throws & target, const free_throws & source){
    target.attempts += source.attempts;
    target.made += source.made;
    set_pc(target);
    return target;
}
