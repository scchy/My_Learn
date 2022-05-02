/*
    *cpp
    *Create Date: 2022-05-02
    *Author: Scc_hy
    *Func: leet code 正则匹配问题
*/
#include<iostream>
using namespace std;


class Solution{
public:
    bool isMatch(string s, string p){
        return __isMatch(s.c_str(), p.c_str());
    }

private:
    bool __isMatch(const char *s, const char *p){
        if(*p == 0) return *s == 0;
        bool fst = *s && (*s == *p || *p == '.');
        if(*(p+1) == '*'){
            return isMatch(s, p+2) || (fst && isMatch(s+1, p));
        }else{
            return fst && isMatch(s+1, p+1);
        }
    }

};


int main(){
    string str_ = "ab";
    string pattern = ".*";
    Solution sl=Solution();
    bool out;
    
    cout << "=====================" << endl;
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;

    cout << "=====================" << endl;
    str_ = "aaa";  pattern = "a*";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;

    cout << "=====================" << endl;
    str_ = "aaa";  pattern = "a*c*";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;


    cout << "=====================" << endl;
    str_ = "ab";  pattern = ".*c";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;

    cout << "=====================" << endl;
    str_ = "aab";  pattern = "c*a*b";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;

    cout << "=====================" << endl;
    str_ = "mississippi";  pattern = "mis*is*p*.";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;


    cout << "=====================" << endl;
    str_ = "aaa";  pattern = "aaaa";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;

    cout << "=====================" << endl;
    str_ = "aaa";  pattern = "ab*a*c*a";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;


    cout << "=====================" << endl;
    str_ = "ab";  pattern = ".*c*";
    out = sl.isMatch(str_, pattern);
    cout << "s=" << str_ << " pattern=" << pattern << " | result=" << out << endl;
    return 0;

}