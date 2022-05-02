# python3
# Create Date: 2022-05-02
# Author: Scc_hy
# Func: leet code 正则匹配问题
# ===========================================================================


def isMatch(s, p):
    if(p == ''):
        return s == '';
    if s == '':
        return False
    print(f's={s[0]} p={p[0]}')
    first_match = s[0] and ((s[0] == p[0]) or (p[0] == '.'))  if s else False
    print(f'first_match={first_match}')
    if(len(p) >= 2 and p[1] == '*'):
        return isMatch(s, p[2:]) or (first_match and isMatch(s[1:], p))

    return first_match and isMatch(s[1:], p[1:])


str_ = "ab";
pattern = ".*c*";
isMatch(str_, pattern)




