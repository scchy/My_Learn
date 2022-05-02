# python3
# Create Date: 2022-05-02
# Author: Scc_hy
# Func: leet code 正则匹配问题
# ===========================================================================


def isMatch(s, p):
    if(p == ''):
        return s == '';

    fst = False
    if s != '':
        fst = s[0] and ((s[0] == p[0]) or (p[0] == '.'))

    print(f's={s} p={p} fst={fst}')
    if(len(p) >= 2 and p[1] == '*'):
        if len(s):
            return isMatch(s, p[2:]) or (fst and isMatch(s[1:], p))
        else:
            return isMatch(s, p[2:]) or (fst and isMatch("", p))
    return fst and isMatch(s[1:], p[1:])


if __name__ == '__main__':
    str_ = "ab";
    pattern = ".*c*";
    result = isMatch(str_, pattern)
    print(f"result={result}")

    str_ = "ab";
    pattern = ".*c";
    result = isMatch(str_, pattern)
    print(f"result={result}")




