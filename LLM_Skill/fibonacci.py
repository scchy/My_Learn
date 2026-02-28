"""
斐波那契数列计算模块
提供多种实现方式：递归、迭代、动态规划
"""

from functools import lru_cache
from typing import List, Iterator


def fibonacci_recursive(n: int) -> int:
    """
    递归实现 - 简单直观但效率较低
    时间复杂度: O(2^n)
    空间复杂度: O(n)
    
    Args:
        n: 斐波那契数列的第n项 (n >= 0)
    
    Returns:
        第n项的值
    
    Raises:
        ValueError: 当n为负数时
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


@lru_cache(maxsize=None)
def fibonacci_memoization(n: int) -> int:
    """
    带缓存的递归实现 - 效率较高
    时间复杂度: O(n)
    空间复杂度: O(n)
    
    Args:
        n: 斐波那契数列的第n项 (n >= 0)
    
    Returns:
        第n项的值
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n <= 1:
        return n
    return fibonacci_memoization(n - 1) + fibonacci_memoization(n - 2)


def fibonacci_iterative(n: int) -> int:
    """
    迭代实现 - 效率高且空间占用小（推荐）
    时间复杂度: O(n)
    空间复杂度: O(1)
    
    Args:
        n: 斐波那契数列的第n项 (n >= 0)
    
    Returns:
        第n项的值
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_sequence(n: int) -> List[int]:
    """
    生成前n项斐波那契数列
    
    Args:
        n: 要生成的项数 (n >= 0)
    
    Returns:
        包含前n项的列表
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence


def fibonacci_generator() -> Iterator[int]:
    """
    斐波那契数列生成器 - 可无限生成
    
    Yields:
        斐波那契数列的每一项
    """
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# 别名，默认使用迭代实现
fibonacci = fibonacci_iterative


if __name__ == "__main__":
    # 测试代码
    print("=== 斐波那契数列测试 ===\n")
    
    # 测试各项实现
    test_n = 10
    print(f"第 {test_n} 项的值:")
    print(f"  递归实现: {fibonacci_recursive(test_n)}")
    print(f"  缓存递归: {fibonacci_memoization(test_n)}")
    print(f"  迭代实现: {fibonacci_iterative(test_n)}")
    print(f"  默认实现: {fibonacci(test_n)}\n")
    
    # 生成数列
    print(f"前 15 项斐波那契数列:")
    print(f"  {fibonacci_sequence(15)}\n")
    
    # 使用生成器
    print("使用生成器生成前 10 项:")
    gen = fibonacci_generator()
    print(f"  {[next(gen) for _ in range(10)]}\n")
    
    # 大数测试
    big_n = 100
    print(f"第 {big_n} 项的值 (迭代实现):")
    print(f"  {fibonacci_iterative(big_n)}")
