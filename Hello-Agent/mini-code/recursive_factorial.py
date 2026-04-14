def factorial(n: int) -> int:
    """Compute n! recursively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


if __name__ == "__main__":
    for i in range(6):
        print(f"{i}! = {factorial(i)}")
