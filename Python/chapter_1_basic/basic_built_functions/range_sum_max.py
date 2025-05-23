#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Built-in Functions Tutorial: range(), sum(), max()
This comprehensive guide demonstrates advanced usage, edge cases, and best practices.
"""

################################################################################
# range() Function
################################################################################
"""
range() - Creates a sequence of numbers
Syntax:
    range(stop)
    range(start, stop)
    range(start, stop, step)

Parameters:
    start: Starting value (inclusive), defaults to 0
    stop: Ending value (exclusive)
    step: Increment between values, defaults to 1

Returns:
    A range object (immutable sequence type)
"""

# Basic usage - range with one parameter (stop)
print("Basic range(5):", list(range(5)))  # [0, 1, 2, 3, 4]

# Two parameters (start, stop)
print("range(2, 7):", list(range(2, 7)))  # [2, 3, 4, 5, 6]

# Three parameters (start, stop, step)
print("range(1, 10, 2):", list(range(1, 10, 2)))  # [1, 3, 5, 7, 9]

# Negative step for counting down
print("range(10, 0, -1):", list(range(10, 0, -1)))  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# Common use case: iterating with an index
fruits = ["apple", "banana", "cherry"]
for i in range(len(fruits)):
    print(f"Index {i}: {fruits[i]}")

# Memory efficiency: range objects don't store all values in memory
import sys
numbers_list = list(range(1000))
numbers_range = range(1000)
print(f"Memory: list={sys.getsizeof(numbers_list)} bytes, range={sys.getsizeof(numbers_range)} bytes")

# Range object features: indexing, slicing, length, membership testing
r = range(0, 100, 10)
print(f"r[3] = {r[3]}")  # 30
print(f"r[2:5] = {list(r[2:5])}")  # [20, 30, 40]
print(f"len(r) = {len(r)}")  # 10
print(f"50 in r: {50 in r}")  # True
print(f"55 in r: {55 in r}")  # False

# Range attributes (Python 3.3+)
r = range(5, 25, 4)
print(f"start={r.start}, stop={r.stop}, step={r.step}")  # start=5, stop=25, step=4

# Empty ranges
print("Empty range:", list(range(5, 5)))  # []
print("Empty range (negative step):", list(range(0, -5, 1)))  # []

# Exception cases
try:
    # TypeError: 'float' object cannot be interpreted as an integer
    range(1.5)
except TypeError as e:
    print(f"Exception: {e}")

try:
    # TypeError: range expected at least 1 argument, got 0
    range()
except TypeError as e:
    print(f"Exception: {e}")

try:
    # ValueError: range() arg 3 must not be zero
    range(1, 10, 0)
except ValueError as e:
    print(f"Exception: {e}")


################################################################################
# sum() Function
################################################################################
"""
sum() - Adds items of an iterable
Syntax:
    sum(iterable, start=0)

Parameters:
    iterable: Collection of items to sum
    start: Value added to the sum, defaults to 0

Returns:
    Sum of all items in the iterable plus the start value
"""

# Basic usage
print("sum([1, 2, 3, 4, 5]):", sum([1, 2, 3, 4, 5]))  # 15

# Using start parameter
print("sum([1, 2, 3], 10):", sum([1, 2, 3], 10))  # 16

# Different iterable types
print("sum((1, 2, 3)):", sum((1, 2, 3)))  # Sum of tuple: 6
print("sum({1, 2, 3}):", sum({1, 2, 3}))  # Sum of set: 6
print("sum(range(1, 6)):", sum(range(1, 6)))  # Sum of range: 15
print("sum(i for i in range(1, 6)):", sum(i for i in range(1, 6)))  # Sum of generator: 15

# Works with floats
print("sum([1.5, 2.5, 3.5]):", sum([1.5, 2.5, 3.5]))  # 7.5

# Empty iterables
print("sum([]):", sum([]))  # 0
print("sum([], 5):", sum([], 5))  # 5

# Floating point precision issues
print("sum([0.1] * 10):", sum([0.1] * 10))  # May not be exactly 1.0 due to floating-point precision

# Better precision with Decimal
from decimal import Decimal
print("sum([Decimal('0.1')] * 10):", sum([Decimal('0.1')] * 10))  # Exactly 1.0

# Common mistakes and exceptions
try:
    # TypeError: sum() can't sum strings (use ''.join(seq) instead)
    sum(['a', 'b', 'c'])
except TypeError as e:
    print(f"Exception: {e}")
    # Correct way to concatenate strings
    print("Correct way to join strings:", ''.join(['a', 'b', 'c']))

try:
    # TypeError: unsupported operand type(s) for +: 'int' and 'str'
    sum([1, 'a', 2])
except TypeError as e:
    print(f"Exception: {e}")

# Performance comparison
import time

def benchmark_sum(n=1000000):
    """Compare performance of built-in sum vs manual loop"""
    numbers = list(range(n))
    
    # Manual summation
    start = time.time()
    total = 0
    for num in numbers:
        total += num
    manual_time = time.time() - start
    
    # Built-in sum
    start = time.time()
    total = sum(numbers)
    builtin_time = time.time() - start
    
    print(f"Manual sum: {manual_time:.6f}s, Built-in sum: {builtin_time:.6f}s")
    print(f"Built-in sum is {manual_time/builtin_time:.2f}x faster")

# Run the benchmark
benchmark_sum()


################################################################################
# max() Function
################################################################################
"""
max() - Returns the largest item in an iterable or the largest of two or more arguments
Syntax:
    max(iterable, *[, key, default])
    max(arg1, arg2, *args[, key])

Parameters:
    iterable: Collection to find maximum value in
    arg1, arg2, ...: Arguments to compare
    key: Function that extracts a comparison value
    default: Value returned if iterable is empty (Python 3.4+)

Returns:
    Largest item from the inputs
"""

# Basic usage with an iterable
print("max([5, 2, 9, 1, 7]):", max([5, 2, 9, 1, 7]))  # 9

# Basic usage with multiple arguments
print("max(5, 2, 9, 1, 7):", max(5, 2, 9, 1, 7))  # 9

# Works with different comparable types
print("max('apple', 'banana', 'cherry'):", max('apple', 'banana', 'cherry'))  # 'cherry' (alphabetically)

# Using key parameter for custom comparison
numbers = [-10, -5, 3, 7, -2]
print("max(numbers):", max(numbers))  # 7 (normal comparison)
print("max(numbers, key=abs):", max(numbers, key=abs))  # -10 (largest absolute value)

# Finding longest string
words = ["apple", "banana", "cherry", "date", "elderberry"]
print("max(words, key=len):", max(words, key=len))  # 'elderberry' (longest string)

# Custom objects comparison
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score
    
    def __repr__(self):
        return f"{self.name}: {self.score}"

students = [
    Student("Alice", 85),
    Student("Bob", 92),
    Student("Charlie", 78)
]

# Find student with highest score
best_student = max(students, key=lambda s: s.score)
print(f"Best student: {best_student}")  # Bob: 92

# Using default parameter for empty iterables (Python 3.4+)
try:
    # ValueError: max() arg is an empty sequence
    max([])
except ValueError as e:
    print(f"Exception: {e}")

print("max([], default='Empty'):", max([], default='Empty'))  # 'Empty'

# When multiple items are maximal, max returns the first one
print("max([5, 9, 2, 9, 3]):", max([5, 9, 2, 9, 3]))  # 9 (first occurrence)

# Finding most common element (max with Counter)
from collections import Counter
items = [1, 2, 3, 1, 2, 1, 4, 5, 1]
counter = Counter(items)
most_common = max(counter.keys(), key=lambda x: counter[x])
print(f"Most common item: {most_common} (appears {counter[most_common]} times)")  # 1 (appears 4 times)

# Finding item with maximum value in a dictionary
prices = {"apple": 0.50, "banana": 0.30, "cherry": 0.70}
most_expensive = max(prices.items(), key=lambda x: x[1])
print(f"Most expensive fruit: {most_expensive[0]} (${most_expensive[1]:.2f})")  # cherry ($0.70)

# Memory efficient max with generator expressions
# Finding maximum even number in a range
largest_even = max((x for x in range(100) if x % 2 == 0))
print(f"Largest even number under 100: {largest_even}")  # 98

# Exception cases
try:
    # TypeError: '>' not supported between instances of 'str' and 'int'
    max([1, 'a', 2])
except TypeError as e:
    print(f"Exception: {e}")

try:
    # TypeError: '<' not supported between instances (with invalid key function)
    max([1, 2, 3], key="not_a_function")
except TypeError as e:
    print(f"Exception: {e}")

# Performance comparison between max() and sorted()[-1]
def benchmark_max(n=1000000):
    """Compare performance of max() vs sorted()[-1]"""
    import random
    numbers = [random.randint(0, n) for _ in range(n)]
    
    # Using max()
    start = time.time()
    result1 = max(numbers)
    max_time = time.time() - start
    
    # Using sorted()[-1]
    start = time.time()
    result2 = sorted(numbers)[-1]
    sorted_time = time.time() - start
    
    print(f"max(): {max_time:.6f}s, sorted()[-1]: {sorted_time:.6f}s")
    print(f"max() is {sorted_time/max_time:.2f}x faster")

# Run the benchmark
benchmark_max()