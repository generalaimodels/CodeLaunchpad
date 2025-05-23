#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python Built-in Functions: min(), abs(), round()
================================================

This module provides a comprehensive guide to these essential Python built-in functions
with examples, edge cases, and practical applications.
"""

# =============================================================================
# min() - Find the smallest item in an iterable or smallest of multiple arguments
# =============================================================================

"""
min() Function Overview:
-----------------------
Signature:
    min(iterable, *[, key, default])  # For iterables
    min(arg1, arg2, *args[, key])     # For multiple arguments

Parameters:
    - iterable: An iterable (list, tuple, string, etc.)
    - arg1, arg2, *args: Two or more comparable objects
    - key: Optional function to extract comparison value
    - default: Optional return value if iterable is empty (Python 3.4+)

Return Value:
    The smallest item in the iterable or smallest of the given arguments
"""

# Basic Usage Examples
# -------------------

# Example 1: Finding minimum of multiple arguments
min_value = min(5, 3, 9, 1, 7)
# Output: 1

# Example 2: Finding minimum in a list
numbers = [5, 3, 9, 1, 7]
min_in_list = min(numbers)
# Output: 1

# Example 3: Finding lexicographically smallest character in a string
min_char = min("PYTHON")
# Output: 'H' (ASCII order)

# Advanced Usage Examples
# ---------------------

# Example 4: Using key parameter with lambda for custom comparison
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]

# Find student with minimum grade
lowest_grade = min(students, key=lambda student: student["grade"])
print("Student with lowest grade:", lowest_grade)

# Output: {'name': 'Charlie', 'grade': 78}

# Example 5: Using the default parameter (Python 3.4+)
empty_list = []
try:
    # This raises ValueError: min() arg is an empty sequence
    min_empty = min(empty_list)
except ValueError as e:
    pass  # ValueError: min() arg is an empty sequence

# Using default parameter handles empty iterables gracefully
min_with_default = min(empty_list, default="No items")

# Output: 'No items'

# Example 6: Using min() with dictionaries
# With dictionaries, min() operates on keys by default
sample_dict = {"apple": 1.2, "banana": 0.75, "cherry": 2.5}
min_key = min(sample_dict)  
# Output: 'apple' (lexicographically smallest key)

# To find key with minimum value
min_value_key = min(sample_dict, key=sample_dict.get)
# Output: 'banana' (has smallest value 0.75)

# Example 7: Using min() with custom objects
import functools

@functools.total_ordering
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius
    
    def __eq__(self, other):
        return self.celsius == other.celsius
    
    def __lt__(self, other):
        return self.celsius < other.celsius
    
    def __repr__(self):
        return f"{self.celsius}°C"

temps = [Temperature(25), Temperature(18), Temperature(30), Temperature(22)]
min_temp = min(temps)
# Output: 18°C

# Exception Handling
# -----------------

# Example 8: TypeError with incomparable types
try:
    min([1, 2, 3j])  # Complex numbers can't be compared
except TypeError as e:
    # TypeError: '<' not supported between instances of 'complex' and 'int'
    pass

# Example 9: ValueError with empty iterable
try:
    min([])
except ValueError as e:
    # ValueError: min() arg is an empty sequence
    pass

# =============================================================================
# abs() - Return the absolute value of a number
# =============================================================================

"""
abs() Function Overview:
-----------------------
Signature:
    abs(x)

Parameters:
    - x: A number (integer, float, or complex)

Return Value:
    - For integers/floats: the absolute value (magnitude without sign)
    - For complex numbers: the magnitude (distance from zero in complex plane)
"""

# Basic Usage Examples
# -------------------

# Example 1: Absolute value of integers
abs_int = abs(-10)
# Output: 10

# Example 2: Absolute value of floating-point numbers
abs_float = abs(-3.14)
# Output: 3.14

# Example 3: Absolute value of complex numbers
# For complex number z = a + bj, abs(z) = sqrt(a² + b²)
complex_num = 3 + 4j
magnitude = abs(complex_num)
# Output: 5.0 (Pythagorean theorem: √(3² + 4²))

# Advanced Usage Examples
# ---------------------

# Example 4: Distance calculation on a number line
point_a = -15
point_b = 5
distance = abs(point_a - point_b)
# Output: 20

# Example 5: Error measurement
actual = 10.5
measured = 10.2
abs_error = abs(actual - measured)
# Output: 0.3

# Example 6: Custom class implementation
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __abs__(self):
        # Euclidean norm (distance from origin)
        return (self.x**2 + self.y**2)**0.5
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v = Vector(3, 4)
v_length = abs(v)
# Output: 5.0

# Exception Handling
# -----------------

# Example 7: TypeError with non-numeric types
try:
    abs("hello")
except TypeError as e:
    # TypeError: bad operand type for abs(): 'str'
    pass

# Example 8: TypeError with objects missing __abs__
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # No __abs__ method defined

try:
    point = Point(3, 4)
    abs(point)
except TypeError as e:
    # TypeError: bad operand type for abs(): 'Point'
    pass

# =============================================================================
# round() - Round a number to a given precision
# =============================================================================

"""
round() Function Overview:
------------------------
Signature:
    round(number[, ndigits])

Parameters:
    - number: The number to round
    - ndigits: Optional precision (decimal places). Default is 0 (round to integer)

Return Value:
    The rounded number at the specified precision

Notes:
    - Uses "banker's rounding" (round to even) for tie-breaking
    - Negative ndigits rounds to tens, hundreds, etc.
"""

# Basic Usage Examples
# -------------------

# Example 1: Rounding to integer (default)
rounded_int = round(3.14159)
# Output: 3

# Example 2: Rounding to specific decimal places
rounded_two_places = round(3.14159, 2)
# Output: 3.14

# Example 3: Rounding negative numbers
rounded_neg = round(-2.7)
# Output: -3

# Advanced Usage Examples
# ---------------------

# Example 4: Banker's rounding (round to even) for .5 cases
round_2_5 = round(2.5)  # Rounds to 2 (even)
round_3_5 = round(3.5)  # Rounds to 4 (even)
# Output: 2, 4

# Example 5: Rounding with negative ndigits
round_to_tens = round(153, -1)
round_to_hundreds = round(153, -2)
# Output: 150, 200

# Example 6: Floating-point representation issues
# Due to binary floating-point representation, some decimals aren't exactly representable
unusual_round = round(2.675, 2)
# Output: 2.67 (might be surprising, but 2.675 isn't exactly representable)

# To avoid floating-point issues, use decimal module
import decimal

decimal.getcontext().prec = 28  # Set precision
d = decimal.Decimal('2.675')
correct_round = round(d, 2)
# Output: 2.68

# Example 7: Practical applications
price = 19.99
tax_rate = 0.0725
tax = round(price * tax_rate, 2)
total = round(price + tax, 2)
# Output: tax = 1.45, total = 21.44

# Exception Handling
# -----------------

# Example 8: TypeError with non-numeric types
try:
    round("3.14")
except TypeError as e:
    # TypeError: type str doesn't define __round__ method
    pass

# Example 9: TypeError with invalid ndigits
try:
    round(3.14, "2")
except TypeError as e:
    # TypeError: 'str' object cannot be interpreted as an integer
    pass

# =============================================================================
# Performance Considerations and Best Practices
# =============================================================================

"""
Performance Tips and Best Practices:

1. min():
   - For finding multiple minimums, use heapq.nsmallest() instead of sorting
   - Use operator.itemgetter() or operator.attrgetter() instead of lambdas
   - For large datasets, consider NumPy's np.min() which is optimized

2. abs():
   - For vectorized operations on large datasets, NumPy's np.abs() is faster
   - With complex numbers, abs(z) is more efficient than manually computing magnitude

3. round():
   - For financial calculations, use decimal.Decimal to avoid floating-point errors
   - Be aware of banker's rounding behavior with ties
   - For consistent rounding in presentation layers, consider format strings:
     f"{value:.2f}" rounds for display without changing the value
"""

import operator
import heapq
import numpy as np
import time

# Example: Efficient min() with operator module
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

# More efficient for multiple uses:
get_age = operator.itemgetter("age")
youngest = min(data, key=get_age)
# Output: {"name": "Bob", "age": 25}

# Example: Using heapq for efficient access to n smallest items
scores = [85, 92, 78, 95, 88, 72, 90, 65]
three_lowest = heapq.nsmallest(3, scores)
# Output: [65, 72, 78]

# Example: NumPy performance benefits (run this with large datasets)
large_list = list(range(-500000, 500000))
large_array = np.array(large_list)

# NumPy vectorized operations are significantly faster for large datasets
# abs_array = np.abs(large_array)  # Much faster than [abs(x) for x in large_list]
# min_value = np.min(large_array)  # Faster than min(large_list)