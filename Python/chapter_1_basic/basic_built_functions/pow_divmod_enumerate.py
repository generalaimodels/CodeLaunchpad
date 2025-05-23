#!/usr/bin/env python3
"""
Python Built-in Functions Deep Dive: pow(), divmod(), enumerate()
=================================================================

This module provides a comprehensive explanation and demonstration of three
powerful Python built-in functions with examples and edge cases.
"""

# =============================================================================
# pow() - Exponentiation with optional modulo
# =============================================================================

"""
The pow() function is used for exponentiation. It can be used with either 
two or three arguments.

Syntax:
    pow(base, exp)     -> Returns base raised to the power of exp
    pow(base, exp, mod) -> Returns (base raised to the power of exp) modulo mod

Parameters:
    base: The base number
    exp: The exponent to which the base is raised
    mod: Optional modulus for modular exponentiation

Return Value:
    The result of base raised to the power of exp, or 
    (base ** exp) % mod if mod is provided

Time Complexity:
    - With 2 arguments: O(log n) where n is the exponent
    - With 3 arguments: O(log n) using efficient modular exponentiation
"""

# Basic usage of pow() with two arguments
print("# Basic pow() examples")
print(f"pow(2, 3) = {pow(2, 3)}")                # 2³ = 8
print(f"pow(5, 2) = {pow(5, 2)}")                # 5² = 25
print(f"pow(10, -2) = {pow(10, -2)}")            # 10⁻² = 0.01
print(f"pow(4, 0.5) = {pow(4, 0.5)}")            # 4^(1/2) = 2.0 (square root)

# Modular exponentiation with three arguments
print("\n# Modular exponentiation examples")
print(f"pow(2, 3, 5) = {pow(2, 3, 5)}")          # (2³) % 5 = 8 % 5 = 3
print(f"pow(10, 2, 7) = {pow(10, 2, 7)}")        # (10²) % 7 = 100 % 7 = 2

# Efficiency demonstration for large numbers
print("\n# Efficiency with large numbers")
large_exp = 10**6  # One million
# Direct calculation would compute full result then take modulo
large_mod_result = pow(2, large_exp, 10)
print(f"pow(2, 10^6, 10) = {large_mod_result}")  # Efficient calculation

"""
Advanced Notes on pow():

1. Modular Efficiency:
   When calculating pow(base, exp, mod), Python uses efficient modular
   exponentiation algorithms, which is much faster than calculating
   (base ** exp) % mod directly for large exponents.

2. Floating Point vs. Integer:
   - If any argument is a float, the result is a float
   - When using the 3-argument form, all arguments must be integers

3. Common Use Cases:
   - Cryptography (modular exponentiation is crucial)
   - Number theory problems
   - Efficient exponentiation in mathematical calculations
"""

# Edge cases and exceptions
print("\n# Edge cases and exceptions")

# Zero and one cases
print(f"pow(0, 5) = {pow(0, 5)}")                # 0
print(f"pow(5, 0) = {pow(5, 0)}")                # 1 (any number raised to power 0 is 1)
print(f"pow(1, 1000000) = {pow(1, 1000000)}")    # 1 (1 raised to any power is 1)

# Negative exponents
print(f"pow(2, -3) = {pow(2, -3)}")              # 1/2³ = 1/8 = 0.125

# Type handling
try:
    pow("2", 3)  # TypeError: unsupported operand type(s)
except TypeError as e:
    print(f"pow(\"2\", 3) -> {type(e).__name__}: {e}")

# Special cases with the 3-argument form
try:
    pow(2, -3, 5)  # ValueError: pow() 3rd argument cannot be negative when exp is negative
except ValueError as e:
    print(f"pow(2, -3, 5) -> {type(e).__name__}: {e}")

try:
    pow(2, 3, 0)  # ValueError: pow() 3rd argument cannot be 0
except ValueError as e:
    print(f"pow(2, 3, 0) -> {type(e).__name__}: {e}")

try:
    pow(2, 3.0, 5)  # TypeError: pow() 3rd argument not allowed unless all arguments are integers
except TypeError as e:
    print(f"pow(2, 3.0, 5) -> {type(e).__name__}: {e}")

"""
Comparison with operator **:
- pow(x, y) is equivalent to x ** y
- pow(x, y, z) is equivalent to (x ** y) % z, but much more efficient
  for large values of y
"""

print("\n# Comparison with ** operator")
print(f"2 ** 3 = {2 ** 3}")                      # Same as pow(2, 3)
print(f"(2 ** 3) % 5 = {(2 ** 3) % 5}")          # Same as pow(2, 3, 5)


# =============================================================================
# divmod() - Division and Modulus in One Operation
# =============================================================================

"""
The divmod() function returns a tuple containing the quotient and the remainder
when dividing two numbers.

Syntax:
    divmod(a, b)

Parameters:
    a: Dividend (numerator)
    b: Divisor (denominator)

Return Value:
    A tuple (quotient, remainder) such that:
    - quotient = a // b (integer division)
    - remainder = a % b (modulo operation)

Time Complexity:
    O(1) - Constant time operation
"""

# Basic usage
print("\n# Basic divmod() examples")
print(f"divmod(13, 5) = {divmod(13, 5)}")        # (2, 3) because 13 = 2*5 + 3
print(f"divmod(20, 7) = {divmod(20, 7)}")        # (2, 6) because 20 = 2*7 + 6
print(f"divmod(10, 2) = {divmod(10, 2)}")        # (5, 0) because 10 = 5*2 + 0

# Floating point numbers
print("\n# Floating point divmod() examples")
print(f"divmod(13.5, 2.5) = {divmod(13.5, 2.5)}")  # (5.0, 1.0)
print(f"divmod(10.0, 3) = {divmod(10.0, 3)}")      # (3.0, 1.0)

"""
Advanced Notes on divmod():

1. Equivalent Operation:
   divmod(a, b) is equivalent to (a // b, a % b), but it's more efficient
   because it performs both operations in a single function call.

2. Floating Point Behavior:
   - With floats, // performs floor division (rounds down)
   - The remainder follows the formula: a % b = a - (a // b) * b

3. Common Use Cases:
   - Converting units (e.g., converting seconds to minutes and seconds)
   - Algorithms involving division and remainder
   - Base conversion algorithms
"""

# Edge cases and exceptions
print("\n# Edge cases and exceptions")

# Negative numbers
print(f"divmod(-13, 5) = {divmod(-13, 5)}")          # (-3, 2) because -13 = -3*5 + 2
print(f"divmod(13, -5) = {divmod(13, -5)}")          # (-3, -2) because 13 = -3*-5 + -2
print(f"divmod(-13, -5) = {divmod(-13, -5)}")        # (2, -3) because -13 = 2*-5 + -3

# Zero dividend
print(f"divmod(0, 5) = {divmod(0, 5)}")              # (0, 0)

# Zero divisor
try:
    divmod(13, 0)  # ZeroDivisionError
except ZeroDivisionError as e:
    print(f"divmod(13, 0) -> {type(e).__name__}: {e}")

# Type errors
try:
    divmod("13", 5)  # TypeError: unsupported operand type(s)
except TypeError as e:
    print(f"divmod(\"13\", 5) -> {type(e).__name__}: {e}")

# Practical examples
print("\n# Practical divmod() examples")

# Converting seconds to minutes and seconds
seconds = 125
minutes, remaining_seconds = divmod(seconds, 60)
print(f"{seconds} seconds = {minutes} minutes and {remaining_seconds} seconds")

# Converting decimal to binary (simplified approach)
def decimal_to_binary(n):
    """Convert a decimal number to its binary representation."""
    if n == 0:
        return "0"
    
    binary = ""
    while n > 0:
        n, remainder = divmod(n, 2)
        binary = str(remainder) + binary
    
    return binary

print(f"10 in binary = {decimal_to_binary(10)}")  # 1010
print(f"42 in binary = {decimal_to_binary(42)}")  # 101010


# =============================================================================
# enumerate() - Add Counter to an Iterable
# =============================================================================

"""
The enumerate() function adds a counter to an iterable and returns an enumerate
object, which generates pairs containing a count (from start, default=0) and
a value from the iterable.

Syntax:
    enumerate(iterable, start=0)

Parameters:
    iterable: Any iterable object (list, tuple, string, etc.)
    start: The starting index for the counter (default is 0)

Return Value:
    An enumerate object that yields tuples containing a count (index) and a value

Time Complexity:
    O(1) for creation, O(n) for iteration through all elements
"""

# Basic usage
print("\n# Basic enumerate() examples")

fruits = ["apple", "banana", "cherry"]

print("Iterating with enumerate():")
for i, fruit in enumerate(fruits):
    print(f"Index {i}: {fruit}")

# Custom start index
print("\nEnumerate with custom start index:")
for i, fruit in enumerate(fruits, start=1):
    print(f"Item #{i}: {fruit}")

# Converting to different collections
print("\nConverting enumerate() results to collections:")
enum_list = list(enumerate(fruits))
print(f"List from enumerate: {enum_list}")

enum_dict = dict(enumerate(fruits))
print(f"Dict from enumerate: {enum_dict}")

"""
Advanced Notes on enumerate():

1. Memory Efficiency:
   enumerate() is a generator that yields values on demand,
   making it memory-efficient for large iterables.

2. Unpacking in loops:
   The most common use is to unpack the index and value directly
   in a for loop: for i, value in enumerate(iterable)

3. Common Use Cases:
   - When both the item and its position in the sequence are needed
   - Tracking the line number when processing file lines
   - Creating lookup dictionaries with indexed values
"""

# Edge cases and examples
print("\n# Edge cases and examples")

# Empty iterable
print(f"list(enumerate([])) = {list(enumerate([]))}")  # []

# String enumeration (iterates over characters)
word = "Python"
print(f"list(enumerate('{word}')) = {list(enumerate(word))}")

# Using with other iterables
print("\n# Using enumerate() with different iterables")

# With a tuple
coordinates = (4, 5, 6)
for i, coord in enumerate(coordinates):
    print(f"Dimension {i+1}: {coord}")

# With a dictionary (enumerates over the keys)
user = {"name": "Alice", "age": 30, "email": "alice@example.com"}
print("\nEnumerating dictionary keys:")
for i, key in enumerate(user):
    print(f"{i+1}. {key}: {user[key]}")

# With a set
unique_numbers = {10, 20, 30, 40}
print("\nEnumerating a set (note: sets are unordered):")
for i, num in enumerate(unique_numbers, 100):
    print(f"Index {i}: {num}")

# With a file (read line by line)
print("\nEnumerating file lines:")
sample_text = "Line 1\nLine 2\nLine 3"
from io import StringIO
sample_file = StringIO(sample_text)

for line_num, line in enumerate(sample_file, 1):
    print(f"Line {line_num}: {line.strip()}")

# Practical examples
print("\n# Practical enumerate() examples")

# Finding indices of specific elements
numbers = [10, 20, 30, 20, 40, 50, 20]
indices_of_20 = [i for i, num in enumerate(numbers) if num == 20]
print(f"Indices of 20 in {numbers}: {indices_of_20}")

# Creating a numbered list from data
todos = ["Buy groceries", "Clean house", "Pay bills"]
numbered_todos = [f"{i}. {task}" for i, task in enumerate(todos, 1)]
print("\nNumbered to-do list:")
for todo in numbered_todos:
    print(todo)

# Parallel iteration with zip and enumerate
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
print("\nCombining enumerate() with zip():")
for i, (name, age) in enumerate(zip(names, ages), 1):
    print(f"Person {i}: {name} is {age} years old")

"""
Performance Considerations:

1. enumerate() vs. manual indexing:
   Using enumerate() is more efficient and readable than maintaining
   a separate counter variable.

2. Memory usage:
   Since enumerate() returns an iterator, it's memory-efficient even
   with large collections.
"""


# =============================================================================
# Additional Tips and Best Practices
# =============================================================================

"""
General Tips for Using These Built-in Functions:

1. pow()
   - Use pow(x, y, z) instead of (x ** y) % z for large exponents
   - Consider using math.sqrt() for square roots instead of pow(x, 0.5)
   - For integer-only operations, ensure all arguments are integers

2. divmod()
   - Use when you need both quotient and remainder
   - Remember the behavior with negative numbers follows Python's division rules
   - Great for unit conversions and base conversions

3. enumerate()
   - Always prefer enumerate() over manual counter variables
   - Use the start parameter when you need to begin counting from a specific number
   - Combine with other iterators like zip() for powerful data processing
"""