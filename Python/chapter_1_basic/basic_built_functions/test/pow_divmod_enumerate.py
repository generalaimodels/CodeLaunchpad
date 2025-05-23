#!/usr/bin/env python3
# Built-in Functions Deep Dive: pow(), divmod(), and enumerate()
# This file demonstrates advanced usage of these built-in functions with comprehensive examples

# =============================================================================
# pow() FUNCTION
# =============================================================================
"""
pow(base, exp[, mod]) returns base raised to power exp, optionally modulo mod.

Parameters:
    base: The base number
    exp: The exponent
    mod: Optional modulus for modular exponentiation

Return Value:
    - Without mod: base raised to power exp (equivalent to base ** exp)
    - With mod: (base ** exp) % mod (but more efficient)
"""

# Basic usage
print("# Basic pow() examples")
print(f"pow(2, 3) = {pow(2, 3)}")              # 8
print(f"pow(2, 3) == 2 ** 3: {pow(2, 3) == 2 ** 3}")  # True
print(f"pow(4, 0.5) = {pow(4, 0.5)}")          # 2.0 (square root)
print(f"pow(2, -3) = {pow(2, -3)}")            # 0.125 (1/8)

# Modular exponentiation (third parameter)
print("\n# Modular exponentiation")
print(f"pow(2, 10, 17) = {pow(2, 10, 17)}")    # 4
print(f"(2 ** 10) % 17 = {(2 ** 10) % 17}")    # 4 (same result, less efficient)

# Performance advantage of modular form for large numbers
import time

def time_comparison(base, exp, mod):
    start = time.time()
    result1 = pow(base, exp, mod)
    t1 = time.time() - start
    
    start = time.time()
    result2 = (base ** exp) % mod
    t2 = time.time() - start
    
    return result1, result2, t1, t2

# Large number example
base, exp, mod = 7, 10**5, 10**9 + 7
r1, r2, t1, t2 = time_comparison(base, exp, mod)
print(f"\n# Performance comparison for pow({base}, {exp}, {mod})")
print(f"pow() with mod: {t1:.6f} seconds")
print(f"** and %: {t2:.6f} seconds")
print(f"Results match: {r1 == r2}")

# Edge cases and exceptions
print("\n# Edge cases and exceptions")
print(f"pow(0, 5) = {pow(0, 5)}")      # 0
print(f"pow(5, 0) = {pow(5, 0)}")      # 1

# ZeroDivisionError: 0 to negative power
try:
    print(pow(0, -1))
except ZeroDivisionError as e:
    print(f"pow(0, -1) raises ZeroDivisionError: {e}")

# TypeError: invalid types
try:
    print(pow("2", 3))
except TypeError as e:
    print(f"pow('2', 3) raises TypeError: {e}")

# ValueError: mod = 0
try:
    print(pow(2, 3, 0))
except ValueError as e:
    print(f"pow(2, 3, 0) raises ValueError: {e}")

# TypeError: pow with mod requires integers
try:
    print(pow(2.5, 3, 2))
except TypeError as e:
    print(f"pow(2.5, 3, 2) raises TypeError: {e}")

# Practical applications
print("\n# Practical applications of pow()")

# 1. Cryptography: Modular exponentiation in RSA
def rsa_encrypt(message, e, n):
    """Simple RSA encryption demonstration"""
    return pow(message, e, n)

message, e, n = 42, 17, 3233  # Toy example values
encrypted = rsa_encrypt(message, e, n)
print(f"RSA encryption: {message} → {encrypted}")

# 2. Fast exponentiation algorithm implementation
def fast_power(base, exp):
    """Manual implementation of fast exponentiation algorithm"""
    if exp == 0:
        return 1
    half = fast_power(base, exp // 2)
    if exp % 2 == 0:
        return half * half
    else:
        return base * half * half

print(f"fast_power(2, 10) = {fast_power(2, 10)}")  # 1024

# =============================================================================
# divmod() FUNCTION
# =============================================================================
"""
divmod(a, b) returns a tuple (a // b, a % b) of quotient and remainder.

Parameters:
    a: Dividend (number being divided)
    b: Divisor (number to divide by)

Return Value:
    Tuple containing:
    - Quotient (a // b)
    - Remainder (a % b)
"""

print("\n\n# Basic divmod() examples")
print(f"divmod(13, 5) = {divmod(13, 5)}")  # (2, 3) means 13 = 5*2 + 3
print(f"divmod(13, 5) == (13 // 5, 13 % 5): {divmod(13, 5) == (13 // 5, 13 % 5)}")  # True

# With floating point numbers
print(f"divmod(13.5, 2.5) = {divmod(13.5, 2.5)}")  # (5.0, 1.0) means 13.5 = 2.5*5 + 1.0

# Working with negative numbers
print("\n# Behavior with negative numbers")
print(f"divmod(-13, 5) = {divmod(-13, 5)}")  # (-3, 2) means -13 = 5*(-3) + 2
print(f"divmod(13, -5) = {divmod(13, -5)}")  # (-3, -2) means 13 = (-5)*(-3) + (-2)
print(f"divmod(-13, -5) = {divmod(-13, -5)}")  # (2, -3) means -13 = (-5)*(2) + (-3)

# Verification:
a, b = -13, 5
q, r = divmod(a, b)
print(f"Verifying {a} = {b} * {q} + {r}: {a == b * q + r}")  # True

# Edge cases and exceptions
print("\n# Edge cases and exceptions")

# ZeroDivisionError: division by zero
try:
    print(divmod(10, 0))
except ZeroDivisionError as e:
    print(f"divmod(10, 0) raises ZeroDivisionError: {e}")

# TypeError: unsupported operand types
try:
    print(divmod("10", 5))
except TypeError as e:
    print(f"divmod('10', 5) raises TypeError: {e}")

# TypeError: complex numbers
try:
    print(divmod(10+2j, 3))
except TypeError as e:
    print(f"divmod(10+2j, 3) raises TypeError: {e}")

# Practical applications
print("\n# Practical applications of divmod()")

# 1. Time conversion
seconds = 9876
minutes, seconds = divmod(seconds, 60)
hours, minutes = divmod(minutes, 60)
print(f"9876 seconds = {hours} hours, {minutes} minutes, {seconds} seconds")

# 2. Number base conversion
def to_base(n, base):
    """Convert decimal integer n to a string representation in given base"""
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n == 0:
        return "0"
    
    result = ""
    while n > 0:
        n, remainder = divmod(n, base)
        result = digits[remainder] + result
    
    return result

print(f"Decimal 255 in binary: {to_base(255, 2)}")       # 11111111
print(f"Decimal 255 in octal: {to_base(255, 8)}")        # 377
print(f"Decimal 255 in hexadecimal: {to_base(255, 16)}") # FF

# 3. Decimal to fraction conversion
def decimal_to_fraction(decimal, max_denominator=1000):
    """Convert decimal to approximate fraction using continued fractions"""
    whole, decimal = divmod(decimal, 1)
    numerator, denominator = 1, 0
    prev_num, prev_denom = 0, 1
    
    while denominator < max_denominator:
        if decimal == 0:
            break
            
        decimal = 1 / decimal
        quotient, decimal = divmod(decimal, 1)
        
        new_num = quotient * numerator + prev_num
        new_denom = quotient * denominator + prev_denom
        
        prev_num, prev_denom = numerator, denominator
        numerator, denominator = new_num, new_denom
    
    return int(whole * denominator + numerator), int(denominator)

pi_approx = 3.14159
num, denom = decimal_to_fraction(pi_approx)
print(f"{pi_approx} ≈ {num}/{denom} = {num/denom}")  # 355/113 = 3.1415929203539825

# =============================================================================
# enumerate() FUNCTION
# =============================================================================
"""
enumerate(iterable, start=0) adds a counter to an iterable.

Parameters:
    iterable: Any object that supports iteration
    start: Starting value for the counter (default: 0)

Return Value:
    An enumerate object (iterator) that yields tuples containing a count and 
    the values from the original iterable
"""

print("\n\n# Basic enumerate() examples")
fruits = ['apple', 'banana', 'cherry']

# Basic usage in a for loop
print("# Iterating with enumerate()")
for i, fruit in enumerate(fruits):
    print(f"Index {i}: {fruit}")

# Custom start index
print("\n# Using a custom start index")
for i, fruit in enumerate(fruits, start=1):
    print(f"Item #{i}: {fruit}")

# Converting enumerate object to a list
print("\n# Convert to list")
enum_obj = enumerate(fruits)
enum_list = list(enum_obj)
print(f"List from enumerate: {enum_list}")  # [(0, 'apple'), (1, 'banana'), (2, 'cherry')]

# enumerate() creates an iterator (one-time use)
print(f"Trying to use exhausted iterator: {list(enum_obj)}")  # []

# Working with different iterables
print("\n# enumerate() with different iterables")

# With a string
for i, char in enumerate("Python"):
    print(f"Character at position {i}: {char}")

# With a dictionary (iterates over keys)
scores = {'Alice': 92, 'Bob': 85, 'Charlie': 78}
for i, name in enumerate(scores):
    print(f"Student {i+1}: {name} scored {scores[name]}")

# With dictionary items
for i, (name, score) in enumerate(scores.items(), 1):
    print(f"Rank {i}: {name} with score {score}")

# Advanced usage
print("\n# Advanced usage")

# Creating a dictionary from enumerated values
letters = ['a', 'b', 'c', 'd']
position_dict = {letter: pos for pos, letter in enumerate(letters, 1)}
print(f"Letter positions: {position_dict}")  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# Finding indices of matching elements
numbers = [10, 20, 30, 20, 10, 20]
indices_of_20 = [i for i, x in enumerate(numbers) if x == 20]
print(f"Indices where value is 20: {indices_of_20}")  # [1, 3, 5]

# Using with zip for parallel iteration with indices
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]
for i, (name, age) in enumerate(zip(names, ages), 1):
    print(f"Person {i}: {name} is {age} years old")

# Edge cases and exceptions
print("\n# Edge cases and exceptions")

# Empty iterable
print(f"enumerate([]) as list: {list(enumerate([]))}")  # []

# Non-iterable
try:
    list(enumerate(123))
except TypeError as e:
    print(f"enumerate(123) raises TypeError: {e}")

# Negative start index (valid)
print(f"enumerate('abc', start=-3) as list: {list(enumerate('abc', start=-3))}")
# [(-3, 'a'), (-2, 'b'), (-1, 'c')]

# Float start index (invalid)
try:
    list(enumerate([1, 2, 3], start=1.5))
except TypeError as e:
    print(f"enumerate([1,2,3], start=1.5) raises TypeError: {e}")

# Practical applications
print("\n# Practical applications of enumerate()")

# 1. Adding line numbers to a file
text = "First line\nSecond line\nThird line"
numbered_text = '\n'.join(f"{i+1}: {line}" for i, line in enumerate(text.splitlines()))
print("Text with line numbers:")
print(numbered_text)

# 2. Finding the first occurrence of an element
def first_index(iterable, value):
    """Return index of first occurrence of value in iterable, or -1 if not found"""
    for i, item in enumerate(iterable):
        if item == value:
            return i
    return -1

data = [5, 3, 9, 7, 3, 8]
print(f"First index of 3 in {data}: {first_index(data, 3)}")  # 1
print(f"First index of 10 in {data}: {first_index(data, 10)}")  # -1

# 3. Generating sequential identifiers
words = ["apple", "banana", "cherry"]
id_dict = {f"id_{i}": word for i, word in enumerate(words, 1)}
print(f"ID dictionary: {id_dict}")  # {'id_1': 'apple', 'id_2': 'banana', 'id_3': 'cherry'}

# 4. Batch processing with enumerate
items = list(range(1, 11))
batch_size = 3

def process_in_batches(items, batch_size):
    for i, item in enumerate(items):
        batch_num = i // batch_size + 1
        position = i % batch_size + 1
        print(f"Processing item {item} (batch {batch_num}, position {position})")

print("\nBatch processing example:")
# Only showing first few items to save space
for i, item in enumerate(items[:6]):
    batch_num = i // batch_size + 1
    position = i % batch_size + 1
    print(f"Item {item} is in batch {batch_num}, position {position}")