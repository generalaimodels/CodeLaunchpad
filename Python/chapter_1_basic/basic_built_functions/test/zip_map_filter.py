#!/usr/bin/env python3
"""
Python Built-in Functions: zip, map, filter
=======================================

This module demonstrates advanced usage of Python's functional programming 
built-in functions: zip, map, and filter. These functions enable more concise, 
readable, and efficient code when working with iterables.
"""

#############################################################################
# 1. ZIP FUNCTION
#############################################################################
"""
zip(*iterables) -> zip object

Purpose: Aggregates elements from multiple iterables into tuples.

Key characteristics:
- Returns an iterator of tuples (lazy evaluation)
- Stops when the shortest input iterable is exhausted
- Can be used with different types of iterables
- Can be unpacked with the * operator

Syntax:
    zip_object = zip(iterable1, iterable2, ...)
    result_list = list(zip_object)
"""

# Example 1: Basic zip usage
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
scores = [95, 85, 90]

# Combining two iterables
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Example 2: Converting zip output to list
pairs = list(zip(names, ages))
print(f"\nPairs as list: {pairs}")  # [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# Example 3: Zipping three or more iterables
for name, age, score in zip(names, ages, scores):
    print(f"{name} is {age} years old and scored {score}")

# Example 4: Unequal length iterables - zip stops at shortest
names_extended = ["Alice", "Bob", "Charlie", "David", "Eva"]
ages_shorter = [25, 30]
result = list(zip(names_extended, ages_shorter))
print(f"\nUnequal lengths: {result}")  # Only pairs for the first two names

# Example 5: Using zip with different iterable types
tuple_data = ("A", "B", "C")
set_data = {10, 20, 30}  # Set order not guaranteed!
dict_keys = {"x": 1, "y": 2, "z": 3}.keys()
mixed_zip = list(zip(tuple_data, set_data, dict_keys))
print(f"\nMixed iterables: {mixed_zip}")

# Example 6: Unzipping (unpacking zipped items)
pairs = [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
names, ages = zip(*pairs)  # Unpacking
print(f"\nUnzipped names: {names}")  # ('Alice', 'Bob', 'Charlie')
print(f"Unzipped ages: {ages}")     # (25, 30, 35)

# Example 7: Creating dictionaries with zip
keys = ["name", "age", "job"]
values = ["Alice", 30, "Engineer"]
person_dict = dict(zip(keys, values))
print(f"\nDictionary from zip: {person_dict}")

# Example 8: zip_longest from itertools (keep all items)
from itertools import zip_longest
names_short = ["Alice", "Bob"]
ages_long = [25, 30, 35, 40]
result = list(zip_longest(names_short, ages_long, fillvalue="Unknown"))
print(f"\nzip_longest result: {result}")
# [('Alice', 25), ('Bob', 30), ('Unknown', 35), ('Unknown', 40)]

# Example 9: Empty zip behavior
empty_zip = list(zip())
print(f"\nEmpty zip result: {empty_zip}")  # []

# Example 10: Exception handling with zip
try:
    # TypeError: zip argument #1 must support iteration
    result = list(zip(123, [1, 2, 3]))
except TypeError as e:
    print(f"\nError: {e}")


#############################################################################
# 2. MAP FUNCTION
#############################################################################
"""
map(function, *iterables) -> map object

Purpose: Applies a given function to each item of an iterable.

Key characteristics:
- Returns an iterator (lazy evaluation)
- Function is applied to each item in parallel across iterables
- Stops at the shortest iterable when multiple iterables provided
- Works with various function types (named functions, lambda, methods)

Syntax:
    map_object = map(function, iterable1, iterable2, ...)
    result_list = list(map_object)
"""

# Example 1: Basic map with a simple function
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))
print(f"\nSquared numbers: {squares}")  # [1, 4, 9, 16, 25]

# Example 2: Using map with a named function
def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

temps_c = [0, 10, 20, 30, 40]
temps_f = list(map(celsius_to_fahrenheit, temps_c))
print(f"\nTemperatures in Fahrenheit: {temps_f}")

# Example 3: Map with multiple iterables
def add_numbers(x, y):
    return x + y

list1 = [1, 2, 3]
list2 = [10, 20, 30]
sums = list(map(add_numbers, list1, list2))
print(f"\nSums from two lists: {sums}")  # [11, 22, 33]

# Example 4: Map with unequal length iterables
list3 = [1, 2, 3, 4, 5]
list4 = [10, 20]
result = list(map(add_numbers, list3, list4))
print(f"\nMap with unequal lengths: {result}")  # Only 2 results

# Example 5: Using map with built-in functions
words = ["hello", "world", "python", "programming"]
lengths = list(map(len, words))
print(f"\nWord lengths: {lengths}")  # [5, 5, 6, 11]

# Example 6: Map with string methods
names = ["alice", "bob", "charlie"]
capitalized = list(map(str.capitalize, names))
print(f"\nCapitalized names: {capitalized}")  # ['Alice', 'Bob', 'Charlie']

# Example 7: Handling exceptions in mapped functions
def safe_inverse(x):
    try:
        return 1/x
    except (ZeroDivisionError, TypeError):
        return None

values = [2, 0, "string", 5, None]
inverses = list(map(safe_inverse, values))
print(f"\nSafe inverses: {inverses}")  # [0.5, None, None, 0.2, None]

# Example 8: Chaining map operations
numbers = [1, 2, 3, 4, 5]
# Square numbers and convert to strings
result = list(map(str, map(lambda x: x**2, numbers)))
print(f"\nChained map operations: {result}")  # ['1', '4', '9', '16', '25']

# Example 9: Map vs. list comprehension (performance comparison)
import time

# Performance test with a large list
large_list = list(range(1000000))

# Using map
start = time.time()
map_result = list(map(lambda x: x**2, large_list))
map_time = time.time() - start

# Using list comprehension
start = time.time()
comp_result = [x**2 for x in large_list]
comp_time = time.time() - start

print(f"\nMap time: {map_time:.4f}s, List comp time: {comp_time:.4f}s")
print(f"Usually list comprehensions are slightly faster in CPython")


#############################################################################
# 3. FILTER FUNCTION
#############################################################################
"""
filter(function, iterable) -> filter object

Purpose: Constructs an iterator from elements that return True when passed to function.

Key characteristics:
- Returns an iterator (lazy evaluation)
- Function should return boolean or equivalent value
- If function is None, identity function is used (filters falsy values)
- Can be combined with map for complex data transformations

Syntax:
    filter_object = filter(function, iterable)
    result_list = list(filter_object)
"""

# Example 1: Basic filtering with a lambda function
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"\nEven numbers: {even_numbers}")  # [2, 4, 6, 8, 10]

# Example 2: Using filter with a named function
def is_prime(n):
    """Check if a number is prime"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

numbers = list(range(1, 20))
primes = list(filter(is_prime, numbers))
print(f"\nPrime numbers: {primes}")  # [2, 3, 5, 7, 11, 13, 17, 19]

# Example 3: Filter with None (removes falsy values)
mixed_data = [0, 1, False, True, "", "hello", None, [], [1, 2], {}, {"a": 1}]
truthy_values = list(filter(None, mixed_data))
print(f"\nTruthy values: {truthy_values}")  # [1, True, 'hello', [1, 2], {'a': 1}]

# Example 4: Filtering strings
words = ["apple", "banana", "cherry", "date", "elderberry", "fig"]
long_words = list(filter(lambda word: len(word) > 5, words))
print(f"\nWords longer than 5 chars: {long_words}")  # ['banana', 'elderberry']

# Example 5: Filtering dictionary items
people = {
    "Alice": 32,
    "Bob": 25,
    "Charlie": 45,
    "David": 19,
    "Eve": 27
}
# Get people older than 30
older_than_30 = dict(filter(lambda item: item[1] > 30, people.items()))
print(f"\nPeople older than 30: {older_than_30}")  # {'Alice': 32, 'Charlie': 45}

# Example 6: Combining filter with map
numbers = list(range(1, 11))
# Get squares of even numbers
even_squares = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
print(f"\nSquares of even numbers: {even_squares}")  # [4, 16, 36, 64, 100]

# Example 7: Using filter with custom objects
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

people = [
    Person("Alice", 32),
    Person("Bob", 17),
    Person("Charlie", 45),
    Person("David", 19)
]

adults = list(filter(lambda person: person.age >= 18, people))
print(f"\nAdults: {adults}")  # [Person('Alice', 32), Person('Charlie', 45), Person('David', 19)]

# Example 8: Filter with generators
def number_generator(n):
    """Generate numbers from 0 to n-1"""
    i = 0
    while i < n:
        yield i
        i += 1

# Filter even numbers from a generator
even_gen = filter(lambda x: x % 2 == 0, number_generator(10))
print(f"\nEven numbers from generator: {list(even_gen)}")  # [0, 2, 4, 6, 8]

# Example 9: Performance: filter vs. list comprehension
large_list = list(range(1000000))

# Using filter
start = time.time()
filter_result = list(filter(lambda x: x % 2 == 0, large_list))
filter_time = time.time() - start

# Using list comprehension
start = time.time()
comp_result = [x for x in large_list if x % 2 == 0]
comp_time = time.time() - start

print(f"\nFilter time: {filter_time:.4f}s, List comp time: {comp_time:.4f}s")


#############################################################################
# COMBINING ZIP, MAP, AND FILTER TOGETHER
#############################################################################
"""
These functions can be combined to create powerful data processing pipelines.
The examples below demonstrate common patterns for combining these functions.
"""

# Example 1: Processing related data with zip, filter, and map
names = ["Alice", "Bob", "Charlie", "David", "Eva"]
ages = [25, 17, 30, 16, 22]
scores = [85, 92, 78, 95, 88]

# Find adult students with passing scores (>=80)
adult_passers = list(filter(
    lambda data: data[1] >= 18 and data[2] >= 80,  # Filter condition
    zip(names, ages, scores)  # Combine data
))

# Format the results
formatted_results = list(map(
    lambda data: f"{data[0]} (Age: {data[1]}, Score: {data[2]})",
    adult_passers
))

print("\nAdult students with passing scores:")
for result in formatted_results:
    print(f"- {result}")

# Example 2: Matrix addition using zip and map
matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6], [7, 8]]

# Add corresponding elements of two matrices
matrix_sum = list(map(
    lambda rows: list(map(
        lambda elements: elements[0] + elements[1],
        zip(*rows)
    )),
    zip(matrix_a, matrix_b)
))

print(f"\nMatrix A: {matrix_a}")
print(f"Matrix B: {matrix_b}")
print(f"Matrix Sum: {matrix_sum}")  # [[6, 8], [10, 12]]

# Example 3: Data transformation and filtering pipeline
customer_data = [
    {"name": "Alice", "purchases": [120, 900, 50, 35]},
    {"name": "Bob", "purchases": [20, 10, 30]},
    {"name": "Charlie", "purchases": [250, 1000, 150]},
    {"name": "David", "purchases": [30, 45, 60, 20]}
]

# Calculate total purchases for each customer
with_totals = list(map(
    lambda customer: {**customer, "total": sum(customer["purchases"])},
    customer_data
))

# Filter high-value customers (total > 500)
high_value_customers = list(filter(
    lambda customer: customer["total"] > 500,
    with_totals
))

# Format for display
formatted_customers = list(map(
    lambda customer: f"{customer['name']} (Total: ${customer['total']})",
    high_value_customers
))

print("\nHigh-value customers:")
for customer in formatted_customers:
    print(f"- {customer}")