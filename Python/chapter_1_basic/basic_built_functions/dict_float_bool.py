#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Built-in Types: dict, float, and bool
=====================================================

This module provides a comprehensive explanation of Python's built-in
dict, float, and bool types with detailed examples, edge cases, and 
advanced usage patterns.
"""

##############################################################################
# DICTIONARIES (dict)
##############################################################################
"""
Dict is a built-in mapping type that stores key-value pairs. Dictionaries are:
- Mutable (can be changed after creation)
- Unordered in Python < 3.6 (no guaranteed order of elements)
- Ordered by insertion since Python 3.6 (implementation detail)
- Officially ordered since Python 3.7
- Optimized for O(1) average-case lookup, insertion, and deletion
"""

# 1. Dictionary Creation
empty_dict = {}                             # Empty dictionary
dict_literal = {'a': 1, 'b': 2, 'c': 3}     # Dictionary literal
dict_constructor = dict(a=1, b=2, c=3)      # Using dict() constructor with keyword arguments
dict_from_tuples = dict([('a', 1), ('b', 2), ('c', 3)])  # From list of tuples
dict_comprehension = {x: x**2 for x in range(5)}  # Dictionary comprehension

# 2. Dictionary Operations
d = {'name': 'Python', 'version': 3.9, 'released': 2020}

# Accessing values - O(1) time complexity
value = d['name']                  # Standard access - raises KeyError if key doesn't exist
safe_value = d.get('creator', 'Unknown')  # Safe access with default value if key doesn't exist

# Adding and modifying - O(1) time complexity
d['creator'] = 'Guido van Rossum'   # Add a new key-value pair
d['version'] = 3.10                 # Modify existing value

# Removing items - O(1) time complexity
del d['released']                  # Raises KeyError if key doesn't exist
popped_value = d.pop('version')    # Removes and returns value, takes optional default
last_item = d.popitem()            # Removes and returns (key, value) pair (LIFO order in 3.7+)
d.clear()                          # Removes all items

# 3. Dictionary Methods
users = {
    'user1': {'name': 'Alice', 'age': 30},
    'user2': {'name': 'Bob', 'age': 25}
}

# Keys, Values, and Items
key_view = users.keys()           # Returns a dynamic view of dictionary keys
value_view = users.values()       # Returns a dynamic view of dictionary values
item_view = users.items()         # Returns a dynamic view of (key, value) pairs

# Dictionary views are dynamic - they update when the dictionary changes
users['user3'] = {'name': 'Charlie', 'age': 35}
# Now key_view, value_view, and item_view all reflect the updated dictionary

# Merging dictionaries
defaults = {'active': True, 'admin': False}
user_with_defaults = {**defaults, **users['user1']}  # Python 3.5+ unpacking
user_with_defaults = defaults | users['user1']       # Python 3.9+ merge operator

# Update method
config = {'debug': False, 'timeout': 30}
config.update({'debug': True, 'retry': 3})  # Updates existing keys, adds new ones

# 4. Dictionary Comprehensions
squares = {x: x**2 for x in range(10)}
filtered_dict = {k: v for k, v in users.items() if v['age'] > 25}

# 5. Advanced Dictionary Operations
# Safely get nested values
def deep_get(dictionary, keys, default=None):
    """Access nested dictionary values safely."""
    result = dictionary
    for key in keys:
        try:
            result = result[key]
        except (KeyError, TypeError):
            return default
    return result

nested_data = {'a': {'b': {'c': 42}}}
value = deep_get(nested_data, ['a', 'b', 'c'])        # Returns 42
value = deep_get(nested_data, ['a', 'x', 'c'], 0)     # Returns 0 (default)

# 6. Exception Handling with Dictionaries
try:
    value = d['nonexistent_key']
except KeyError as e:
    # This block runs if key doesn't exist
    print(f"KeyError would occur: {e}")

# 7. Dictionary Performance Considerations
"""
- Hash collisions can degrade performance from O(1) to O(n)
- Dict keys must be hashable (immutable) - lists and dicts cannot be keys
- Memory overhead: dictionaries use more memory than lists for same data size
"""

# Example: Attempting to use an unhashable type (list) as a key
try:
    bad_dict = {[1, 2]: 'value'}  # This will raise TypeError
except TypeError as e:
    # This block will run
    print(f"TypeError would occur: {e}")

# 8. Dictionary Subclasses
from collections import defaultdict, OrderedDict, Counter

# defaultdict - provides default values for missing keys
word_count = defaultdict(int)  # Default value is 0 for int()
for word in "how much wood would a woodchuck chuck".split():
    word_count[word] += 1  # No KeyError for first access

# OrderedDict - maintains insertion order (less relevant since Python 3.7)
ordered = OrderedDict([('first', 1), ('second', 2)])

# Counter - specialized dict for counting hashable objects
inventory = Counter(['apple', 'banana', 'apple', 'orange', 'banana', 'banana'])
most_common = inventory.most_common(2)  # Returns [('banana', 3), ('apple', 2)]

##############################################################################
# FLOAT
##############################################################################
"""
Float is a built-in numeric type representing floating-point numbers in Python.
Floats are:
- Implemented using C's double type (IEEE 754 double-precision binary float)
- 64 bits (8 bytes) in memory
- Have approximately 15-17 significant decimal digits of precision
- Range roughly ±1.8e308 with smallest non-zero magnitude around 2.2e-308
"""

# 1. Float Creation
f1 = 3.14159                  # Decimal notation
f2 = 2.5e6                    # Scientific notation (2,500,000.0)
f3 = float(42)                # From integer
f4 = float("3.14159")         # From string
f5 = float("inf")             # Positive infinity
f6 = float("-inf")            # Negative infinity
f7 = float("nan")             # Not a Number

# 2. Float Operations
a, b = 3.0, 2.0
addition = a + b              # 5.0
subtraction = a - b           # 1.0
multiplication = a * b        # 6.0
division = a / b              # 1.5
floor_division = a // b       # 1.0 (rounds down)
modulo = a % b                # 1.0 (remainder)
power = a ** b                # 9.0 (exponentiation)

# 3. Float Precision and Representation Issues
"""
Due to binary representation of decimal values, some decimal fractions
cannot be represented exactly as binary fractions, leading to precision issues.
"""

# Example of precision issue
x = 0.1 + 0.2                   # Expected: 0.3, Actual: 0.30000000000000004
is_equal = x == 0.3             # False due to precision error

# Solutions to precision issues:
# a) Using decimal module for precise decimal arithmetic
import decimal
from decimal import Decimal

d1 = Decimal('0.1')
d2 = Decimal('0.2')
d_sum = d1 + d2                # Exactly 0.3
is_decimal_equal = d_sum == Decimal('0.3')  # True

# b) Using round or math.isclose for approximate equality
import math
is_close = math.isclose(0.1 + 0.2, 0.3)     # True with default tolerance

# c) Using a small epsilon value for comparison
epsilon = 1e-10
is_approximately_equal = abs(x - 0.3) < epsilon  # True

# 4. Special Float Values
inf = float('inf')            # Positive infinity
neg_inf = float('-inf')       # Negative infinity
nan = float('nan')            # Not a Number

# Testing for special values
import math
is_inf = math.isinf(inf)                    # True
is_nan = math.isnan(nan)                    # True
is_finite = math.isfinite(42.5)             # True

# Arithmetic with special values
inf_plus = inf + 100                        # Still inf
inf_multiply = inf * 0                      # NaN
inf_comparison = inf > 1e308                # True

# 5. Float Methods and Functions
# Round to specified precision
rounded = round(3.14159, 2)                 # 3.14
truncated = math.trunc(3.99)                # 3 (truncates to integer)
floored = math.floor(3.99)                  # 3 (largest integer <= x)
ceiled = math.ceil(3.01)                    # 4 (smallest integer >= x)

# Floating-point functions
absolute = abs(-42.5)                       # 42.5
exponential = math.exp(1.0)                 # e^1 ≈ 2.718
logarithm = math.log(10.0)                  # Natural log of 10
log_base_10 = math.log10(100.0)             # Log base 10 of 100 (2.0)
square_root = math.sqrt(16.0)               # Square root of 16 (4.0)

# 6. Float Conversion and Formatting
# Converting to other types
float_to_int = int(3.9)                     # 3 (truncates, doesn't round)
float_to_string = str(3.14159)              # "3.14159"

# Formatting options
formatted1 = f"{math.pi:.4f}"               # "3.1416" (4 decimal places)
formatted2 = "{:.2e}".format(1500.0)        # "1.50e+03" (scientific notation)
formatted3 = "{:+.1f}".format(-42.5)        # "-42.5" (with sign)

# 7. Float Exceptions and Edge Cases
# Division by zero
try:
    result = 1.0 / 0.0  # In Python, this returns inf, not an error
    # But this behavior might be unexpected
except ZeroDivisionError:
    # This won't execute for float division by zero
    pass

# Overflow and Underflow
try:
    too_large = 1.8e308 * 2.0  # Results in inf, not exception
    too_small = 2.2e-308 / 2.0  # Results in 0.0 (underflow)
except:
    # No exception raised for float overflow/underflow
    pass

# Type conversion errors
try:
    float("not a number")  # This will raise ValueError
except ValueError as e:
    # This block will run
    print(f"ValueError would occur: {e}")

##############################################################################
# BOOLEAN (bool)
##############################################################################
"""
Bool is a built-in type in Python representing binary values:
- Subclass of int with only two instances: True and False
- True has integer value 1, False has integer value 0
- Used for logical operations and control flow
"""

# 1. Boolean Creation
b1 = True                            # Boolean literal True
b2 = False                           # Boolean literal False
b3 = bool(1)                         # True (non-zero values are True)
b4 = bool(0)                         # False (zero is False)
b5 = bool("Hello")                   # True (non-empty strings are True)
b6 = bool("")                        # False (empty strings are False)
b7 = bool([1, 2, 3])                 # True (non-empty containers are True)
b8 = bool([])                        # False (empty containers are False)
b9 = bool(None)                      # False (None is False)

# 2. Boolean Operations
a, b = True, False

# Logical operations
and_result = a and b                 # False (AND)
or_result = a or b                   # True (OR)
not_result = not a                   # False (NOT)

# Comparison operations
equal = (a == b)                     # False
not_equal = (a != b)                 # True
greater = (a > b)                    # True (True > False since True=1, False=0)

# 3. Boolean in Control Flow
# If-else
if True:
    pass  # This block always executes
else:
    pass  # This block never executes

# While loop
counter = 3
while counter:  # Implicitly converts to bool
    counter -= 1  # Loop runs until counter becomes 0 (False)

# For loop with early exit
found = False
for item in [1, 2, 3, 4, 5]:
    if item == 3:
        found = True
        break  # Exit loop early

# 4. Truth Value Testing
"""
Objects can be tested for truth value in boolean contexts. 
By default, an object is considered True unless:
- It is None
- It is False
- It is zero (0, 0.0, 0j, Decimal(0), Fraction(0, 1))
- It is an empty collection ('', (), [], {}, set(), range(0))
"""

# Custom truth testing with __bool__ and __len__
class CustomBool:
    def __init__(self, value):
        self.value = value
    
    def __bool__(self):
        # Define custom truthy behavior
        return self.value > 0

# Usage:
custom_obj = CustomBool(5)
if custom_obj:  # Calls __bool__, returns True
    pass

# 5. Short-Circuit Evaluation
"""
Python uses short-circuit evaluation for boolean operations:
- x and y: if x is False, return x without evaluating y
- x or y: if x is True, return x without evaluating y
"""

# Short-circuit AND
x = False
# y = expensive_function()  # This would never execute due to short-circuit

# Short-circuit OR with default value pattern
# result = user_input or "default"  # Returns "default" if user_input is falsy

# 6. Boolean as Integer
"""
Because bool is a subclass of int, Booleans can be used in arithmetic operations.
True behaves like 1, False behaves like 0.
"""

count = sum([True, False, True, True])  # 3
indexed = [10, 20, 30][True]            # 20 (accesses index 1)

# 7. Boolean Pitfalls and Best Practices

# a) Equality vs Identity
"""
Use == for equality comparison and is for identity comparison:
- == checks if values are equal
- is checks if objects are the same instance
"""
a = [1, 2, 3]
b = [1, 2, 3]
equality = a == b    # True (values are equal)
identity = a is b    # False (different objects)

# b) Be careful with boolean expressions that mix types
result = 1 == True   # True (1 is equal to True in value)
result = 1 is True   # False (1 and True are different objects)

# c) Boolean traps in function parameters
# Bad:
def process_data(data, ignore_errors=False):
    pass

# Better (more explicit):
def process_data(data, ignore_errors=False):
    pass

# Call site is clearer with named parameters
# process_data(my_data, ignore_errors=True)

# 8. All and Any Functions
"""
- all(): Returns True if all elements in iterable are truthy
- any(): Returns True if at least one element in iterable is truthy
"""

all_true = all([True, True, True])           # True
any_true = any([False, False, True])         # True
empty_all = all([])                          # True (vacuously true)
empty_any = any([])                          # False (vacuously false)

# Practical examples
valid = all(x > 0 for x in [1, 2, 3, 4])      # True if all positive
exists = any(x == 'target' for x in ['a', 'b', 'target', 'c'])  # True

##############################################################################
# INTERACTIONS AND ADVANCED USAGE
##############################################################################
"""
This section demonstrates how dict, float, and bool interact with each other
and advanced usage patterns combining these types.
"""

# 1. Dictionary with float keys and boolean values
performance_thresholds = {
    0.8: True,    # 80% and above is passing
    0.6: False,   # 60% and above is borderline
    0.0: False    # Below 60% is failing
}

def evaluate_score(score):
    """Evaluate a score using thresholds."""
    for threshold in sorted(performance_thresholds.keys(), reverse=True):
        if score >= threshold:
            return performance_thresholds[threshold]
    return False  # Default if no threshold matches

# 2. Using bool to filter dictionary items
scores = {'Alice': 85.5, 'Bob': 92.1, 'Charlie': 65.3, 'David': 59.8}

# Get passing students (>= 70) using dictionary comprehension and boolean logic
passing = {name: score for name, score in scores.items() if score >= 70.0}

# 3. Converting between types with potential issues
# Float to bool - any non-zero float becomes True
float_to_bool = bool(0.0001)  # True

# Bool to float - True becomes 1.0, False becomes 0.0
bool_to_float = float(True)   # 1.0

# Dict to bool - any non-empty dict becomes True
dict_to_bool = bool({'key': 'value'})  # True

# 4. Handling Floating Point in Dictionary Keys
"""
Due to floating-point precision issues, using floats as dictionary keys
can lead to unexpected behavior.
"""

precision_issue = {0.1 + 0.2: 'value'}  # Key is actually 0.30000000000000004
try:
    # This will fail!
    value = precision_issue[0.3]
except KeyError:
    # Use math.isclose for approximate matching
    key = next((k for k in precision_issue if math.isclose(k, 0.3)), None)
    if key is not None:
        value = precision_issue[key]

# 5. Complex Data Structures
# Dictionary using tuple keys containing floats and bools
point_properties = {
    (1.5, 2.5, True): {"color": "red", "size": 5},
    (1.5, 2.5, False): {"color": "blue", "size": 3}
}

# 6. Functional Programming with these Types
from functools import reduce

# Calculate average score using reduce and dictionary values
average = reduce(lambda acc, score: acc + score, scores.values(), 0.0) / len(scores)

# Filter and map operations with booleans and floats
scores_list = list(scores.values())
passing_scores = filter(lambda x: x >= 70.0, scores_list)  # Filter using boolean condition
curved_scores = map(lambda x: min(100.0, x * 1.1), scores_list)  # Map with float transformation

# 7. Serialization and Deserialization
import json

# JSON handles these types well, with caveats for float precision
data = {
    "config": {
        "threshold": 0.85,
        "active": True
    },
    "results": [1.0, 2.5, 3.7]
}

# Serialize to JSON
json_data = json.dumps(data)

# Deserialize from JSON
parsed_data = json.loads(json_data)


# Note: NaN and Infinity are not supported by standard JSON
try:
    json.dumps({"value": float('inf')})  # Raises ValueError
except ValueError as e:
    # This block will run
    print(f"ValueError would occur: {e}")

# 8. Memory and Performance Considerations
"""
- bool is very memory efficient (essentially an int)
- float takes 8 bytes per number
- dict has overhead but offers O(1) average lookup, insert, delete
"""

# Memory usage demonstration
import sys

bool_size = sys.getsizeof(True)              # Typically 28 bytes (implementation detail)
float_size = sys.getsizeof(1.0)              # Typically 24 bytes (implementation detail)
empty_dict_size = sys.getsizeof({})          # Varies by Python version
small_dict_size = sys.getsizeof({'a': 1})    # More than empty_dict_size

# Example: calculating approximate memory usage for a dataset
def estimate_memory(num_records):
    """Estimate memory for a dataset of floating-point records with boolean flags."""
    per_record = 2 * float_size + bool_size + 100  # 2 floats, 1 bool, plus dict overhead
    return num_records * per_record / (1024 * 1024)  # in MB

estimated_mb = estimate_memory(1000000)  # Memory estimate for 1 million records