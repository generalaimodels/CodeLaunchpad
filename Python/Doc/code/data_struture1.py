# -*- coding: utf-8 -*-
"""
Data Structures - Advanced Exploration of Tuples, Sets, Dictionaries, and Looping

This script provides an expert-level examination of Python's core data structures and looping techniques,
designed for seasoned Python developers. We transcend basic usage to dissect advanced properties,
performance characteristics, and idiomatic patterns for optimal data manipulation and algorithmic efficiency.

This rigorous exploration will cover:

    - Tuples and Sequences: Deconstructing tuples as immutable sequences, emphasizing their role in data integrity and advanced sequence operations.
    - Sets: In-depth analysis of sets for efficient membership testing and set-theoretic operations, including performance nuances and specialized use cases.
    - Dictionaries: Mastering dictionaries as highly optimized mappings, focusing on advanced access patterns, view objects, and efficient data retrieval strategies.
    - Looping Techniques: Exploring advanced looping idioms, including `enumerate`, `zip`, iterator protocols, and performance-optimized iteration patterns.
    - More on Conditions: Deepening understanding of Python's conditional expressions, truthiness rules, and crafting sophisticated conditional logic.
    - Comparing Sequences and Other Types: Examining sequence comparison mechanisms, type comparison nuances, and custom comparison strategies for complex objects.

Expect a focus on:

    - Performance optimization for data structure operations and looping constructs.
    - Pythonic coding idioms for advanced data manipulation.
    - In-depth understanding of data structure internals and implementation details.
    - Advanced use cases and less commonly known features of each data structure and technique.
    - Robust error handling and defensive programming strategies for data-intensive operations.
    - Understanding the interplay between data structures and algorithm design in Python.

Let's embark on this advanced journey to solidify your Python data structure expertise and elevate your coding proficiency.
"""

################################################################################
# 5. Data Structures (Continued)
################################################################################

print("\n--- 5. Data Structures (Continued) ---\n")

################################################################################
# 5.3. Tuples and Sequences
################################################################################

print("\n--- 5.3. Tuples and Sequences ---\n")

# Tuples are immutable ordered sequences, often used to represent fixed collections of items.
# They are defined using parentheses `()` and are a crucial part of Python's sequence type hierarchy, sharing many operations with lists and strings.
# Immutability is the defining characteristic, providing data integrity and enabling usage in contexts where mutability is undesirable (e.g., dictionary keys, set elements).

# --- Tuple Creation and Immutability ---
print("\n--- Tuple Creation and Immutability ---")
example_tuple = (1, 2, 'three', 4.0)
print(f"Example tuple: {example_tuple}, Type: {type(example_tuple)}")

# Immutability in action:
try:
    example_tuple[0] = 10 # Attempting to modify a tuple item - TypeError!
except TypeError as e:
    print(f"Tuple immutability - TypeError: {e}")

# Creating tuples with single elements requires a trailing comma:
single_element_tuple = (5,) # Note the comma
print(f"Single element tuple: {single_element_tuple}, Type: {type(single_element_tuple)}")
not_a_tuple = (5) # Without comma, it's just an integer in parentheses
print(f"Not a tuple: {not_a_tuple}, Type: {type(not_a_tuple)}")

# Tuple packing and unpacking:
packed_tuple = 10, 20, 30 # Tuple packing - parentheses are optional in many contexts
x, y, z = packed_tuple      # Tuple unpacking - assigning tuple elements to variables
print(f"Packed tuple: {packed_tuple}, Unpacked values: x={x}, y={y}, z={z}")

# --- Sequence Operations on Tuples ---
print("\n--- Sequence Operations on Tuples ---")
tuple1 = (1, 2, 3)
tuple2 = (4, 5)

# Concatenation (+)
concatenated_tuple = tuple1 + tuple2
print(f"Concatenation: {tuple1} + {tuple2} = {concatenated_tuple}")

# Repetition (*)
repeated_tuple = tuple1 * 3
print(f"Repetition: {tuple1} * 3 = {repeated_tuple}")

# Indexing and Slicing - Same as lists and strings (zero-based indexing)
print("\n--- Tuple Indexing and Slicing ---")
indexing_tuple = ('a', 'b', 'c', 'd', 'e')
print(f"Tuple: {indexing_tuple}")
print(f"Index 0: {indexing_tuple[0]}, Index -1: {indexing_tuple[-1]}")
print(f"Slice [1:4]: {indexing_tuple[1:4]}, Slice [:3]: {indexing_tuple[:3]}")

# Membership testing (in, not in)
print("\n--- Tuple Membership Testing ---")
print(f"'c' in {indexing_tuple}: {'c' in indexing_tuple}")
print(f"'f' not in {indexing_tuple}: {'f' not in indexing_tuple}")

# Length (len())
print("\n--- Tuple Length ---")
print(f"Length of {tuple1}: {len(tuple1)}")

# Iteration
print("\n--- Tuple Iteration ---")
for item in example_tuple:
    print(item)

# --- Use Cases for Tuples ---
print("\n--- Use Cases for Tuples ---")
# - Data integrity: Immutability ensures data remains unchanged after creation.
# - Dictionary keys: Tuples (if they contain only immutable elements) can be used as dictionary keys.
# - Set elements: Similarly, tuples of immutable elements can be members of sets.
# - Function return values: Returning multiple values efficiently as a tuple.
# - Representing records: Grouping related data points together (e.g., coordinates, RGB colors).

# --- Tuple Performance Considerations ---
print("\n--- Tuple Performance Considerations ---")
# - Tuple creation is generally slightly faster than list creation.
# - Tuple indexing is as fast as list indexing (O(1)).
# - Tuples are more memory-efficient than lists due to their immutability (no need to allocate extra space for potential resizing).
# - Hashable if all elements are hashable, making them usable as dictionary keys and set elements.

# --- Exception Handling with Tuples ---
print("\n--- Tuple Exception Handling ---")
try:
    out_of_range_index_tuple = indexing_tuple[10] # IndexError: tuple index out of range
    print(f"Out-of-range index access (should raise error): {out_of_range_index_tuple}") # Not reached
except IndexError as e:
    print(f"IndexError encountered: {e}")
except TypeError as e: # TypeError can occur with incompatible operations
    print(f"TypeError encountered with tuples: {e}")

################################################################################
# 5.4. Sets
################################################################################

print("\n--- 5.4. Sets ---\n")

# Sets are unordered collections of *unique* and *hashable* elements.
# They are highly optimized for membership testing and removing duplicate entries.
# Sets come in two flavors: mutable (`set`) and immutable (`frozenset`). Mutable sets can be modified after creation, while frozensets cannot.

# --- Set Creation and Properties ---
print("\n--- Set Creation and Properties ---")
example_set = {1, 2, 3, 2, 1} # Duplicate elements are automatically removed
print(f"Example set: {example_set}, Type: {type(example_set)}") # Output: {1, 2, 3} (order may vary)

# Sets are unordered: element order is not guaranteed and may change.
# Sets only contain unique elements: duplicates are automatically eliminated.
# Set elements must be hashable: immutable objects (e.g., integers, floats, strings, tuples of immutable objects). Lists and dictionaries are not hashable and cannot be set elements.

# Creating sets from iterables:
set_from_list = set([4, 5, 6, 6])
set_from_string = set("hello")
print(f"Set from list: {set_from_list}")
print(f"Set from string: {set_from_string}")

# Frozen sets (immutable sets) - created using frozenset() constructor
frozen_set = frozenset([7, 8, 9])
print(f"Frozen set: {frozen_set}, Type: {type(frozen_set)}")
try:
    frozen_set.add(10) # Attempting to modify a frozen set - AttributeError!
except AttributeError as e:
    print(f"Frozen set immutability - AttributeError: {e}")

# --- Set Operations ---
print("\n--- Set Operations ---")
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}

# Union (| or .union()) - All elements in either set
union_set = set_a | set_b # or set_a.union(set_b)
print(f"Union: {set_a} | {set_b} = {union_set}")

# Intersection (& or .intersection()) - Elements common to both sets
intersection_set = set_a & set_b # or set_a.intersection(set_b)
print(f"Intersection: {set_a} & {set_b} = {intersection_set}")

# Difference (- or .difference()) - Elements in set_a but not in set_b
difference_set_a_b = set_a - set_b # or set_a.difference(set_b)
print(f"Difference (a - b): {set_a} - {set_b} = {difference_set_a_b}")
difference_set_b_a = set_b - set_a # or set_b.difference(set_a)
print(f"Difference (b - a): {set_b} - {set_a} = {difference_set_b_a}")

# Symmetric Difference (^ or .symmetric_difference()) - Elements in either set but not in both
symmetric_difference_set = set_a ^ set_b # or set_a.symmetric_difference(set_b)
print(f"Symmetric Difference: {set_a} ^ {set_b} = {symmetric_difference_set}")

# Subset and Superset (<=, <, >=, >, .issubset(), .issuperset())
print("\n--- Set Subset and Superset ---")
set_c = {1, 2, 3}
print(f"Is {set_c} subset of {set_a}?: {set_c <= set_a} (or {set_c.issubset(set_a)})")
print(f"Is {set_a} superset of {set_c}?: {set_a >= set_c} (or {set_a.issuperset(set_c)})")
print(f"Is {set_c} proper subset of {set_a}?: {set_c < set_a}") # Proper subset - subset and not equal
print(f"Is {set_a} proper superset of {set_c}?: {set_a > set_c}") # Proper superset - superset and not equal

# Disjoint (.isdisjoint()) - Sets have no common elements
print("\n--- Set Disjoint ---")
set_d = {9, 10}
print(f"Are {set_a} and {set_d} disjoint?: {set_a.isdisjoint(set_d)}")
print(f"Are {set_a} and {set_b} disjoint?: {set_a.isdisjoint(set_b)}")

# --- Set Comprehensions ---
print("\n--- Set Comprehensions ---")
numbers = [1, 2, 3, 4, 5, 1, 2, 3]
squared_set_lc = {number**2 for number in numbers} # Set comprehension - creates a set of squared numbers (duplicates removed)
print(f"Squared set (set comprehension): {squared_set_lc}")

# --- Use Cases for Sets ---
print("\n--- Use Cases for Sets ---")
# - Membership testing: Highly efficient (average O(1) time complexity).
# - Removing duplicates: Automatically eliminates duplicate elements.
# - Mathematical set operations: Union, intersection, difference, etc.
# - Finding unique elements in a collection.

# --- Set Performance Considerations ---
print("\n--- Set Performance Considerations ---")
# - Membership testing (in, not in) is extremely fast (average O(1) time complexity).
# - Adding and removing elements is also typically fast (average O(1)).
# - Set operations (union, intersection, etc.) are generally efficient.
# - Sets are implemented using hash tables, which contribute to their fast performance for these operations.

# --- Set Exception Handling ---
print("\n--- Set Exception Handling ---")
try:
    unhashable_set = {[1, 2], 3} # TypeError: unhashable type: 'list' - Lists are not hashable
    print(f"Unhashable set (should raise error): {unhashable_set}") # Not reached
except TypeError as e:
    print(f"TypeError encountered: {e}")

################################################################################
# 5.5. Dictionaries
################################################################################

print("\n--- 5.5. Dictionaries ---\n")

# Dictionaries are highly optimized mappings, storing key-value pairs. Keys must be unique and hashable, while values can be of any type.
# Dictionaries are mutable and provide efficient key-based lookup, insertion, and deletion operations.

# --- Dictionary Creation and Properties ---
print("\n--- Dictionary Creation and Properties ---")
example_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}
print(f"Example dictionary: {example_dict}, Type: {type(example_dict)}")

# Keys must be hashable and unique:
# - Immutable types (integers, floats, strings, tuples) are typically used as keys.
# - If duplicate keys are provided during creation, the last key-value pair will overwrite previous ones.
# Values can be of any type (mutable or immutable).

# Creating dictionaries using dict() constructor:
dict_from_pairs = dict([('a', 1), ('b', 2), ('c', 3)]) # From list of key-value pairs
dict_from_kwargs = dict(d=4, e=5, f=6) # From keyword arguments
print(f"Dictionary from pairs: {dict_from_pairs}")
print(f"Dictionary from kwargs: {dict_from_kwargs}")

# --- Dictionary Operations ---
print("\n--- Dictionary Operations ---")
person_dict = {'name': 'Bob', 'age': 25, 'city': 'London'}

# Accessing values by key (indexing with [] or .get())
print("\n--- Dictionary Access ---")
name_value = person_dict['name'] # Using indexing - KeyError if key not found
print(f"Name: {name_value}")
age_value_get = person_dict.get('age') # Using .get() - Returns None if key not found, or a default value if provided
print(f"Age (using get): {age_value_get}")
city_value_get_default = person_dict.get('country', 'Unknown') # Default value if key not found
print(f"Country (using get with default): {city_value_get_default}")
try:
    non_existent_key = person_dict['country'] # KeyError if key not in dict
    print(non_existent_key) # Not reached
except KeyError as e:
    print(f"KeyError: {e}")

# Modifying and Adding key-value pairs
print("\n--- Dictionary Modification and Addition ---")
person_dict['age'] = 26 # Modify existing value
print(f"Dictionary after age modification: {person_dict}")
person_dict['occupation'] = 'Engineer' # Add new key-value pair
print(f"Dictionary after occupation addition: {person_dict}")

# Deleting key-value pairs (del or .popitem(), .pop())
print("\n--- Dictionary Deletion ---")
del person_dict['city'] # Delete by key - KeyError if key not found
print(f"Dictionary after city deletion: {person_dict}")
removed_item = person_dict.pop('age') # Remove and return value by key - KeyError if key not found
print(f"Removed item (pop 'age'): {removed_item}, Dictionary: {person_dict}")
last_item = person_dict.popitem() # Remove and return last inserted key-value pair (LIFO in Python 3.7+), raises KeyError if dict is empty
print(f"Removed last item (popitem): {last_item}, Dictionary: {person_dict}")
try:
    empty_dict = {}
    empty_dict.popitem() # popitem from empty dict
except KeyError as e:
    print(f"popitem from empty dict - KeyError: {e}")

# --- Dictionary Views (keys(), values(), items()) ---
print("\n--- Dictionary Views ---")
view_dict = {'a': 1, 'b': 2, 'c': 3}
keys_view = view_dict.keys() # View object of keys
values_view = view_dict.values() # View object of values
items_view = view_dict.items() # View object of key-value pairs (tuples)

print(f"Keys view: {keys_view}, Values view: {values_view}, Items view: {items_view}")

# Views are dynamic: they reflect changes in the dictionary
view_dict['d'] = 4
print(f"After adding 'd': Keys view: {keys_view}, Values view: {values_view}, Items view: {items_view}")

# Iterating through dictionaries (keys, values, items)
print("\n--- Dictionary Iteration ---")
print("Iterating keys:")
for key in view_dict: # Iterates over keys by default
    print(key)
print("\nIterating values:")
for value in view_dict.values():
    print(value)
print("\nIterating items (key-value pairs):")
for key, value in view_dict.items():
    print(f"Key: {key}, Value: {value}")

# --- Dictionary Comprehensions ---
print("\n--- Dictionary Comprehensions ---")
numbers_list = [1, 2, 3, 4, 5]
squared_dict_lc = {number: number**2 for number in numbers_list} # Dictionary comprehension - creates a dictionary mapping numbers to their squares
print(f"Squared dictionary (dictionary comprehension): {squared_dict_lc}")

# --- Use Cases for Dictionaries ---
print("\n--- Use Cases for Dictionaries ---")
# - Mapping keys to values: Configuration settings, symbol tables, caches, etc.
# - Data lookup: Efficient retrieval of values based on keys.
# - Representing structured data: JSON-like structures, records.
# - Counting occurrences of items (using keys as items and values as counts).

# --- Dictionary Performance Considerations ---
print("\n--- Dictionary Performance Considerations ---")
# - Key lookup, insertion, deletion are extremely fast (average O(1) time complexity) due to hash table implementation.
# - Iteration order is insertion order in Python 3.7+ (CPython 3.6+).
# - Dictionaries are highly optimized for key-based operations.

# --- Dictionary Exception Handling ---
print("\n--- Dictionary Exception Handling ---")
try:
    unhashable_key_dict = {[1, 2]: 'value'} # TypeError: unhashable type: 'list' - Lists are not hashable keys
    print(f"Unhashable key dictionary (should raise error): {unhashable_key_dict}") # Not reached
except TypeError as e:
    print(f"TypeError encountered: {e}")

################################################################################
# 5.6. Looping Techniques
################################################################################

print("\n--- 5.6. Looping Techniques ---\n")

# Python offers several advanced looping techniques that enhance code readability, efficiency, and expressiveness when working with iterables.

# --- enumerate() - Index and Value Iteration ---
print("\n--- enumerate() - Index and Value Iteration ---")
items_list = ['a', 'b', 'c', 'd']
print("enumerate(items_list):")
for index, item in enumerate(items_list): # enumerate() yields (index, value) pairs
    print(f"Index: {index}, Item: {item}")

print("\nenumerate(items_list, start=1):")
for index, item in enumerate(items_list, start=1): # Start index from 1
    print(f"Index (starting from 1): {index}, Item: {item}")

# --- zip() - Parallel Iteration over Multiple Iterables ---
print("\n--- zip() - Parallel Iteration over Multiple Iterables ---")
names = ['Alice', 'Bob', 'Charlie']
ages = [30, 25, 40]
cities = ['New York', 'London', 'Paris']

print("\nzip(names, ages, cities):")
for name, age, city in zip(names, ages, cities): # zip() iterates in parallel until the shortest iterable is exhausted
    print(f"Name: {name}, Age: {age}, City: {city}")

# Handling unequal length iterables with itertools.zip_longest (more advanced)
import itertools
print("\nitertools.zip_longest(names, ages, cities, fillvalue='N/A'):")
for name, age, city in itertools.zip_longest(names, ages, cities, fillvalue='N/A'): # Fills missing values with 'N/A'
    print(f"Name: {name}, Age: {age}, City: {city}")

# --- Looping through Dictionaries ---
print("\n--- Looping through Dictionaries ---")
dictionary_loop = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

print("\nLooping through keys (default):")
for key in dictionary_loop: # Iterates over keys by default
    print(f"Key: {key}")

print("\nLooping through values:")
for value in dictionary_loop.values():
    print(f"Value: {value}")

print("\nLooping through items (key-value pairs):")
for key, value in dictionary_loop.items(): # Iterates over key-value pairs
    print(f"Key: {key}, Value: {value}")

# --- Iterator Protocol (Advanced - Briefly Mention) ---
print("\n--- Iterator Protocol (Briefly Mention) ---")
# For loops in Python work with iterators. An iterator is an object that implements the iterator protocol, which consists of two methods:
# - __iter__(): Returns the iterator object itself.
# - __next__(): Returns the next item in the sequence. When there are no more items, it raises StopIteration exception.

# Example (manual iteration - typically handled implicitly by for loops):
iterator = iter(items_list) # Get iterator for the list
try:
    while True:
        item = next(iterator) # Get next item
        print(f"Manually iterated item: {item}")
except StopIteration: # StopIteration signals end of iteration
    print("End of iteration reached.")

# --- Performance Considerations for Looping ---
print("\n--- Performance Considerations for Looping ---")
# - Python's for loops are generally efficient for iterating over sequences and iterables.
# - List comprehensions and generator expressions can often be more performant than explicit for loops for simple transformations.
# - For very performance-critical loops, especially in numerical computations, consider using NumPy or Cython for optimized iteration.

# --- Exception Handling in Loops ---
print("\n--- Exception Handling in Loops ---")
# Exceptions within the loop body are handled in the standard way using try-except blocks.
# StopIteration is used internally to signal the end of iteration and is typically handled implicitly by for loops; you don't usually catch it explicitly in normal for loop usage.

################################################################################
# 5.7. More on Conditions
################################################################################

print("\n--- 5.7. More on Conditions ---\n")

# Python's conditional expressions and truthiness rules offer powerful ways to express complex conditions concisely and effectively.

# --- Truthiness and Falsiness - Deep Dive ---
print("\n--- Truthiness and Falsiness - Deep Dive ---")
# Recap: Falsy values in Python: False, None, numeric zero (0, 0.0, 0j), empty sequences ('', [], (), {}), empty sets, and objects of classes that define __bool__() or __len__() returning False or 0.
# All other values are truthy.

print(f"bool(False): {bool(False)}")
print(f"bool(None): {bool(None)}")
print(f"bool(0): {bool(0)}")
print(f"bool(0.0): {bool(0.0)}")
print(f"bool(''): {bool('')}")
print(f"bool([]): {bool([])}")
print(f"bool(()): {bool(())}")
# print(f"bool({}): {bool({})}")
print(f"bool(set()): {bool(set())}")

print(f"bool(True): {bool(True)}")
print(f"bool(1): {bool(1)}")
print(f"bool(10.5): {bool(10.5)}")
print(f"bool('hello'): {bool('hello')}")
print(f"bool([1, 2]): {bool([1, 2])}")
print(f"bool((1, 2)): {bool((1, 2))}")
print(f"bool({{1: 2}}): {bool({{1: 2}})}")
print(f"bool({{1}}): {bool({{1}})}")

# Custom truthiness using __bool__() or __len__() methods in classes (more advanced OOP)

# --- Advanced Conditional Logic ---
print("\n--- Advanced Conditional Logic ---")
x = 10
y = 5
z = 20

# Chained comparisons (highly Pythonic and readable)
print("\n--- Chained Comparisons ---")
print(f"0 < x < 15: {0 < x < 15}") # Equivalent to (0 < x) and (x < 15)
print(f"x > y >= 0: {x > y >= 0}") # Equivalent to (x > y) and (y >= 0)

# Boolean operators (and, or, not) - Short-circuit evaluation
print("\n--- Boolean Operators and Short-Circuit Evaluation ---")
# 'and': Returns first falsy operand, or the last operand if all are truthy.
# 'or': Returns first truthy operand, or the last operand if all are falsy.
# 'not': Negates the truthiness of the operand.

def is_truthy(value):
    print(f"Evaluating truthiness of: {value}")
    return bool(value)

print("\nShort-circuit 'and':")
result_and = is_truthy(True) and is_truthy(False) and is_truthy(True) # 'False' is encountered first, no further evaluation after that.
print(f"Result of 'True and False and True': {result_and}")

print("\nShort-circuit 'or':")
result_or = is_truthy(False) or is_truthy(True) or is_truthy(False) # 'True' is encountered first, no further evaluation after that.
print(f"Result of 'False or True or False': {result_or}")

# Conditional expressions (ternary operator) - Concise if-else in a single line
print("\n--- Conditional Expressions (Ternary Operator) ---")
status = "adult" if x >= 18 else "minor"
print(f"Status for age {x}: {status}")

# --- Use Cases for Advanced Conditions ---
print("\n--- Use Cases for Advanced Conditions ---")
# - Complex validation logic.
# - Concise expression of conditional assignments.
# - Efficient short-circuiting in boolean expressions for performance optimization.

# --- Exception Handling in Conditions ---
# Exceptions within conditional expressions or conditions are handled as they normally would be.

################################################################################
# 5.8. Comparing Sequences and Other Types
################################################################################

print("\n--- 5.8. Comparing Sequences and Other Types ---\n")

# Python supports comparison operations between sequences (lists, tuples, strings) and between different types, with specific rules and behaviors.

# --- Sequence Comparison (Lexicographical) ---
print("\n--- Sequence Comparison (Lexicographical) ---")
# Sequences are compared lexicographically: element by element, from left to right.
# Comparison stops as soon as a difference is found. If one sequence is a prefix of the other, the shorter sequence is considered smaller.

list1 = [1, 2, 3]
list2 = [1, 2, 4]
list3 = [1, 2]
list4 = [1, 2, 3]

print(f"{list1} == {list4}: {list1 == list4}") # Element-wise equality
print(f"{list1} < {list2}: {list1 < list2}") # Comparison at index 2 (3 < 4)
print(f"{list1} > {list2}: {list1 > list2}")
print(f"{list3} < {list1}: {list3 < list1}") # list3 is a prefix of list1

tuple1 = (1, 2, 3)
tuple2 = (1, 2, 4)
print(f"{tuple1} < {tuple2}: {tuple1 < tuple2}") # Lexicographical comparison works for tuples too

string1 = "apple"
string2 = "banana"
string3 = "apple"
print(f"'{string1}' < '{string2}': {string1 < string2}") # Lexicographical comparison for strings (alphabetical order)
print(f"'{string1}' == '{string3}': {string1 == string3}")

# --- Type Comparison and Identity vs. Equality ---
print("\n--- Type Comparison and Identity vs. Equality ---")
# Equality (==): Compares the *values* of objects.
# Identity (is): Compares the *memory addresses* (object identity).

list_a = [1, 2, 3]
list_b = [1, 2, 3]
list_c = list_a # Reference to the same object

print(f"{list_a} == {list_b}: {list_a == list_b}") # True - values are the same
print(f"{list_a} is {list_b}: {list_a is list_b}") # False - different objects in memory
print(f"{list_a} is {list_c}: {list_a is list_c}") # True - same object in memory

# Comparing objects of different types - Python 3 behavior
print("\n--- Comparing Different Types (Python 3+) ---")
print(f"10 < '2': {10 < '2'}") # TypeError: '<' not supported between instances of 'int' and 'str' - In Python 3, most comparisons between unrelated types raise TypeError.

# --- Custom Comparison Methods (Advanced - Briefly Mention) ---
# For custom classes, you can define rich comparison methods (__lt__, __le__, __eq__, __ne__, __gt__, __ge__) to customize how instances of your classes are compared.
# functools.total_ordering decorator can help implement all rich comparison methods if you define just a few (e.g., __eq__ and __lt__).

# --- Exception Handling in Comparisons ---
print("\n--- Exception Handling in Comparisons ---")
try:
    comparison_error = 10 < "string" # TypeError when types are not comparable
    print(comparison_error) # Not reached
except TypeError as e:
    print(f"TypeError during comparison: {e}")

print("\n--- End of Data Structures - Advanced Concepts ---")