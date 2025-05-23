#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Built-in Functions: zip, map, filter
====================================================

This tutorial covers three powerful built-in functions in Python that are essential
for writing efficient, idiomatic, and Pythonic code. These functions implement
functional programming concepts and enable operations on iterables with clean syntax.
"""


#####################################
# 1. THE ZIP FUNCTION
#####################################
"""
The zip() function takes multiple iterables and returns an iterator of tuples,
where the i-th tuple contains the i-th element from each of the input iterables.

Syntax: zip(*iterables)

Key characteristics:
- Returns a zip object (iterator of tuples)
- Stops when the shortest input iterable is exhausted
- Commonly used to pair related values from different sequences
"""

# Basic usage: combining related elements from multiple lists
def zip_basic_example():
    names = ["Alice", "Bob", "Charlie"]
    ages = [25, 30, 35]
    
    # Zip the two lists together
    paired_data = zip(names, ages)
    
    # Convert to list to see the result (not needed in actual use)
    print(f"Zipped result: {list(paired_data)}")
    
    # More common usage: iterate directly through the zip object
    for name, age in zip(names, ages):
        print(f"{name} is {age} years old")


# Handling iterables of different lengths (stops at shortest)
def zip_length_example():
    letters = ["a", "b", "c"]
    numbers = [1, 2, 3, 4, 5]  # Longer than letters
    
    # Zip will only pair the first 3 elements from numbers
    print(f"Zipped with different lengths: {list(zip(letters, numbers))}")
    
    # EXCEPTION CASE: If you need all elements, use itertools.zip_longest
    import itertools
    print("Using zip_longest:")
    print(list(itertools.zip_longest(letters, numbers, fillvalue="missing")))


# Unzipping: using zip to "transpose" data
def zip_unpack_example():
    # Data in paired format
    pairs = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
    
    # Unpack using zip(*iterable) with the * operator
    names, ages = zip(*pairs)
    print(f"Unpacked names: {names}")
    print(f"Unpacked ages: {ages}")


# Dictionary operations with zip
def zip_dict_operations():
    keys = ["name", "age", "job"]
    values = ["Alice", 25, "Engineer"]
    
    # Create a dictionary from keys and values
    person = dict(zip(keys, values))
    print(f"Created dictionary: {person}")
    
    # Swap keys and values in a dictionary
    inverted_dict = dict(zip(person.values(), person.keys()))
    print(f"Inverted dictionary: {inverted_dict}")
    
    # EXCEPTION CASE: Warning about non-hashable values
    # If values contain lists or other unhashable types, this will fail
    try:
        bad_keys = ["a", "b", "c"]
        bad_values = [[1, 2], [3, 4], [5, 6]]  # Lists are unhashable
        bad_dict = dict(zip(bad_values, bad_keys))
    except TypeError as e:
        print(f"Exception when using unhashable types: {e}")


# Advanced: Zipping multiple iterables
def zip_multiple_iterables():
    names = ["Alice", "Bob", "Charlie"]
    ages = [25, 30, 35]
    jobs = ["Engineer", "Designer", "Manager"]
    cities = ["New York", "San Francisco", "Seattle"]
    
    # Zip can handle any number of iterables
    for name, age, job, city in zip(names, ages, jobs, cities):
        print(f"{name} is a {age}-year-old {job} in {city}")


# Performance consideration: zip is lazy (returns an iterator)
def zip_performance():
    # This creates a large range but doesn't consume memory
    # zip() won't process these until elements are requested
    large_range1 = range(10**6)
    large_range2 = range(10**6)
    
    # This creates an iterator, not a full list of tuples
    zipped = zip(large_range1, large_range2)
    
    # Only when we iterate does it compute values
    print("First 5 elements:")
    for i, (a, b) in enumerate(zipped):
        if i >= 5:
            break
        print(f"({a}, {b})")


#####################################
# 2. THE MAP FUNCTION
#####################################
"""
The map() function applies a specified function to each item of an iterable
and returns an iterator of the results.

Syntax: map(function, iterable, [iterable2, iterable3, ...])

Key characteristics:
- Returns a map object (iterator)
- Can apply a function to multiple iterables (like zip)
- Processes elements one at a time (lazy evaluation)
- Useful for transforming data without explicit loops
"""

# Basic usage: applying a function to each element
def map_basic_example():
    numbers = [1, 2, 3, 4, 5]
    
    # Apply the built-in square function to each number
    squares = map(lambda x: x**2, numbers)
    print(f"Squared numbers: {list(squares)}")
    
    # Using a named function
    def celsius_to_fahrenheit(c):
        return c * 9/5 + 32
    
    temperatures_c = [0, 10, 20, 30, 40]
    temperatures_f = map(celsius_to_fahrenheit, temperatures_c)
    print(f"Temperatures in Fahrenheit: {list(temperatures_f)}")


# Map with multiple iterables
def map_multiple_iterables():
    base_prices = [10, 20, 30, 40]
    tax_rates = [0.07, 0.05, 0.08, 0.09]
    
    # Calculate final prices using both lists
    def calculate_final_price(price, tax):
        return price * (1 + tax)
    
    final_prices = map(calculate_final_price, base_prices, tax_rates)
    print(f"Final prices: {list(final_prices)}")
    
    # EXCEPTION CASE: Like zip, map with multiple iterables stops at shortest
    short_list = [1, 2]
    long_list = [1, 2, 3, 4, 5]
    result = map(lambda x, y: x + y, short_list, long_list)
    print(f"Map with different lengths: {list(result)}")


# Using map with non-numeric data
def map_string_example():
    names = ["alice", "bob", "charlie", "david"]
    
    # Capitalize first letter of each name
    capitalized = map(str.capitalize, names)
    print(f"Capitalized names: {list(capitalized)}")
    
    # Extract lengths
    name_lengths = map(len, names)
    print(f"Name lengths: {list(name_lengths)}")


# Map with None as function (identity function)
def map_with_none():
    # Using None as function just returns the elements unchanged
    # Useful for converting an iterable to a list/tuple efficiently
    data = range(5)  # Range object
    # data_list = list(map(None, data))  # Convert to list
    # print(f"Map with None: {data_list}") -> TypeError: 'NoneType' object is not callable
    
    # EXCEPTION CASE: This only works in Python 2, in Python 3 it raises a TypeError
    try:
        result = list(map(None, [1, 2, 3]))
    except TypeError as e:
        print(f"Exception in Python 3 with map(None, ...): {e}")
        print("In Python 3, use list(iterable) instead")


# Map with class methods
def map_with_classes():
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
            
        def description(self):
            return f"{self.name} ({self.age})"
        
        def birthday(self):
            self.age += 1
            return self
    
    people = [
        Person("Alice", 25),
        Person("Bob", 30),
        Person("Charlie", 35)
    ]
    
    # Get descriptions for all people
    descriptions = map(Person.description, people)
    print(f"Descriptions: {list(descriptions)}")
    
    # Increment everyone's age (using an instance method)
    updated_people = map(Person.birthday, people)
    descriptions_after = map(Person.description, updated_people)
    print(f"After birthdays: {list(descriptions_after)}")


# Performance considerations
def map_vs_list_comprehension():
    numbers = list(range(1000))
    
    # Using map (returns iterator, not list)
    import time
    
    start = time.time()
    squares_map = map(lambda x: x**2, numbers)
    # Processing is deferred until iteration or conversion
    squared_list = list(squares_map)
    map_time = time.time() - start
    
    # Using list comprehension (returns list immediately)
    start = time.time()
    squares_comp = [x**2 for x in numbers]
    comp_time = time.time() - start
    
    print(f"Map time: {map_time:.6f} seconds")
    print(f"List comprehension time: {comp_time:.6f} seconds")
    print("Note: For simple operations, list comprehensions can be faster "
          "and more readable, but map shines with reusable functions and "
          "memory efficiency for large datasets")


#####################################
# 3. THE FILTER FUNCTION
#####################################
"""
The filter() function constructs an iterator from elements of an iterable for
which a function returns True.

Syntax: filter(function, iterable)

Key characteristics:
- Returns a filter object (iterator)
- Only keeps elements for which the function returns True
- If function is None, keeps only elements that are truthy
- Useful for selecting a subset of data without explicit loops
"""

# Basic usage: filtering elements by a condition
def filter_basic_example():
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Filter even numbers
    def is_even(n):
        return n % 2 == 0
    
    even_numbers = filter(is_even, numbers)
    print(f"Even numbers: {list(even_numbers)}")
    
    # Using lambda for compact syntax
    odd_numbers = filter(lambda n: n % 2 == 1, numbers)
    print(f"Odd numbers: {list(odd_numbers)}")


# Filter with None (removes falsy values)
def filter_none_example():
    mixed_data = [0, 1, False, True, "", "hello", None, [], [1, 2]]
    
    # None as function argument keeps only truthy values
    truthy_values = filter(None, mixed_data)
    print(f"Truthy values: {list(truthy_values)}")
    
    # Equivalent with explicit lambda
    is_truthy = lambda x: bool(x)
    explicit_filter = filter(is_truthy, mixed_data)
    print(f"Explicitly filtered: {list(explicit_filter)}")


# Filtering dictionaries and complex objects
def filter_complex_example():
    people = [
        {"name": "Alice", "age": 25, "active": True},
        {"name": "Bob", "age": 17, "active": False},
        {"name": "Charlie", "age": 30, "active": True},
        {"name": "Dave", "age": 16, "active": True}
    ]
    
    # Filter for active people over 18
    def is_active_adult(person):
        return person["active"] and person["age"] >= 18
    
    active_adults = filter(is_active_adult, people)
    print(f"Active adults: {list(active_adults)}")
    
    # EXCEPTION CASE: Note that filter doesn't transform the data
    # If you need to both filter and transform, you'd use a combination
    # of filter and map or a comprehension
    names_of_active_adults = map(
        lambda person: person["name"],
        filter(is_active_adult, people)
    )
    print(f"Names of active adults: {list(names_of_active_adults)}")


# Filtering strings and text processing
def filter_string_example():
    text = "Hello, this is an example with special ch@racters! 123"
    
    # Filter for alphabetic characters
    alpha_only = filter(str.isalpha, text)
    print(f"Alphabetic only: {''.join(alpha_only)}")
    
    # Filter for digits
    digits_only = filter(str.isdigit, text)
    print(f"Digits only: {''.join(digits_only)}")
    
    # Filter out whitespace
    no_whitespace = filter(lambda c: not c.isspace(), text)
    print(f"No whitespace: {''.join(no_whitespace)}")


# Performance considerations
def filter_vs_comprehension():
    numbers = list(range(1000))
    
    import time
    
    # Using filter
    start = time.time()
    even_filter = filter(lambda x: x % 2 == 0, numbers)
    even_list = list(even_filter)
    filter_time = time.time() - start
    
    # Using list comprehension
    start = time.time()
    even_comp = [x for x in numbers if x % 2 == 0]
    comp_time = time.time() - start
    
    print(f"Filter time: {filter_time:.6f} seconds")
    print(f"List comprehension time: {comp_time:.6f} seconds")
    
    # Memory efficiency demonstration:
    large_range = range(10**7)  # 10 million elements
    
    # This is memory efficient (returns an iterator)
    filtered = filter(lambda x: x % 1000 == 0, large_range)
    
    print("First 5 elements from filtering 10 million numbers:")
    for i, num in enumerate(filtered):
        if i >= 5:
            break
        print(num)


#####################################
# 4. COMBINING ZIP, MAP AND FILTER
#####################################
"""
These functions can be combined to create powerful data processing pipelines.
"""

def combined_examples():
    names = ["Alice", "Bob", "Charlie", "David", "Eve"]
    ages = [25, 17, 30, 16, 22]
    jobs = ["Engineer", "Student", "Doctor", "Student", "Designer"]
    
    # Combine the data
    people = zip(names, ages, jobs)
    
    # Filter for adults
    adults = filter(lambda person: person[1] >= 18, people)
    
    # Transform to formatted strings
    formatted = map(
        lambda person: f"{person[0]} ({person[1]}) - {person[2]}",
        adults
    )
    
    print("Adults with their jobs:")
    for person_info in formatted:
        print(person_info)
    
    # IMPORTANT: Note that since these functions return iterators,
    # once consumed they're exhausted. If you need to reuse the data,
    # convert to a list or other container first.
    
    # Combining in a single expression (read from right to left)
    people_data = list(zip(names, ages, jobs))
    result = list(map(
        lambda person: f"{person[0]} ({person[1]}) - {person[2]}",
        filter(lambda person: person[1] >= 18, people_data)
    ))
    
    print("\nSame result with combined functions:")
    for person_info in result:
        print(person_info)
    
    # Alternative using list comprehension (often more readable)
    comprehension_result = [
        f"{name} ({age}) - {job}"
        for name, age, job in zip(names, ages, jobs)
        if age >= 18
    ]
    
    print("\nUsing list comprehension:")
    for person_info in comprehension_result:
        print(person_info)


# Run all examples
if __name__ == "__main__":
    print("\n=== ZIP FUNCTION EXAMPLES ===\n")
    zip_basic_example()
    print("\n")
    zip_length_example()
    print("\n")
    zip_unpack_example()
    print("\n")
    zip_dict_operations()
    print("\n")
    zip_multiple_iterables()
    print("\n")
    zip_performance()
    
    print("\n=== MAP FUNCTION EXAMPLES ===\n")
    map_basic_example()
    print("\n")
    map_multiple_iterables()
    print("\n")
    map_string_example()
    print("\n")
    map_with_none()
    print("\n")
    map_with_classes()
    print("\n")
    map_vs_list_comprehension()
    
    print("\n=== FILTER FUNCTION EXAMPLES ===\n")
    filter_basic_example()
    print("\n")
    filter_none_example()
    print("\n")
    filter_complex_example()
    print("\n")
    filter_string_example()
    print("\n")
    filter_vs_comprehension()
    
    print("\n=== COMBINED EXAMPLES ===\n")
    combined_examples()