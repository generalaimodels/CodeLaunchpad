#!/usr/bin/env python3
"""
Advanced Python Built-in Functions Guide: sorted(), reversed(), iter()
-----------------------------------------------------------------------
This guide provides in-depth explanations and examples of three powerful
built-in functions in Python that every advanced coder should master.
"""

##############################################################################
# SORTED() FUNCTION
##############################################################################
"""
The sorted() function returns a new sorted list from the elements of any iterable.

Syntax: sorted(iterable, *, key=None, reverse=False)

Parameters:
- iterable: Required. A sequence (list, tuple, string) or collection (set, dictionary, 
  frozen set) or any other iterator
- key: Optional. A function that serves as a key for the sort comparison
- reverse: Optional. If True, the sorted list is reversed (descending order)

Return Value:
- A new sorted list containing all items from the iterable

Time Complexity: O(n log n) where n is the length of the iterable
Space Complexity: O(n) as it creates a new list
"""

# Basic Usage with Different Iterables
print("# Basic Usage of sorted()")
# Sorting a list
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(numbers)
print(f"Original list: {numbers}")
print(f"Sorted list: {sorted_numbers}")
# Note: original list remains unchanged

# Sorting a tuple (returns a list)
tuple_values = (3, 1, 4, 1, 5, 9)
sorted_tuple = sorted(tuple_values)
print(f"Original tuple: {tuple_values}")
print(f"Sorted tuple (as list): {sorted_tuple}")

# Sorting a string (returns a list of characters)
text = "python"
sorted_text = sorted(text)
print(f"Original string: {text}")
print(f"Sorted string (as list): {sorted_text}")
print(f"Sorted string (joined): {''.join(sorted_text)}")

# Sorting a dictionary (sorts by keys)
my_dict = {'c': 3, 'a': 1, 'b': 2}
sorted_dict_keys = sorted(my_dict)
print(f"Original dict: {my_dict}")
print(f"Sorted dict keys: {sorted_dict_keys}")

# Sorting a set
my_set = {3, 1, 4, 1, 5, 9}  # Note: duplicates are removed in sets
sorted_set = sorted(my_set)
print(f"Original set: {my_set}")
print(f"Sorted set (as list): {sorted_set}")

print("\n# Advanced Usage of sorted()")
# Sorting in reverse order
reverse_sorted = sorted(numbers, reverse=True)
print(f"Reverse sorted: {reverse_sorted}")

# Sorting with a key function (sort by absolute value)
numbers_with_negatives = [3, -1, 4, -5, 2, -6]
abs_sorted = sorted(numbers_with_negatives, key=abs)
print(f"Original list: {numbers_with_negatives}")
print(f"Sorted by absolute value: {abs_sorted}")

# Sorting with a lambda key function (sort by length)
words = ["apple", "banana", "cherry", "date", "elderberry"]
length_sorted = sorted(words, key=lambda x: len(x))
print(f"Original words: {words}")
print(f"Sorted by length: {length_sorted}")

# Sorting complex objects (list of tuples)
students = [
    ("Alice", 23, "Computer Science"),
    ("Bob", 21, "Mathematics"),
    ("Charlie", 22, "Physics"),
    ("David", 21, "Engineering")
]

# Sort by age
age_sorted = sorted(students, key=lambda student: student[1])
print(f"Sorted by age: {age_sorted}")

# Sort by name
name_sorted = sorted(students, key=lambda student: student[0])
print(f"Sorted by name: {name_sorted}")

# Sort by multiple criteria (first by age, then by name for ties)
from operator import itemgetter
multi_sorted = sorted(students, key=itemgetter(1, 0))
print(f"Sorted by age then name: {multi_sorted}")

# Using custom class with sorted
print("\n# Sorting Custom Objects")
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

people = [
    Person("Alice", 30),
    Person("Bob", 25),
    Person("Charlie", 35),
    Person("David", 30)
]

# Sort by age
people_by_age = sorted(people, key=lambda person: person.age)
print(f"People sorted by age: {people_by_age}")

# Sort by name
people_by_name = sorted(people, key=lambda person: person.name)
print(f"People sorted by name: {people_by_name}")

# Sort by age then name (for same age)
people_by_age_name = sorted(people, key=lambda person: (person.age, person.name))
print(f"People sorted by age then name: {people_by_age_name}")

print("\n# Edge Cases and Exceptions")
# Empty iterable
print(f"Sorted empty list: {sorted([])}")  # Returns empty list

# Heterogeneous lists (mixing types can cause issues)
try:
    sorted([1, 'a', 2, 'b'])  # TypeError: '<' not supported between instances of 'str' and 'int'
except TypeError as e:
    print(f"Error sorting mixed types: {e}")

# Non-comparable objects
try:
    sorted([{1}, {2}, {3}])  # TypeError: '<' not supported between instances of 'set' and 'set'
except TypeError as e:
    print(f"Error sorting non-comparable objects: {e}")

# Non-iterable object
try:
    sorted(123)  # TypeError: 'int' object is not iterable
except TypeError as e:
    print(f"Error sorting non-iterable: {e}")

# Using a key function that may raise exceptions
try:
    # Will raise ZeroDivisionError for the 0 element
    sorted([1, 2, 0, 3], key=lambda x: 1/x)
except ZeroDivisionError as e:
    print(f"Error in key function: {e}")

print("\n# Performance and Best Practices")
# Using key= vs. custom comparator in Python 2 (cmp parameter)
# In modern Python, always use key= as it's more efficient

# Timsort - Python's sorting algorithm
# - Stable sort (preserves order of equal elements)
# - Adaptive (performs well on partially sorted data)
# - Best case: O(n) for already sorted data
# - Average and worst case: O(n log n)

# For very large datasets, consider using specialized libraries or algorithms
import random
import time

# Example showing performance difference
large_list = list(range(10000))
random.shuffle(large_list)

# Time to sort large list
start = time.time()
sorted(large_list)
end = time.time()
print(f"Time to sort 10,000 items: {(end - start) * 1000:.2f} ms")

# Pre-computing keys for repeated sorting
students = [
    ("Alice", 23, "Computer Science"),
    ("Bob", 21, "Mathematics"),
    # ... imagine 10,000 students
]

# Instead of:
# sorted(students, key=lambda s: complex_calculation(s))

# Better to do:
# decorated = [(complex_calculation(s), s) for s in students]
# decorated.sort()  # or decorated = sorted(decorated)
# result = [s for _, s in decorated]

##############################################################################
# REVERSED() FUNCTION
##############################################################################
"""
The reversed() function returns a reverse iterator that accesses the given sequence
in the reverse order.

Syntax: reversed(seq)

Parameters:
- seq: Required. A sequence or any object that implements the __reversed__() 
  or the __getitem__() and __len__() methods

Return Value:
- A reverse iterator (not a list)

Time Complexity: O(1) for the call itself, as it returns an iterator
Space Complexity: O(1) as it's just an iterator, not a new copy of the data
"""

print("\n" + "#" * 70)
print("# REVERSED() FUNCTION")
print("#" * 70)

print("\n# Basic Usage of reversed()")
# Reversing a list
numbers = [1, 2, 3, 4, 5]
reversed_iterator = reversed(numbers)
print(f"Original list: {numbers}")
print(f"reversed() result (iterator): {reversed_iterator}")
print(f"List from reversed(): {list(reversed(numbers))}")

# Reversing a tuple
my_tuple = (1, 2, 3, 4, 5)
print(f"Original tuple: {my_tuple}")
print(f"Tuple from reversed(): {tuple(reversed(my_tuple))}")

# Reversing a string
text = "Python"
print(f"Original string: {text}")
print(f"String from reversed(): {''.join(reversed(text))}")

# Reversing a range
my_range = range(1, 6)
print(f"Original range: {list(my_range)}")
print(f"Range from reversed(): {list(reversed(my_range))}")

print("\n# Advanced Usage of reversed()")
# Using reversed in a for loop
print("Counting down:")
for i in reversed(range(1, 6)):
    print(i, end=" ")
print()

# Converting to different types
numbers = [1, 2, 3, 4, 5]
reversed_list = list(reversed(numbers))
reversed_tuple = tuple(reversed(numbers))
reversed_set = set(reversed(numbers))  # Note: sets don't maintain order

print(f"Reversed as list: {reversed_list}")
print(f"Reversed as tuple: {reversed_tuple}")
print(f"Reversed as set (no order): {reversed_set}")

# Using with custom objects
print("\n# Using reversed() with Custom Objects")
class CountDown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        # Forward iterator
        n = 1
        while n <= self.start:
            yield n
            n += 1
    
    def __reversed__(self):
        # Reverse iterator
        n = self.start
        while n >= 1:
            yield n
            n -= 1

countdown = CountDown(5)
print("Forward iteration:", end=" ")
for i in countdown:
    print(i, end=" ")
print()

print("Reversed iteration:", end=" ")
for i in reversed(countdown):
    print(i, end=" ")
print()

# Using reversed on a custom sequence
class CustomSequence:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __repr__(self):
        return f"CustomSequence({self.data})"

seq = CustomSequence([1, 2, 3, 4, 5])
print(f"Custom sequence: {seq}")
print(f"Reversed custom sequence: {list(reversed(seq))}")

print("\n# Edge Cases and Exceptions")
# Empty sequence
print(f"Reversed empty list: {list(reversed([]))}")  # Empty list

# Non-sequence object without __reversed__, __len__, or __getitem__
try:
    reversed(123)  # TypeError: 'int' object is not reversible
except TypeError as e:
    print(f"Error reversing non-sequence: {e}")

# Object with __len__ but no __getitem__
class LenOnly:
    def __len__(self):
        return 5

try:
    reversed(LenOnly())  # TypeError: 'LenOnly' object is not reversible
except TypeError as e:
    print(f"Error with incomplete sequence protocol: {e}")

# Dictionary (only keys are reversed in older Python versions, or view objects in newer versions)
my_dict = {'a': 1, 'b': 2, 'c': 3}
print(f"Original dict: {my_dict}")

# In modern Python, you need to specify what to reverse
print(f"Reversed dict keys: {list(reversed(list(my_dict.keys())))}")

print("\n# Performance and Best Practices")
# reversed() is more memory efficient than creating a reversed copy
import time
import sys

large_list = list(range(1000000))

# Using reversed (iterator approach)
start = time.time()
sum_rev = sum(reversed(large_list))
end = time.time()
rev_time = end - start

# Using slicing to create a reversed copy
start = time.time()
sum_slice = sum(large_list[::-1])
end = time.time()
slice_time = end - start

print(f"Time with reversed(): {rev_time:.6f} seconds")
print(f"Time with slicing: {slice_time:.6f} seconds")

# Memory usage comparison
rev_iterator = reversed(large_list)
reversed_list = large_list[::-1]

# Note: sys.getsizeof doesn't include the size of contained objects for containers
# This is just to demonstrate the size difference between an iterator and a full list
print(f"Size of reversed iterator: {sys.getsizeof(rev_iterator)} bytes")
print(f"Size of reversed list copy: {sys.getsizeof(reversed_list)} bytes")

# When to use reversed():
# 1. When you need to process elements in reverse order without modifying the original
# 2. When working with large sequences to save memory
# 3. When you only need to iterate once over the reversed sequence

# When NOT to use reversed():
# 1. When you need random access to the reversed elements (use slicing instead)
# 2. When you need to reverse in-place (use .reverse() for lists)

##############################################################################
# ITER() FUNCTION
##############################################################################
"""
The iter() function returns an iterator object for the given object.

Syntax: iter(object, sentinel)

Parameters:
- object: Required. An object that implements the __iter__() method or the __getitem__() method
- sentinel: Optional. If provided, object must be a callable, and the iterator will call
  object until it returns sentinel

Return Value:
- An iterator object

Time Complexity: O(1) for the call itself
Space Complexity: O(1) as it returns an iterator object, not a copy of the data
"""

print("\n" + "#" * 70)
print("# ITER() FUNCTION")
print("#" * 70)

print("\n# Basic Usage of iter()")
# Creating an iterator from a list
numbers = [1, 2, 3, 4, 5]
iterator = iter(numbers)
print(f"Original list: {numbers}")
print(f"Iterator object: {iterator}")

# Using next() with iterator
print("Values from iterator:", end=" ")
print(next(iterator), end=" ")  # 1
print(next(iterator), end=" ")  # 2
print(next(iterator), end=" ")  # 3
print(next(iterator), end=" ")  # 4
print(next(iterator))           # 5

# StopIteration exception when iterator is exhausted
try:
    next(iterator)  # StopIteration exception
except StopIteration:
    print("Iterator exhausted")

# Creating iterators from different iterable types
list_iter = iter([1, 2, 3])
tuple_iter = iter((4, 5, 6))
string_iter = iter("abc")
dict_iter = iter({"x": 1, "y": 2, "z": 3})  # Iterates over keys
set_iter = iter({7, 8, 9})

print(f"List iterator: {next(list_iter)}")
print(f"Tuple iterator: {next(tuple_iter)}")
print(f"String iterator: {next(string_iter)}")
print(f"Dict iterator: {next(dict_iter)}")
print(f"Set iterator: {next(set_iter)}")

print("\n# Advanced Usage of iter()")
# Using the two-argument form of iter() with a sentinel value
# This creates an iterator that calls the function until it returns the sentinel value
import random

# Example: Read lines from a file until an empty string is encountered
from io import StringIO

# Create a mock file-like object
mock_file = StringIO("Line 1\nLine 2\nLine 3\n")

# Create an iterator that reads lines until an empty string
line_iterator = iter(mock_file.readline, "")

print("Lines from file:")
for line in line_iterator:
    print(f"  {line.strip()}")

# Example: Generate random numbers until 0 is encountered
def random_until_zero():
    """Returns a random integer between 0 and 9."""
    return random.randint(0, 9)

random_iterator = iter(random_until_zero, 0)

print("Random numbers until 0:")
numbers = []
for num in random_iterator:
    numbers.append(num)
print(f"  {numbers}")

# Using iter() with custom objects
print("\n# Using iter() with Custom Objects")
class CountUp:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0
    
    def __iter__(self):
        # Return self as the iterator
        self.current = 0
        return self
    
    def __next__(self):
        # Implement the next method for the iterator protocol
        if self.current < self.limit:
            self.current += 1
            return self.current
        else:
            raise StopIteration

count = CountUp(5)
iterator = iter(count)

print("Custom iterator values:", end=" ")
try:
    while True:
        print(next(iterator), end=" ")
except StopIteration:
    print()

# Reusable iterators with __iter__
print("\nReusing the iterable:")
for i in count:  # This works because count has __iter__
    print(i, end=" ")
print()

# Another iteration (reusable)
for i in count:
    print(i, end=" ")
print()

# Example with a generator function
def fibonacci(n):
    """Generate first n Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

fib_iter = iter(fibonacci(10))
print("Fibonacci numbers:", end=" ")
for num in fib_iter:
    print(num, end=" ")
print()

print("\n# Edge Cases and Exceptions")
# Empty iterable
empty_iter = iter([])
try:
    next(empty_iter)  # StopIteration
except StopIteration:
    print("Empty iterator raises StopIteration immediately")

# Non-iterable object
try:
    iter(123)  # TypeError: 'int' object is not iterable
except TypeError as e:
    print(f"Error with non-iterable: {e}")

# Calling iter() on an iterator returns the iterator itself
numbers = [1, 2, 3]
iterator1 = iter(numbers)
iterator2 = iter(iterator1)  # Returns the same iterator
print(f"iterator1 is iterator2: {iterator1 is iterator2}")  # True

# The sentinel form with a non-callable first argument
try:
    iter("not callable", None)  # TypeError: 'str' object is not callable
except TypeError as e:
    print(f"Error with non-callable in sentinel form: {e}")

print("\n# Performance and Best Practices")
# Iterators are memory efficient for large datasets
import sys

# Memory usage for a large range
large_range = range(1000000)
large_list = list(large_range)

range_iterator = iter(large_range)
list_iterator = iter(large_list)

print(f"Size of range object: {sys.getsizeof(large_range)} bytes")
print(f"Size of list: {sys.getsizeof(large_list)} bytes")
print(f"Size of range iterator: {sys.getsizeof(range_iterator)} bytes")
print(f"Size of list iterator: {sys.getsizeof(list_iterator)} bytes")

# Using iterators for lazy evaluation
print("\n# Lazy Evaluation with Iterators")
# This doesn't calculate all values at once
lazy_squares = map(lambda x: x**2, range(10))
print(f"Lazy evaluation object: {lazy_squares}")
print(f"First few squares: {next(lazy_squares)}, {next(lazy_squares)}, {next(lazy_squares)}")

# Creating infinite sequences with iterators
from itertools import count

infinite_counter = count(1)  # Starts from 1 and counts indefinitely
print(f"Infinite sequence elements: {next(infinite_counter)}, {next(infinite_counter)}, {next(infinite_counter)}")

# Practical applications of iterators
print("\n# Practical Applications")

# 1. Processing large files line by line
print("1. File processing (simulated):")
mock_file = StringIO("Line 1\nLine 2\nLine 3\n")
for line in iter(mock_file.readline, ""):
    print(f"  Processing: {line.strip()}")

# 2. Working with database cursors
print("2. Database cursor iteration (simulated):")
class MockCursor:
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def fetchone(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        return None

mock_db = MockCursor(["Record 1", "Record 2", "Record 3"])
record_iter = iter(mock_db.fetchone, None)
for record in record_iter:
    print(f"  Processing: {record}")

# 3. Custom iterator for pairwise elements
print("3. Pairwise iteration:")
import itertools

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iter(iterable))
    next(b, None)
    return zip(a, b)

data = [1, 2, 3, 4, 5]
for pair in pairwise(data):
    print(f"  Pair: {pair}")

# 4. Implementing a circular buffer with an iterator
print("4. Circular buffer:")
import itertools

def circular_iterator(iterable):
    """Returns an infinite iterator that cycles through the elements."""
    return itertools.cycle(iterable)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
day_iter = circular_iterator(days)

for _ in range(10):
    print(f"  {next(day_iter)}", end=" ")
print()

##############################################################################
# COMMON ITERATOR TOOLS AND PATTERNS
##############################################################################
"""
Combining sorted(), reversed(), and iter() with other iterator tools can provide
powerful data processing capabilities.
"""

print("\n" + "#" * 70)
print("# COMBINING BUILT-IN FUNCTIONS WITH ITERTOOLS")
print("#" * 70)

import itertools

print("\n# Efficient Data Processing Examples")

# 1. Taking a slice of a large sorted dataset
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
# Get the 3 smallest unique values
smallest_three = list(itertools.islice(sorted(set(data)), 3))
print(f"Three smallest unique values: {smallest_three}")

# 2. Chunking data with sorted ordering
def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk

sorted_data = sorted(data)
for chunk in chunked(sorted_data, 3):
    print(f"Chunk: {chunk}")

# 3. Finding most frequent items
from collections import Counter
most_common = Counter(data).most_common(2)
print(f"Two most common elements: {most_common}")

# 4. Working with permutations and combinations
letters = 'ABC'
print("Permutations of 'ABC':")
for p in itertools.permutations(letters):
    print(f"  {''.join(p)}", end=" ")
print()

print("Combinations of 'ABC' (choose 2):")
for c in itertools.combinations(letters, 2):
    print(f"  {''.join(c)}", end=" ")
print()

# 5. Grouping data
people = [
    ("John", 25, "Engineer"),
    ("Jane", 30, "Doctor"),
    ("Mike", 25, "Teacher"),
    ("Alex", 30, "Engineer"),
]

# Sort by age first
sorted_people = sorted(people, key=lambda x: x[1])
# Then group by age
for age, group in itertools.groupby(sorted_people, key=lambda x: x[1]):
    print(f"Age {age}:")
    for person in group:
        print(f"  {person[0]} - {person[2]}")

# 6. Applying multiple sorted orders
print("\n# Multiple Sorted Orders")
# Sort by multiple fields in reverse order
data = [
    ("apple", 5, 1.5),
    ("banana", 3, 2.0),
    ("cherry", 5, 1.0),
    ("date", 2, 2.5),
]

# Sort by field 1 (descending), then field 2 (ascending)
multi_sorted = sorted(
    data,
    key=lambda x: (-x[1], x[2])  # Negative for descending order
)
print("Items sorted by quantity (desc) then price (asc):")
for item in multi_sorted:
    print(f"  {item[0]}: {item[1]} units at ${item[2]:.2f} each")

# 7. Reversing specific parts of data
print("\n# Partial Reversal")
words = ["apple", "banana", "cherry", "date"]
# Sort alphabetically, but display the individual letters reversed
backwards_sorted = sorted(
    words, 
    key=lambda x: ''.join(reversed(x))
)
reversed_words = [''.join(reversed(word)) for word in words]
print(f"Words: {words}")
print(f"Reversed words: {reversed_words}")
print(f"Sorted by reversed spelling: {backwards_sorted}")

# 8. Infinite iterator with sentinel
print("\n# Using Infinite Iterators")
count_iter = itertools.count()  # 0, 1, 2, ...
squares = map(lambda x: x**2, count_iter)
# Get first 5 square numbers
print("First 5 square numbers:", end=" ")
for _ in range(5):
    print(next(squares), end=" ")
print()

##############################################################################
# CONCLUSION
##############################################################################
"""
These built-in functions (sorted(), reversed(), and iter()) are fundamental 
tools for effective Python programming:

sorted(): Creates a new sorted list from an iterable with full control over 
          sorting criteria through the key and reverse parameters.

reversed(): Creates a reverse iterator that traverses the sequence in reverse
            order without creating a copy of the data.

iter(): Returns an iterator from an iterable object or creates a callable-based
        iterator using a sentinel value.

Mastering these functions and combining them with other tools like itertools
allows for efficient, pythonic code that handles data processing tasks
with elegance and performance.
"""