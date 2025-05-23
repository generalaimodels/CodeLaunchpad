"""
Python Tuple Methods - Comprehensive Guide
-----------------------------------------

This file provides a detailed explanation of all built-in tuple methods in Python.
Each method includes explanation, syntax, examples, edge cases, time complexity,
and exception handling.

Tuples are immutable sequences, typically used to store collections of heterogeneous
data. They are similar to lists but cannot be modified after creation.

Since tuples are immutable, they have only two built-in methods:
1. count() - counts occurrences of a value
2. index() - finds the position of a value

Time complexity notation:
- O(n): Linear time - operation time grows linearly with tuple size
"""

# -----------------------------------------------------------------------------
# tuple.count(value)
# -----------------------------------------------------------------------------
"""
PURPOSE: Counts the number of occurrences of a specified value in the tuple
PARAMETERS: value - The element to count occurrences of
RETURN VALUE: An integer representing the count of occurrences
TIME COMPLEXITY: O(n) where n is the length of the tuple
"""

# Basic usage with numbers
numbers = (1, 2, 3, 1, 4, 1, 5)
count_of_1 = numbers.count(1)
print(f"Count of 1 in {numbers}: {count_of_1}")  # Output: 3

# Basic usage with strings
fruits = ("apple", "banana", "apple", "cherry", "apple", "dragonfruit")
count_of_apple = fruits.count("apple")
print(f"Count of 'apple' in {fruits}: {count_of_apple}")  # Output: 3

# Case sensitivity with strings
words = ("Python", "python", "PYTHON", "Python")
count_python_exact = words.count("Python")
print(f"Count of 'Python' (case-sensitive) in {words}: {count_python_exact}")  # Output: 2

# Counting elements that don't exist returns 0 (no exception)
nonexistent_count = numbers.count(99)
print(f"Count of 99 in {numbers}: {nonexistent_count}")  # Output: 0

# Counting in an empty tuple
empty_tuple = ()
print(f"Count of anything in an empty tuple: {empty_tuple.count(5)}")  # Output: 0

# Counting with mixed data types
mixed_tuple = (1, "hello", 3.14, True, 1, "hello", None)
print(f"Count of 'hello' in mixed tuple: {mixed_tuple.count('hello')}")  # Output: 2
print(f"Count of 1 in mixed tuple: {mixed_tuple.count(1)}")  # Output: 2
# Note: True is considered equal to 1 in Python, so this might count as 3 instead of 2
# depending on your Python version and implementation

# Counting complex objects (demonstrates equality vs identity)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

# Two distinct objects with the same values
point_tuple = (Point(1, 2), Point(3, 4), Point(1, 2))
# This counts objects that are equal (not identical)
print(f"Count of Point(1, 2): {point_tuple.count(Point(1, 2))}")  # Output: 2

# Counting with objects without custom equality
class SimplePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Without __eq__, object identity is used instead of equality
simple_points = (SimplePoint(1, 2), SimplePoint(3, 4), SimplePoint(1, 2))
print(f"Count of SimplePoint(1, 2): {simple_points.count(SimplePoint(1, 2))}")  # Output: 0
# The count is 0 because the SimplePoint(1, 2) we're looking for is a different object

# -----------------------------------------------------------------------------
# tuple.index(value[, start[, end]])
# -----------------------------------------------------------------------------
"""
PURPOSE: Returns the first index of the specified value in the tuple
PARAMETERS: 
    value - The element to find
    start - Optional starting position (default: 0)
    end - Optional ending position (default: end of tuple)
RETURN VALUE: Integer index of the first occurrence
TIME COMPLEXITY: O(n) where n is the length of the tuple
EXCEPTIONS: ValueError if the value is not found
"""

# Basic usage
colors = ("red", "green", "blue", "yellow", "green", "purple")
index_of_blue = colors.index("blue")
print(f"Index of 'blue' in {colors}: {index_of_blue}")  # Output: 2

# Finding first occurrence when multiple exist
index_of_green = colors.index("green")
print(f"Index of first 'green' in {colors}: {index_of_green}")  # Output: 1

# Using the start parameter (begin searching from index 2)
index_of_green_after_2 = colors.index("green", 2)
print(f"Index of 'green' starting from position 2: {index_of_green_after_2}")  # Output: 4

# Using both start and end parameters (search between indices 1 and 3)
index_of_yellow_range = colors.index("yellow", 1, 5)
print(f"Index of 'yellow' between positions 1 and 5: {index_of_yellow_range}")  # Output: 3

# EXCEPTION: ValueError when element is not found
try:
    colors.index("black")
except ValueError as e:
    print(f"Error when finding 'black': {e}")  # Output: 'black' is not in tuple

# EXCEPTION: ValueError when element is not found in the specified range
try:
    colors.index("red", 1)  # "red" is at index 0, outside the search range
except ValueError as e:
    print(f"Error when finding 'red' after index 1: {e}")  # Output: 'red' is not in tuple

# Using index with numeric tuples
numbers = (10, 20, 30, 40, 30, 50)
print(f"Index of 30: {numbers.index(30)}")  # Output: 2 (first occurrence)
print(f"Index of 30 after position 3: {numbers.index(30, 3)}")  # Output: 4 (second occurrence)

# Working with boolean values
bool_tuple = (True, False, True, True)
print(f"Index of True: {bool_tuple.index(True)}")  # Output: 0 (first occurrence)
print(f"Index of False: {bool_tuple.index(False)}")  # Output: 1

# Edge case: Using very large start/end values doesn't cause IndexError
print(f"Search with large end value: {colors.index('purple', 0, 1000)}")  # Output: 5
try:
    # This will still raise ValueError because the search is valid, 
    # but the element doesn't exist in that range
    colors.index('purple', 1000)
except ValueError as e:
    print(f"Error when searching out of range: {e}")

# Working with objects and equality
point_tuple = (Point(1, 2), Point(3, 4), Point(1, 2))
print(f"Index of Point(1, 2): {point_tuple.index(Point(1, 2))}")  # Output: 0

# EXCEPTION: Using index on an empty tuple always raises ValueError
try:
    empty_tuple = ()
    empty_tuple.index(5)
except ValueError as e:
    print(f"Error when using index on an empty tuple: {e}")

# -----------------------------------------------------------------------------
# PRACTICAL EXAMPLES: Real-world usage of tuple methods
# -----------------------------------------------------------------------------

# Example 1: Finding duplicates in a dataset
def find_duplicates(data):
    """Return a list of items that appear more than once in the data."""
    return [item for item in set(data) if data.count(item) > 1]

sample_data = ("user123", "user456", "user789", "user123", "user456", "user999")
duplicates = find_duplicates(sample_data)
print(f"\nDuplicate users: {duplicates}")  # Output: ['user123', 'user456']

# Example 2: Simple tokenizer that preserves token positions
def tokenize_sentence(sentence):
    """
    Split a sentence into tokens and provide ability to 
    find original positions of specific words.
    """
    tokens = tuple(sentence.lower().split())
    
    def find_all_positions(word):
        positions = []
        start = 0
        while True:
            try:
                pos = tokens.index(word.lower(), start)
                positions.append(pos)
                start = pos + 1
            except ValueError:
                break
        return positions
    
    return tokens, find_all_positions

text = "Python is great and Python is easy to learn"
tokens, find_positions = tokenize_sentence(text)
print(f"\nTokens: {tokens}")
print(f"Positions of 'python': {find_positions('python')}")  # Output: [0, 3]
print(f"Positions of 'is': {find_positions('is')}")  # Output: [1, 4]
print(f"Positions of 'not_found': {find_positions('not_found')}")  # Output: []

# Example 3: Implementing a basic frequency counter
def frequency_counter(items):
    """Return a dictionary with items and their frequencies."""
    unique_items = set(items)
    return {item: items.count(item) for item in unique_items}

words = ("apple", "banana", "apple", "cherry", "banana", "apple")
frequencies = frequency_counter(words)
print(f"\nWord frequencies: {frequencies}")
# Output: {'apple': 3, 'banana': 2, 'cherry': 1}

# Example 4: Finding the most frequent element
def most_frequent(items):
    """Return the most frequently occurring item."""
    if not items:
        return None
    
    frequencies = frequency_counter(items)
    return max(frequencies, key=frequencies.get)

numbers = (1, 2, 3, 2, 2, 3, 1, 2, 4, 5)
most_common = most_frequent(numbers)
print(f"\nMost frequent number: {most_common}")  # Output: 2

# Example 5: Implementing a basic version tracker
def track_versions(version_history):
    """Track software versions and provide version information."""
    versions = tuple(version_history)
    
    def get_version_info(version):
        try:
            index = versions.index(version)
            return {
                "exists": True,
                "position": index,
                "is_latest": index == len(versions) - 1,
                "occurrences": versions.count(version),
                "newer_versions": len(versions) - index - 1
            }
        except ValueError:
            return {"exists": False}
    
    return get_version_info

release_history = ("1.0.0", "1.0.1", "1.1.0", "1.1.1", "2.0.0")
version_tracker = track_versions(release_history)

print("\nVersion information:")
print(f"v1.0.1: {version_tracker('1.0.1')}")
# Output: {'exists': True, 'position': 1, 'is_latest': False, 'occurrences': 1, 'newer_versions': 3}
print(f"v2.0.0: {version_tracker('2.0.0')}")
# Output: {'exists': True, 'position': 4, 'is_latest': True, 'occurrences': 1, 'newer_versions': 0}
print(f"v3.0.0: {version_tracker('3.0.0')}")
# Output: {'exists': False}