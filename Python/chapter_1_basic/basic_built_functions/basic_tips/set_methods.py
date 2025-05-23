"""
Python Set Methods - Comprehensive Guide
----------------------------------------

This file provides a detailed explanation of all built-in set methods in Python.
Each method includes explanation, syntax, examples, edge cases, and time complexity.

Sets are unordered collections of unique elements. They support mathematical operations
like union, intersection, difference, and symmetric difference.

Time complexity notation:
- O(1): Constant time - operation takes the same time regardless of set size
- O(n): Linear time - operation time grows linearly with set size
- O(len(s)): Time depends on the length of set s
"""

# -----------------------------------------------------------------------------
# set.add(elem)
# -----------------------------------------------------------------------------
"""
PURPOSE: Adds an element to the set if it's not already present
PARAMETERS: elem - The element to add to the set
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(1) average case
"""

# Basic usage
fruits = {"apple", "banana", "cherry"}
fruits.add("orange")
print(f"After add('orange'): {fruits}")  # Output: {'apple', 'banana', 'cherry', 'orange'}

# Adding an existing element (no change)
fruits.add("apple")
print(f"After add('apple'): {fruits}")  # Output: {'apple', 'banana', 'cherry', 'orange'}

# Adding different types (sets can contain different types, but must be hashable)
mixed_set = {1, "hello", 3.14}
mixed_set.add(True)
print(f"Mixed set after adding bool: {mixed_set}")

# EXCEPTION: Adding unhashable types causes TypeError
try:
    fruits.add([1, 2, 3])  # Lists are mutable (unhashable)
except TypeError as e:
    print(f"Error when adding list: {e}")

# -----------------------------------------------------------------------------
# set.clear()
# -----------------------------------------------------------------------------
"""
PURPOSE: Removes all elements from the set
PARAMETERS: None
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(1)
"""

numbers = {1, 2, 3, 4, 5}
print(f"Before clear: {numbers}")  # Output: {1, 2, 3, 4, 5}
numbers.clear()
print(f"After clear: {numbers}")   # Output: set()

# Empty set behavior
empty_set = set()
empty_set.clear()  # No error, still results in empty set
print(f"Empty set after clear: {empty_set}")  # Output: set()

# -----------------------------------------------------------------------------
# set.copy()
# -----------------------------------------------------------------------------
"""
PURPOSE: Returns a shallow copy of the set
PARAMETERS: None
RETURN VALUE: A new set containing the same elements
TIME COMPLEXITY: O(n) where n is the number of elements
"""

original = {1, 2, 3}
copied = original.copy()
print(f"Original: {original}, Copy: {copied}")  # Both: {1, 2, 3}

# Demonstrating that it's a separate object
original.add(4)
print(f"Original after modification: {original}")  # {1, 2, 3, 4}
print(f"Copy remains unchanged: {copied}")        # {1, 2, 3}

# Shallow copy behavior with nested objects
nested = {1, 2, (3, 4)}
nested_copy = nested.copy()

# With immutable nested objects, modification isn't possible so no concern for shallow copy

# -----------------------------------------------------------------------------
# set.difference(other_set, ...)
# -----------------------------------------------------------------------------
"""
PURPOSE: Returns a new set with elements that are in this set but not in others
PARAMETERS: other_set - Another set or iterable
            ... - Additional sets or iterables (can take multiple arguments)
RETURN VALUE: A new set containing the difference
TIME COMPLEXITY: O(len(self) + len(other_set1) + len(other_set2) + ...)
ALTERNATIVE SYNTAX: set1 - set2 - set3 - ...
"""

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7}

# Basic difference
diff = set1.difference(set2)
print(f"set1.difference(set2): {diff}")  # {1, 2, 3}

# Operator syntax
operator_diff = set1 - set2
print(f"set1 - set2: {operator_diff}")   # {1, 2, 3}

# Multiple sets difference
set3 = {1, 8, 9}
multi_diff = set1.difference(set2, set3)
print(f"set1.difference(set2, set3): {multi_diff}")  # {2, 3}

# Empty set behavior
print(f"set1.difference(set()): {set1.difference(set())}")  # {1, 2, 3, 4, 5}
print(f"set().difference(set1): {set().difference(set1)}")  # set()

# With other iterables
print(f"set1.difference([4, 5, 6]): {set1.difference([4, 5, 6])}")  # {1, 2, 3}

# -----------------------------------------------------------------------------
# set.difference_update(other_set, ...)
# -----------------------------------------------------------------------------
"""
PURPOSE: Removes elements found in other sets from this set (in-place modification)
PARAMETERS: other_set - Another set or iterable
            ... - Additional sets or iterables (can take multiple arguments)
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(len(self) + len(other_set1) + len(other_set2) + ...)
ALTERNATIVE SYNTAX: set1 -= other_set
"""

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6}
print(f"Before difference_update: {set1}")  # {1, 2, 3, 4, 5}

set1.difference_update(set2)
print(f"After difference_update: {set1}")   # {1, 2, 3}

# Operator syntax
set1 = {1, 2, 3, 4, 5}  # Reset set1
set1 -= set2
print(f"After -= operator: {set1}")  # {1, 2, 3}

# Multiple sets
set1 = {1, 2, 3, 4, 5}  # Reset set1
set3 = {1, 8, 9}
set1.difference_update(set2, set3)
print(f"After difference_update with multiple sets: {set1}")  # {2, 3}

# With other iterables
set1 = {1, 2, 3, 4, 5}  # Reset set1
set1.difference_update([4, 5, 6])
print(f"After difference_update with list: {set1}")  # {1, 2, 3}

# Empty set has no effect
set1 = {1, 2, 3}  # Reset set1
set1.difference_update(set())
print(f"After difference_update with empty set: {set1}")  # {1, 2, 3}

# -----------------------------------------------------------------------------
# set.discard(elem)
# -----------------------------------------------------------------------------
"""
PURPOSE: Removes an element from the set if it exists
PARAMETERS: elem - The element to remove
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(1) average case
"""

fruits = {"apple", "banana", "cherry"}
print(f"Before discard: {fruits}")  # {'apple', 'banana', 'cherry'}

# Removing an existing element
fruits.discard("banana")
print(f"After discard('banana'): {fruits}")  # {'apple', 'cherry'}

# Key difference from remove(): discarding a non-existent element does NOT raise an error
fruits.discard("mango")  # No error
print(f"After discard('mango'): {fruits}")  # {'apple', 'cherry'}

# Empty set behavior
empty_set = set()
empty_set.discard("anything")  # No error
print(f"Empty set after discard: {empty_set}")  # set()

# -----------------------------------------------------------------------------
# set.intersection(other_set, ...)
# -----------------------------------------------------------------------------
"""
PURPOSE: Returns a new set with elements common to this set and all others
PARAMETERS: other_set - Another set or iterable
            ... - Additional sets or iterables (can take multiple arguments)
RETURN VALUE: A new set containing the intersection
TIME COMPLEXITY: O(min(len(self), len(other_set))) for two sets
                 More complex with multiple sets
ALTERNATIVE SYNTAX: set1 & set2 & set3 & ...
"""

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7}
set3 = {1, 4, 9}

# Basic intersection
common = set1.intersection(set2)
print(f"set1.intersection(set2): {common}")  # {4, 5}

# Operator syntax
operator_common = set1 & set2
print(f"set1 & set2: {operator_common}")  # {4, 5}

# Multiple sets
multi_common = set1.intersection(set2, set3)
print(f"Intersection of 3 sets: {multi_common}")  # {4}

# Empty intersection
no_common = set1.intersection({8, 9, 10})
print(f"No common elements: {no_common}")  # set()

# With other iterables
print(f"set1.intersection([4, 5, 6]): {set1.intersection([4, 5, 6])}")  # {4, 5}

# Empty set intersection always results in empty set
print(f"set1.intersection(set()): {set1.intersection(set())}")  # set()

# -----------------------------------------------------------------------------
# set.intersection_update(other_set, ...)
# -----------------------------------------------------------------------------
"""
PURPOSE: Updates the set, keeping only elements found in both it and all others (in-place)
PARAMETERS: other_set - Another set or iterable
            ... - Additional sets or iterables (can take multiple arguments)
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(len(self) + len(other_set1) + len(other_set2) + ...)
ALTERNATIVE SYNTAX: set1 &= other_set
"""

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7}
print(f"Before intersection_update: {set1}")  # {1, 2, 3, 4, 5}

set1.intersection_update(set2)
print(f"After intersection_update: {set1}")  # {4, 5}

# Operator syntax
set1 = {1, 2, 3, 4, 5}  # Reset set1
set1 &= set2
print(f"After &= operator: {set1}")  # {4, 5}

# Multiple sets
set1 = {1, 2, 3, 4, 5}  # Reset set1
set3 = {1, 4, 9}
set1.intersection_update(set2, set3)
print(f"After intersection_update with multiple sets: {set1}")  # {4}

# With other iterables
set1 = {1, 2, 3, 4, 5}  # Reset set1
set1.intersection_update([4, 5, 6])
print(f"After intersection_update with list: {set1}")  # {4, 5}

# Empty set intersection makes the set empty
set1 = {1, 2, 3}  # Reset set1
set1.intersection_update(set())
print(f"After intersection_update with empty set: {set1}")  # set()

# -----------------------------------------------------------------------------
# set.isdisjoint(other_set)
# -----------------------------------------------------------------------------
"""
PURPOSE: Returns True if the set has no elements in common with other_set
PARAMETERS: other_set - Another set or iterable
RETURN VALUE: Boolean (True if sets are disjoint, False otherwise)
TIME COMPLEXITY: O(len(self)) if len(self) < len(other_set), else O(len(other_set))
"""

set1 = {1, 2, 3}
set2 = {4, 5, 6}
set3 = {3, 4, 5}

# Sets with no common elements
print(f"set1.isdisjoint(set2): {set1.isdisjoint(set2)}")  # True

# Sets with common elements
print(f"set1.isdisjoint(set3): {set1.isdisjoint(set3)}")  # False

# With other iterables
print(f"set1.isdisjoint([4, 5, 6]): {set1.isdisjoint([4, 5, 6])}")  # True
print(f"set1.isdisjoint([1, 5, 6]): {set1.isdisjoint([1, 5, 6])}")  # False

# Empty set is disjoint with any set
print(f"set1.isdisjoint(set()): {set1.isdisjoint(set())}")  # True
print(f"set().isdisjoint(set1): {set().isdisjoint(set1)}")  # True

# -----------------------------------------------------------------------------
# set.issubset(other_set)
# -----------------------------------------------------------------------------
"""
PURPOSE: Tests if every element in this set is in other_set
PARAMETERS: other_set - Another set or iterable
RETURN VALUE: Boolean (True if set is subset, False otherwise)
TIME COMPLEXITY: O(len(self))
ALTERNATIVE SYNTAX: set1 <= set2 (subset), set1 < set2 (proper subset)
"""

set1 = {1, 2}
set2 = {1, 2, 3, 4, 5}
set3 = {1, 2}

# Basic subset check
print(f"set1.issubset(set2): {set1.issubset(set2)}")  # True

# Set is a subset of itself
print(f"set1.issubset(set1): {set1.issubset(set1)}")  # True

# Using operators
print(f"set1 <= set2: {set1 <= set2}")  # True (subset)
print(f"set1 < set2: {set1 < set2}")    # True (proper subset)
print(f"set1 <= set3: {set1 <= set3}")  # True (subset)
print(f"set1 < set3: {set1 < set3}")    # False (not a proper subset because they're equal)

# With other iterables
print(f"set1.issubset([1, 2, 3]): {set1.issubset([1, 2, 3])}")  # True

# Empty set is a subset of any set
empty_set = set()
print(f"empty_set.issubset(set1): {empty_set.issubset(set1)}")  # True
print(f"set1.issubset(empty_set): {set1.issubset(empty_set)}")  # False

# -----------------------------------------------------------------------------
# set.issuperset(other_set)
# -----------------------------------------------------------------------------
"""
PURPOSE: Tests if this set contains every element in other_set
PARAMETERS: other_set - Another set or iterable
RETURN VALUE: Boolean (True if set is superset, False otherwise)
TIME COMPLEXITY: O(len(other_set))
ALTERNATIVE SYNTAX: set1 >= set2 (superset), set1 > set2 (proper superset)
"""

set1 = {1, 2, 3, 4, 5}
set2 = {1, 2}
set3 = {1, 2, 3, 4, 5}

# Basic superset check
print(f"set1.issuperset(set2): {set1.issuperset(set2)}")  # True

# Set is a superset of itself
print(f"set1.issuperset(set1): {set1.issuperset(set1)}")  # True

# Using operators
print(f"set1 >= set2: {set1 >= set2}")  # True (superset)
print(f"set1 > set2: {set1 > set2}")    # True (proper superset)
print(f"set1 >= set3: {set1 >= set3}")  # True (superset)
print(f"set1 > set3: {set1 > set3}")    # False (not a proper superset because they're equal)

# With other iterables
print(f"set1.issuperset([1, 2]): {set1.issuperset([1, 2])}")  # True

# Any set is a superset of the empty set
empty_set = set()
print(f"set1.issuperset(empty_set): {set1.issuperset(empty_set)}")  # True
print(f"empty_set.issuperset(set1): {empty_set.issuperset(set1)}")  # False

# -----------------------------------------------------------------------------
# set.pop()
# -----------------------------------------------------------------------------
"""
PURPOSE: Removes and returns an arbitrary element from the set
PARAMETERS: None
RETURN VALUE: The removed element
TIME COMPLEXITY: O(1)
EXCEPTIONS: KeyError if the set is empty
"""

fruits = {"apple", "banana", "cherry"}
print(f"Before pop: {fruits}")  # {'apple', 'banana', 'cherry'}

# Note: Since sets are unordered, we can't predict which element will be popped
popped = fruits.pop()
print(f"Popped element: {popped}")
print(f"After pop: {fruits}")  # Two remaining elements

# Popping all elements
while fruits:
    print(f"Popping: {fruits.pop()}")

# EXCEPTION: Popping from an empty set raises KeyError
try:
    empty_set = set()
    empty_set.pop()
except KeyError as e:
    print(f"Error when popping from empty set: {e}")

# -----------------------------------------------------------------------------
# set.remove(elem)
# -----------------------------------------------------------------------------
"""
PURPOSE: Removes the specified element from the set
PARAMETERS: elem - The element to remove
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(1) average case
EXCEPTIONS: KeyError if the element is not found
"""

fruits = {"apple", "banana", "cherry"}
print(f"Before remove: {fruits}")  # {'apple', 'banana', 'cherry'}

# Removing an existing element
fruits.remove("banana")
print(f"After remove('banana'): {fruits}")  # {'apple', 'cherry'}

# EXCEPTION: Key difference from discard(): removing a non-existent element raises KeyError
try:
    fruits.remove("mango")
except KeyError as e:
    print(f"Error when removing non-existent element: {e}")

# -----------------------------------------------------------------------------
# set.symmetric_difference(other_set)
# -----------------------------------------------------------------------------
"""
PURPOSE: Returns a new set with elements in either this set or other_set but not both
PARAMETERS: other_set - Another set or iterable
RETURN VALUE: A new set with symmetric difference
TIME COMPLEXITY: O(len(self) + len(other_set))
ALTERNATIVE SYNTAX: set1 ^ set2
"""

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Basic symmetric difference
sym_diff = set1.symmetric_difference(set2)
print(f"set1.symmetric_difference(set2): {sym_diff}")  # {1, 2, 5, 6}

# Operator syntax
operator_sym_diff = set1 ^ set2
print(f"set1 ^ set2: {operator_sym_diff}")  # {1, 2, 5, 6}

# Symmetric difference with itself is always empty
print(f"set1.symmetric_difference(set1): {set1.symmetric_difference(set1)}")  # set()

# With other iterables
print(f"set1.symmetric_difference([3, 4, 5, 6]): {set1.symmetric_difference([3, 4, 5, 6])}")  # {1, 2, 5, 6}

# Symmetric difference with empty set gives the original set
print(f"set1.symmetric_difference(set()): {set1.symmetric_difference(set())}")  # {1, 2, 3, 4}

# -----------------------------------------------------------------------------
# set.symmetric_difference_update(other_set)
# -----------------------------------------------------------------------------
"""
PURPOSE: Updates the set with symmetric difference (in-place)
PARAMETERS: other_set - Another set or iterable
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(len(self) + len(other_set))
ALTERNATIVE SYNTAX: set1 ^= other_set
"""

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print(f"Before symmetric_difference_update: {set1}")  # {1, 2, 3, 4}

set1.symmetric_difference_update(set2)
print(f"After symmetric_difference_update: {set1}")  # {1, 2, 5, 6}

# Operator syntax
set1 = {1, 2, 3, 4}  # Reset set1
set1 ^= set2
print(f"After ^= operator: {set1}")  # {1, 2, 5, 6}

# With other iterables
set1 = {1, 2, 3, 4}  # Reset set1
set1.symmetric_difference_update([3, 4, 5, 6])
print(f"After update with list: {set1}")  # {1, 2, 5, 6}

# Update with itself makes it empty
set1 = {1, 2, 3, 4}  # Reset set1
set1.symmetric_difference_update(set1)
print(f"After update with itself: {set1}")  # set()

# -----------------------------------------------------------------------------
# set.union(other_set, ...)
# -----------------------------------------------------------------------------
"""
PURPOSE: Returns a new set with elements from this set and all others
PARAMETERS: other_set - Another set or iterable
            ... - Additional sets or iterables (can take multiple arguments)
RETURN VALUE: A new set containing the union
TIME COMPLEXITY: O(len(self) + len(other_set1) + len(other_set2) + ...)
ALTERNATIVE SYNTAX: set1 | set2 | set3 | ...
"""

set1 = {1, 2, 3}
set2 = {3, 4, 5}
set3 = {5, 6, 7}

# Basic union
combined = set1.union(set2)
print(f"set1.union(set2): {combined}")  # {1, 2, 3, 4, 5}

# Operator syntax
operator_combined = set1 | set2
print(f"set1 | set2: {operator_combined}")  # {1, 2, 3, 4, 5}

# Multiple sets
multi_combined = set1.union(set2, set3)
print(f"Union of 3 sets: {multi_combined}")  # {1, 2, 3, 4, 5, 6, 7}

# With other iterables
print(f"set1.union([3, 4, 5]): {set1.union([3, 4, 5])}")  # {1, 2, 3, 4, 5}

# Union with empty set is the original set
print(f"set1.union(set()): {set1.union(set())}")  # {1, 2, 3}

# -----------------------------------------------------------------------------
# set.update(other_set, ...)
# -----------------------------------------------------------------------------
"""
PURPOSE: Updates the set, adding elements from all others (in-place)
PARAMETERS: other_set - Another set or iterable
            ... - Additional sets or iterables (can take multiple arguments)
RETURN VALUE: None (modifies the set in-place)
TIME COMPLEXITY: O(len(other_set1) + len(other_set2) + ...)
ALTERNATIVE SYNTAX: set1 |= other_set
"""

set1 = {1, 2, 3}
set2 = {3, 4, 5}
print(f"Before update: {set1}")  # {1, 2, 3}

set1.update(set2)
print(f"After update: {set1}")  # {1, 2, 3, 4, 5}

# Operator syntax
set1 = {1, 2, 3}  # Reset set1
set1 |= set2
print(f"After |= operator: {set1}")  # {1, 2, 3, 4, 5}

# Multiple updates
set1 = {1, 2, 3}  # Reset set1
set3 = {5, 6, 7}
set1.update(set2, set3)
print(f"After update with multiple sets: {set1}")  # {1, 2, 3, 4, 5, 6, 7}

# With other iterables
set1 = {1, 2, 3}  # Reset set1
set1.update([3, 4, 5], (6, 7))
print(f"After update with list and tuple: {set1}")  # {1, 2, 3, 4, 5, 6, 7}

# Update with empty set has no effect
set1 = {1, 2, 3}  # Reset set1
set1.update(set())
print(f"After update with empty set: {set1}")  # {1, 2, 3}

# -----------------------------------------------------------------------------
# PRACTICE SECTION: Real-world examples demonstrating set operations
# -----------------------------------------------------------------------------

# Example 1: Finding common interests between users
user1_interests = {"python", "data science", "machine learning", "web development"}
user2_interests = {"javascript", "web development", "ui/ux", "python"}

common_interests = user1_interests.intersection(user2_interests)
print(f"\nCommon interests: {common_interests}")  # {'python', 'web development'}

all_interests = user1_interests.union(user2_interests)
print(f"All unique interests: {all_interests}")

# Example 2: Removing duplicates from a list while preserving order
def remove_duplicates(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

numbers_with_duplicates = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
unique_ordered = remove_duplicates(numbers_with_duplicates)
print(f"\nOriginal list: {numbers_with_duplicates}") 
print(f"After removing duplicates (preserving order): {unique_ordered}")

# Example 3: Set operations for data analysis
all_students = {"Alice", "Bob", "Charlie", "David", "Eve"}
passed_math = {"Alice", "Charlie", "Eve"}
passed_science = {"Bob", "Charlie", "David"}

# Students who passed both subjects
passed_both = passed_math & passed_science
print(f"\nPassed both Math and Science: {passed_both}")  # {'Charlie'}

# Students who passed at least one subject
passed_either = passed_math | passed_science
print(f"Passed either Math or Science: {passed_either}")  # {'Alice', 'Bob', 'Charlie', 'David', 'Eve'}

# Students who passed Math but not Science
math_only = passed_math - passed_science
print(f"Passed only Math: {math_only}")  # {'Alice', 'Eve'}

# Students who passed exactly one subject
passed_exactly_one = passed_math ^ passed_science
print(f"Passed exactly one subject: {passed_exactly_one}")  # {'Alice', 'Bob', 'David', 'Eve'}

# Students who failed both subjects
failed_both = all_students - passed_either
print(f"Failed both subjects: {failed_both}")  # set() (everyone passed at least one)