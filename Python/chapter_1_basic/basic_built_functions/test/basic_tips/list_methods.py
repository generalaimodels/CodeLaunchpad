#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python List Methods - Comprehensive Guide
=========================================

This module provides a comprehensive exploration of Python's built-in list methods.
Each method is thoroughly explained with examples, time complexity analysis,
edge cases, and common pitfalls.

Author: 
Date: 2025-03-14

NOTE: Run this file to see execution of all examples.
"""

import timeit
import sys
from typing import Any, List, TypeVar, Optional, Callable, Union

T = TypeVar('T')


def separator(method_name: str) -> None:
    """Print a separator with the method name for better readability."""
    print(f"\n{'=' * 80}\n{method_name.upper()}\n{'=' * 80}")


# =============================================================================
# list.append() - Add a single element to the end of the list
# =============================================================================

def demonstrate_append() -> None:
    """
    list.append(x) - Add item x to the end of the list
    
    Time Complexity: O(1) - Amortized constant time
    Space Complexity: O(1) - In-place operation (though may trigger reallocation)
    
    Key Points:
    - Adds exactly one element to the end of the list
    - Modifies the list in-place and returns None
    - Works with any object type (immutable or mutable)
    - Appending to large lists is still fast due to over-allocation strategy
    
    Common Pitfalls:
    - Using append() in a loop to build a list can be inefficient compared to comprehensions
    - Mistakenly trying to append multiple items at once (use extend() instead)
    - Ignoring the None return value (append modifies in-place)
    """
    separator("list.append()")
    
    # Basic usage - appending different data types
    numbers = [1, 2, 3]
    numbers.append(4)
    print(f"After appending 4: {numbers}")
    
    # Appending different data types
    mixed_list = []
    mixed_list.append("string")
    mixed_list.append(42)
    mixed_list.append(True)
    mixed_list.append([1, 2, 3])  # Note: This adds the entire list as a single element
    mixed_list.append({"key": "value"})
    print(f"Mixed list after appends: {mixed_list}")
    
    # Nested list example (append adds the entire list as a single element)
    matrix = [[1, 2], [3, 4]]
    matrix.append([5, 6])
    print(f"Matrix after append: {matrix}")
    
    # Return value is None (modifies in-place)
    result = numbers.append(5)
    print(f"Return value of append(): {result}")
    print(f"List after append: {numbers}")
    
    # Performance example - appending to large lists remains efficient
    big_list = []
    start_time = timeit.default_timer()
    for i in range(10000):
        big_list.append(i)
    end_time = timeit.default_timer()
    print(f"Time to append 10,000 elements: {end_time - start_time:.6f} seconds")
    
    # IMPORTANT: List comprehension is more efficient than append in a loop
    start_time = timeit.default_timer()
    better_list = [i for i in range(10000)]
    end_time = timeit.default_timer()
    print(f"Time for equivalent list comprehension: {end_time - start_time:.6f} seconds")


# =============================================================================
# list.clear() - Remove all items from the list
# =============================================================================

def demonstrate_clear() -> None:
    """
    list.clear() - Remove all items from the list
    
    Time Complexity: O(n) - Linear time (proportional to list size)
    Space Complexity: O(1) - In-place operation
    
    Key Points:
    - Removes ALL elements from the list, resulting in an empty list
    - Modifies the list in-place and returns None
    - Equivalent to del list[:] but more readable
    - Maintains the same list object identity
    
    Common Pitfalls:
    - Setting list = [] creates a new empty list rather than clearing the existing one
      (this can cause issues if other variables reference the original list)
    - Does not affect other lists that might contain the cleared list as an element
    """
    separator("list.clear()")
    
    # Basic usage
    numbers = [1, 2, 3, 4, 5]
    print(f"Before clear(): {numbers}")
    numbers.clear()
    print(f"After clear(): {numbers}")
    
    # Comparison with list = []
    original = [1, 2, 3]
    reference = original  # Both variables now reference the same list
    
    # Method 1: Using clear() - maintains identity
    print(f"\nUsing clear() method:")
    print(f"Before: original={original}, reference={reference}, id(original)={id(original)}")
    original.clear()
    print(f"After:  original={original}, reference={reference}, id(original)={id(original)}")
    # Note that both original and reference are now empty lists with the same identity
    
    # Method 2: Using list = [] - creates new list
    original = [1, 2, 3]
    reference = original
    print(f"\nUsing assignment (list = []):")
    print(f"Before: original={original}, reference={reference}, id(original)={id(original)}")
    original = []  # This creates a new empty list and assigns it to 'original'
    print(f"After:  original={original}, reference={reference}, id(original)={id(original)}")
    # Note that reference still contains [1, 2, 3] because original now points to a different list
    
    # Method 3: Using del list[:] - equivalent to clear()
    original = [1, 2, 3]
    reference = original
    print(f"\nUsing del list[:]:")
    print(f"Before: original={original}, reference={reference}, id(original)={id(original)}")
    del original[:]
    print(f"After:  original={original}, reference={reference}, id(original)={id(original)}")
    
    # Effect on nested lists
    nested = [[1, 2], [3, 4]]
    matrix = [nested, [5, 6]]
    print(f"\nNested example - before: {matrix}")
    nested.clear()
    print(f"After clearing nested: {matrix}")
    # Note that nested list was cleared but still exists in matrix


# =============================================================================
# list.copy() - Return a shallow copy of the list
# =============================================================================

def demonstrate_copy() -> None:
    """
    list.copy() - Return a shallow copy of the list
    
    Time Complexity: O(n) - Linear time (proportional to list size)
    Space Complexity: O(n) - Creates a new list of same size
    
    Key Points:
    - Creates a new list containing the same elements as the original
    - Returns a shallow copy - doesn't copy nested objects, just references to them
    - Equivalent to list[:] but more readable
    - Alternative to copy.copy() for simple list copying
    
    Common Pitfalls:
    - Mistaking shallow copy for deep copy (nested mutable objects are shared)
    - Not considering that modifying mutable elements affects both lists
    - For deep copying (recursive copying of all nested objects), use copy.deepcopy()
    """
    separator("list.copy()")
    
    # Basic usage
    original = [1, 2, 3, 4, 5]
    copied = original.copy()
    
    print(f"Original list: {original}")
    print(f"Copied list: {copied}")
    print(f"Are they the same object? {original is copied}")
    print(f"Original ID: {id(original)}, Copy ID: {id(copied)}")
    
    # Demonstrate independence of lists
    print("\nModifying original list:")
    original.append(6)
    print(f"Original after append: {original}")
    print(f"Copy after original was modified: {copied}")
    
    # Shallow copy behavior with nested objects
    print("\nShallow copy with nested mutable objects:")
    nested_original = [[1, 2], [3, 4], {'a': 1}]
    nested_copied = nested_original.copy()
    
    print(f"Original nested: {nested_original}")
    print(f"Copied nested: {nested_copied}")
    
    # Modifying a nested object affects both lists
    print("\nModifying a nested object in original:")
    nested_original[0].append(99)
    nested_original[2]['b'] = 2
    
    print(f"Original after nested modification: {nested_original}")
    print(f"Copy after original's nested object was modified: {nested_copied}")
    
    # How to perform a deep copy (when needed)
    import copy
    print("\nDeep copy comparison:")
    deep_original = [[1, 2], [3, 4], {'a': 1}]
    deep_copied = copy.deepcopy(deep_original)
    
    deep_original[0].append(99)
    deep_original[2]['b'] = 2
    
    print(f"Deep original after modification: {deep_original}")
    print(f"Deep copy after original's modification: {deep_copied}")
    
    # Performance comparison of copying methods
    large_list = list(range(100000))
    
    time_copy = timeit.timeit(lambda: large_list.copy(), number=100)
    time_slice = timeit.timeit(lambda: large_list[:], number=100)
    time_list = timeit.timeit(lambda: list(large_list), number=100)
    
    print("\nPerformance comparison for 100 operations on list with 100,000 elements:")
    print(f"list.copy(): {time_copy:.6f} seconds")
    print(f"list[:]: {time_slice:.6f} seconds")
    print(f"list(original): {time_list:.6f} seconds")


# =============================================================================
# list.count() - Count occurrences of an element
# =============================================================================

def demonstrate_count() -> None:
    """
    list.count(x) - Return the number of times x appears in the list
    
    Time Complexity: O(n) - Linear time (must check every element)
    Space Complexity: O(1) - Constant space
    
    Key Points:
    - Returns an integer count of how many times the value appears
    - Uses object equality (==) for comparison, not identity (is)
    - Returns 0 if the item is not found
    - Can be used with any object that supports equality comparison
    
    Common Pitfalls:
    - Counting mutable objects might not work as expected if their content changes
    - Using count() in a loop for multiple elements can be inefficient (use Counter instead)
    - Not considering that count() is case-sensitive for strings
    """
    separator("list.count()")
    
    # Basic usage
    numbers = [1, 2, 3, 2, 4, 2, 5]
    count_2 = numbers.count(2)
    print(f"List: {numbers}")
    print(f"Count of 2: {count_2}")
    
    # Counting various data types
    mixed = ['apple', 42, True, 'apple', 3.14, False, True, 'apple']
    print(f"\nMixed list: {mixed}")
    print(f"Count of 'apple': {mixed.count('apple')}")
    print(f"Count of True: {mixed.count(True)}")
    print(f"Count of 99 (not in list): {mixed.count(99)}")
    
    # Case sensitivity with strings
    words = ['Apple', 'apple', 'APPLE', 'Orange']
    print(f"\nWords list: {words}")
    print(f"Count of 'apple': {words.count('apple')}")
    print(f"Count of 'Apple': {words.count('Apple')}")
    
    # Counting mutable objects
    list1 = [1, 2]
    list2 = [1, 2]
    list3 = [3, 4]
    
    container = [list1, list2, list3, [1, 2]]
    print(f"\nContainer of lists: {container}")
    print(f"Count of [1, 2]: {container.count([1, 2])}")
    
    # What happens if we modify a list after adding it?
    list1.append(3)
    print(f"After modifying list1 to {list1}")
    print(f"Count of [1, 2]: {container.count([1, 2])}")
    print(f"Count of [1, 2, 3]: {container.count([1, 2, 3])}")
    
    # More efficient counting of multiple items
    from collections import Counter
    
    data = [1, 2, 3, 1, 2, 1, 4, 5, 1, 6]
    
    # Standard approach (less efficient for multiple counts)
    standard_time = timeit.timeit(lambda: (data.count(1), data.count(2), data.count(3)), number=10000)
    
    # Using Counter (more efficient for multiple counts)
    counter_time = timeit.timeit(lambda: Counter(data), number=10000)
    
    print("\nPerformance comparison for counting multiple elements:")
    print(f"Using list.count() multiple times: {standard_time:.6f} seconds")
    print(f"Using collections.Counter: {counter_time:.6f} seconds")
    
    # Example with Counter
    data_counter = Counter(data)
    print(f"\nCounter object: {data_counter}")
    print(f"Count of 1: {data_counter[1]}")


# =============================================================================
# list.extend() - Add multiple elements from an iterable
# =============================================================================

def demonstrate_extend() -> None:
    """
    list.extend(iterable) - Extend list by appending elements from the iterable
    
    Time Complexity: O(k) - Linear in the size of the iterable being added
    Space Complexity: O(1) - In-place operation (though may trigger reallocation)
    
    Key Points:
    - Adds multiple elements from an iterable to the end of the list
    - Modifies the list in-place and returns None
    - Equivalent to: for item in iterable: list.append(item)
    - More efficient than multiple append() calls
    - Works with any iterable: lists, tuples, sets, strings, etc.
    
    Common Pitfalls:
    - Confusing extend() with append() (extend adds individual elements, append adds one object)
    - Using += with lists (which calls extend()) vs. + (which creates a new list)
    - Forgetting that extend modifies the original list and returns None
    - When extending with strings, each character becomes a separate list element
    """
    separator("list.extend()")
    
    # Basic usage
    numbers = [1, 2, 3]
    print(f"Original list: {numbers}")
    
    numbers.extend([4, 5, 6])
    print(f"After extend with [4, 5, 6]: {numbers}")
    
    # Comparison with append
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    
    list1.extend([4, 5])
    list2.append([4, 5])
    
    print(f"\nWith extend(): {list1}")
    print(f"With append(): {list2}")
    
    # Extending with different iterables
    base = ['a', 'b', 'c']
    print(f"\nBase list: {base}")
    
    # With tuple
    base_copy = base.copy()
    base_copy.extend(('d', 'e'))
    print(f"After extend with tuple ('d', 'e'): {base_copy}")
    
    # With set
    base_copy = base.copy()
    base_copy.extend({'d', 'e'})  # Note: Order not guaranteed with sets
    print(f"After extend with set {{'d', 'e'}}: {base_copy}")
    
    # With string (each character becomes an element)
    base_copy = base.copy()
    base_copy.extend("de")
    print(f"After extend with string 'de': {base_copy}")
    
    # With dictionary (extends with the keys)
    base_copy = base.copy()
    base_copy.extend({'d': 1, 'e': 2})
    print(f"After extend with dict {{'d': 1, 'e': 2}}: {base_copy}")
    
    # Performance: extend vs. multiple appends
    def using_append():
        result = []
        for i in range(10000):
            result.append(i)
        return result
    
    def using_extend():
        result = []
        result.extend(range(10000))
        return result
    
    append_time = timeit.timeit(using_append, number=100)
    extend_time = timeit.timeit(using_extend, number=100)
    
    print("\nPerformance comparison (lower is better):")
    print(f"Multiple append() calls: {append_time:.6f} seconds")
    print(f"Single extend() call: {extend_time:.6f} seconds")
    
    # List += operator uses extend
    original = [1, 2, 3]
    reference = original
    
    print(f"\nUsing += operator (which calls extend):")
    print(f"Before: original={original}, reference={reference}, id={id(original)}")
    original += [4, 5]  # Same as original.extend([4, 5])
    print(f"After: original={original}, reference={reference}, id={id(original)}")
    
    # List + operator creates a new list
    original = [1, 2, 3]
    reference = original
    
    print(f"\nUsing + operator (creates new list):")
    print(f"Before: original={original}, reference={reference}, id={id(original)}")
    original = original + [4, 5]  # Creates a new list
    print(f"After: original={original}, reference={reference}, id={id(original)}")


# =============================================================================
# list.index() - Find the index of an element
# =============================================================================

def demonstrate_index() -> None:
    """
    list.index(x[, start[, end]]) - Return index of first occurrence of x
    
    Time Complexity: O(n) - Linear time (must search elements one by one)
    Space Complexity: O(1) - Constant space
    
    Key Points:
    - Returns the index of the first occurrence of an element
    - Optional start and end parameters limit the search to a slice
    - Raises ValueError if the element is not found
    - Uses equality (==) for comparison, not identity (is)
    
    Common Pitfalls:
    - Not handling ValueError when element might not exist
    - Forgetting that it only returns the FIRST occurrence
    - Inefficient use in loops (better to use enumerate() or dict lookups)
    - Using with mutable objects whose contents might change
    """
    separator("list.index()")
    
    # Basic usage
    fruits = ['apple', 'banana', 'orange', 'apple', 'pear']
    
    print(f"List: {fruits}")
    apple_index = fruits.index('apple')
    print(f"Index of 'apple': {apple_index}")
    
    # Using start and end parameters
    print(f"Index of 'apple' starting from position 1: {fruits.index('apple', 1)}")
    
    # Limiting search to a specific range
    sub_list = fruits[2:4]  # ['orange', 'apple']
    print(f"Sub-list {sub_list}")
    print(f"Index of 'apple' in sub-list: {sub_list.index('apple')}")
    print(f"Index of 'apple' in original list with range [2:4]: {fruits.index('apple', 2, 4)}")
    
    # Handling ValueErrors
    try:
        print(fruits.index('mango'))
    except ValueError:
        print("'mango' not found in list")
    
    # Safe indexing function
    def safe_index(lst, value, default=-1):
        try:
            return lst.index(value)
        except ValueError:
            return default
    
    print(f"Safe index of 'mango': {safe_index(fruits, 'mango')}")
    
    # Performance considerations - index vs. alternatives
    large_list = list(range(10000))
    target = 9500
    
    # Standard index call
    def standard_index():
        return large_list.index(target)
    
    # Using enumerate (often clearer in loops)
    def using_enumerate():
        for i, val in enumerate(large_list):
            if val == target:
                return i
        return -1
    
    # Using a dictionary for O(1) lookups
    def using_dict():
        lookup = {val: idx for idx, val in enumerate(large_list)}
        return lookup.get(target, -1)
    
    standard_time = timeit.timeit(standard_index, number=1000)
    enumerate_time = timeit.timeit(using_enumerate, number=1000)
    dict_time = timeit.timeit(using_dict, number=1000)
    
    print("\nPerformance comparison for 1000 lookups in a list with 10,000 elements:")
    print(f"Using list.index(): {standard_time:.6f} seconds")
    print(f"Using enumerate loop: {enumerate_time:.6f} seconds")
    print(f"Using dictionary lookup: {dict_time:.6f} seconds")
    
    # Finding all occurrences (index only finds the first one)
    def find_all_indices(lst, value):
        return [i for i, x in enumerate(lst) if x == value]
    
    fruits.append('apple')  # Add another apple
    all_apple_indices = find_all_indices(fruits, 'apple')
    print(f"\nList with multiple apples: {fruits}")
    print(f"All indices of 'apple': {all_apple_indices}")


# =============================================================================
# list.insert() - Insert an element at a specific position
# =============================================================================

def demonstrate_insert() -> None:
    """
    list.insert(i, x) - Insert item x at position i
    
    Time Complexity: O(n) - Linear time worst case (all elements after i must shift)
    Space Complexity: O(1) - In-place operation
    
    Key Points:
    - Inserts an element at a specified position
    - All elements at and after position i are shifted right by one
    - If i is negative, it's treated as len(list) + i
    - If i is beyond the list bounds, the item is appended
    - Modifies the list in-place and returns None
    
    Common Pitfalls:
    - Inefficient when inserting many items at the beginning of large lists
    - Forgetting that indices >= i get shifted right
    - Using insert() in a loop (consider building a new list instead)
    - Inserting at a negative index (functions, but can be confusing)
    """
    separator("list.insert()")
    
    # Basic usage
    numbers = [1, 2, 3, 5]
    print(f"Original list: {numbers}")
    
    # Insert at specific position
    numbers.insert(3, 4)
    print(f"After insert(3, 4): {numbers}")
    
    # Insert at beginning
    numbers.insert(0, 0)
    print(f"After insert(0, 0): {numbers}")
    
    # Insert with negative index
    numbers.insert(-2, 3.5)  # Same as insert(len(numbers)-2, 3.5)
    print(f"After insert(-2, 3.5): {numbers}")
    
    # Insert beyond list bounds
    numbers.insert(100, 6)  # Same as append(6)
    print(f"After insert(100, 6): {numbers}")
    
    # Insert with very negative index
    numbers.insert(-100, -1)  # Same as insert(0, -1)
    print(f"After insert(-100, -1): {numbers}")
    
    # Performance comparison: inserting at beginning vs. end
    def insert_beginning(n):
        lst = []
        for i in range(n):
            lst.insert(0, i)
        return lst
    
    def insert_end(n):
        lst = []
        for i in range(n):
            lst.insert(len(lst), i)  # Same as append(i)
        return lst
    
    # More efficient alternatives
    def prepend_then_reverse(n):
        lst = []
        for i in range(n):
            lst.append(i)
        lst.reverse()
        return lst
    
    def use_deque(n):
        from collections import deque
        d = deque()
        for i in range(n):
            d.appendleft(i)
        return list(d)
    
    # Time the operations
    small_n = 1000
    
    time_begin = timeit.timeit(lambda: insert_beginning(small_n), number=5)
    time_end = timeit.timeit(lambda: insert_end(small_n), number=5)
    time_reverse = timeit.timeit(lambda: prepend_then_reverse(small_n), number=5)
    time_deque = timeit.timeit(lambda: use_deque(small_n), number=5)
    
    print("\nPerformance comparison for inserting 1,000 elements:")
    print(f"Inserting at beginning (insert(0, x)): {time_begin:.6f} seconds")
    print(f"Inserting at end (insert(len, x)): {time_end:.6f} seconds")
    print(f"Appending then reversing: {time_reverse:.6f} seconds")
    print(f"Using collections.deque: {time_deque:.6f} seconds")
    
    # Insert elements from an iterable at a specific position
    def insert_multiple(lst, index, items):
        for item in reversed(items):  # Insert in reverse to maintain order
            lst.insert(index, item)
        return lst
    
    numbers = [1, 5]
    insert_multiple(numbers, 1, [2, 3, 4])
    print(f"\nAfter inserting multiple items: {numbers}")


# =============================================================================
# list.pop() - Remove and return an element at a specific position
# =============================================================================

def demonstrate_pop() -> None:
    """
    list.pop([i]) - Remove and return item at index i (default last)
    
    Time Complexity: 
    - O(1) for pop() or pop(-1) (end of list)
    - O(n) for pop(i) where i is not the end (elements must shift left)
    
    Space Complexity: O(1) - In-place operation
    
    Key Points:
    - Removes the item at specified position and returns it
    - Default is the last item if no index specified
    - If i is negative, it's treated as len(list) + i
    - Modifies the list in-place
    - Raises IndexError if list is empty or index out of range
    
    Common Pitfalls:
    - Using pop(0) repeatedly on large lists is inefficient (use collections.deque)
    - Forgetting to handle possible IndexError
    - Not using the returned value when needed
    - Confusing with remove() which removes by value, not index
    """
    separator("list.pop()")
    
    # Basic usage
    fruits = ['apple', 'banana', 'orange', 'pear', 'grape']
    print(f"Original list: {fruits}")
    
    # Pop from end (default)
    last_fruit = fruits.pop()
    print(f"Popped item: {last_fruit}")
    print(f"List after pop(): {fruits}")
    
    # Pop from specific index
    second_fruit = fruits.pop(1)
    print(f"Popped item from index 1: {second_fruit}")
    print(f"List after pop(1): {fruits}")
    
    # Pop with negative index
    fruit_from_end = fruits.pop(-2)  # Second-to-last item
    print(f"Popped item from index -2: {fruit_from_end}")
    print(f"List after pop(-2): {fruits}")
    
    # Handling IndexError
    try:
        fruits.pop(10)  # Index out of range
    except IndexError as e:
        print(f"Error when popping index 10: {e}")
    
    # Empty list
    empty_list = []
    try:
        empty_list.pop()
    except IndexError as e:
        print(f"Error when popping from empty list: {e}")
    
    # Using as a stack (LIFO - Last In, First Out)
    stack = []
    for i in range(5):
        stack.append(i)
    
    print(f"\nStack after pushing 0-4: {stack}")
    while stack:
        print(f"Popped: {stack.pop()}")
    
    # Using as a queue (less efficient, better to use collections.deque)
    queue = []
    for i in range(5):
        queue.append(i)
    
    print(f"\nQueue after enqueuing 0-4: {queue}")
    while queue:
        print(f"Dequeued: {queue.pop(0)}")
    
    # Performance comparison: pop(0) vs. pop() vs. deque
    def pop_beginning(n):
        lst = list(range(n))
        result = []
        while lst:
            result.append(lst.pop(0))
        return result
    
    def pop_end(n):
        lst = list(range(n))
        result = []
        while lst:
            result.append(lst.pop())
        return result
    
    def deque_popleft(n):
        from collections import deque
        d = deque(range(n))
        result = []
        while d:
            result.append(d.popleft())
        return result
    
    medium_n = 1000
    
    time_begin = timeit.timeit(lambda: pop_beginning(medium_n), number=5)
    time_end = timeit.timeit(lambda: pop_end(medium_n), number=5)
    time_deque = timeit.timeit(lambda: deque_popleft(medium_n), number=5)
    
    print("\nPerformance comparison for popping 1,000 elements:")
    print(f"Popping from beginning (pop(0)): {time_begin:.6f} seconds")
    print(f"Popping from end (pop()): {time_end:.6f} seconds")
    print(f"Using collections.deque.popleft(): {time_deque:.6f} seconds")


# =============================================================================
# list.remove() - Remove an element by value
# =============================================================================

def demonstrate_remove() -> None:
    """
    list.remove(x) - Remove first occurrence of value x
    
    Time Complexity: O(n) - Linear time (must search for the value)
    Space Complexity: O(1) - In-place operation
    
    Key Points:
    - Removes the first occurrence of the specified value
    - Modifies the list in-place and returns None
    - Raises ValueError if the value is not found
    - Elements after the removed item shift left by one position
    - Uses equality (==) for comparison, not identity (is)
    
    Common Pitfalls:
    - Only removes the first occurrence, not all occurrences
    - Not handling ValueError when element might not exist
    - Removing elements while iterating (can cause index errors)
    - Confusing with pop() which removes by index
    """
    separator("list.remove()")
    
    # Basic usage
    numbers = [1, 2, 3, 2, 4, 5]
    print(f"Original list: {numbers}")
    
    # Remove first occurrence of a value
    numbers.remove(2)
    print(f"After remove(2): {numbers}")
    
    # Note that only the first occurrence was removed
    print(f"2 still in list? {'Yes' if 2 in numbers else 'No'}")
    
    # Handling ValueError
    try:
        numbers.remove(10)  # Value not in list
    except ValueError as e:
        print(f"Error when removing 10: {e}")
    
    # Safe removal function
    def safe_remove(lst, value):
        try:
            lst.remove(value)
            return True
        except ValueError:
            return False
    
    result = safe_remove(numbers, 10)
    print(f"Safe remove of 10 succeeded? {result}")
    
    # Removing all occurrences of a value
    repeated = [1, 2, 3, 1, 2, 1, 4, 1]
    print(f"\nList with repeated values: {repeated}")
    
    # INCORRECT approach - Don't do this!
    def remove_all_incorrect(lst, value):
        for i in range(len(lst)):
            if lst[i] == value:
                lst.remove(value)
        return lst
    
    # Correct approaches
    def remove_all_while(lst, value):
        while value in lst:
            lst.remove(value)
        return lst
    
    def remove_all_filter(lst, value):
        return [x for x in lst if x != value]
    
    test1 = repeated.copy()
    remove_all_while(test1, 1)
    print(f"After removing all 1s (while loop): {test1}")
    
    test2 = repeated.copy()
    test2 = remove_all_filter(test2, 1)
    print(f"After removing all 1s (list comprehension): {test2}")
    
    # Removing while iterating (common mistake)
    print("\nRemoving during iteration - common mistake:")
    values = [1, 2, 3, 4, 5, 6]
    
    # INCORRECT - this causes items to be skipped
    print(f"Starting with: {values}")
    for value in values[:]:  # Create a copy for illustration
        if value % 2 == 0:  # Remove even numbers
            print(f"Removing {value}")
            values.remove(value)
    print(f"Result with incorrect approach: {values}")
    
    # CORRECT - iterate over a copy
    values = [1, 2, 3, 4, 5, 6]
    for value in values[:]:  # Create a copy for safe iteration
        if value % 2 == 0:
            values.remove(value)
    print(f"Result with correct approach: {values}")
    
    # Performance: remove vs filter for removing multiple items
    data = [i % 10 for i in range(10000)]  # List with repeated values
    
    def using_remove(lst, value):
        result = lst.copy()
        while value in result:
            result.remove(value)
        return result
    
    def using_filter(lst, value):
        return [x for x in lst if x != value]
    
    remove_time = timeit.timeit(lambda: using_remove(data, 5), number=10)
    filter_time = timeit.timeit(lambda: using_filter(data, 5), number=10)
    
    print("\nPerformance comparison for removing all occurrences of a value:")
    print(f"Using list.remove() in a loop: {remove_time:.6f} seconds")
    print(f"Using list comprehension: {filter_time:.6f} seconds")


# =============================================================================
# list.reverse() - Reverse the elements in place
# =============================================================================

def demonstrate_reverse() -> None:
    """
    list.reverse() - Reverse elements of the list in place
    
    Time Complexity: O(n) - Linear time (must process each element)
    Space Complexity: O(1) - In-place operation
    
    Key Points:
    - Reverses the order of elements in the list
    - Modifies the list in-place and returns None
    - Equivalent to lst[:] = lst[::-1] but more efficient
    - Simple and efficient way to reverse a list
    
    Common Pitfalls:
    - Confusing with reversed() built-in which returns an iterator, not a list
    - Forgetting that reverse() returns None (modifies in-place)
    - Not creating a copy when the original order is still needed
    - Calling reverse() multiple times when not needed (two reverses cancel out)
    """
    separator("list.reverse()")
    
    # Basic usage
    numbers = [1, 2, 3, 4, 5]
    print(f"Original list: {numbers}")
    
    numbers.reverse()
    print(f"After reverse(): {numbers}")
    
    # Return value is None
    result = numbers.reverse()  # Reverse back to original
    print(f"Return value: {result}")
    print(f"List after second reverse(): {numbers}")
    
    # Comparison with other methods
    original = [1, 2, 3, 4, 5]
    print(f"\nStarting with: {original}")
    
    # Method 1: list.reverse()
    method1 = original.copy()
    method1.reverse()
    print(f"After list.reverse(): {method1}")
    
    # Method 2: Using slicing
    method2 = original[::-1]
    print(f"Using slicing [::-1]: {method2}")
    
    # Method 3: Using reversed() built-in
    method3 = list(reversed(original))
    print(f"Using list(reversed()): {method3}")
    
    # Reversing other sequence types
    text = "Python"
    print(f"\nReversing string '{text}':")
    print(f"Using slicing: '{text[::-1]}'")
    print(f"Using reversed(): '{''.join(reversed(text))}'")
    
    # Performance comparison
    large_list = list(range(100000))
    
    def using_reverse():
        lst = large_list.copy()
        lst.reverse()
        return lst
    
    def using_slicing():
        return large_list[::-1]
    
    def using_reversed():
        return list(reversed(large_list))
    
    reverse_time = timeit.timeit(using_reverse, number=100)
    slicing_time = timeit.timeit(using_slicing, number=100)
    reversed_time = timeit.timeit(using_reversed, number=100)
    
    print("\nPerformance comparison for reversing a list with 100,000 elements:")
    print(f"list.reverse(): {reverse_time:.6f} seconds")
    print(f"list[::-1]: {slicing_time:.6f} seconds")
    print(f"list(reversed(list)): {reversed_time:.6f} seconds")
    
    # Memory usage considerations
    print("\nMemory usage considerations:")
    original_id = id(original)
    
    original.reverse()
    print(f"After list.reverse(), same object? {id(original) == original_id}")
    
    sliced = original[::-1]
    print(f"After list[::-1], same object? {id(sliced) == original_id}")
    
    # Practical applications
    # 1. Checking for palindromes
    def is_palindrome(s):
        # Convert to lowercase and remove non-alphanumeric characters
        s = ''.join(c.lower() for c in s if c.isalnum())
        return s == s[::-1]
    
    test_strings = ["radar", "hello", "A man, a plan, a canal: Panama"]
    for s in test_strings:
        print(f"'{s}' is palindrome? {is_palindrome(s)}")
    
    # 2. Reversing words in a sentence
    sentence = "Python is awesome"
    words = sentence.split()
    words.reverse()
    print(f"\nReversed words: '{' '.join(words)}'")


# =============================================================================
# list.sort() - Sort the list in place
# =============================================================================

def demonstrate_sort() -> None:
    """
    list.sort(*, key=None, reverse=False) - Sort the list in place
    
    Time Complexity: O(n log n) - Timsort algorithm
    Space Complexity: O(n) - Requires temporary storage during sorting
    
    Key Points:
    - Sorts the list in-place and returns None
    - Default is ascending order
    - Optional 'key' parameter specifies a function to extract comparison key
    - Optional 'reverse' parameter sorts in descending order when True
    - Uses Timsort algorithm (hybrid of merge sort and insertion sort)
    - Stable sort (preserves relative order of equal elements)
    
    Common Pitfalls:
    - Confusing with sorted() built-in which returns a new list
    - Forgetting that sort() returns None (modifies in-place)
    - Not creating a copy when the original order is still needed
    - Trying to sort mixed data types without a proper key function
    - Performance impact when using complex key functions
    """
    separator("list.sort()")
    
    # Basic usage
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    print(f"Original list: {numbers}")
    
    numbers.sort()
    print(f"After sort(): {numbers}")
    
    # Descending order
    numbers.sort(reverse=True)
    print(f"After sort(reverse=True): {numbers}")
    
    # Return value is None
    result = numbers.sort()
    print(f"Return value: {result}")
    
    # Sorting strings (case-sensitive)
    words = ['banana', 'Apple', 'cherry', 'Date']
    print(f"\nOriginal strings: {words}")
    
    words.sort()
    print(f"After sort() (case-sensitive): {words}")
    
    # Case-insensitive sorting using key function
    words = ['banana', 'Apple', 'cherry', 'Date']
    words.sort(key=str.lower)
    print(f"After sort(key=str.lower): {words}")
    
    # Sorting complex objects
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
        Person("David", 25)
    ]
    
    print(f"\nOriginal people: {people}")
    
    # Sort by age
    people.sort(key=lambda person: person.age)
    print(f"After sort by age: {people}")
    
    # Sort by name
    people.sort(key=lambda person: person.name)
    print(f"After sort by name: {people}")
    
    # Multiple sort criteria (sort by age, then by name)
    people.sort(key=lambda person: (person.age, person.name))
    print(f"After sort by age, then name: {people}")
    
    # Sorting mixed fields (age ascending, name descending)
    people.sort(key=lambda person: (person.age, -ord(person.name[0])))
    print(f"After sort by age (asc), then name initial (desc): {people}")
    
    # Comparison with sorted()
    original = [5, 2, 3, 1, 4]
    print(f"\nOriginal: {original}")
    
    # Using list.sort()
    sorted_in_place = original.copy()
    sorted_in_place.sort()
    print(f"After list.sort(): {sorted_in_place}")
    
    # Using sorted() built-in
    sorted_new = sorted(original)
    print(f"Using sorted(): {sorted_new}")
    print(f"Original after sorted(): {original}")
    
    # Performance comparison for different sorting approaches
    data = [i for i in range(10000)]
    import random
    random.shuffle(data)
    
    # Regular sort
    def regular_sort():
        lst = data.copy()
        lst.sort()
        return lst
    
    # Sort with simple key function
    def key_sort_simple():
        lst = data.copy()
        lst.sort(key=lambda x: x)
        return lst
    
    # Sort with complex key function
    def key_sort_complex():
        lst = data.copy()
        lst.sort(key=lambda x: (x % 10, x // 10))
        return lst
    
    # Using sorted()
    def using_sorted():
        return sorted(data)
    
    regular_time = timeit.timeit(regular_sort, number=10)
    key_simple_time = timeit.timeit(key_sort_simple, number=10)
    key_complex_time = timeit.timeit(key_sort_complex, number=10)
    sorted_time = timeit.timeit(using_sorted, number=10)
    
    print("\nPerformance comparison for sorting 10,000 elements:")
    print(f"list.sort(): {regular_time:.6f} seconds")
    print(f"list.sort(key=lambda x: x): {key_simple_time:.6f} seconds")
    print(f"list.sort() with complex key: {key_complex_time:.6f} seconds")
    print(f"sorted(): {sorted_time:.6f} seconds")
    
    # Sort stability demonstration
    data = [(1, 'B'), (2, 'A'), (1, 'A'), (2, 'B')]
    print(f"\nDemonstrating sort stability with {data}")
    
    # Sort by first element
    data_copy = data.copy()
    data_copy.sort(key=lambda x: x[0])
    print(f"After sorting by first element: {data_copy}")
    # Notice how the relative order of (1, 'B')/(1, 'A') and (2, 'A')/(2, 'B') is preserved


# =============================================================================
# Comprehensive list method examples and demos
# =============================================================================

def run_all_demos() -> None:
    """Execute all list method demonstrations."""
    demonstrate_append()
    demonstrate_clear()
    demonstrate_copy()
    demonstrate_count()
    demonstrate_extend()
    demonstrate_index()
    demonstrate_insert()
    demonstrate_pop()
    demonstrate_remove()
    demonstrate_reverse()
    demonstrate_sort()


if __name__ == "__main__":
    # Set larger recursion limit for deep examples
    sys.setrecursionlimit(3000)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║ Python List Methods - Comprehensive Guide                                 ║
║ --------------------------------------------------------------           ║
║ Running complete demonstrations of all Python list methods.               ║
║ Each method includes detailed explanations, examples, and                 ║
║ performance considerations.                                               ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    run_all_demos()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║ Summary of List Method Time & Space Complexity                           ║
║ --------------------------------------------------------------           ║
║ list.append(x)    | O(1) time - amortized  | O(1) space                  ║
║ list.clear()      | O(n) time              | O(1) space                  ║
║ list.copy()       | O(n) time              | O(n) space                  ║
║ list.count(x)     | O(n) time              | O(1) space                  ║
║ list.extend(iter) | O(k) time (k=len(iter))| O(1) space                  ║
║ list.index(x)     | O(n) time              | O(1) space                  ║
║ list.insert(i, x) | O(n) time worst case   | O(1) space                  ║
║ list.pop([i])     | O(1) time for end      | O(1) space                  ║
║                   | O(n) time for i != end |                             ║
║ list.remove(x)    | O(n) time              | O(1) space                  ║
║ list.reverse()    | O(n) time              | O(1) space                  ║
║ list.sort()       | O(n log n) time        | O(n) space                  ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)