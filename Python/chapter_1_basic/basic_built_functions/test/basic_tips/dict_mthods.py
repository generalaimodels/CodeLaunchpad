#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Built-in List Methods - Comprehensive Guide
================================================

This module provides a detailed overview of all standard list methods in Python,
with examples, time complexity analysis, edge cases, and exception handling.
"""


# =============================================================================
# 1. list.append(x) - O(1)
# =============================================================================
def demonstrate_append():
    """
    list.append(x):
    - Adds element x to the end of the list
    - Time Complexity: O(1) amortized - constant time operation
    - In-place operation: returns None, modifies the original list
    - Works with any data type
    """
    # Basic usage
    fruits = ["apple", "banana"]
    fruits.append("cherry")
    print(f"After append: {fruits}")  # ['apple', 'banana', 'cherry']
    
    # Appending different data types
    mixed_list = [1, "two"]
    mixed_list.append(3.0)
    mixed_list.append(True)
    mixed_list.append([4, 5])  # Note: This adds the entire list as a single element
    print(f"Appending different types: {mixed_list}")  # [1, 'two', 3.0, True, [4, 5]]
    
    # Common mistake: trying to append multiple items
    numbers = [1, 2, 3]
    # This adds [4, 5] as a single element, not individual elements
    numbers.append([4, 5])
    print(f"Appending a list: {numbers}")  # [1, 2, 3, [4, 5]]
    
    # Return value is None (in-place operation)
    result = numbers.append(6)
    print(f"Return value of append: {result}")  # None
    print(f"Updated list: {numbers}")  # [1, 2, 3, [4, 5], 6]


# =============================================================================
# 2. list.clear() - O(1)
# =============================================================================
def demonstrate_clear():
    """
    list.clear():
    - Removes all items from the list
    - Time Complexity: O(1) - constant time operation
    - In-place operation: returns None, modifies the original list
    - Equivalent to del list[:]
    """
    # Basic usage
    numbers = [1, 2, 3, 4, 5]
    numbers.clear()
    print(f"After clear: {numbers}")  # []
    
    # Clear on an empty list (no error)
    empty_list = []
    empty_list.clear()
    print(f"Clear empty list: {empty_list}")  # []
    
    # Effect on references
    original = [1, 2, 3]
    reference = original  # Both point to the same list
    original.clear()
    print(f"Original after clear: {original}")  # []
    print(f"Reference after clear: {reference}")  # [] (affected because it's the same object)
    
    # Alternative ways to clear a list
    alt_list = [1, 2, 3, 4]
    del alt_list[:]  # Equivalent to clear()
    print(f"After del[:]: {alt_list}")  # []
    
    # Note: list = [] creates a new empty list rather than clearing the existing one
    another_list = [1, 2, 3]
    ref_to_another = another_list
    another_list = []  # Creates a new empty list, doesn't affect the original
    print(f"another_list: {another_list}")  # []
    print(f"ref_to_another: {ref_to_another}")  # [1, 2, 3] (not affected)


# =============================================================================
# 3. list.copy() - O(n)
# =============================================================================
def demonstrate_copy():
    """
    list.copy():
    - Returns a shallow copy of the list
    - Time Complexity: O(n) where n is the length of the list
    - Does not modify the original list
    - Equivalent to list[:] and list(original_list)
    - Creates a new list object but does not copy nested objects
    """
    # Basic usage
    original = [1, 2, 3]
    copied = original.copy()
    print(f"Original: {original}, Copy: {copied}")  # [1, 2, 3], [1, 2, 3]
    
    # Proving they are different objects
    original.append(4)
    print(f"Original after append: {original}")  # [1, 2, 3, 4]
    print(f"Copy after original's append: {copied}")  # [1, 2, 3] (unaffected)
    
    # Shallow copy - nested objects are shared
    nested = [[1, 2], [3, 4]]
    nested_copy = nested.copy()
    
    # Modifying a nested list affects both original and copy
    nested[0].append(5)
    print(f"Original after nested modification: {nested}")  # [[1, 2, 5], [3, 4]]
    print(f"Copy after original's nested modification: {nested_copy}")  # [[1, 2, 5], [3, 4]]
    
    # But adding a new list doesn't affect the copy
    nested.append([6, 7])
    print(f"Original after append: {nested}")  # [[1, 2, 5], [3, 4], [6, 7]]
    print(f"Copy after original's append: {nested_copy}")  # [[1, 2, 5], [3, 4]] (unaffected)
    
    # Alternative ways to copy a list
    slice_copy = original[:]
    constructor_copy = list(original)
    print(f"Slice copy: {slice_copy}")  # [1, 2, 3, 4]
    print(f"Constructor copy: {constructor_copy}")  # [1, 2, 3, 4]
    
    # For deep copy (copying nested structures too)
    import copy
    nested2 = [[1, 2], [3, 4]]
    deep_copy = copy.deepcopy(nested2)
    nested2[0].append(5)
    print(f"Original after deep copy and modification: {nested2}")  # [[1, 2, 5], [3, 4]]
    print(f"Deep copy after original's modification: {deep_copy}")  # [[1, 2], [3, 4]] (unaffected)


# =============================================================================
# 4. list.count(x) - O(n)
# =============================================================================
def demonstrate_count():
    """
    list.count(x):
    - Returns the number of times element x appears in the list
    - Time Complexity: O(n) where n is the length of the list
    - Does not modify the original list
    - Returns 0 if the element is not found
    - Uses the == operator for comparison
    """
    # Basic usage
    numbers = [1, 2, 2, 3, 2, 4, 5, 2]
    count_2 = numbers.count(2)
    print(f"Count of 2: {count_2}")  # 4
    
    # Counting items that don't exist
    count_9 = numbers.count(9)
    print(f"Count of 9: {count_9}")  # 0
    
    # Counting with different data types
    mixed = [1, "hello", True, 1.0, 1, "hello"]
    print(f"Count of 'hello': {mixed.count('hello')}")  # 2
    
    # Note: In Python, 1 == True is True
    print(f"Count of 1: {mixed.count(1)}")  # 3 (counts both 1 and True)
    print(f"Count of True: {mixed.count(True)}")  # 3 (same reason)
    print(f"Count of 1.0: {mixed.count(1.0)}")  # 3 (1 == 1.0 is True)
    
    # Counting with custom objects
    class Person:
        def __init__(self, name):
            self.name = name
        
        def __eq__(self, other):
            # Two Person objects are equal if they have the same name
            if isinstance(other, Person):
                return self.name == other.name
            return False
    
    people = [Person("Alice"), Person("Bob"), Person("Alice")]
    alice_count = people.count(Person("Alice"))
    print(f"Count of Person('Alice'): {alice_count}")  # 2
    
    # Counting None values
    with_none = [1, None, 2, None, 3]
    print(f"Count of None: {with_none.count(None)}")  # 2


# =============================================================================
# 5. list.extend(iterable) - O(k)
# =============================================================================
def demonstrate_extend():
    """
    list.extend(iterable):
    - Extends the list by appending elements from the iterable
    - Time Complexity: O(k) where k is the length of the iterable
    - In-place operation: returns None, modifies the original list
    - Works with any iterable (lists, tuples, sets, strings, etc.)
    - Equivalent to: list[len(list):] = iterable
    """
    # Basic usage
    numbers = [1, 2, 3]
    numbers.extend([4, 5, 6])
    print(f"After extend with list: {numbers}")  # [1, 2, 3, 4, 5, 6]
    
    # Extend with different iterables
    numbers.extend((7, 8))  # Tuple
    print(f"After extend with tuple: {numbers}")  # [1, 2, 3, 4, 5, 6, 7, 8]
    
    numbers.extend({9, 10})  # Set (order not guaranteed)
    print(f"After extend with set: {numbers}")  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (order may vary)
    
    # Extend with a string (adds each character)
    chars = ['A', 'B']
    chars.extend("CD")
    print(f"After extend with string: {chars}")  # ['A', 'B', 'C', 'D']
    
    # Difference between extend and append
    list1 = [1, 2, 3]
    list1.append([4, 5])
    print(f"After append: {list1}")  # [1, 2, 3, [4, 5]]
    
    list2 = [1, 2, 3]
    list2.extend([4, 5])
    print(f"After extend: {list2}")  # [1, 2, 3, 4, 5]
    
    # Return value is None (in-place operation)
    result = list2.extend([6])
    print(f"Return value of extend: {result}")  # None
    print(f"Updated list: {list2}")  # [1, 2, 3, 4, 5, 6]
    
    # Empty iterable doesn't change the list
    list2.extend([])
    print(f"After extending with empty list: {list2}")  # [1, 2, 3, 4, 5, 6]
    
    # Alternative syntax
    list3 = [1, 2, 3]
    list3[len(list3):] = [4, 5]  # Equivalent to extend
    print(f"Using slice assignment: {list3}")  # [1, 2, 3, 4, 5]
    
    # Extending with a dictionary (adds only keys)
    list4 = [1, 2]
    list4.extend({"a": 1, "b": 2})
    print(f"After extending with dict: {list4}")  # [1, 2, 'a', 'b']
    
    # Caution with generators - they get consumed
    def gen_func():
        yield 10
        yield 20
    
    generator = gen_func()
    list5 = [1, 2]
    list5.extend(generator)
    print(f"After extending with generator: {list5}")  # [1, 2, 10, 20]
    
    # The generator is now exhausted
    list6 = [3, 4]
    list6.extend(generator)  # Generator is already consumed
    print(f"After extending with consumed generator: {list6}")  # [3, 4]


# =============================================================================
# 6. list.index(x[, start[, end]]) - O(n)
# =============================================================================
def demonstrate_index():
    """
    list.index(x[, start[, end]]):
    - Returns the first index of element x in the list
    - Optional start and end parameters to limit search to a slice
    - Time Complexity: O(n) where n is the length of the list (or slice)
    - Does not modify the original list
    - Raises ValueError if the element is not found
    - Uses the == operator for comparison
    """
    # Basic usage
    fruits = ["apple", "banana", "cherry", "banana", "date"]
    banana_index = fruits.index("banana")
    print(f"Index of 'banana': {banana_index}")  # 1 (first occurrence)
    
    # Using optional start parameter
    second_banana = fruits.index("banana", 2)  # Start search from index 2
    print(f"Index of 'banana' starting from index 2: {second_banana}")  # 3
    
    # Using both start and end parameters
    letters = ["a", "b", "c", "d", "e", "f", "g"]
    print(f"Index of 'e' from 2 to 6: {letters.index('e', 2, 6)}")  # 4
    
    # Error handling: element not found
    try:
        letters.index("z")
    except ValueError as e:
        print(f"Error when searching for 'z': {e}")  # 'z' is not in list
    
    # Error handling: element exists but not in the specified range
    try:
        letters.index("a", 1)  # 'a' exists at index 0, but we're starting from 1
    except ValueError as e:
        print(f"Error when searching for 'a' after index 1: {e}")  # 'a' is not in list
    
    # Works with any comparable type
    mixed = [1, 2.0, "three", True, [5]]
    print(f"Index of 'three': {mixed.index('three')}")  # 2
    print(f"Index of True: {mixed.index(True)}")  # 3
    
    # Note: In Python, 1 == True is True, so may give unexpected results
    numbers_with_true = [0, 1, 2, True]
    print(f"Index of 1: {numbers_with_true.index(1)}")  # 1 (not 3)
    print(f"Index of True: {numbers_with_true.index(True)}")  # 1 (not 3)
    
    # Searching for mutable objects
    list_with_lists = [[1], [2], [3]]
    search_list = [2]
    print(f"Index of [2]: {list_with_lists.index(search_list)}")  # 1
    
    # Using index with custom objects
    class Book:
        def __init__(self, title):
            self.title = title
            
        def __eq__(self, other):
            if isinstance(other, Book):
                return self.title == other.title
            return False
    
    library = [Book("Moby Dick"), Book("Pride and Prejudice"), Book("1984")]
    print(f"Index of Book('1984'): {library.index(Book('1984'))}")  # 2


# =============================================================================
# 7. list.insert(i, x) - O(n)
# =============================================================================
def demonstrate_insert():
    """
    list.insert(i, x):
    - Inserts element x at position i in the list
    - Time Complexity: O(n) where n is the length of the list (worst case)
    - In-place operation: returns None, modifies the original list
    - If i is negative, treated as len(list) + i
    - If i is beyond list bounds, equivalent to append
    """
    # Basic usage
    fruits = ["apple", "banana", "cherry"]
    fruits.insert(1, "blueberry")
    print(f"After insert at index 1: {fruits}")  # ['apple', 'blueberry', 'banana', 'cherry']
    
    # Insert at beginning (index 0)
    fruits.insert(0, "avocado")
    print(f"After insert at beginning: {fruits}")  # ['avocado', 'apple', 'blueberry', 'banana', 'cherry']
    
    # Insert at end (index = length of list)
    fruits.insert(len(fruits), "elderberry")
    print(f"After insert at end: {fruits}")  # [..., 'cherry', 'elderberry']
    
    # Insert after end (index > length) - same as append
    fruits.insert(100, "fig")  # Equivalent to append
    print(f"After insert beyond end: {fruits}")  # [..., 'elderberry', 'fig']
    
    # Insert with negative index
    numbers = [1, 2, 3, 4]
    numbers.insert(-1, 3.5)  # Insert before the last element
    print(f"After insert at index -1: {numbers}")  # [1, 2, 3, 3.5, 4]
    
    numbers.insert(-2, 2.5)  # Insert before the second-to-last element
    print(f"After insert at index -2: {numbers}")  # [1, 2, 3, 2.5, 3.5, 4]
    
    # Return value is None (in-place operation)
    result = numbers.insert(0, 0)
    print(f"Return value of insert: {result}")  # None
    print(f"Updated list: {numbers}")  # [0, 1, 2, 3, 2.5, 3.5, 4]
    
    # Inserting different data types
    mixed = [1, "two"]
    mixed.insert(1, [1.5, 1.75])  # Insert a list as a single element
    print(f"After inserting a list: {mixed}")  # [1, [1.5, 1.75], 'two']
    
    # Performance consideration
    # For large lists, insert near the beginning is slower than near the end
    # because all subsequent elements must be shifted


# =============================================================================
# 8. list.pop([i]) - O(n)
# =============================================================================
def demonstrate_pop():
    """
    list.pop([i]):
    - Removes and returns the item at position i
    - If i is not specified, removes and returns the last item
    - Time Complexity: O(n) where n is the length of the list (worst case for first element)
    - Time Complexity: O(1) for popping the last element (i = -1 or default)
    - In-place operation: modifies the original list
    - Raises IndexError if list is empty or index is out of range
    """
    # Basic usage - pop last element (default)
    fruits = ["apple", "banana", "cherry", "date"]
    last_fruit = fruits.pop()
    print(f"Popped last element: {last_fruit}")  # date
    print(f"List after pop: {fruits}")  # ['apple', 'banana', 'cherry']
    
    # Pop with specified index
    second_fruit = fruits.pop(1)
    print(f"Popped element at index 1: {second_fruit}")  # banana
    print(f"List after pop(1): {fruits}")  # ['apple', 'cherry']
    
    # Pop with negative index
    numbers = [1, 2, 3, 4, 5]
    second_last = numbers.pop(-2)  # Same as numbers.pop(len(numbers) - 2)
    print(f"Popped second-to-last element: {second_last}")  # 4
    print(f"List after pop(-2): {numbers}")  # [1, 2, 3, 5]
    
    # Error handling: pop from empty list
    empty_list = []
    try:
        empty_list.pop()
    except IndexError as e:
        print(f"Error when popping from empty list: {e}")  # pop from empty list
    
    # Error handling: index out of range
    short_list = [1, 2, 3]
    try:
        short_list.pop(10)
    except IndexError as e:
        print(f"Error when popping index 10: {e}")  # pop index out of range
    
    # Using pop to implement a stack (LIFO - Last In, First Out)
    stack = []
    stack.append("a")
    stack.append("b")
    stack.append("c")
    print(f"Stack: {stack}")  # ['a', 'b', 'c']
    
    while stack:
        print(f"Popped from stack: {stack.pop()}")
    # Outputs: 'c', 'b', 'a'
    
    # Using pop to implement a queue (FIFO - First In, First Out)
    queue = []
    queue.append("a")
    queue.append("b")
    queue.append("c")
    print(f"Queue: {queue}")  # ['a', 'b', 'c']
    
    # Not efficient for large queues (O(n) time complexity for each pop(0))
    while queue:
        print(f"Dequeued: {queue.pop(0)}")
    # Outputs: 'a', 'b', 'c'
    
    # Note: For efficient queue, use collections.deque
    from collections import deque
    efficient_queue = deque(["a", "b", "c"])
    print(f"Efficient queue: {efficient_queue}")
    print(f"Dequeued efficiently: {efficient_queue.popleft()}")  # 'a' in O(1) time


# =============================================================================
# 9. list.remove(x) - O(n)
# =============================================================================
def demonstrate_remove():
    """
    list.remove(x):
    - Removes the first occurrence of element x from the list
    - Time Complexity: O(n) where n is the length of the list
    - In-place operation: returns None, modifies the original list
    - Raises ValueError if the element is not found
    - Uses the == operator for comparison
    """
    # Basic usage
    fruits = ["apple", "banana", "cherry", "banana", "date"]
    fruits.remove("banana")  # Removes first occurrence only
    print(f"After removing 'banana': {fruits}")  # ['apple', 'cherry', 'banana', 'date']
    
    # Removing again
    fruits.remove("banana")  # Removes the second original banana
    print(f"After removing 'banana' again: {fruits}")  # ['apple', 'cherry', 'date']
    
    # Error handling: element not found
    try:
        fruits.remove("banana")  # No more bananas to remove
    except ValueError as e:
        print(f"Error when removing 'banana' again: {e}")  # list.remove(x): x not in list
    
    # Return value is None (in-place operation)
    numbers = [1, 2, 3]
    result = numbers.remove(2)
    print(f"Return value of remove: {result}")  # None
    print(f"List after remove: {numbers}")  # [1, 3]
    
    # Caution with == comparisons (like with index)
    mixed = [0, 1, 2, True, False]
    mixed.remove(True)  # 1 == True is True in Python
    print(f"After removing True: {mixed}")  # [0, 2, False] (1 is removed)
    
    mixed = [0, 1, 2, True, False]
    mixed.remove(1)  # Also removes True since 1 == True
    print(f"After removing 1: {mixed}")  # [0, 2, False] (True is removed)
    
    # Removing objects by value vs identity
    list_a = [[1, 2], [3, 4]]
    list_b = [[1, 2], [3, 4]]  # Same values but different objects
    
    # This works because [1, 2] == [1, 2] is True
    list_a.remove([1, 2])
    print(f"After removing [1, 2]: {list_a}")  # [[3, 4]]
    
    # With custom objects
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
            
        def __eq__(self, other):
            if isinstance(other, Person):
                return self.name == other.name  # Compare only by name
            return False
            
        def __repr__(self):
            return f"Person('{self.name}', {self.age})"
    
    people = [Person("Alice", 25), Person("Bob", 30), Person("Alice", 40)]
    
    # Removes first "Alice" even though the age is different
    people.remove(Person("Alice", 999))  # Age is ignored in __eq__
    print(f"After removing Alice: {people}")  # [Person('Bob', 30), Person('Alice', 40)]


# =============================================================================
# 10. list.reverse() - O(n)
# =============================================================================
def demonstrate_reverse():
    """
    list.reverse():
    - Reverses the elements of the list in place
    - Time Complexity: O(n) where n is the length of the list
    - In-place operation: returns None, modifies the original list
    - Alternative ways to get reversed lists without modifying original:
      * list[::-1] - creates a reversed copy
      * reversed(list) - returns an iterator
    """
    # Basic usage
    numbers = [1, 2, 3, 4, 5]
    numbers.reverse()
    print(f"After reverse: {numbers}")  # [5, 4, 3, 2, 1]
    
    # Empty list
    empty = []
    empty.reverse()
    print(f"Reversing empty list: {empty}")  # [] (no effect)
    
    # Single element
    single = [42]
    single.reverse()
    print(f"Reversing single element list: {single}")  # [42] (no visible effect)
    
    # Return value is None (in-place operation)
    result = numbers.reverse()
    print(f"Return value of reverse: {result}")  # None
    print(f"List after reverse again: {numbers}")  # [1, 2, 3, 4, 5] (back to original)
    
    # Alternative 1: Slicing to create a reversed copy
    original = [1, 2, 3, 4]
    reversed_copy = original[::-1]  # Creates a new list
    print(f"Original: {original}")  # [1, 2, 3, 4] (unchanged)
    print(f"Reversed copy: {reversed_copy}")  # [4, 3, 2, 1]
    
    # Alternative 2: reversed() function
    reversed_iterator = reversed(original)
    print(f"Type of reversed(): {type(reversed_iterator)}")  # <class 'list_reverseiterator'>
    reversed_list = list(reversed_iterator)
    print(f"List from reversed(): {reversed_list}")  # [4, 3, 2, 1]
    
    # Reversing a list of mutable objects (only order changes, not the objects)
    nested = [[1, 2], [3, 4], [5, 6]]
    nested.reverse()
    print(f"Reversed nested list: {nested}")  # [[5, 6], [3, 4], [1, 2]]
    
    # Modifying a nested element
    nested[0].append(7)
    print(f"After modifying nested element: {nested}")  # [[5, 6, 7], [3, 4], [1, 2]]
    
    # Practical examples
    # 1. Reversing a string by converting to list and back
    s = "hello"
    char_list = list(s)
    char_list.reverse()
    reversed_s = ''.join(char_list)
    print(f"Reversed string: {reversed_s}")  # 'olleh'
    
    # 2. Checking if a word is a palindrome
    def is_palindrome(word):
        letters = list(word.lower())
        reversed_letters = letters.copy()
        reversed_letters.reverse()
        return letters == reversed_letters
    
    print(f"'radar' is palindrome: {is_palindrome('radar')}")  # True
    print(f"'hello' is palindrome: {is_palindrome('hello')}")  # False


# =============================================================================
# 11. list.sort(*, key=None, reverse=False) - O(n log n)
# =============================================================================
def demonstrate_sort():
    """
    list.sort(*, key=None, reverse=False):
    - Sorts the list in place (uses TimSort algorithm)
    - Time Complexity: O(n log n) where n is the length of the list
    - In-place operation: returns None, modifies the original list
    - Parameters:
      * key: a function that takes an element and returns a key for sorting
      * reverse: if True, sort in descending order
    - For a new sorted list without changing original, use sorted(list)
    - Elements must be comparable to each other
    """
    # Basic usage
    numbers = [3, 1, 4, 1, 5, 9, 2]
    numbers.sort()
    print(f"After sort: {numbers}")  # [1, 1, 2, 3, 4, 5, 9]
    
    # Reverse sort
    numbers.sort(reverse=True)
    print(f"After reverse sort: {numbers}")  # [9, 5, 4, 3, 2, 1, 1]
    
    # Empty list
    empty = []
    empty.sort()
    print(f"Sorting empty list: {empty}")  # [] (no effect)
    
    # Return value is None (in-place operation)
    result = numbers.sort()
    print(f"Return value of sort: {result}")  # None
    print(f"List after sort again: {numbers}")  # [1, 1, 2, 3, 4, 5, 9]
    
    # Sorting strings (lexicographic order)
    words = ["banana", "apple", "Cherry", "date"]
    words.sort()
    print(f"Sorted words: {words}")  # ['Cherry', 'apple', 'banana', 'date'] (capital letters come first)
    
    # Case-insensitive sort using key function
    words.sort(key=str.lower)
    print(f"Case-insensitive sort: {words}")  # ['apple', 'banana', 'Cherry', 'date']
    
    # Sorting based on length
    words.sort(key=len)
    print(f"Sorted by length: {words}")  # ['date', 'apple', 'Cherry', 'banana']
    
    # Complex example: sort by length, then alphabetically
    words = ["cat", "dog", "ant", "elephant", "bee", "zebra"]
    
    def length_then_alpha(s):
        return (len(s), s.lower())  # Return a tuple to sort by multiple criteria
    
    words.sort(key=length_then_alpha)
    print(f"Sorted by length, then alphabetically: {words}")  # ['ant', 'bee', 'cat', 'dog', 'zebra', 'elephant']
    
    # Sorting dictionaries by value
    scores = [{"name": "Alice", "score": 95},
              {"name": "Bob", "score": 87},
              {"name": "Charlie", "score": 92}]
    
    scores.sort(key=lambda x: x["score"], reverse=True)
    print(f"Scores sorted by score (descending):\n{scores}")
    # Output: [{'name': 'Alice', 'score': 95}, {'name': 'Charlie', 'score': 92}, {'name': 'Bob', 'score': 87}]
    
    # Alternative: sorted() function (returns a new list)
    original = [3, 1, 4, 1, 5, 9, 2]
    sorted_copy = sorted(original)
    print(f"Original: {original}")  # [3, 1, 4, 1, 5, 9, 2] (unchanged)
    print(f"Sorted copy: {sorted_copy}")  # [1, 1, 2, 3, 4, 5, 9]
    
    # Error handling: incomparable types
    try:
        mixed = [1, "two", 3.0]
        mixed.sort()
    except TypeError as e:
        print(f"Error sorting mixed types: {e}")  # '<' not supported between instances of 'str' and 'int'
    
    # Sorting with None values
    with_none = [3, None, 1, None, 2]
    with_none.sort()  # None is considered smaller than any number
    print(f"Sorted with None: {with_none}")  # [None, None, 1, 2, 3]
    
    # Using the key parameter to handle None values differently
    with_none = [3, None, 1, None, 2]
    # Put None values at the end
    with_none.sort(key=lambda x: (x is None, x))
    print(f"Sorted with None at end: {with_none}")  # [1, 2, 3, None, None]
    
    # Stability of sort
    data = [("Alice", 3), ("Bob", 2), ("Charlie", 3), ("David", 1)]
    data.sort(key=lambda x: x[1])  # Sort by the number only
    print(f"Stable sort preserves order of equal elements: {data}")
    # Output: [('David', 1), ('Bob', 2), ('Alice', 3), ('Charlie', 3)]
    # Note that Alice still comes before Charlie because sort is stable


# =============================================================================
# Main execution - demonstrates all list methods
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("1. list.append(x)")
    print("="*50)
    demonstrate_append()
    
    print("\n" + "="*50)
    print("2. list.clear()")
    print("="*50)
    demonstrate_clear()
    
    print("\n" + "="*50)
    print("3. list.copy()")
    print("="*50)
    demonstrate_copy()
    
    print("\n" + "="*50)
    print("4. list.count(x)")
    print("="*50)
    demonstrate_count()
    
    print("\n" + "="*50)
    print("5. list.extend(iterable)")
    print("="*50)
    demonstrate_extend()
    
    print("\n" + "="*50)
    print("6. list.index(x[, start[, end]])")
    print("="*50)
    demonstrate_index()
    
    print("\n" + "="*50)
    print("7. list.insert(i, x)")
    print("="*50)
    demonstrate_insert()
    
    print("\n" + "="*50)
    print("8. list.pop([i])")
    print("="*50)
    demonstrate_pop()
    
    print("\n" + "="*50)
    print("9. list.remove(x)")
    print("="*50)
    demonstrate_remove()
    
    print("\n" + "="*50)
    print("10. list.reverse()")
    print("="*50)
    demonstrate_reverse()
    
    print("\n" + "="*50)
    print("11. list.sort(*, key=None, reverse=False)")
    print("="*50)
    demonstrate_sort()