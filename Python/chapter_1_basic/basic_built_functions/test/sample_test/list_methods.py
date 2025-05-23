#!/usr/bin/env python3
"""
Comprehensive demonstration of Python list methods:
- append: Add an element to the end of the list
- clear: Remove all items from the list
- copy: Create a shallow copy of the list
- count: Count occurrences of an element
- extend: Add elements from another iterable
- index: Find the position of an element
- insert: Insert an element at a specific position
- pop: Remove and return an element
- remove: Remove first occurrence of a value
- reverse: Reverse the list in-place
- sort: Sort the list in-place
"""


def demonstrate_append():
    """
    append(x): Add item x to the end of the list.
    Time Complexity: O(1) - Amortized constant time
    """
    # Basic usage
    numbers = [1, 2, 3]
    numbers.append(4)
    print(f"After append(4): {numbers}")  # [1, 2, 3, 4]
    
    # Appending different data types
    mixed_list = []
    mixed_list.append(42)
    mixed_list.append("hello")
    mixed_list.append([1, 2, 3])  # Appending a list as a single element
    print(f"Mixed list after appends: {mixed_list}")  # [42, 'hello', [1, 2, 3]]
    
    # Common mistake: append vs extend
    numbers = [1, 2, 3]
    numbers.append([4, 5])  # This adds the list as a single item
    print(f"After append([4, 5]): {numbers}")  # [1, 2, 3, [4, 5]]


def demonstrate_clear():
    """
    clear(): Remove all items from the list.
    Time Complexity: O(n) - Linear time
    """
    # Basic usage
    numbers = [1, 2, 3, 4, 5]
    print(f"Before clear(): {numbers}")
    numbers.clear()
    print(f"After clear(): {numbers}")  # []
    
    # Clearing nested lists (note: doesn't affect nested lists themselves)
    nested = [[1, 2], [3, 4]]
    inner = nested[0]
    nested.clear()
    print(f"After clear() on nested: {nested}")  # []
    print(f"Inner list reference: {inner}")  # [1, 2] - still exists
    
    # Alternative ways to clear a list (not recommended)
    numbers = [1, 2, 3, 4, 5]
    numbers[:] = []
    print(f"After slice assignment: {numbers}")  # []
    
    numbers = [1, 2, 3, 4, 5]
    del numbers[:]
    print(f"After del slice: {numbers}")  # []


def demonstrate_copy():
    """
    copy(): Return a shallow copy of the list.
    Time Complexity: O(n) - Linear time
    """
    # Basic usage
    original = [1, 2, 3]
    copied = original.copy()
    print(f"Original: {original}, Copy: {copied}")
    
    # Modifying the copy doesn't affect original
    copied.append(4)
    print(f"After modifying copy - Original: {original}, Copy: {copied}")
    
    # Shallow vs. deep copy demonstration
    nested = [[1, 2], [3, 4]]
    shallow = nested.copy()
    
    # Modifying inner list affects both because it's a shallow copy
    shallow[0].append(99)
    print(f"Original nested after modifying shallow copy: {nested}")  # [[1, 2, 99], [3, 4]]
    print(f"Shallow copy: {shallow}")  # [[1, 2, 99], [3, 4]]
    
    # Deep copy example using the copy module
    import copy
    nested = [[1, 2], [3, 4]]
    deep = copy.deepcopy(nested)
    deep[0].append(99)
    print(f"Original nested after modifying deep copy: {nested}")  # [[1, 2], [3, 4]]
    print(f"Deep copy: {deep}")  # [[1, 2, 99], [3, 4]]
    
    # Alternative ways to create a shallow copy
    original = [1, 2, 3]
    copy1 = list(original)  # Using the list() constructor
    copy2 = original[:]     # Using slice notation
    print(f"copy() method: {original.copy()}")
    print(f"list() constructor: {copy1}")
    print(f"Slice notation: {copy2}")


def demonstrate_count():
    """
    count(x): Return the number of occurrences of x in the list.
    Time Complexity: O(n) - Linear time
    """
    # Basic usage
    numbers = [1, 2, 2, 3, 2, 4, 5]
    count_of_2 = numbers.count(2)
    print(f"Count of 2 in {numbers}: {count_of_2}")  # 3
    
    # Counting elements that don't exist
    count_of_9 = numbers.count(9)
    print(f"Count of 9 in {numbers}: {count_of_9}")  # 0
    
    # Counting with different data types
    mixed = [1, "hello", 1.5, True, "hello", [1, 2], [1, 2]]
    print(f"Count of 'hello': {mixed.count('hello')}")  # 2
    print(f"Count of True: {mixed.count(True)}")  # 1
    
    # Lists are compared by identity for equality
    list_item = [1, 2]
    mixed.append(list_item)
    print(f"Count of [1, 2]: {mixed.count([1, 2])}")  # 2
    print(f"Count of list_item object: {mixed.count(list_item)}")  # 1
    
    # Count with more complex objects
    class Person:
        def __init__(self, name):
            self.name = name
    
    people = [Person("Alice"), Person("Bob"), Person("Alice")]
    # This counts object instances, not equal names
    alice_count = sum(1 for person in people if person.name == "Alice")
    print(f"Count of people named Alice: {alice_count}")  # 2


def demonstrate_extend():
    """
    extend(iterable): Add all items from the iterable to the end of the list.
    Time Complexity: O(k) where k is the length of the iterable
    """
    # Basic usage
    numbers = [1, 2, 3]
    numbers.extend([4, 5, 6])
    print(f"After extend([4, 5, 6]): {numbers}")  # [1, 2, 3, 4, 5, 6]
    
    # Extend with other iterable types
    chars = ['a', 'b', 'c']
    # Using a tuple
    chars.extend(('d', 'e'))
    print(f"After extend with tuple: {chars}")  # ['a', 'b', 'c', 'd', 'e']
    
    # Using a string (each character is added)
    chars.extend("fg")
    print(f"After extend with string: {chars}")  # ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    
    # Using a set (order not guaranteed)
    set_example = {1, 2, 3}
    numbers = []
    numbers.extend(set_example)
    print(f"After extend with set: {numbers}")  # Order may vary
    
    # Extend with a dictionary (only keys are added)
    dict_example = {'a': 1, 'b': 2}
    letters = []
    letters.extend(dict_example)
    print(f"After extend with dict: {letters}")  # ['a', 'b']
    
    # Common mistake: append vs extend
    list1 = [1, 2, 3]
    list2 = [1, 2, 3]
    
    list1.append([4, 5])  # Adds [4, 5] as a single element
    list2.extend([4, 5])  # Adds 4 and 5 as separate elements
    
    print(f"After append([4, 5]): {list1}")  # [1, 2, 3, [4, 5]]
    print(f"After extend([4, 5]): {list2}")  # [1, 2, 3, 4, 5]
    
    # Alternative to extend using concatenation
    list1 = [1, 2, 3]
    list2 = list1 + [4, 5]  # Creates a new list
    print(f"Using + operator: {list2}")  # [1, 2, 3, 4, 5]
    
    # Alternative using slice assignment
    list1 = [1, 2, 3]
    list1[len(list1):] = [4, 5]
    print(f"Using slice assignment: {list1}")  # [1, 2, 3, 4, 5]


def demonstrate_index():
    """
    index(x[, start[, end]]): Return the index of the first occurrence of x.
    Raises ValueError if x is not found.
    Time Complexity: O(n) - Linear time
    """
    # Basic usage
    fruits = ["apple", "banana", "cherry", "banana", "elderberry"]
    banana_index = fruits.index("banana")
    print(f"Index of 'banana': {banana_index}")  # 1 (first occurrence)
    
    # Using start and end parameters
    # Find 'banana' starting from position 2
    banana_index_from_2 = fruits.index("banana", 2)
    print(f"Index of 'banana' from position 2: {banana_index_from_2}")  # 3
    
    # Find 'cherry' between positions 1 and 3
    cherry_index = fruits.index("cherry", 1, 4)
    print(f"Index of 'cherry' between positions 1-4: {cherry_index}")  # 2
    
    # Error handling for non-existent elements
    try:
        mango_index = fruits.index("mango")
    except ValueError as e:
        print(f"Error finding 'mango': {e}")
    
    # Alternative: Safe index finding
    def safe_index(lst, item, default=-1):
        """Return index of item in lst or default if not found."""
        try:
            return lst.index(item)
        except ValueError:
            return default
    
    safe_mango_index = safe_index(fruits, "mango")
    print(f"Safe index of 'mango': {safe_mango_index}")  # -1
    
    # Working with different data types
    mixed = [1, 2.5, "three", True, None, [4, 5]]
    print(f"Index of True: {mixed.index(True)}")  # 3
    print(f"Index of None: {mixed.index(None)}")  # 4


def demonstrate_insert():
    """
    insert(i, x): Insert item x at position i.
    Time Complexity: O(n) - Linear time (because elements after i need to be shifted)
    """
    # Basic usage
    fruits = ["apple", "cherry", "elderberry"]
    fruits.insert(1, "banana")
    print(f"After insert at position 1: {fruits}")  # ['apple', 'banana', 'cherry', 'elderberry']
    
    # Insert at the beginning (equivalent to prepend)
    fruits.insert(0, "apricot")
    print(f"After insert at position 0: {fruits}")  # ['apricot', 'apple', 'banana', 'cherry', 'elderberry']
    
    # Insert at the end (though append() is more efficient for this)
    fruits.insert(len(fruits), "fig")
    print(f"After insert at the end: {fruits}")  # ['apricot', 'apple', 'banana', 'cherry', 'elderberry', 'fig']
    
    # Insert with negative indices
    numbers = [1, 2, 3, 5]
    # Insert at position -1 (before the last element)
    numbers.insert(-1, 4)
    print(f"After insert at position -1: {numbers}")  # [1, 2, 3, 4, 5]
    
    # Insert with out-of-range indices
    letters = ['a', 'b', 'c']
    # If i > len(list), it's equivalent to append()
    letters.insert(100, 'd')
    print(f"After insert at position 100: {letters}")  # ['a', 'b', 'c', 'd']
    
    # If i < 0 and abs(i) > len(list), it's inserted at position 0
    letters.insert(-100, 'z')
    print(f"After insert at position -100: {letters}")  # ['z', 'a', 'b', 'c', 'd']


def demonstrate_pop():
    """
    pop([i]): Remove and return item at position i (default is last).
    Raises IndexError if list is empty or i is out of range.
    Time Complexity: O(1) for the last element, O(n) for arbitrary position
    """
    # Basic usage - pop from the end (most efficient)
    fruits = ["apple", "banana", "cherry", "date"]
    last_fruit = fruits.pop()
    print(f"Popped element: {last_fruit}")  # date
    print(f"List after pop(): {fruits}")  # ['apple', 'banana', 'cherry']
    
    # Pop from a specific position
    second_fruit = fruits.pop(1)
    print(f"Popped element at index 1: {second_fruit}")  # banana
    print(f"List after pop(1): {fruits}")  # ['apple', 'cherry']
    
    # Pop with negative index
    fruits = ["apple", "banana", "cherry", "date"]
    second_last = fruits.pop(-2)
    print(f"Popped element at index -2: {second_last}")  # cherry
    print(f"List after pop(-2): {fruits}")  # ['apple', 'banana', 'date']
    
    # Error handling for empty lists
    empty_list = []
    try:
        empty_list.pop()
    except IndexError as e:
        print(f"Error popping from empty list: {e}")
    
    # Error handling for out-of-range indices
    short_list = [1, 2, 3]
    try:
        short_list.pop(5)
    except IndexError as e:
        print(f"Error popping from out-of-range index: {e}")
    
    # Using pop() to implement a stack (LIFO - Last In, First Out)
    stack = []
    stack.append(1)  # Push
    stack.append(2)  # Push
    stack.append(3)  # Push
    print(f"Stack: {stack}")  # [1, 2, 3]
    
    top_item = stack.pop()  # Pop from top
    print(f"Popped from stack: {top_item}")  # 3
    print(f"Stack after pop: {stack}")  # [1, 2]


def demonstrate_remove():
    """
    remove(x): Remove the first occurrence of item x.
    Raises ValueError if item is not found.
    Time Complexity: O(n) - Linear time
    """
    # Basic usage
    fruits = ["apple", "banana", "cherry", "banana", "date"]
    fruits.remove("banana")  # Removes only the first occurrence
    print(f"After removing 'banana': {fruits}")  # ['apple', 'cherry', 'banana', 'date']
    
    # Error handling for non-existent elements
    try:
        fruits.remove("mango")
    except ValueError as e:
        print(f"Error removing 'mango': {e}")
    
    # Removing all occurrences of an element
    numbers = [1, 2, 3, 2, 4, 2, 5]
    element_to_remove = 2
    
    # Method 1: Using a while loop with try/except
    numbers_copy1 = numbers.copy()
    try:
        while True:
            numbers_copy1.remove(element_to_remove)
    except ValueError:
        pass
    print(f"After removing all 2s (method 1): {numbers_copy1}")  # [1, 3, 4, 5]
    
    # Method 2: Using list comprehension (more efficient)
    numbers_copy2 = [x for x in numbers if x != element_to_remove]
    print(f"After removing all 2s (method 2): {numbers_copy2}")  # [1, 3, 4, 5]
    
    # Removing elements based on a condition
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Note: Use a copy for iteration to avoid issues with changing the list during iteration
    for num in numbers[:]:
        if num % 2 == 0:
            numbers.remove(num)
    print(f"After removing even numbers: {numbers}")  # [1, 3, 5, 7, 9]


def demonstrate_reverse():
    """
    reverse(): Reverse the elements of the list in-place.
    Time Complexity: O(n) - Linear time
    """
    # Basic usage
    numbers = [1, 2, 3, 4, 5]
    numbers.reverse()
    print(f"After reverse(): {numbers}")  # [5, 4, 3, 2, 1]
    
    # Reversing an empty list
    empty = []
    empty.reverse()
    print(f"After reversing empty list: {empty}")  # []
    
    # Reversing a list with one element
    singleton = [42]
    singleton.reverse()
    print(f"After reversing singleton list: {singleton}")  # [42]
    
    # Note: reverse() modifies the list in-place and returns None
    fruits = ["apple", "banana", "cherry"]
    result = fruits.reverse()
    print(f"Return value of reverse(): {result}")  # None
    print(f"List after reverse(): {fruits}")  # ['cherry', 'banana', 'apple']
    
    # Alternative ways to reverse a list
    
    # 1. Using the reversed() function (returns an iterator)
    numbers = [1, 2, 3, 4, 5]
    reversed_iterator = reversed(numbers)
    reversed_list = list(reversed_iterator)
    print(f"Original list: {numbers}")  # [1, 2, 3, 4, 5]
    print(f"Reversed list (using reversed()): {reversed_list}")  # [5, 4, 3, 2, 1]
    
    # 2. Using slicing with negative step
    numbers = [1, 2, 3, 4, 5]
    reversed_list = numbers[::-1]  # Creates a new list
    print(f"Original list: {numbers}")  # [1, 2, 3, 4, 5]
    print(f"Reversed list (using slicing): {reversed_list}")  # [5, 4, 3, 2, 1]


def demonstrate_sort():
    """
    sort(*, key=None, reverse=False): Sort the list in-place.
    key: Function to extract a comparison key
    reverse: If True, sort in descending order
    Time Complexity: O(n log n) - where n is the length of the list
    """
    # Basic usage
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    numbers.sort()
    print(f"Sorted numbers: {numbers}")  # [1, 1, 2, 3, 4, 5, 6, 9]
    
    # Sorting in reverse order
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    numbers.sort(reverse=True)
    print(f"Reverse sorted numbers: {numbers}")  # [9, 6, 5, 4, 3, 2, 1, 1]
    
    # Sorting strings (lexicographical order)
    fruits = ["banana", "apple", "Cherry", "date", "elderberry"]
    fruits.sort()
    print(f"Sorted fruits (case-sensitive): {fruits}")  # ['Cherry', 'apple', 'banana', 'date', 'elderberry']
    
    # Case-insensitive sorting using the key parameter
    fruits = ["banana", "apple", "Cherry", "date", "elderberry"]
    fruits.sort(key=str.lower)
    print(f"Sorted fruits (case-insensitive): {fruits}")  # ['apple', 'banana', 'Cherry', 'date', 'elderberry']
    
    # Sorting with a custom key function
    people = [
        {"name": "Alice", "age": 25},
        {"name": "Bob", "age": 20},
        {"name": "Charlie", "age": 30}
    ]
    
    # Sort by age
    people.sort(key=lambda person: person["age"])
    print(f"Sorted by age: {[p['name'] for p in people]}")  # ['Bob', 'Alice', 'Charlie']
    
    # Sort by name
    people.sort(key=lambda person: person["name"])
    print(f"Sorted by name: {[p['name'] for p in people]}")  # ['Alice', 'Bob', 'Charlie']
    
    # Sorting with multiple criteria using a tuple as key
    students = [
        ("Alice", "A", 15),
        ("Bob", "B", 12),
        ("Charlie", "A", 20),
        ("David", "C", 10),
        ("Eve", "B", 14)
    ]
    
    # Sort by grade, then by age (ascending)
    students.sort(key=lambda x: (x[1], x[2]))
    print(f"Sorted by grade, then age: {students}")
    
    # Sort by grade ascending, then by age descending
    students.sort(key=lambda x: (x[1], -x[2]))
    print(f"Sorted by grade ascending, age descending: {students}")
    
    # Stability of sort
    # Python's sort is stable - equal items maintain their relative order
    data = [("Alice", 3), ("Bob", 2), ("Charlie", 3), ("David", 1)]
    data.sort(key=lambda x: x[1])  # Sort by the second element
    print(f"Stable sort result: {data}")  
    # [('David', 1), ('Bob', 2), ('Alice', 3), ('Charlie', 3)]


def main():
    """Execute demonstrations of each list method."""
    methods = [
        demonstrate_append,
        demonstrate_clear,
        demonstrate_copy,
        demonstrate_count,
        demonstrate_extend,
        demonstrate_index,
        demonstrate_insert,
        demonstrate_pop,
        demonstrate_remove,
        demonstrate_reverse,
        demonstrate_sort
    ]
    
    for method in methods:
        print(f"\n{'-'*50}")
        print(f"Demonstrating: {method.__name__.replace('demonstrate_', '')}")
        print(f"{'-'*50}")
        method()


if __name__ == "__main__":
    main()