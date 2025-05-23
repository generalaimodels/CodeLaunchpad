# -*- coding: utf-8 -*-
"""
Data Structures - Advanced Python List Mastery and Beyond

This script offers a deep dive into Python's fundamental data structures, focusing primarily on lists and the `del` statement.
It is tailored for advanced Python developers seeking a rigorous understanding of list operations, their application in stack and queue implementations, list comprehensions (including nested forms), and the nuanced behavior of the `del` statement.

This exploration will cover:

    - More on Lists: Advanced list manipulations, performance considerations, and memory management.
    - Using Lists as Stacks: Implementing stack data structures using Python lists, analyzing LIFO behavior and efficiency.
    - Using Lists as Queues: Implementing queue data structures, discussing FIFO behavior and efficiency trade-offs for list-based queues.
    - List Comprehensions: Mastering list comprehensions for concise and efficient list creation, including conditional comprehensions.
    - Nested List Comprehensions: Constructing complex lists using nested comprehensions, exploring multi-dimensional data structures.
    - The del statement: In-depth analysis of the `del` statement for item removal, slice deletion, and variable unbinding, including memory implications.

Expect a focus on:

    - Performance optimization in list operations and comprehensions.
    - Pythonic idioms for data structure manipulation.
    - Advanced use cases for stacks and queues in Python.
    - Comprehensive understanding of list comprehensions for complex data transformations.
    - Memory management aspects related to lists and the `del` statement.
    - Handling potential exceptions and edge cases in data structure operations.

Let's embark on this advanced exploration of Python data structures, pushing beyond the basics to achieve true mastery.
"""

################################################################################
# 5. Data Structures
################################################################################

print("\n--- 5. Data Structures ---\n")

################################################################################
# 5.1. More on Lists
################################################################################

print("\n--- 5.1. More on Lists ---\n")

# Python lists are highly versatile, dynamic arrays capable of holding heterogeneous items.
# They offer a rich set of built-in methods for manipulation, making them a cornerstone of Python programming.
# Beyond basic usage, understanding list internals and advanced operations is crucial for efficient and robust code.

# --- List Methods - Recap and Advanced Insights ---
print("\n--- List Methods - Recap and Advanced Insights ---")

sample_list = [10, 20, 30, 20, 40]

# append(x): Adds item x to the end of the list. O(1) amortized time complexity.
sample_list.append(50)
print(f"append(50): {sample_list}")

# extend(iterable): Extends the list by appending all items from the iterable. Amortized O(k) where k is the length of iterable.
sample_list.extend([60, 70])
print(f"extend([60, 70]): {sample_list}")

# insert(i, x): Inserts item x at a given position i. O(n) time complexity due to potential element shifting.
sample_list.insert(2, 25)
print(f"insert(2, 25): {sample_list}")

# remove(x): Removes the *first* occurrence of item x. O(n) time complexity in the worst case (item not found or at the end). ValueError if x is not in the list.
sample_list.remove(20) # Removes the first 20
print(f"remove(20): {sample_list}")
try:
    sample_list.remove(100) # Non-existent element
except ValueError as e:
    print(f"remove(100) - ValueError: {e}")

# pop([i]): Removes and returns the item at index i (or the last item if index is not specified). O(1) for pop() (last item), O(n) for pop(i) (arbitrary index due to shifting). IndexError if list is empty or index out of range.
popped_item_last = sample_list.pop() # Removes and returns last item
print(f"pop(): Popped item = {popped_item_last}, List: {sample_list}")
popped_item_index = sample_list.pop(1) # Removes and returns item at index 1
print(f"pop(1): Popped item = {popped_item_index}, List: {sample_list}")
try:
    empty_list = []
    empty_list.pop() # Pop from empty list
except IndexError as e:
    print(f"pop() from empty list - IndexError: {e}")

# clear(): Removes all items from the list (equivalent to del a[:]). O(1) time complexity.
cleared_list = sample_list.copy() # Operate on a copy to preserve original
cleared_list.clear()
print(f"clear(): {cleared_list}")

# index(x[, start[, end]]): Returns zero-based index of the *first* occurrence of item x. ValueError if x is not in the list. Optional start and end arguments for search range. O(n) in the worst case.
index_25 = sample_list.index(25)
print(f"index(25): {index_25}")
index_20_from_3 = sample_list.index(20, 3) # Search for 20 starting from index 3
print(f"index(20, 3): {index_20_from_3}")
try:
    sample_list.index(100) # Non-existent element
except ValueError as e:
    print(f"index(100) - ValueError: {e}")

# count(x): Returns the number of times x appears in the list. O(n) time complexity.
count_20 = sample_list.count(20)
print(f"count(20): {count_20}")

# sort(*, key=None, reverse=False): Sorts the list *in place*. Stable sorting algorithm (Timsort). Average O(n log n) and worst-case O(n log n).
unsorted_list = [3, 1, 4, 1, 5, 9, 2, 6]
unsorted_list.sort() # Ascending order
print(f"sort(): {unsorted_list}")
unsorted_list.sort(reverse=True) # Descending order
print(f"sort(reverse=True): {unsorted_list}")
unsorted_list.sort(key=len) # Sort by length (example - if list of strings) - TypeError here as it's int list.

# reverse(): Reverses the elements of the list *in place*. O(n) time complexity.
reversed_list = unsorted_list.copy() # Operate on a copy
reversed_list.reverse()
print(f"reverse(): {reversed_list}")

# copy(): Returns a shallow copy of the list. O(n) time complexity.
copied_list = sample_list.copy()
print(f"copy(): {copied_list}, Are they the same object? {copied_list is sample_list}") # False - different objects, but shallow copy

# --- List Mutability and Memory ---
print("\n--- List Mutability and Memory ---")
# Lists are mutable: their contents can be changed after creation.
# When you modify a list in place (e.g., append, insert, remove, sort, reverse), you are operating on the *same* list object in memory.
# Assignment (e.g., `list_b = list_a`) does *not* create a copy; it creates a new reference to the *same* list object. To create a copy, use `list_a.copy()` or `list(list_a)` or slicing `list_a[:]`.

original_list = [1, 2, [3, 4]]
reference_list = original_list # Reference, not a copy
copied_list_shallow = original_list.copy() # Shallow copy
# copied_list_deep = 
import copy;copied_list_deep =  copy.deepcopy(original_list) # Deep copy (import copy required)

original_list[0] = 100 # Modify original list
original_list[2][0] = 300 # Modify nested list in original list

print(f"Original list after modification: {original_list}")
print(f"Reference list (same object): {reference_list}") # Reflects changes in original
print(f"Shallow copied list: {copied_list_shallow}") # Top-level items are different, but nested list is still shared!
print(f"Deep copied list: {copied_list_deep}") # Completely independent copy

# --- Performance Considerations ---
print("\n--- Performance Considerations ---")
# - Appending and popping from the end of a list are generally efficient (amortized O(1)).
# - Inserting or removing elements at the beginning or middle of a list is less efficient (O(n)) due to element shifting.
# - Searching for an element (index, remove, in) is O(n) in the worst case.
# - Sorting is O(n log n).
# - For performance-critical applications requiring frequent insertions/deletions at both ends, consider `collections.deque` (double-ended queue).
# - For numerical operations on large datasets, NumPy arrays are significantly more efficient than Python lists.

################################################################################
# 5.1.1. Using Lists as Stacks
################################################################################

print("\n--- 5.1.1. Using Lists as Stacks ---\n")

# Stacks are Last-In, First-Out (LIFO) data structures. The last element added is the first one removed.
# Python lists can be efficiently used as stacks using `append()` to push (add) elements onto the stack and `pop()` to pop (remove and retrieve) elements from the top of the stack.

# --- Stack Implementation using List ---
print("\n--- Stack Implementation using List ---")
stack = [] # Initialize an empty list as a stack

# Push elements onto the stack using append()
stack.append('a')
stack.append('b')
stack.append('c')
print(f"Stack after pushes: {stack}")

# Pop elements from the stack using pop()
top_element = stack.pop() # Removes and returns the last added element ('c')
print(f"Popped element: {top_element}, Stack after pop: {stack}")
top_element = stack.pop() # Removes and returns 'b'
print(f"Popped element: {top_element}, Stack after pop: {stack}")

# Check if stack is empty
if not stack:
    print("Stack is empty")
else:
    print("Stack is not empty")

# --- Stack Underflow Handling ---
print("\n--- Stack Underflow Handling ---")
# Attempting to pop from an empty stack will raise an IndexError. Handle this exception to prevent program crashes.
empty_stack = []
try:
    popped_from_empty = empty_stack.pop() # Attempt to pop from empty stack
except IndexError as e:
    print(f"Stack underflow - IndexError: {e}")

# --- Stack Use Cases ---
print("\n--- Stack Use Cases ---")
# - Function call stack (internal to Python interpreter).
# - Expression evaluation (e.g., postfix notation).
# - Backtracking algorithms (e.g., depth-first search).
# - Undo/redo functionality.
# - Browser history.

# --- Efficiency of List as Stack ---
print("\n--- Efficiency of List as Stack ---")
# - push (append): O(1) amortized.
# - pop: O(1).
# - Top element access (stack[-1]): O(1).
# - Excellent performance for stack operations using lists.

################################################################################
# 5.1.2. Using Lists as Queues
################################################################################

print("\n--- 5.1.2. Using Lists as Queues ---\n")

# Queues are First-In, First-Out (FIFO) data structures. The first element added is the first one removed.
# While lists can be used as queues, they are *not efficient* for this purpose for larger queues.
# Using `append()` to enqueue (add to the back) is efficient (O(1)), but using `pop(0)` to dequeue (remove from the front) is *inefficient* (O(n)) because it requires shifting all subsequent elements.

# --- Queue Implementation using List (Inefficient for large queues) ---
print("\n--- Queue Implementation using List (Inefficient) ---")
queue = [] # Initialize an empty list as a queue

# Enqueue elements using append() (add to the back)
queue.append('item1')
queue.append('item2')
queue.append('item3')
print(f"Queue after enqueues: {queue}")

# Dequeue elements using pop(0) (remove from the front - INEFFICIENT!)
first_item = queue.pop(0) # Removes and returns the first added element ('item1') - O(n) operation!
print(f"Dequeued item: {first_item}, Queue after dequeue: {queue}")
first_item = queue.pop(0) # Removes and returns 'item2' - O(n) operation again!
print(f"Dequeued item: {first_item}, Queue after dequeue: {queue}")

# Check if queue is empty
if not queue:
    print("Queue is empty")
else:
    print("Queue is not empty")

# --- Queue Underflow Handling ---
print("\n--- Queue Underflow Handling ---")
# Similar to stacks, popping from an empty queue (using pop(0) or pop()) raises IndexError.
empty_queue = []
try:
    dequeued_from_empty = empty_queue.pop(0) # Attempt to dequeue from empty queue
except IndexError as e:
    print(f"Queue underflow - IndexError: {e}")

# --- Queue Use Cases ---
print("\n--- Queue Use Cases ---")
# - Task scheduling (e.g., print queue, job queue).
# - Breadth-first search (BFS) algorithms.
# - Message queues in inter-process communication.
# - Buffering data streams.

# --- Efficiency of List as Queue (Inefficient Dequeue) ---
print("\n--- Efficiency of List as Queue (Inefficient Dequeue) ---")
# - enqueue (append): O(1) amortized.
# - dequeue (pop(0)): O(n) - INEFFICIENT for large queues.
# - Front element access (queue[0]): O(1).
# - Due to O(n) dequeue, lists are *not recommended* for queue implementations that require frequent dequeues, especially for large queues.

# --- Efficient Queue Implementation using collections.deque ---
print("\n--- Efficient Queue using collections.deque ---")
from collections import deque # Import deque from collections module

efficient_queue = deque() # Initialize deque as an efficient queue

# Enqueue using append() (adds to the right end - efficient for deque)
efficient_queue.append('task1')
efficient_queue.append('task2')
efficient_queue.append('task3')
print(f"Efficient queue after enqueues: {efficient_queue}")

# Dequeue using popleft() (removes from the left end - efficient for deque)
first_task = efficient_queue.popleft() # Removes and returns the leftmost element - O(1) operation in deque!
print(f"Dequeued task: {first_task}, Efficient queue after dequeue: {efficient_queue}")
first_task = efficient_queue.popleft() # O(1) dequeue again!
print(f"Dequeued task: {first_task}, Efficient queue after dequeue: {efficient_queue}")

# collections.deque is optimized for appends and pops from both ends (left and right), making it highly efficient for both stack and queue implementations.
# For queue implementations, always prefer `collections.deque` over lists for performance reasons, especially when dealing with potentially large queues.

################################################################################
# 5.1.3. List Comprehensions
################################################################################

print("\n--- 5.1.3. List Comprehensions ---\n")

# List comprehensions provide a concise and readable way to create lists in Python.
# They offer a more compact syntax compared to traditional `for` loops for building lists, often with performance advantages in CPython implementations.
# Basic syntax: `[expression for item in iterable if condition]` (optional if condition)

# --- Basic List Comprehension - Squaring Numbers ---
print("\n--- Basic List Comprehension - Squaring Numbers ---")
numbers = [1, 2, 3, 4, 5]
squared_numbers_lc = [number**2 for number in numbers] # List comprehension for squaring
print(f"Original numbers: {numbers}")
print(f"Squared numbers (list comprehension): {squared_numbers_lc}")

# --- List Comprehension with Conditional Filtering - Even Numbers ---
print("\n--- List Comprehension with Conditional Filtering - Even Numbers ---")
even_numbers_lc = [number for number in numbers if number % 2 == 0] # List comprehension with if condition
print(f"Even numbers (list comprehension with condition): {even_numbers_lc}")

# --- List Comprehension with Transformation and Condition ---
print("\n--- List Comprehension with Transformation and Condition ---")
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
long_words_uppercase_lc = [word.upper() for word in words if len(word) > 5] # Transform to uppercase and filter by length
print(f"Original words: {words}")
print(f"Long words (uppercase, list comprehension with condition): {long_words_uppercase_lc}")

# --- List Comprehension vs. Traditional for loop ---
print("\n--- List Comprehension vs. Traditional for loop ---")
# List comprehension (more concise and often faster):
lc_squares = [x**2 for x in range(10)]

# Equivalent for loop (more verbose):
loop_squares = []
for x in range(10):
    loop_squares.append(x**2)

print(f"List comprehension squares: {lc_squares}")
print(f"For loop squares: {loop_squares}")

# --- Performance of List Comprehensions ---
print("\n--- Performance of List Comprehensions ---")
# In CPython, list comprehensions are generally faster than equivalent for loops for list creation.
# This is because list comprehensions are often optimized at the bytecode level, reducing interpreter overhead.
# However, for very complex expressions or operations within the comprehension, the performance difference might become less significant.
# Readability is often a more compelling reason to use list comprehensions for simple list creation and transformations.

# --- Handling potential exceptions within List Comprehensions ---
print("\n--- List Comprehension Exception Handling ---")
data_with_errors = ['1', '2', 'invalid', '4', '5']
# Directly handling exceptions *within* a list comprehension is not straightforward.
# For robust error handling, it's often better to use a for loop with try-except blocks, or pre-process the data.
# However, you can use conditional expressions to avoid errors in some cases.

# Example - Safely converting to integers, skipping invalid entries (using conditional expression)
safe_integers_lc = [int(item) if item.isdigit() else None for item in data_with_errors] # Use None for invalid
print(f"Data with potential errors: {data_with_errors}")
print(f"Safe integers (list comprehension with conditional expression): {safe_integers_lc}") # Contains None for 'invalid'

# For more complex error handling or logging, a traditional for loop with try-except is often preferred.

################################################################################
# 5.1.4. Nested List Comprehensions
################################################################################

print("\n--- 5.1.4. Nested List Comprehensions ---\n")

# Nested list comprehensions are used to create lists of lists (multi-dimensional lists) or to perform operations involving nested iterations.
# They can be powerful but can become less readable if nesting becomes too deep.
# Syntax for nested loop equivalent:
# outer_list = []
# for outer_item in outer_iterable:
#     inner_list = []
#     for inner_item in inner_iterable:
#         inner_list.append(expression(outer_item, inner_item))
#     outer_list.append(inner_list)

# List comprehension equivalent:
# [ [expression(outer_item, inner_item) for inner_item in inner_iterable] for outer_item in outer_iterable ]

# --- Basic Nested List Comprehension - Creating a Matrix ---
print("\n--- Basic Nested List Comprehension - Creating a Matrix ---")
matrix_2x3_lc = [[0 for _ in range(3)] for _ in range(2)] # 2 rows, 3 columns initialized to 0
print(f"2x3 Matrix (nested list comprehension): {matrix_2x3_lc}")

# --- Nested List Comprehension with Iteration over Multiple Iterables ---
print("\n--- Nested List Comprehension with Multiple Iterables ---")
colors = ['red', 'green', 'blue']
fruits = ['apple', 'banana', 'cherry']
combinations_lc = [(color, fruit) for color in colors for fruit in fruits] # Nested loops to create combinations
print(f"Colors: {colors}")
print(f"Fruits: {fruits}")
print(f"Combinations (nested list comprehension): {combinations_lc}")

# --- Nested List Comprehension with Conditional Filtering in Inner and Outer Loops ---
print("\n--- Nested List Comprehension with Conditional Filtering ---")
matrix_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Extract even numbers from each row, only for rows where the first element is even
even_numbers_nested_lc = [[num for num in row if num % 2 == 0]
                            for row in matrix_data if row[0] % 2 == 0] # Outer condition on row, inner on number
print(f"Matrix data: {matrix_data}")
print(f"Even numbers from even-starting rows (nested list comprehension with conditions): {even_numbers_nested_lc}") # Empty list as no row starts with even.

matrix_data_2 = [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
even_numbers_nested_lc_2 = [[num for num in row if num % 2 == 0]
                            for row in matrix_data_2 if row[0] % 2 == 0] # Outer condition on row, inner on number
print(f"Matrix data 2: {matrix_data_2}")
print(f"Even numbers from even-starting rows (nested list comprehension with conditions - example 2): {even_numbers_nested_lc_2}")

# --- Readability and Complexity of Nested List Comprehensions ---
print("\n--- Readability and Complexity ---")
# While powerful, deeply nested list comprehensions can become hard to read and understand.
# For complex data transformations or multiple nested iterations, consider breaking down the logic into separate for loops or helper functions for better readability and maintainability.
# Aim for clarity over extreme conciseness, especially in collaborative projects.

################################################################################
# 5.2. The del statement
################################################################################

print("\n--- 5.2. The del statement ---\n")

# The `del` statement in Python is used to delete objects. It can delete items from lists, slices of lists, variables, and attributes.
# Importantly, `del` unbinds names from objects, and for mutable objects like lists, it can modify the list in place.
# If the object is no longer referenced after `del`, it becomes eligible for garbage collection.

# --- Deleting List Items by Index ---
print("\n--- Deleting List Items by Index ---")
delete_list_index = ['a', 'b', 'c', 'd', 'e']
del delete_list_index[2] # Delete item at index 2 ('c')
print(f"List before del[2]: ['a', 'b', 'c', 'd', 'e'], List after del[2]: {delete_list_index}")

try:
    del delete_list_index[10] # Index out of range
except IndexError as e:
    print(f"del delete_list_index[10] - IndexError: {e}")

# --- Deleting List Slices ---
print("\n--- Deleting List Slices ---")
delete_list_slice = ['p', 'q', 'r', 's', 't', 'u']
del delete_list_slice[1:4] # Delete slice from index 1 up to (but not including) 4 ('q', 'r', 's')
print(f"List before del[1:4]: ['p', 'q', 'r', 's', 't', 'u'], List after del[1:4]: {delete_list_slice}")

del delete_list_slice[:] # Delete all elements in the list (clear list)
print(f"List before del[:]: ['p', 't', 'u'], List after del[:]: {delete_list_slice}") # Empty list

# --- Deleting Variables (Unbinding Names) ---
print("\n--- Deleting Variables (Unbinding Names) ---")
variable_to_delete = 100
del variable_to_delete # Unbind the name 'variable_to_delete' from the object 100
# After del, the name 'variable_to_delete' is no longer defined in the current scope.
try:
    print(variable_to_delete) # Accessing deleted variable
except NameError as e:
    print(f"Accessing deleted variable - NameError: {e}")

# --- del and Memory Management ---
print("\n--- del and Memory Management ---")
# `del` does *not* directly deallocate memory in Python. It unbinds names.
# If a name was the last reference to an object, the object becomes eligible for garbage collection.
# Python's garbage collector reclaims memory occupied by unreferenced objects.
# In most cases, you don't need to explicitly use `del` for memory management. Python's automatic garbage collection is generally sufficient.
# `del` is more about removing names from namespaces and modifying mutable objects in place.

# Example:
list_for_gc = [1, 2, 3, 4, 5]
another_reference = list_for_gc
del list_for_gc # 'list_for_gc' name is unbound, but the list object still exists because 'another_reference' still points to it.
print(f"List still accessible via another reference: {another_reference}")

del another_reference # Now, no more references to the list (assuming no other references elsewhere). The list object is now eligible for garbage collection.

# --- del on Attributes of Objects (Classes - more advanced) ---
# `del object.attribute` can be used to delete attributes from objects. (Covered in OOP sections)

# --- Exception Handling with del ---
# `del` itself generally does not raise exceptions other than IndexError (for list index/slice out of range) and NameError (if you try to del an unbound name - though this is rare in normal usage as you're usually deleting something that *was* defined).

print("\n--- End of Data Structures - Lists and del ---")