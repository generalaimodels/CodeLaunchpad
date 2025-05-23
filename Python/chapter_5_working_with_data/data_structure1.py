# Chapter 5: Working with Data Structures 🗂️
# Lists, Tuples, Dictionaries, Sets

# 5.1 Lists in Detail 📜 (Mutable Sequences)

# Example 1: Creating a list using square brackets []
fruits = ['apple', 'banana', 'cherry']
# 'fruits' is a list of fruit names 🍎🍌🍒

# Example 2: Accessing elements by index 📍
first_fruit = fruits[0]  # 'apple'
# Indexing starts at 0

# Example 3: Negative indexing (from the end)
last_fruit = fruits[-1]  # 'cherry'

# Example 4: Slicing lists ✂️
sub_list = fruits[0:2]  # ['apple', 'banana']
# Slices from index 0 up to but not including 2

# Example 5: Appending an item to the list 🛠️
fruits.append('orange')  # Now fruits is ['apple', 'banana', 'cherry', 'orange']

# Example 6: Inserting an item at a specific index
fruits.insert(1, 'mango')  # Insert 'mango' at index 1
# Now fruits is ['apple', 'mango', 'banana', 'cherry', 'orange']

# Example 7: Removing an item by value
fruits.remove('banana')  # Removes first occurrence of 'banana'
# Now fruits is ['apple', 'mango', 'cherry', 'orange']

# Example 8: Removing an item by index with pop()
popped_fruit = fruits.pop(2)  # Removes and returns item at index 2
# popped_fruit is 'cherry', fruits is now ['apple', 'mango', 'orange']

# Example 9: Reversing a list
fruits.reverse()  # Now fruits is ['orange', 'mango', 'apple']

# Example 10: Sorting a list
fruits.sort()  # Now fruits is ['apple', 'mango', 'orange']

# Example 11: List length
number_of_fruits = len(fruits)  # 3

# Example 12: Checking if an item exists in the list
has_apple = 'apple' in fruits  # True 🍏

# Example 13: List concatenation
more_fruits = ['pineapple', 'grape']
all_fruits = fruits + more_fruits
# all_fruits is ['apple', 'mango', 'orange', 'pineapple', 'grape']

# Example 14: List comprehension ⚡
numbers = [1, 2, 3, 4, 5]
squares = [num ** 2 for num in numbers]
# squares is [1, 4, 9, 16, 25]

# Example 15: Nested lists (lists within lists) 📦📦
matrix = [[1, 2], [3, 4], [5, 6]]
# matrix is a list of lists, can represent a 2D array

# Example 16: Modifying elements in a list (since lists are mutable)
fruits[0] = 'pear'  # Change 'apple' to 'pear'
# fruits is now ['pear', 'mango', 'orange']

# Example 17: Iterating over a list
for fruit in fruits:
    print(fruit)  # Prints each fruit in the list

# Example 18: Extending a list
fruits.extend(['kiwi', 'strawberry'])
# fruits is now ['pear', 'mango', 'orange', 'kiwi', 'strawberry']

# Example 19: Deleting an element by index
del fruits[1]  # Deletes 'mango' (index 1)
# fruits is now ['pear', 'orange', 'kiwi', 'strawberry']

# Example 20: Clearing a list
fruits.clear()  # Now fruits is []

# Possible mistakes/exceptions:
# Example 21: IndexError when accessing out-of-range index
# invalid_fruit = fruits[5]  # IndexError: list index out of range

# Example 22: ValueError when removing a non-existent item
# fruits.remove('banana')  # ValueError: list.remove(x): x not in list

# Example 23: TypeError when using invalid types
# mixed_list = [1, 2, 'three']
# mixed_list.sort()  # TypeError: '<' not supported between instances of 'str' and 'int'

# Handling exceptions
try:
    fruits.remove('banana')
except ValueError:
    print("Banana not found in fruits 🍌❌")

# 5.2 Tuples in Detail 🔒📜 (Immutable Sequences)

# Example 1: Creating a tuple using parentheses ()
colors = ('red', 'green', 'blue')
# 'colors' is a tuple of color names 🟥🟩🟦

# Example 2: Accessing tuple elements by index 📍
first_color = colors[0]  # 'red'

# Example 3: Negative indexing
last_color = colors[-1]  # 'blue'

# Example 4: Slicing tuples ✂️
sub_colors = colors[0:2]  # ('red', 'green')

# Example 5: Tuple unpacking 📦➡️➡️➡️
point = (10, 20)
x, y = point 
# x is 10, y is 20

# Example 6: Single-element tuple (note the comma)
single_element_tuple = (42,)
# Without comma, it would be just an integer

# Example 7: Immutable nature of tuples
# colors[0] = 'yellow'  # TypeError: 'tuple' object does not support item assignment

# Example 8: Counting occurrences
count_green = colors.count('green')  # 1

# Example 9: Finding index
index_of_blue = colors.index('blue')  # 2

# Example 10: Nested tuples 📦📦
nested_tuple = ((1, 2), (3, 4))
# nested_tuple contains tuples inside

# Example 11: Converting between lists and tuples
colors_list = list(colors)  # Converts tuple to list
colors_tuple = tuple(colors_list)  # Converts list back to tuple

# Example 12: Checking if an item exists in a tuple
has_red = 'red' in colors  # True

# Example 13: Tuple concatenation
more_colors = ('yellow', 'purple')
all_colors = colors + more_colors
# all_colors is ('red', 'green', 'blue', 'yellow', 'purple')

# Example 14: Tuple of numbers
numbers_tuple = (1, 2, 3, 4)
sum_numbers = sum(numbers_tuple)  # 10

# Example 15: Functions returning multiple values using tuples
def get_min_max(numbers):
    return (min(numbers), max(numbers))  # Returns a tuple

min_num, max_num = get_min_max([5, 2, 9, 1])
# min_num is 1, max_num is 9

# Example 16: Tuples as keys in dictionaries (since tuples are immutable)
coordinates = {}
coordinates[(10, 20)] = 'Point A'
coordinates[(30, 40)] = 'Point B'
# coordinates is a dictionary with tuple keys

# Example 17: Length of a tuple
colors_length = len(colors)  # 3

# Possible mistakes/exceptions:
# Example 18: TypeError when attempting to modify a tuple
try:
    colors[0] = 'orange'
except TypeError:
    print("Cannot modify tuple elements 🔒❌")

# 5.3 Dictionaries in Detail 📒 (Key-Value Mappings)

# Example 1: Creating a dictionary using curly braces {}
person = {'name': 'Alice', 'age': 30}
# 'person' is a dictionary with keys 'name' and 'age'

# Example 2: Accessing values using keys 🔑
person_name = person['name']  # 'Alice'

# Example 3: Adding a new key-value pair
person['city'] = 'New York'
# Now person is {'name': 'Alice', 'age': 30, 'city': 'New York'}

# Example 4: Updating a value
person['age'] = 31  # Update age to 31

# Example 5: Deleting a key-value pair
del person['city']
# Now person is {'name': 'Alice', 'age': 31}

# Example 6: Using get() method to access values safely
salary = person.get('salary', 0)  # Returns 0 if 'salary' key is not found

# Example 7: Keys, Values, and Items
keys = person.keys()      # dict_keys(['name', 'age'])
values = person.values()  # dict_values(['Alice', 31])
items = person.items()    # dict_items([('name', 'Alice'), ('age', 31)])

# Example 8: Iterating over a dictionary
for key, value in person.items():
    print(f"{key}: {value}")

# Example 9: Checking if a key exists
has_name = 'name' in person  # True

# Example 10: Dictionary comprehension ⚡
numbers = [1, 2, 3, 4]
square_dict = {num: num ** 2 for num in numbers}
# square_dict is {1: 1, 2: 4, 3: 9, 4: 16}

# Example 11: Merging dictionaries
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged_dict = {**dict1, **dict2}  # {'a': 1, 'b': 3, 'c': 4}
# Note: 'b' in dict2 overwrites 'b' in dict1

# Example 12: Nested dictionaries 📦📦
student = {
    'name': 'Bob',
    'grades': {'math': 90, 'science': 85}
}

# Access nested value
math_grade = student['grades']['math']  # 90

# Example 13: Updating dictionaries with update() method
student.update({'age': 20})
# Now student has an 'age' key

# Example 14: Removing a key with pop()
age = student.pop('age')  # Removes 'age' key and returns the value 20

# Example 15: Clearing a dictionary
student.clear()  # Now student is {}

# Example 16: Using immutable types as keys (e.g., tuple)
locations = { (40.7128, -74.0060): 'New York', (34.0522, -118.2437): 'Los Angeles' }

# Example 17: Dictionary from sequences
keys = ['name', 'age', 'city']
values = ['Carol', 28, 'Chicago']
person_dict = dict(zip(keys, values))
# person_dict is {'name': 'Carol', 'age': 28, 'city': 'Chicago'}

# Example 18: Fromkeys method
keys = ['a', 'b', 'c']
default_dict = dict.fromkeys(keys, 0)
# default_dict is {'a': 0, 'b': 0, 'c': 0}

# Possible mistakes/exceptions:
# Example 19: KeyError when accessing non-existent key
try:
    salary = person['salary']
except KeyError:
    print("Key 'salary' does not exist 🔑❌")

# Example 20: Using mutable types as keys (not allowed)
# invalid_dict = { [1, 2]: 'list as key' }  # TypeError: unhashable type: 'list'

# 5.4 Sets in Detail 🎒 (Unique Collections)

# Example 1: Creating a set using curly braces {}
numbers_set = {1, 2, 3, 4, 5}
# 'numbers_set' is a set of numbers

# Example 2: Creating a set using set()
empty_set = set()  # Creates an empty set

# Example 3: Sets remove duplicates automatically
duplicates_list = [1, 2, 2, 3, 3, 3]
unique_numbers = set(duplicates_list)  # {1, 2, 3}

# Example 4: Adding an element to a set
numbers_set.add(6)  # Now numbers_set is {1, 2, 3, 4, 5, 6}

# Example 5: Removing an element from a set
numbers_set.remove(3)  # Now numbers_set is {1, 2, 4, 5, 6}
# If 3 is not present, remove() raises KeyError

# Example 6: Discarding an element (won't raise error if not present)
numbers_set.discard(10)  # No error, even though 10 is not in the set

# Example 7: Set union ⋃
set_a = {1, 2, 3}
set_b = {3, 4, 5}
union_set = set_a | set_b  # {1, 2, 3, 4, 5}

# Example 8: Set intersection ⋂
intersection_set = set_a & set_b  # {3}

# Example 9: Set difference ➖
difference_set = set_a - set_b  # {1, 2}

# Example 10: Set symmetric difference ▵
sym_diff_set = set_a ^ set_b  # {1, 2, 4, 5}

# Example 11: Checking membership
has_two = 2 in set_a  # True
has_ten = 10 in set_a  # False

# Example 12: Iterating over a set
for number in numbers_set:
    print(number)

# Example 13: Set comprehension ⚡
squares_set = {num ** 2 for num in range(1, 6)}
# squares_set is {1, 4, 9, 16, 25}

# Example 14: Frozen sets (immutable sets)
frozen_numbers = frozenset([1, 2, 3, 4, 5])
# Cannot add or remove elements from frozen_numbers

# Example 15: Sets of immutable types (cannot have lists or other sets inside)
valid_set = {(1, 2), (3, 4)}  # Set of tuples
# invalid_set = {[1, 2], [3, 4]}  # TypeError: unhashable type: 'list'

# Possible mistakes/exceptions:
# Example 16: KeyError when removing non-existent element
try:
    numbers_set.remove(10)
except KeyError:
    print("Element 10 not found in set ❌")

# Example 17: TypeError when adding mutable types
try:
    numbers_set.add([7, 8])
except TypeError:
    print("Cannot add mutable type (list) to set ❌")

# Example 18: Sets are unordered, cannot access elements by index
# first_element = numbers_set[0]  # TypeError: 'set' object is not subscriptable

# Example 19: Union method
union_method_set = set_a.union(set_b, {6, 7})
# union_method_set is {1, 2, 3, 4, 5, 6, 7}

# Example 20: Clearing a set
numbers_set.clear()  # Now numbers_set is set()

# Additional Examples

# List Example: Using enumerate() with lists
fruits = ['apple', 'banana', 'cherry']
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# Output:
# 0: apple
# 1: banana
# 2: cherry

# Tuple Example: Swapping variables
a = 10
b = 20
a, b = b, a  # Swap values
# Now a is 20, b is 10

# Dictionary Example: Default values with defaultdict
from collections import defaultdict
dd = defaultdict(int)
dd['count'] += 1  # If 'count' doesn't exist, default to 0
# dd is {'count': 1}

# Set Example: Removing duplicates from a list
names = ['Alice', 'Bob', 'Alice', 'Charlie']
unique_names = list(set(names))  # ['Alice', 'Bob', 'Charlie']

# End of Chapter 5 Examples

# This concludes the detailed examples for Lists 📜, Tuples 🔒📜, Dictionaries 📒, and Sets 🎒 in Python.
# Each example has been carefully crafted to help understand the concepts in great depth.
# Remember to handle exceptions and be mindful of common mistakes! 🛡️