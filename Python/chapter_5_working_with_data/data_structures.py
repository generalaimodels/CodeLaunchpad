# Chapter 5: Working with Data Structures: Lists, Tuples, Dictionaries, Sets 🗂️ (Organized Storage)

# 5.1 Lists in Detail 📜 (Mutable Sequences)

# Example 1: Creating a simple list
fruits = ['apple', 'banana', 'cherry']  # 🍎 Simple list of fruits
print(fruits)

# Example 2: Accessing elements by index
first_fruit = fruits[0]  # 📍 Accessing first element
print(first_fruit)

# Example 3: Negative indexing
last_fruit = fruits[-1]  # 📍 Accessing the last element
print(last_fruit)

# Example 4: Slicing a list
some_fruits = fruits[0:2]  # ✂️ Slicing the first two elements
print(some_fruits)

# Example 5: Changing elements (mutable)
fruits[1] = 'blueberry'  # ✏️ Changing 'banana' to 'blueberry'
print(fruits)

# Example 6: Appending elements
fruits.append('date')  # ➕ Adding 'date' to the end
print(fruits)

# Example 7: Inserting elements
fruits.insert(1, 'blackberry')  # 🛠️ Inserting 'blackberry' at index 1
print(fruits)

# Example 8: Removing elements by value
fruits.remove('apple')  # 🗑️ Removing 'apple' from the list
print(fruits)

# Example 9: Popping elements by index
popped_fruit = fruits.pop(2)  # 🎈 Popping element at index 2
print(f"Popped fruit: {popped_fruit}")
print(fruits)

# Example 10: Extending a list
more_fruits = ['elderberry', 'fig']
fruits.extend(more_fruits)  # 🔗 Extending fruits with more_fruits
print(fruits)

# Example 11: Sorting a list
fruits.sort()  # ↕️ Sorting the list alphabetically
print(fruits)

# Example 12: Reversing a list
fruits.reverse()  # 🔄 Reversing the list order
print(fruits)

# Example 13: List comprehensions
numbers = [1, 2, 3, 4, 5]
squares = [num ** 2 for num in numbers]  # ⚡ Creating list of squares
print(squares)

# Example 14: Nested lists
matrix = [[1, 2], [3, 4]]  # 📦 2x2 matrix as a nested list
print(matrix)

# Example 15: Length of a list
length = len(fruits)  # 📏 Getting the number of elements in fruits
print(f"Number of fruits: {length}")

# 5.2 Tuples in Detail 🔒📜 (Immutable Sequences)

# Example 1: Creating a simple tuple
point = (1, 2)  # 🎯 Tuple representing a point
print(point)

# Example 2: Accessing tuple elements
x = point[0]  # 📍 Accessing first element
y = point[1]  # 📍 Accessing second element
print(f"x: {x}, y: {y}")

# Example 3: Single element tuple
singleton = (42,)  # 🚨 Note the comma, needed for single element
print(singleton)

# Example 4: Tuple without parentheses
coordinates = 3, 4  # 🎯 Parentheses are optional
print(coordinates)

# Example 5: Packing and unpacking
a, b = coordinates  # 📦 Unpacking tuple into variables
print(f"a: {a}, b: {b}")

# Example 6: Immutable nature of tuples
# point[0] = 10  # ❌ This would raise a TypeError
# Uncommenting the above line would cause an error

# Example 7: Tuple methods - index()
animals = ('cat', 'dog', 'bird')
index = animals.index('dog')  # 🔍 Getting index of 'dog'
print(f"Index of 'dog': {index}")

# Example 8: Tuple methods - count()
count = animals.count('cat')  # 🔢 Counting occurrences of 'cat'
print(f"Number of 'cat': {count}")

# Example 9: Nested tuples
nested_tuple = ((1, 2), (3, 4))  # 📦 Tuples within tuples
print(nested_tuple)

# Example 10: Returning multiple values from functions
def divide_remainder(a, b):
    # ✨ Returns quotient and remainder
    return a // b, a % b  # 🎁 Returning a tuple

quotient, remainder = divide_remainder(10, 3)  # 📦 Unpacking result
print(f"Quotient: {quotient}, Remainder: {remainder}")

# Example 11: Tuple of different data types
person = ('Alice', 30, 'Engineer')  # 👩 Name, age, profession
print(person)

# Example 12: Tuples as keys in dictionaries
location = (40.7128, -74.0060)  # 🗺️ Latitude and longitude
city_info = {location: 'New York'}  # 📒 Using tuple as key
print(city_info)

# Example 13: Length of a tuple
length = len(animals)  # 📏 Getting the number of elements in animals
print(f"Number of animals: {length}")

# Example 14: Slicing tuples
sub_tuple = animals[0:2]  # ✂️ Slicing the first two elements
print(sub_tuple)

# Example 15: Conversion between lists and tuples
numbers_list = [1, 2, 3]
numbers_tuple = tuple(numbers_list)  # 🔄 Converting list to tuple
print(numbers_tuple)
new_list = list(numbers_tuple)  # 🔄 Converting tuple back to list
print(new_list)

# 5.3 Dictionaries in Detail 📒 (Key-Value Mappings)

# Example 1: Creating a simple dictionary
person = {'name': 'Bob', 'age': 25}  # 👤 Person's info
print(person)

# Example 2: Accessing values by keys
name = person['name']  # 🔑 Accessing value of 'name' key
print(name)

# Example 3: Adding a new key-value pair
person['profession'] = 'Developer'  # ➕ Adding 'profession'
print(person)

# Example 4: Updating a value
person['age'] = 26  # ✏️ Updating 'age'
print(person)

# Example 5: Removing a key-value pair
del person['age']  # 🗑️ Deleting 'age' key
print(person)

# Example 6: Using get() method
age = person.get('age', 'Unknown')  # 🔍 Getting 'age' or default
print(f"Age: {age}")

# Example 7: Keys method
keys = person.keys()  # 🗝️ Getting all keys
print(keys)

# Example 8: Values method
values = person.values()  # 📄 Getting all values
print(values)

# Example 9: Items method
items = person.items()  # 📝 Getting all key-value pairs
print(items)

# Example 10: Looping through a dictionary
for key, value in person.items():
    print(f"{key}: {value}")  # 🔄 Iterating over items

# Example 11: Dictionary comprehensions
numbers = [1, 2, 3]
square_dict = {num: num ** 2 for num in numbers}  # ⚡ Creating dict
print(square_dict)

# Example 12: Nested dictionaries
employees = {
    'emp1': {'name': 'John', 'age': 30},
    'emp2': {'name': 'Anna', 'age': 28}
}  # 📦 Dictionaries within a dictionary
print(employees)

# Example 13: Checking for key existence
if 'name' in person:
    print("Name is present")  # ✅ Key exists
else:
    print("Name is not present")

# Example 14: Updating dictionary with another
additional_info = {'hobby': 'Painting', 'city': 'Paris'}
person.update(additional_info)  # 🔄 Merging dictionaries
print(person)

# Example 15: Removing all items
person.clear()  # 🧹 Clearing dictionary
print(person)

# 5.4 Sets in Detail 🎒 (Unique Collections)

# Example 1: Creating a set
numbers_set = {1, 2, 3}  # 🔢 Set of numbers
print(numbers_set)

# Example 2: Removing duplicates from a list
numbers_list = [1, 2, 2, 3, 3, 3]
unique_numbers = set(numbers_list)  # 🎒 Removes duplicates
print(unique_numbers)

# Example 3: Adding elements to a set
unique_numbers.add(4)  # ➕ Adding 4
print(unique_numbers)

# Example 4: Removing elements from a set
unique_numbers.remove(2)  # 🗑️ Removing 2
print(unique_numbers)

# Example 5: Discarding elements (doesn't error if not present)
unique_numbers.discard(5)  # 🚫 Safe remove
print(unique_numbers)

# Example 6: Union of sets
set_a = {1, 2, 3}
set_b = {3, 4, 5}
union_set = set_a | set_b  # ⋃ Union of sets
print(union_set)

# Example 7: Intersection of sets
intersection_set = set_a & set_b  # ⋂ Intersection
print(intersection_set)

# Example 8: Difference of sets
difference_set = set_a - set_b  # ➖ Difference (set_a - set_b)
print(difference_set)

# Example 9: Symmetric difference of sets
symmetric_diff_set = set_a ^ set_b  # ▵ Symmetric difference
print(symmetric_diff_set)

# Example 10: Checking membership
if 3 in set_a:
    print("3 is in set_a")  # ✅ Membership test
else:
    print("3 is not in set_a")

# Example 11: Iterating over a set
for num in unique_numbers:
    print(num)  # 🔄 Sets are iterable

# Example 12: Frozen set (immutable set)
frozen = frozenset([1, 2, 3])  # ❄️ Immutable set
print(frozen)

# Example 13: Length of a set
length = len(set_a)  # 📏 Number of elements in set_a
print(f"Length of set_a: {length}")

# Example 14: Set comprehension
squared_set = {num ** 2 for num in range(1, 5)}  # ⚡ Creating set
print(squared_set)

# Example 15: Clearing a set
unique_numbers.clear()  # 🧹 Clearing all elements
print(unique_numbers)