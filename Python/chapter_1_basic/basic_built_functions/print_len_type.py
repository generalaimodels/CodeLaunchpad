#!/usr/bin/env python3
"""
Python Built-in Functions Deep Dive: print(), len(), and type()
This file provides a comprehensive explanation of these essential built-in functions
with examples, edge cases, and best practices following PEP-8 standards.
"""

###############################################################################
# print() function
###############################################################################
# print() writes objects to the text stream file, separated by sep and followed by end.
# 
# Syntax: print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
# 
# Parameters:
# - *objects: Zero or more objects to print (multiple objects accepted)
# - sep: String inserted between objects (default: ' ')
# - end: String appended after the last object (default: '\n')
# - file: File-like object to write to (default: sys.stdout)
# - flush: Whether to force flush the stream (default: False)
#
# Return value: None

# Basic usage - printing a single value
print("Hello, World!")  # Output: Hello, World!

# Printing multiple values (automatically separated by spaces)
print("Python", "is", "powerful")  # Output: Python is powerful

# Using custom separator
print("Python", "is", "powerful", sep="-")  # Output: Python-is-powerful

# Using custom end character (no newline)
print("Hello", end="")  # Output: Hello (without newline)
print("World")  # Output: HelloWorld (notice no space between lines)

# Combining sep and end
print("Step", "1", "2", "3", sep="->", end="!\n")  # Output: Step->1->2->3!

# Printing different data types
print(42)              # Integer: 42
print(3.14159)         # Float: 3.14159
print(True)            # Boolean: True
print([1, 2, 3])       # List: [1, 2, 3]
print({"name": "John"})  # Dictionary: {'name': 'John'}

# Printing variables
name = "Alice"
age = 30
print(name, "is", age, "years old")  # Output: Alice is 30 years old

# Using f-strings (Python 3.6+) with print (modern approach)
print(f"{name} is {age} years old")  # Output: Alice is 30 years old

# Redirecting output to a file
try:
    with open("output.txt", "w") as file:
        print("This will be written to a file", file=file)
        # No visible output (written to file instead)
except IOError:
    # Exception handling for file operations
    print("Error writing to file")

# Handling non-printable objects
class WithoutStr:
    pass

try:
    # All objects can be printed - Python calls str() or repr()
    print(WithoutStr())  # Output: <__main__.WithoutStr object at 0x...>
except Exception as e:
    print(f"Error: {e}")  # This won't execute as all objects are printable

# Printing escape characters
print("New\nLine")    # Output: New (then a newline) Line
print("Tab\tCharacter")  # Output: Tab     Character
print("Backslash: \\")  # Output: Backslash: \

# Print nothing (empty print)
print()  # Outputs just a newline

# Flush parameter example (useful for progress indicators)
import time
import sys

# Without flush, this might not show until buffer is filled
print("Loading", end="")
for _ in range(3):
    time.sleep(0.5)
    print(".", end="", flush=True)  # Forces output without buffer
print(" Done!")

###############################################################################
# len() function
###############################################################################
# len() returns the length (number of items) of an object.
# 
# Syntax: len(object)
# 
# Parameters:
# - object: A sequence (string, list, tuple, range) or collection (dictionary, set)
#
# Return value: An integer representing the number of items

# Strings - counts characters
text = "Hello, Python!"
print(f"Length of '{text}': {len(text)}")  # Output: Length of 'Hello, Python!': 14

# Lists - counts elements
numbers = [1, 2, 3, 4, 5]
print(f"Length of list: {len(numbers)}")  # Output: Length of list: 5

# Tuples - counts elements
coordinates = (10, 20, 30)
print(f"Length of tuple: {len(coordinates)}")  # Output: Length of tuple: 3

# Dictionaries - counts key-value pairs
person = {"name": "Bob", "age": 25, "city": "New York"}
print(f"Length of dictionary: {len(person)}")  # Output: Length of dictionary: 3

# Sets - counts unique elements
unique_numbers = {1, 2, 3, 3, 2, 1, 4}  # Duplicates are automatically removed
print(f"Length of set: {len(unique_numbers)}")  # Output: Length of set: 4

# Ranges - efficiently calculates length without generating all elements
number_range = range(1, 1001)
print(f"Length of range: {len(number_range)}")  # Output: Length of range: 1000

# Bytes and bytearray
byte_data = b'hello'
print(f"Length of bytes: {len(byte_data)}")  # Output: Length of bytes: 5

byte_array = bytearray([65, 66, 67])
print(f"Length of bytearray: {len(byte_array)}")  # Output: Length of bytearray: 3

# Unicode strings - counts code points, not bytes
unicode_string = "こんにちは"  # Japanese "Hello"
print(f"Length of '{unicode_string}': {len(unicode_string)}")  # Output: Length of 'こんにちは': 5

# Custom objects - implementing __len__
class CustomContainer:
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)

custom = CustomContainer([1, 2, 3, 4])
print(f"Length of custom object: {len(custom)}")  # Output: Length of custom object: 4

# Empty containers
print(f"Length of empty string: {len('')}")  # Output: Length of empty string: 0
print(f"Length of empty list: {len([])}")  # Output: Length of empty list: 0
print(f"Length of empty dict: {len({})}")  # Output: Length of empty dict: 0

# Common exceptions with len()

# TypeError: object of type 'int' has no len()
try:
    len(42)
except TypeError as e:
    print(f"Error with len(42): {e}")  # Output: Error with len(42): object of type 'int' has no len()

# TypeError: object of type 'NoneType' has no len()
try:
    len(None)
except TypeError as e:
    print(f"Error with len(None): {e}")  # Output: Error with len(None): object of type 'NoneType' has no len()

# TypeError: object of type 'function' has no len()
try:
    len(print)
except TypeError as e:
    print(f"Error with len(print): {e}")  # Output: Error with len(print): object of type 'function' has no len()

# Performance note: len() is an O(1) operation for built-in types
# It doesn't need to iterate through all elements, it just looks up a stored length value

###############################################################################
# type() function
###############################################################################
# type() returns the type of an object or creates a new type object.
# 
# Syntax 1 (get type): type(object)
# Syntax 2 (create type): type(name, bases, dict)
# 
# Parameters for Syntax 1:
# - object: Any Python object
#
# Parameters for Syntax 2:
# - name: Name of the new type
# - bases: Tuple of base classes
# - dict: Dictionary containing attribute definitions
#
# Return value: The type of the object or a new type object

# Basic usage - getting the type of various objects
print(f"Type of 42: {type(42)}")                  # Output: Type of 42: <class 'int'>
print(f"Type of 3.14: {type(3.14)}")              # Output: Type of 3.14: <class 'float'>
print(f"Type of 'hello': {type('hello')}")        # Output: Type of 'hello': <class 'str'>
print(f"Type of [1, 2, 3]: {type([1, 2, 3])}")    # Output: Type of [1, 2, 3]: <class 'list'>
print(f"Type of (1, 2, 3): {type((1, 2, 3))}")    # Output: Type of (1, 2, 3): <class 'tuple'>
print(f"Type of {{'a': 1}}: {type({'a': 1})}")    # Output: Type of {'a': 1}: <class 'dict'>
print(f"Type of {1, 2, 3}: {type({1, 2, 3})}")    # Output: Type of {1, 2, 3}: <class 'set'>
print(f"Type of True: {type(True)}")              # Output: Type of True: <class 'bool'>
print(f"Type of None: {type(None)}")              # Output: Type of None: <class 'NoneType'>

# Type checking with conditional statements
value = "42"
if type(value) is str:
    print("Value is a string")
    # Output: Value is a string

# Type comparison
print(f"type(42) == int: {type(42) == int}")  # Output: type(42) == int: True

# Getting types of complex objects
class Person:
    def __init__(self, name):
        self.name = name

person = Person("Alice")
print(f"Type of person: {type(person)}")  # Output: Type of person: <class '__main__.Person'>

# Getting types of functions and methods
def example_function():
    pass

print(f"Type of example_function: {type(example_function)}")  # Output: Type of example_function: <class 'function'>

# Getting type of a class
print(f"Type of Person class: {type(Person)}")  # Output: Type of Person class: <class 'type'>

# Using type() for type checking (though isinstance() is usually preferred)
def process_number(num):
    if type(num) is int:
        return f"Processing integer: {num * 2}"
    elif type(num) is float:
        return f"Processing float: {num:.2f}"
    else:
        return "Not a number type"

print(process_number(5))      # Output: Processing integer: 10
print(process_number(3.14))   # Output: Processing float: 3.14
print(process_number("42"))   # Output: Not a number type

# IMPORTANT: isinstance() vs type()
# isinstance() is generally preferred for type checking as it handles inheritance

class Animal:
    pass

class Dog(Animal):
    pass

dog = Dog()
print(f"type(dog) is Dog: {type(dog) is Dog}")          # Output: type(dog) is Dog: True
print(f"type(dog) is Animal: {type(dog) is Animal}")    # Output: type(dog) is Animal: False
print(f"isinstance(dog, Dog): {isinstance(dog, Dog)}")     # Output: isinstance(dog, Dog): True
print(f"isinstance(dog, Animal): {isinstance(dog, Animal)}")  # Output: isinstance(dog, Animal): True

# Advanced usage: Creating a new type dynamically
# This is metaprogramming and used in advanced scenarios

# Creating a class dynamically using type()
DynamicClass = type('DynamicClass', (object,), {
    'greeting': 'Hello',
    'say_hello': lambda self: f"{self.greeting}, dynamic world!"
})

dynamic_instance = DynamicClass()
print(f"Dynamic class type: {type(dynamic_instance)}")  # Output: Dynamic class type: <class '__main__.DynamicClass'>
print(dynamic_instance.say_hello())  # Output: Hello, dynamic world!

# Type of built-in functions
print(f"Type of len function: {type(len)}")  # Output: Type of len function: <class 'builtin_function_or_method'>

# Getting the type name as a string (for logging or display)
num_type = type(42)
type_name = num_type.__name__
print(f"Type name of 42: {type_name}")  # Output: Type name of 42: int

# Type checking limitation with None (use is instead)
value = None
# This is correct but verbose:
if type(value) is type(None):
    print("Value is None")
    
# This is the Pythonic way:
if value is None:
    print("Value is None")  # Output: Value is None