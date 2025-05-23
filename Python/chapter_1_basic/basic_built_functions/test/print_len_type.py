#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Built-in Functions: print(), len(), type()
===========================================================

This module provides comprehensive examples and explanations of three 
fundamental Python built-in functions. Understanding these functions deeply
is essential for writing efficient and effective Python code.
"""

###############################################################################
# 1. print() Function
###############################################################################

"""
The print() function outputs the specified message to the screen or another 
standard output device.

Syntax:
    print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)

Parameters:
    *objects  : Zero or more objects to print (multiple objects separated by commas)
    sep       : String inserted between objects (default: space)
    end       : String appended after the last object (default: newline)
    file      : File-like object to write to (default: sys.stdout)
    flush     : Whether to forcibly flush the stream (default: False)
"""

# Basic usage
print("Hello, World!")  # Outputs: Hello, World!

# Printing multiple items
print("Python", "is", "awesome")  # Outputs: Python is awesome

# Using the sep parameter to change separator
print("Python", "is", "awesome", sep="-")  # Outputs: Python-is-awesome

# Using the end parameter to avoid newline
print("Hello", end=" ")
print("World")  # Outputs: Hello World

# Printing variables of different types
name = "Alice"
age = 30
height = 5.7
print(name, age, height)  # Outputs: Alice 30 5.7

# Formatting output with f-strings (Python 3.6+)
print(f"{name} is {age} years old and {height} feet tall")

# Using print with file parameter
import sys
print("This goes to stderr", file=sys.stderr)

# Writing to a file
with open("output.txt", "w") as f:
    print("This will be written to a file", file=f)

# Using flush parameter for immediate output (useful for logs or progress bars)
import time
for i in range(5):
    print(f"Processing {i}...", end="\r", flush=True)
    time.sleep(0.5)
print("Done!            ")  # Extra spaces to overwrite previous line

# Exception handling: print() is very robust and rarely raises exceptions
# However, trying to print non-serializable objects can cause issues
try:
    # Attempting to print a complex object that doesn't have a string representation
    class ComplexObj:
        __slots__ = ['x']  # This restricts the object from having a __dict__
    
    obj = ComplexObj()
    print(obj)  # This will print the object's default string representation
except Exception as e:
    print(f"Exception occurred: {e}")

###############################################################################
# 2. len() Function
###############################################################################

"""
The len() function returns the number of items in an object.

Syntax:
    len(object)

Parameters:
    object : A sequence (string, list, tuple, etc.) or collection (dictionary, set, etc.)

Return Value:
    An integer representing the length of the object
"""

# Length of a string
text = "Python"
print(f"Length of '{text}': {len(text)}")  # Outputs: Length of 'Python': 6

# Length of a list
fruits = ["apple", "banana", "cherry"]
print(f"Number of fruits: {len(fruits)}")  # Outputs: Number of fruits: 3

# Length of a tuple
coordinates = (10, 20, 30)
print(f"Dimensions: {len(coordinates)}")  # Outputs: Dimensions: 3

# Length of a dictionary (number of key-value pairs)
person = {"name": "John", "age": 30, "city": "New York"}
print(f"Number of attributes: {len(person)}")  # Outputs: Number of attributes: 3

# Length of a set
unique_numbers = {1, 2, 3, 3, 2, 1}  # Note: duplicates are removed
print(f"Number of unique values: {len(unique_numbers)}")  # Outputs: Number of unique values: 3

# Length of a bytes object
binary_data = b"hello"
print(f"Length of binary data: {len(binary_data)}")  # Outputs: Length of binary data: 5

# Length of a range object
r = range(0, 10, 2)  # [0, 2, 4, 6, 8]
print(f"Length of range: {len(r)}")  # Outputs: Length of range: 5

# Length of custom objects
class CustomCollection:
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)

custom = CustomCollection([1, 2, 3, 4])
print(f"Length of custom collection: {len(custom)}")  # Outputs: Length of custom collection: 4

# Exception cases
try:
    # Integer has no length
    len(42)
except TypeError as e:
    print(f"TypeError: {e}")  # Outputs: TypeError: object of type 'int' has no len()

try:
    # Float has no length
    len(3.14)
except TypeError as e:
    print(f"TypeError: {e}")  # Outputs: TypeError: object of type 'float' has no len()

try:
    # None has no length
    len(None)
except TypeError as e:
    print(f"TypeError: {e}")  # Outputs: TypeError: object of type 'NoneType' has no len()

# Creating a custom object without __len__ method
class NoLenObject:
    pass

try:
    obj = NoLenObject()
    len(obj)
except TypeError as e:
    print(f"TypeError: {e}")  # Outputs: TypeError: object of type 'NoLenObject' has no len()

###############################################################################
# 3. type() Function
###############################################################################

"""
The type() function has two different forms:
1. When called with one argument, it returns the type of the object
2. When called with three arguments, it creates a new type object (advanced usage)

Syntax:
    type(object)
    type(name, bases, dict)

Parameters:
    object : The object whose type is to be returned
    name   : Name of the class to create
    bases  : Tuple of base classes
    dict   : Dictionary containing methods and attributes

Return Value:
    The type of the object, or a new type object
"""

# Getting the type of various Python objects
print(f"Type of 42: {type(42)}")                 # Outputs: Type of 42: <class 'int'>
print(f"Type of 3.14: {type(3.14)}")             # Outputs: Type of 3.14: <class 'float'>
print(f"Type of 'hello': {type('hello')}")       # Outputs: Type of 'hello': <class 'str'>
print(f"Type of [1, 2, 3]: {type([1, 2, 3])}")   # Outputs: Type of [1, 2, 3]: <class 'list'>
print(f"Type of (1, 2, 3): {type((1, 2, 3))}")   # Outputs: Type of (1, 2, 3): <class 'tuple'>
print(f"Type of {1, 2, 3}: {type({1, 2, 3})}")   # Outputs: Type of {1, 2, 3}: <class 'set'>
print(f"Type of {{1: 'one'}}: {type({1: 'one'})}")  # Outputs: Type of {1: 'one'}: <class 'dict'>

# Type of functions and methods
def sample_function():
    pass

print(f"Type of sample_function: {type(sample_function)}")  # Outputs: Type of sample_function: <class 'function'>

# Type of custom objects
class Person:
    def __init__(self, name):
        self.name = name

john = Person("John")
print(f"Type of john: {type(john)}")  # Outputs: Type of john: <class '__main__.Person'>

# Using type() for type checking (comparing types)
x = 42
if type(x) is int:
    print("x is an integer")
else:
    print("x is not an integer")

# Better alternative for type checking (using isinstance())
if isinstance(x, int):
    print("x is an integer (checked with isinstance)")

# Note: isinstance() is preferred over type() for type checking because
# it handles inheritance properly

# Advanced: Creating a new type dynamically
# This is metaclass programming and is rarely needed in everyday coding

# Creating a class dynamically using type()
DynamicClass = type('DynamicClass', (), {'x': 10, 'get_x': lambda self: self.x})

# Using the dynamically created class
obj = DynamicClass()
print(f"DynamicClass.x: {DynamicClass.x}")    # Outputs: DynamicClass.x: 10
print(f"obj.get_x(): {obj.get_x()}")          # Outputs: obj.get_x(): 10

# Exception cases for type()
# type() doesn't typically raise exceptions as it can return the type of any object

# However, when used as a class factory, it may raise exceptions
try:
    # Bases must be a tuple
    InvalidClass = type('InvalidClass', 'not a tuple', {})
except TypeError as e:
    print(f"TypeError: {e}")  # Outputs: TypeError: bases must be a tuple

try:
    # Dict must be a dictionary
    InvalidClass = type('InvalidClass', (), "not a dict")
except TypeError as e:
    print(f"TypeError: {e}")  # Outputs: TypeError: type() argument 3 must be dict, not str

###############################################################################
# Performance Considerations
###############################################################################

"""
Performance tips for the discussed built-in functions:

1. print():
   - For large amounts of data, consider writing to a file directly instead of print()
   - Use string concatenation or f-strings for complex formatting instead of multiple print calls
   - Avoid excessive printing in performance-critical code sections

2. len():
   - len() is O(1) (constant time) for most built-in types
   - For custom classes, implement an efficient __len__() method
   - For frequently accessed lengths, consider caching the length value

3. type():
   - isinstance() is generally preferred over type() for type checking as it handles inheritance
   - Avoid extensive type checking in performance-critical loops
   - For checking against multiple types, use isinstance(obj, (type1, type2, ...))
"""

###############################################################################
# Real-world Applications
###############################################################################

# Example: Data validation using type() and len()
def validate_user_data(data):
    """Validate user data using type and length checks."""
    if not isinstance(data, dict):
        raise TypeError("User data must be a dictionary")
    
    # Validate required fields
    required_fields = ["username", "email", "password"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate username length
    if not isinstance(data["username"], str):
        raise TypeError("Username must be a string")
    if len(data["username"]) < 3 or len(data["username"]) > 20:
        raise ValueError("Username must be between 3 and 20 characters")
    
    # Validate email format (simplified)
    if not isinstance(data["email"], str):
        raise TypeError("Email must be a string")
    if "@" not in data["email"] or "." not in data["email"]:
        raise ValueError("Invalid email format")
    
    # Validate password strength
    if not isinstance(data["password"], str):
        raise TypeError("Password must be a string")
    if len(data["password"]) < 8:
        raise ValueError("Password must be at least 8 characters")
    
    return True

# Test the validation function
try:
    user_data = {
        "username": "johndoe",
        "email": "john@example.com",
        "password": "securepass123"
    }
    if validate_user_data(user_data):
        print("User data is valid!")
except (TypeError, ValueError) as e:
    print(f"Validation error: {e}")

# Example: Custom logging function using print()
def log(message, level="INFO", to_file=False):
    """Simple logging function with different output options."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{level}] {timestamp} - {message}"
    
    if to_file:
        with open("app.log", "a") as f:
            print(log_message, file=f)
    else:
        # Different colors for different log levels
        if level == "ERROR":
            # ANSI red
            print(f"\033[91m{log_message}\033[0m")
        elif level == "WARNING":
            # ANSI yellow
            print(f"\033[93m{log_message}\033[0m")
        else:
            print(log_message)

# Test the logging function
log("Application started")
log("Configuration file missing", level="WARNING")
log("Failed to connect to database", level="ERROR")
log("This message goes to the log file", to_file=True)