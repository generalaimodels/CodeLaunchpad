# Chapter 6: Modules and Packages 📦
# Expanding Python's Power

# 6.1 What are Modules? Code Libraries 📚 (Toolboxes)

# Modules are files containing Python definitions and statements.
# They help organize code into reusable and manageable sections.

# Example 1: Creating a module (simulated here)
# Let's assume we have a module file named 'my_module.py' with the following content:

# Content of 'my_module.py':
# def greet(name):
#     print(f"Hello, {name}! Welcome to my_module.")

# Since we cannot create separate files here, we'll define the function directly.

def greet(name):
    print(f"Hello, {name}! Welcome to my_module.")

# Example 2: Using the module's function
greet("Alice")  # Output: Hello, Alice! Welcome to my_module.

# Example 3: Importing a built-in module
import math  # Math module 📐

# Using functions from the math module
square_root = math.sqrt(25)  # Calculate square root
print(square_root)  # Output: 5.0

# Example 4: Accessing module content using dot notation
pi_value = math.pi  # Accessing constant pi
print(pi_value)  # Output: 3.141592653589793

# Example 5: Importing specific functions
from math import factorial  # Import only factorial function

result = factorial(5)  # Calculate 5!
print(result)  # Output: 120

# Example 6: Importing multiple functions
from math import sin, cos, tan  # Trigonometric functions

angle = math.radians(90)  # Convert 90 degrees to radians
sin_value = sin(angle)
print(sin_value)  # Output: 1.0

# Example 7: Importing all contents (not recommended)
from math import *  # Imports everything from math module

log_value = log(100, 10)  # Base-10 logarithm
print(log_value)  # Output: 2.0

# Example 8: Using alias for module names
import datetime as dt  # Alias datetime module

current_date = dt.date.today()
print(current_date)  # Output: YYYY-MM-DD

# Example 9: Using alias for imported functions
from math import pow as power  # Alias 'pow' as 'power'

power_value = power(2, 3)  # 2 raised to the power of 3
print(power_value)  # Output: 8.0

# Example 10: Module search path
import sys  # sys module ⚙️

# sys.path contains the list of directories Python looks for modules
print(sys.path)  # Outputs list of paths

# Example 11: Creating a custom module (simulated)
# Assume 'calculator.py' module with basic arithmetic functions

# Content of 'calculator.py':
# def add(a, b):
#     return a + b
# def subtract(a, b):
#     return a - b

# Simulating module functions
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# Using the module's functions
sum_result = add(10, 5)
difference = subtract(10, 5)
print(sum_result)      # Output: 15
print(difference)      # Output: 5

# Example 12: Handling exceptions while importing modules
try:
    import nonexistent_module  # This module does not exist
except ImportError:
    print("Module not found 🚫")

# Example 13: Reloading a module
import importlib  # Module to reload other modules

# Assuming 'my_module' has been updated elsewhere
importlib.reload(sys)  # Reloads the 'sys' module

# Example 14: Avoiding namespace conflicts
from math import sqrt
from cmath import sqrt as csqrt  # Complex square root

real_sqrt = sqrt(9)    # math.sqrt
complex_sqrt = csqrt(-9)  # cmath.sqrt
print(real_sqrt)       # Output: 3.0
print(complex_sqrt)    # Output: 3j

# Example 15: Checking available functions in a module
import random  # random module 🎲

# Use dir() to list attributes of the module
print(dir(random))  # Lists functions and variables in random module

# Example 16: Writing reusable code with modules
# Let's create utility functions in module 'utils.py'
# Content of 'utils.py':
# def is_even(number):
#     return number % 2 == 0

def is_even(number):
    return number % 2 == 0

# Using the utility function
print(is_even(4))  # Output: True
print(is_even(5))  # Output: False

# Example 17: Conditional imports
import platform  # Provides information about the platform

if platform.system() == 'Windows':
    print("Running on Windows 🖥️")
else:
    print("Running on non-Windows system 🐧")

# Example 18: Using __name__ == '__main__'
# This is used to check if the script is run directly or imported
if __name__ == '__main__':
    print("Script is run directly 🚀")
else:
    print("Script is imported as a module 📦")

# Example 19: Importing modules within functions
def calculate():
    import time  # Importing time module inside a function
    print("Calculation starts...")
    time.sleep(1)  # Pause for 1 second
    print("Calculation ends.")

calculate()

# Example 20: Circular imports (Common mistake)
# Module A imports Module B, and Module B imports Module A
# This can lead to ImportError or AttributeError

# In 'module_a.py':
# import module_b
# def function_a():
#     module_b.function_b()

# In 'module_b.py':
# import module_a
# def function_b():
#     module_a.function_a()

# To fix circular imports, restructure code or use lazy imports.

# Example 21: Using __all__ to control import behavior
# In a module 'shapes.py':
# __all__ = ['circle_area']
# def circle_area(r):
#     return 3.14 * r * r
# def square_area(a):
#     return a * a

# From another script:
# from shapes import *
# Now only 'circle_area' is imported, 'square_area' is not.

# Example 22: Importing from a module in a different directory (Common issue)
# Add directory to sys.path
import sys
sys.path.append('/path/to/module_directory')
# Now you can import the module
# import my_custom_module

# Example 23: Importing modules with same name (Name collision)
# Use aliases to differentiate
# import module1 as m1
# import module2 as m2

# Example 24: Built-in modules vs third-party modules
# time is a built-in module
import time  # Built-in module ⏱️

# requests is a third-party module (needs installation)
# import requests  # Install via pip if needed

# Example 25: Compiled Python files (.pyc)
# Python compiles modules to bytecode files with .pyc extension for faster loading
# This process is automatic and usually transparent to the user

# 6.2 What are Packages? Module Organizers 📦📦 (Tool Sheds)

# Packages are directories containing multiple modules and an __init__.py file.

# Example 1: Creating a package structure (simulated)

# my_package/              (Package directory)
#     __init__.py          (Marks directory as a package)
#     module1.py           (Module in package)
#     module2.py           (Another module)
#     sub_package/         (Sub-package)
#         __init__.py
#         sub_module.py

# Since we can't create directories, we'll simulate modules.

# Content of 'my_package/__init__.py':
# (Can be empty or contain package initialization code)

# Content of 'my_package/module1.py':
# def module1_func():
#     print("Function in module1")

# Content of 'my_package/module2.py':
# def module2_func():
#     print("Function in module2")

# Example 2: Importing modules from a package
# Simulate module functions
def module1_func():
    print("Function in module1")

def module2_func():
    print("Function in module2")

# Importing (simulated)
# from my_package import module1
# module1.module1_func()

# Since we can't actually create packages here, we'll just call the functions
module1_func()  # Output: Function in module1
module2_func()  # Output: Function in module2

# Example 3: Importing functions directly
# from my_package.module1 import module1_func
# module1_func()

# Example 4: Using sub-packages
# Content of 'my_package/sub_package/sub_module.py':
# def sub_module_func():
#     print("Function in sub_module")

def sub_module_func():
    print("Function in sub_module")

# Importing from sub-package (simulated)
# from my_package.sub_package import sub_module
# sub_module.sub_module_func()

sub_module_func()  # Output: Function in sub_module

# Example 5: Relative imports within packages
# In 'module2.py' inside 'my_package':
# from .module1 import module1_func  # Import from the same package

# Example 6: __init__.py usage
# The __init__.py file can initialize package-level variables

# Content of 'my_package/__init__.py':
# __all__ = ['module1', 'module2']  # Controls what is imported with *

# Example 7: Importing all modules from a package
# from my_package import *  # Imports modules listed in __all__

# Example 8: Common mistake - forgetting __init__.py
# Without __init__.py, Python won't recognize the directory as a package

# Example 9: Namespaces in packages
# Modules within the same package can have the same names as modules in other packages without conflict

# Example 10: Nested packages (packages within packages)
# Can create deep hierarchical structures for large projects

# Example 11: Using setup.py to create distributable packages
# This is for packaging and distributing your package via tools like pip

# Example 12: Installing packages
# Using pip to install packages from PyPI
# pip install package_name

# Example 13: Virtual environments
# Isolate package installations for different projects
# Commands:
# python -m venv myenv
# source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Example 14: The import statement and __pycache__
# Python caches compiled bytecode in __pycache__ directories for efficiency

# Example 15: Accessing package data files
# Using pkgutil or importlib.resources to access non-code files within packages

# Example 16: Common mistake - circular imports in packages
# Just like modules, packages can suffer from circular imports leading to errors

# Example 17: Using absolute vs relative imports
# Absolute import:
# import my_package.module1
# Relative import:
# from . import module1

# Example 18: Handling exceptions during imports
try:
    from my_package import unknown_module
except ImportError:
    print("Module not found in package 🚫")

# Example 19: __init__.py executing code
# Code in __init__.py is executed when the package is imported
# Can be used to initialize package state

# Example 20: Namespaces packages (PEP 420)
# Packages without __init__.py files, allowing for distributed namespaces

# 6.3 Standard Library: Python's Built-in Modules 🔋 (Pre-made Toolboxes)

# Python's standard library provides a rich set of modules ready to use.

# Example 1: Using the 'os' module 📁
import os  # Operating system interfaces

current_directory = os.getcwd()  # Get current working directory
print(current_directory)

# Example 2: Listing files in a directory
files = os.listdir('.')  # Lists files in current directory
print(files)

# Example 3: Using 'sys' module ⚙️
import sys

print(sys.version)  # Python version
print(sys.platform)  # Platform information

# Example 4: Command-line arguments
# Run script as: python script.py arg1 arg2
print(sys.argv)  # List of command-line arguments

# Example 5: Using 'random' module 🎲
import random

rand_int = random.randint(1, 100)  # Random integer between 1 and 100
print(rand_int)

# Example 6: Random choice from a list
choices = ['apple', 'banana', 'cherry']
fruit = random.choice(choices)
print(fruit)

# Example 7: Using 'datetime' module 📅⏱️
import datetime

current_time = datetime.datetime.now()
print(current_time.strftime("%Y-%m-%d %H:%M:%S"))

# Example 8: Date arithmetic
tomorrow = datetime.date.today() + datetime.timedelta(days=1)
print(tomorrow)

# Example 9: Using 'json' module ⇄ JSON
import json

data = {'name': 'Alice', 'age': 30}
json_str = json.dumps(data)  # Convert to JSON string
print(json_str)

# Parsing JSON string back to Python dict
parsed_data = json.loads(json_str)
print(parsed_data)

# Example 10: Using 're' module 🔍regex
import re

text = "The rain in Spain"
match = re.search(r'\bS\w+', text)  # Finds words starting with 'S'
if match:
    print(match.group())  # Output: Spain

# Example 11: Handling exceptions with try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero! 🚫")

# Example 12: Using 'urllib' module 🌐🔗
import urllib.request

try:
    response = urllib.request.urlopen('http://www.example.com/')
    html = response.read()
    print(html)
except urllib.error.URLError:
    print("Failed to retrieve URL 🚫")

# Example 13: File operations
with open('sample.txt', 'w') as file:
    file.write('Hello, World!')

with open('sample.txt', 'r') as file:
    content = file.read()
    print(content)

# Example 14: Using 'collections' module
from collections import Counter

counts = Counter([1, 2, 2, 3, 3, 3])
print(counts)  # Output: Counter({3: 3, 2: 2, 1: 1})

# Example 15: Using 'itertools' module
import itertools

perms = itertools.permutations([1, 2, 3])
for perm in perms:
    print(perm)

# Example 16: Using 'threading' module
import threading

def print_numbers():
    for i in range(5):
        print(i)

thread = threading.Thread(target=print_numbers)
thread.start()
thread.join()  # Wait for the thread to finish

# Example 17: Using 'logging' module
import logging

logging.basicConfig(level=logging.INFO)
logging.info("This is an info message")

# Example 18: Using 'decimal' module for precise decimal arithmetic
from decimal import Decimal

a = Decimal('0.1')
b = Decimal('0.2')
c = a + b
print(c)  # Output: 0.3

# Example 19: Using 'functools' module
from functools import lru_cache

@lru_cache(maxsize=32)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))  # Output: 55

# Example 20: Using 'math' module 📐
import math

angle = math.radians(45)
print(math.sin(angle))  # Output: 0.7071067811865475

# Example 21: Using 'subprocess' module to run external commands
import subprocess

result = subprocess.run(['echo', 'Hello World'], capture_output=True, text=True)
print(result.stdout)  # Output: Hello World

# Example 22: Using 'hashlib' module for hashing
import hashlib

hash_object = hashlib.sha256(b'Hello World')
hex_dig = hash_object.hexdigest()
print(hex_dig)  # Outputs SHA256 hash

# Example 23: Using 'copy' module for deep and shallow copies
import copy

original = [1, 2, [3, 4]]
shallow_copy = copy.copy(original)
deep_copy = copy.deepcopy(original)

# Example 24: Using 'argparse' module for command-line arguments
import argparse

parser = argparse.ArgumentParser(description='Sample argparse script')
parser.add_argument('echo', help='Echo the string you use here')
args = parser.parse_args()
print(args.echo)

# Example 25: Using 'time' module for performance measurement
import time

start_time = time.time()
# Perform some operations
time.sleep(1)
end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds")

# Possible mistakes/exceptions:
# Example 26: ImportError when module is not found
try:
    import non_existent_module
except ImportError:
    print("Module not found 🚫")

# Example 27: AttributeError accessing non-existent function
try:
    random.non_existent_function()
except AttributeError:
    print("Function not found in module 🚫")

# Example 28: SyntaxError in imported module (common during development)
# Ensure that all modules are syntactically correct to avoid this error.

# This concludes the detailed examples for Modules and Packages in Python.
# We've covered creating and using modules 📚, organizing code with packages 📦📦,
# and utilizing the Standard Library 🔋 to expand Python's power.
# Remember to handle exceptions and be mindful of common mistakes! 🛡️