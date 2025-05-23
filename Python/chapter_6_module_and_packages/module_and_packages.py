# Chapter 6: Modules and Packages: Expanding Python's Power 📦 (Toolboxes and Tool Sheds)

# 6.1 What are Modules? Code Libraries 📚 (Toolboxes)

# Note: In practice, modules are separate .py files that you import.
# Since we are writing in a single .py file, we'll simulate module code using comments.

# Example 1: Importing a built-in module
import math  # 🧮 Importing the math module
result = math.sqrt(16)  # ➗ Using sqrt function from math
print(f"The square root of 16 is {result}")

# Example 2: Importing a module with an alias
import datetime as dt  # 🕒 Alias datetime module as dt
current_time = dt.datetime.now()
print(f"The current time is {current_time}")

# Example 3: Importing specific functions from a module
from math import pi, sin  # 🎯 Importing specific functions
print(f"The value of pi is {pi}")
print(f"The sine of pi/2 is {sin(pi/2)}")

# Example 4: Importing all functions from a module (not recommended)
from math import *  # ⚠️ Importing all names (can cause conflicts)
print(f"Cosine of 0 is {cos(0)}")  # Using cos() without module prefix

# Example 5: Creating a custom module (simulated)
# Imagine this is in a separate file named my_module.py
# def greet(name):
#     print(f"Hello, {name}!")

# Using the custom module
# import my_module
# my_module.greet("Alice")  # 👋 Calling greet from my_module

# Example 6: Handling ModuleNotFoundError
try:
    import non_existent_module  # 🚫 This module doesn't exist
except ModuleNotFoundError:
    print("Module not found!")  # ⚠️ Handling the exception

# Example 7: Using sys.path to include module locations
import sys  # ⚙️ System-specific parameters and functions
# sys.path.append('/path/to/your/modules')  # 🛣️ Adding custom path
# Now you can import modules from that directory

# Example 8: Reloading a module
import importlib  # 🔄 Module for importing/reloading
# import my_module
# importlib.reload(my_module)  # ♻️ Reloading my_module after changes

# Example 9: Using __name__ variable
if __name__ == "__main__":
    # 🏃 Code runs when the script is executed directly
    print("This script is running as the main program.")
else:
    # 🧩 Code runs when the script is imported as a module
    print("This script is imported as a module.")

# Example 10: Module with variables
# In my_module.py:
# number = 42
# In main script:
# import my_module
# print(my_module.number)  # 🔢 Accessing a variable from my_module

# Example 11: Module with classes
# In my_module.py:
# class MyClass:
#     def __init__(self, name):
#         self.name = name
#     def greet(self):
#         print(f"Hello, {self.name}!")
# In main script:
# from my_module import MyClass
# obj = MyClass("Bob")  # 👤 Creating an instance
# obj.greet()  # 👋 Calling a method

# Example 12: Importing a module inside a function
def calculate_area(radius):
    import math  # 📦 Importing inside function scope
    area = math.pi * radius ** 2
    return area

area = calculate_area(5)
print(f"Area of the circle: {area}")

# Example 13: Circular imports (avoidance)
# Module A imports Module B, and Module B imports Module A
# ⚠️ This can lead to ImportError
# Solution: Restructure code to avoid circular dependencies

# Example 14: Using dir() to inspect module contents
import random  # 🎲 Random module
print(dir(random))  # 📜 Listing all attributes and methods

# Example 15: Creating a module with __all__ attribute
# In my_module.py:
# __all__ = ['function_a']
# def function_a():
#     pass
# def function_b():
#     pass
# In main script:
# from my_module import *  # Imports only function_a due to __all__

# 6.2 What are Packages? Module Organizers 📦📦 (Tool Sheds)

# Note: Packages are directories containing an __init__.py file.
# Since we are in a single .py file, we'll simulate package structures.

# Example 1: Creating a simple package
# my_package/
#     __init__.py
#     module1.py
# Importing module1 from my_package
# from my_package import module1
# module1.some_function()  # 🛠️ Using function from module1

# Example 2: Importing sub-packages
# my_package/
#     __init__.py
#     sub_package/
#         __init__.py
#         sub_module.py
# Importing sub_module
# from my_package.sub_package import sub_module
# sub_module.another_function()  # 🛠️ Using function from sub_module

# Example 3: Using __init__.py for package initialization
# In __init__.py:
# from .module1 import function_a
# Now you can import directly:
# from my_package import function_a

# Example 4: Absolute vs Relative Imports
# In module within package:
# Absolute import:
# from my_package.module1 import function_a
# Relative import:
# from .module1 import function_a  # 🧭 Relative to current package

# Example 5: Namespaces in packages
# Avoid naming conflicts by properly structuring packages and modules
# 🗂️ Organize code logically

# Example 6: Importing everything from a package
# In __init__.py:
# __all__ = ['module1', 'module2']
# Now import all specified modules:
# from my_package import *

# Example 7: Installing packages with pip (third-party)
# pip install requests  # 🌐 Installing requests package
import requests  # 📨 Now you can import and use it

# Example 8: Handling ImportError
try:
    from my_package import non_existent_module  # 🚫 Does not exist
except ImportError:
    print("Module cannot be imported!")  # ⚠️ Handling the exception

# Example 9: Package Data (non-code files)
# Packages can include data files (e.g., images, configs)
# Access using package resources or pkgutil

# Example 10: Editable installs for development
# pip install -e .  # 📝 Install package in editable mode
# Allows changes to be reflected immediately

# Example 11: Using setup.py for package distribution
# setup.py contains package metadata and installation instructions
# Allows sharing packages via PyPI

# Example 12: __main__.py in packages
# Allows a package to be executable
# python -m my_package  # 🏃 Executes __main__.py

# Example 13: Compiled Python files (.pyc)
# Python caches compiled bytecode in __pycache__ directories
# ⚡ For faster loading

# Example 14: Zipped packages
# Python can import modules from zip archives
# import sys
# sys.path.append('my_archive.zip')  # 👜 Add zip to path

# Example 15: Using virtual environments
# Isolate package installations per project
# python -m venv venv  # 🌐 Create virtual environment
# source venv/bin/activate  # 🔋 Activate environment

# 6.3 Standard Library: Python's Built-in Modules 🔋 (Pre-made Toolboxes)

# Example 1: Using the os module
import os  # 📁 Operating system interface
current_dir = os.getcwd()  # 📂 Get current working directory
print(f"Current directory: {current_dir}")

# Example 2: Using the sys module
import sys  # ⚙️ System-specific parameters and functions
print(f"Python version: {sys.version}")
print(f"Command-line arguments: {sys.argv}")

# Example 3: Using the math module
import math  # 🧮 Math functions
print(f"Factorial of 5: {math.factorial(5)}")

# Example 4: Using the random module
import random  # 🎲 Random number generators
rand_number = random.randint(1, 100)  # 🔢 Random integer between 1 and 100
print(f"Random number: {rand_number}")

# Example 5: Using the datetime module
import datetime  # 📅 Date and time
now = datetime.datetime.now()
print(f"Current date and time: {now}")

# Example 6: Using the json module
import json  # 🔄 JSON encoding and decoding
data = {'name': 'Eve', 'age': 29}
json_str = json.dumps(data)  # 📝 Convert to JSON string
print(f"JSON string: {json_str}")
parsed_data = json.loads(json_str)  # 📝 Parse JSON string
print(f"Parsed data: {parsed_data}")

# Example 7: Using the re module
import re  # 🔍 Regular expressions
pattern = r'\d+'  # 🔢 Matches one or more digits
text = "There are 24 hours in a day."
matches = re.findall(pattern, text)
print(f"Found numbers: {matches}")

# Example 8: Using the time module
import time  # ⏰ Time-related functions
print("Sleeping for 2 seconds...")
time.sleep(2)  # 💤 Pause execution for 2 seconds
print("Done sleeping.")

# Example 9: Using the threading module
import threading  # 🧵 Multi-threading
def worker():
    print("Thread is running")
thread = threading.Thread(target=worker)  # 🎯 Create new thread
thread.start()
thread.join()  # ⏳ Wait for thread to finish
print("Thread has finished")

# Example 10: Using the urllib.request module
from urllib.request import urlopen  # 🌐 Open URLs
response = urlopen('http://www.example.com')
html = response.read()
print("Fetched web page content.")  # 🌍

# Example 11: Using the collections module
from collections import Counter  # 🧮 Specialized container datatypes
data = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
counter = Counter(data)  # 🔢 Count occurrences
print(f"Item counts: {counter}")

# Example 12: Using the itertools module
import itertools  # ♾️ Iterator functions
perms = itertools.permutations([1, 2, 3])  # 🔄 All permutations
print("Permutations of [1, 2, 3]:")
for perm in perms:
    print(perm)

# Example 13: Using the socket module
import socket  # 🛜 Network communications
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print(f"Hostname: {hostname}, IP Address: {ip_address}")

# Example 14: Using the decimal module
from decimal import Decimal, getcontext  # 💰 Fixed-point and floating-point arithmetic
getcontext().prec = 4  # 🔧 Set precision
num = Decimal(1) / Decimal(7)
print(f"Decimal division result: {num}")

# Example 15: Handling exceptions with traceback module
import traceback  # 🐛 Error traceback details
try:
    result = 1 / 0  # 🚫 Division by zero error
except ZeroDivisionError:
    print("An error occurred:")
    traceback.print_exc()  # 📄 Prints the traceback