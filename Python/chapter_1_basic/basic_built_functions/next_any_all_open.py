#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Python Built-in Functions Guide
---------------------------------------
This guide covers the powerful built-in functions:
- next()
- any()
- all()
- open()

Each section includes detailed explanations, examples, and edge cases.
"""

##############################################################################
# NEXT() FUNCTION
##############################################################################
"""
The next() function returns the next item from an iterator.

Syntax:
next(iterator, default)

Parameters:
- iterator: Required. An iterator object from which to get the next item
- default: Optional. Value to return if the iterator is exhausted

Return Value:
- The next item from the iterator
- If iterator is exhausted, returns the default value if provided
- If no default value is provided and iterator is exhausted, raises StopIteration
"""

print("====== next() FUNCTION ======")

# Example 1: Basic usage with a list iterator
print("\n# Example 1: Basic usage")
numbers = [10, 20, 30, 40]
iter_numbers = iter(numbers)  # Create an iterator from the list

print(next(iter_numbers))  # Output: 10
print(next(iter_numbers))  # Output: 20
print(next(iter_numbers))  # Output: 30
print(next(iter_numbers))  # Output: 40

# Example 2: Using the default parameter
print("\n# Example 2: Using default parameter")
iter_numbers = iter([1, 2, 3])
print(next(iter_numbers, "End"))  # Output: 1
print(next(iter_numbers, "End"))  # Output: 2
print(next(iter_numbers, "End"))  # Output: 3
print(next(iter_numbers, "End"))  # Output: "End" (iterator exhausted)

# Example 3: StopIteration exception
print("\n# Example 3: StopIteration exception")
iter_numbers = iter([5])
print(next(iter_numbers))  # Output: 5
try:
    print(next(iter_numbers))  # Raises StopIteration
except StopIteration:
    print("StopIteration exception raised!")

# Example 4: Using next() with a custom iterator
print("\n# Example 4: Custom iterator")
class CountDown:
    def __init__(self, start):
        self.start = start
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

countdown = CountDown(3)
print(next(countdown))  # Output: 3
print(next(countdown))  # Output: 2
print(next(countdown))  # Output: 1
try:
    print(next(countdown))  # Raises StopIteration
except StopIteration:
    print("Countdown completed!")

# Example 5: Using next() with generators
print("\n# Example 5: With generators")
def fibonacci_generator(limit):
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1

fib = fibonacci_generator(5)
print(next(fib))  # Output: 0
print(next(fib))  # Output: 1
print(next(fib))  # Output: 1
print(next(fib))  # Output: 2
print(next(fib))  # Output: 3
# print(next(fib))  # Would raise StopIteration


##############################################################################
# ANY() FUNCTION
##############################################################################
"""
The any() function returns True if any element in the iterable is True.
If the iterable is empty, it returns False.

Syntax:
any(iterable)

Parameters:
- iterable: Required. An iterable object (list, tuple, dictionary, etc.)

Return Value:
- True if at least one item in the iterable evaluates to True
- False if all items in the iterable evaluate to False
- False if the iterable is empty
"""

print("\n\n====== any() FUNCTION ======")

# Example 1: Basic usage with lists
print("\n# Example 1: Basic usage with lists")
print(any([True, False, False]))  # Output: True
print(any([False, False, False]))  # Output: False
print(any([]))  # Output: False (empty iterable)

# Example 2: Using with numeric values
print("\n# Example 2: Using with numeric values")
print(any([0, 0, 1, 0]))  # Output: True (1 evaluates to True)
print(any([0, 0, 0, 0]))  # Output: False (all 0s evaluate to False)

# Example 3: Using with strings
print("\n# Example 3: Using with strings")
print(any(["", "", "Hello"]))  # Output: True ("Hello" is non-empty)
print(any(["", "", ""]))  # Output: False (all strings are empty)

# Example 4: Using with conditions in a list comprehension
print("\n# Example 4: Using with list comprehension")
numbers = [1, 2, 3, 4, 5]
print(any(num > 3 for num in numbers))  # Output: True (4 and 5 are > 3)
print(any(num > 10 for num in numbers))  # Output: False (no number > 10)

# Example 5: Practical use case - validation
print("\n# Example 5: Validation use case")
def has_uppercase(password):
    """Check if password has at least one uppercase letter."""
    return any(char.isupper() for char in password)

print(has_uppercase("password"))  # Output: False
print(has_uppercase("Password"))  # Output: True

# Example 6: Working with dictionaries 
print("\n# Example 6: With dictionaries")
# In dictionaries, any() checks the keys by default
dict1 = {0: "False", 1: "True"}
dict2 = {"": "Empty", 0: "Zero"}
print(any(dict1))  # Output: True (has key 1 which is truthy)
print(any(dict2))  # Output: False (all keys are falsy)

# Checking values in a dictionary
print(any(dict1.values()))  # Output: True (all values are non-empty strings)
print(any(value.startswith('T') for value in dict1.values()))  # Output: True


##############################################################################
# ALL() FUNCTION
##############################################################################
"""
The all() function returns True if all elements in the iterable are True.
If the iterable is empty, it returns True.

Syntax:
all(iterable)

Parameters:
- iterable: Required. An iterable object (list, tuple, dictionary, etc.)

Return Value:
- True if all items in the iterable evaluate to True
- True if the iterable is empty
- False if at least one item in the iterable evaluates to False
"""

print("\n\n====== all() FUNCTION ======")

# Example 1: Basic usage
print("\n# Example 1: Basic usage")
print(all([True, True, True]))  # Output: True
print(all([True, False, True]))  # Output: False
print(all([]))  # Output: True (empty iterable)

# Example 2: Using with numeric values
print("\n# Example 2: Using with numeric values")
print(all([1, 2, 3]))  # Output: True (all non-zero numbers are truthy)
print(all([1, 0, 3]))  # Output: False (0 is falsy)

# Example 3: Using with strings
print("\n# Example 3: Using with strings")
print(all(["Hello", "World", "Python"]))  # Output: True (all non-empty)
print(all(["Hello", "", "Python"]))  # Output: False (empty string is falsy)

# Example 4: Using with conditions in a list comprehension
print("\n# Example 4: Using with list comprehension")
numbers = [2, 4, 6, 8, 10]
print(all(num % 2 == 0 for num in numbers))  # Output: True (all even)
numbers = [2, 4, 5, 8, 10]
print(all(num % 2 == 0 for num in numbers))  # Output: False (5 is odd)

# Example 5: Practical use case - validation
print("\n# Example 5: Validation use case")
def validate_user_input(values, min_val=0, max_val=100):
    """Check if all values are within the specified range."""
    return all(min_val <= value <= max_val for value in values)

print(validate_user_input([10, 50, 75]))  # Output: True
print(validate_user_input([10, 150, 75]))  # Output: False (150 > max_val)

# Example 6: Working with dictionaries
print("\n# Example 6: With dictionaries")
# In dictionaries, all() checks the keys by default
dict1 = {1: "One", 2: "Two", 3: "Three"}  # All keys are truthy
dict2 = {0: "Zero", 1: "One", 2: "Two"}   # Key 0 is falsy
print(all(dict1))  # Output: True
print(all(dict2))  # Output: False

# Checking values in a dictionary
print(all(len(val) > 2 for val in dict1.values()))  # Output: True
print(all(val.startswith('T') for val in dict1.values()))  # Output: False


##############################################################################
# OPEN() FUNCTION
##############################################################################
"""
The open() function opens a file and returns a file object.

Syntax:
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)

Common Parameters:
- file: Required. Path and name of the file to open
- mode: Optional. The mode to open the file in (default is 'r' for read)
- encoding: Optional. Encoding to use when reading/writing the file

Common Mode Values:
- 'r': Read (default) - opens file for reading
- 'w': Write - creates a new file or truncates existing file
- 'a': Append - opens file for writing, appending to the end
- 'x': Create - creates a new file, fails if file exists
- 'b': Binary mode
- 't': Text mode (default)
- '+': Update (read and write)

Return Value:
- A file object that can be used to read, write, or manipulate the file

Note: It's important to close files when done with them using the close() method.
Better yet, use the 'with' statement which automatically handles closing.
"""

print("\n\n====== open() FUNCTION ======")

# Example 1: Writing to a file
print("\n# Example 1: Writing to a file")
# Writing text to a file
with open("example.txt", "w", encoding="utf-8") as file:
    file.write("Hello, World!\n")
    file.write("This is a test file.\n")
    file.write("Python is awesome!\n")
print("File written successfully.")

# Example 2: Reading an entire file
print("\n# Example 2: Reading an entire file")
with open("example.txt", "r", encoding="utf-8") as file:
    content = file.read()
    print("File content:")
    print(content)

# Example 3: Reading a file line by line
print("\n# Example 3: Reading a file line by line")
with open("example.txt", "r", encoding="utf-8") as file:
    print("Reading line by line:")
    for line_number, line in enumerate(file, 1):
        print(f"Line {line_number}: {line.strip()}")

# Example 4: Appending to a file
print("\n# Example 4: Appending to a file")
with open("example.txt", "a", encoding="utf-8") as file:
    file.write("This line was appended.\n")
    file.write("File handling in Python is easy!\n")
print("Data appended successfully.")

# Verify the append operation worked
with open("example.txt", "r", encoding="utf-8") as file:
    print("Updated file content:")
    print(file.read())

# Example 5: Reading specific number of characters
print("\n# Example 5: Reading specific characters")
with open("example.txt", "r", encoding="utf-8") as file:
    print("First 10 characters:", file.read(10))
    print("Next 10 characters:", file.read(10))
    # File position is maintained between read operations

# Example 6: Using seek() and tell() methods
print("\n# Example 6: Using seek() and tell()")
with open("example.txt", "r", encoding="utf-8") as file:
    print("Initial position:", file.tell())  # Output: 0
    file.read(5)  # Read and discard 5 characters
    print("After reading 5 chars:", file.tell())  # Output: 5
    file.seek(0)  # Go back to the beginning
    print("After seek(0):", file.tell())  # Output: 0
    print("First line:", file.readline().strip())

# Example 7: Error handling with file operations
print("\n# Example 7: Error handling")
try:
    # Attempting to open a file that doesn't exist
    with open("nonexistent_file.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("Error: The file does not exist!")

try:
    # Attempting to write to a file without proper permissions
    # This is commented out as it's system-dependent
    # with open("/root/restricted.txt", "w") as file:
    #     file.write("Test")
    # For demonstration purposes:
    print("Permission error would occur when writing to protected locations")
except PermissionError:
    print("Error: No permission to write to this location!")

# Example 8: Working with binary files
print("\n# Example 8: Binary mode")
# Writing binary data
with open("binary_example.bin", "wb") as file:
    file.write(b"\x00\x01\x02\x03\x04")
    print("Binary data written")

# Reading binary data
with open("binary_example.bin", "rb") as file:
    binary_data = file.read()
    print("Binary data read:", binary_data)
    # Output bytes representation

# Clean up the example files
import os
try:
    os.remove("example.txt")
    os.remove("binary_example.bin")
    print("\nExample files cleaned up.")
except:
    print("\nCouldn't clean up one or more files.")


"""
SUMMARY OF KEY POINTS:

1. next():
   - Returns the next item from an iterator
   - Can provide a default value to return when the iterator is exhausted
   - Raises StopIteration if iterator is exhausted and no default provided
   - Commonly used with iterators, generators, and custom iterator classes

2. any():
   - Returns True if at least one element in the iterable is True
   - Returns False if all elements are False or if the iterable is empty
   - Evaluates elements for truthiness (not just boolean True/False)
   - Great for validations and checking conditions

3. all():
   - Returns True if all elements in the iterable are True
   - Returns True if the iterable is empty
   - Returns False if at least one element is False
   - Perfect for validating that multiple conditions are all met

4. open():
   - Creates a file object for reading, writing, or appending
   - Supports various modes (r, w, a, x, b, t, +)
   - Best used with 'with' statement to ensure proper file closing
   - Works with text and binary files
   - Includes methods like read(), write(), readline(), seek(), tell()
   - Requires proper error handling for FileNotFoundError, PermissionError, etc.
"""