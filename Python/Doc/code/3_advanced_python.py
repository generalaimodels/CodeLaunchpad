# -*- coding: utf-8 -*-
"""
An Informal Introduction to Python - Deep Dive for Advanced Developers

This script meticulously explores fundamental Python concepts, mirroring the initial chapters of the official Python tutorial.
It's crafted for experienced developers seeking a concise yet technically rich refresher, emphasizing nuances and best practices often glossed over in beginner materials.

We'll delve into using Python as a sophisticated calculator, manipulating numbers, text, and lists with precision and awareness of underlying mechanisms.
Furthermore, we'll lay the groundwork for structured programming, anticipating potential pitfalls and showcasing robust coding habits from the outset.

Throughout this exploration, expect a focus on:

    - Data types and their inherent properties (mutability, immutability, etc.)
    - Operator precedence and associativity, crucial for predictable expression evaluation.
    - Built-in functions and their optimized implementations.
    - Error handling strategies, even in seemingly simple operations, demonstrating defensive programming.
    - Pythonic idioms and efficient coding styles.

Let's embark on this journey to solidify your Python foundations with an advanced perspective.
"""

################################################################################
# 3. An Informal Introduction to Python
################################################################################

print("\n--- 3. An Informal Introduction to Python ---\n")

################################################################################
# 3.1. Using Python as a Calculator
################################################################################

print("\n--- 3.1. Using Python as a Calculator ---\n")

# Python, at its core, functions as a powerful interactive calculator.
# It interprets expressions you type and returns the results.

################################################################################
# 3.1.1. Numbers
################################################################################

print("\n--- 3.1.1. Numbers ---\n")

# Python supports several numerical types: integers (int), floating-point numbers (float), and complex numbers (complex).
# Integer precision is arbitrary in Python 3.x, limited only by available memory. Floats are typically implemented as double-precision.
# Complex numbers have a real and imaginary part, both represented as floats.

# --- Integers (int) ---
print("\n--- Integers (int) ---")
integer_example = 10
print(f"Integer example: {integer_example}, Type: {type(integer_example)}")

# Arithmetic operations:
print("\n--- Integer Arithmetic ---")
print(f"Addition: 5 + 3 = {5 + 3}")
print(f"Subtraction: 10 - 4 = {10 - 4}")
print(f"Multiplication: 6 * 7 = {6 * 7}")
print(f"Division (true division - float result): 15 / 4 = {15 / 4}") # Always returns a float
print(f"Floor Division (integer division - truncates towards negative infinity): 15 // 4 = {15 // 4}, -15 // 4 = {-15 // 4}") # Integer result, truncates
print(f"Modulo (remainder): 15 % 4 = {15 % 4}")
print(f"Exponentiation: 2 ** 3 = {2 ** 3}")

# Operator precedence: Follows standard mathematical conventions (PEMDAS/BODMAS)
print("\n--- Operator Precedence ---")
print(f"Precedence: 5 + 3 * 2 = {5 + 3 * 2} (Multiplication before addition)")
print(f"Parentheses for grouping: (5 + 3) * 2 = {(5 + 3) * 2} (Parentheses override precedence)")

# Handling potential exceptions with integers:
print("\n--- Integer Exception Handling ---")
try:
    large_integer = 2**1000  # Python integers can be arbitrarily large
    print(f"Large integer calculation successful: {large_integer}")
except OverflowError as e: # OverflowError is practically non-existent for standard Python integers
    print(f"OverflowError encountered: {e}") # Unlikely to be triggered with Python integers

# --- Floating-point numbers (float) ---
print("\n--- Floating-point numbers (float) ---")
float_example = 3.14159
print(f"Float example: {float_example}, Type: {type(float_example)}")

# Floating-point arithmetic: Similar operators as integers.
print("\n--- Float Arithmetic ---")
print(f"Float Addition: 2.5 + 1.7 = {2.5 + 1.7}")
print(f"Float Division: 7.0 / 2.0 = {7.0 / 2.0}")

# Beware of floating-point representation issues (inherent in binary floating-point):
print("\n--- Floating-point Precision ---")
print(f"Floating-point representation issue: 0.1 + 0.2 = {0.1 + 0.2} (Not exactly 0.3 due to binary representation)")
# For precise decimal arithmetic, consider the `decimal` module.
import decimal
precise_decimal = decimal.Decimal('0.1') + decimal.Decimal('0.2')
print(f"Using decimal for precision: decimal.Decimal('0.1') + decimal.Decimal('0.2') = {precise_decimal}")


# Handling potential exceptions with floats:
print("\n--- Float Exception Handling ---")
try:
    division_by_zero_float = 1.0 / 0.0 # Returns 'inf' (infinity) for floats, not an error in standard operations.
    print(f"Float division by zero (inf): {division_by_zero_float}")

    # However, using float('inf') or float('-inf') in certain operations might lead to specific issues:
    infinity_float = float('inf')
    print(f"Infinity float: {infinity_float}")
    print(f"Infinity + 1: {infinity_float + 1}")
    print(f"Infinity * 2: {infinity_float * 2}")
    print(f"Infinity / Infinity: {infinity_float / infinity_float} (NaN - Not a Number)") # Results in NaN

except ZeroDivisionError as e: # ZeroDivisionError is generally not directly raised by float division by zero in basic operations
    print(f"ZeroDivisionError encountered (unlikely in basic float division): {e}")
except OverflowError as e: # OverflowError can occur in extreme float calculations
    print(f"OverflowError encountered with floats: {e}")
except ValueError as e: # ValueError may occur in specific float related functions
    print(f"ValueError encountered with floats: {e}")


# --- Complex numbers (complex) ---
print("\n--- Complex numbers (complex) ---")
complex_example = 3 + 4j # 'j' or 'J' denotes the imaginary part
print(f"Complex example: {complex_example}, Type: {type(complex_example)}")
print(f"Real part: {complex_example.real}, Imaginary part: {complex_example.imag}")

# Complex number arithmetic:
print("\n--- Complex Arithmetic ---")
complex1 = 2 + 3j
complex2 = 1 - 2j
print(f"Complex Addition: {complex1} + {complex2} = {complex1 + complex2}")
print(f"Complex Multiplication: {complex1} * {complex2} = {complex1 * complex2}")
print(f"Complex Conjugate: Conjugate of {complex1} is {complex1.conjugate()}")

# Handling potential exceptions with complex numbers:
print("\n--- Complex Exception Handling ---")
try:
    complex_division_by_zero = (1+1j) / 0 # ZeroDivisionError will occur if dividing by zero *integer*
    print(f"Complex division by zero: {complex_division_by_zero}") # This will raise ZeroDivisionError

except ZeroDivisionError as e:
    print(f"ZeroDivisionError encountered with complex division by integer zero: {e}")
except TypeError as e: # TypeError can occur if operations are not defined for complex numbers and other types
    print(f"TypeError encountered with complex numbers: {e}")

# Type conversion (casting):
print("\n--- Type Conversion ---")
int_to_float = float(integer_example)
print(f"Integer to float: int({integer_example}) = {int_to_float}, Type: {type(int_to_float)}")
float_to_int = int(float_example) # Truncates towards zero
print(f"Float to integer (truncation): int({float_example}) = {float_to_int}, Type: {type(float_to_int)}")
float_to_int_round = round(float_example) # Rounds to nearest integer (even number in case of 0.5)
print(f"Float to integer (rounding): round({float_example}) = {float_to_int_round}, Type: {type(float_to_int_round)}")
int_to_complex = complex(integer_example)
print(f"Integer to complex: complex({integer_example}) = {int_to_complex}, Type: {type(int_to_complex)}")


################################################################################
# 3.1.2. Text (Strings)
################################################################################

print("\n--- 3.1.2. Text (Strings) ---\n")

# Python excels at string manipulation. Strings are immutable sequences of characters.
# They can be enclosed in single quotes ('...'), double quotes ("..."), or triple quotes ('''...''' or \"\"\"...\"\"\") for multiline strings.

# --- String Literals ---
print("\n--- String Literals ---")
single_quoted_string = 'Hello, Python!'
double_quoted_string = "Hello, Python!"
multiline_string = '''This is a
multiline string
using triple single quotes.'''
multiline_string_double = """This is another
multiline string
using triple double quotes."""

print(f"Single quoted: '{single_quoted_string}'")
print(f"Double quoted: \"{double_quoted_string}\"")
print(f"Multiline (single quotes):\n{multiline_string}")
print(f"Multiline (double quotes):\n{multiline_string_double}")

# --- String Operations ---
print("\n--- String Operations ---")

# Concatenation (+)
string1 = "Python"
string2 = " is awesome"
concatenated_string = string1 + string2
print(f"Concatenation: '{string1}' + '{string2}' = '{concatenated_string}'")

# Repetition (*)
repeated_string = string1 * 3
print(f"Repetition: '{string1}' * 3 = '{repeated_string}'")

# Indexing (accessing individual characters) - zero-based indexing
print("\n--- String Indexing ---")
python_string = "Python"
print(f"String: '{python_string}'")
print(f"First character (index 0): {python_string[0]}")
print(f"Second character (index 1): {python_string[1]}")
print(f"Last character (index -1): {python_string[-1]}") # Negative indices count from the end

# Slicing (extracting substrings) - [start:stop:step]
print("\n--- String Slicing ---")
print(f"Slice [0:2]: {python_string[0:2]} (characters from index 0 up to, but not including, index 2)")
print(f"Slice [2:]: {python_string[2:]} (characters from index 2 to the end)")
print(f"Slice [:3]: {python_string[:3]} (characters from the beginning up to, but not including, index 3)")
print(f"Slice [1:5:2]: {python_string[1:5:2]} (characters from index 1 to 5 with a step of 2)")
print(f"Slice [::-1]: {python_string[::-1]} (reverses the string)") # Common idiom for reversing a string

# Immutability - Strings cannot be changed in place. Operations create new strings.
print("\n--- String Immutability ---")
original_string = "immutable"
# original_string[0] = 'I' # This will raise a TypeError: 'str' object does not support item assignment

modified_string = 'I' + original_string[1:] # Creating a new string instead
print(f"Original string: '{original_string}'")
print(f"Modified string (new string created): '{modified_string}'")


# Escape sequences - Represent special characters within strings
print("\n--- Escape Sequences ---")
escaped_string = "This string has a newline character: \\n and a tab: \\t" # Backslash is the escape character
print(escaped_string)
print("Quoting within strings: \'single quote\' and \"double quote\"") # Escaping quotes within strings of different types

# String methods - Built-in functions to manipulate strings
print("\n--- String Methods ---")
method_string = "  python programming  "
print(f"Original string: '{method_string}'")
print(f"Length (len()): {len(method_string)}")
print(f"Uppercase (upper()): '{method_string.upper()}'")
print(f"Lowercase (lower()): '{method_string.lower()}'")
print(f"Strip whitespace (strip()): '{method_string.strip()}'") # Removes leading/trailing whitespace
print(f"Find substring 'thon' (find()): {method_string.find('thon')}") # Returns index of first occurrence, -1 if not found
print(f"Replace 'python' with 'Java' (replace()): '{method_string.replace('python', 'Java')}'")
print(f"Split into words (split()): {method_string.strip().split()}") # Splits by whitespace by default, returns a list of strings

# Handling potential exceptions with strings:
print("\n--- String Exception Handling ---")
try:
    out_of_range_index = python_string[10] # IndexError: string index out of range
    print(f"Out-of-range index access (should raise error): {out_of_range_index}") # This will not be reached
except IndexError as e:
    print(f"IndexError encountered: {e}")
except TypeError as e: # TypeError can occur with incompatible operations
    print(f"TypeError encountered with strings: {e}")


################################################################################
# 3.1.3. Lists
################################################################################

print("\n--- 3.1.3. Lists ---\n")

# Lists in Python are ordered, mutable sequences of items. They can contain items of different types.
# Lists are defined using square brackets `[...]` and items are separated by commas.

# --- List Creation ---
print("\n--- List Creation ---")
empty_list = []
print(f"Empty list: {empty_list}, Type: {type(empty_list)}")
integer_list = [1, 2, 3, 4, 5]
print(f"Integer list: {integer_list}")
mixed_list = [1, "hello", 3.14, True] # Lists can hold different data types
print(f"Mixed list: {mixed_list}")
nested_list = [integer_list, mixed_list, ["a", "b"]] # Lists can be nested
print(f"Nested list: {nested_list}")

# --- List Operations ---
print("\n--- List Operations ---")

# Indexing and slicing - similar to strings
print("\n--- List Indexing and Slicing ---")
my_list = ['a', 'b', 'c', 'd', 'e']
print(f"List: {my_list}")
print(f"First element (index 0): {my_list[0]}")
print(f"Slice [1:4]: {my_list[1:4]}")
print(f"Slice [-2:]: {my_list[-2:]}")

# Mutability - Lists are mutable, elements can be changed in place
print("\n--- List Mutability ---")
mutable_list = [1, 2, 3]
print(f"Original mutable list: {mutable_list}")
mutable_list[0] = 10 # Modifying the first element
print(f"List after modification: {mutable_list}")
mutable_list[1:3] = [20, 30] # Modifying a slice
print(f"List after slice modification: {mutable_list}")

# List concatenation (+) and repetition (*) - same as strings but operate on lists
print("\n--- List Concatenation and Repetition ---")
list1 = [1, 2]
list2 = [3, 4]
concatenated_list = list1 + list2
print(f"List Concatenation: {list1} + {list2} = {concatenated_list}")
repeated_list = list1 * 2
print(f"List Repetition: {list1} * 2 = {repeated_list}")

# --- List Methods ---
print("\n--- List Methods ---")
method_list = [10, 20, 30]
print(f"Original list: {method_list}")

method_list.append(40) # Adds element to the end of the list
print(f"Append 40 (append()): {method_list}")

method_list.insert(1, 15) # Inserts element at a specific index
print(f"Insert 15 at index 1 (insert()): {method_list}")

method_list.remove(20) # Removes the first occurrence of a value
print(f"Remove 20 (remove()): {method_list}") # ValueError if value not found

popped_element = method_list.pop() # Removes and returns the last element (or element at index if specified)
print(f"Pop last element (pop()): Popped element = {popped_element}, List after pop: {method_list}")

del method_list[0] # Deletes element at a specific index (or slice)
print(f"Delete element at index 0 (del): {method_list}")

list_to_sort = [3, 1, 4, 1, 5, 9, 2, 6]
list_to_sort.sort() # Sorts the list in place (ascending order by default)
print(f"Sorted list (sort()): {list_to_sort}")
list_to_sort.sort(reverse=True) # Sort in descending order
print(f"Sorted list (reverse=True): {list_to_sort}")

list_to_reverse = [1, 2, 3, 4, 5]
list_to_reverse.reverse() # Reverses the list in place
print(f"Reversed list (reverse()): {list_to_reverse}")

count_of_ones = list_to_sort.count(1) # Counts occurrences of a value
print(f"Count of 1s (count()): {count_of_ones}")

index_of_nine = list_to_sort.index(9) # Returns index of the first occurrence of a value
print(f"Index of 9 (index()): {index_of_nine}") # ValueError if value not found

list_length = len(method_list) # Returns the length of the list
print(f"Length (len()): {list_length}")

is_30_in_list = 30 in method_list # Checks if an element is in the list (returns boolean)
print(f"Is 30 in list (in): {is_30_in_list}")
is_100_not_in_list = 100 not in method_list # Checks if an element is NOT in the list
print(f"Is 100 not in list (not in): {is_100_not_in_list}")

# Handling potential exceptions with lists:
print("\n--- List Exception Handling ---")
error_list = [1, 2, 3]
try:
    out_of_range_index_list = error_list[5] # IndexError: list index out of range
    print(f"Out-of-range list index (should raise error): {out_of_range_index_list}") # Not reached
except IndexError as e:
    print(f"IndexError encountered with list indexing: {e}")

try:
    error_list.remove(10) # ValueError: list.remove(x): x not in list
    print(f"Remove non-existent element (should raise error): {error_list}") # Not reached
except ValueError as e:
    print(f"ValueError encountered with list.remove(): {e}")

try:
    index_of_non_existent = error_list.index(10) # ValueError: 10 is not in list
    print(f"Index of non-existent element (should raise error): {index_of_non_existent}") # Not reached
except ValueError as e:
    print(f"ValueError encountered with list.index(): {e}")

try:
    # Incorrect type for list operation (example: trying to add a list to a number)
    # This will depend on the specific operation, in this case, '+' is not defined between list and int
    # result = error_list + 5 # TypeError: can only concatenate list (not "int") to list
    pass # TypeError example commented out as it is context-dependent and less common for basic list operations
except TypeError as e:
    print(f"TypeError encountered with list operations: {e}")


################################################################################
# 3.2. First Steps Towards Programming
################################################################################

print("\n--- 3.2. First Steps Towards Programming ---\n")

# Building upon the calculator analogy, we now introduce fundamental programming concepts: variables, assignment, and basic input/output.
# These are the building blocks for creating more complex and reusable code.

# --- Variables and Assignment ---
print("\n--- Variables and Assignment ---")

# Variables are symbolic names that refer to values. Assignment uses the `=` operator to bind a name to a value.
# Python is dynamically typed, meaning you don't need to explicitly declare the type of a variable. The type is inferred at runtime.

variable_name = "Python is dynamic" # Variable 'variable_name' is assigned a string value
print(f"Variable assignment: variable_name = 'Python is dynamic'")
print(f"Value of variable_name: {variable_name}, Type: {type(variable_name)}")

age = 30 # Variable 'age' is assigned an integer value
print(f"Variable assignment: age = 30")
print(f"Value of age: {age}, Type: {type(age)}")

price = 99.99 # Variable 'price' is assigned a float value
print(f"Variable assignment: price = 99.99")
print(f"Value of price: {price}, Type: {type(price)}")

# Variable naming conventions:
# - Must start with a letter or underscore (_).
# - Can contain letters, numbers, and underscores.
# - Case-sensitive (my_variable and My_variable are different).
# - Avoid using reserved keywords (e.g., if, for, while, print).
# - Use descriptive names (e.g., user_name instead of un).
# - Pythonic style is to use snake_case (lowercase words separated by underscores).

# --- Input and Output ---
print("\n--- Input and Output ---")

# The `print()` function is used to display output to the console. We've used it extensively already.
# The `input()` function is used to get input from the user as a string.

name = input("Please enter your name: ") # Prompts the user to enter their name and stores it in the 'name' variable
print(f"Hello, {name}!") # Using an f-string for formatted output

try:
    user_age_str = input("Please enter your age: ") # Input is always a string
    user_age = int(user_age_str) # Convert the input string to an integer
    print(f"Your age is: {user_age}, Type: {type(user_age)}")
except ValueError as e: # Handle cases where the user enters non-numeric input
    print(f"Invalid input for age. Please enter a number. Error: {e}")

# --- Comments ---
print("\n--- Comments ---")

# Comments are used to explain code and make it more readable.
# In Python, comments start with a hash symbol (#) and continue to the end of the line.

# This is a single-line comment.
# Comments are ignored by the Python interpreter.

'''
This is a multi-line comment
using triple single quotes.
It's often used for docstrings (documentation strings)
but can also be used for general multi-line comments.
'''

"""
This is another multi-line comment
using triple double quotes.
Also commonly used for docstrings.
"""

# --- Basic Program Flow - Sequential Execution ---
print("\n--- Basic Program Flow - Sequential Execution ---")

print("This is the first line of code.")
print("This is the second line of code.")
# Python executes code line by line, from top to bottom, unless control flow statements (like loops and conditionals) are used (which we'll explore later).

print("\n--- End of Informal Introduction ---")