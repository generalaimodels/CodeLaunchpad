# Chapter 2: Data: The Building Blocks 🧱 of Python

# ===========================
# 2.1 Variables: Containers for Information 📦
# ===========================

# Example 1: Assigning a string to a variable
name = "Alice"       # Variable 'name' stores the string "Alice"

# Example 2: Assigning an integer to a variable
age = 30             # Variable 'age' stores the integer 30

# Example 3: Assigning a boolean to a variable
is_student = False   # Variable 'is_student' stores the boolean value False

# Example 4: Assigning a float to a variable
salary = 75000.50    # Variable 'salary' stores the float 75000.50

# Example 5: Valid variable names using letters, digits, and underscores
user_name = "bob_smith"  # Valid variable name with underscore

# Example 6: Invalid variable name starting with a digit (will cause SyntaxError)
# 1st_place = "John"     # ❌ Invalid: variable names cannot start with a digit

# Correct version:
first_place = "John"     # ✔️ Valid variable name starting with a letter

# Example 7: Variables are case-sensitive
Var = 10
var = 20  # 'Var' and 'var' are different variables

# Example 8: Using reserved keywords as variable names (will cause SyntaxError)
# def = 5           # ❌ Invalid: 'def' is a reserved keyword

# Correct version:
definition = 5      # ✔️ Valid variable name

# Example 9: Multiple assignments in one line
x = y = z = 0       # Assigns 0 to x, y, and z

# Example 10: Assigning multiple variables in one line
a, b, c = 1, 2, 3   # Assigns 1 to a, 2 to b, and 3 to c

# Example 11: Dynamic typing in Python
variable = 10       # Initially an integer
variable = "Ten"    # Now a string; variable type changed dynamically

# Example 12: Modifying variable values
counter = 0         # Start at 0
counter = counter + 1   # Increment counter by 1 (now 1)

# Example 13: Self-assignment operators
counter += 1        # Another way to increment counter by 1 (now 2)

# Example 14: Using variables before assignment (will cause NameError)
# print(value)      # ❌ Error: 'value' is not defined

# Correct version:
value = 100
print(value)        # ✔️ Outputs 100

# Example 15: Deleting a variable
temp = 5
del temp            # 'temp' variable is deleted
# print(temp)       # ❌ Error: 'temp' is not defined

# ===========================
# 2.2 Data Types: Kinds of Information 📊
# ===========================

# Example 1: Integer data type
quantity = 15       # 'quantity' is an integer

# Example 2: Float data type
temperature = 36.6  # 'temperature' is a float

# Example 3: String data type
message = "Hello, Python!"  # 'message' is a string

# Example 4: Boolean data type
is_valid = True     # 'is_valid' is a boolean

# Example 5: List data type (mutable)
colors = ["red", "green", "blue"]   # 'colors' is a list

# Example 6: Modifying a list
colors.append("yellow")   # Adds "yellow" to the list; lists are mutable

# Example 7: Tuple data type (immutable)
coordinates = (10.0, 20.0)    # 'coordinates' is a tuple

# Example 8: Trying to change a tuple (will cause TypeError)
# coordinates[0] = 15.0    # ❌ Error: tuples are immutable

# Example 9: Dictionary data type
student = {"name": "Bob", "age": 22}   # 'student' is a dictionary

# Example 10: Accessing dictionary values
student_name = student["name"]   # Retrieves the value corresponding to "name"

# Example 11: Set data type
unique_ids = {101, 102, 103, 103}   # 'unique_ids' is a set; duplicates are removed

# Example 12: NoneType
result = None    # 'result' represents the absence of a value

# Example 13: Complex numbers
complex_number = 2 + 3j    # 'complex_number' is a complex number

# Example 14: Bytes data type
byte_data = b"Byte data"   # 'byte_data' is a bytes object

# Example 15: Type conversion
num_str = "100"
num_int = int(num_str)     # Converts string to integer; 'num_int' is 100

# Example 16: Checking data types
print(type(num_int))       # Outputs: <class 'int'>

# Example 17: Mutable vs Immutable types
list_example = [1, 2, 3]
list_example[0] = 10       # Lists are mutable; changes 'list_example' to [10, 2, 3]

tuple_example = (1, 2, 3)
# tuple_example[0] = 10    # ❌ Error: tuples are immutable

# Example 18: String indexing
char = message[0]          # 'char' is 'H' (the first character in 'message')

# Example 19: Slicing strings
substring = message[0:5]   # 'substring' is 'Hello'

# Example 20: Lists with mixed data types
mixed_list = [1, "two", 3.0, True]  # List containing different data types

# Example 21: Nested Lists
matrix = [[1, 2], [3, 4]]  # 'matrix' is a list of lists

# Example 22: Accessing Nested Lists
first_row = matrix[0]      # 'first_row' is [1, 2]

# Example 23: Dictionary of Dictionaries
employees = {
    "emp1": {"name": "Alice", "age": 30},
    "emp2": {"name": "Bob", "age": 35}
}                          # Nested dictionaries

# Example 24: Accessing Nested Dictionaries
emp1_name = employees["emp1"]["name"]  # 'emp1_name' is 'Alice'

# Example 25: Set operations
set_a = {1, 2, 3}
set_b = {3, 4, 5}
intersection = set_a & set_b  # Intersection of sets; result is {3}

# ===========================
# 2.3 Operators: Performing Actions ➕➖✖️➗
# ===========================

# Arithmetic Operators

# Example 1: Addition
sum_result = 7 + 3          # sum_result is 10

# Example 2: Subtraction
difference = 7 - 3          # difference is 4

# Example 3: Multiplication
product = 7 * 3             # product is 21

# Example 4: Division
quotient = 7 / 3            # quotient is approximately 2.3333

# Example 5: Floor Division
floor_division = 7 // 3     # floor_division is 2

# Example 6: Modulo (remainder)
remainder = 7 % 3           # remainder is 1

# Example 7: Exponentiation
power = 2 ** 3              # power is 8 (2 raised to the power of 3)

# Example 8: Order of operations
result = 2 + 3 * 4          # result is 14 (multiplication before addition)

# Example 9: Using parentheses to change order
result = (2 + 3) * 4        # result is 20

# Example 10: Dividing by zero (will cause ZeroDivisionError)
# error_result = 10 / 0     # ❌ Error: division by zero

# Comparison Operators

# Example 11: Equal to
is_equal = (5 == 5)         # is_equal is True

# Example 12: Not equal to
is_not_equal = (5 != 3)     # is_not_equal is True

# Example 13: Greater than
is_greater = (5 > 3)        # is_greater is True

# Example 14: Less than
is_less = (5 < 3)           # is_less is False

# Example 15: Greater than or equal to
is_greater_equal = (5 >= 5) # is_greater_equal is True

# Example 16: Less than or equal to
is_less_equal = (3 <= 5)    # is_less_equal is True

# Assignment Operators

# Example 17: Simple assignment
number = 10                 # Assigns 10 to 'number'

# Example 18: Add and assign
number += 5                 # 'number' is now 15

# Example 19: Subtract and assign
number -= 3                 # 'number' is now 12

# Example 20: Multiply and assign
number *= 2                 # 'number' is now 24

# Example 21: Divide and assign
number /= 4                 # 'number' is now 6.0

# Example 22: Modulo and assign
number %= 5                 # 'number' is now 1.0

# Example 23: Floor divide and assign
number = 7
number //= 2                # 'number' is now 3

# Example 24: Exponentiate and assign
number **= 3                # 'number' is now 27

# Logical Operators

# Example 25: Logical AND
logical_and = True and False    # logical_and is False

# Example 26: Logical OR
logical_or = True or False      # logical_or is True

# Example 27: Logical NOT
logical_not = not True          # logical_not is False

# Example 28: Combining logical operators
result = not (True and False) or (False and True)  # result is True

# Membership Operators

# Example 29: 'in' operator
my_list = [1, 2, 3, 4]
is_in_list = 3 in my_list       # is_in_list is True

# Example 30: 'not in' operator
is_not_in_list = 5 not in my_list   # is_not_in_list is True

# Identity Operators

# Example 31: 'is' operator
a_list = [1, 2, 3]
b_list = a_list
print(a_list is b_list)         # Outputs True; both refer to the same object

# Example 32: 'is not' operator
c_list = [1, 2, 3]
print(a_list is not c_list)     # Outputs True; different objects

# Bitwise Operators

# Example 33: Bitwise AND
bitwise_and = 5 & 3             # bitwise_and is 1 (binary 0101 & 0011)

# Example 34: Bitwise OR
bitwise_or = 5 | 3              # bitwise_or is 7 (binary 0101 | 0011)

# Example 35: Bitwise XOR
bitwise_xor = 5 ^ 3             # bitwise_xor is 6 (binary 0101 ^ 0011)

# Example 36: Bitwise NOT
bitwise_not = ~5                # bitwise_not is -6 (inverts bits and adds 1)

# Example 37: Left Shift
left_shift = 5 << 1             # left_shift is 10 (binary 0101 becomes 1010)

# Example 38: Right Shift
right_shift = 5 >> 1            # right_shift is 2 (binary 0101 becomes 0010)

# Avoid common mistakes

# Example 39: Chaining comparison operators
is_between = 3 < 5 < 7          # is_between is True; equivalent to (3 < 5) and (5 < 7)

# Example 40: Using '==' vs '='
# if x = 5:                     # ❌ Error: assignment in conditional
x = 5
if x == 5:
    print("x is 5")             # ✔️ Correct usage of '=='

# Example 41: String concatenation with '+'
greeting = "Hello, " + "World!" # Concatenates strings

# Example 42: TypeError when adding different types
# result = 5 + "3"              # ❌ Error: can't add int and str

# Correct version:
result = 5 + int("3")           # result is 8

# Example 43: Modulo with negative numbers
negative_mod = -7 % 3           # negative_mod is 2

# Example 44: Exponentiation with negative exponents
inverse = 2 ** -1               # inverse is 0.5

# Example 45: Floor division with negative numbers
neg_floor_div = -7 // 3         # neg_floor_div is -3

# Type conversions during operations

# Example 46: Implicit type conversion
mixed_sum = 5 + 3.2             # mixed_sum is 8.2 (int + float results in float)

# Example 47: Explicit type conversion
string_sum = str(5) + "3"       # string_sum is '53' (both operands are strings)

# Avoiding ZeroDivisionError

# Example 48:
numerator = 10
denominator = 0
# safe_division = numerator / denominator   # ❌ Error: division by zero

# Correct version:
if denominator != 0:
    safe_division = numerator / denominator
else:
    safe_division = None    # Handle division by zero appropriately

# Edge cases with floating point numbers

# Example 49:
sum_floats = 0.1 + 0.2       # May not be exactly 0.3 due to floating point precision

# Correct way to compare floating point numbers
import math
is_close = math.isclose(sum_floats, 0.3)   # True if values are close within tolerance

# Example 50: Augmented assignment with unsupported types
# my_string = "Hello"
# my_string += 5       # ❌ Error: can't add int to str

# Correct version:
my_string = "Hello"
my_string += str(5)    # my_string is 'Hello5'