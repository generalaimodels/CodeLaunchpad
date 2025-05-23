# Chapter 2: Data - The Building Blocks 🧱 of Python

# ============================================================================
# 2.1 Variables: Containers for Information 📦 (Labeled Boxes)
# ============================================================================

# Variables are like labeled boxes that store data values.

# Example 1: Assigning an integer value to a variable
age = 25  # 'age' stores the integer 25

# Example 2: Assigning a string value to a variable
name = "Alice"  # 'name' stores the string "Alice"

# Example 3: Assigning a float value to a variable
height = 5.7  # 'height' stores the float 5.7

# Example 4: Assigning a boolean value to a variable
is_student = True  # 'is_student' stores the boolean True

# Example 5: Variable naming with underscores
user_name = "Bob"  # Valid variable name using an underscore

# Example 6: Variable names are case-sensitive
City = "New York"  # 'City' is different from 'city'
city = "Los Angeles"

# Example 7: Reassigning a variable to a different data type (Dynamic Typing)
count = 10      # 'count' is initially an integer
count = "Ten"   # Now, 'count' is a string

# Example 8: Variable names cannot start with a number (Invalid)
# 2nd_place = "Silver"  # SyntaxError: invalid syntax

# Correct way:
second_place = "Silver"  # Valid variable name

# Example 9: Using special characters in variable names is not allowed
# total$ = 100  # SyntaxError: invalid syntax

# Correct way:
total_amount = 100  # Valid variable name

# Example 10: Avoiding reserved keywords as variable names
# if = 5  # SyntaxError: invalid syntax because 'if' is a reserved keyword

# Correct way:
if_value = 5  # Valid variable name

# Example 11: Multiple assignments in one line
a = b = c = 0  # 'a', 'b', and 'c' all store 0

# Example 12: Swapping variable values
x = 5
y = 10
x, y = y, x  # Now 'x' is 10 and 'y' is 5

# Example 13: Assigning values to multiple variables in one line
width, height = 1920, 1080  # 'width' stores 1920, 'height' stores 1080

# Example 14: Using variables in expressions
radius = 7
area = 3.14 * radius ** 2  # 'area' calculates the area of a circle

# Example 15: Deleting a variable (uncommon but possible)
temp = 100
del temp  # 'temp' is deleted and no longer exists

# ============================================================================
# 2.2 Data Types: Kinds of Information 📊 (Different Box Types)
# ============================================================================

# Understanding different data types in Python.

# Example 1: Integer (int)
num_apples = 5  # 'num_apples' stores the integer 5

# Example 2: Float (float)
price = 9.99  # 'price' stores the float 9.99

# Example 3: String (str)
greeting = "Hello, World!"  # 'greeting' stores a string

# Example 4: Boolean (bool)
is_open = False  # 'is_open' stores the boolean False

# Example 5: List (list)
fruits = ["apple", "banana", "cherry"]  # 'fruits' stores a list of strings

# Example 6: Accessing list elements
first_fruit = fruits[0]  # 'first_fruit' is "apple"

# Example 7: Modifying a list (mutable)
fruits.append("date")  # Adds "date" to the end of the list

# Example 8: Tuple (tuple)
coordinates = (10.0, 20.0)  # 'coordinates' stores a tuple of floats

# Example 9: Immutable nature of tuples
# coordinates[0] = 15.0  # TypeError: 'tuple' object does not support item assignment

# Example 10: Dictionary (dict)
student = {"name": "Alice", "age": 25}  # 'student' stores a dict with keys and values

# Example 11: Accessing dictionary values
student_name = student["name"]  # 'student_name' is "Alice"

# Example 12: Adding a new key-value pair to a dictionary
student["major"] = "Computer Science"  # Adds a new key 'major'

# Example 13: Set (set)
unique_numbers = {1, 2, 3, 2, 1}  # 'unique_numbers' stores {1, 2, 3}

# Example 14: Adding an element to a set
unique_numbers.add(4)  # 'unique_numbers' is now {1, 2, 3, 4}

# Example 15: NoneType (None)
result = None  # 'result' represents the absence of a value

# ============================================================================
# 2.3 Operators: Performing Actions ➕➖✖️➗ (Action Verbs)
# ============================================================================

# Using operators to manipulate data.

# Arithmetic Operators
# Example 1: Addition
sum_result = 10 + 5  # sum_result is 15

# Example 2: Subtraction
difference = 10 - 5  # difference is 5

# Example 3: Multiplication
product = 10 * 5  # product is 50

# Example 4: Division
quotient = 10 / 5  # quotient is 2.0

# Example 5: Floor Division
floor_div_result = 10 // 3  # floor_div_result is 3

# Example 6: Modulo
remainder = 10 % 3  # remainder is 1

# Example 7: Exponentiation
power = 2 ** 3  # power is 8

# Comparison Operators
# Example 8: Equal to
is_equal = (10 == 5)  # is_equal is False

# Example 9: Not equal to
is_not_equal = (10 != 5)  # is_not_equal is True

# Example 10: Greater than
is_greater = (10 > 5)  # is_greater is True

# Example 11: Less than
is_less = (10 < 5)  # is_less is False

# Example 12: Greater than or equal to
is_greater_equal = (10 >= 5)  # is_greater_equal is True

# Example 13: Less than or equal to
is_less_equal = (10 <= 5)  # is_less_equal is False

# Assignment Operators
# Example 14: Simple assignment
counter = 0  # counter is initialized to 0

# Example 15: Incrementing assignment
counter += 1  # counter is now 1

# Logical Operators
# Example 16: Logical AND
result_and = True and False  # result_and is False

# Example 17: Logical OR
result_or = True or False  # result_or is True

# Example 18: Logical NOT
result_not = not True  # result_not is False

# Membership Operators
# Example 19: 'in' operator
colors = ["red", "green", "blue"]
is_green_in_colors = "green" in colors  # is_green_in_colors is True

# Example 20: 'not in' operator
is_yellow_in_colors = "yellow" not in colors  # is_yellow_in_colors is True

# Identity Operators
# Example 21: 'is' operator
a = [1, 2, 3]
b = a
is_a_b = (a is b)  # is_a_b is True

# Example 22: 'is not' operator
c = [1, 2, 3]
is_a_c = (a is not c)  # is_a_c is True

# Example 23: Difference between '==' and 'is'
list1 = [1, 2, 3]
list2 = [1, 2, 3]
equal_lists = (list1 == list2)  # equal_lists is True
same_identity = (list1 is list2)  # same_identity is False

# Special Operators
# Example 24: Using '+=', '-=', '*=', '/=' operators
number = 10
number += 5  # number is now 15
number -= 3  # number is now 12
number *= 2  # number is now 24
number /= 4  # number is now 6.0

# Example 25: Chaining comparison operators
age = 25
is_young_adult = 18 < age < 30  # is_young_adult is True

# Example 26: Bitwise Operators
# '&' (AND), '|' (OR), '^' (XOR), '<<' (Left shift), '>>' (Right shift)
bitwise_and = 5 & 3  # bitwise_and is 1
bitwise_or = 5 | 3   # bitwise_or is 7
bitwise_xor = 5 ^ 3  # bitwise_xor is 6
left_shift = 5 << 1  # left_shift is 10
right_shift = 5 >> 1  # right_shift is 2

# Example 27: Order of Operations (Operator Precedence)
calculation = 2 + 3 * 4  # calculation is 14, multiplication before addition

# Correcting with parentheses
correct_calculation = (2 + 3) * 4  # correct_calculation is 20

# Example 28: Using operators with different data types
message = "Hello " + "World!"  # message is "Hello World!"

# Example 29: Multiplying strings
repeat_message = "Ha" * 3  # repeat_message is "HaHaHa"

# Example 30: Using modulo with floats
mod_float = 7.5 % 2.5  # mod_float is 0.0

# ============================================================================
# End of Chapter 2 Examples
# ============================================================================