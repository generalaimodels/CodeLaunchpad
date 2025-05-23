# -*- coding: utf-8 -*-
"""
More Control Flow Tools - Advanced Python Deep Dive

This script provides an in-depth exploration of Python's control flow mechanisms, designed for advanced developers seeking a nuanced understanding.
We move beyond basic syntax to dissect the underlying behavior, performance implications, and idiomatic usage of control flow statements.

This exploration will cover:

    - Conditional execution with `if`, `elif`, and `else` statements, including ternary operators and advanced conditional expressions.
    - Iteration using `for` loops, focusing on iterators, generators, and efficient loop constructs.
    - The versatile `range()` function, revealing its implementation and optimization strategies.
    - Loop control with `break` and `continue`, emphasizing their impact on loop execution and program efficiency.
    - The often-overlooked `else` clause in loops, clarifying its precise semantics and practical applications.
    - The placeholder `pass` statement, demonstrating its role in structural integrity and code scaffolding.
    - (Python 3.10+) Pattern matching with `match` statements, showcasing its power and advanced pattern capabilities.
    - Function definition with `def`, delving into scope, argument passing mechanisms, and advanced function design.

Expect a focus on:

    - Performance optimization within loops and conditional blocks.
    - Pythonic coding styles for control flow, emphasizing readability and maintainability.
    - Comprehensive error handling strategies within control flow structures.
    - Advanced use cases and less commonly known features of each control flow tool.
    - Understanding the underlying interpreter behavior for efficient code design.

Let's delve into mastering Python's control flow with an advanced and critical perspective.
"""

################################################################################
# 4. More Control Flow Tools
################################################################################

print("\n--- 4. More Control Flow Tools ---\n")

################################################################################
# 4.1. if Statements
################################################################################

print("\n--- 4.1. if Statements ---\n")

# `if` statements are fundamental for conditional execution in Python. They allow branching based on the truthiness of an expression.
# Python's `if` structure is straightforward yet powerful, supporting `elif` (else if) and `else` clauses for comprehensive conditional logic.

# --- Basic if statement ---
print("\n--- Basic if statement ---")
x = 10
if x > 0:
    print("x is positive")

# --- if-else statement ---
print("\n--- if-else statement ---")
y = -5
if y > 0:
    print("y is positive")
else:
    print("y is not positive (i.e., zero or negative)")

# --- if-elif-else statement ---
print("\n--- if-elif-else statement ---")
z = 0
if z > 0:
    print("z is positive")
elif z < 0:
    print("z is negative")
else:
    print("z is zero")

# --- Nested if statements ---
print("\n--- Nested if statements ---")
a = 15
if a > 10:
    print("a is greater than 10")
    if a % 2 == 0:
        print("a is also even")
    else:
        print("a is odd")
else:
    print("a is not greater than 10")

# --- Truthiness and Falsiness ---
print("\n--- Truthiness and Falsiness ---")
# Python evaluates various data types as truthy or falsy in conditional contexts.
# Falsy values: False, None, numeric zero (0, 0.0, 0j), empty sequences ('', [], (), {}), empty sets, and objects of classes that define __bool__() or __len__() methods returning False or 0.
# All other values are truthy.

if True: print("True is truthy")
if 1: print("1 is truthy")
if [1, 2]: print("[1, 2] is truthy")
if "hello": print("'hello' is truthy")

if False: print("False is truthy (this will not print)")
if 0: print("0 is truthy (this will not print)")
if []: print("[] is truthy (this will not print)")
if "": print("'' is truthy (this will not print)")
if None: print("None is truthy (this will not print)")

# --- Conditional Expressions (Ternary Operator) ---
print("\n--- Conditional Expressions (Ternary Operator) ---")
# A concise way to write simple if-else statements in a single line.
age = 20
status = "adult" if age >= 18 else "minor"
print(f"Age: {age}, Status: {status}")

# --- Advanced Conditional Expressions and Logic ---
print("\n--- Advanced Conditional Expressions and Logic ---")
# Using boolean operators (and, or, not) for complex conditions.
p = 5
q = 7
if p > 0 and q < 10:
    print("Both conditions are true")

if not (p < 0): # Using 'not' to negate a condition
    print("p is not negative")

# Chained comparisons (Pythonic and efficient)
r = 3
if 0 < r < 5: # Equivalent to (0 < r) and (r < 5), but more readable and potentially optimized
    print("r is between 0 and 5 (exclusive)")

# Handling potential exceptions within if statements:
print("\n--- if Statement Exception Handling ---")
def might_fail():
    raise ValueError("Function failed")
    return True # Never reached

try:
    if might_fail(): # Function call within the condition
        print("Function returned True (this won't be printed)")
    else:
        print("Function returned False (this won't be printed either)")
except ValueError as e:
    print(f"ValueError caught in if condition: {e}")

################################################################################
# 4.2. for Statements
################################################################################

print("\n--- 4.2. for Statements ---\n")

# `for` loops in Python are primarily used for iterating over sequences (like lists, tuples, strings) or other iterable objects.
# Unlike some languages with index-based for loops, Python's `for` loop is a "for-each" loop, emphasizing iteration over elements.

# --- Basic for loop iterating over a list ---
print("\n--- Basic for loop iterating over a list ---")
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)

# --- Iterating over a string (sequence of characters) ---
print("\n--- Iterating over a string ---")
message = "Python"
for char in message:
    print(char, end=" ") # end=" " to print characters on the same line separated by spaces
print() # Newline after printing characters

# --- Iterating over a tuple ---
print("\n--- Iterating over a tuple ---")
coordinates = (10, 20, 30)
for coord in coordinates:
    print(coord)

# --- Iterating over a dictionary (iterates over keys by default) ---
print("\n--- Iterating over a dictionary (keys) ---")
person = {'name': 'Alice', 'age': 30, 'city': 'New York'}
for key in person: # Iterates over keys
    print(key)

# --- Iterating over dictionary items (key-value pairs) ---
print("\n--- Iterating over dictionary items (key-value pairs) ---")
for key, value in person.items(): # .items() method returns key-value pairs as tuples
    print(f"Key: {key}, Value: {value}")

# --- Iterating with index using enumerate() ---
print("\n--- Iterating with index using enumerate() ---")
# `enumerate()` provides both the index and the value during iteration.
for index, fruit in enumerate(fruits):
    print(f"Index: {index}, Fruit: {fruit}")

# --- Nested for loops ---
print("\n--- Nested for loops ---")
colors = ['red', 'green', 'blue']
for color in colors:
    for fruit in fruits:
        print(f"Color: {color}, Fruit: {fruit}")

# --- Iterating over custom iterables (advanced concept - iterators and generators) ---
print("\n--- Iterating over custom iterables (generators) ---")
def count_up_to(limit):
    """Generator function to yield numbers up to a limit."""
    n = 0
    while n <= limit:
        yield n # 'yield' makes this a generator
        n += 1

for number in count_up_to(5):
    print(number)

# --- Performance considerations with for loops ---
print("\n--- Performance considerations with for loops ---")
# For computationally intensive loops, consider:
# - Vectorized operations using NumPy for numerical tasks (significantly faster than Python loops).
# - List comprehensions or generator expressions for concise and often faster iterations compared to explicit for loops in simple cases.
# - Just-In-Time (JIT) compilation using libraries like Numba to accelerate loop execution.

# --- Handling potential exceptions within for loops ---
print("\n--- for loop Exception Handling ---")
data_list = [1, 2, 'invalid', 4, 5]
for item in data_list:
    try:
        result = item * 2 # Potential TypeError if item is not numeric
        print(f"Item: {item}, Result: {result}")
    except TypeError as e:
        print(f"TypeError encountered for item '{item}': {e}")
    except Exception as e: # Catch-all for other potential exceptions
        print(f"An unexpected error occurred for item '{item}': {e}")

################################################################################
# 4.3. The range() Function
################################################################################

print("\n--- 4.3. The range() Function ---\n")

# `range()` is a built-in function that generates a sequence of numbers. It is commonly used in `for` loops to iterate a specific number of times.
# `range(stop)`: Generates numbers from 0 up to (but not including) `stop`.
# `range(start, stop)`: Generates numbers from `start` up to (but not including) `stop`.
# `range(start, stop, step)`: Generates numbers from `start` to `stop` with a specified increment `step`.

# --- Basic range(stop) ---
print("\n--- Basic range(stop) ---")
for i in range(5): # Generates 0, 1, 2, 3, 4
    print(i)

# --- range(start, stop) ---
print("\n--- range(start, stop) ---")
for j in range(2, 7): # Generates 2, 3, 4, 5, 6
    print(j)

# --- range(start, stop, step) ---
print("\n--- range(start, stop, step) ---")
for k in range(0, 10, 2): # Generates 0, 2, 4, 6, 8
    print(k)

# --- Negative step in range() ---
print("\n--- Negative step in range() ---")
for l in range(10, 0, -1): # Generates 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    print(l)

# --- range() is a sequence type, but memory-efficient ---
print("\n--- range() is memory-efficient ---")
# In Python 3.x, `range()` returns a range object, which is a sequence type that generates numbers on demand, rather than storing them all in memory.
# This makes `range()` very efficient for iterating over large sequences of numbers.
range_object = range(1000000) # Creates a range object, not a list of a million numbers
print(f"Range object: {range_object}, Type: {type(range_object)}")
# Accessing elements of a range is still efficient:
print(f"First element of range: {range_object[0]}")
print(f"Last element of range (calculated on the fly): {range_object[-1]}")

# --- Converting range to a list (for demonstration or when a list is explicitly needed) ---
print("\n--- Converting range to list ---")
range_list = list(range(5)) # Explicitly create a list from the range
print(f"Range converted to list: {range_list}, Type: {type(range_list)}")

# --- Using range() with len() to iterate over indices of a sequence ---
print("\n--- range() with len() for index iteration ---")
data_items = ['a', 'b', 'c', 'd']
for index in range(len(data_items)): # Iterate over indices of the list
    print(f"Index: {index}, Item: {data_items[index]}")

# --- Handling potential exceptions with range() ---
print("\n--- range() Exception Handling ---")
try:
    invalid_range = range(10, 5) # Empty range if start >= stop with positive step
    print(f"Invalid range (empty): {list(invalid_range)}") # Will produce an empty list

    invalid_step_range = range(5, 10, 0) # ValueError: range() arg 3 must not be zero
    print(f"Invalid step range (should raise error): {list(invalid_step_range)}") # Not reached
except ValueError as e:
    print(f"ValueError encountered with range(): {e}")

################################################################################
# 4.4. break and continue Statements
################################################################################

print("\n--- 4.4. break and continue Statements ---\n")

# `break` and `continue` statements are used to alter the normal flow of loop execution.
# `break`: Terminates the loop execution immediately and jumps to the statement following the loop.
# `continue`: Skips the rest of the current iteration and proceeds to the next iteration of the loop.

# --- break statement ---
print("\n--- break statement ---")
for number in range(10):
    if number == 5:
        break # Terminate the loop when number is 5
    print(f"Number: {number}")
print("Loop terminated using break") # Executed after break

# --- continue statement ---
print("\n--- continue statement ---")
for number in range(10):
    if number % 2 == 0:
        continue # Skip even numbers and proceed to the next iteration
    print(f"Odd Number: {number}") # Only odd numbers will be printed
print("Loop continued using continue") # Executed after loop completion

# --- break in nested loops ---
print("\n--- break in nested loops ---")
for i in range(3):
    print(f"Outer loop iteration: {i}")
    for j in range(3):
        if j == 1:
            break # break only exits the inner loop
        print(f"  Inner loop iteration: {j}")
    print("Inner loop finished (due to break or completion)")

# --- continue in nested loops ---
print("\n--- continue in nested loops ---")
for i in range(3):
    print(f"Outer loop iteration: {i}")
    for j in range(3):
        if j == 1:
            continue # continue only skips the current iteration of the inner loop
        print(f"  Inner loop iteration: {j}")
    print("Inner loop finished (due to completion)")

# --- break and continue with while loops ---
print("\n--- break and continue with while loops ---")
count = 0
while count < 10:
    count += 1
    if count == 3:
        continue # Skip iteration when count is 3
    if count == 7:
        break # Terminate loop when count is 7
    print(f"While loop count: {count}")
print("While loop terminated (using break or completion)")

# --- Idiomatic use of break for search loops ---
print("\n--- Idiomatic use of break for search loops ---")
def find_item(items, target):
    """Search for a target item in a list and return True if found, False otherwise."""
    found = False
    for item in items:
        if item == target:
            found = True
            break # Efficiently exit loop once item is found
    return found

my_items = ['apple', 'banana', 'orange', 'grape']
target_item = 'orange'
if find_item(my_items, target_item):
    print(f"'{target_item}' found in the list")
else:
    print(f"'{target_item}' not found in the list")

# --- No specific exception handling directly related to break/continue themselves.
# --- They control loop flow, and exceptions are handled within the loop body as usual.

################################################################################
# 4.5. else Clauses on Loops
################################################################################

print("\n--- 4.5. else Clauses on Loops ---\n")

# Python allows an `else` clause with `for` and `while` loops.
# The `else` block is executed when the loop completes its iterations *without* encountering a `break` statement.
# It's a subtle but powerful feature for scenarios where you need to know if a loop finished naturally or was interrupted.

# --- else with for loop (loop completed without break) ---
print("\n--- else with for loop (no break) ---")
for number in range(5):
    print(f"Loop number: {number}")
else: # else block executed because the loop completed all iterations
    print("For loop finished successfully (no break)")

# --- else with for loop (loop terminated by break) ---
print("\n--- else with for loop (with break) ---")
for number in range(5):
    print(f"Loop number: {number}")
    if number == 3:
        break # Loop terminated prematurely
else: # else block NOT executed because the loop was broken
    print("This else block will NOT be executed because of break")
print("Code after for loop (regardless of break)")

# --- else with while loop (loop condition becomes false) ---
print("\n--- else with while loop (no break, condition becomes false) ---")
count_while = 0
while count_while < 3:
    print(f"While loop count: {count_while}")
    count_while += 1
else: # else block executed because the while condition became false
    print("While loop finished because condition became false (no break)")

# --- else with while loop (loop terminated by break) ---
print("\n--- else with while loop (with break) ---")
count_while_break = 0
while count_while_break < 5:
    print(f"While loop count: {count_while_break}")
    if count_while_break == 2:
        break # Loop terminated prematurely
    count_while_break += 1
else: # else block NOT executed because the loop was broken
    print("This else block will NOT be executed because of break in while loop")
print("Code after while loop (regardless of break)")

# --- Common use case: Search loop with else for 'not found' scenario ---
print("\n--- Search loop with else for 'not found' ---")
def find_item_with_else(items, target):
    """Search for a target item and use else to indicate if not found."""
    for item in items:
        if item == target:
            print(f"'{target}' found in the list")
            break # Item found, exit loop
    else: # Executed if the loop completes without finding the target and without break
        print(f"'{target}' not found in the list")

my_items_else = ['apple', 'banana', 'orange', 'grape']
find_item_with_else(my_items_else, 'banana') # Item found
find_item_with_else(my_items_else, 'kiwi')   # Item not found - else block executed

# --- No specific exception handling related to loop else clauses themselves.
# --- Exceptions within the loop body are handled normally. The else clause is about loop completion type (break or no break).

################################################################################
# 4.6. pass Statements
################################################################################

print("\n--- 4.6. pass Statements ---\n")

# `pass` is a null operation in Python. When it is executed, nothing happens.
# It is used as a placeholder where syntactically a statement is required, but you don't want to execute any code.
# Common use cases:
# - Empty code blocks (e.g., in if statements, loops, function definitions) during development.
# - Minimal classes or functions that inherit from base classes but don't add new functionality.
# - As a placeholder in abstract methods (in abstract base classes).

# --- pass in if statement ---
print("\n--- pass in if statement ---")
condition = False
if condition:
    pass # Placeholder - code to be added later
else:
    print("Condition is false")

# --- pass in for loop ---
print("\n--- pass in for loop ---")
items_to_process = [1, 2, 3, 4, 5]
for item in items_to_process:
    if item % 2 == 0:
        pass # Placeholder - process even items later
    else:
        print(f"Processing odd item: {item}")

# --- pass in function definition ---
print("\n--- pass in function definition ---")
def empty_function():
    pass # Placeholder - function implementation to be added later

empty_function() # Calling the function does nothing

# --- pass in class definition ---
print("\n--- pass in class definition ---")
class MyMinimalClass:
    pass # Placeholder - class attributes and methods to be added later

instance = MyMinimalClass() # Creating an instance of the minimal class
print(f"Instance of MyMinimalClass: {instance}, Type: {type(instance)}")

# --- pass as a placeholder in exception handling (though generally not recommended for production) ---
print("\n--- pass in exception handling (placeholder - use with caution) ---")
try:
    result = 10 / 0
except ZeroDivisionError:
    pass # Placeholder - handle ZeroDivisionError later (better to log or handle appropriately)
    print("ZeroDivisionError occurred, but pass was used as a placeholder.")
    # In production code, you would typically log the error, raise a custom exception, or provide a default value, not just 'pass'.
else:
    print("Division successful")

# --- pass in abstract methods (in abstract base classes - more advanced OOP) ---
print("\n--- pass in abstract methods (Abstract Base Classes - OOP) ---")
import abc

class AbstractClass(abc.ABC):
    @abc.abstractmethod
    def abstract_method(self):
        """Abstract method - must be implemented in subclasses."""
        pass # pass is essential here to define an abstract method without implementation

class ConcreteClass(AbstractClass):
    def abstract_method(self):
        print("Implementation of abstract_method in ConcreteClass")

concrete_instance = ConcreteClass()
concrete_instance.abstract_method()

# --- pass is purely a syntactic placeholder and doesn't introduce any exceptions itself.
# --- Exceptions within the block where 'pass' is used are handled as they normally would be.

################################################################################
# 4.7. match Statements (Python 3.10+)
################################################################################

print("\n--- 4.7. match Statements (Python 3.10+) ---\n")

# Introduced in Python 3.10, `match` statements provide structured pattern matching, similar to `switch` in other languages but significantly more powerful.
# It allows for complex pattern matching based on the structure and values of data.

# --- Basic match statement with literal patterns ---
print("\n--- Basic match statement with literal patterns ---")
http_status = 404
match http_status:
    case 200:
        print("OK")
    case 400:
        print("Bad Request")
    case 404:
        print("Not Found") # Match found here
    case 500:
        print("Internal Server Error")
    case _: # Wildcard case - default case if no other case matches
        print("Unknown status code")

# --- Matching against multiple literals in a case ---
print("\n--- Matching against multiple literals ---")
command = "delete"
match command:
    case "start" | "run" | "execute": # OR pattern
        print("Starting process")
    case "stop" | "quit" | "exit":
        print("Stopping process")
    case "delete" | "remove":
        print("Deleting data") # Match found here
    case _:
        print("Unknown command")

# --- Matching against variables (capture patterns) ---
print("\n--- Matching against variables (capture patterns) ---")
point = (0, 1)
match point:
    case (0, 0):
        print("Origin")
    case (0, y): # Capture the second element into variable 'y'
        print(f"Y-axis, y={y}") # 'y' is now accessible
    case (x, 0): # Capture the first element into variable 'x'
        print(f"X-axis, x={x}") # 'x' is now accessible
    case (x, y): # Capture both elements
        print(f"Point in plane, x={x}, y={y}")
    case _:
        print("Not a 2D point")

# --- Matching sequences (lists, tuples) ---
print("\n--- Matching sequences (lists, tuples) ---")
data = ["log", "warning", "Disk space low"]
match data:
    case ["log", level, message]: # Match list of length 3, capture elements
        print(f"Log entry - Level: {level}, Message: {message}") # Match found here
    case ["config", setting, value]:
        print(f"Configuration - Setting: {setting}, Value: {value}")
    case _:
        print("Unknown data format")

# --- Matching with conditions (guard clauses) ---
print("\n--- Matching with conditions (guard clauses) ---")
age_group = 25
match age_group:
    case age if age < 18:
        print("Minor")
    case age if 18 <= age < 65: # Guard clause - additional condition
        print("Adult") # Match found here as 18 <= 25 < 65 is True
    case age: # No guard clause - matches any remaining age
        print("Senior")

# --- Matching classes and attributes (class patterns) ---
print("\n--- Matching classes and attributes (class patterns) ---")
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

origin = Point(0, 0)
p1 = Point(1, 5)

def process_point(point_obj):
    match point_obj:
        case Point(x=0, y=0): # Match Point object where x=0 and y=0
            print("Origin point")
        case Point(x=0, y=y): # Match Point object where x=0, capture y
            print(f"Point on Y-axis, y={y}")
        case Point(x=x, y=0): # Match Point object where y=0, capture x
            print(f"Point on X-axis, x={x}")
        case Point(x=x, y=y) if x == y: # Match Point where x == y with guard
            print(f"Point on diagonal, x=y={x}")
        case Point(x=x, y=y): # Match any Point object, capture x and y
            print(f"General point, x={x}, y={y}")
        case _:
            print("Not a Point object")

process_point(origin) # Origin point
process_point(p1)     # General point

# --- OR patterns and AS patterns (more advanced - refer to Python 3.10+ documentation for full details) ---
# --- Exhaustiveness checks (match must cover all possible cases or have a wildcard '_') ---

# --- Exception handling in match statements is standard. Exceptions within the case blocks are caught normally.
# --- Match statements themselves don't inherently raise specific exceptions beyond those raised by the code within the cases.
# --- Ensure your patterns are well-defined and handle potential input variations gracefully, especially in wildcard cases.

################################################################################
# 4.8. Defining Functions
################################################################################

print("\n--- 4.8. Defining Functions ---\n")

# Functions are fundamental building blocks for code modularity and reusability in Python.
# Functions encapsulate blocks of code that perform specific tasks. They can accept arguments (inputs) and return values (outputs).
# Function definition uses the `def` keyword, followed by the function name, parentheses for parameters, and a colon.
# The function body is indented.

# --- Basic function definition with no arguments and no return value ---
print("\n--- Basic function definition (no args, no return) ---")
def greet():
    """This function prints a greeting message.""" # Docstring - function documentation
    print("Hello, world!")

greet() # Calling the function

# --- Function with arguments ---
print("\n--- Function with arguments ---")
def greet_name(name):
    """Greets the person passed in as a parameter."""
    print(f"Hello, {name}!")

greet_name("Alice") # Calling with an argument
greet_name("Bob")

# --- Function with return value ---
print("\n--- Function with return value ---")
def add_numbers(num1, num2):
    """Returns the sum of two numbers."""
    return num1 + num2

sum_result = add_numbers(5, 3)
print(f"Sum: {sum_result}")

# --- Function with default argument values ---
print("\n--- Function with default argument values ---")
def power(base, exponent=2): # 'exponent' has a default value of 2
    """Calculates base to the power of exponent (default exponent is 2)."""
    return base ** exponent

square = power(5) # Uses default exponent (2)
cube = power(3, 3) # Overrides default exponent with 3
print(f"Square of 5: {square}")
print(f"Cube of 3: {cube}")

# --- Function with variable number of arguments (*args - positional arguments) ---
print("\n--- Function with *args (variable positional arguments) ---")
def sum_all(*args):
    """Sums up all the positional arguments passed to the function."""
    total = 0
    for num in args:
        total += num
    return total

sum_1_2_3 = sum_all(1, 2, 3)
sum_many = sum_all(1, 2, 3, 4, 5, 6, 7)
print(f"Sum of 1, 2, 3: {sum_1_2_3}")
print(f"Sum of many numbers: {sum_many}")
print(f"Sum of no numbers: {sum_all()}") # Works even with no arguments

# --- Function with variable number of keyword arguments (**kwargs) ---
print("\n--- Function with **kwargs (variable keyword arguments) ---")
def describe_person(**kwargs):
    """Prints key-value pairs of keyword arguments (person's attributes)."""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

describe_person(name="Charlie", age=35, city="London")
describe_person(occupation="Engineer", skills=["Python", "Java"], experience=10)
describe_person() # Works even with no keyword arguments

# --- Function annotations (type hints - for documentation and static analysis, not enforced at runtime by default) ---
print("\n--- Function annotations (type hints) ---")
def multiply(a: int, b: int) -> int: # Annotations for argument types and return type
    """Multiplies two integers and returns an integer."""
    return a * b

product = multiply(4, 7)
print(f"Product: {product}, Type: {type(product)}")

# --- Function scope (local, global, nonlocal) - important for understanding variable access within functions ---
# --- Lambda functions (anonymous functions) - concise function definitions for simple operations ---
# --- Function decorators (advanced - for modifying function behavior) ---
# --- Generators and function-based iterators (advanced - for memory-efficient iteration) ---
# --- Recursive functions (functions that call themselves) ---

# --- Exception handling within functions ---
print("\n--- Function Exception Handling ---")
def divide(x, y):
    """Divides x by y, handling potential ZeroDivisionError."""
    try:
        result = x / y
        return result
    except ZeroDivisionError as e:
        print(f"Error: Division by zero. {e}")
        return None # Or raise a custom exception, log, etc.

division_result_valid = divide(10, 2)
division_result_invalid = divide(10, 0)
print(f"Valid division result: {division_result_valid}")
print(f"Invalid division result: {division_result_invalid}")

print("\n--- End of More Control Flow Tools ---")