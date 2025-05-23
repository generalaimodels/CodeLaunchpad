# -*- coding: utf-8 -*-
"""
More on Defining Functions - Advanced Python Function Mastery

This script delves into the intricacies of function definition in Python, targeting advanced developers who seek a comprehensive understanding beyond basic syntax.
We will dissect advanced argument handling, function design principles, and best practices for creating robust and maintainable functions.

This exploration covers:

    - Default Argument Values: Unpacking the nuances of default values and the crucial concept of mutable defaults.
    - Keyword Arguments: Mastering the use of keyword arguments for enhanced readability and flexibility in function calls.
    - Special Parameters: A deep dive into positional-only, keyword-only, and positional-or-keyword arguments, including practical examples and recap.
    - Arbitrary Argument Lists: Leveraging *args for positional and **kwargs for keyword variable arguments.
    - Unpacking Argument Lists: Utilizing * and ** operators to dynamically pass arguments to functions.
    - Lambda Expressions: Exploring anonymous functions for concise operations and functional programming paradigms.
    - Documentation Strings (Docstrings): Crafting effective docstrings for function documentation and API clarity.
    - Function Annotations: Employing type hints for improved code understanding and static analysis benefits.
    - Intermezzo: Coding Style: A brief discussion on Pythonic coding style and adherence to PEP 8 for professional development.

Expect a focus on:

    - Advanced argument parsing and validation techniques.
    - Performance implications of different argument types.
    - Best practices for function design and API creation.
    - Handling complex argument scenarios and edge cases.
    - Pythonic idioms and efficient function implementation.
    - Understanding the underlying mechanisms of function calls in Python.

Let's embark on this journey to master Python function definitions with an advanced and critical eye.
"""

################################################################################
# 4.9. More on Defining Functions
################################################################################

print("\n--- 4.9. More on Defining Functions ---\n")

################################################################################
# 4.9.1. Default Argument Values
################################################################################

print("\n--- 4.9.1. Default Argument Values ---\n")

# Default argument values provide a fallback when arguments are not explicitly passed during a function call.
# They are evaluated *once*, at the point of function definition, not at each call. This behavior is crucial, especially with mutable default values.

# --- Function with a simple default argument (immutable) ---
print("\n--- Simple default argument (immutable) ---")
def increment(number, step=1):
    """Increments a number by a specified step (default step is 1)."""
    return number + step

print(f"increment(5): {increment(5)}")         # Uses default step=1
print(f"increment(10, 3): {increment(10, 3)}") # Overrides default step with 3

# --- The pitfall of mutable default arguments ---
print("\n--- Mutable default argument PITFALL! ---")
def append_to_list(value, my_list=[]): # WARNING: Mutable default argument!
    """Appends a value to a list. Be cautious with mutable defaults."""
    my_list.append(value)
    return my_list

print(f"append_to_list(1): {append_to_list(1)}")     # Initial call, my_list is created as []
print(f"append_to_list(2): {append_to_list(2)}")     # Subsequent call, my_list is the *same* list from the first call!
print(f"append_to_list(3, [100]): {append_to_list(3, [100])}") # Explicitly passing a new list works as expected.
print(f"append_to_list(4): {append_to_list(4)}")     # Continues to modify the *shared* default list.

# Explanation: The default list `[]` is created only once when the function is defined. Subsequent calls without providing `my_list` reuse this *same* list object.
# This is a common source of bugs for developers new to Python.

# --- Correct approach: Using None as default for mutable arguments ---
print("\n--- Correct approach: None as default for mutable arguments ---")
def append_to_list_correct(value, my_list=None):
    """Appends a value to a list. Correctly handles mutable defaults using None."""
    if my_list is None:
        my_list = [] # Create a *new* list if None is passed (or default is used)
    my_list.append(value)
    return my_list

print(f"append_to_list_correct(1): {append_to_list_correct(1)}")
print(f"append_to_list_correct(2): {append_to_list_correct(2)}") # Now each call gets a *new* list by default
print(f"append_to_list_correct(3, [200]): {append_to_list_correct(3, [200])}") # Explicit list still works
print(f"append_to_list_correct(4): {append_to_list_correct(4)}")

# Best practice: For mutable default arguments (lists, dictionaries, sets, etc.), use `None` as the default and create a new mutable object inside the function if the argument is not provided.

################################################################################
# 4.9.2. Keyword Arguments
################################################################################

print("\n--- 4.9.2. Keyword Arguments ---\n")

# Keyword arguments allow you to pass arguments to a function using parameter names.
# This enhances code readability, especially for functions with many parameters, and allows arguments to be passed in any order (for keyword arguments).

# --- Function using keyword arguments ---
print("\n--- Function with keyword arguments ---")
def create_report(name, age, city, occupation="Unknown"): # 'occupation' has a default
    """Creates a report string using keyword arguments."""
    report = f"Name: {name}, Age: {age}, City: {city}, Occupation: {occupation}"
    return report

# Calling with positional arguments (standard way)
report1 = create_report("Alice", 30, "New York")
print(f"Report 1 (positional): {report1}")

# Calling with keyword arguments (more explicit and order-independent for keywords)
report2 = create_report(name="Bob", city="London", age=25) # Order doesn't matter for keyword arguments
print(f"Report 2 (keyword): {report2}")

report3 = create_report("Charlie", 40, "Paris", occupation="Engineer") # Overriding default with keyword
print(f"Report 3 (keyword override): {report3}")

# Mixing positional and keyword arguments (positional arguments must come first)
report4 = create_report("David", 35, city="Berlin", occupation="Artist") # 'name' and 'age' are positional, 'city' and 'occupation' are keyword
print(f"Report 4 (mixed): {report4}")

# --- Keyword arguments improve readability and reduce errors, especially in complex function calls.
# --- They are crucial for functions with optional parameters or a large number of arguments.

# --- Potential TypeError if required positional arguments are missing or incorrectly ordered when mixing.
# --- However, keyword arguments themselves are designed to *prevent* argument order errors for the keyword part.

################################################################################
# 4.9.3. Special parameters
################################################################################

print("\n--- 4.9.3. Special parameters ---\n")

# Python offers special syntax to define the *kind* of arguments a function can accept.
# This provides finer control over function signatures and enhances API design, especially for library functions.
# There are three main kinds of parameters:

# 4.9.3.1. Positional-or-Keyword Arguments
# 4.9.3.2. Positional-Only Parameters
# 4.9.3.3. Keyword-Only Arguments
# 4.9.3.4. Function Examples
# 4.9.3.5. Recap

################################################################################
# 4.9.3.1. Positional-or-Keyword Arguments
################################################################################

print("\n--- 4.9.3.1. Positional-or-Keyword Arguments ---\n")

# These are the *default* kind of parameters in Python functions.
# Arguments can be passed either positionally or as keyword arguments.
# All parameters defined in functions we've seen so far are positional-or-keyword.

# Example (reusing create_report from 4.9.2):
# def create_report(name, age, city, occupation="Unknown"): # All are positional-or-keyword
#     ...

# As demonstrated in 4.9.2, you can call `create_report` using positional arguments, keyword arguments, or a mix of both.

################################################################################
# 4.9.3.2. Positional-Only Parameters
################################################################################

print("\n--- 4.9.3.2. Positional-Only Parameters ---\n")

# Positional-only parameters are specified using a `/` (slash) in the function definition.
# Arguments for positional-only parameters *must* be passed positionally. They cannot be used as keyword arguments.
# Positional-only parameters are useful when:
#   - The parameter names are not intended to be part of the API.
#   - Enforcing argument order is semantically important.
#   - Performance optimization (in some internal CPython implementations, though less relevant in typical Python code).

# --- Function with positional-only parameters ---
print("\n--- Function with positional-only parameters ---")
def positional_only_example(pos_only1, pos_only2, /, pos_or_kw):
    """Example function with positional-only parameters (pos_only1, pos_only2)."""
    return f"pos_only1: {pos_only1}, pos_only2: {pos_only2}, pos_or_kw: {pos_or_kw}"

print(f"positional_only_example(1, 2, 3): {positional_only_example(1, 2, 3)}") # All positional - OK
print(f"positional_only_example(1, 2, pos_or_kw=3): {positional_only_example(1, 2, pos_or_kw=3)}") # pos_or_kw as keyword - OK

try:
    print(f"positional_only_example(pos_only1=1, pos_only2=2, pos_or_kw=3): {positional_only_example(pos_only1=1, pos_only2=2, pos_or_kw=3)}") # pos_only1 and pos_only2 as keywords - TypeError!
except TypeError as e:
    print(f"TypeError: {e}") # TypeError: positional_only_example() got some keyword-only arguments passed as positional-only: 'pos_only1, pos_only2'

# Explanation: `pos_only1` and `pos_only2` are before the `/`, making them positional-only.
# `pos_or_kw` is after the `/` (and before any `*` if keyword-only parameters were present), making it positional-or-keyword.

################################################################################
# 4.9.3.3. Keyword-Only Arguments
################################################################################

print("\n--- 4.9.3.3. Keyword-Only Arguments ---\n")

# Keyword-only arguments are specified *after* a `*` (asterisk) in the function definition.
# Arguments for keyword-only parameters *must* be passed as keyword arguments. They cannot be passed positionally.
# Keyword-only arguments enhance clarity when a function has many optional parameters, forcing explicit naming.

# --- Function with keyword-only arguments ---
print("\n--- Function with keyword-only arguments ---")
def keyword_only_example(arg1, *, kw_only1, kw_only2):
    """Example function with keyword-only parameters (kw_only1, kw_only2)."""
    return f"arg1: {arg1}, kw_only1: {kw_only1}, kw_only2: {kw_only2}"

print(f"keyword_only_example(1, kw_only1=2, kw_only2=3): {keyword_only_example(1, kw_only1=2, kw_only2=3)}") # arg1 positional, kw_only1/2 keyword - OK

try:
    print(f"keyword_only_example(1, 2, 3): {keyword_only_example(1, 2, 3)}") # All positional - TypeError!
except TypeError as e:
    print(f"TypeError: {e}") # TypeError: keyword_only_example() takes 1 positional argument but 3 were given

try:
    print(f"keyword_only_example(arg1=1, kw_only1=2, kw_only2=3): {keyword_only_example(arg1=1, kw_only1=2, kw_only2=3)}") # arg1 as keyword - TypeError!
except TypeError as e:
    print(f"TypeError: {e}") # TypeError: keyword_only_example() got some positional-only arguments passed as keyword arguments: 'arg1'

# Explanation: `kw_only1` and `kw_only2` are after the `*`, making them keyword-only.
# `arg1` is before the `*`, and since there's no `/` before the `*`, it's positional-or-keyword (but in this context, effectively positional since keyword-only parameters are present).

################################################################################
# 4.9.3.4. Function Examples
################################################################################

print("\n--- 4.9.3.4. Function Examples ---\n")

# --- Combined example: Positional-only, positional-or-keyword, and keyword-only ---
print("\n--- Combined parameter types example ---")
def complex_function(pos_only, /, pos_or_kw, *, kw_only="default"):
    """Function with positional-only, positional-or-keyword, and keyword-only parameters."""
    return f"pos_only: {pos_only}, pos_or_kw: {pos_or_kw}, kw_only: {kw_only}"

print(f"complex_function(1, 2, kw_only='custom'): {complex_function(1, 2, kw_only='custom')}") # OK
print(f"complex_function(1, 2): {complex_function(1, 2)}") # OK, uses default for kw_only

try:
    print(f"complex_function(pos_only=1, pos_or_kw=2, kw_only='custom'): {complex_function(pos_only=1, pos_or_kw=2, kw_only='custom')}") # pos_only as keyword - TypeError!
except TypeError as e:
    print(f"TypeError: {e}") # TypeError: complex_function() got some keyword-only arguments passed as positional-only: 'pos_only'

try:
    print(f"complex_function(1, pos_or_kw=2, kw_only='custom'): {complex_function(1, pos_or_kw=2, kw_only='custom')}") # pos_or_kw as keyword - OK (it's positional-or-keyword)
except TypeError as e:
    print(f"TypeError: {e}") # No TypeError

try:
    print(f"complex_function(1, kw_only='custom', pos_or_kw=2): {complex_function(1, kw_only='custom', pos_or_kw=2)}") # Order of keyword args doesn't matter, but positional must be first
except TypeError as e:
    print(f"TypeError: {e}") # No TypeError - Keyword arguments can be in any order after positional ones.

################################################################################
# 4.9.3.5. Recap
################################################################################

print("\n--- 4.9.3.5. Recap ---\n")

# Parameter types recap:

# 1. Positional-or-Keyword (default): Arguments can be passed positionally or by keyword.
# 2. Positional-Only (`/`): Arguments *must* be passed positionally (before the `/`).
# 3. Keyword-Only (`*`): Arguments *must* be passed as keywords (after the `*`).

# Syntax Summary:
# def function_name(pos_only1, pos_only2, /, pos_or_kw1, pos_or_kw2, *, kw_only1="default", kw_only2):
#     ...

# Use cases:
# - Positional-only: For internal parameters or when argument order is semantically crucial.
# - Keyword-only: For optional parameters, improving function call clarity, especially with many options.
# - Positional-or-keyword: The standard, most flexible type for general-purpose functions.

################################################################################
# 4.9.4. Arbitrary Argument Lists
################################################################################

print("\n--- 4.9.4. Arbitrary Argument Lists ---\n")

# Arbitrary argument lists allow functions to accept a variable number of arguments.
# Two main types:
#   - `*args`: For variable positional arguments (passed as a tuple inside the function).
#   - `**kwargs`: For variable keyword arguments (passed as a dictionary inside the function).

# --- Function with *args (variable positional arguments) ---
print("\n--- Function with *args ---")
def print_arguments(*args):
    """Prints all positional arguments passed to the function."""
    print(f"Arguments tuple: {args}, Type: {type(args)}")
    for index, arg in enumerate(args):
        print(f"Argument {index+1}: {arg}")

print_arguments(1, 2, 3, "hello")
print_arguments("a", "b")
print_arguments() # No arguments - args is an empty tuple

# --- Function with **kwargs (variable keyword arguments) ---
print("\n--- Function with **kwargs ---")
def print_keyword_arguments(**kwargs):
    """Prints all keyword arguments passed to the function."""
    print(f"Keyword arguments dictionary: {kwargs}, Type: {type(kwargs)}")
    for key, value in kwargs.items():
        print(f"Keyword: {key}, Value: {value}")

print_keyword_arguments(name="Eve", age=28, city="Sydney")
print_keyword_arguments(param1=100, param2=200)
print_keyword_arguments() # No keyword arguments - kwargs is an empty dictionary

# --- Combining *args and **kwargs ---
print("\n--- Combining *args and **kwargs ---")
def combined_arguments(arg1, arg2, *args, **kwargs):
    """Function accepting positional, variable positional, and variable keyword arguments."""
    print(f"arg1: {arg1}, arg2: {arg2}")
    print(f"*args tuple: {args}")
    print(f"**kwargs dictionary: {kwargs}")

combined_arguments(10, 20, 30, 40, key1="value1", key2="value2")
combined_arguments("first", "second", "extra1", name="John", city="Chicago")

# --- *args and **kwargs are powerful for creating flexible functions, especially for decorators, wrappers, and functions that need to accept arbitrary input.
# --- Argument order is important: positional arguments first, then *args, then keyword arguments, then **kwargs.

################################################################################
# 4.9.5. Unpacking Argument Lists
################################################################################

print("\n--- 4.9.5. Unpacking Argument Lists ---\n")

# Argument unpacking allows you to pass sequences (like lists, tuples) or dictionaries as arguments to functions using `*` and `**` operators.
# This is useful when you have arguments stored in a data structure and want to pass them to a function.

# --- Unpacking lists/tuples with * operator (for positional arguments) ---
print("\n--- Unpacking lists/tuples with * ---")
def calculate_sum_diff(x, y):
    """Calculates sum and difference of two numbers."""
    return x + y, x - y

arguments_tuple = (15, 7)
sum_diff_tuple = calculate_sum_diff(*arguments_tuple) # Unpacking tuple into positional arguments
print(f"Unpacked tuple arguments: {arguments_tuple}, Result: {sum_diff_tuple}")

arguments_list = [20, 5]
sum_diff_list = calculate_sum_diff(*arguments_list) # Unpacking list into positional arguments
print(f"Unpacked list arguments: {arguments_list}, Result: {sum_diff_list}")

# --- Unpacking dictionaries with ** operator (for keyword arguments) ---
print("\n--- Unpacking dictionaries with ** ---")
def create_profile(name, city, age):
    """Creates a profile string using keyword arguments."""
    return f"Name: {name}, City: {city}, Age: {age}"

profile_data = {'name': 'Grace', 'city': 'Seattle', 'age': 32}
profile_string = create_profile(**profile_data) # Unpacking dictionary into keyword arguments
print(f"Unpacked dictionary arguments: {profile_data}, Profile: {profile_string}")

# --- Combining * and ** unpacking ---
print("\n--- Combining * and ** unpacking ---")
def combined_unpacking(a, b, c, name, city):
    """Function using both positional and keyword arguments for unpacking."""
    return f"a: {a}, b: {b}, c: {c}, Name: {name}, City: {city}"

positional_args = [1, 2, 3]
keyword_args_dict = {'name': 'Henry', 'city': 'Austin'}
combined_result = combined_unpacking(*positional_args, **keyword_args_dict)
print(f"Combined unpacking - positional: {positional_args}, keyword: {keyword_args_dict}, Result: {combined_result}")

# --- Unpacking is essential for dynamic function calls and working with data structures that represent function arguments.
# --- Ensure that the unpacked sequence/dictionary matches the function's parameter signature to avoid TypeError.

################################################################################
# 4.9.6. Lambda Expressions
################################################################################

print("\n--- 4.9.6. Lambda Expressions ---\n")

# Lambda expressions (anonymous functions) are small, unnamed functions defined using the `lambda` keyword.
# They are typically used for short, simple operations that can be expressed in a single line.
# Syntax: `lambda arguments: expression`

# --- Basic lambda function: squaring a number ---
print("\n--- Basic lambda expression ---")
square_lambda = lambda x: x**2
print(f"Lambda square(5): {square_lambda(5)}")

# --- Lambda function with multiple arguments: adding three numbers ---
print("\n--- Lambda with multiple arguments ---")
add_three_lambda = lambda x, y, z: x + y + z
print(f"Lambda add_three(2, 3, 4): {add_three_lambda(2, 3, 4)}")

# --- Lambda function with conditional expression ---
print("\n--- Lambda with conditional expression ---")
max_lambda = lambda a, b: a if a > b else b
print(f"Lambda max(10, 5): {max_lambda(10, 5)}")

# --- Using lambda functions with higher-order functions (map, filter, sorted) ---
print("\n--- Lambda with higher-order functions ---")
numbers = [1, 2, 3, 4, 5]

squared_numbers = list(map(lambda x: x**2, numbers)) # Using lambda with map()
print(f"Squared numbers using map(lambda): {squared_numbers}")

even_numbers = list(filter(lambda x: x % 2 == 0, numbers)) # Using lambda with filter()
print(f"Even numbers using filter(lambda): {even_numbers}")

sorted_numbers_desc = sorted(numbers, key=lambda x: -x) # Using lambda as key in sorted() for descending order
print(f"Sorted descending using sorted(key=lambda): {sorted_numbers_desc}")

# --- Lambda functions are concise but limited to single expressions. For more complex logic, use regular `def` functions.
# --- They are especially powerful for functional programming paradigms and when you need short, throwaway functions.

################################################################################
# 4.9.7. Documentation Strings
################################################################################

print("\n--- 4.9.7. Documentation Strings ---\n")

# Documentation strings (docstrings) are string literals that appear as the first statement in a function, module, class, or method definition.
# They are used to document the object and are accessible via the `__doc__` attribute and the `help()` function.
# Good docstrings are crucial for code understandability, maintainability, and API documentation generation.

# --- Function with a docstring ---
print("\n--- Function with docstring ---")
def calculate_area(radius):
    """
    Calculates the area of a circle.

    Args:
        radius (float or int): The radius of the circle.

    Returns:
        float: The area of the circle.

    Raises:
        TypeError: if radius is not a number.
        ValueError: if radius is negative.
    """
    if not isinstance(radius, (int, float)):
        raise TypeError("Radius must be a number.")
    if radius < 0:
        raise ValueError("Radius cannot be negative.")
    return 3.14159 * radius**2

print(f"Docstring of calculate_area:\n{calculate_area.__doc__}") # Accessing docstring using __doc__
help(calculate_area) # Accessing docstring using help()

# --- Docstring conventions (PEP 257):
#   - First line should be a concise summary of the object's purpose.
#   - Leave a blank line after the summary line.
#   - For more detailed documentation, include parameters, return values, exceptions, and examples.
#   - Use triple quotes (`"""Docstring goes here"""`).
#   - For simple one-line docstrings, the closing quotes can be on the same line as the opening quotes.

# --- Docstrings are essential for creating self-documenting code and are used by documentation generators (e.g., Sphinx).
# --- Writing clear and informative docstrings is a hallmark of professional Python development.

################################################################################
# 4.9.8. Function Annotations
################################################################################

print("\n--- 4.9.8. Function Annotations ---\n")

# Function annotations (type hints) are metadata about the types of function parameters and return values.
# They are specified using the `:` and `->` syntax in function definitions.
# Annotations are primarily for documentation, static analysis tools (like MyPy), and IDEs to improve code understanding and catch type-related errors *before* runtime.
# Python itself does *not* enforce type annotations at runtime by default (though libraries like `pydantic` or `beartype` can enable runtime type checking).

# --- Function with annotations ---
print("\n--- Function with annotations ---")
def greet_person(name: str, age: int) -> str:
    """Greets a person with their name and age.

    Args:
        name: The name of the person (string).
        age: The age of the person (integer).

    Returns:
        A greeting string.
    """
    return f"Hello, {name}! You are {age} years old."

greeting = greet_person("Alice", 28)
print(f"Greeting: {greeting}")

print(f"Annotations of greet_person: {greet_person.__annotations__}") # Accessing annotations using __annotations__

# --- Annotations are arbitrary expressions, but typically type hints from the `typing` module are used for clarity and static analysis.
# --- Common type hints from `typing`: List, Tuple, Dict, Set, Union, Optional, Callable, etc.

from typing import List, Tuple, Dict, Union, Optional

def process_data(items: List[int]) -> Optional[Tuple[int, int]]:
    """Processes a list of integers and returns a tuple or None.

    Args:
        items: A list of integers.

    Returns:
        A tuple of two integers or None if processing fails.
    """
    if not items:
        return None
    return min(items), max(items)

data_result = process_data([3, 1, 4, 1, 5, 9])
print(f"process_data result: {data_result}")

# --- Function annotations enhance code readability, enable static type checking, and improve collaboration in larger projects.
# --- They are increasingly adopted in modern Python development for better code quality and maintainability.

################################################################################
# 4.10. Intermezzo: Coding Style
################################################################################

print("\n--- 4.10. Intermezzo: Coding Style ---\n")

# Coding style is crucial for writing clean, readable, and maintainable Python code.
# Adhering to established style guides like PEP 8 (Style Guide for Python Code) is highly recommended for professional Python development.

# Key aspects of Pythonic coding style (PEP 8):

# - Readability: Code should be easy to read and understand.
# - Consistency: Follow consistent naming conventions, indentation, and code structure.
# - Indentation: Use 4 spaces per indentation level (no tabs).
# - Line length: Limit lines to 79 characters (72 for docstrings/comments).
# - Blank lines: Use blank lines to separate logical sections of code.
# - Imports: Place imports at the top of the file, one import per line.
# - Naming conventions:
#     - `snake_case` for variables, functions, and modules.
#     - `CamelCase` for classes.
#     - `UPPER_CASE` for constants.
# - Comments: Use comments to explain complex logic, not to restate obvious code.
# - Docstrings: Write clear and informative docstrings for all public modules, functions, classes, and methods.

# Tools for style checking and code formatting:
# - `flake8`: Comprehensive style checker (combines PEP 8, pyflakes, and McCabe).
# - `pylint`: More in-depth static analysis and style checking.
# - `black`: Opinionated code formatter that automatically formats code to conform to PEP 8 style.
# - `autopep8`: Another automatic code formatter.

# Adopting a consistent and Pythonic coding style significantly improves code quality, collaboration, and long-term maintainability of Python projects.

print("\n--- End of More on Defining Functions ---")