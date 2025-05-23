#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Built-ins: slice, eval, compile, exec
====================================================

This module demonstrates powerful but often misunderstood Python built-in 
functions that can significantly enhance code capabilities when used properly.
"""

##############################################################################
# slice
##############################################################################
"""
The slice built-in creates slice objects that can be used for advanced slicing
operations on sequences.

Key characteristics:
- Creates a slice object representing indices for slicing
- Takes start, stop, and step parameters
- Can be used with __getitem__, __setitem__ for custom slicing behavior
- Allows for named slices that can be reused
- More powerful than the normal slice notation [start:stop:step]
"""

# Basic slice object creation
simple_slice = slice(1, 5)  # Equivalent to [1:5]
print(f"Simple slice object: {simple_slice}")
print(f"Slice attributes - start: {simple_slice.start}, stop: {simple_slice.stop}, step: {simple_slice.step}")

# Using slice objects with sequences
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Original list: {numbers}")
print(f"Using slice object: {numbers[simple_slice]}")  # Same as numbers[1:5]

# Creating more complex slices
reverse_slice = slice(None, None, -1)  # Equivalent to [::-1]
step_slice = slice(1, 8, 2)  # Equivalent to [1:8:2]
print(f"Reversed: {numbers[reverse_slice]}")
print(f"Every other item from 1 to 7: {numbers[step_slice]}")

# Named slices for reuse
FIRST_QUARTER = slice(0, 3)
MIDDLE_HALF = slice(3, 7)
LAST_QUARTER = slice(7, None)

quarterly_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"First quarter: {quarterly_data[FIRST_QUARTER]}")
print(f"Middle half: {quarterly_data[MIDDLE_HALF]}")
print(f"Last quarter: {quarterly_data[LAST_QUARTER]}")

# Using slices with different data types
text = "Python Programming"
print(f"Original text: {text}")
print(f"Every other character: {text[step_slice]}")
print(f"Reversed text: {text[reverse_slice]}")

# Using slice with custom objects
class SliceableObject:
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, slc):
        """Supports advanced slicing with slice objects"""
        if isinstance(slc, slice):
            print(f"Custom slicing with start={slc.start}, stop={slc.stop}, step={slc.step}")
            # Apply the slice to our internal data
            return self.data[slc]
        # Handle integer indexing
        return self.data[slc]
    
    def __setitem__(self, slc, value):
        """Supports slice-based assignment"""
        if isinstance(slc, slice):
            print(f"Setting values with slice: start={slc.start}, stop={slc.stop}, step={slc.step}")
            self.data[slc] = value
        else:
            self.data[slc] = value


# Using custom sliceable object
sliceable = SliceableObject([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"Custom slice result: {sliceable[slice(2, 8, 2)]}")

# Setting values using slice
my_list = [0, 1, 2, 3, 4, 5]
print(f"Before slice assignment: {my_list}")
my_list[slice(1, 4)] = [10, 20, 30]
print(f"After slice assignment: {my_list}")

# EXCEPTION CASE: Slices with negative indices can be confusing
neg_slice = slice(-5, -1)
print(f"Negative slice on list: {numbers[neg_slice]}")  # Works with negative indices

# EXCEPTION CASE: Empty slices
empty_slice1 = slice(5, 1)  # start > stop with default step
empty_slice2 = slice(1, 5, -1)  # positive range with negative step
print(f"Empty slice 1: {numbers[empty_slice1]}")  # Returns empty list
print(f"Empty slice 2: {numbers[empty_slice2]}")  # Returns empty list

# EXCEPTION CASE: None values are interpreted as defaults
default_slice = slice(None, None, None)  # Equivalent to [:]
print(f"Default slice (full copy): {numbers[default_slice]}")

# Advanced: Using slices with numpy arrays (if available)
try:
    import numpy as np
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"NumPy array:\n{arr}")
    
    # Complex slice with NumPy
    complex_slice = (slice(0, 2), slice(1, 3))
    print(f"Complex NumPy slice (first 2 rows, columns 1-2):\n{arr[complex_slice]}")
except ImportError:
    print("NumPy not available for advanced slice demonstration")


##############################################################################
# eval
##############################################################################
"""
eval() evaluates a Python expression from a string and returns the result.

Key characteristics:
- Executes an expression string and returns the result
- Can access variables from the current scope
- Takes optional globals and locals dictionaries for environment control
- Powerful but potentially dangerous with untrusted input
- Should always be used with caution for security reasons
"""

# Basic eval usage
simple_expression = "2 + 2"
result = eval(simple_expression)
print(f"Evaluated '{simple_expression}' to: {result}")

# Using eval with variables from current scope
x = 10
y = 5
expression_with_vars = "x * y + 2"
result = eval(expression_with_vars)
print(f"Evaluated '{expression_with_vars}' with x={x}, y={y} to: {result}")

# Controlling the execution environment with globals/locals
safe_globals = {"__builtins__": {}}  # Empty builtins for safety
safe_locals = {"x": 42, "y": 10}

try:
    # This will fail because we removed builtins
    result = eval("max(x, y)", safe_globals)
except NameError as e:
    print(f"Expected error with restricted globals: {e}")

# Providing needed functions explicitly
math_globals = {
    "__builtins__": {},  # Empty builtins
    "max": max,          # Explicitly allow max
    "min": min,          # Explicitly allow min
    "abs": abs           # Explicitly allow abs
}
result = eval("max(x, y)", math_globals, safe_locals)
print(f"Evaluated 'max(x, y)' in controlled environment: {result}")

# Complex expressions
complex_expr = "[x*y for x in range(5) for y in range(3) if x*y > 0]"
result = eval(complex_expr)
print(f"List comprehension via eval: {result}")

# Creating a simple calculator
def safe_eval_calculator(expression):
    """A simple calculator that safely evaluates basic math expressions."""
    # Define a safe environment with only math operations
    safe_dict = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "pow": pow,
        "int": int,
        "float": float,
    }
    
    try:
        # Add basic math operators and functions from math module
        import math
        for name in dir(math):
            if not name.startswith('_'):  # Skip private attributes
                safe_dict[name] = getattr(math, name)
        
        return eval(expression, {"__builtins__": {}}, safe_dict)
    except Exception as e:
        return f"Error: {str(e)}"


# Test the calculator
calc_expressions = [
    "2 + 2 * 3",
    "sin(0.5) + cos(0.5)",
    "sqrt(16) + pow(2, 3)",
    "log(10) * 2"
]

print("\nCalculator results:")
for expr in calc_expressions:
    result = safe_eval_calculator(expr)
    print(f" {expr} = {result}")

# EXCEPTION CASE: Code injection vulnerability with unrestricted eval
dangerous_input = "str(__import__('os').system('echo Potential code injection'))"
print("\nDemonstrating security risk (normally you would never do this):")
try:
    # Using a try block to prevent actual execution in demonstration
    print(f"Dangerous input: {dangerous_input}")
    print("If evaluated without restrictions, this could execute arbitrary code!")
    # result = eval(dangerous_input)  # This would actually execute the code!
except Exception as e:
    print(f"Error: {e}")

# EXCEPTION CASE: SyntaxError with invalid expressions
try:
    eval("x + y +")  # Incomplete expression
except SyntaxError as e:
    print(f"\nSyntax error with invalid expression: {e}")

# EXCEPTION CASE: NameError when variable doesn't exist
try:
    eval("undefined_variable + 10")
except NameError as e:
    print(f"Name error with undefined variable: {e}")

# EXCEPTION CASE: TypeError with incompatible types
try:
    eval("'string' + 10")
except TypeError as e:
    print(f"Type error with incompatible types: {e}")

# Best practice: Using ast.literal_eval for safer evaluation of literals
import ast
literal_expression = "[1, 2, 3] + [4, 5, 6]"
try:
    # This will fail because ast.literal_eval only evaluates literals, not operations
    result = ast.literal_eval(literal_expression)
except ValueError as e:
    print(f"\nast.literal_eval safely rejected non-literal: {e}")

# What ast.literal_eval can safely evaluate
safe_literals = [
    "{'a': 1, 'b': 2}",  # Dictionary
    "[1, 2, 3]",         # List
    "(1, 2, 3)",         # Tuple
    "{'a', 'b', 'c'}",   # Set
    "True",              # Boolean
    "None",              # None
    "123",               # Integer
    "123.456",           # Float
    "'string'",          # String
]

print("\nSafe literal evaluation with ast.literal_eval:")
for literal in safe_literals:
    result = ast.literal_eval(literal)
    print(f"  {literal} => {result} ({type(result).__name__})")


##############################################################################
# compile
##############################################################################
"""
compile() converts source code to a code object that can be executed.

Key characteristics:
- Compiles source into a code object for later execution
- Takes source string, filename, and compile mode
- Modes: 'exec' (statements), 'eval' (single expression), 'single' (interactive)
- Creates reusable code objects that can be executed multiple times
- More efficient than repeatedly using eval() or exec() on the same code
- Advanced tool for meta-programming and dynamic code execution
"""

# Basic compilation and execution
simple_code = "result = 5 * 5"
compiled_code = compile(simple_code, "<string>", "exec") # compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)
print(f"Compiled code object: {compiled_code}")

# Execute the compiled code in the current namespace
exec(compiled_code) #exec
print(f"After execution, result = {result}")

# Compile mode 'eval' - for expressions that return a value
expression = "2 ** 10"
compiled_expr = compile(expression, "<string>", "eval")
result = eval(compiled_expr) #eval
print(f"Compiled and evaluated expression '{expression}': {result}")

# Compile mode 'single' - for single interactive statement
interactive_code = "print('This simulates interactive mode')"
compiled_interactive = compile(interactive_code, "<string>", "single")
exec(compiled_interactive)

# Compiling and executing a function definition
function_def = """
def greet(name):
    return f"Hello, {name}!"
"""
compiled_function = compile(function_def, "<string>", "exec")
namespace = {}
exec(compiled_function, namespace)  # Execute in a separate namespace

# Now use the function from the namespace
greeting = namespace["greet"]("Python User")
print(f"Using compiled function: {greeting}") #compile  and execute

# Compiling a class definition
class_def = """
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def __str__(self):
        return f"Rectangle({self.width}×{self.height})"
"""
compiled_class = compile(class_def, "<string>", "exec") # "<string>", exec, compile, namespace={}
namespace = {}
exec(compiled_class, namespace)

# Use the compiled class
rect = namespace["Rectangle"](5, 3)
print(f"Using compiled class: {rect}, area: {rect.area()}")

# Performance advantage of pre-compilation
import time

# Code to be executed many times
complex_code = """
total = 0
for i in range(1000):
    total += i
"""

# Measure time without compilation (using exec directly)
start_time = time.time()
for _ in range(1000):
    exec(complex_code)
direct_exec_time = time.time() - start_time

# Measure time with pre-compilation
compiled = compile(complex_code, "<string>", "exec")
start_time = time.time()
for _ in range(1000):
    exec(compiled)
compiled_exec_time = time.time() - start_time

print(f"\nPerformance comparison:")
print(f"  Direct exec time: {direct_exec_time:.6f} seconds")
print(f"  Compiled exec time: {compiled_exec_time:.6f} seconds")
print(f"  Speedup factor: {direct_exec_time / compiled_exec_time:.2f}x")

# EXCEPTION CASE: SyntaxError during compilation
try:
    compile("for i in range(10) print(i)", "<string>", "exec")  # Missing colon
except SyntaxError as e:
    print(f"\nSyntax error during compilation: {e}")
    print(f"  Line: {e.lineno}, Offset: {e.offset}")
    print(f"  Text: {e.text}")

# EXCEPTION CASE: Using the wrong mode
try:
    # Using 'eval' mode for a statement (should be 'exec')
    compile("x = 10", "<string>", "eval")
except SyntaxError as e:
    print(f"Wrong mode error: {e}")

try:
    # Using 'exec' mode for an expression that returns a value
    result = eval(compile("2 + 2", "<string>", "exec"))
except TypeError as e:
    print(f"Cannot eval 'exec' mode code: {e}")

# Advanced: Getting code object attributes
code_obj = compile("x = [i**2 for i in range(10)]", "<string>", "exec")
print("\nCode object attributes:")
print(f"  co_name: {code_obj.co_name}")
print(f"  co_filename: {code_obj.co_filename}")
print(f"  co_argcount: {code_obj.co_argcount}")
print(f"  co_code length: {len(code_obj.co_code)} bytes")
print(f"  co_consts: {code_obj.co_consts}")
print(f"  co_names: {code_obj.co_names}")
print(f"  co_varnames: {code_obj.co_varnames}")

# Advanced: Dynamically generated recursive factorial function
factorial_code = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n-1)
"""
compiled_factorial = compile(factorial_code, "<string>", "exec")
namespace = {}
exec(compiled_factorial, namespace)

# Use the dynamically created factorial function
for i in range(1, 6):
    print(f"factorial({i}) = {namespace['factorial'](i)}")


##############################################################################
# exec
##############################################################################
"""
exec() executes Python code dynamically as a statement.

Key characteristics:
- Executes Python code stored in strings or code objects
- Returns None (unlike eval() which returns the expression result)
- Can execute multiple statements, including function and class definitions
- Takes optional globals and locals dictionaries for environment control
- Powerful but potentially dangerous with untrusted input
- Should always be used with caution for security reasons
"""

# Basic exec usage - executing a simple statement
exec("x = 100")  # Creates or modifies x in the current namespace
print(f"After exec, x = {x}")

# Executing multiple statements
code_block = """
y = 200
z = x + y
print(f"Inside exec: z = {z}")
"""
exec(code_block)
# print(f"After exec, y = {y}, z = {z}")

# Using custom namespace with exec
namespace = {"a": 10, "b": 20}
exec("c = a + b", namespace)
print(f"exec in custom namespace: {namespace}")

# Both globals and locals dictionaries
globals_dict = {"g_var": "global"}
locals_dict = {"l_var": "local"}
exec("result = g_var + ' and ' + l_var", globals_dict, locals_dict)
print(f"Result from separate globals/locals: {locals_dict['result']}")

# Creating a function dynamically
function_code = """
def multiply(a, b):
    return a * b
"""
exec(function_code)
# print(f"Using dynamically created function: {multiply(6, 7)}")

# Creating a class dynamically
class_code = """
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        import math
        return math.pi * self.radius ** 2
"""
exec(class_code)
# circle = Circle(5)
# print(f"Area of circle with radius 5: {circle.area():.2f}")

# Executing code with pre-compiled code object (more efficient)
compiled_code = compile("for i in range(5): print(f'Iteration {i}')", "<string>", "exec")
print("\nExecuting pre-compiled code:")
exec(compiled_code)

# Creating a simple REPL (Read-Eval-Print Loop)
def simple_repl(global_context=None):
    """A very simple Python REPL implementation using exec."""
    if global_context is None:
        global_context = {
            "__builtins__": __builtins__,
            "print": print,
            "input": input,
        }
    
    print("Simple Python REPL (type 'exit()' to quit)")
    while True:
        try:
            # Read
            code = input(">>> ")
            if code.strip() == "exit()":
                break
            
            # Execute
            exec(code, global_context)
            
            # No explicit print step - user must use print() themselves
        except Exception as e:
            print(f"Error: {e}")


# To use the REPL (commented out to avoid interactive execution)
# simple_repl()

# EXCEPTION CASE: SyntaxError with invalid code
try:
    exec("for i in range(5) print(i)")  # Missing colon
except SyntaxError as e:
    print(f"\nSyntax error in exec: {e}")

# EXCEPTION CASE: NameError when variable doesn't exist
try:
    exec("print(undefined_variable)")
except NameError as e:
    print(f"Name error in exec: {e}")

# EXCEPTION CASE: Security risk with unrestricted access
dangerous_code = """
import os
print("This could execute dangerous commands if not properly restricted")
# os.system("rm -rf /")  # This would be VERY dangerous!
"""
print("\nPotentially dangerous code (restricted execution):")
# Using restricted execution environment
safe_globals = {"__builtins__": {"print": print}}  # Only allow print
try:
    exec(dangerous_code, safe_globals)
except NameError as e:
    print(f"Safely prevented: {e}")

# Safe execution with logging
def safe_exec(code, filename="<string>"):
    """Execute code safely with logging."""
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("safe_exec")
    
    # Set up a safe environment
    safe_globals = {
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "__name__": "__main__",
    }
    
    # Create a restricted locals dictionary
    safe_locals = {}
    
    try:
        logger.info(f"Executing code: {filename}")
        exec(code, safe_globals, safe_locals)
        logger.info("Execution completed successfully")
        return safe_locals
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise


# Example of safe execution
safe_code = """
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
total = sum(squared)  # This will fail in our safe environment
"""

try:
    print("\nAttempting safe execution:")
    result = safe_exec(safe_code)
    print(f"Execution result: {result}")
except Exception as e:
    print(f"Caught expected error: {e}")

# Creating a plugin system with exec
print("\nSimple plugin system example:")

class PluginSystem:
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name, code):
        """Compile and register a plugin from source code."""
        namespace = {"__name__": name}
        try:
            exec(code, namespace)
            if "initialize" not in namespace:
                raise ValueError("Plugin must define an initialize() function")
            self.plugins[name] = namespace
            print(f"Registered plugin: {name}")
            return True
        except Exception as e:
            print(f"Failed to register plugin {name}: {e}")
            return False
    
    def initialize_plugins(self):
        """Initialize all registered plugins."""
        for name, plugin in self.plugins.items():
            try:
                plugin["initialize"]()
            except Exception as e:
                print(f"Failed to initialize plugin {name}: {e}")


# Create sample plugins
plugin1_code = """
def initialize():
    print("Plugin 1 initialized")

def process_data(data):
    return data.upper()
"""

plugin2_code = """
def initialize():
    print("Plugin 2 initialized")

def process_data(data):
    return data.split()
"""

# Create and use the plugin system
plugin_system = PluginSystem()
plugin_system.register_plugin("StringUpper", plugin1_code)
plugin_system.register_plugin("StringSplitter", plugin2_code)

# Initialize all plugins
plugin_system.initialize_plugins()

# Use a plugin's functionality
test_data = "hello world"
upper_result = plugin_system.plugins["StringUpper"]["process_data"](test_data)
split_result = plugin_system.plugins["StringSplitter"]["process_data"](test_data)

print(f"Plugin 1 result: {upper_result}")
print(f"Plugin 2 result: {split_result}")

# Summary of safety considerations for eval, exec, and compile
print("\nSafety considerations for dynamic code execution:")
print("1. Never use with untrusted input without strict validation")
print("2. Use restricted environments with minimal privileges")
print("3. Consider ast.literal_eval for parsing data structures")
print("4. Use try/except to handle potential errors")
print("5. Prefer higher-level alternatives when possible")
print("6. Always document security implications in your code")