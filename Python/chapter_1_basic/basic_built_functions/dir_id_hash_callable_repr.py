"""
advanced_python_builtins.py - Deep Dive into Python Built-in Functions
======================================================================

This comprehensive guide explores important Python built-in functions:
dir(), id(), hash(), callable(), and repr()

Each function is explained with detailed examples, practical applications,
and exception cases to help you master these essential tools.
"""

# =============================================================================
# dir() FUNCTION
# =============================================================================
"""
The dir() function returns a sorted list of names in the current scope or
attributes of the specified object.

Syntax:
    dir([object])

Parameters:
    object (optional): The object to inspect.
    If omitted, returns names in the current scope.

Return Value:
    A sorted list of strings.

Use Cases:
    1. Exploring object attributes and methods
    2. Discovering available variables in current namespace
    3. Interactive debugging and learning
    4. Exploring new modules and libraries
"""

# Example 1: Using dir() without arguments (shows local namespace)
# ----------------------------------------------------------------
x = 10
y = "Python"

def example_function():
    pass

print("Example 1: dir() without arguments")
local_names = dir()  # Returns names in current local scope
print(local_names)
# Output includes: ['__builtins__', '__doc__', '__file__', 'example_function', 'x', 'y', ...]

# Example 2: Exploring built-in types with dir()
# ----------------------------------------------
string_example = "Python"
string_methods = dir(string_example)
print("\nExample 2: Some string methods:")
print([method for method in string_methods if not method.startswith('__')][:5])
# Output: ['capitalize', 'casefold', 'center', 'count', 'encode']

list_example = [1, 2, 3]
list_methods = dir(list_example)
print("\nSome list methods:")
print([method for method in list_methods if not method.startswith('__')][:5])
# Output: ['append', 'clear', 'copy', 'count', 'extend']

# Example 3: Exploring a custom class with dir()
# ----------------------------------------------
class ExampleClass:
    class_variable = "I'm a class variable"
    
    def __init__(self):
        self.instance_variable = "I'm an instance variable"
    
    def method(self):
        pass

obj = ExampleClass()
print("\nExample 3: Custom class attributes:")
print([attr for attr in dir(obj) if not attr.startswith('__')])
# Output: ['class_variable', 'instance_variable', 'method']

# Example 4: Module exploration with dir()
# ---------------------------------------
import math
print("\nExample 4: Math module functions:")
math_functions = [item for item in dir(math) if not item.startswith('__') and callable(getattr(math, item))]
print(math_functions[:5])  # Show first 5 functions
# Output: ['acos', 'acosh', 'asin', 'asinh', 'atan']

print("\nMath module constants:")
math_constants = [item for item in dir(math) if not item.startswith('__') and not callable(getattr(math, item))]
print(math_constants)
# Output: ['e', 'inf', 'nan', 'pi', 'tau']

# Example 5: Exception cases
# -------------------------
print("\nExample 5: dir() with None")
print(dir(None))
# Output: List of methods available on None object

try:
    z = 100
    del z
    print(dir(z))  # This will raise NameError
except NameError as e:
    print(f"Error: {e}")
    # Output: Error: name 'z' is not defined

# =============================================================================
# id() FUNCTION
# =============================================================================
"""
The id() function returns the identity (unique integer) of an object.
In CPython, this represents the memory address of the object.

Syntax:
    id(object)

Parameters:
    object: Any Python object

Return Value:
    An integer representing the object's identity

Use Cases:
    1. Checking if variables reference the same object
    2. Understanding object identity vs equality
    3. Examining Python's memory management
    4. Debugging reference issues
"""

# Example 1: Basic usage of id()
# -----------------------------
print("\nExample 1: Basic id() usage")
a = 42
b = 42  # Small integers are cached in Python
c = "hello"
d = "hello"  # Strings may be interned

print(f"id(a): {id(a)}")
print(f"id(b): {id(b)}")
print(f"Are a and b the same object? {a is b}")  # True due to integer caching

print(f"id(c): {id(c)}")
print(f"id(d): {id(d)}")
print(f"Are c and d the same object? {c is d}")  # Often True due to string interning

# Example 2: Mutable vs. immutable objects
# ---------------------------------------
print("\nExample 2: Mutable vs immutable objects")

# Immutable objects (new object created on modification)
x = 10
print(f"id(x) before: {id(x)}")
x += 1
print(f"id(x) after x += 1: {id(x)}")
print(f"Did id change? {id(10) != id(x)}")  # True - different objects

# Mutable objects (same object modified in-place)
my_list = [1, 2, 3]
original_id = id(my_list)
print(f"id(my_list) before: {original_id}")
my_list.append(4)
print(f"id(my_list) after append: {id(my_list)}")
print(f"Did id change? {original_id != id(my_list)}")  # False - same object

# Example 3: Object identity vs equality
# ------------------------------------
print("\nExample 3: Identity vs equality")
list1 = [1, 2, 3]
list2 = [1, 2, 3]  # Different object with same content
list3 = list1  # Reference to the same object

print(f"list1 == list2: {list1 == list2}")  # True (equal values)
print(f"list1 is list2: {list1 is list2}")  # False (different objects)

print(f"list1 == list3: {list1 == list3}")  # True (equal values)
print(f"list1 is list3: {list1 is list3}")  # True (same object)

# Example 4: Python's memory optimization
# -------------------------------------
print("\nExample 4: Python's memory optimization")

# Small integer caching (-5 to 256 in CPython)
small_int1 = 42
small_int2 = 42
print(f"small_int1 is small_int2: {small_int1 is small_int2}")  # True

# Large integers are not cached
large_int1 = 1000000
large_int2 = 1000000
print(f"large_int1 is large_int2: {large_int1 is large_int2}")  # Usually False

# Example 5: Exception cases (rare with id())
# -----------------------------------------
print("\nExample 5: id() exceptions")
try:
    print(id(non_existent_variable))  # Will raise NameError
except NameError as e:
    print(f"Error: {e}")
    # Output: Error: name 'non_existent_variable' is not defined

# =============================================================================
# hash() FUNCTION
# =============================================================================
"""
The hash() function returns the hash value of an object if it has one.
Hash values are integers used for quick dictionary and set lookups.

Syntax:
    hash(object)

Parameters:
    object: The object whose hash value to return.
    Must be hashable (immutable).

Return Value:
    An integer hash value

Use Cases:
    1. Dictionary key operations
    2. Set membership tests
    3. Implementing __hash__ for custom classes
    4. Fast data retrieval in hash-based collections
"""

# Example 1: Hashing immutable objects
# ----------------------------------
print("\nExample 1: Hashing immutable objects")
print(f"hash(42): {hash(42)}")
print(f"hash(3.14): {hash(3.14)}")
print(f"hash('hello'): {hash('hello')}")
print(f"hash((1, 2, 3)): {hash((1, 2, 3))}")  # Tuples of hashable objects are hashable
print(f"hash(None): {hash(None)}")
print(f"hash(True): {hash(True)}")

# Example 2: Unhashable (mutable) objects
# -------------------------------------
print("\nExample 2: Unhashable objects")
try:
    print(hash([1, 2, 3]))  # Lists are mutable, not hashable
except TypeError as e:
    print(f"Error hashing list: {e}")
    # Output: Error hashing list: unhashable type: 'list'

try:
    print(hash({1: 'one'}))  # Dictionaries are mutable, not hashable
except TypeError as e:
    print(f"Error hashing dict: {e}")
    # Output: Error hashing dict: unhashable type: 'dict'

try:
    print(hash({1, 2, 3}))  # Sets are mutable, not hashable
except TypeError as e:
    print(f"Error hashing set: {e}")
    # Output: Error hashing set: unhashable type: 'set'

# Example 3: Custom hashable objects
# --------------------------------
print("\nExample 3: Custom hashable objects")

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))  # Using tuple hash

p1 = Point(1, 2)
p2 = Point(1, 2)  # Different object, same values
p3 = Point(3, 4)

print(f"hash(p1): {hash(p1)}")
print(f"hash(p2): {hash(p2)}")
print(f"p1 == p2: {p1 == p2}")  # True
print(f"p1 is p2: {p1 is p2}")  # False
print(f"hash(p1) == hash(p2): {hash(p1) == hash(p2)}")  # True

# Using our hashable object in a dictionary
points_dict = {p1: "Point at (1,2)"}
print(f"Dictionary lookup with p2: {points_dict[p2]}")  # Works! Same hash & equality

# Example 4: Hash consistency and object mutation warning
# ----------------------------------------------------
print("\nExample 4: Hash consistency issues")

class MutablePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, MutablePoint):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

mp = MutablePoint(5, 10)
mp_dict = {mp: "Original Point"}

print(f"Initial hash: {hash(mp)}")
mp.x = 15  # DANGEROUS! Changing the object's state after using as key
print(f"Hash after modification: {hash(mp)}")  # Hash value changed!

# Now the object is in the wrong hash bucket
try:
    print(mp_dict[mp])  # This will likely fail
except KeyError:
    print("KeyError: Modified object can't be found in dictionary")

# Example 5: Extreme hash collisions
# -------------------------------
print("\nExample 5: Hash collisions")

class FixedHash:
    def __init__(self, value):
        self.value = value
    
    def __hash__(self):
        return 42  # All instances return the same hash!
    
    def __eq__(self, other):
        if not isinstance(other, FixedHash):
            return False
        return self.value == other.value

a = FixedHash(1)
b = FixedHash(2)

print(f"hash(a): {hash(a)}")
print(f"hash(b): {hash(b)}")
print(f"a == b: {a == b}")  # False, different values

collision_dict = {}
collision_dict[a] = "Value A"
collision_dict[b] = "Value B"

print(f"Lookup with same hash but different equality: {collision_dict[a]}")
print(f"Python handles hash collisions using equality checks")

# =============================================================================
# callable() FUNCTION
# =============================================================================
"""
The callable() function checks if an object can be called like a function.

Syntax:
    callable(object)

Parameters:
    object: Any Python object to check

Return Value:
    Boolean: True if the object is callable, False otherwise

Use Cases:
    1. Checking if an object is a function
    2. Validating callback functions
    3. Working with higher-order functions
    4. Interface/protocol checking
"""

# Example 1: Basic callable objects
# -------------------------------
print("\nExample 1: Callable objects")

# Functions are callable
def example_function():
    return "I'm callable"

# Lambda functions are callable
lambda_func = lambda x: x * 2

# Classes are callable (they create instances)
class ExampleClass:
    pass

# Methods are callable
class ClassWithMethod:
    def method(self):
        pass

obj = ClassWithMethod()

print(f"callable(example_function): {callable(example_function)}")  # True
print(f"callable(lambda_func): {callable(lambda_func)}")  # True
print(f"callable(ExampleClass): {callable(ExampleClass)}")  # True
print(f"callable(obj.method): {callable(obj.method)}")  # True

# Example 2: Non-callable objects
# -----------------------------
print("\nExample 2: Non-callable objects")
number = 42
string = "hello"
list_obj = [1, 2, 3]
dict_obj = {"a": 1}

print(f"callable(number): {callable(number)}")  # False
print(f"callable(string): {callable(string)}")  # False
print(f"callable(list_obj): {callable(list_obj)}")  # False
print(f"callable(dict_obj): {callable(dict_obj)}")  # False

# Example 3: Making objects callable with __call__
# ----------------------------------------------
print("\nExample 3: Custom callable objects with __call__")

class CallableObject:
    def __call__(self, *args, **kwargs):
        args_str = ", ".join(str(arg) for arg in args)
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"Called with args: {args_str} and kwargs: {kwargs_str}"

callable_obj = CallableObject()
print(f"callable(callable_obj): {callable(callable_obj)}")  # True
print(f"Result of calling: {callable_obj(1, 2, name='Python')}")
# Output: Called with args: 1, 2 and kwargs: name=Python

# Example 4: Practical application with callback
# -------------------------------------------
print("\nExample 4: Practical callback example")

def process_with_callback(value, callback=None):
    """Process a value and optionally apply a callback."""
    result = value * 2
    
    if callback is not None:
        if callable(callback):
            return callback(result)
        else:
            raise TypeError("Callback must be callable")
    return result

# With a valid callback
print(f"With valid callback: {process_with_callback(5, lambda x: x + 10)}")  # 20

# Without a callback
print(f"Without callback: {process_with_callback(5)}")  # 10

# With invalid callback
try:
    process_with_callback(5, "not_callable")
except TypeError as e:
    print(f"Error: {e}")  # Error: Callback must be callable

# Example 5: Edge cases
# -------------------
print("\nExample 5: Special callable cases")

# Built-in functions are callable
print(f"callable(len): {callable(len)}")  # True
print(f"callable(print): {callable(print)}")  # True

# Type constructors are callable
print(f"callable(int): {callable(int)}")  # True
print(f"callable(list): {callable(list)}")  # True

# Methods of built-in objects
print(f"callable('hello'.upper): {callable('hello'.upper)}")  # True
print(f"callable([].append): {callable([].append)}")  # True

# =============================================================================
# repr() FUNCTION
# =============================================================================
"""
The repr() function returns a string containing a printable representation
of an object. Ideally, repr(obj) should return a string that, when passed to
eval(), would recreate the object.

Syntax:
    repr(object)

Parameters:
    object: Any Python object

Return Value:
    A string representation of the object

Use Cases:
    1. Debugging and logging
    2. Creating unambiguous object representations
    3. Implementing __repr__ for custom classes
    4. Creating eval()-able string representations
"""

# Example 1: Basic usage with different types
# -----------------------------------------
print("\nExample 1: Basic repr() usage")

# Simple types
print(f"repr(42): {repr(42)}")  # '42'
print(f"repr(3.14): {repr(3.14)}")  # '3.14'
print(f"repr('hello'): {repr('hello')}")  # "'hello'"
print(f"repr(True): {repr(True)}")  # 'True'
print(f"repr(None): {repr(None)}")  # 'None'

# Collections
print(f"repr([1, 2, 3]): {repr([1, 2, 3])}")  # '[1, 2, 3]'
print(f"repr((1, 'a')): {repr((1, 'a'))}")  # "(1, 'a')"
# print(f"repr({'a': 1}): {repr({'a': 1})}")  # "{'a': 1}"  # ValueError: Space not allowed in string format specifier

# Example 2: repr() vs str()
# ------------------------
print("\nExample 2: repr() vs str()")

# For many objects, str() is human-readable, repr() is unambiguous
import datetime
now = datetime.datetime.now()

print(f"str(now): {str(now)}")  # e.g. '2023-01-01 12:34:56.789012'
print(f"repr(now): {repr(now)}")  # e.g. 'datetime.datetime(2023, 1, 1, 12, 34, 56, 789012)'

# Another example with escape characters
text = "Hello\nWorld"
print(f"str(text):\n{str(text)}")  # Shows actual newline
print(f"repr(text): {repr(text)}")  # Shows escape sequence: 'Hello\\nWorld'

# Example 3: Custom __repr__ implementation
# --------------------------------------
print("\nExample 3: Custom __repr__ implementation")

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        # String that could recreate this object when passed to eval()
        return f"Person('{self.name}', {self.age})"
    
    def __str__(self):
        # More human-readable representation
        return f"{self.name}, {self.age} years old"

person = Person("Alice", 30)
print(f"repr(person): {repr(person)}")  # Person('Alice', 30)
print(f"str(person): {str(person)}")    # Alice, 30 years old

# Example 4: Using eval() with repr() output
# ----------------------------------------
print("\nExample 4: eval() and repr()")

# For many built-in types, eval(repr(x)) == x
x = [1, 2, 3]
x_repr = repr(x)
x_eval = eval(x_repr)

print(f"Original x: {x}")
print(f"repr(x): {x_repr}")
print(f"eval(repr(x)): {x_eval}")
print(f"eval(repr(x)) == x: {x_eval == x}")  # True

# Example 5: repr() for debugging
# -----------------------------
print("\nExample 5: repr() for debugging")

def debug_value(value):
    """Demonstrate how repr() helps with debugging."""
    print(f"Type: {type(value).__name__}")
    print(f"Value (repr): {repr(value)}")
    
    if hasattr(value, '__dict__'):
        print(f"Attributes: {repr(value.__dict__)}")

# Test with different types
debug_value("hello\nworld")  # Shows escape characters
debug_value([1, None, True])  # Shows exact values including None
debug_value(person)          # Uses our custom __repr__

# Example 6: Edge cases
# -------------------
print("\nExample 6: repr() edge cases")

# Default repr for custom object
class BasicClass:
    pass

basic = BasicClass()
print(f"Default repr: {repr(basic)}")  # <__main__.BasicClass object at 0x...>

# Recursive data structures
recursive_list = [1, 2, 3]
recursive_list.append(recursive_list)  # List contains itself
print(f"Recursive list repr: {repr(recursive_list)}")  # Will show [...] for recursive part