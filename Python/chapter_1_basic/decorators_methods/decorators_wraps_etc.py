# Python Special Methods and Decorators Guide

import abc
import functools
import contextlib
import asyncio
import typing
import dataclasses
import time
from functools import wraps
from contextlib import asynccontextmanager

# =====================================
# @abstractmethod
# =====================================
"""
Marks a method as abstract, requiring implementation by concrete subclasses.
Part of the Abstract Base Class (ABC) mechanism.
"""

class Shape(abc.ABC):
    @abc.abstractmethod
    def area(self):
        """Calculate the area of the shape."""
        pass  # Implementation required in subclasses
    
    @abc.abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape."""
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# rect = Rectangle(5, 3)  # Works because all abstract methods are implemented
# shape = Shape()  # TypeError: Can't instantiate abstract class

# =====================================
# @cached_property
# =====================================
"""
Combines @property with caching - computes value once, then caches it as instance attribute.
Available in Python 3.8+ in functools.
"""

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @functools.cached_property
    def processed_result(self):
        print("Performing expensive calculation...")
        time.sleep(0.5)  # Simulate expensive operation
        return sum(x * 2 for x in self.data)

# processor = DataProcessor([1, 2, 3, 4, 5])
# print(processor.processed_result)  # Calculates and prints result
# print(processor.processed_result)  # Uses cached value (no recalculation)

# =====================================
# @functools.wraps
# =====================================
"""
Preserves metadata (name, docstring, etc.) of the decorated function.
Essential for creating well-behaved decorators.
"""

def log_execution(func):
    @functools.wraps(func)  # Preserves func metadata
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@log_execution
def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b

# print(add_numbers(5, 3))  # Logs execution and returns 8
# print(add_numbers.__name__)  # Prints "add_numbers" (preserved)
# print(add_numbers.__doc__)   # Prints docstring (preserved)

# =====================================
# @functools.lru_cache
# =====================================
"""
Implements memoization - caches function results based on arguments.
Uses LRU (Least Recently Used) strategy for limited cache size.
"""

@functools.lru_cache(maxsize=128)
def fibonacci(n):
    """Calculate nth Fibonacci number with memoization."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Without caching, this would be extremely slow for large n
# print(fibonacci(30))  # Fast calculation due to caching
# print(fibonacci.cache_info())  # Shows hits, misses, maxsize, etc.
# fibonacci.cache_clear()  # Clears the cache

# =====================================
# @functools.total_ordering
# =====================================
"""
Automatically generates all comparison methods (__lt__, __gt__, __le__, __ge__)
when you define just __eq__ and one other comparison method.
"""

@functools.total_ordering
class Version:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch
    
    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __repr__(self):
        return f"Version({self.major}, {self.minor}, {self.patch})"

# v1 = Version(1, 5, 0)
# v2 = Version(2, 0, 1)
# print(v1 < v2)   # True (explicitly defined)
# print(v1 > v2)   # False (auto-generated)
# print(v1 <= v2)  # True (auto-generated)
# print(v1 >= v2)  # False (auto-generated)

# =====================================
# @functools.singledispatch
# =====================================
"""
Implements single dispatch generic functions - allows different function
implementations based on the type of the first argument.
"""

@functools.singledispatch
def serialize(obj):
    """Default serialization for unknown types."""
    return str(obj)

@serialize.register
def _(obj: list):
    """Serialize lists by serializing each element."""
    return [serialize(item) for item in obj]

@serialize.register
def _(obj: dict):
    """Serialize dictionaries by serializing each key/value."""
    return {key: serialize(value) for key, value in obj.items()}

@serialize.register(int)  # Alternative syntax
def _(obj):
    """Serialize integers as hex strings."""
    return f"0x{obj:x}"

# print(serialize(42))               # '0x2a' (int impl)
# print(serialize("hello"))          # 'hello' (default impl)
# print(serialize([1, "two", 3.0]))  # ['0x1', 'two', '3.0'] (list impl)
# print(serialize({"a": 1, "b": [2, 3]}))  # {'a': '0x1', 'b': ['0x2', '0x3']}

# =====================================
# @functools.singledispatchmethod
# =====================================
"""
Similar to singledispatch but for class methods - dispatches based on
type of first argument after 'self'.
"""

class Formatter:
    @functools.singledispatchmethod
    def format(self, obj):
        """Default formatter for unknown types."""
        return str(obj)
    
    @format.register
    def _(self, obj: int):
        """Format integers with commas as thousand separators."""
        return f"{obj:,}"
    
    @format.register
    def _(self, obj: float):
        """Format floats with 2 decimal places."""
        return f"{obj:.2f}"
    
    @format.register
    def _(self, obj: list):
        """Format lists as comma-separated items."""
        return ", ".join(self.format(item) for item in obj)

# formatter = Formatter()
# print(formatter.format(1234567))  # '1,234,567'
# print(formatter.format(3.14159))  # '3.14'
# print(formatter.format([1, 2.5, "three"]))  # '1, 2.50, three'

# =====================================
# @functools.cache
# =====================================
"""
Simpler version of lru_cache with unlimited size (Python 3.9+).
Caches all function calls based on arguments.
"""

@functools.cache
def factorial(n):
    """Calculate factorial with unlimited caching."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

# print(factorial(10))  # 3628800 (calculated efficiently using caching)
# print(factorial.cache_info())  # Cache statistics

# =====================================
# @contextlib.contextmanager
# =====================================
"""
Transforms a generator function into a context manager (with statement).
Simplifies resource management with setup/cleanup patterns.
"""

@contextlib.contextmanager
def open_file(filename, mode="r"):
    """Context manager for file handling with proper cleanup."""
    try:
        file = open(filename, mode)
        print(f"Opened {filename} in {mode} mode")
        yield file  # Provide resource to with block
    finally:
        file.close()  # Always executed for cleanup
        print(f"Closed {filename}")

# Usage with 'with' statement:
# with open_file("example.txt", "w") as f:
#     f.write("Hello, World!")

# =====================================
# @asyncio.coroutine
# =====================================
"""
Marks generator-based coroutines for async operations.
DEPRECATED: Use 'async def' and 'await' syntax instead.
"""

@asyncio.coroutine  # Deprecated approach
def old_style_coroutine():
    """Legacy generator-based coroutine."""
    print("Starting")
    yield from asyncio.sleep(1)  # Non-blocking sleep
    print("Finished")
    return "Result"

# Modern equivalent using async/await:
async def modern_coroutine():
    """Modern native coroutine."""
    print("Starting")
    await asyncio.sleep(1)  # Non-blocking sleep
    print("Finished")
    return "Result"

# To run either coroutine:
# asyncio.run(modern_coroutine())

# =====================================
# @typing.overload
# =====================================
"""
Provides multiple type signatures for a function (type hinting).
Doesn't affect runtime behavior, only static type checking.
"""

@typing.overload
def normalize(value: int) -> float:
    ...

@typing.overload
def normalize(value: list) -> list[float]:
    ...

@typing.overload
def normalize(value: dict) -> dict[str, float]:
    ...

def normalize(value):
    """
    Normalize numerical values in different data structures.
    The @overload decorators provide type hints to static type checkers.
    """
    if isinstance(value, int):
        return float(value)
    elif isinstance(value, list):
        return [float(x) for x in value]
    elif isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    raise TypeError(f"Unsupported type: {type(value)}")

# print(normalize(42))        # 42.0
# print(normalize([1, 2, 3])) # [1.0, 2.0, 3.0]
# print(normalize({"a": 1, "b": 2}))  # {'a': 1.0, 'b': 2.0}

# =====================================
# @dataclasses.dataclass
# =====================================
"""
Automatically generates boilerplate code for classes (Python 3.7+):
__init__, __repr__, __eq__, etc. Creates data container classes easily.
"""

@dataclasses.dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    
    def total_value(self):
        """Calculate total value of product inventory."""
        return self.price * self.quantity

# Advanced usage with options
@dataclasses.dataclass(frozen=True, order=True)
class Point:
    x: float
    y: float
    # Field with custom metadata
    label: str = dataclasses.field(default="", compare=False, repr=True)
    # Calculated field
    distance: float = dataclasses.field(init=False, repr=True)
    
    def __post_init__(self):
        """Called after __init__ to calculate derived values."""
        object.__setattr__(self, "distance", (self.x**2 + self.y**2)**0.5)

# p1 = Product("Laptop", 999.99, 5)
# p2 = Product("Laptop", 999.99, 5)
# print(p1)  # Product(name='Laptop', price=999.99, quantity=5)
# print(p1 == p2)  # True (automatic equality comparison)
# print(p1.total_value())  # 4999.95

# point = Point(3, 4, "Origin")
# print(point)  # Point(x=3.0, y=4.0, label='Origin', distance=5.0)

# =====================================
# @asynccontextmanager
# =====================================
"""
Creates asynchronous context managers for use with 'async with'.
Similar to @contextmanager but works with async/await syntax.
"""

@asynccontextmanager
async def async_db_transaction(db_connection):
    """Async context manager for database transactions."""
    # Setup - begin transaction
    print("Starting transaction")
    await db_connection.execute("BEGIN TRANSACTION")
    
    try:
        yield db_connection  # Provide connection to async with block
    except Exception as e:
        # Exception - rollback transaction
        print(f"Error: {e}, rolling back")
        await db_connection.execute("ROLLBACK")
        raise
    else:
        # No exception - commit transaction
        print("Committing transaction")
        await db_connection.execute("COMMIT")

# Usage with 'async with':
# async def main():
#     db = await create_db_connection()
#     async with async_db_transaction(db) as conn:
#         await conn.execute("INSERT INTO table VALUES (1, 2, 3)")
#     # Transaction automatically committed
# asyncio.run(main())

# =====================================
# @wraps
# =====================================
"""
Alias for functools.wraps - preserves function metadata in decorators.
Imported directly from functools or used via functools.wraps.
"""

from functools import wraps

def timing_decorator(func):
    @wraps(func)  # Preserves metadata (name, docstring, etc.)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timing_decorator
def slow_function(delay):
    """A function that takes some time to execute."""
    time.sleep(delay)
    return f"Completed after {delay} seconds"

# print(slow_function(0.5))  # Times execution and returns result
# print(slow_function.__name__)  # 'slow_function' (preserved)