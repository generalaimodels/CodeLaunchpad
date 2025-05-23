"""
================================================================================
Python Advanced Standard Library Utilities: Detailed Guide with Examples
================================================================================

This file provides an in-depth explanation and usage examples for the following
Python standard library features, focusing on best practices, edge cases, and
developer-centric insights. Each section is self-contained and includes at least
two standard examples with detailed explanations.

Covered Topics:
    - GenericAlias
    - RLock
    - WRAPPER_ASSIGNMENTS
    - WRAPPER_UPDATES
    - cache
    - cached_property
    - cmp_to_key
    - get_cache_token
    - lru_cache
    - namedtuple
    - partial
    - partialmethod
    - recursive_repr
    - reduce
    - singledispatch
    - singledispatchmethod
    - total_ordering
    - update_wrapper
    - wraps

================================================================================
"""

# ==============================================================================
# 1. GenericAlias (from typing module, Python 3.9+)
# ==============================================================================

"""
GenericAlias is the internal type used for parameterized generics, e.g., list[int].
It is rarely used directly, but understanding it is crucial for advanced type
manipulation and introspection.
"""

from typing import GenericAlias

# Example 1: Inspecting a parameterized type
ListInt = list[int]
print("Example 1: Type of list[int]:", type(ListInt))  # <class 'types.GenericAlias'>
print("Is ListInt a GenericAlias?", isinstance(ListInt, GenericAlias))  # True

# Example 2: Accessing __origin__ and __args__ for introspection
print("Example 2: __origin__ of list[int]:", ListInt.__origin__)  # <class 'list'>
print("Example 2: __args__ of list[int]:", ListInt.__args__)      # (<class 'int'>,)

# ==============================================================================
# 2. RLock (threading.RLock)
# ==============================================================================

"""
RLock (Reentrant Lock) allows a thread to acquire the same lock multiple times.
Useful for recursive locking in complex threaded code.
"""

import threading

# Example 1: Basic RLock usage
lock = threading.RLock()

def safe_increment(counter):
    with lock:
        counter[0] += 1

counter = [0]
safe_increment(counter)
print("Example 1: Counter after increment:", counter[0])  # 1

# Example 2: Recursive function with RLock
def recursive_sum(n, acc=0):
    with lock:
        if n == 0:
            return acc
        return recursive_sum(n-1, acc+n)

print("Example 2: Recursive sum with RLock:", recursive_sum(5))  # 15

# ==============================================================================
# 3. WRAPPER_ASSIGNMENTS and WRAPPER_UPDATES (functools)
# ==============================================================================

"""
These are used by functools.update_wrapper and functools.wraps to specify which
attributes of the wrapped function are assigned or updated on the wrapper.
"""

from functools import WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES

print("WRAPPER_ASSIGNMENTS:", WRAPPER_ASSIGNMENTS)  # ('__module__', '__name__', '__qualname__', '__doc__', '__annotations__')
print("WRAPPER_UPDATES:", WRAPPER_UPDATES)          # ('__dict__',)

# Example 1: Custom wrapper using update_wrapper
import functools

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before call")
        return func(*args, **kwargs)
    functools.update_wrapper(wrapper, func)
    return wrapper

@my_decorator
def greet(name):
    """Greets a person."""
    print(f"Hello, {name}!")

print("Example 1: Wrapper docstring:", greet.__doc__)  # "Greets a person."
greet("LOkita")

# Example 2: Changing WRAPPER_ASSIGNMENTS
def my_decorator_custom(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    functools.update_wrapper(wrapper, func, assigned=('__name__',))
    return wrapper

@my_decorator_custom
def foo():
    pass

print("Example 2: Custom assigned attributes:", foo.__name__)  # "foo"

# ==============================================================================
# 4. cache (functools.cache, Python 3.9+)
# ==============================================================================

"""
cache is a simple unbounded cache decorator for functions with no arguments.
"""

from functools import cache

# Example 1: Caching a function with no arguments
@cache
def get_magic_number():
    print("Computing magic number...")
    return 42

print("Example 1:", get_magic_number())  # Computes and prints 42
print("Example 1:", get_magic_number())  # Uses cache, prints 42

# Example 2: Caching a function with arguments (raises TypeError)
try:
    @cache
    def add(a, b):
        return a + b
except TypeError as e:
    print("Example 2: Error using cache with arguments:", e)

# ==============================================================================
# 5. cached_property (functools.cached_property, Python 3.8+)
# ==============================================================================

"""
cached_property is a property that is computed once per instance and then cached.
"""

from functools import cached_property

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @cached_property
    def area(self):
        print("Computing area...")
        return 3.14159 * self.radius ** 2

# Example 1: Using cached_property
c = Circle(3)
print("Example 1: Area (first access):", c.area)  # Computes
print("Example 1: Area (second access):", c.area) # Cached

# Example 2: Deleting cached property
del c.area
print("Example 2: Area (after delete):", c.area)  # Recomputes

# ==============================================================================
# 6. cmp_to_key (functools.cmp_to_key)
# ==============================================================================

"""
cmp_to_key converts an old-style comparison function to a key function for sorting.
"""

from functools import cmp_to_key

# Example 1: Sorting with cmp_to_key
def compare_len(a, b):
    return len(a) - len(b)

words = ["pear", "apple", "banana", "kiwi"]
sorted_words = sorted(words, key=cmp_to_key(compare_len))
print("Example 1: Sorted by length:", sorted_words)

# Example 2: Descending order
def reverse_compare(a, b):
    return b - a

numbers = [5, 2, 9, 1]
sorted_numbers = sorted(numbers, key=cmp_to_key(reverse_compare))
print("Example 2: Descending order:", sorted_numbers)

# ==============================================================================
# 7. get_cache_token (functools.get_cache_token, Python 3.8+)
# ==============================================================================

"""
get_cache_token returns a value that changes whenever the internal cache for
functools.lru_cache should be invalidated (e.g., after typing changes).
"""

from functools import get_cache_token

# Example 1: Get cache token
token1 = get_cache_token()
print("Example 1: Cache token:", token1)

# Example 2: Token changes after typing changes (rarely used directly)
# (Demonstration only; in practice, this is for advanced use cases.)

# ==============================================================================
# 8. lru_cache (functools.lru_cache)
# ==============================================================================

"""
lru_cache is a decorator that caches the results of function calls using a
Least Recently Used (LRU) cache.
"""

from functools import lru_cache

# Example 1: Fibonacci with lru_cache
@lru_cache(maxsize=4)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print("Example 1: fib(5):", fib(5))  # Uses cache for efficiency

# Example 2: lru_cache info and clear
print("Example 2: Cache info:", fib.cache_info())
fib.cache_clear()
print("Example 2: Cache cleared. Info:", fib.cache_info())

# ==============================================================================
# 9. namedtuple (collections.namedtuple)
# ==============================================================================

"""
namedtuple creates tuple subclasses with named fields for readability.
"""

from collections import namedtuple

# Example 1: Creating and using a namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print("Example 1: Point:", p, "x:", p.x, "y:", p.y)

# Example 2: _replace and _asdict
p2 = p._replace(x=10)
print("Example 2: Replaced x:", p2)
print("Example 2: As dict:", p2._asdict())

# ==============================================================================
# 10. partial (functools.partial)
# ==============================================================================

"""
partial creates a new function with some arguments fixed.
"""

from functools import partial

# Example 1: Fixing arguments
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
print("Example 1: square(5):", square(5))

# Example 2: Using partial with map
add = lambda x, y: x + y
add_five = partial(add, 5)
print("Example 2: add_five(10):", add_five(10))

# ==============================================================================
# 11. partialmethod (functools.partialmethod)
# ==============================================================================

"""
partialmethod is like partial, but for methods in classes.
"""

from functools import partialmethod

class MyClass:
    def greet(self, name, punctuation):
        return f"Hello, {name}{punctuation}"

    greet_exclaim = partialmethod(greet, punctuation="!")

# Example 1: Using partialmethod
obj = MyClass()
print("Example 1:", obj.greet_exclaim("Alice"))

# Example 2: Another partialmethod
class Math:
    def power(self, base, exp):
        return base ** exp
    square = partialmethod(power, exp=2)

m = Math()
print("Example 2: m.square(4):", m.square(4))

# ==============================================================================
# 12. recursive_repr (functools.recursive_repr)
# ==============================================================================

"""
recursive_repr helps avoid infinite recursion in __repr__ for recursive data structures.
"""

from functools import recursive_repr

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

    @recursive_repr()
    def __repr__(self):
        return f"Node({self.value}, next={self.next})"

# Example 1: Recursive structure
a = Node(1)
b = Node(2)
a.next = b
b.next = a  # Cycle

print("Example 1: Recursive repr:", a)

# Example 2: Non-recursive
c = Node(3)
print("Example 2: Non-recursive repr:", c)

# ==============================================================================
# 13. reduce (functools.reduce)
# ==============================================================================

"""
reduce applies a function cumulatively to the items of a sequence.
"""

from functools import reduce

# Example 1: Sum of a list
nums = [1, 2, 3, 4]
total = reduce(lambda x, y: x + y, nums)
print("Example 1: Sum:", total)

# Example 2: Product of a list
product = reduce(lambda x, y: x * y, nums)
print("Example 2: Product:", product)

# ==============================================================================
# 14. singledispatch (functools.singledispatch)
# ==============================================================================

"""
singledispatch creates generic functions that dispatch on the type of the first argument.
"""

from functools import singledispatch

@singledispatch
def describe(obj):
    return f"Object: {obj}"

@describe.register
def _(obj: int):
    return f"Integer: {obj}"

@describe.register
def _(obj: list):
    return f"List of length {len(obj)}"

# Example 1: Dispatching on int
print("Example 1:", describe(10))

# Example 2: Dispatching on list
print("Example 2:", describe([1, 2, 3]))

# ==============================================================================
# 15. singledispatchmethod (functools.singledispatchmethod, Python 3.8+)
# ==============================================================================

"""
singledispatchmethod is like singledispatch, but for methods in classes.
"""

from functools import singledispatchmethod

class Handler:
    @singledispatchmethod
    def handle(self, arg):
        return f"Default: {arg}"

    @handle.register
    def _(self, arg: int):
        return f"Handling int: {arg}"

    @handle.register
    def _(self, arg: str):
        return f"Handling str: {arg}"

# Example 1: Handling int
h = Handler()
print("Example 1:", h.handle(42))

# Example 2: Handling str
print("Example 2:", h.handle("hello"))

# ==============================================================================
# 16. total_ordering (functools.total_ordering)
# ==============================================================================

"""
total_ordering fills in missing ordering methods if at least __eq__ and one other
(__lt__, __le__, __gt__, or __ge__) are defined.
"""

from functools import total_ordering

@total_ordering
class Person:
    def __init__(self, age):
        self.age = age

    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        return self.age == other.age

    def __lt__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        return self.age < other.age

# Example 1: Using total_ordering
alice = Person(30)
bob = Person(25)
print("Example 1: alice > bob:", alice > bob)
print("Example 1: alice <= bob:", alice <= bob)

# Example 2: Sorting
people = [Person(40), Person(20), Person(30)]
sorted_people = sorted(people)
print("Example 2: Sorted ages:", [p.age for p in sorted_people])

# ==============================================================================
# 17. update_wrapper (functools.update_wrapper)
# ==============================================================================

"""
update_wrapper copies attributes from the wrapped function to the wrapper.
"""

def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    functools.update_wrapper(wrapper, func)
    return wrapper

# Example 1: Preserving function metadata
@decorator
def hello():
    """Say hello."""
    return "Hello!"

print("Example 1: __doc__:", hello.__doc__)

# Example 2: Custom assigned/updated attributes
def decorator_custom(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    functools.update_wrapper(wrapper, func, assigned=('__name__',), updated=())
    return wrapper

@decorator_custom
def foo2():
    pass

print("Example 2: __name__:", foo2.__name__)

# ==============================================================================
# 18. wraps (functools.wraps)
# ==============================================================================

"""
wraps is a decorator that applies update_wrapper to the wrapper function.
"""

from functools import wraps

def logging_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Example 1: Using wraps
@logging_decorator
def add(a, b):
    """Adds two numbers."""
    return a + b

print("Example 1: add(2, 3):", add(2, 3))
print("Example 1: __doc__:", add.__doc__)

# Example 2: Without wraps (metadata lost)
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@bad_decorator
def subtract(a, b):
    """Subtracts two numbers."""
    return a - b

print("Example 2: subtract.__doc__ (should be None):", subtract.__doc__)

# ==============================================================================
# END OF FILE
# ==============================================================================