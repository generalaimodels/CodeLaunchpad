#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Built-ins: A Comprehensive Guide
===============================================

This module demonstrates Python's powerful built-in functions that are often 
overlooked but can significantly enhance code quality and developer productivity.
"""

##############################################################################
# __import__
##############################################################################
"""
The __import__() function is a low-level function used by the 'import' statement.
It allows for dynamic importing of modules at runtime.

Key characteristics:
- Takes module name as a string
- Can import nested modules with 'fromlist' parameter
- Returns the module object
- More flexible but less readable than regular import statements
- Recommended to use importlib.import_module() instead in modern Python
"""

# Basic usage of __import__()
math_module = __import__('math')
print(f"Pi from dynamically imported math module: {math_module.pi}")

# Importing nested modules (submodules)
# Regular import: from os.path import join
# With __import__:
os_path = __import__('os.path', fromlist=['join'])
print(f"Join function from os.path: {os_path.join('/tmp', 'file.txt')}")

# Conditional importing
module_name = 'datetime' if True else 'time'
time_module = __import__(module_name)
print(f"Imported module: {time_module.__name__}")

# EXCEPTION CASE: Module not found
try:
    non_existent = __import__('non_existent_module')
except ImportError as e:
    print(f"Import Error: {e}")

# NOTE: Modern alternative using importlib (recommended approach)
import importlib
json_module = importlib.import_module('json')
print(f"JSON module version: {json_module.__version__ if hasattr(json_module, '__version__') else 'No version info'}")


##############################################################################
# classmethod
##############################################################################
"""
classmethod is a decorator that converts a method to a class method.

Key characteristics:
- Receives the class as the first argument (cls) instead of instance (self)
- Can be called on the class itself, not just on instances
- Commonly used for alternative constructors
- Cannot modify instance state (as it doesn't have access to instance)
- Useful for factory methods that return class instances
"""

class Person:
    # Class variable shared by all instances
    population = 0
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
        Person.population += 1
    
    # Regular instance method
    def display(self):
        return f"{self.name} is {self.age} years old"
    
    # Class method - note it takes cls, not self
    @classmethod
    def from_birth_year(cls, name, birth_year):
        """Alternative constructor that creates Person from birth year instead of age"""
        from datetime import date
        age = date.today().year - birth_year
        return cls(name, age)  # Creates new Person instance
    
    @classmethod
    def get_population(cls):
        """Returns the total population count"""
        return cls.population
    
    # EXCEPTION CASE: Trying to access instance attributes in class method will fail
    @classmethod
    def invalid_access(cls):
        # This would raise an error if uncommented
        # return cls.name  # AttributeError: type object 'Person' has no attribute 'name'
        return "Cannot access instance attributes"


# Using the regular constructor
person1 = Person("Alice", 30)
print(person1.display())

# Using the class method as an alternative constructor
person2 = Person.from_birth_year("Bob", 1990)
print(person2.display())

# Using class method to access class variables
print(f"Total population: {Person.get_population()}")

# Class methods can also be called on instances (though not common practice)
print(f"Population via instance: {person1.get_population()}")


##############################################################################
# staticmethod
##############################################################################
"""
staticmethod is a decorator that defines a static method in a class.

Key characteristics:
- Does not receive any special first argument (no self, no cls)
- Cannot access or modify class/instance state directly
- Conceptually similar to a regular function but belongs to class namespace
- Used for utility functions related to the class but not dependent on state
- More for logical organization than functionality that requires class/instance
"""

class MathUtils:
    PI = 3.14159
    
    def __init__(self, value):
        self.value = value
    
    # Instance method - has access to instance via self
    def calculate_area(self):
        """Calculate area of circle with the instance radius"""
        return MathUtils.PI * self.value * self.value
    
    # Static method - no access to instance or class
    @staticmethod
    def is_prime(num):
        """Check if a number is prime - doesn't need class or instance state"""
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    # Class method for comparison - has access to class via cls
    @classmethod
    def get_pi(cls):
        """Return the PI value defined in the class"""
        return cls.PI
    
    # EXCEPTION CASE: Static methods can't directly access class or instance attributes
    @staticmethod
    def invalid_direct_access():
        # This would raise an error if uncommented
        # return PI  # NameError: name 'PI' is not defined
        return "Cannot directly access class attributes"


# Using the static method from the class
print(f"Is 17 prime? {MathUtils.is_prime(17)}")

# Static methods can also be called on instances
math_instance = MathUtils(5)
print(f"Is 15 prime? {math_instance.is_prime(15)}")

# Comparing usage with instance and class methods
print(f"Area of circle with radius 5: {math_instance.calculate_area()}")
print(f"PI value from class method: {MathUtils.get_pi()}")


##############################################################################
# property
##############################################################################
"""
property is a built-in function that creates managed attributes in classes.

Key characteristics:
- Allows getter, setter, and deleter methods for an attribute
- Enables validation, computation, and encapsulation
- Makes attributes act like regular attributes while being methods behind the scenes
- Can make attributes read-only, write-only, or fully managed
- Helps implement the descriptor protocol in a simpler way
"""

class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius  # Protected attribute with underscore convention
    
    # Getter method
    @property
    def celsius(self):
        """Get the temperature in Celsius"""
        return self._celsius
    
    # Setter method
    @celsius.setter
    def celsius(self, value):
        """Set the temperature in Celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value
    
    # Another property that calculates fahrenheit on-the-fly
    @property
    def fahrenheit(self):
        """Get the temperature in Fahrenheit (calculated)"""
        return (self.celsius * 9/5) + 32
    
    # Setter for fahrenheit that converts to celsius
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set the temperature using Fahrenheit value"""
        if value < -459.67:
            raise ValueError("Temperature below absolute zero is not possible")
        self.celsius = (value - 32) * 5/9
    
    # Property with only a getter (read-only)
    @property
    def kelvin(self):
        """Get the temperature in Kelvin (read-only)"""
        return self.celsius + 273.15
    
    # EXCEPTION CASE: No setter defined for kelvin
    # If we tried to set kelvin directly, it would raise AttributeError


# Create a temperature instance
temp = Temperature(25)

# Access properties like regular attributes
print(f"Temperature in Celsius: {temp.celsius}°C")
print(f"Temperature in Fahrenheit: {temp.fahrenheit}°F")
print(f"Temperature in Kelvin: {temp.kelvin}K")

# Set temperature using different scales
temp.celsius = 30
print(f"Updated Celsius: {temp.celsius}°C")
print(f"Updated Fahrenheit: {temp.fahrenheit}°F")

temp.fahrenheit = 68
print(f"Updated via Fahrenheit: {temp.celsius}°C")

# EXCEPTION CASE: Demonstrating validation
try:
    temp.celsius = -300  # Below absolute zero
except ValueError as e:
    print(f"Validation error: {e}")

# EXCEPTION CASE: Demonstrating read-only property
try:
    temp.kelvin = 300  # No setter defined
except AttributeError as e:
    print(f"Read-only property error: {str(e)}")


##############################################################################
# super
##############################################################################
"""
super() returns a temporary object that allows access to methods of parent class.

Key characteristics:
- Enables cooperative multiple inheritance
- Allows calling methods from parent class(es)
- Commonly used in __init__ to initialize parent class
- Can be used with explicit parameters: super(Class, instance)
- Follows Method Resolution Order (MRO) in complex inheritance hierarchies
"""

class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some generic animal sound"
    
    def identify(self):
        return f"I am {self.name}, an animal"


class Mammal(Animal):
    def __init__(self, name, fur_color):
        super().__init__(name)  # Call parent's __init__
        self.fur_color = fur_color
    
    def speak(self):
        return "Some mammal sound"
    
    def identify(self):
        # Extend parent's identify method
        return f"{super().identify()} and a mammal with {self.fur_color} fur"


class Dog(Mammal):
    def __init__(self, name, fur_color, breed):
        # Call parent's __init__
        super().__init__(name, fur_color)
        self.breed = breed
    
    def speak(self):
        return "Woof!"
    
    def identify(self):
        # We can choose which parent method to extend
        return f"{super().identify()}, specifically a {self.breed} dog"


# Multiple inheritance example to show MRO complexity
class Robot:
    def __init__(self, model):
        self.model = model
    
    def identify(self):
        return f"I am robot model {self.model}"
    
    def speak(self):
        return "Beep boop"


class RoboDog(Dog, Robot):
    def __init__(self, name, fur_color, breed, model):
        # With multiple inheritance, we need to be explicit about which parent to initialize
        Dog.__init__(self, name, fur_color, breed)
        Robot.__init__(self, model)
    
    def speak(self):
        # Use super() to call the first method in MRO
        return f"{super().speak()} {Robot.speak(self)}"
    
    def identify(self):
        # Explicitly call both parent methods
        return f"{Dog.identify(self)} and {Robot.identify(self)}"


# Create instances and test inheritance
animal = Animal("Generic")
print(animal.identify())

dog = Dog("Rex", "brown", "German Shepherd")
print(dog.identify())
print(dog.speak())

robo_dog = RoboDog("K9", "metallic", "RoboDog", "X1000")
print(robo_dog.identify())
print(robo_dog.speak())

# EXCEPTION CASE: Demonstrate Method Resolution Order (MRO)
print(f"Method Resolution Order for RoboDog: {[cls.__name__ for cls in RoboDog.__mro__]}")

# EXCEPTION CASE: Using super with explicit arguments
# This is equivalent to the implicit super() inside a method
parent_identify = super(Dog, dog).identify()
print(f"Explicit super call: {parent_identify}")


##############################################################################
# object
##############################################################################
"""
object is the base class for all classes in Python.

Key characteristics:
- Every class implicitly inherits from object if no parent is specified
- Provides default implementations for standard methods
- Common methods: __init__, __str__, __repr__, __eq__, __hash__
- Used as a type for generic type hints
- Can be instantiated but typically used as a base class
"""

# All classes inherit from object implicitly
class SimpleClass:  # Equivalent to class SimpleClass(object):
    pass


# Demonstrate object as the base class
print(f"SimpleClass base classes: {SimpleClass.__bases__}")
print(f"Is SimpleClass subclass of object? {issubclass(SimpleClass, object)}")

# Object's default methods
obj = object()
print(f"Default object.__str__: {obj}")
print(f"Default object.__repr__: {repr(obj)}")
print(f"Object's dir: {dir(obj)[-10:]}")  # Last 10 methods

# Custom class overriding object methods
class Person(object):  # Explicit inheritance
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name}, {self.age} years old"
    
    def __repr__(self):
        return f"Person('{self.name}', {self.age})"
    
    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        return self.name == other.name and self.age == other.age
    
    def __hash__(self):
        return hash((self.name, self.age))


# Using the custom Person class
person = Person("Alice", 30)
print(f"Custom __str__: {person}")
print(f"Custom __repr__: {repr(person)}")

# Equality comparison and hashing
person2 = Person("Alice", 30)
person3 = Person("Bob", 25)
print(f"Equality: {person == person2}")
print(f"Inequality: {person == person3}")
print(f"Hash value: {hash(person)}")

# EXCEPTION CASE: Using object() as a sentinel value
SENTINEL = object()

def get_value(dictionary, key, default=SENTINEL):
    result = dictionary.get(key, SENTINEL)
    if result is SENTINEL:
        if default is SENTINEL:
            raise KeyError(f"Key '{key}' not found and no default provided")
        return default
    return result

# Using the sentinel
sample_dict = {"a": 1, "b": 2, "c": None}
print(f"Key exists: {get_value(sample_dict, 'a')}")
print(f"Key with default: {get_value(sample_dict, 'x', 'default')}")
print(f"Key with None value: {get_value(sample_dict, 'c')}")

try:
    get_value(sample_dict, 'x')  # No default provided
except KeyError as e:
    print(f"Expected error: {e}")


##############################################################################
# credits
##############################################################################
"""
credits() is a built-in function that displays credits for the Python interpreter.

Key characteristics:
- Interactive function intended for use in the Python REPL
- Displays information about Python development contributors
- Similar to other help functions like help() and copyright()
- Not typically used in production code, more for interactive sessions
- Returns None, primarily used for its side effect of printing information
"""

# To avoid actually executing this function in our script, we'll just describe it
# If you run credits() in the interactive Python shell, it will display:
# "Thanks to CWI, CNRI, BeOpen.com, Zope Corporation and a cast of thousands
# for supporting Python development. See www.python.org for more information."

# How it would normally be used in the Python REPL:
# >>> credits()

# How to programmatically get the credits as a string (safer than executing the function)
import sys
print("Python credits information is available through the 'credits' built-in function")
print(f"Python version: {sys.version.split()[0]}")

# EXCEPTION CASE: Not meant to be used programmatically
# If you need to access this information programmatically, use sys.version instead


##############################################################################
# license
##############################################################################
"""
license() is a built-in function that displays Python's license information.

Key characteristics:
- Interactive function intended for use in the Python REPL
- Displays Python's license text (typically the Python Software Foundation License)
- Similar to other help functions like help() and copyright()
- Not typically used in production code, more for interactive sessions
- Returns None, primarily used for its side effect of printing information
"""

# Like credits(), this is mainly for interactive use
# If you run license() in the interactive Python shell, it will display the Python license

# How it would normally be used in the Python REPL:
# >>> license()

# How to programmatically get license information
import sys
print("Python license information is available through the 'license' built-in function")
print(f"Python is released under the Python Software Foundation License")
print(f"For full details, run license() in an interactive Python session or visit python.org")

# EXCEPTION CASE: Not meant to be used programmatically
# If you need to check license compatibility in code, you should use hardcoded license identifiers


##############################################################################
# aiter
##############################################################################
"""
aiter() is a built-in function for getting an asynchronous iterator from an object.

Key characteristics:
- Used with asynchronous iteration and async for loops
- Pairs with anext() for asynchronous iteration
- Returns an asynchronous iterator directly from an async iterable
- Part of Python's asynchronous programming features
- Only usable with objects that implement __aiter__ protocol
"""

import asyncio

# Define an asynchronous iterable class
class AsyncRange:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
    
    # This makes the class an async iterable
    def __aiter__(self):
        return AsyncRangeIterator(self.start, self.stop)


# Define an asynchronous iterator
class AsyncRangeIterator:
    def __init__(self, start, stop):
        self.current = start
        self.stop = stop
    
    # This makes the class an async iterator
    async def __anext__(self):
        await asyncio.sleep(0.1)  # Simulate I/O delay
        if self.current >= self.stop:
            raise StopAsyncIteration
        value = self.current
        self.current += 1
        return value


# Define an async function to demonstrate aiter() and anext()
async def demonstrate_aiter():
    # Create an async iterable
    async_range = AsyncRange(1, 5)
    
    # Get async iterator using aiter()
    iterator = aiter(async_range)
    
    # Get values using anext()
    print("Using anext() explicitly:")
    try:
        while True:
            value = await anext(iterator)
            print(f"Got value: {value}")
    except StopAsyncIteration:
        print("Iterator exhausted")
    
    # Equivalent using async for (more common)
    print("\nUsing async for:")
    async for value in AsyncRange(1, 5):
        print(f"Got value: {value}")
    
    # EXCEPTION CASE: Using aiter on non-async iterable
    try:
        aiter([1, 2, 3])  # Regular list doesn't implement __aiter__
    except TypeError as e:
        print(f"\nException with non-async iterable: {e}")


# Run the async demonstration
# To actually execute this, you would need to run the async function:
# asyncio.run(demonstrate_aiter())
print("\nTo run the aiter/anext demonstration, execute:")
print("asyncio.run(demonstrate_aiter())")

# For demonstration purposes, let's show the equivalent synchronous code
print("\nEquivalent synchronous code:")
numbers = range(1, 5)
iterator = iter(numbers)  # Like aiter() but for regular iterables
try:
    while True:
        value = next(iterator)  # Like anext() but for regular iterators
        print(f"Got value: {value}")
except StopIteration:
    print("Iterator exhausted")


##############################################################################
# anext
##############################################################################
"""
anext() is a built-in function that gets the next item from an asynchronous iterator.

Key characteristics:
- Used with asynchronous iteration alongside aiter()
- Must be awaited in an async function
- Can take a default value to return instead of raising StopAsyncIteration
- Part of Python's asynchronous programming features
- Works with any object that implements __anext__ protocol
"""

import asyncio

# Define an async generator function (simpler way to create async iterables)
async def async_generator(max_value):
    for i in range(max_value):
        await asyncio.sleep(0.1)  # Simulate I/O delay
        yield i


# Define an async function to demonstrate anext()
async def demonstrate_anext():
    # Create an async generator
    gen = async_generator(3)
    
    # Get values using anext()
    print("Using anext() with async generator:")
    
    # First value
    value = await anext(gen)
    print(f"First value: {value}")
    
    # Second value
    value = await anext(gen)
    print(f"Second value: {value}")
    
    # Third value
    value = await anext(gen)
    print(f"Third value: {value}")
    
    # Fourth value - would raise StopAsyncIteration, but we provide default
    value = await anext(gen, "END")
    print(f"Fourth value (default): {value}")
    
    # EXCEPTION CASE: Using anext without default on exhausted iterator
    try:
        value = await anext(gen)
    except StopAsyncIteration:
        print("Iterator exhausted as expected")
    
    # EXCEPTION CASE: Using anext on non-async iterator
    try:
        await anext(iter([1, 2, 3]))  # Regular iterator doesn't have __anext__
    except TypeError as e:
        print(f"Exception with non-async iterator: {e}")


# Run the async demonstration
# To actually execute this, you would need to run the async function:
# asyncio.run(demonstrate_anext())
print("\nTo run the anext demonstration, execute:")
print("asyncio.run(demonstrate_anext())")

# For demonstration purposes, let's show a more practical example
async def fetch_items(urls):
    """Simulates fetching items asynchronously from URLs"""
    async for url in async_generator(len(urls)):
        print(f"Fetching from {urls[url]}")
        yield f"Data from {urls[url]}"


async def process_urls():
    urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
    fetcher = fetch_items(urls)
    
    # Process first item immediately
    first_item = await anext(fetcher)
    print(f"Processed first item immediately: {first_item}")
    
    # Process remaining items in loop
    async for item in fetcher:
        print(f"Processed item in loop: {item}")


print("\nPractical URL processing example:")
print("To run: asyncio.run(process_urls())")