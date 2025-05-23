#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Attribute & Type Functions Guide
================================================

This guide covers the following built-in functions:
- isinstance(): Type checking with inheritance support
- hasattr(): Attribute existence checking
- getattr(): Safe attribute access with defaults
- setattr(): Dynamic attribute setting
- delattr(): Dynamic attribute deletion

These functions enable dynamic object introspection and manipulation,
which are key for writing flexible, maintainable, and powerful Python code.
"""

#######################################################
# isinstance(obj, class_or_tuple) - Type Checking
#######################################################

def isinstance_examples():
    """
    isinstance(object, classinfo) -> bool
    
    Returns True if the object is an instance of the class or a subclass thereof.
    If classinfo is a tuple of classes, return True if object is an instance of any of these classes.
    """
    
    # Basic usage with primitive types
    num = 42
    text = "Hello, World!"
    decimal = 3.14
    
    print(f"num is int: {isinstance(num, int)}")                  # True
    print(f"text is str: {isinstance(text, str)}")                # True #isinstance isinstance
    print(f"decimal is float: {isinstance(decimal, float)}")      # True
    
    # Checking against multiple types (using a tuple)
    print(f"num is int or float: {isinstance(num, (int, float))}")  # True isinstance
    print(f"text is int or float: {isinstance(text, (int, float))}") # False
    
    # isinstance works with inheritance
    class Animal:
        pass
    
    class Dog(Animal):
        pass
    
    class Cat(Animal):
        pass
    
    dog = Dog()
    cat = Cat()
    
    print(f"dog is Dog: {isinstance(dog, Dog)}")           # True
    print(f"dog is Animal: {isinstance(dog, Animal)}")     # True - inheritance!
    print(f"dog is Cat: {isinstance(dog, Cat)}")           # False
    
    # Works with built-in inheritance too
    class MyDict(dict):
        pass
    
    my_dict = MyDict()
    print(f"my_dict is MyDict: {isinstance(my_dict, MyDict)}")  # True
    print(f"my_dict is dict: {isinstance(my_dict, dict)}")      # True
    
    # Advanced usage with abstract base classes
    from collections.abc import Sequence, Mapping
    
    print(f"list is Sequence: {isinstance([], Sequence)}")      # True
    print(f"dict is Mapping: {isinstance({}, Mapping)}")        # True
    print(f"str is Sequence: {isinstance('abc', Sequence)}")    # True
    
    # IMPORTANT: isinstance vs type()
    # isinstance respects inheritance, type() doesn't
    
    print(f"type(dog) is Animal: {type(dog) is Animal}")       # False
    print(f"isinstance(dog, Animal): {isinstance(dog, Animal)}") # True
    
    # Exception cases:
    # 1. Second argument must be a type or tuple of types
    try:
        isinstance(42, "int")  # TypeError
    except TypeError as e:
        print(f"Error with non-type second argument: {e}")
    
    # 2. None is not considered a subclass of any type
    print(f"None is int: {isinstance(None, int)}")  # False


#######################################################
# hasattr(obj, name) - Attribute Existence Checking
#######################################################

def hasattr_examples():
    """
    hasattr(object, name) -> bool
    
    Returns True if the object has an attribute with the given name.
    This is done by calling getattr(object, name) and catching AttributeError.
    """
    
    class Person:
        def __init__(self):
            self.name = "Alice"
            self.age = 30
        
        def greet(self):
            return f"Hello, my name is {self.name}"
    
    person = Person()
    
    # Basic attribute checking
    print(f"person has 'name': {hasattr(person, 'name')}")         # True #hasattr
    print(f"person has 'age': {hasattr(person, 'age')}")           # True
    print(f"person has 'address': {hasattr(person, 'address')}")   # False
    
    # Checking for methods
    print(f"person has 'greet': {hasattr(person, 'greet')}")       # True
    
    # Checking built-in attributes
    print(f"person has '__dict__': {hasattr(person, '__dict__')}")  # True
    print(f"person has '__class__': {hasattr(person, '__class__')}") # True
    
    # Checking attributes on built-in types
    print(f"list has 'append': {hasattr([], 'append')}")           # True
    print(f"str has 'lower': {hasattr('hello', 'lower')}")         # True
    
    # For exception handling, 'hasattr' catches AttributeError internally
    # Dynamic class example
    class DynamicObject:
        def __getattr__(self, name):
            if name.startswith('compute_'):
                # Dangerous: this could raise other exceptions not caught by hasattr
                return lambda x: int(name.split('_')[1]) * x
            raise AttributeError(f"{name} not found")
    
    dyn = DynamicObject()
    
    # hasattr internally uses try-except AttributeError
    print(f"dyn has 'compute_10': {hasattr(dyn, 'compute_10')}")  # True
    
    # IMPORTANT: If __getattr__ raises a different exception, hasattr will propagate it
    class BuggyObject:
        def __getattr__(self, name):
            if name == "trigger_error":
                1/0  # ZeroDivisionError
            return None
    
    buggy = BuggyObject()
    
    try:
        result = hasattr(buggy, "trigger_error")  # ZeroDivisionError
    except ZeroDivisionError:
        print("hasattr propagates non-AttributeError exceptions!")


#######################################################
# getattr(obj, name[, default]) - Safe Attribute Access
#######################################################

def getattr_examples():
    """
    getattr(object, name[, default]) -> value
    
    Returns the value of the named attribute of object. If not found,
    it returns the default value if provided, otherwise raises AttributeError.
    """
    
    class Config:
        def __init__(self):
            self.host = "localhost"
            self.port = 8080
            self.debug = True
    
    config = Config()
    
    # Basic attribute access
    host = getattr(config, "host")
    print(f"Host: {host}")  # localhost
    
    # Accessing non-existent attribute with default
    timeout = getattr(config, "timeout", 30)  # Returns default 30
    print(f"Timeout: {timeout}")
    
    # Without default - will raise AttributeError
    try:
        missing = getattr(config, "missing_attribute")  # AttributeError
    except AttributeError as e:
        print(f"Error: {e}")
    
    # Getting methods and calling them
    class Calculator:
        def add(self, a, b):
            return a + b
        
        def subtract(self, a, b):
            return a - b
    
    calc = Calculator()
    
    # Get method dynamically and call it
    operation = "add"
    method = getattr(calc, operation)
    result = method(5, 3)
    print(f"{operation}: {result}")  # add: 8
    
    # Using with built-in methods
    string_method = getattr("hello world", "upper")
    print(f"Result of string_method(): {string_method()}")  # HELLO WORLD
    
    # Advanced use case: Mock configuration system
    class FeatureFlags:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def is_enabled(self, feature_name, default=False):
            return getattr(self, feature_name, default)
    
    features = FeatureFlags(dark_mode=True, beta_feature=False)
    
    print(f"Dark mode enabled: {features.is_enabled('dark_mode')}")         # True
    print(f"Beta feature enabled: {features.is_enabled('beta_feature')}")    # False
    print(f"New feature enabled: {features.is_enabled('new_feature')}")      # False (default)
    
    # IMPORTANT: getattr with properties behaves differently than direct access
    class User:
        def __init__(self, name):
            self._name = name
        
        @property
        def name(self):
            print("Property accessed")
            return self._name
    
    user = User("Bob")
    
    # Direct access triggers the property
    # user.name  # Would print "Property accessed"
    
    # getattr also triggers the property
    name = getattr(user, "name")  # Prints "Property accessed"
    print(f"User name: {name}")


#######################################################
# setattr(obj, name, value) - Dynamic Attribute Setting
#######################################################

def setattr_examples():
    """
    setattr(object, name, value)
    
    Sets the named attribute on the given object to the specified value.
    Equivalent to object.name = value
    """
    
    class Person:
        def __init__(self):
            self.name = "Unknown"
            self.age = 0
    
    person = Person()
    
    # Basic attribute setting
    print(f"Before: {person.name}")  # Unknown
    setattr(person, "name", "Charlie")
    print(f"After: {person.name}")   # Charlie
    
    # Setting non-existent attributes (creates them)
    setattr(person, "email", "charlie@example.com")
    print(f"New attribute: {person.email}")  # charlie@example.com
    
    # Setting attributes dynamically from user input
    user_field = "location"
    user_value = "New York"
    setattr(person, user_field, user_value)
    print(f"Dynamic attribute: {person.location}")  # New York
    
    # Using with a dictionary of values
    person_data = {
        "occupation": "Engineer",
        "salary": 75000,
        "department": "R&D"
    }
    
    for key, value in person_data.items():
        setattr(person, key, value)
    
    print(f"Occupation: {person.occupation}")  # Engineer
    print(f"Salary: {person.salary}")          # 75000
    
    # Setting methods dynamically
    def sing(self):
        return f"{self.name} is singing"
    
    setattr(Person, "sing", sing)
    print(person.sing())  # Charlie is singing
    
    # Special cases and limitations:
    
    # 1. setattr respects property setters
    class ProtectedPerson:
        def __init__(self):
            self._age = 0
        
        @property
        def age(self):
            return self._age
        
        @age.setter
        def age(self, value):
            if value < 0:
                raise ValueError("Age cannot be negative")
            self._age = value
    
    protected = ProtectedPerson()
    
    try:
        setattr(protected, "age", -10)  # Raises ValueError from the setter
    except ValueError as e:
        print(f"Error: {e}")
    
    setattr(protected, "age", 25)  # Works fine
    print(f"Protected age: {protected.age}")  # 25
    
    # 2. Cannot set attributes on built-in immutable types
    try:
        setattr("hello", "new_attr", "value")  # TypeError
    except TypeError as e:
        print(f"Error setting attribute on string: {e}")
    
    # 3. Setting attributes on classes vs instances
    class Example:
        class_var = "I'm a class variable"
    
    ex1 = Example()
    ex2 = Example()
    
    # Setting attribute on instance
    setattr(ex1, "instance_var", "I'm an instance variable")
    print(f"ex1.instance_var: {ex1.instance_var}")
    print(f"hasattr(ex2, 'instance_var'): {hasattr(ex2, 'instance_var')}")  # False
    
    # Setting attribute on class affects all instances
    setattr(Example, "new_class_var", "I'm a new class variable")
    print(f"ex1.new_class_var: {ex1.new_class_var}")
    print(f"ex2.new_class_var: {ex2.new_class_var}")


#######################################################
# delattr(obj, name) - Dynamic Attribute Deletion
#######################################################

def delattr_examples():
    """
    delattr(object, name)
    
    Deletes the named attribute from the given object.
    Equivalent to `del object.name`
    """
    
    class ConfigObject:
        def __init__(self):
            self.server = "production"
            self.port = 443
            self.debug = False
            self.temp_value = "This will be deleted"
    
    config = ConfigObject()
    
    # Basic attribute deletion
    print(f"Before deletion: {hasattr(config, 'temp_value')}")  # True
    delattr(config, "temp_value")
    print(f"After deletion: {hasattr(config, 'temp_value')}")   # False
    
    # Deleting non-existent attribute raises AttributeError
    try:
        delattr(config, "non_existent")  # AttributeError
    except AttributeError as e:
        print(f"Error: {e}")
    
    # Dynamic deletion based on conditions
    attributes_to_clean = ["debug", "port"]
    
    for attr in attributes_to_clean:
        if hasattr(config, attr):
            delattr(config, attr)
    
    print(f"After cleaning: debug exists: {hasattr(config, 'debug')}")  # False
    print(f"After cleaning: port exists: {hasattr(config, 'port')}")    # False
    print(f"After cleaning: server exists: {hasattr(config, 'server')}") # True
    
    # Limitations and special cases:
    
    # 1. delattr respects property deleters
    class ProtectedConfig:
        def __init__(self):
            self._protected = "sensitive data"
            self.normal = "normal data"
        
        @property
        def protected(self):
            return self._protected
        
        @protected.deleter
        def protected(self):
            print("Running custom delete logic")
            self._protected = None
    
    p_config = ProtectedConfig()
    
    # This calls the deleter method
    delattr(p_config, "protected")
    print(f"Protected value after deletion: {p_config._protected}")  # None
    
    # 2. Cannot delete required attributes
    class RequiredFields:
        def __init__(self):
            self.id = 1
            self.name = "Required"
        
        def __delattr__(self, name):
            if name in ["id", "name"]:
                raise AttributeError(f"Cannot delete required attribute: {name}")
            super().__delattr__(name)
    
    rf = RequiredFields()
    rf.optional = "Can be deleted"
    
    try:
        delattr(rf, "id")  # AttributeError from __delattr__
    except AttributeError as e:
        print(f"Error: {e}")
    
    # Can delete optional attributes
    delattr(rf, "optional")
    print(f"optional exists: {hasattr(rf, 'optional')}")  # False
    
    # 3. Cannot delete attributes on built-in immutable types
    try:
        delattr("hello", "__class__")  # TypeError
    except TypeError as e:
        print(f"Error: {e}")
    
    # 4. Deleting class attributes vs instance attributes
    class Example:
        class_var = "I'm a class variable"
        another_var = "Another class variable"
    
    ex = Example()
    ex.instance_var = "I'm an instance variable"
    
    # Delete instance attribute
    delattr(ex, "instance_var")
    print(f"instance_var exists: {hasattr(ex, 'instance_var')}")  # False
    
    # Delete class attribute through the class
    delattr(Example, "another_var")
    print(f"another_var exists on class: {hasattr(Example, 'another_var')}")  # False
    print(f"another_var exists on instance: {hasattr(ex, 'another_var')}")    # False


#######################################################
# Practical Applications and Use Cases
#######################################################

def practical_examples():
    """
    This section demonstrates practical examples combining these functions.
    """
    # Creating a flexible configuration class
    class FlexibleConfig:
        """A configuration class that can be dynamically updated."""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def get(self, name, default=None):
            """Safely get a configuration value."""
            return getattr(self, name, default)
        
        def set(self, name, value):
            """Set a configuration value."""
            setattr(self, name, value)
        
        def has(self, name):
            """Check if a configuration exists."""
            return hasattr(self, name)
        
        def remove(self, name):
            """Remove a configuration if it exists."""
            if hasattr(self, name):
                delattr(self, name)
                return True
            return False
        
        def update(self, **kwargs):
            """Update multiple configurations at once."""
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Creating a plugin system
    class PluginManager:
        """Simple plugin manager that can load and manage plugins."""
        def __init__(self):
            self.plugins = {}
        
        def register_plugin(self, name, plugin_class):
            """Register a plugin if it has the required interface."""
            required_methods = ['initialize', 'run', 'cleanup']
            
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    raise ValueError(f"Plugin {name} missing required method: {method}")
            
            self.plugins[name] = plugin_class()
            print(f"Plugin {name} registered successfully")
        
        def execute_plugin(self, name, method, *args, **kwargs):
            """Execute a plugin method if the plugin exists."""
            if name not in self.plugins:
                return f"Plugin {name} not found"
            
            if not hasattr(self.plugins[name], method):
                return f"Method {method} not found in plugin {name}"
            
            plugin_method = getattr(self.plugins[name], method)
            return plugin_method(*args, **kwargs)
    
    # Example plugin
    class LoggerPlugin:
        def initialize(self):
            self.logs = []
            return "Logger initialized"
        
        def run(self, message):
            self.logs.append(message)
            return f"Logged: {message}"
        
        def cleanup(self):
            count = len(self.logs)
            self.logs = []
            return f"Cleaned up {count} logs"
    
    # Dynamic attribute factory pattern
    class AttributeFactory:
        """Creates and manages attributes dynamically based on data type."""
        def create_attribute(self, obj, name, value):
            """Create an attribute with appropriate validation."""
            if not hasattr(obj, '_validators'):
                setattr(obj, '_validators', {})
            
            validators = getattr(obj, '_validators')
            
            # Set validator based on the value type
            if isinstance(value, int):
                validators[name] = lambda x: isinstance(x, int)
            elif isinstance(value, str):
                validators[name] = lambda x: isinstance(x, str)
            elif isinstance(value, list):
                validators[name] = lambda x: isinstance(x, list)
            else:
                validators[name] = lambda x: True  # No validation
            
            # Create property with validation
            def getter(obj):
                return getattr(obj, f"_{name}", None)
            
            def setter(obj, val):
                validator = getattr(obj, '_validators').get(name)
                if not validator(val):
                    raise TypeError(f"Invalid type for {name}")
                setattr(obj, f"_{name}", val)
            
            prop = property(getter, setter)
            setattr(type(obj), name, prop)
            
            # Set the initial value
            setter(obj, value)
    
    # Demo the practical examples
    print("\n=== Flexible Config Demo ===")
    config = FlexibleConfig(host="localhost", port=8080, debug=True)
    print(f"Host: {config.get('host')}")  # localhost
    config.set("timeout", 30)
    print(f"Has timeout: {config.has('timeout')}")  # True
    config.remove("debug")
    print(f"Has debug: {config.has('debug')}")  # False
    config.update(host="new-server", max_connections=100)
    print(f"New host: {config.get('host')}")  # new-server
    
    print("\n=== Plugin Manager Demo ===")
    pm = PluginManager()
    pm.register_plugin("logger", LoggerPlugin)
    print(pm.execute_plugin("logger", "initialize"))
    print(pm.execute_plugin("logger", "run", "Test message"))
    print(pm.execute_plugin("logger", "cleanup"))
    
    print("\n=== Attribute Factory Demo ===")
    class User:
        pass
    
    user = User()
    factory = AttributeFactory()
    
    factory.create_attribute(user, "name", "John Doe")
    factory.create_attribute(user, "age", 30)
    factory.create_attribute(user, "tags", ["user", "premium"])
    
    print(f"User name: {user.name}")  # John Doe
    print(f"User age: {user.age}")    # 30
    
    # This would raise TypeError due to validation
    try:
        user.age = "thirty"  # TypeError: Invalid type for age
    except TypeError as e:
        print(f"Error: {e}")
    
    user.age = 31  # Works fine
    print(f"Updated age: {user.age}")  # 31


#######################################################
# Run the examples
#######################################################

if __name__ == "__main__":
    print("\n\n=== isinstance() Examples ===\n")
    isinstance_examples()
    
    print("\n\n=== hasattr() Examples ===\n")
    hasattr_examples()
    
    print("\n\n=== getattr() Examples ===\n")
    getattr_examples()
    
    print("\n\n=== setattr() Examples ===\n")
    setattr_examples()
    
    print("\n\n=== delattr() Examples ===\n")
    delattr_examples()
    
    print("\n\n=== Practical Applications ===\n")
    practical_examples()