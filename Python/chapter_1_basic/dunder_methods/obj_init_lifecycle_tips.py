#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Special Methods (Dunder Methods) - Object Initialization & Lifecycle
==========================================================================

__new__, __init__, and __del__ methods control the lifecycle of Python objects.
"""

# ========================
# __new__ Method
# ========================
"""
__new__ is a static method responsible for creating and returning a new instance.
- Called before __init__
- Receives the class as first argument, followed by any args passed to constructor
- Must return an instance (usually of the class being instantiated)
- Rarely overridden except for:
  1. Implementing singletons
  2. Subclassing immutable types (str, int, tuple)
  3. Customizing instance creation and metaclass behavior
"""

class CustomNew:
    def __new__(cls, *args, **kwargs):
        print(f"1. __new__ called with class: {cls.__name__}")
        print(f"   Arguments: {args}, {kwargs}")
        # Create and return instance by calling parent's __new__
        instance = super().__new__(cls)
        print(f"2. Instance created: {instance}")
        return instance
    
    def __init__(self, value):
        print(f"3. __init__ called with self: {self}")
        print(f"   Initializing with value: {value}")
        self.value = value

# Example: Singleton pattern using __new__
class Singleton:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print(f"Creating new singleton instance: {cls._instance}")
        else:
            print(f"Returning existing singleton: {cls._instance}")
        return cls._instance
    
    def __init__(self, name=None):
        # Only initialize if name isn't already set (first creation)
        if not hasattr(self, 'name'):
            self.name = name
            print(f"Initializing singleton with name: {name}")
        else:
            print(f"Not re-initializing existing singleton")


# ========================
# __init__ Method
# ========================
"""
__init__ initializes the newly created instance.
- Called after __new__ returns an instance
- Receives the instance as self, followed by constructor arguments
- Should only initialize the instance, not create it
- Must return None (any other return value raises TypeError)
- Most common special method that you'll override
"""

class Person:
    def __init__(self, name, age):
        print(f"Initializing {name}, {age}")
        self.name = name
        self.age = age
        # Derived attributes
        self.is_adult = age >= 18
    
    def __str__(self):
        return f"{self.name} ({self.age})"


# ========================
# __del__ Method
# ========================
"""
__del__ is the finalizer, called when object is about to be destroyed.
- Called when object's reference count reaches zero
- Not a destructor - Python uses garbage collection
- Not guaranteed to be called (e.g., program crash, circular references)
- Primarily used for releasing external resources (file handles, network connections)
- Avoid creating new references to self inside __del__
- Should never raise exceptions
"""

class ResourceManager:
    def __init__(self, resource_id):
        print(f"Acquiring resource: {resource_id}")
        self.resource_id = resource_id
        # Simulate acquiring external resource
        self.resource = {"id": resource_id, "data": "Important data"}
    
    def __del__(self):
        print(f"Releasing resource: {self.resource_id}")
        # Clean up external resources
        self.resource = None


# ========================
# Complete Lifecycle Example
# ========================
class LifecycleExample:
    def __new__(cls, name, *args, **kwargs):
        print(f"\n1. __new__ called for {cls.__name__} with name={name}")
        instance = super().__new__(cls)
        print(f"2. Created instance: {instance}")
        return instance
    
    def __init__(self, name, data=None):
        print(f"3. __init__ called for {self}")
        self.name = name
        self.data = data or {}
        print(f"4. Initialized with name={name}, data={data}")
    
    def __del__(self):
        print(f"5. __del__ called for {self}")
        print(f"6. Cleaning up {self.name}")
    
    def __str__(self):
        return f"LifecycleExample(name={self.name})"


# =================================
# Execution Flow and Demonstration
# =================================
if __name__ == "__main__":
    print("\n=== CustomNew Example ===")
    obj = CustomNew(42)
    
    print("\n=== Singleton Example ===")
    s1 = Singleton("First")
    s2 = Singleton("Second")  # Should reuse instance, not reinitialize
    print(f"Same instance? {s1 is s2}")  # Should be True
    
    print("\n=== Person Example ===")
    person = Person("Alice", 30)
    print(person)
    
    print("\n=== ResourceManager Example ===")
    # Context (with) blocks ensure proper cleanup via __exit__, not __del__
    resource = ResourceManager("DB_CONNECTION")
    # Force deletion of resource
    print("Deleting resource explicitly...")
    del resource
    
    print("\n=== Lifecycle Example ===")
    # # Create, use, and delete an instance to demonstrate full lifecycle
    # lifecycle = LifecycleExample("test_object", {"purpose": "demonstration"})
    # print(f"Object created: {lifecycle}")
    
    print("\nExiting program - remaining objects will be cleaned up...")
    # LifecycleExample.__del__ will be called when lifecycle goes out of scope