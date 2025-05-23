#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Special Methods (Dunder Methods) - Object Initialization & Lifecycle
==========================================================================

Comprehensive demonstration of __new__, __init__, and __del__ methods.
"""

# ========================
# __new__ Method
# ========================
class NewExample:
    """
    __new__ is a static method responsible for creating a new instance.
    - First step in instance creation, called before __init__
    - Receives the class as first parameter (cls), then constructor arguments
    - Must return an instance (typically of the class being instantiated)
    - One of the few methods that doesn't receive self as first parameter
    - Rare to override except for singleton pattern, immutable subclassing, or metaclass behavior
    """
    def __new__(cls, value):
        print(f"[NewExample.__new__] Creating instance of {cls.__name__} with value={value}")
        # Create the instance by calling the parent's __new__ method
        instance = super().__new__(cls)
        # You can set attributes here, but it's usually done in __init__
        instance._created_in_new = True
        print(f"[NewExample.__new__] Returning new instance: {hex(id(instance))}")
        return instance
    
    def __init__(self, value):
        print(f"[NewExample.__init__] Initializing instance {hex(id(self))} with value={value}")
        self.value = value
        print(f"[NewExample.__init__] Was created in __new__? {self._created_in_new}")


# Immutable type subclassing with __new__
class ImmutablePoint(tuple):
    """Example of subclassing immutable type (tuple) using __new__."""
    def __new__(cls, x, y):
        print(f"[ImmutablePoint.__new__] Creating point with coordinates ({x}, {y})")
        # For immutable types, we must create the complete object in __new__
        return super().__new__(cls, (x, y))
    
    def __init__(self, x, y):
        # This actually doesn't do anything for immutable objects
        print(f"[ImmutablePoint.__init__] Init called, but tuple is already created")
        # Cannot modify the tuple here - it's already created and immutable
        # self[0] = x  # This would raise TypeError
    
    @property
    def x(self):
        return self[0]
    
    @property
    def y(self):
        return self[1]


# Singleton pattern with __new__
class Singleton:
    """Demonstration of singleton pattern using __new__."""
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print(f"[Singleton.__new__] First call - creating singleton instance")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        else:
            print(f"[Singleton.__new__] Returning existing instance: {hex(id(cls._instance))}")
        return cls._instance
    
    def __init__(self, name=None):
        if not self._initialized:
            print(f"[Singleton.__init__] First initialization with name: {name}")
            self.name = name
            self._initialized = True
        else:
            print(f"[Singleton.__init__] Already initialized, ignoring name: {name}")


# Control instance creation with __new__
class Restricted:
    """Restricts instance creation based on arguments."""
    _max_instances = 3
    _instances_created = 0
    
    def __new__(cls, value):
        if value < 0:
            print(f"[Restricted.__new__] Rejected negative value: {value}")
            return None  # Reject negative values by returning None instead of an instance
        
        if cls._instances_created >= cls._max_instances:
            print(f"[Restricted.__new__] Max instances ({cls._max_instances}) reached, rejecting")
            return None
        
        print(f"[Restricted.__new__] Creating instance #{cls._instances_created+1} with value {value}")
        instance = super().__new__(cls)
        cls._instances_created += 1
        return instance
    
    def __init__(self, value):
        # This only executes if __new__ returned an instance
        print(f"[Restricted.__init__] Initializing with value: {value}")
        self.value = value


# ========================
# __init__ Method
# ========================
class Person:
    """
    __init__ initializes an instance after it's created by __new__.
    - Second step in instance creation
    - Receives the new instance as self, followed by constructor arguments
    - Should initialize attributes, not create the instance
    - Must return None (any other return value raises TypeError)
    - Most commonly overridden dunder method
    """
    def __init__(self, name, age, email=None):
        print(f"[Person.__init__] Initializing person: {name}, {age}")
        # Basic attributes
        self.name = name
        self.age = age
        self.email = email
        
        # Derived attributes calculated during initialization
        self.is_adult = age >= 18
        self._contacts = []
    
    def add_contact(self, contact):
        self._contacts.append(contact)
    
    def __str__(self):
        return f"Person(name={self.name}, age={self.age}, is_adult={self.is_adult})"


# Class hierarchy with __init__ parameter forwarding
class Shape:
    def __init__(self, color):
        print(f"[Shape.__init__] Base initialization with color: {color}")
        self.color = color

class Circle(Shape):
    def __init__(self, radius, color="black"):
        print(f"[Circle.__init__] Initializing circle with radius: {radius}, color: {color}")
        # Call parent's __init__ to initialize inherited attributes
        super().__init__(color)
        self.radius = radius
        
    def __str__(self):
        return f"Circle(radius={self.radius}, color={self.color})"


# ========================
# __del__ Method
# ========================
class ResourceManager:
    """
    __del__ is the finalizer method called when an object is being destroyed.
    - Called when object reference count reaches zero
    - Not guaranteed to be called in all cases (e.g., program termination)
    - Used primarily for cleanup of external resources
    - Should handle exceptions internally to prevent issues with garbage collection
    - Not a true destructor - Python uses garbage collection
    """
    def __init__(self, resource_id):
        print(f"[ResourceManager.__init__] Acquiring resource: {resource_id}")
        self.resource_id = resource_id
        # Simulate acquiring external resource
        self.resource = {"id": resource_id, "status": "open", "data": "Resource data"}
        
    def use_resource(self):
        print(f"[ResourceManager.use] Using resource: {self.resource_id}")
        if not hasattr(self, 'resource') or self.resource is None:
            raise ValueError("Resource already released")
        return self.resource["data"]
        
    def __del__(self):
        print(f"[ResourceManager.__del__] Finalizer called for resource: {self.resource_id}")
        if hasattr(self, 'resource') and self.resource is not None:
            print(f"[ResourceManager.__del__] Explicitly releasing resource: {self.resource_id}")
            # Cleanup code - would close files, network connections, etc.
            self.resource["status"] = "closed"
            self.resource = None
        else:
            print(f"[ResourceManager.__del__] Resource {self.resource_id} already released")


# File-like resource with proper cleanup
class FileResource:
    def __init__(self, filename):
        print(f"[FileResource.__init__] Opening file: {filename}")
        self.filename = filename
        # In a real scenario, this would be: self.file = open(filename, 'w')
        self.file = {"name": filename, "status": "open"}
        
    def write(self, data):
        print(f"[FileResource.write] Writing to {self.filename}: {data}")
        if self.file["status"] != "open":
            raise ValueError("File already closed")
        # In a real scenario: self.file.write(data)
        
    def close(self):
        print(f"[FileResource.close] Explicitly closing file: {self.filename}")
        if self.file["status"] == "open":
            # In a real scenario: self.file.close()
            self.file["status"] = "closed"
            
    def __del__(self):
        print(f"[FileResource.__del__] Finalizer called for file: {self.filename}")
        # Ensure file is closed if not explicitly closed
        if hasattr(self, 'file') and self.file["status"] == "open":
            print(f"[FileResource.__del__] Auto-closing file in finalizer: {self.filename}")
            self.close()
        else:
            print(f"[FileResource.__del__] File already closed: {self.filename}")


# ========================
# Complete Lifecycle Example
# ========================
class LifecycleDemo:
    """Demonstrates complete object lifecycle with all three methods."""
    
    def __new__(cls, name, *args, **kwargs):
        print(f"\n[LifecycleDemo.__new__] 1. Creating instance with name: {name}")
        instance = super().__new__(cls)
        print(f"[LifecycleDemo.__new__] 2. Instance created: {hex(id(instance))}")
        # We can set attributes in __new__, but it's cleaner to do it in __init__
        instance._created_at = "step 1"
        return instance
    
    def __init__(self, name, data=None):
        print(f"[LifecycleDemo.__init__] 3. Initializing instance {hex(id(self))}")
        self.name = name
        self.data = data or {}
        self._created_at = "step 2"  # Override the attribute set in __new__
        print(f"[LifecycleDemo.__init__] 4. Initialization complete: {self}")
    
    def __del__(self):
        print(f"[LifecycleDemo.__del__] 5. Finalizing instance {hex(id(self))}: {self.name}")
        print(f"[LifecycleDemo.__del__] 6. Cleaning up resources for: {self.name}")
    
    def __str__(self):
        return f"LifecycleDemo(name='{self.name}', data={self.data})"


# =================================
# Execution Flow and Demonstration
# =================================
def test_new_method():
    """Test cases for __new__ method."""
    print("\n===== TESTING __new__ METHOD =====")
    
    print("\n1. Basic __new__ override:")
    obj = NewExample(42)
    print(f"Created object: {obj}, value: {obj.value}")
    
    print("\n2. Immutable subclassing with __new__:")
    point = ImmutablePoint(10, 20)
    print(f"Created point: {point}, coordinates: ({point.x}, {point.y})")
    
    print("\n3. Singleton pattern with __new__:")
    s1 = Singleton("First")
    s2 = Singleton("Second")
    s3 = Singleton("Third")
    print(f"s1: {s1.name}, id: {hex(id(s1))}")
    print(f"s2: {s2.name}, id: {hex(id(s2))}")
    print(f"s3: {s3.name}, id: {hex(id(s3))}")
    print(f"All references to same object: {s1 is s2 is s3}")
    
    print("\n4. Controlling instance creation with __new__:")
    r1 = Restricted(10)  # Should create instance
    r2 = Restricted(20)  # Should create instance
    r3 = Restricted(30)  # Should create instance
    r4 = Restricted(40)  # Should be rejected (max instances)
    r5 = Restricted(-5)  # Should be rejected (negative value)
    
    instances = [r1, r2, r3, r4, r5]
    print(f"Created instances: {len([i for i in instances if i is not None])}/5")
    for i, instance in enumerate(instances):
        if instance is not None:
            print(f"  r{i+1}: value={instance.value}")
        else:
            print(f"  r{i+1}: None (rejected)")

def test_init_method():
    """Test cases for __init__ method."""
    print("\n===== TESTING __init__ METHOD =====")
    
    print("\n1. Basic initialization:")
    person = Person("Alice", 30, "alice@example.com")
    print(f"Person created: {person}")
    print(f"Person attributes: name={person.name}, age={person.age}, is_adult={person.is_adult}")
    
    print("\n2. Inheritance with __init__ chaining:")
    circle = Circle(5.0, "blue")
    print(f"Circle created: {circle}")
    print(f"Circle attributes: radius={circle.radius}, color={circle.color}")
    
    # Default parameter
    default_circle = Circle(3.0)
    print(f"Default circle: {default_circle}")

def test_del_method():
    """Test cases for __del__ method."""
    print("\n===== TESTING __del__ METHOD =====")
    
    print("\n1. Basic resource cleanup:")
    resource = ResourceManager("DB_CONNECTION")
    data = resource.use_resource()
    print(f"Resource data: {data}")
    print("Explicitly deleting resource...")
    del resource  # This should trigger __del__
    
    print("\n2. File resource with explicit/implicit cleanup:")
    # Scenario 1: Explicit cleanup
    file1 = FileResource("example1.txt")
    file1.write("Hello, World!")
    print("Explicitly closing file1...")
    file1.close()
    print("Deleting file1 reference...")
    del file1
    
    # Scenario 2: Implicit cleanup via __del__
    file2 = FileResource("example2.txt")
    file2.write("This file will be closed by __del__")
    print("Deleting file2 without explicit close...")
    del file2

def test_complete_lifecycle():
    """Test the complete object lifecycle."""
    print("\n===== TESTING COMPLETE LIFECYCLE =====")
    
    # Create the object
    print("\nCreating lifecycle demo object:")
    lifecycle = LifecycleDemo("test_object", {"key": "value"})
    
    # Use the object
    print("\nUsing the object:")
    print(f"Object name: {lifecycle.name}")
    print(f"Object data: {lifecycle.data}")
    print(f"Object created at: {lifecycle._created_at}")
    
    # Delete the object
    print("\nExplicitly deleting the object:")
    del lifecycle
    
    # Create and let it be garbage collected
    print("\nCreating temporary object to be garbage collected:")
    LifecycleDemo("temporary")
    print("Temporary object goes out of scope here")


if __name__ == "__main__":
    test_new_method()
    test_init_method()
    test_del_method()
    test_complete_lifecycle()
    
    print("\n===== PROGRAM EXITING =====")
    print("Any remaining objects with __del__ will be cleaned up now")