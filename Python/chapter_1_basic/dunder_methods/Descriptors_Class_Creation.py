"""
Comprehensive Guide to Python Special Methods (Dunder Methods) and Descriptors

This file provides an in-depth explanation of Python's special methods (dunder methods) 
and descriptors, focusing on the requested topics. Each concept is explained in detail 
with examples, adhering to PEP-8 standards and using high-level DSA concepts where applicable.

The content is structured as follows:
1. Descriptors (__get__, __set__, __delete__, __set_name__)
2. Class Creation (__prepare__, __instancecheck__, __subclasscheck__)
"""

# === Part 1: Descriptors ===
"""
Descriptors are a powerful feature in Python that allow you to customize attribute access, 
modification, and deletion. A descriptor is an object that defines at least one of the 
following methods: __get__, __set__, or __delete__. Descriptors are typically used to 
implement properties, methods, or managed attributes in a class.

Key Concepts:
- A descriptor is "invoked" when an attribute is accessed, set, or deleted.
- Descriptors are only meaningful when defined as class attributes (not instance attributes).
- There are two types of descriptors:
  1. Data Descriptors: Implement both __get__ and __set__ (and optionally __delete__).
  2. Non-Data Descriptors: Implement only __get__.
- The __set_name__ method (introduced in Python 3.6) is used to inform the descriptor about 
  the name of the attribute it manages.
"""

class DescriptorExample:
    """
    A custom descriptor class demonstrating __get__, __set__, __delete__, and __set_name__.
    This descriptor manages an attribute and enforces type checking.
    """
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self._name = None  # Will be set by __set_name__

    def __set_name__(self, owner, name):
        """
        __set_name__(self, owner, name)
        - Called when the descriptor is assigned to a class attribute.
        - Parameters:
            - owner: The class where the descriptor is defined.
            - name: The name of the attribute the descriptor is assigned to.
        - Purpose: Allows the descriptor to know the name of the attribute it manages.
        """
        self._name = name

    def __get__(self, obj, objtype=None):
        """
        __get__(self, obj, objtype=None)
        - Called when the attribute is accessed (e.g., obj.attr).
        - Parameters:
            - obj: The instance of the class (None if accessed via the class, e.g., MyClass.attr).
            - objtype: The class of the instance (always provided, even for class access).
        - Returns: The value of the managed attribute.
        - Purpose: Customizes attribute access.
        """
        if obj is None:
            # If accessed via the class (e.g., MyClass.attr), return the descriptor itself
            return self
        # Retrieve the value from the instance's __dict__ using the attribute name
        return obj.__dict__.get(self._name, None)

    def __set__(self, obj, value):
        """
        __set__(self, obj, value)
        - Called when the attribute is set (e.g., obj.attr = value).
        - Parameters:
            - obj: The instance of the class.
            - value: The value being assigned to the attribute.
        - Purpose: Customizes attribute modification, e.g., for type checking or validation.
        """
        if not isinstance(value, self.expected_type):
            raise TypeError(f"Expected {self.expected_type}, got {type(value)}")
        # Store the value in the instance's __dict__
        obj.__dict__[self._name] = value

    def __delete__(self, obj):
        """
        __delete__(self, obj)
        - Called when the attribute is deleted (e.g., del obj.attr).
        - Parameters:
            - obj: The instance of the class.
        - Purpose: Customizes attribute deletion.
        """
        if self._name in obj.__dict__:
            del obj.__dict__[self._name]
        else:
            raise AttributeError(f"{self._name} is not set")

# Example usage of the descriptor
class Person:
    """
    A class that uses the DescriptorExample to manage its 'age' attribute.
    """
    age = DescriptorExample(expected_type=int)

    def __init__(self, age):
        self.age = age  # Calls __set__ in DescriptorExample

# Test the descriptor
if __name__ == "__main__":
    print("=== Testing Descriptors ===")
    person = Person(age=30)
    print(f"Person's age: {person.age}")  # Calls __get__, outputs 30

    try:
        person.age = "invalid"  # Calls __set__, raises TypeError
    except TypeError as e:
        print(f"TypeError: {e}")

    del person.age  # Calls __delete__
    try:
        print(person.age)  # Calls __get__, outputs None
    except AttributeError as e:
        print(f"AttributeError: {e}")

    # Accessing descriptor via class
    print(f"Descriptor object: {Person.age}")  # Returns the descriptor instance


# === Part 2: Class Creation ===
"""
Python provides several special methods that allow you to customize the creation and 
behavior of classes. These methods are typically defined in metaclasses, which are classes 
that create other classes. The methods we will cover are:
- __prepare__: Customizes the namespace preparation before class creation.
- __instancecheck__: Customizes behavior of isinstance().
- __subclasscheck__: Customizes behavior of issubclass().
"""

# === __prepare__ ===
"""
__prepare__(metacls, name, bases, **kwargs)
- Called before the class body is executed, during class creation.
- Parameters:
    - metacls: The metaclass (the class of the class being created).
    - name: The name of the class being created.
    - bases: A tuple of base classes.
    - **kwargs: Additional keyword arguments (e.g., from class definition).
- Returns: A mapping (usually a dictionary) that will be used as the namespace for the class.
- Purpose: Allows customization of the namespace, e.g., using an ordered dictionary or 
  a custom mapping to control attribute order or behavior.
"""

from collections import OrderedDict

class OrderedMeta(type):
    """
    A metaclass that uses __prepare__ to ensure class attributes are stored in an 
    OrderedDict, preserving the order of definition.
    """
    @classmethod
    def __prepare__(metacls, name, bases, **kwargs):
        """
        Customizes the namespace to use an OrderedDict instead of a regular dict.
        """
        return OrderedDict()

    def __new__(metacls, name, bases, namespace, **kwargs):
        """
        Creates the class, preserving the ordered namespace.
        """
        print(f"Namespace during class creation: {list(namespace.keys())}")
        return super().__new__(metacls, name, bases, dict(namespace))

# Example usage of __prepare__
class OrderedClass(metaclass=OrderedMeta):
    x = 1
    y = 2
    z = 3

if __name__ == "__main__":
    print("\n=== Testing __prepare__ ===")
    print(f"OrderedClass attributes: {list(OrderedClass.__dict__.keys())}")


# === __instancecheck__ and __subclasscheck__ ===
"""
These methods are used to customize the behavior of isinstance() and issubclass(), 
respectively. They are typically defined in a metaclass.

- __instancecheck__(cls, instance)
    - Called by isinstance(instance, cls).
    - Parameters:
        - cls: The class being checked against.
        - instance: The object being tested.
    - Returns: True if instance is considered an instance of cls, False otherwise.
    - Purpose: Allows custom instance checking, e.g., for duck typing or virtual subclasses.

- __subclasscheck__(cls, subclass)
    - Called by issubclass(subclass, cls).
    - Parameters:
        - cls: The class being checked against.
        - subclass: The class being tested.
    - Returns: True if subclass is considered a subclass of cls, False otherwise.
    - Purpose: Allows custom subclass checking, e.g., for virtual subclasses.
"""

class CustomMeta(type):
    """
    A metaclass that customizes isinstance() and issubclass() behavior.
    """
    def __instancecheck__(cls, instance):
        """
        Customizes isinstance(). Here, we consider an object an instance of CustomClass 
        if it has a 'quack' method (duck typing).
        """
        return hasattr(instance, "quack")

    def __subclasscheck__(cls, subclass):
        """
        Customizes issubclass(). Here, we consider a class a subclass of CustomClass 
        if it defines a 'quack' method in its namespace.
        """
        return "quack" in subclass.__dict__

# Example usage of __instancecheck__ and __subclasscheck__
class CustomClass(metaclass=CustomMeta):
    pass

class Duck:
    def quack(self):
        print("Quack!")

class FakeDuck:
    def quack(self):
        print("Fake quack!")

if __name__ == "__main__":
    print("\n=== Testing __instancecheck__ and __subclasscheck__ ===")
    duck = Duck()
    fake_duck = FakeDuck()

    # Test __instancecheck__
    print(f"Is duck an instance of CustomClass? {isinstance(duck, CustomClass)}")  # True
    print(f"Is fake_duck an instance of CustomClass? {isinstance(fake_duck, CustomClass)}")  # True
    print(f"Is int an instance of CustomClass? {isinstance(1, CustomClass)}")  # False

    # Test __subclasscheck__
    print(f"Is Duck a subclass of CustomClass? {issubclass(Duck, CustomClass)}")  # True
    print(f"Is FakeDuck a subclass of CustomClass? {issubclass(FakeDuck, CustomClass)}")  # True
    print(f"Is int a subclass of CustomClass? {issubclass(int, CustomClass)}")  # False
