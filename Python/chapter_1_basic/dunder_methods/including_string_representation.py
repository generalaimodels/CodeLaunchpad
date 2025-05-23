"""
Python Special Methods (Dunder Methods) - Detailed Explanation and Implementation

This file provides an in-depth explanation of Python's special methods (dunder methods) related to:
1. Object Initialization & Lifecycle
2. String Representation

Each method is explained in detail with examples, use cases, and best practices.
All code follows PEP-8 standards for readability and maintainability.
"""

# Section 1: Object Initialization & Lifecycle
"""
Object Initialization & Lifecycle Methods
These methods are responsible for creating, initializing, and destroying objects in Python.
They form the backbone of object lifecycle management in Python's object-oriented programming.
"""

class LifecycleDemo:
    """
    A class to demonstrate object lifecycle methods: __new__, __init__, and __del__.
    """

    def __new__(cls, *args, **kwargs):
        """
        __new__ Method:
        - Purpose: Responsible for creating a new instance of the class.
        - Called before __init__.
        - Returns a new instance of the class (or another object if needed).
        - Rarely overridden unless you need to control object creation (e.g., singletons).
        - Static method, takes 'cls' as the first parameter (class itself).
        - Use case: Singleton pattern, custom object creation logic.

        Parameters:
        - cls: The class being instantiated.
        - *args, **kwargs: Arguments passed during object creation.

        Returns:
        - A new instance of the class.
        """
        print(f"__new__ called: Creating a new instance of {cls.__name__}")
        # Call the parent class's __new__ to create the instance
        instance = super(LifecycleDemo, cls).__new__(cls)
        return instance

    def __init__(self, name):
        """
        __init__ Method:
        - Purpose: Initializes the instance after it is created by __new__.
        - Does not return anything (implicitly returns None).
        - Called automatically after __new__ returns the instance.
        - Use case: Set up initial attributes, state, or resources.

        Parameters:
        - self: The instance being initialized.
        - name: A string parameter to set as an instance attribute.
        """
        print(f"__init__ called: Initializing instance with name '{name}'")
        self.name = name

    def __del__(self):
        """
        __del__ Method:
        - Purpose: Called when the instance is about to be destroyed (garbage collected).
        - Not guaranteed to be called (e.g., if program exits abruptly).
        - Use case: Clean up resources (e.g., close files, release memory).
        - Avoid circular references as they may prevent __del__ from being called.

        Parameters:
        - self: The instance being destroyed.
        """
        print(f"__del__ called: Destroying instance with name '{self.name}'")

# Example usage of LifecycleDemo
print("=== Object Lifecycle Example ===")
obj = LifecycleDemo("TestObject")
print(f"Object name: {obj.name}")
# Object will be destroyed when it goes out of scope or is explicitly deleted
del obj
print("Object deletion complete\n")


# Section 2: String Representation
"""
String Representation Methods
These methods control how objects are represented as strings, formatted, or converted to bytes.
They are essential for debugging, logging, and user-facing output.
"""

class StringRepresentationDemo:
    """
    A class to demonstrate string representation methods: __str__, __repr__, __format__, __bytes__.
    """

    def __init__(self, value, encoding="utf-8"):
        """
        Initialize the instance with a value and encoding.

        Parameters:
        - value: An integer value to store in the object.
        - encoding: Encoding to use for byte conversion (default: 'utf-8').
        """
        self.value = value
        self.encoding = encoding

    def __str__(self):
        """
        __str__ Method:
        - Purpose: Provides a user-friendly string representation of the object.
        - Called by str() and print().
        - Should return a string that is readable and informative for end users.
        - Use case: Display object in a human-readable format.

        Returns:
        - A string representing the object in a user-friendly manner.
        """
        return f"StringRepresentationDemo object with value: {self.value}"

    def __repr__(self):
        """
        __repr__ Method:
        - Purpose: Provides a developer-friendly string representation of the object.
        - Called by repr() and when object is inspected in interactive shells.
        - Should return a string that, ideally, could be used to recreate the object.
        - Use case: Debugging, logging, developer tools.

        Returns:
        - A string representing the object in a detailed, unambiguous manner.
        """
        return f"StringRepresentationDemo(value={self.value}, encoding='{self.encoding}')"

    def __format__(self, format_spec):
        """
        __format__ Method:
        - Purpose: Controls how the object is formatted when using string formatting methods.
        - Called by format() and f-strings with format specifiers.
        - Format specifiers are passed as the 'format_spec' parameter.
        - Use case: Custom formatting for numbers, dates, etc.

        Parameters:
        - format_spec: A string specifying the format (e.g., '.2f', 'x').

        Returns:
        - A formatted string based on the format specification.
        """
        if format_spec == "x":
            return f"{self.value:x}"  # Hexadecimal lowercase
        elif format_spec == "X":
            return f"{self.value:X}"  # Hexadecimal uppercase
        elif format_spec == "b":
            return f"{self.value:b}"  # Binary
        else:
            return str(self.value)  # Default to string representation

    def __bytes__(self):
        """
        __bytes__ Method:
        - Purpose: Converts the object to a bytes representation.
        - Called by bytes().
        - Use case: Serialize object data, network communication, file storage.

        Returns:
        - A bytes object representing the instance.
        """
        return str(self.value).encode(self.encoding)

# Example usage of StringRepresentationDemo
print("=== String Representation Example ===")
obj = StringRepresentationDemo(255)

# __str__ example
print("str(obj):", str(obj))
print("print(obj):", obj)

# __repr__ example
print("repr(obj):", repr(obj))
print("Interactive shell representation:", obj.__repr__())

# __format__ example
print("format(obj, 'x'):", format(obj, "x"))  # Hexadecimal lowercase
print("format(obj, 'X'):", format(obj, "X"))  # Hexadecimal uppercase
print("format(obj, 'b'):", format(obj, "b"))  # Binary
print(f"f-string example: {obj:x}")  # Using f-string with format specifier

# __bytes__ example
print("bytes(obj):", bytes(obj))
print("Object lifecycle and string representation demo complete")

# Best Practices and Notes:
"""
1. Always implement __repr__ for debugging purposes; it should be as detailed as possible.
2. Implement __str__ for user-friendly output; fall back to __repr__ if __str__ is not defined.
3. Use __new__ sparingly, only when you need to control instance creation.
4. Be cautious with __del__; it is not a destructor in the traditional sense and is not guaranteed to run.
5. Use __format__ for custom formatting needs, especially for numerical or specialized data types.
6. Use __bytes__ when dealing with binary data or serialization.
"""