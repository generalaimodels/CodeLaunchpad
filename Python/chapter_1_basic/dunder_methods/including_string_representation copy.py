# ====================================================================================
# PYTHON SPECIAL METHODS (DUNDER METHODS) GUIDE
# ====================================================================================

# ====================================================================================
# OBJECT INITIALIZATION & LIFECYCLE
# ====================================================================================

# 1. __new__(cls, *args, **kwargs)
# ------------------------------------------
# Purpose: Creates and returns a new instance of the class
# Called: Before __init__, when an instance is created (e.g., x = MyClass())
# Returns: Usually the new instance object
# Note: Rarely overridden unless implementing singletons, caching, or immutable types

class NewExample:
    def __new__(cls, *args, **kwargs):
        print(f"1. __new__ called with class: {cls.__name__}")
        # Must call parent's __new__ to actually create the instance
        instance = super().__new__(cls)
        # You can modify the instance before __init__ is called
        instance._created_at = "pre-initialization"
        return instance

    def __init__(self, value):
        print(f"2. __init__ called with value: {value}")
        print(f"   _created_at from __new__: {self._created_at}")
        self.value = value
        self._created_at = "post-initialization"


# 2. __init__(self, *args, **kwargs)
# ------------------------------------------
# Purpose: Initializes the newly created instance
# Called: Immediately after __new__ returns
# Returns: None (returning anything else raises TypeError)
# Note: Most commonly overridden special method for setting up an object

class Person:
    def __init__(self, name, age):
        print(f"Initializing Person object")
        # Validate inputs
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        if not isinstance(age, int) or age < 0:
            raise ValueError("Age must be a positive integer")
        
        # Initialize attributes
        self.name = name
        self.age = age
        # Private attributes convention uses underscore
        self._id = id(self)


# 3. __del__(self)
# ------------------------------------------
# Purpose: Finalizer method, called when object is being garbage collected
# Called: When object's reference count reaches zero (not immediately after del x)
# Returns: None
# Note: Not reliable for cleanup; use context managers (with) instead

class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Resource {name} acquired")
        self.file = open(f"{name}.temp", "w")  # Create a temporary file
    
    def __del__(self):
        print(f"Resource {self.name} is being cleaned up")
        # Cleanup resource when object is garbage collected
        try:
            self.file.close()
            # In real code, you might also delete the file here
            # import os
            # os.remove(f"{self.name}.temp")
        except:
            pass  # Handle any exceptions during cleanup


# ====================================================================================
# STRING REPRESENTATION
# ====================================================================================

# 4. __str__(self)
# ------------------------------------------
# Purpose: Human-readable string representation
# Called: By str(), print(), and format() when {} is used
# Returns: String
# Note: Should be readable for end users

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        # Return user-friendly representation
        return f"Point at ({self.x}, {self.y})"


# 5. __repr__(self)
# ------------------------------------------
# Purpose: Unambiguous string representation for developers
# Called: By repr(), in debuggers, and when __str__ is not defined
# Returns: String (ideally one that could recreate the object)
# Note: Should follow the convention that eval(repr(obj)) == obj when possible

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def __str__(self):
        return f"Rectangle with width={self.width} and height={self.height}"
    
    def __repr__(self):
        # Return code-like representation that could recreate the object
        return f"Rectangle({self.width}, {self.height})"


# 6. __format__(self, format_spec)
# ------------------------------------------
# Purpose: Customizes formatting with format() and f-strings
# Called: By format(), str.format(), and f-strings with format specs
# Returns: Formatted string
# Note: format_spec is the part after the colon in f"{obj:format_spec}"

class Money:
    def __init__(self, amount, currency="USD"):
        self.amount = amount
        self.currency = currency
    
    def __str__(self):
        return f"{self.currency} {self.amount:.2f}"
    
    def __format__(self, format_spec):
        # Default format
        if not format_spec:
            return str(self)
        
        # Format specifiers:
        # 'p' - plain (just number)
        # 's' - symbol
        # 'c' - code (e.g., USD)
        # 'f:2' - float with 2 decimal places
        
        symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
        symbol = symbols.get(self.currency, "")
        
        if format_spec == 'p':
            return f"{self.amount:.2f}"
        elif format_spec == 's':
            return f"{symbol}{self.amount:.2f}"
        elif format_spec == 'c':
            return f"{self.currency} {self.amount:.2f}"
        elif format_spec.startswith('f:'):
            # Extract decimal places from format spec
            try:
                decimal_places = int(format_spec[2:])
                return f"{self.amount:.{decimal_places}f}"
            except:
                # Fall back to default 2 decimal places
                return f"{self.amount:.2f}"
        else:
            # Handle other format specifiers as normal float formatting
            return f"{self.amount:{format_spec}}"


# 7. __bytes__(self)
# ------------------------------------------
# Purpose: Returns bytes representation of the object
# Called: By bytes(obj)
# Returns: A bytes object
# Note: Used for serialization or producing a byte-string version of the object

class Image:
    def __init__(self, width, height, data=None):
        self.width = width
        self.height = height
        # Simple example: each pixel is one byte (0-255)
        self.data = data or bytearray(width * height)
    
    def __str__(self):
        return f"Image({self.width}x{self.height})"
    
    def __bytes__(self):
        # Create a simple header with dimensions (4 bytes each) and data
        # Real implementations would use proper image format like PNG
        header = self.width.to_bytes(4, byteorder='big') + self.height.to_bytes(4, byteorder='big')
        return header + bytes(self.data)
    
    @classmethod
    def from_bytes(cls, data):
        # Recreate object from bytes representation
        width = int.from_bytes(data[0:4], byteorder='big')
        height = int.from_bytes(data[4:8], byteorder='big')
        pixel_data = data[8:]
        return cls(width, height, pixel_data)


# ====================================================================================
# USAGE EXAMPLES
# ====================================================================================

# __new__ and __init__ example
instance = NewExample("test value")
# Output:
# 1. __new__ called with class: NewExample
# 2. __init__ called with value: test value
#    _created_at from __new__: pre-initialization

# __str__ and __repr__ example
p = Point(3, 4)
print(str(p))           # Uses __str__: "Point at (3, 4)"
print(repr(p))          # Falls back to __str__ since __repr__ not defined

r = Rectangle(5, 10)
print(str(r))           # Uses __str__: "Rectangle with width=5 and height=10"
print(repr(r))          # Uses __repr__: "Rectangle(5, 10)"

# __format__ example
m = Money(42.5)
print(f"{m}")            # Default: "USD 42.50"
print(f"{m:p}")          # Plain: "42.50"
print(f"{m:s}")          # Symbol: "$42.50"
print(f"{m:c}")          # Code: "USD 42.50"
print(f"{m:f:3}")        # Custom precision: "42.500"

# __bytes__ example
img = Image(100, 50)
serialized = bytes(img)
# First 8 bytes are width and height, rest is pixel data
print(f"Serialized image size: {len(serialized)} bytes")
# Recreate image from bytes
recreated = Image.from_bytes(serialized)
print(f"Recreated image: {recreated}")