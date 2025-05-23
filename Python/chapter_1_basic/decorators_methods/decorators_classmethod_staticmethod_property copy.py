"""
Python Special Methods and Decorators
====================================

This file demonstrates Python special methods (dunder methods) and decorators
including @classmethod, @staticmethod, @property and its modifiers.
"""


#############################
# Python Special Methods (Dunder Methods)
#############################
# Dunder (double underscore) methods are special methods in Python that enable
# operator overloading and implementing specific behaviors for custom objects.
# They're automatically called by Python in response to certain operations.

class Vector:
    def __init__(self, x, y):
        """
        Constructor method - called when creating a new instance
        Initializes the object's attributes.
        """
        self.x = x
        self.y = y
    
    def __str__(self):
        """
        String representation for human-readable output
        Called by str() and print()
        """
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        """
        Official string representation for debugging/development
        Called by repr() and in interactive sessions
        Should ideally be unambiguous and allow recreation of the object
        """
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        """
        Defines behavior for the + operator
        Called when + is used on this object
        """
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented  # Let Python try other methods
    
    def __eq__(self, other):
        """
        Defines behavior for the == operator
        Called when == is used to compare this object
        """
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y
    
    def __len__(self):
        """
        Defines behavior for len()
        Often returns size/dimensions/count of elements
        """
        # Return the Euclidean length (rounded to nearest int)
        return int((self.x**2 + self.y**2)**0.5)
    
    def __getitem__(self, index):
        """
        Defines behavior for accessing items with indexing syntax: obj[index]
        Makes the object behave like a sequence
        """
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector index out of range")
    
    def __bool__(self):
        """
        Defines behavior for bool() to determine truthiness
        Called for truth value testing (if statements, etc.)
        """
        return self.x != 0 or self.y != 0
    
    def __call__(self, scalar):
        """
        Makes the object callable like a function: object(arguments)
        """
        return Vector(self.x * scalar, self.y * scalar)


#############################
# Class Methods, Static Methods, and Properties
#############################

class Temperature:
    def __init__(self, celsius):
        """
        Initialize with temperature in Celsius
        """
        self._celsius = celsius  # Using leading underscore for private attribute

    #############################
    # @classmethod
    #############################
    # Class methods are bound to the class, not instances
    # First parameter is always the class (cls) itself, not an instance
    # Can access and modify class state but not instance state
    # Commonly used as factory methods or alternative constructors
    
    @classmethod
    def from_fahrenheit(cls, fahrenheit):
        """
        Alternative constructor that creates a Temperature instance
        from a value in Fahrenheit.
        
        - Takes the class as first parameter (cls)
        - Can create and return instances of the class
        - Can access/modify class variables but not instance variables
        """
        celsius = (fahrenheit - 32) * 5 / 9
        return cls(celsius)  # Create new instance using the class
    
    @classmethod
    def from_kelvin(cls, kelvin):
        """Another factory method creating instances from Kelvin"""
        celsius = kelvin - 273.15
        return cls(celsius)

    #############################
    # @staticmethod
    #############################
    # Static methods don't receive any special first parameter
    # They're regular functions contained in a class namespace
    # Cannot access or modify class or instance state
    # Used for utility functions related to the class but not dependent on its state
    
    @staticmethod
    def is_freezing(temp_celsius):
        """
        Utility method related to temperature but not dependent on instance data.
        
        - Takes no special first parameter (no self, no cls)
        - Cannot access instance or class variables directly
        - Logically related to the class but independent of any specific instance
        """
        return temp_celsius <= 0
    
    @staticmethod
    def celsius_to_fahrenheit(celsius):
        """Utility conversion method"""
        return celsius * 9/5 + 32

    #############################
    # @property
    #############################
    # Properties allow access to attributes through methods
    # Enables getter behavior - accessing attribute-like syntax calls a method
    # Supports encapsulation, validation, calculation of derived values
    
    @property
    def celsius(self):
        """
        Getter method for celsius temperature.
        
        - Called when accessing obj.celsius
        - Allows read access to private attribute
        - Can perform validation or computation
        - Preserves encapsulation
        """
        return self._celsius
    
    @property
    def fahrenheit(self):
        """Calculated property for Fahrenheit equivalent"""
        return self._celsius * 9/5 + 32
    
    @property
    def kelvin(self):
        """Calculated property for Kelvin equivalent"""
        return self._celsius + 273.15

    #############################
    # @property.setter
    #############################
    # Setter methods allow setting attributes through property methods
    # Enables validation, transformation, or other logic when setting values
    # Must have a corresponding @property method with the same name
    
    @celsius.setter
    def celsius(self, value):
        """
        Setter method for celsius temperature.
        
        - Called when assigning to obj.celsius
        - Allows validation before setting the attribute
        - Must have corresponding @property with same name
        - Maintains encapsulation
        """
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value

    #############################
    # @property.deleter
    #############################
    # Deleter methods define behavior when deleting an attribute
    # Called when using 'del obj.attribute'
    # Allows cleanup or other actions when attribute is deleted
    
    @celsius.deleter
    def celsius(self):
        """
        Deleter method for celsius temperature.
        
        - Called when executing 'del obj.celsius'
        - Defines behavior for attribute deletion
        - Must have corresponding @property with same name
        - Can be used for cleanup or resetting to default
        """
        print("Resetting temperature to absolute zero")
        self._celsius = -273.15


# Demo usage
if __name__ == "__main__":
    # Dunder methods demonstration
    v1 = Vector(3, 4)
    v2 = Vector(2, 3)
    
    print(f"v1: {v1}")  # Uses __str__
    print(f"v1 + v2: {v1 + v2}")  # Uses __add__
    print(f"v1 == v2: {v1 == v2}")  # Uses __eq__
    print(f"Length of v1: {len(v1)}")  # Uses __len__
    print(f"v1[0]: {v1[0]}, v1[1]: {v1[1]}")  # Uses __getitem__
    print(f"bool(v1): {bool(v1)}")  # Uses __bool__
    print(f"v1(2): {v1(2)}")  # Uses __call__
    
    # Class/static methods and properties demonstration
    t1 = Temperature(25)  # Regular initialization
    t2 = Temperature.from_fahrenheit(77)  # Using class method
    t3 = Temperature.from_kelvin(300)  # Using another class method
    
    print(f"\nRegular temp: {t1.celsius}°C")
    print(f"From Fahrenheit: {t2.celsius}°C")
    print(f"From Kelvin: {t3.celsius}°C")
    
    # Using properties
    print(f"\nt1 in Celsius: {t1.celsius}°C")
    print(f"t1 in Fahrenheit: {t1.fahrenheit}°F")
    print(f"t1 in Kelvin: {t1.kelvin}K")
    
    # Using static methods
    print(f"\nIs 0°C freezing? {Temperature.is_freezing(0)}")
    print(f"Is 5°C freezing? {Temperature.is_freezing(5)}")
    
    # Using setter
    try:
        t1.celsius = -300  # Will raise ValueError
    except ValueError as e:
        print(f"\nError when setting invalid temperature: {e}")
    
    t1.celsius = 100
    print(f"New temperature: {t1.celsius}°C")
    
    # Using deleter
    del t1.celsius
    print(f"After deletion: {t1.celsius}°C")