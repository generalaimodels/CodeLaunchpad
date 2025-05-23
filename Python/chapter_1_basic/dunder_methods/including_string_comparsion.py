# ============================================================================
# PYTHON SPECIAL METHODS (DUNDER METHODS) COMPREHENSIVE GUIDE
# ============================================================================

"""
This file demonstrates and explains Python special methods (dunder methods)
for string representation and comparison operations.
"""


# ============================================================================
# STRING REPRESENTATION METHODS
# ============================================================================

class Product:
    """Example class demonstrating string representation methods."""
    
    def __init__(self, name, price, category):
        self.name = name
        self.price = price
        self.category = category
    
    def __str__(self):
        """
        Defines informal string representation, called by str() and print().
        Should be readable by end users.
        
        Returns:
            str: User-friendly string representation
        """
        return f"{self.name} (${self.price:.2f})"
    
    def __repr__(self):
        """
        Defines official string representation, called by repr().
        Should ideally be an expression that recreates the object.
        
        Returns:
            str: String that could be used to recreate the object
        """
        return f"Product(name='{self.name}', price={self.price}, category='{self.category}')"
    
    def __format__(self, format_spec):
        """
        Called by format() function and string formatting operations.
        The format_spec parameter comes from the format specifier.
        
        Args:
            format_spec (str): Format specification
            
        Returns:
            str: Formatted string representation
            
        Format options implemented:
        - 'short': Just the name
        - 'full': Name, price, and category
        - 'price': Just the price with $ symbol
        - Default (empty string): Same as __str__
        """
        if format_spec == "short":
            return self.name
        elif format_spec == "full":
            return f"{self.name} (${self.price:.2f}) - {self.category}"
        elif format_spec == "price":
            return f"${self.price:.2f}"
        else:  # Default format
            return str(self)
    
    def __bytes__(self):
        """
        Called by bytes() function to create a bytes representation.
        Useful for serialization or binary protocols.
        
        Returns:
            bytes: Binary representation of the object
        """
        # Creating a simple byte representation by encoding a string
        data_string = f"{self.name}|{self.price}|{self.category}"
        return data_string.encode('utf-8')


# String Representation Demo
def demo_string_representation():
    """Demonstrates string representation methods."""
    laptop = Product("MacBook Pro", 1299.99, "Electronics")
    
    # __str__ demo
    print("\n# __str__ demo:")
    print(f"str(laptop): {str(laptop)}")
    print(f"print(laptop): {laptop}")  # print() implicitly calls __str__
    
    # __repr__ demo
    print("\n# __repr__ demo:")
    print(f"repr(laptop): {repr(laptop)}")
    print(f"In interactive mode: {laptop!r}")  # !r in f-string calls __repr__
    
    # __format__ demo
    print("\n# __format__ demo:")
    print(f"format(laptop): {format(laptop)}")
    print(f"format(laptop, 'short'): {format(laptop, 'short')}")
    print(f"format(laptop, 'full'): {format(laptop, 'full')}")
    print(f"format(laptop, 'price'): {format(laptop, 'price')}")
    print(f"In f-string: {laptop:short}")  # :spec in f-string calls __format__
    
    # __bytes__ demo
    print("\n# __bytes__ demo:")
    print(f"bytes(laptop): {bytes(laptop)}")
    print(f"Decoded: {bytes(laptop).decode('utf-8')}")


# ============================================================================
# COMPARISON METHODS
# ============================================================================

class Temperature:
    """Example class demonstrating comparison methods."""
    
    def __init__(self, celsius):
        self.celsius = celsius
    
    def __eq__(self, other):
        """
        Equality comparison (==).
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if objects are equal, False otherwise
        """
        if not isinstance(other, Temperature):
            # Handle comparison with other types
            if isinstance(other, (int, float)):
                return self.celsius == other
            return NotImplemented
        return self.celsius == other.celsius
    
    def __ne__(self, other):
        """
        Not equal comparison (!=).
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if objects are not equal, False otherwise
            
        Note: Modern Python can use __eq__ and negate the result,
        but implementing __ne__ explicitly gives more control.
        """
        if not isinstance(other, Temperature):
            if isinstance(other, (int, float)):
                return self.celsius != other
            return NotImplemented
        return self.celsius != other.celsius
    
    def __lt__(self, other):
        """
        Less than comparison (<).
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if self < other, False otherwise
        """
        if not isinstance(other, Temperature):
            if isinstance(other, (int, float)):
                return self.celsius < other
            return NotImplemented
        return self.celsius < other.celsius
    
    def __le__(self, other):
        """
        Less than or equal comparison (<=).
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if self <= other, False otherwise
        """
        if not isinstance(other, Temperature):
            if isinstance(other, (int, float)):
                return self.celsius <= other
            return NotImplemented
        return self.celsius <= other.celsius
    
    def __gt__(self, other):
        """
        Greater than comparison (>).
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if self > other, False otherwise
        """
        if not isinstance(other, Temperature):
            if isinstance(other, (int, float)):
                return self.celsius > other
            return NotImplemented
        return self.celsius > other.celsius
    
    def __ge__(self, other):
        """
        Greater than or equal comparison (>=).
        
        Args:
            other: Object to compare with
            
        Returns:
            bool: True if self >= other, False otherwise
        """
        if not isinstance(other, Temperature):
            if isinstance(other, (int, float)):
                return self.celsius >= other
            return NotImplemented
        return self.celsius >= other.celsius
    
    def __hash__(self):
        """
        Generates a hash value for the object.
        
        Returns:
            int: Hash value
            
        Note: Only immutable objects should be hashable. If your class
        has mutable attributes, you should either not implement __hash__
        or make the object effectively immutable.
        
        Objects that compare equal should have the same hash value.
        """
        return hash(self.celsius)
    
    def __bool__(self):
        """
        Defines truth value testing and used by bool() function.
        
        Returns:
            bool: True if the object is considered True, False otherwise
            
        Note: If __bool__ is not defined, Python falls back to __len__,
        and if neither is defined, all objects are considered True.
        """
        # In this example, we'll consider temperatures at or below absolute zero (-273.15°C) as False
        return self.celsius > -273.15
    
    def __str__(self):
        """For demonstration purposes."""
        return f"{self.celsius}°C"


# Comparison Methods Demo
def demo_comparison_methods():
    """Demonstrates comparison methods."""
    temp1 = Temperature(25)  # 25°C
    temp2 = Temperature(30)  # 30°C
    temp3 = Temperature(25)  # 25°C (same as temp1)
    absolute_zero = Temperature(-273.15)  # Absolute zero
    
    # __eq__ and __ne__ demo
    print("\n# __eq__ and __ne__ demo:")
    print(f"temp1 == temp2: {temp1 == temp2}")  # Calls __eq__
    print(f"temp1 == temp3: {temp1 == temp3}")  # Calls __eq__
    print(f"temp1 == 25: {temp1 == 25}")  # Calls __eq__ with int
    print(f"temp1 != temp2: {temp1 != temp2}")  # Calls __ne__
    
    # __lt__, __le__, __gt__, __ge__ demo
    print("\n# Comparison operators demo:")
    print(f"temp1 < temp2: {temp1 < temp2}")   # Calls __lt__
    print(f"temp1 <= temp3: {temp1 <= temp3}") # Calls __le__
    print(f"temp1 > temp2: {temp1 > temp2}")   # Calls __gt__
    print(f"temp1 >= temp3: {temp1 >= temp3}") # Calls __ge__
    
    # __hash__ demo
    print("\n# __hash__ demo:")
    temp_dict = {temp1: "Room temperature", temp2: "Hot day"}
    print(f"hash(temp1): {hash(temp1)}")
    print(f"hash(temp3): {hash(temp3)}") # Same as temp1 because they're equal
    print(f"Using Temperature objects as dictionary keys: {temp_dict}")
    
    # __bool__ demo
    print("\n# __bool__ demo:")
    print(f"bool(temp1): {bool(temp1)}")  # Normal temperature
    print(f"bool(absolute_zero): {bool(absolute_zero)}")  # Absolute zero
    print(f"In if statement: {'Valid' if temp1 else 'Invalid'}")  # Using in control flow


# ============================================================================
# RUN DEMOS
# ============================================================================

if __name__ == "__main__":
    demo_string_representation()
    demo_comparison_methods()