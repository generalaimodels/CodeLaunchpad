# ============================================================================
# PYTHON SPECIAL METHODS (DUNDER METHODS) - REFLECTED OPERATIONS
# ============================================================================

"""
This file demonstrates and explains Python's reflected operator methods - these
are special methods that are called when the left operand of an operation
doesn't support the operation with the right operand.

Reflected operations are essential for handling mixed-type operations in a
symmetrical way, enabling proper numeric type conversions and interoperability.
"""


class Number:
    """
    A class demonstrating all reflected numeric operations.
    
    When operations are performed with different types, Python first tries the 
    direct operation on the left operand. If that returns NotImplemented,
    it then tries the reflected operation on the right operand.
    
    Example flow for x + y:
    1. Try x.__add__(y)
    2. If that returns NotImplemented, try y.__radd__(x)
    3. If both fail, raise TypeError
    """
    
    def __init__(self, value):
        self.value = value
    
    # ========================================================================
    # Regular operations (for comparison with reflected operations)
    # ========================================================================
    
    def __add__(self, other):
        """Regular addition: self + other"""
        if isinstance(other, Number):
            return Number(self.value + other.value)
        elif isinstance(other, (int, float)):
            return Number(self.value + other)
        return NotImplemented  # This triggers the reflected operation
    
    def __sub__(self, other):
        """Regular subtraction: self - other"""
        if isinstance(other, Number):
            return Number(self.value - other.value)
        elif isinstance(other, (int, float)):
            return Number(self.value - other)
        return NotImplemented
    
    def __mul__(self, other):
        """Regular multiplication: self * other"""
        if isinstance(other, Number):
            return Number(self.value * other.value)
        elif isinstance(other, (int, float)):
            return Number(self.value * other)
        return NotImplemented
    
    def __matmul__(self, other):
        """Regular matrix multiplication: self @ other"""
        if isinstance(other, Number):
            # Simplified implementation for demonstration
            return Number(self.value * other.value)
        return NotImplemented
    
    def __truediv__(self, other):
        """Regular true division: self / other"""
        if isinstance(other, Number):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            return Number(self.value / other.value)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return Number(self.value / other)
        return NotImplemented
    
    def __floordiv__(self, other):
        """Regular floor division: self // other"""
        if isinstance(other, Number):
            if other.value == 0:
                raise ZeroDivisionError("Division by zero")
            return Number(self.value // other.value)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return Number(self.value // other)
        return NotImplemented
    
    def __mod__(self, other):
        """Regular modulo: self % other"""
        if isinstance(other, Number):
            if other.value == 0:
                raise ZeroDivisionError("Modulo by zero")
            return Number(self.value % other.value)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Modulo by zero")
            return Number(self.value % other)
        return NotImplemented
    
    def __divmod__(self, other):
        """Regular divmod: divmod(self, other)"""
        if isinstance(other, Number):
            if other.value == 0:
                raise ZeroDivisionError("divmod with zero divisor")
            q = self.value // other.value
            r = self.value % other.value
            return (Number(q), Number(r))
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("divmod with zero divisor")
            q = self.value // other
            r = self.value % other
            return (Number(q), Number(r))
        return NotImplemented
    
    def __pow__(self, other, modulo=None):
        """Regular power: self ** other or pow(self, other[, modulo])"""
        if isinstance(other, Number):
            power = other.value
        elif isinstance(other, (int, float)):
            power = other
        else:
            return NotImplemented
            
        if modulo is None:
            return Number(self.value ** power)
        else:
            mod_value = modulo.value if isinstance(modulo, Number) else modulo
            return Number(pow(self.value, power, mod_value))
    
    # ========================================================================
    # Reflected operations
    # ========================================================================
    
    def __radd__(self, other):
        """
        Reflected addition: other + self
        
        Called when 'other' does not support the addition operation with 'self'.
        For example, when doing 5 + num where num is a Number instance.
        
        Args:
            other: The left operand of the + operator
            
        Returns:
            Number: A new Number instance with the result
            
        Note:
            Reflected operations typically convert the other operand to the
            current type and then perform the operation.
        """
        if isinstance(other, (int, float)):
            return Number(other + self.value)
        return NotImplemented
    
    def __rsub__(self, other):
        """
        Reflected subtraction: other - self
        
        Called when 'other' does not support the subtraction operation with 'self'.
        For example, when doing 5 - num where num is a Number instance.
        
        Args:
            other: The left operand of the - operator
            
        Returns:
            Number: A new Number instance with the result
            
        Note:
            Be careful with the order of operands in reflected operations!
            For subtraction, it's other - self, not self - other.
        """
        if isinstance(other, (int, float)):
            return Number(other - self.value)
        return NotImplemented
    
    def __rmul__(self, other):
        """
        Reflected multiplication: other * self
        
        Called when 'other' does not support the multiplication operation with 'self'.
        For example, when doing 5 * num where num is a Number instance.
        
        Args:
            other: The left operand of the * operator
            
        Returns:
            Number: A new Number instance with the result
        """
        if isinstance(other, (int, float)):
            return Number(other * self.value)
        return NotImplemented
    
    def __rmatmul__(self, other):
        """
        Reflected matrix multiplication: other @ self
        
        Called when 'other' does not support the matrix multiplication with 'self'.
        For example, when using @ operator where self is on the right.
        
        Args:
            other: The left operand of the @ operator
            
        Returns:
            Number: A new Number instance with the result
            
        Note:
            Matrix multiplication is typically used with arrays/matrices.
            This is a simplified implementation for demonstration.
        """
        if isinstance(other, (int, float)):
            # Simplified implementation for demonstration
            return Number(other * self.value)
        return NotImplemented
    
    def __rtruediv__(self, other):
        """
        Reflected true division: other / self
        
        Called when 'other' does not support the division operation with 'self'.
        For example, when doing 10 / num where num is a Number instance.
        
        Args:
            other: The left operand of the / operator
            
        Returns:
            Number: A new Number instance with the result
            
        Note:
            Remember to handle division by zero cases.
        """
        if isinstance(other, (int, float)):
            if self.value == 0:
                raise ZeroDivisionError("Division by zero")
            return Number(other / self.value)
        return NotImplemented
    
    def __rfloordiv__(self, other):
        """
        Reflected floor division: other // self
        
        Called when 'other' does not support the floor division operation with 'self'.
        For example, when doing 10 // num where num is a Number instance.
        
        Args:
            other: The left operand of the // operator
            
        Returns:
            Number: A new Number instance with the result
        """
        if isinstance(other, (int, float)):
            if self.value == 0:
                raise ZeroDivisionError("Division by zero")
            return Number(other // self.value)
        return NotImplemented
    
    def __rmod__(self, other):
        """
        Reflected modulo: other % self
        
        Called when 'other' does not support the modulo operation with 'self'.
        For example, when doing 10 % num where num is a Number instance.
        
        Args:
            other: The left operand of the % operator
            
        Returns:
            Number: A new Number instance with the result
        """
        if isinstance(other, (int, float)):
            if self.value == 0:
                raise ZeroDivisionError("Modulo by zero")
            return Number(other % self.value)
        return NotImplemented
    
    def __rdivmod__(self, other):
        """
        Reflected divmod: divmod(other, self)
        
        Called when 'other' does not support the divmod operation with 'self'.
        For example, when doing divmod(10, num) where num is a Number instance.
        
        Args:
            other: The first argument to divmod
            
        Returns:
            tuple: A tuple of (quotient, remainder)
        """
        if isinstance(other, (int, float)):
            if self.value == 0:
                raise ZeroDivisionError("divmod with zero divisor")
            q = other // self.value
            r = other % self.value
            return (Number(q), Number(r))
        return NotImplemented
    
    def __rpow__(self, other):
        """
        Reflected power: other ** self
        
        Called when 'other' does not support the power operation with 'self'.
        For example, when doing 2 ** num where num is a Number instance.
        
        Args:
            other: The left operand of the ** operator
            
        Returns:
            Number: A new Number instance with the result
        """
        if isinstance(other, (int, float)):
            return Number(other ** self.value)
        return NotImplemented
    
    def __str__(self):
        """String representation for better readability in examples."""
        return f"Number({self.value})"


# Define a class that only works with reflected operations
class OtherType:
    """
    A class that does not define regular operator methods,
    only meant to demonstrate when reflected operations are called.
    """
    
    def __init__(self, value):
        self.value = value
    
    # This class intentionally doesn't implement __add__, __sub__, etc.,
    # so when an operation is performed with a Number, the Number's 
    # corresponding method will be called.
    
    def __str__(self):
        return f"OtherType({self.value})"


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demonstrate_normal_vs_reflected():
    """
    Demonstrates the difference between normal and reflected operations.
    Shows when each type of method gets called.
    """
    print("\n# Normal vs. Reflected Operations")
    
    num = Number(10)
    print(f"num = {num}")
    
    # Regular operations where Number is on the left
    print("\n## Regular operations (Number on left):")
    print(f"num + 5 = {num + 5}")         # Uses __add__
    print(f"num - 3 = {num - 3}")         # Uses __sub__
    print(f"num * 2 = {num * 2}")         # Uses __mul__
    print(f"num / 2 = {num / 2}")         # Uses __truediv__
    
    # Reflected operations where Number is on the right
    print("\n## Reflected operations (Number on right):")
    print(f"5 + num = {5 + num}")         # Uses __radd__
    print(f"20 - num = {20 - num}")       # Uses __rsub__
    print(f"3 * num = {3 * num}")         # Uses __rmul__
    print(f"100 / num = {100 / num}")     # Uses __rtruediv__


def demonstrate_method_resolution_order():
    """
    Demonstrates the method resolution order for operations.
    Shows how Python decides which method to call.
    """
    print("\n# Method Resolution Order")
    
    num1 = Number(10)
    num2 = Number(5)
    
    # 1. When both operands are the same type, the direct method is used
    print("\n## Both operands same type:")
    print(f"num1 + num2 = {num1 + num2}")  # Uses Number.__add__
    
    # 2. When operands are different types:
    #    a. Try left.__op__(right)
    #    b. If that returns NotImplemented, try right.__rop__(left)
    #    c. If both fail, raise TypeError
    
    # Create an object of a different type that will trigger reflected operations
    other = OtherType(20)
    print(f"\n## Different types (Number + OtherType):")
    
    try:
        # This will eventually fail since OtherType doesn't implement __radd__
        # but it demonstrates the attempt sequence
        result = num1 + other
        print(f"num1 + other = {result}")
    except TypeError as e:
        print(f"num1 + other -> TypeError: {e}")
        print("  1. Tried Number.__add__(OtherType) -> returned NotImplemented")
        print("  2. Tried OtherType.__radd__(Number) -> not implemented")
        print("  3. Result: TypeError")


def demonstrate_mixed_types():
    """
    Demonstrates operations with mixed types,
    showing which method gets called in each case.
    """
    print("\n# Operations with Mixed Types")
    
    num = Number(10)
    
    # Built-in types (int, float) with Number
    print("\n## Built-in types with Number:")
    
    # When Number is on the left:
    print("# Number on left:")
    print(f"num + 5 = {num + 5}")        # Uses __add__
    print(f"num - 3.5 = {num - 3.5}")    # Uses __sub__
    
    # When Number is on the right:
    print("\n# Number on right:")
    print(f"5 + num = {5 + num}")        # Uses __radd__
    print(f"15.5 - num = {15.5 - num}")  # Uses __rsub__


def demonstrate_all_reflected_operations():
    """
    Demonstrates all the reflected operations,
    showing how each one works with built-in types.
    """
    print("\n# All Reflected Operations")
    
    num = Number(5)
    
    print(f"\n## Arithmetic operations with num = {num}:")
    print(f"10 + num = {10 + num}")         # __radd__
    print(f"10 - num = {10 - num}")         # __rsub__
    print(f"10 * num = {10 * num}")         # __rmul__
    print(f"10 @ num = {10 @ num}")         # __rmatmul__
    print(f"10 / num = {10 / num}")         # __rtruediv__
    print(f"10 // num = {10 // num}")       # __rfloordiv__
    print(f"10 % num = {10 % num}")         # __rmod__
    
    print("\n## Advanced operations:")
    q, r = divmod(100, num)
    print(f"divmod(100, num) = ({q}, {r})")  # __rdivmod__
    print(f"2 ** num = {2 ** num}")          # __rpow__


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_normal_vs_reflected()
    demonstrate_method_resolution_order()
    demonstrate_mixed_types()
    demonstrate_all_reflected_operations()