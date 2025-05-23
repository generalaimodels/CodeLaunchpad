# ============================================================================
# PYTHON SPECIAL METHODS (DUNDER METHODS) - NUMERIC OPERATIONS
# ============================================================================

"""
This file demonstrates and explains Python special methods (dunder methods)
for numeric operations, including arithmetic, matrix operations, and more.
"""


class Number:
    """
    A class demonstrating all numeric special methods (dunder methods).
    This class simulates a custom numeric type to show how Python's
    arithmetic operators are implemented behind the scenes.
    """
    
    def __init__(self, value):
        """Initialize with a numeric value."""
        self.value = value
    
    def __add__(self, other):
        """
        Implements the addition operation (self + other).
        Called when the + operator is used.
        
        Args:
            other: The right operand of the + operator
            
        Returns:
            Number: A new Number instance with the result
            
        Examples:
            num1 + num2
            num + 5
        """
        if isinstance(other, Number):
            return Number(self.value + other.value)
        elif isinstance(other, (int, float)):
            return Number(self.value + other)
        # Return NotImplemented if we don't know how to handle the operation
        # This allows Python to try other.radd(self) if available
        return NotImplemented
    
    def __sub__(self, other):
        """
        Implements the subtraction operation (self - other).
        Called when the - operator is used.
        
        Args:
            other: The right operand of the - operator
            
        Returns:
            Number: A new Number instance with the result
            
        Examples:
            num1 - num2
            num - 10
        """
        if isinstance(other, Number):
            return Number(self.value - other.value)
        elif isinstance(other, (int, float)):
            return Number(self.value - other)
        return NotImplemented
    
    def __mul__(self, other):
        """
        Implements the multiplication operation (self * other).
        Called when the * operator is used.
        
        Args:
            other: The right operand of the * operator
            
        Returns:
            Number: A new Number instance with the result
            
        Examples:
            num1 * num2
            num * 3
            
        Note:
            Can also be used for scaling sequences (like lists) if
            implemented on a sequence-like object.
        """
        if isinstance(other, Number):
            return Number(self.value * other.value)
        elif isinstance(other, (int, float)):
            return Number(self.value * other)
        return NotImplemented
    
    def __matmul__(self, other):
        """
        Implements the matrix multiplication operation (self @ other).
        Called when the @ operator is used (Python 3.5+).
        
        Args:
            other: The right operand of the @ operator
            
        Returns:
            Number: A new Number instance with the result
            
        Examples:
            matrix1 @ matrix2
            
        Note:
            This is typically used for matrix multiplication in libraries
            like NumPy. For our simple Number class, we'll use a placeholder
            implementation to demonstrate the concept.
        """
        if isinstance(other, Number):
            # Simplified example: in real matrices, this would involve
            # complex multiplication and summation operations
            return Number(self.value * other.value)  # Not real matrix multiplication
        return NotImplemented
    
    def __truediv__(self, other):
        """
        Implements true division (self / other).
        Called when the / operator is used.
        
        Args:
            other: The right operand of the / operator
            
        Returns:
            Number: A new Number instance with the result
            
        Examples:
            num1 / num2
            num / 2
            
        Note:
            In Python 3, / always performs true division (returns float).
            This is different from Python 2 where / performed floor division
            for integers.
        """
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
        """
        Implements floor division (self // other).
        Called when the // operator is used.
        
        Args:
            other: The right operand of the // operator
            
        Returns:
            Number: A new Number instance with the result
            
        Examples:
            num1 // num2
            num // 2
            
        Note:
            Floor division returns the largest integer less than or equal
            to the true division result.
        """
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
        """
        Implements the modulo operation (self % other).
        Called when the % operator is used.
        
        Args:
            other: The right operand of the % operator
            
        Returns:
            Number: A new Number instance with the remainder
            
        Examples:
            num1 % num2
            num % 5
            
        Note:
            Returns the remainder when dividing self by other.
        """
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
        """
        Implements the divmod operation (divmod(self, other)).
        Returns a tuple containing (self // other, self % other).
        
        Args:
            other: The right operand of the divmod operation
            
        Returns:
            tuple: A tuple of (quotient, remainder)
            
        Examples:
            divmod(num1, num2)
            divmod(num, 3)
            
        Note:
            This operation is useful when you need both the quotient and
            remainder of a division operation.
        """
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
        """
        Implements the power operation (self ** other).
        Called when the ** operator is used or pow(self, other[, modulo]) is called.
        
        Args:
            other: The exponent
            modulo: Optional modulus for the power operation
            
        Returns:
            Number: A new Number instance with the result
            
        Examples:
            num1 ** num2
            num ** 2
            pow(num, 2, 5)  # Computes (num ** 2) % 5
            
        Note:
            The modulo parameter is used when the 3-argument form of pow()
            is called, which is more efficient than (x ** y) % z.
        """
        if isinstance(other, Number):
            if modulo is None:
                return Number(self.value ** other.value)
            else:
                if isinstance(modulo, Number):
                    return Number(pow(self.value, other.value, modulo.value))
                return Number(pow(self.value, other.value, modulo))
        elif isinstance(other, (int, float)):
            if modulo is None:
                return Number(self.value ** other)
            else:
                if isinstance(modulo, Number):
                    return Number(pow(self.value, other, modulo.value))
                return Number(pow(self.value, other, modulo))
        return NotImplemented
    
    def __round__(self, ndigits=None):
        """
        Implements the round operation (round(self[, ndigits])).
        
        Args:
            ndigits: Number of decimal places to round to
            
        Returns:
            Number: A new Number instance with the rounded value
            
        Examples:
            round(num)      # Round to nearest integer
            round(num, 2)   # Round to 2 decimal places
            
        Note:
            If ndigits is omitted or None, returns the nearest integer.
            Otherwise rounds to the specified number of decimal places.
        """
        if ndigits is None:
            return Number(round(self.value))
        return Number(round(self.value, ndigits))
    
    def __abs__(self):
        """
        Implements the absolute value operation (abs(self)).
        
        Returns:
            Number: A new Number instance with the absolute value
            
        Examples:
            abs(num)
            
        Note:
            Returns a non-negative version of the value.
        """
        return Number(abs(self.value))
    
    def __neg__(self):
        """
        Implements the negation operation (-self).
        
        Returns:
            Number: A new Number instance with the negated value
            
        Examples:
            -num
            
        Note:
            Negation flips the sign of the value.
        """
        return Number(-self.value)
    
    def __pos__(self):
        """
        Implements the unary plus operation (+self).
        
        Returns:
            Number: A new Number instance with the same value
            
        Examples:
            +num
            
        Note:
            In most cases, this just returns the object unchanged,
            but it can be overridden for type conversion.
        """
        return Number(+self.value)
    
    def __invert__(self):
        """
        Implements the bitwise inversion operation (~self).
        
        Returns:
            Number: A new Number instance with the inverted value
            
        Examples:
            ~num
            
        Note:
            For integers, this computes the bitwise NOT (one's complement).
            For our Number class, we'll implement it for integer values.
        """
        if isinstance(self.value, int):
            return Number(~self.value)
        else:
            # For non-integer types, bitwise inversion doesn't make sense
            # but we'll convert to int and invert for demonstration
            return Number(~int(self.value))
    
    def __str__(self):
        """String representation for better readability in examples."""
        return f"Number({self.value})"


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_basic_arithmetic():
    """Demonstrates basic arithmetic operations."""
    a = Number(10)
    b = Number(3)
    
    print("\n# Basic Arithmetic Operations")
    print(f"a = {a}, b = {b}")
    print(f"a + b = {a + b}")         # __add__
    print(f"a - b = {a - b}")         # __sub__
    print(f"a * b = {a * b}")         # __mul__
    print(f"a / b = {a / b}")         # __truediv__
    print(f"a // b = {a // b}")       # __floordiv__
    print(f"a % b = {a % b}")         # __mod__
    print(f"a ** b = {a ** b}")       # __pow__


def demo_matrix_and_advanced_operations():
    """Demonstrates matrix and advanced operations."""
    a = Number(10)
    b = Number(3)
    
    print("\n# Matrix and Advanced Operations")
    print(f"a @ b = {a @ b}")         # __matmul__ (simplified)
    
    # divmod demonstration
    quot, rem = divmod(a, b)
    print(f"divmod(a, b) = ({quot}, {rem})")  # __divmod__
    
    # pow with modulo
    print(f"pow(a, b, 7) = {pow(a, b, 7)}")   # __pow__ with modulo


def demo_unary_operations():
    """Demonstrates unary operations."""
    a = Number(10)
    b = Number(-5)
    
    print("\n# Unary Operations")
    print(f"a = {a}, b = {b}")
    print(f"round(a) = {round(a)}")           # __round__
    print(f"round(a/b, 2) = {round(a/b, 2)}") # __round__ with ndigits
    print(f"abs(b) = {abs(b)}")               # __abs__
    print(f"-a = {-a}")                       # __neg__
    print(f"+b = {+b}")                       # __pos__
    print(f"~a = {~a}")                       # __invert__


def demo_mixed_operands():
    """Demonstrates operations with mixed operand types."""
    a = Number(10)
    
    print("\n# Operations with Mixed Types")
    print(f"a = {a}")
    print(f"a + 5 = {a + 5}")                 # Number + int
    print(f"a * 2.5 = {a * 2.5}")             # Number * float
    print(f"a / 2 = {a / 2}")                 # Number / int
    print(f"a ** 0.5 = {a ** 0.5}")           # Number ** float


if __name__ == "__main__":
    # Run all demonstrations
    demo_basic_arithmetic()
    demo_matrix_and_advanced_operations()
    demo_unary_operations()
    demo_mixed_operands()