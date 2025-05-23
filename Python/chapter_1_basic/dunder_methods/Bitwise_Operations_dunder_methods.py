# Python Special Methods (Dunder Methods): Bitwise Operations

class BitwiseExample:
    """
    This class demonstrates all bitwise dunder methods in Python.
    
    Bitwise operations in Python work on integers at the binary level.
    These special methods allow custom classes to respond to bitwise operators.
    """
    
    def __init__(self, value):
        self.value = value
    
    # Left shift operations (<<)
    def __lshift__(self, other):
        """
        Left shift operation: self << other
        Shifts bits to the left by 'other' positions.
        
        Example: 5 (101) << 1 = 10 (1010)
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        return BitwiseExample(self.value << other)
    
    def __rlshift__(self, other):
        """
        Reflected left shift: other << self
        Called when the left operand doesn't support the operation.
        
        Example: 5 << BitwiseExample(2) calls BitwiseExample.__rlshift__(5)
        """
        return BitwiseExample(other << self.value)
    
    def __ilshift__(self, other):
        """
        In-place left shift: self <<= other
        Modifies self directly rather than creating a new instance.
        
        Example: a = BitwiseExample(5); a <<= 1
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        self.value <<= other
        return self
    
    # Right shift operations (>>)
    def __rshift__(self, other):
        """
        Right shift operation: self >> other
        Shifts bits to the right by 'other' positions.
        
        Example: 10 (1010) >> 1 = 5 (101)
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        return BitwiseExample(self.value >> other)
    
    def __rrshift__(self, other):
        """
        Reflected right shift: other >> self
        Called when the left operand doesn't support the operation.
        
        Example: 10 >> BitwiseExample(1) calls BitwiseExample.__rrshift__(10)
        """
        return BitwiseExample(other >> self.value)
    
    def __irshift__(self, other):
        """
        In-place right shift: self >>= other
        Modifies self directly rather than creating a new instance.
        
        Example: a = BitwiseExample(10); a >>= 1
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        self.value >>= other
        return self
    
    # Bitwise AND operations (&)
    def __and__(self, other):
        """
        Bitwise AND: self & other
        Applies bitwise AND to each pair of corresponding bits.
        
        Example: 5 (101) & 3 (011) = 1 (001)
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        return BitwiseExample(self.value & other)
    
    def __rand__(self, other):
        """
        Reflected bitwise AND: other & self
        Called when the left operand doesn't support the operation.
        
        Example: 5 & BitwiseExample(3) calls BitwiseExample.__rand__(5)
        """
        return BitwiseExample(other & self.value)
    
    def __iand__(self, other):
        """
        In-place bitwise AND: self &= other
        Modifies self directly rather than creating a new instance.
        
        Example: a = BitwiseExample(5); a &= 3
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        self.value &= other
        return self
    
    # Bitwise OR operations (|)
    def __or__(self, other):
        """
        Bitwise OR: self | other
        Applies bitwise OR to each pair of corresponding bits.
        
        Example: 5 (101) | 3 (011) = 7 (111)
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        return BitwiseExample(self.value | other)
    
    def __ror__(self, other):
        """
        Reflected bitwise OR: other | self
        Called when the left operand doesn't support the operation.
        
        Example: 5 | BitwiseExample(3) calls BitwiseExample.__ror__(5)
        """
        return BitwiseExample(other | self.value)
    
    def __ior__(self, other):
        """
        In-place bitwise OR: self |= other
        Modifies self directly rather than creating a new instance.
        
        Example: a = BitwiseExample(5); a |= 3
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        self.value |= other
        return self
    
    # Bitwise XOR operations (^)
    def __xor__(self, other):
        """
        Bitwise XOR: self ^ other
        Applies bitwise XOR to each pair of corresponding bits.
        
        Example: 5 (101) ^ 3 (011) = 6 (110)
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        return BitwiseExample(self.value ^ other)
    
    def __rxor__(self, other):
        """
        Reflected bitwise XOR: other ^ self
        Called when the left operand doesn't support the operation.
        
        Example: 5 ^ BitwiseExample(3) calls BitwiseExample.__rxor__(5)
        """
        return BitwiseExample(other ^ self.value)
    
    def __ixor__(self, other):
        """
        In-place bitwise XOR: self ^= other
        Modifies self directly rather than creating a new instance.
        
        Example: a = BitwiseExample(5); a ^= 3
        """
        if isinstance(other, BitwiseExample):
            other = other.value
        self.value ^= other
        return self
    
    def __str__(self):
        """String representation showing decimal and binary formats."""
        return f"Value: {self.value} (bin: {bin(self.value)})"


# Demonstration of BitwiseExample class
def demonstrate_bitwise_operations():
    # Create instances
    a = BitwiseExample(5)     # 101 in binary
    b = BitwiseExample(3)     # 011 in binary
    
    # Left shift operations
    left_shift = a << 1       # 5 << 1 = 10 (1010)
    rleft_shift = 4 << a      # 4 << 5 = 128 (10000000)
    a_copy = BitwiseExample(5)
    a_copy <<= 2              # 5 <<= 2 = 20 (10100)
    
    # Right shift operations
    right_shift = a >> 1      # 5 >> 1 = 2 (10)
    rright_shift = 16 >> a    # 16 >> 5 = 0 (0)
    a_copy = BitwiseExample(5)
    a_copy >>= 1              # 5 >>= 1 = 2 (10)
    
    # Bitwise AND operations
    and_result = a & b        # 5 & 3 = 1 (001)
    rand_result = 7 & a       # 7 & 5 = 5 (101)
    a_copy = BitwiseExample(5)
    a_copy &= 3               # 5 &= 3 = 1 (001)
    
    # Bitwise OR operations
    or_result = a | b         # 5 | 3 = 7 (111)
    ror_result = 8 | a        # 8 | 5 = 13 (1101)
    a_copy = BitwiseExample(5)
    a_copy |= 3               # 5 |= 3 = 7 (111)
    
    # Bitwise XOR operations
    xor_result = a ^ b        # 5 ^ 3 = 6 (110)
    rxor_result = 9 ^ a       # 9 ^ 5 = 12 (1100)
    a_copy = BitwiseExample(5)
    a_copy ^= 3               # 5 ^= 3 = 6 (110)
    
    # Results will be printed when this function is executed
    print(f"Original a: {a}")
    print(f"Original b: {b}")
    print(f"a << 1: {left_shift}")
    print(f"4 << a: {rleft_shift}")
    print(f"a >> 1: {right_shift}")
    print(f"16 >> a: {rright_shift}")
    print(f"a & b: {and_result}")
    print(f"7 & a: {rand_result}")
    print(f"a | b: {or_result}")
    print(f"8 | a: {ror_result}")
    print(f"a ^ b: {xor_result}")
    print(f"9 ^ a: {rxor_result}")


# Bitwise operations pattern explanations:
# 1. __x__      - Standard operation: a @ b (where @ is the operator)
# 2. __rx__     - Reflected operation: b @ a (when b doesn't implement __x__)
# 3. __ix__     - In-place operation: a @= b (modifies a)

# Key concepts to understand:
# - Bitwise operations work on the binary representation of integers
# - Each special method defines how a custom class responds to a specific operator
# - Reflected methods (__r*__) are called when the left operand doesn't support the operation
# - In-place methods (__i*__) modify the object directly instead of creating a new one
# - When implementing these methods, always consider type checking and conversion

# To run the demonstration, call: 
demonstrate_bitwise_operations()