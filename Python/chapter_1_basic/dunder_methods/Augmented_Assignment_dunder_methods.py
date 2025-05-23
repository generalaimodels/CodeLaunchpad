# Python Special Methods (Dunder Methods): Augmented Assignment Operators
# ========================================================================


"""
Dunder methods (double underscore methods) allow Python classes to emulate built-in
behavior and operations. Augmented assignment dunder methods handle operations like 
+=, -=, etc., providing in-place modification semantics.

Key characteristics:
1. Return self (for chainable operations)
2. Modify the object in-place when possible
3. Fall back to __add__ + assignment if not implemented
"""

# Important notes on augmented assignment dunder methods:
"""
1. Performance: These methods can be more efficient than their non-augmented 
   counterparts (like __add__) because they modify the object in-place when possible.

2. Fallback behavior: If an object doesn't implement the appropriate __i*__ method,
   Python falls back to the regular operator (__add__, etc.) and then assigns the result.

3. Return value: Always return self to support chaining operations (x += y += z).

4. Immutable types: For immutable types (like int, str, tuple), Python still uses
   these methods internally, but they must return a new object since the original
   cannot be modified.

5. List extension behavior: Note that list's __iadd__ behaves like extend(),
   while list + list creates a new list.
"""

# 1. __iadd__ (+=): In-place addition
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    # Without __iadd__, v1 += v2 would use __add__ and create a new object
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    # With __iadd__, v1 += v2 modifies v1 in-place (more efficient)
    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self  # Must return self for chaining operations

# 2. __isub__ (-=): In-place subtraction
class Counter:
    def __init__(self, value=0):
        self.value = value
    
    def __repr__(self):
        return f"Counter({self.value})"
    
    def __isub__(self, amount):
        self.value -= amount
        return self

# 3. __imul__ (*=): In-place multiplication
class Matrix:
    def __init__(self, data):
        self.data = data  # Assume 2D list
    
    def __repr__(self):
        return f"Matrix({self.data})"
    
    def __imul__(self, scalar):
        # Scale every element in the matrix
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                self.data[i][j] *= scalar
        return self

# 4. __imatmul__ (@=): In-place matrix multiplication (Python 3.5+)
class TensorWrapper:
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return f"TensorWrapper({self.data})"
    
    def __imatmul__(self, other):
        # Simplified example: element-wise multiply
        # Real matrix multiplication would be more complex
        for i in range(len(self.data)):
            self.data[i] *= other.data[i]
        return self

# 5. __itruediv__ (/=): In-place true division
class Budget:
    def __init__(self, amount):
        self.amount = amount
    
    def __repr__(self):
        return f"Budget({self.amount})"
    
    def __itruediv__(self, divisor):
        self.amount /= divisor
        return self

# 6. __ifloordiv__ (//=): In-place floor division
class ResourceAllocator:
    def __init__(self, total):
        self.total = total
    
    def __repr__(self):
        return f"ResourceAllocator({self.total})"
    
    def __ifloordiv__(self, parts):
        self.total //= parts  # Integer division
        return self

# 7. __imod__ (%=): In-place modulo
class CircularCounter:
    def __init__(self, value, max_value):
        self.value = value
        self.max_value = max_value
    
    def __repr__(self):
        return f"CircularCounter({self.value}, max={self.max_value})"
    
    def __imod__(self, other):
        self.value %= other
        return self

# 8. __ipow__ (**=): In-place exponentiation
class GrowthModel:
    def __init__(self, base):
        self.base = base
    
    def __repr__(self):
        return f"GrowthModel({self.base})"
    
    def __ipow__(self, exponent):
        self.base **= exponent
        return self

# Demonstration of all operators
def demonstrate_operators():
    # __iadd__ demonstration
    v1 = Vector(1, 2)
    v2 = Vector(3, 4)
    print(f"Before __iadd__: {v1}")
    v1 += v2  # Uses __iadd__
    print(f"After __iadd__: {v1}\n")
    
    # __isub__ demonstration
    c = Counter(10)
    print(f"Before __isub__: {c}")
    c -= 3  # Uses __isub__
    print(f"After __isub__: {c}\n")
    
    # __imul__ demonstration
    m = Matrix([[1, 2], [3, 4]])
    print(f"Before __imul__: {m}")
    m *= 2  # Uses __imul__
    print(f"After __imul__: {m}\n")
    
    # __imatmul__ demonstration
    t1 = TensorWrapper([1, 2, 3])
    t2 = TensorWrapper([4, 5, 6])
    print(f"Before __imatmul__: {t1}")
    t1 @= t2  # Uses __imatmul__
    print(f"After __imatmul__: {t1}\n")
    
    # __itruediv__ demonstration
    b = Budget(1000)
    print(f"Before __itruediv__: {b}")
    b /= 2  # Uses __itruediv__
    print(f"After __itruediv__: {b}\n")
    
    # __ifloordiv__ demonstration
    r = ResourceAllocator(100)
    print(f"Before __ifloordiv__: {r}")
    r //= 3  # Uses __ifloordiv__
    print(f"After __ifloordiv__: {r}\n")
    
    # __imod__ demonstration
    cc = CircularCounter(17, 10)
    print(f"Before __imod__: {cc}")
    cc %= 10  # Uses __imod__
    print(f"After __imod__: {cc}\n")
    
    # __ipow__ demonstration
    g = GrowthModel(2)
    print(f"Before __ipow__: {g}")
    g **= 3  # Uses __ipow__
    print(f"After __ipow__: {g}")



# Run the demonstration
if __name__ == "__main__":
    demonstrate_operators()