#################################################
# PYTHON BUILT-IN FUNCTIONS DEEP DIVE
# min(), abs(), and round() functions explained
#################################################

#####################
# min() FUNCTION
#####################
"""
min() function returns the smallest item in an iterable or the smallest of two or more arguments.

Syntax:
    min(iterable, *[, key, default])
    min(arg1, arg2, *args[, key])

Parameters:
    - iterable: An iterable such as list, tuple, set, dictionary, etc.
    - arg1, arg2, *args: Any number of positional arguments
    - key: A function that serves as a key for the sort comparison (optional)
    - default: A value to return if the iterable is empty (added in Python 3.4)

Return Value:
    The smallest item in the iterable or the smallest of the positional arguments
"""

# BASIC USAGE
# Example 1: Finding minimum in a list of numbers
numbers = [5, 2, 9, 1, 7]
min_number = min(numbers)
print(f"Minimum number in {numbers}: {min_number}")  # Output: Minimum number in [5, 2, 9, 1, 7]: 1

# Example 2: Finding minimum of multiple arguments
min_value = min(8, 3, 12, 2)
print(f"Minimum of 8, 3, 12, 2: {min_value}")  # Output: Minimum of 8, 3, 12, 2: 2

# Example 3: Finding minimum string (lexicographically)
words = ["apple", "banana", "cherry", "date"]
min_word = min(words)
print(f"Minimum word in {words}: {min_word}")  # Output: Minimum word in ['apple', 'banana', 'cherry', 'date']: apple

# ADVANCED USAGE
# Example 4: Using key parameter to customize comparison
# Find the shortest string
strings = ["python", "java", "c", "javascript", "go"]
shortest = min(strings, key=len)
print(f"Shortest string in {strings}: {shortest}")  # Output: Shortest string in ['python', 'java', 'c', 'javascript', 'go']: c

# Example 5: Using key with custom objects
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f"Person({self.name}, {self.age})"

people = [
    Person("Alice", 25),
    Person("Bob", 19),
    Person("Charlie", 32)
]

youngest = min(people, key=lambda p: p.age)
print(f"Youngest person: {youngest}")  # Output: Youngest person: Person(Bob, 19)

# Example 6: Using default parameter (Python 3.4+)
empty_list = []
try:
    # This will raise ValueError without default
    print(min(empty_list))
except ValueError as e:
    print(f"Error without default: {e}")  # Output: Error without default: min() arg is an empty sequence

# Using default parameter to handle empty iterables
min_with_default = min(empty_list, default=0)
print(f"Min with default: {min_with_default}")  # Output: Min with default: 0

# Example 7: Working with dictionaries
student_scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
# Get student with minimum score
min_student = min(student_scores, key=student_scores.get)
print(f"Student with minimum score: {min_student}")  # Output: Student with minimum score: Charlie

# EDGE CASES AND EXCEPTIONS
# Example 8: Type mixing (will throw TypeError)
try:
    min_mixed = min([1, 2, "3", 4])
except TypeError as e:
    print(f"Type mixing error: {e}")  # Output: Type mixing error: '<' not supported between instances of 'str' and 'int'

# Example 9: NaN values in floating point comparison
import math
numbers_with_nan = [5.0, 2.5, float('nan'), 1.0]
# Note: NaN is not considered the minimum value
try:
    print(min(numbers_with_nan))
except ValueError as e:
    print(f"NaN comparison error: {e}")  # Output depends on implementation, NaN comparisons are problematic

# Example 10: min() with unhashable types
# This works fine
min_list_of_lists = min([[3, 2], [1, 5], [2, 1]], key=lambda x: x[0])
print(f"Min list based on first element: {min_list_of_lists}")  # Output: Min list based on first element: [1, 5]


#####################
# abs() FUNCTION
#####################
"""
abs() function returns the absolute value of a number.

Syntax:
    abs(number)

Parameters:
    - number: A number (integer, float, or complex)

Return Value:
    The absolute value of the number
    For complex numbers, it returns the magnitude (sqrt(real² + imag²))
"""

# BASIC USAGE
# Example 1: Absolute value of integers
print(f"abs(-5): {abs(-5)}")  # Output: abs(-5): 5
print(f"abs(5): {abs(5)}")    # Output: abs(5): 5

# Example 2: Absolute value of floating-point numbers
print(f"abs(-3.14): {abs(-3.14)}")  # Output: abs(-3.14): 3.14
print(f"abs(0.0): {abs(0.0)}")      # Output: abs(0.0): 0.0

# ADVANCED USAGE
# Example 3: Absolute value of complex numbers
complex_num = complex(3, 4)  # 3 + 4j
abs_complex = abs(complex_num)
print(f"abs({complex_num}): {abs_complex}")  # Output: abs((3+4j)): 5.0 (magnitude: √(3² + 4²) = 5)

# Example 4: Using abs() in custom classes with __abs__ method
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __abs__(self):
        return (self.x**2 + self.y**2)**0.5
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

vector = Vector(3, 4)
print(f"abs({vector}): {abs(vector)}")  # Output: abs(Vector(3, 4)): 5.0

# Example 5: Using abs() in mathematical calculations
def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return abs(complex(p2[0] - p1[0], p2[1] - p1[1]))

point1 = (0, 0)
point2 = (3, 4)
print(f"Distance between {point1} and {point2}: {distance(point1, point2)}")  # Output: Distance between (0, 0) and (3, 4): 5.0

# EDGE CASES AND EXCEPTIONS
# Example 6: Non-numeric types
try:
    abs("string")
except TypeError as e:
    print(f"TypeError: {e}")  # Output: TypeError: bad operand type for abs(): 'str'

# Example 7: Overflow with large numbers
import sys
# Will work but can potentially overflow on some systems with limited precision
big_number = -2**31
print(f"abs({big_number}): {abs(big_number)}")

# Example 8: abs() with custom __abs__ implementation that returns non-numeric value
class BadAbs:
    def __abs__(self):
        return "not a number"

try:
    abs(BadAbs())
except TypeError as e:
    print(f"Bad __abs__ implementation: {e}")  # Error: __abs__ returned non-number type


#####################
# round() FUNCTION
#####################
"""
round() function returns a floating-point number rounded to the specified number of decimals.

Syntax:
    round(number[, ndigits])

Parameters:
    - number: A numeric value to round
    - ndigits: Number of decimal places to round to (default is 0)

Return Value:
    The rounded value
"""

# BASIC USAGE
# Example 1: Rounding to the nearest integer (default)
print(f"round(3.7): {round(3.7)}")        # Output: round(3.7): 4
print(f"round(3.2): {round(3.2)}")        # Output: round(3.2): 3
print(f"round(-3.7): {round(-3.7)}")      # Output: round(-3.7): -4
print(f"round(-3.2): {round(-3.2)}")      # Output: round(-3.2): -3

# Example 2: Rounding to specified decimal places
print(f"round(3.14159, 2): {round(3.14159, 2)}")    # Output: round(3.14159, 2): 3.14
print(f"round(3.14159, 4): {round(3.14159, 4)}")    # Output: round(3.14159, 4): 3.1416

# Example 3: Rounding large numbers with negative ndigits
print(f"round(1234, -2): {round(1234, -2)}")        # Output: round(1234, -2): 1200
print(f"round(5678, -3): {round(5678, -3)}")        # Output: round(5678, -3): 6000

# IMPORTANT BEHAVIOR TO UNDERSTAND
# Example 4: Python's "Banker's Rounding" - ties are rounded to the nearest even digit
print(f"round(2.5): {round(2.5)}")    # Output: round(2.5): 2
print(f"round(3.5): {round(3.5)}")    # Output: round(3.5): 4

# This is the IEEE-754 rounding standard to minimize rounding bias in statistical computations
# It rounds ties to the nearest even number: 0.5 rounds to 0, 1.5 to 2, 2.5 to 2, 3.5 to 4, etc.

# Example 5: Rounding with precision issues in floating point
print(f"round(2.675, 2): {round(2.675, 2)}")  # Might output 2.67 instead of 2.68 due to floating point precision

# ADVANCED USAGE
# Example 6: Rounding in financial calculations (where banker's rounding might be desired)
def calculate_interest(principal, rate, years):
    """Calculate simple interest rounded to 2 decimal places"""
    interest = principal * rate * years
    return round(interest, 2)

print(f"Interest on $1000 at 5.5% for 2 years: ${calculate_interest(1000, 0.055, 2)}")
# Output: Interest on $1000 at 5.5% for 2 years: $110.0

# EDGE CASES AND EXCEPTIONS
# Example 7: Non-numeric types
try:
    round("3.14")
except TypeError as e:
    print(f"TypeError: {e}")  # Output: TypeError: type str doesn't define __round__ method

# Example 8: Rounding with custom classes
class Price:
    def __init__(self, amount):
        self.amount = amount
    
    def __round__(self, ndigits=None):
        if ndigits is None:
            return round(self.amount)
        return round(self.amount, ndigits)
    
    def __repr__(self):
        return f"Price(${self.amount})"

price = Price(19.995)
print(f"Original: {price}, Rounded: {round(price, 2)}")  # Output: Original: Price($19.995), Rounded: 20.0

# Example 9: Avoiding floating point issues using Decimal
from decimal import Decimal, ROUND_HALF_UP

# More predictable rounding behavior
d = Decimal('2.675')
rounded_decimal = d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
print(f"Decimal('2.675') rounded to 2 places: {rounded_decimal}")  # Output: Decimal('2.675') rounded to 2 places: 2.68

# Example 10: Practical application - rounding percentages
scores = [85.6, 92.3, 78.9, 90.5]
rounded_scores = [round(score) for score in scores]
print(f"Original scores: {scores}")
print(f"Rounded scores: {rounded_scores}")
# Output: 
# Original scores: [85.6, 92.3, 78.9, 90.5]
# Rounded scores: [86, 92, 79, 90]