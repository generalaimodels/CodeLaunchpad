#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Python Built-in Types: frozenset, complex, and memoryview
A comprehensive guide with examples and edge cases
"""

###############################################################################
# FROZENSET
###############################################################################
"""
frozenset is an immutable version of the set type. Once created, its elements
cannot be modified (no add, remove, or update operations).

Key characteristics:
- Immutable (hashable)
- Unordered
- No duplicates
- Can contain only hashable elements
- Supports set operations like union, intersection

Use cases:
- Dictionary keys
- Set elements
- When you need a constant set of values
- In situations where you need to guarantee that a collection won't change
"""

# Creating frozensets
empty_frozenset = frozenset()
from_list = frozenset([1, 2, 3, 2])  # Duplicates are removed
from_string = frozenset("hello")  # Each character becomes an element
from_dict = frozenset({"a": 1, "b": 2})  # Only keys are used
from_range = frozenset(range(5))

print("Basic frozenset examples:")
print(f"Empty: {empty_frozenset}")
print(f"From list: {from_list}")
print(f"From string: {from_string}")  # Order is not guaranteed
print(f"From dict: {from_dict}")
print(f"From range: {from_range}")
print()

# Set operations with frozensets
set_a = frozenset([1, 2, 3, 4])
set_b = frozenset([3, 4, 5, 6])

print("Set operations:")
print(f"A: {set_a}")
print(f"B: {set_b}")
print(f"Union (A | B): {set_a | set_b}")
print(f"Intersection (A & B): {set_a & set_b}")
print(f"Difference (A - B): {set_a - set_b}")
print(f"Symmetric Difference (A ^ B): {set_a ^ set_b}")
print()

# Common methods
print("Common frozenset methods:")
print(f"Is A a subset of B? {set_a.issubset(set_b)}")
print(f"Is A a superset of B? {set_a.issuperset(set_b)}")
print(f"Are A and B disjoint? {set_a.isdisjoint(set_b)}")
print()

# Using frozenset as a dictionary key (this works unlike with regular sets)
dict_with_frozenset_keys = {
    frozenset([1, 2]): "set1",
    frozenset([3, 4]): "set2"
}
print(f"Dictionary with frozenset keys: {dict_with_frozenset_keys}")
print(f"Accessing value: {dict_with_frozenset_keys[frozenset([1, 2])]}")
print()

# Nested sets - regular sets can't contain sets, but can contain frozensets
regular_set = {1, 2, frozenset([3, 4])}
print(f"Set containing a frozenset: {regular_set}")
print()

# Common exception cases with frozensets
try:
    # TypeError: frozenset objects are immutable
    from_list.add(4)
except TypeError as e:
    print(f"Exception when trying to modify a frozenset: {e}")

try:
    # TypeError: unhashable type: 'list'
    frozenset([[1, 2], [3, 4]])
except TypeError as e:
    print(f"Exception when creating frozenset with unhashable elements: {e}")
print()

# Performance comparison (frozenset vs set)
import sys
print("Memory usage comparison:")
test_list = list(range(1000))
test_set = set(test_list)
test_frozenset = frozenset(test_list)
print(f"Size of list: {sys.getsizeof(test_list)} bytes")
print(f"Size of set: {sys.getsizeof(test_set)} bytes")
print(f"Size of frozenset: {sys.getsizeof(test_frozenset)} bytes")
print()

###############################################################################
# COMPLEX
###############################################################################
"""
complex is a built-in type for complex numbers, which have a real and 
imaginary part. They take the form a + bj where a is the real part, b is 
the imaginary part, and j is the imaginary unit (√-1).

Key characteristics:
- Immutable
- Consists of a real and imaginary part
- Supports arithmetic operations
- Useful for mathematical and engineering calculations
- Important in signal processing, control systems, and physics
"""

# Creating complex numbers
# Method 1: Using the complex constructor
c1 = complex(2, 3)  # 2 + 3j
# Method 2: Using the j notation directly
c2 = 4 + 5j

print("Complex number examples:")
print(f"c1 = {c1}")
print(f"c2 = {c2}")
print(f"Type: {type(c1)}")
print()

# Accessing parts of a complex number
print("Complex number components:")
print(f"Real part of c1: {c1.real}")
print(f"Imaginary part of c1: {c1.imag}")
print()

# Basic arithmetic with complex numbers
print("Complex arithmetic:")
print(f"Addition: {c1} + {c2} = {c1 + c2}")
print(f"Subtraction: {c1} - {c2} = {c1 - c2}")
print(f"Multiplication: {c1} * {c2} = {c1 * c2}")
print(f"Division: {c1} / {c2} = {c1 / c2}")
print()

# Complex conjugate and absolute value
print("Complex operations:")
print(f"Conjugate of {c1}: {c1.conjugate()}")
print(f"Absolute value (magnitude) of {c1}: {abs(c1)}")
print()

# Using complex with other numeric types
c3 = c1 + 5  # Adding integer
c4 = c2 * 2.5  # Multiplying by float
print(f"Complex + integer: {c1} + 5 = {c3}")
print(f"Complex * float: {c2} * 2.5 = {c4}")
print()

# Other operations
import math
import cmath  # Complex math module

print("Advanced complex operations:")
print(f"Square root of -1: {cmath.sqrt(-1)}")
print(f"e^(πi): {cmath.exp(complex(0, math.pi))}")  # Should be close to -1
print(f"Polar form of {c1}: {cmath.polar(c1)}")
print(f"Phase (argument) of {c1}: {cmath.phase(c1)} radians")
print()

# Exception cases with complex numbers
try:
    # Can't take square root of negative number with math.sqrt
    math.sqrt(-1)
except ValueError as e:
    print(f"Exception with math.sqrt(-1): {e}")
    print(f"Solution: Use cmath.sqrt(-1) = {cmath.sqrt(-1)}")

try:
    # Complex numbers don't support ordering operations
    c1 < c2
except TypeError as e:
    print(f"Exception when comparing complex numbers: {e}")
print()

# Converting between formats
print("Converting to and from complex:")
# From polar form (magnitude and phase) to complex
magnitude, phase = 5, math.pi/4
c5 = cmath.rect(magnitude, phase)
print(f"Complex from polar (mag={magnitude}, phase={phase}): {c5}")

# String to complex
c6 = complex("3+4j")
print(f"Complex from string '3+4j': {c6}")
print()

# Applications example: Solving quadratic equations with complex roots
print("Application: Solving quadratic equation ax² + bx + c = 0")
a, b, c = 1, 2, 5  # Has complex roots
discriminant = b**2 - 4*a*c

if discriminant >= 0:
    x1 = (-b + math.sqrt(discriminant)) / (2*a)
    x2 = (-b - math.sqrt(discriminant)) / (2*a)
    roots = (x1, x2)
else:
    x1 = (-b + cmath.sqrt(discriminant)) / (2*a)
    x2 = (-b - cmath.sqrt(discriminant)) / (2*a)
    roots = (x1, x2)

print(f"Equation: {a}x² + {b}x + {c} = 0")
print(f"Discriminant: {discriminant}")
print(f"Roots: {roots[0]} and {roots[1]}")
print()

###############################################################################
# MEMORYVIEW
###############################################################################
"""
memoryview provides a way to access the internal data of an object that
supports the buffer protocol without copying. This is efficient for large
data structures and operations on binary data.

Key characteristics:
- Provides direct access to an object's memory
- No data copying (more memory efficient)
- Works with objects that support the buffer protocol (bytes, bytearray, array.array)
- Allows slicing and modification of data in-place
- Useful for large binary data processing

Use cases:
- Working with binary files
- Image processing
- Network protocols
- Any situation where you need to manipulate large chunks of binary data
"""

# Creating a memoryview
print("Basic memoryview examples:")
# From bytes (read-only)
byte_data = bytes(range(10))
view1 = memoryview(byte_data)
print(f"Memoryview from bytes: {view1}")
print(f"Memoryview contents: {list(view1)}")  # Convert to list to see contents

# From bytearray (mutable)
byte_array = bytearray(range(10))
view2 = memoryview(byte_array)
print(f"Memoryview from bytearray: {view2}")
print(f"Original bytearray: {list(byte_array)}")
print()

# Attributes and information
print("Memoryview information:")
print(f"Item size (bytes): {view1.itemsize}")
print(f"Format: {view1.format}")  # 'B' for unsigned char
print(f"Number of dimensions: {view1.ndim}")
print(f"Shape: {view1.shape}")
print(f"Total size (bytes): {view1.nbytes}")
print(f"Is read-only? {view1.readonly}")
print()

# Slicing memoryviews
print("Slicing memoryviews:")
# Get a slice of the view
slice_view = view2[2:6]
print(f"Slice view (indices 2-5): {list(slice_view)}")
print()

# Modifying data through memoryview (only works with mutable sources like bytearray)
print("Modifying data through memoryview:")
print(f"Original bytearray: {list(byte_array)}")

# Modify specific elements
view2[3] = 255
print(f"After modifying index 3: {list(byte_array)}")

# Modify a slice - must use compatible type
view2[5:8] = bytes([100, 101, 102])
print(f"After modifying slice 5:8: {list(byte_array)}")
print()

# Converting memoryview to other types
print("Converting memoryviews:")
print(f"To bytes: {bytes(view1)}")
print(f"To list: {list(view1)}")
print(f"To bytearray: {bytearray(view1)}")
print()

# Using memoryview with other binary types
import array

# With array.array
arr = array.array('i', [1, 2, 3, 4, 5])  # array of integers
arr_view = memoryview(arr)
print(f"Array: {arr}")
print(f"Memoryview of array: {arr_view}")
print(f"Memoryview format: {arr_view.format}")  # 'i' for signed int
print(f"Item size: {arr_view.itemsize} bytes")  # 4 bytes per integer on most systems
print()

# Memory efficiency demonstration
print("Memory efficiency demonstration:")
large_bytes = bytes(range(1000000))  # 1 million bytes

import time
import sys

# Without memoryview - creates copies
start_time = time.time()
sum_normal = sum(large_bytes)  # Creates a copy
normal_time = time.time() - start_time
print(f"Sum without memoryview: {sum_normal}")
print(f"Time taken: {normal_time:.6f} seconds")

# With memoryview - no copies
start_time = time.time()
view = memoryview(large_bytes)
sum_view = sum(view)  # No copy
view_time = time.time() - start_time
print(f"Sum with memoryview: {sum_view}")
print(f"Time taken: {view_time:.6f} seconds")
print(f"Memory of bytes: {sys.getsizeof(large_bytes)} bytes")
print(f"Memory of memoryview: {sys.getsizeof(view)} bytes")
print()

# Exception cases with memoryviews
print("Exception cases:")
try:
    # TypeError: memoryview: a bytes-like object is required, not 'str'
    memoryview("hello")
except TypeError as e:
    print(f"Creating memoryview from string: {e}")

try:
    # TypeError: cannot modify read-only memory
    view1[0] = 99  # view1 is from bytes which is immutable
except TypeError as e:
    print(f"Modifying read-only memoryview: {e}")

try:
    # ValueError: memoryview assignment: lvalue and rvalue have different structures
    view2[2:4] = [42, 43]  # Incorrect assignment type
except ValueError as e:
    print(f"Assigning wrong type to memoryview: {e}")
print()

# Advanced usage: Working with multi-dimensional arrays
print("Multi-dimensional memoryviews:")

# Create a 2D array using bytearray
width, height = 4, 3
matrix_data = bytearray(width * height)
for i in range(width * height):
    matrix_data[i] = i

# Create a 2D memoryview
matrix_view = memoryview(matrix_data).cast('B', (height, width))
print("2D matrix view:")
for row in matrix_view:
    print(list(row))

# Modify using 2D indexing
matrix_view[1, 2] = 99
print("\nAfter modification:")
for row in matrix_view:
    print(list(row))
print()

# Real-world example: Simple image data manipulation
print("Example: Manipulating image data")

# Simulate grayscale image data (0-255 for each pixel)
img_width, img_height = 5, 4
img_data = bytearray(img_width * img_height)
for i in range(img_width * img_height):
    img_data[i] = i * 10 % 256  # Some arbitrary pixel values

# Create a 2D view of the image
img_view = memoryview(img_data).cast('B', (img_height, img_width))

print("Original 'image':")
for row in img_view:
    print(list(row))

# Invert the image (255 - pixel_value)
for y in range(img_height):
    for x in range(img_width):
        img_view[y, x] = 255 - img_view[y, x]

print("\nInverted 'image':")
for row in img_view:
    print(list(row))