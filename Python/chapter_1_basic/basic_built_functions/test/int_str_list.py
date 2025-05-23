# ==================================================================================
# COMPREHENSIVE GUIDE TO PYTHON'S BUILT-IN TYPES AND THEIR FUNCTIONS
# ==================================================================================
# This file explains the built-in functions and methods for int, str, and list types
# with detailed examples and exception cases to help you become an advanced Python coder.
# ==================================================================================


# ==================================================================================
# INT TYPE
# ==================================================================================
# The int type represents whole numbers (integers) in Python

# ----- INT CREATION AND CONVERSION -----

# Creating integers
x = 42  # Direct assignment
y = int(3.14)  # Convert float to int (truncates decimal part)
z = int("100")  # Convert string to int
b = int("1010", 2)  # Convert binary string to int (base 2)
h = int("1A", 16)  # Convert hex string to int (base 16)
o = int("12", 8)  # Convert octal string to int (base 8)

# Exception cases:
# int("3.14")  # ValueError: invalid literal for int() with base 10
# int("ABC")   # ValueError: invalid literal for int() with base 10
# int("", 16)  # ValueError: invalid literal for int() with base 16
# int(None)    # TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'

# ----- INT OPERATIONS -----

# Basic arithmetic
add = 5 + 3  # Addition: 8
sub = 5 - 3  # Subtraction: 2
mul = 5 * 3  # Multiplication: 15
div = 5 / 3  # Division (returns float): 1.6666666666666667
floordiv = 5 // 3  # Floor division (returns int): 1
mod = 5 % 3  # Modulus (remainder): 2
power = 5 ** 3  # Exponentiation: 125
neg = -5  # Negation: -5

# Bitwise operations
bitwise_and = 5 & 3  # Bitwise AND: 1  (101 & 011 = 001)
bitwise_or = 5 | 3  # Bitwise OR: 7   (101 | 011 = 111)
bitwise_xor = 5 ^ 3  # Bitwise XOR: 6  (101 ^ 011 = 110)
bitwise_not = ~5  # Bitwise NOT: -6  (~101 = ...111010)
left_shift = 5 << 1  # Left shift: 10  (101 << 1 = 1010)
right_shift = 5 >> 1  # Right shift: 2  (101 >> 1 = 10)

# ----- INT METHODS -----

# bit_length() - returns number of bits needed to represent the integer
bit_len = (5).bit_length()  # 3 bits needed for 5 (binary 101)
large_num = (1024).bit_length()  # 11 bits needed for 1024 (binary 10000000000)

# to_bytes() - convert int to bytes representation
bytes_representation = (1024).to_bytes(2, byteorder='big')  # b'\x04\x00'
bytes_representation_little = (1024).to_bytes(2, byteorder='little')  # b'\x00\x04'
# Exception case:
# (256).to_bytes(1, byteorder='big')  # OverflowError: int too big to convert

# from_bytes() - convert bytes to int
int_from_bytes = int.from_bytes(b'\x04\x00', byteorder='big')  # 1024
int_from_bytes_little = int.from_bytes(b'\x00\x04', byteorder='little')  # 1024

# ----- INT CASTING AND CONVERSION -----

# Converting int to other types
int_to_float = float(42)  # 42.0
int_to_str = str(42)  # "42"
int_to_bool = bool(42)  # True (any non-zero int is True)
int_to_bool_zero = bool(0)  # False (zero is False)

# hex(), bin(), oct() - built-in functions for base conversion
hex_value = hex(42)  # '0x2a'
bin_value = bin(42)  # '0b101010'
oct_value = oct(42)  # '0o52'

# ----- INT SPECIAL ATTRIBUTES AND LIMITATIONS -----

max_int = pow(2, 63) - 1  # Large int example
# Python can handle arbitrary precision integers, limited only by available memory
very_large_int = 9999 ** 99  # Python handles this without overflow

# Exception cases:
# a = 1 / 0  # ZeroDivisionError: division by zero
# b = 5 // 0  # ZeroDivisionError: integer division or modulo by zero



# ==================================================================================
# STR TYPE
# ==================================================================================
# The str type represents sequences of characters (text) in Python

# ----- STRING CREATION AND CONVERSION -----

# Creating strings
s1 = "Hello"  # Double quotes
s2 = 'World'  # Single quotes
s3 = """Multiline
string"""  # Triple quotes for multiline strings
s4 = str(42)  # Convert int to string: "42"
s5 = str(3.14)  # Convert float to string: "3.14"
s6 = str([1, 2, 3])  # Convert list to string: "[1, 2, 3]"

# Raw strings (backslashes are treated as literal)
raw_str = r"C:\Users\name\documents"  # Backslashes are not escape characters

# f-strings (formatted string literals) - Python 3.6+
name = "Alice"
age = 30
f_string = f"Name: {name}, Age: {age}"  # "Name: Alice, Age: 30"
f_expr = f"2 + 2 = {2 + 2}"  # "2 + 2 = 4"

# ----- STRING INDEXING AND SLICING -----

# Indexing (0-based)
s = "Python"
first_char = s[0]  # 'P'
last_char = s[-1]  # 'n'
# Exception case:
# error_char = s[10]  # IndexError: string index out of range

# Slicing [start:end:step]
substring = s[1:4]  # 'yth' (from index 1 to 3)
from_start = s[:3]  # 'Pyt' (from start to index 2)
to_end = s[2:]  # 'thon' (from index 2 to end)
reversed_str = s[::-1]  # 'nohtyP' (reversed string)
step_by_two = s[::2]  # 'Pto' (every second character)

# ----- STRING METHODS: SEARCHING -----

# find(), index(), rfind(), rindex()
text = "Python is amazing and Python is fun"
find_pos = text.find("Python")  # 0 (first occurrence)
rfind_pos = text.rfind("Python")  # 21 (last occurrence)
find_missing = text.find("Java")  # -1 (not found)
# Exception case:
# index_missing = text.index("Java")  # ValueError: substring not found

# count() - count occurrences
count_python = text.count("Python")  # 2
count_empty = text.count("")  # 36 (length + 1)

# startswith(), endswith()
starts_with = text.startswith("Python")  # True
ends_with = text.endswith("fun")  # True
starts_with_tuple = text.startswith(("Java", "Python"))  # True

# ----- STRING METHODS: CASE MANIPULATION -----

# upper(), lower(), capitalize(), title(), swapcase()
upper_case = "hello".upper()  # "HELLO"
lower_case = "HELLO".lower()  # "hello"
capitalized = "hello world".capitalize()  # "Hello world"
title_case = "hello world".title()  # "Hello World"
swapped = "Hello World".swapcase()  # "hELLO wORLD"

# casefold() - more aggressive lowercase (for comparisons)
german_ss = "Straße".casefold()  # "strasse"
lower_german_ss = "Straße".lower()  # "straße"

# ----- STRING METHODS: MODIFICATION -----

# replace() - replace occurrences
replaced = "Hello World".replace("World", "Python")  # "Hello Python"
replace_limit = "aaa".replace("a", "b", 2)  # "bba"

# strip(), lstrip(), rstrip() - remove whitespace or specified chars
whitespace = "  hello  "
stripped = whitespace.strip()  # "hello"
left_stripped = whitespace.lstrip()  # "hello  "
right_stripped = whitespace.rstrip()  # "  hello"
custom_strip = "xxx hello xxx".strip("x")  # " hello "

# split(), rsplit(), splitlines() - split string into list
words = "apple,banana,orange".split(",")  # ['apple', 'banana', 'orange']
rsplit_limit = "a:b:c:d".rsplit(":", 2)  # ['a:b', 'c', 'd']
lines = "line1\nline2\nline3".splitlines()  # ['line1', 'line2', 'line3']
split_empty = "".split(",")  # [''] (one empty string)

# join() - join list of strings
joined = ",".join(["apple", "banana", "orange"])  # "apple,banana,orange"
# Exception case:
# ",".join([1, 2, 3])  # TypeError: sequence item 0: expected str instance, int found

# ----- STRING METHODS: TESTING -----

# isalpha(), isdigit(), isalnum(), isspace(), etc.
alpha_check = "Hello".isalpha()  # True
digit_check = "123".isdigit()  # True
alnum_check = "Hello123".isalnum()  # True
space_check = "   ".isspace()  # True
lower_check = "hello".islower()  # True
upper_check = "HELLO".isupper()  # True
title_check = "Hello World".istitle()  # True
empty_check = "".isalpha()  # False (empty string returns False)

# ----- STRING METHODS: FORMATTING -----

# format() - format string with placeholders
template = "Name: {}, Age: {}"
formatted = template.format("Alice", 30)  # "Name: Alice, Age: 30"
named_format = "Name: {name}, Age: {age}".format(name="Bob", age=25)  # "Name: Bob, Age: 25"

# center(), ljust(), rjust(), zfill() - alignment/padding
centered = "hello".center(11)  # "   hello   "
left_just = "hello".ljust(10, '-')  # "hello-----"
right_just = "hello".rjust(10, '-')  # "-----hello"
zeroes = "42".zfill(5)  # "00042"

# ----- STRING ENCODING/DECODING -----

# encode(), decode() - convert between str and bytes
encoded = "Pythön".encode("utf-8")  # b'Pyth\xc3\xb6n'
decoded = encoded.decode("utf-8")  # "Pythön"
# Exception cases:
# "Pythön".encode("ascii")  # UnicodeEncodeError: 'ascii' codec can't encode character '\xf6'
# b'\xff'.decode("utf-8")  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff

# ----- STRING OPERATORS -----

# Concatenation (+)
concat = "Hello" + " " + "World"  # "Hello World"

# Repetition (*)
repeat = "abc" * 3  # "abcabcabc"

# Membership (in)
contains = "y" in "Python"  # True

# Comparison operators
equal = "apple" == "apple"  # True
greater = "banana" > "apple"  # True (lexicographical comparison)
less = "apple" < "banana"  # True

# ----- STRING IMMUTABILITY -----

# Strings are immutable - cannot be changed after creation
s = "hello"
# s[0] = "H"  # TypeError: 'str' object does not support item assignment

# To modify a string, create a new one
s = "H" + s[1:]  # "Hello"

# ----- STRING CONSTANTS -----

import string

ascii_letters = string.ascii_letters  # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
ascii_lowercase = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
ascii_uppercase = string.ascii_uppercase  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
digits = string.digits  # '0123456789'
punctuation = string.punctuation  # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'



# ==================================================================================
# LIST TYPE
# ==================================================================================
# Lists are ordered, mutable collections that can hold any type of object

# ----- LIST CREATION -----

# Creating lists
empty_list = []  # Empty list
numbers = [1, 2, 3, 4, 5]  # List of integers
mixed = [1, "hello", 3.14, True]  # List of mixed types
nested = [1, [2, 3], [4, [5, 6]]]  # Nested lists

# List constructor
list_from_tuple = list((1, 2, 3))  # [1, 2, 3]
list_from_string = list("hello")  # ['h', 'e', 'l', 'l', 'o']
list_from_range = list(range(5))  # [0, 1, 2, 3, 4]

# List comprehensions (powerful way to create lists)
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
even_squares = [x**2 for x in range(10) if x % 2 == 0]  # [0, 4, 16, 36, 64]

# ----- LIST INDEXING AND SLICING -----

# Indexing (0-based)
lst = [10, 20, 30, 40, 50]
first = lst[0]  # 10
last = lst[-1]  # 50
# Exception case:
# out_of_bounds = lst[10]  # IndexError: list index out of range

# Slicing [start:end:step]
subset = lst[1:4]  # [20, 30, 40]
from_start = lst[:3]  # [10, 20, 30]
to_end = lst[2:]  # [30, 40, 50]
reversed_list = lst[::-1]  # [50, 40, 30, 20, 10]
every_other = lst[::2]  # [10, 30, 50]

# ----- LIST MODIFICATION -----

# Changing elements
lst = [10, 20, 30, 40, 50]
lst[0] = 100  # [100, 20, 30, 40, 50]

# append() - add single element to end
lst.append(60)  # [100, 20, 30, 40, 50, 60]

# insert() - insert element at specific position
lst.insert(1, 15)  # [100, 15, 20, 30, 40, 50, 60]
lst.insert(100, 70)  # Insert beyond end: [100, 15, 20, 30, 40, 50, 60, 70]

# extend() - add multiple elements
lst.extend([80, 90])  # [100, 15, 20, 30, 40, 50, 60, 70, 80, 90]

# List concatenation (+)
lst2 = [110, 120]
combined = lst + lst2  # Creates new list: [100, 15, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120]

# remove() - remove first occurrence by value
lst = [10, 20, 30, 20, 40]
lst.remove(20)  # [10, 30, 20, 40]
# Exception case:
# lst.remove(100)  # ValueError: list.remove(x): x not in list

# pop() - remove by index and return value
lst = [10, 20, 30, 40, 50]
popped = lst.pop(2)  # popped = 30, lst = [10, 20, 40, 50]
popped_last = lst.pop()  # Default removes last: popped_last = 50, lst = [10, 20, 40]
# Exception case:
# lst.pop(100)  # IndexError: pop index out of range

# clear() - remove all elements
lst.clear()  # lst = []

# del statement - delete elements or slices
lst = [10, 20, 30, 40, 50]
del lst[0]  # [20, 30, 40, 50]
del lst[1:3]  # [20, 50]

# ----- LIST OPERATIONS -----

# len() - get list length
length = len([10, 20, 30])  # 3

# count() - count occurrences of a value
occurrences = [1, 2, 2, 3, 2, 4].count(2)  # 3

# index() - find first index of a value
idx = [10, 20, 30, 20].index(20)  # 1
# With bounds
idx_in_range = [10, 20, 30, 20, 40].index(20, 2)  # 3 (search from index 2)
# Exception case:
# [10, 20, 30].index(40)  # ValueError: 40 is not in list

# in operator - check if value exists
exists = 20 in [10, 20, 30]  # True
not_exists = 40 in [10, 20, 30]  # False

# ----- LIST SORTING AND ORDERING -----

# sort() - sort list in-place
lst = [3, 1, 4, 1, 5, 9]
lst.sort()  # [1, 1, 3, 4, 5, 9]
lst.sort(reverse=True)  # [9, 5, 4, 3, 1, 1]

# Custom sort with key function
words = ["apple", "Banana", "cherry"]
words.sort()  # ['Banana', 'apple', 'cherry'] (capital letters come first)
words.sort(key=str.lower)  # ['apple', 'Banana', 'cherry'] (case-insensitive)

# sorted() - returns new sorted list (original unchanged)
lst = [3, 1, 4, 1, 5, 9]
sorted_lst = sorted(lst)  # sorted_lst = [1, 1, 3, 4, 5, 9], lst unchanged
desc_sorted = sorted(lst, reverse=True)  # [9, 5, 4, 3, 1, 1]

# reverse() - reverse list in-place
lst = [1, 2, 3, 4]
lst.reverse()  # [4, 3, 2, 1]

# reversed() - returns iterator over reversed list
lst = [1, 2, 3, 4]
rev_iterator = reversed(lst)  # Iterator
rev_list = list(reversed(lst))  # [4, 3, 2, 1]

# ----- LIST COPYING -----

# Copy issues (reference vs. value)
original = [1, 2, 3]
reference = original  # Both variables point to same list
reference.append(4)  # original also changes: [1, 2, 3, 4]

# Shallow copy methods
shallow1 = original.copy()  # Method 1
shallow2 = list(original)  # Method 2
shallow3 = original[:]  # Method 3

# Shallow copy limitations with nested lists
nested = [1, [2, 3]]
shallow = nested.copy()
shallow[1].append(4)  # Changes nested too: nested = [1, [2, 3, 4]]

# Deep copy for nested structures
import copy
deep = copy.deepcopy(nested)
deep[1].append(5)  # nested remains [1, [2, 3, 4]]

# ----- LIST COMPREHENSIONS (ADVANCED) -----

# Multiple conditions
complex_comp = [x for x in range(100) if x % 2 == 0 if x % 5 == 0]  # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# If-else in list comprehension
conditional = ["even" if x % 2 == 0 else "odd" for x in range(5)]  # ['even', 'odd', 'even', 'odd', 'even']

# Nested list comprehensions
matrix = [[i + j for j in range(3)] for i in range(3)]
# [[0, 1, 2], [1, 2, 3], [2, 3, 4]]

# Flattening a 2D list
flat = [item for sublist in matrix for item in sublist]
# [0, 1, 2, 1, 2, 3, 2, 3, 4]

# ----- LIST CONVERSION -----

# Convert to other types
lst = [1, 2, 3]
as_tuple = tuple(lst)  # (1, 2, 3)
as_set = set(lst)  # {1, 2, 3}
as_string = str(lst)  # "[1, 2, 3]"

# Join list of strings
words = ["Hello", "World"]
sentence = " ".join(words)  # "Hello World"

# ----- LIST PERFORMANCE CONSIDERATIONS -----

# Time complexity:
# - Accessing by index: O(1)
# - Append/pop at end: O(1) amortized
# - Insert/pop at beginning: O(n)
# - Search (in operator): O(n)
# - Sort: O(n log n)

# Memory usage:
# Lists use more memory than tuples
# Lists have overhead for potential growth

# ----- LIST UNPACKING -----

# Basic unpacking
lst = [1, 2, 3]
a, b, c = lst  # a=1, b=2, c=3
# Exception case:
# a, b = lst  # ValueError: too many values to unpack

# Extended unpacking (Python 3.x)
first, *middle, last = [1, 2, 3, 4, 5]  # first=1, middle=[2, 3, 4], last=5
head, *tail = [1, 2, 3]  # head=1, tail=[2, 3]
*beginning, end = [1, 2, 3]  # beginning=[1, 2], end=3

# ----- LIST METHODS SUMMARY -----

# Adding elements:
# - append(x): Add item to end
# - insert(i, x): Insert item at position
# - extend(iterable): Add multiple items

# Removing elements:
# - remove(x): Remove first occurrence
# - pop([i]): Remove and return item
# - clear(): Remove all items

# Information:
# - index(x[, start[, end]]): Find position
# - count(x): Count occurrences

# Ordering:
# - sort(*, key=None, reverse=False): Sort in-place
# - reverse(): Reverse in-place

# Copying:
# - copy(): Return shallow copy