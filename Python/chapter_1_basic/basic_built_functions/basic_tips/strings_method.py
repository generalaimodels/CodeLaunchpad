#!/usr/bin/env python3
"""
COMPREHENSIVE GUIDE TO PYTHON STRING METHODS
===========================================

This file provides detailed examples and explanations of all built-in string methods in Python.
Each section includes example code, detailed explanations, and edge cases to help you
master string manipulation in Python.
"""

#=============================================================================
# str.capitalize()
#=============================================================================
"""
str.capitalize() converts the first character of a string to uppercase and 
the rest to lowercase.

Syntax: string.capitalize()
Parameters: None
Return: A new string with the first character capitalized and the rest lowercased

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "hello world"
print(f"'hello world'.capitalize() → '{text.capitalize()}'")  # 'Hello world'

# If the first character is already uppercase, it remains unchanged
print(f"'Hello world'.capitalize() → '{'Hello world'.capitalize()}'")  # 'Hello world'

# If the first character is not a letter, it remains unchanged
print(f"'123 hello'.capitalize() → '{'123 hello'.capitalize()}'")  # '123 hello'

# All other letters are converted to lowercase
print(f"'hELLO WORLD'.capitalize() → '{'hELLO WORLD'.capitalize()}'")  # 'Hello world'

# Empty string handling
print(f"''.capitalize() → '{''.capitalize()}'")  # ''

# Works with unicode characters too
print(f"'éducation'.capitalize() → '{'éducation'.capitalize()}'")  # 'Éducation'


#=============================================================================
# str.casefold()
#=============================================================================
"""
str.casefold() returns a casefolded copy of the string - this is an aggressive
lowercase method intended for caseless matching.

This is more aggressive than .lower() and removes all case distinctions in a string.
For example, the German lowercase letter 'ß' is equivalent to "ss".

Syntax: string.casefold()
Parameters: None
Return: A casefolded copy of the string

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage - similar to lower()
text = "Hello World"
print(f"'Hello World'.casefold() → '{text.casefold()}'")  # 'hello world'

# Difference between lower() and casefold() with special characters
text = "Straße"  # German word for "street"
print(f"'Straße'.lower() → '{text.lower()}'")      # 'straße'
print(f"'Straße'.casefold() → '{text.casefold()}'")  # 'strasse'

# Other examples with unicode characters
print(f"'Ĥĕľŀŏ'.casefold() → '{'Ĥĕľŀŏ'.casefold()}'")

# Practical use case: Case-insensitive string comparison
def case_insensitive_compare(str1, str2):
    return str1.casefold() == str2.casefold()

print(f"Comparing 'Straße' and 'STRASSE': {case_insensitive_compare('Straße', 'STRASSE')}")  # True


#=============================================================================
# str.center()
#=============================================================================
"""
str.center() returns a centered string of specified width.

Syntax: string.center(width[, fillchar])
Parameters:
    width: The total width of the resulting string
    fillchar (optional): The character to fill the extra space (default is space)
Return: A centered string padded with specified character

Time Complexity: O(n) where n is the width
"""

# Basic usage
text = "Python"
print(f"'Python'.center(10) → '{text.center(10)}'")  # '  Python  '

# Using a different fill character
print(f"'Python'.center(10, '*') → '{text.center(10, '*')}'")  # '**Python**'

# If width is less than or equal to the string length, the original string is returned
print(f"'Python'.center(6) → '{text.center(6)}'")  # 'Python'
print(f"'Python'.center(4) → '{text.center(4)}'")  # 'Python'

# If width - len(string) is odd, the extra padding goes on the right
print(f"'Python'.center(11) → '{text.center(11)}'")  # '  Python   '

# Empty string handling
print(f"''.center(5, '*') → '{''.center(5, '*')}'")  # '*****'

# fillchar must be exactly one character
try:
    "Python".center(10, "**")
except TypeError as e:
    print(f"Error when using multiple characters as fillchar: {e}")


#=============================================================================
# str.count()
#=============================================================================
"""
str.count() returns the number of non-overlapping occurrences of a substring in the string.

Syntax: string.count(substring[, start[, end]])
Parameters:
    substring: The substring to count
    start (optional): The starting index (default is 0)
    end (optional): The ending index (default is the end of the string)
Return: An integer representing the count of non-overlapping occurrences

Time Complexity: O(n*m) where n is the length of the string and m is the length of the substring
"""

# Basic usage
text = "hello world, hello python, hello programmer"
print(f"Count of 'hello' in '{text}': {text.count('hello')}")  # 3

# With start and end parameters
print(f"Count of 'hello' from index 10 to end: {text.count('hello', 10)}")  # 2
print(f"Count of 'hello' from index 10 to 25: {text.count('hello', 10, 25)}")  # 1

# Case sensitivity
print(f"Count of 'Hello' (with capital H): {text.count('Hello')}")  # 0

# Overlapping substrings are not counted
text = "abababa"
print(f"'abababa'.count('aba'): {text.count('aba')}")  # 2, not 3 (non-overlapping)

# Empty substring counts between each character plus one
print(f"'hello'.count(''): {'hello'.count('')}")  # 6 (length + 1)

# If start is greater than end, returns 0
print(f"Count when start > end: {'abababa'.count('a', 5, 3)}")  # 0

# If start is negative, it's treated as 0
print(f"Count with negative start: {'abababa'.count('a', -5)}")  # 4

# If end is negative, it's counted from the end of the string
print(f"Count with negative end: {'abababa'.count('a', 0, -2)}")  # 3


#=============================================================================
# str.encode()
#=============================================================================
"""
str.encode() returns an encoded version of the string as bytes.

Syntax: string.encode([encoding[, errors]])
Parameters:
    encoding (optional): The encoding to use (default is UTF-8)
    errors (optional): How to handle encoding errors (default is 'strict')
Return: A bytes object

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "Hello, World!"
print(f"'Hello, World!'.encode() → {text.encode()}")  # b'Hello, World!'

# Different encodings
print(f"UTF-8 encoding: {text.encode('utf-8')}")
print(f"ASCII encoding: {text.encode('ascii')}")
print(f"Latin-1 encoding: {text.encode('latin-1')}")

# Handling encoding errors
text_with_unicode = "Hello, 你好!"

# 'strict' - raises UnicodeEncodeError for characters that can't be encoded
try:
    text_with_unicode.encode('ascii', 'strict')
except UnicodeEncodeError as e:
    print(f"Error with 'strict' error handling: {e}")

# 'ignore' - ignores characters that can't be encoded
print(f"'ignore' handling: {text_with_unicode.encode('ascii', 'ignore')}")

# 'replace' - replaces characters that can't be encoded with a placeholder
print(f"'replace' handling: {text_with_unicode.encode('ascii', 'replace')}")

# 'xmlcharrefreplace' - replaces with XML character reference
print(f"'xmlcharrefreplace' handling: {text_with_unicode.encode('ascii', 'xmlcharrefreplace')}")

# 'backslashreplace' - replaces with backslashed escape sequence
print(f"'backslashreplace' handling: {text_with_unicode.encode('ascii', 'backslashreplace')}")


#=============================================================================
# str.endswith()
#=============================================================================
"""
str.endswith() returns True if the string ends with the specified suffix.

Syntax: string.endswith(suffix[, start[, end]])
Parameters:
    suffix: The suffix to check for (can be a tuple of suffixes)
    start (optional): The starting index (default is 0)
    end (optional): The ending index (default is the end of the string)
Return: A boolean indicating whether the string ends with the suffix

Time Complexity: O(n) where n is the length of the string or suffix
"""

# Basic usage
text = "hello.txt"
print(f"'hello.txt'.endswith('.txt') → {text.endswith('.txt')}")  # True

# Using a tuple of suffixes
print(f"'hello.txt'.endswith(('.txt', '.pdf')) → {text.endswith(('.txt', '.pdf'))}")  # True
print(f"'hello.txt'.endswith(('.doc', '.pdf')) → {text.endswith(('.doc', '.pdf'))}")  # False

# With start and end parameters
text = "hello.txt.bak"
print(f"'hello.txt.bak'.endswith('.txt', 0, 9) → {text.endswith('.txt', 0, 9)}")  # True

# Case sensitivity
print(f"'hello.txt'.endswith('.TXT') → {'hello.txt'.endswith('.TXT')}")  # False

# Empty string handling
print(f"''.endswith('x') → {''.endswith('x')}")  # False
print(f"''.endswith('') → {''.endswith('')}")  # True

# If suffix is an empty string, returns True
print(f"'hello'.endswith('') → {'hello'.endswith('')}")  # True

# If start is greater than end, returns False
print(f"Check when start > end: {'hello.txt'.endswith('.txt', 5, 3)}")  # False

# Error handling: suffix must be a string or a tuple of strings
try:
    "hello.txt".endswith(5)
except TypeError as e:
    print(f"Error when using non-string suffix: {e}")


#=============================================================================
# str.expandtabs()
#=============================================================================
"""
str.expandtabs() replaces tab characters ('\t') with spaces.

Syntax: string.expandtabs([tabsize])
Parameters:
    tabsize (optional): The number of spaces to replace a tab with (default is 8)
Return: A new string with tabs expanded

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "hello\tworld"
print(f"'hello\\tworld'.expandtabs() → '{text.expandtabs()}'")  # Default tabsize is 8

# Specifying different tabsize
print(f"With tabsize 4: '{text.expandtabs(4)}'")
print(f"With tabsize 2: '{text.expandtabs(2)}'")
print(f"With tabsize 16: '{text.expandtabs(16)}'")

# Multiple tabs and text alignment
text = "Name\tAge\tCountry"
print(f"Original: '{text}'")
print(f"Expanded (8): '{text.expandtabs(8)}'")

text = "John\t25\tUSA"
print(f"Original: '{text}'")
print(f"Expanded (8): '{text.expandtabs(8)}'")

# Tab position matters
text = "a\tb\tc\td"
print(f"Original: '{text}'")
print(f"Expanded (4): '{text.expandtabs(4)}'")

text = "ab\tc\td"
print(f"Original: '{text}'")
print(f"Expanded (4): '{text.expandtabs(4)}'")

# If tabsize is negative, it's treated as 0
print(f"With negative tabsize: '{text.expandtabs(-4)}'")

str.split()
#=============================================================================
# str.find()
#=============================================================================
"""
str.find() returns the lowest index where the substring is found.

Syntax: string.find(substring[, start[, end]])
Parameters:
    substring: The substring to search for
    start (optional): The starting index (default is 0)
    end (optional): The ending index (default is the end of the string)
Return: The lowest index where substring is found, or -1 if not found

Time Complexity: O(n*m) where n is the length of the string and m is the length of the substring
"""

# Basic usage
text = "hello world"
print(f"'hello world'.find('world') → {text.find('world')}")  # 6

# Substring not found
print(f"'hello world'.find('python') → {text.find('python')}")  # -1

# With start and end parameters
text = "hello world, hello python"
print(f"Find 'hello' from start: {text.find('hello')}")  # 0
print(f"Find 'hello' from index 5: {text.find('hello', 5)}")  # 13
print(f"Find 'hello' from index 5 to 15: {text.find('hello', 5, 15)}")  # 13

# Case sensitivity
print(f"Find 'Hello' (with capital H): {text.find('Hello')}")  # -1

# Empty substring
print(f"Find empty string: {text.find('')}")  # 0
print(f"Find empty string from index 5: {text.find('', 5)}")  # 5

# If start is greater than end, returns -1
print(f"Find when start > end: {text.find('hello', 5, 3)}")  # -1

# Comparison with index() - find() returns -1 when not found, index() raises ValueError
try:
    pos = text.index("python!")
except ValueError as e:
    print(f"Error with index(): {e}")

pos = text.find("python!")
print(f"Find result for 'python!': {pos}")  # -1


#=============================================================================
# str.format()
#=============================================================================
"""
str.format() formats the string by replacing placeholders with values.

Syntax: string.format(*args, **kwargs)
Parameters:
    *args: Positional arguments
    **kwargs: Keyword arguments
Return: A formatted string

Time Complexity: Depends on the complexity of the formatting
"""

# Basic usage
template = "Hello, {}!"
print(f"'Hello, {{}}!'.format('World') → '{template.format('World')}'")  # 'Hello, World!'

# Multiple placeholders
template = "{} plus {} equals {}"
print(f"'{template}'.format(2, 3, 5) → '{template.format(2, 3, 5)}'")

# Using indices
template = "{0} {1} {0}"
print(f"'{template}'.format('hello', 'world') → '{template.format('hello', 'world')}'")  # 'hello world hello'

# Using named placeholders
template = "Hello, {name}! You are {age} years old."
print(f"With named placeholders: '{template.format(name='Alice', age=30)}'")

# Formatting numbers
template = "Value: {:.2f}"  # Two decimal places
print(f"Formatting float: '{template.format(3.14159)}'")  # 'Value: 3.14'

# Padding and alignment
# <: left-align, >: right-align, ^: center
print("Left align: '{:<10}'.".format('test'))  # 'test      '
print("Right align: '{:>10}'.".format('test'))  # '      test'
print("Center: '{:^10}'.".format('test'))      # '   test   '

# With specific character for padding
print("Pad with '-': '{:-^10}'.".format('test'))  # '---test---'

# Thousands separator
print("With thousand separator: '{:,}'.".format(1234567))  # '1,234,567'

# Binary, octal, hex formats
print("Binary: '{:b}'.".format(10))    # '1010'
print("Octal: '{:o}'.".format(10))     # '12'
print("Hex: '{:x}'.".format(10))       # 'a'
print("HEX: '{:X}'.".format(10))       # 'A'

# Datetime formatting
import datetime
now = datetime.datetime(2023, 1, 1, 12, 30, 45)
print("Date: '{:%Y-%m-%d %H:%M:%S}'.".format(now))  # '2023-01-01 12:30:45'

# Nested formatting
data = {'name': 'Alice', 'age': 30}
template = "Name: {0[name]}, Age: {0[age]}"
print(f"Accessing dict elements: '{template.format(data)}'")

# Handling potential errors
try:
    "{} {}".format("too", "many", "arguments")
except Exception as e:
    print(f"Error with too many args: {e}")

try:
    "{}".format()
except Exception as e:
    print(f"Error with no args: {e}")


#=============================================================================
# str.format_map()
#=============================================================================
"""
str.format_map() formats the string using a mapping to replace placeholders.

Syntax: string.format_map(mapping)
Parameters:
    mapping: A mapping (like a dictionary)
Return: A formatted string

Time Complexity: Depends on the complexity of the formatting and mapping access
"""

# Basic usage
template = "Hello, {name}! You are {age} years old."
data = {'name': 'Alice', 'age': 30}
print(f"Using format_map with {data} → '{template.format_map(data)}'")

# Using a dict with missing keys - raises KeyError
template = "Name: {name}, Age: {age}, Country: {country}"
data = {'name': 'Bob', 'age': 25}
try:
    template.format_map(data)
except KeyError as e:
    print(f"Error with missing key: {e}")

# Custom mapping class to handle missing keys
class DefaultDict(dict):
    def __missing__(self, key):
        return f"<{key} missing>"

data = DefaultDict({'name': 'Bob', 'age': 25})
print(f"With custom mapping: '{template.format_map(data)}'")

# Advanced usage with object attributes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

template = "Person: {person.name}, Age: {person.age}"
data = {'person': Person('Charlie', 40)}
print(f"Using object attributes: '{template.format_map(data)}'")

# Error handling - must be a mapping
try:
    "Hello, {name}".format_map(["Alice"])
except TypeError as e:
    print(f"Error when using non-mapping: {e}")


#=============================================================================
# str.index()
#=============================================================================
"""
str.index() returns the lowest index where the substring is found.
Like find(), but raises ValueError when the substring is not found.

Syntax: string.index(substring[, start[, end]])
Parameters:
    substring: The substring to search for
    start (optional): The starting index (default is 0)
    end (optional): The ending index (default is the end of the string)
Return: The lowest index where substring is found
Raises: ValueError if substring is not found

Time Complexity: O(n*m) where n is the length of the string and m is the length of the substring
"""

# Basic usage
text = "hello world"
print(f"'hello world'.index('world') → {text.index('world')}")  # 6

# Substring not found - raises ValueError
try:
    text.index("python")
except ValueError as e:
    print(f"Error when substring not found: {e}")

# With start and end parameters
text = "hello world, hello python"
print(f"Index of 'hello' from start: {text.index('hello')}")  # 0
print(f"Index of 'hello' from index 5: {text.index('hello', 5)}")  # 13
print(f"Index of 'hello' from index 5 to 20: {text.index('hello', 5, 20)}")  # 13

# Out of range start/end
try:
    text.index('hello', 20, 30)
except ValueError as e:
    print(f"Error when substring not found in range: {e}")

# Case sensitivity
try:
    text.index('Hello')  # Case sensitive
except ValueError as e:
    print(f"Error with case sensitivity: {e}")

# Empty substring
print(f"Index of empty string: {text.index('')}")  # 0
print(f"Index of empty string from position 5: {text.index('', 5)}")  # 5

# Comparison with find() - main difference is error handling
pos1 = text.find("python")
print(f"find() result for 'python': {pos1}")  # Returns index

try:
    pos2 = text.index("python")
    print(f"index() result: {pos2}")
except ValueError:
    print("index() raises ValueError when substring not found")


#=============================================================================
# str.isalnum()
#=============================================================================
"""
str.isalnum() returns True if all characters in the string are alphanumeric
(letters or numbers) and there is at least one character.

Syntax: string.isalnum()
Parameters: None
Return: A boolean indicating whether all characters are alphanumeric

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "Python3"
print(f"'Python3'.isalnum() → {text.isalnum()}")  # True

# With spaces
text = "Python 3"
print(f"'Python 3'.isalnum() → {text.isalnum()}")  # False (space is not alphanumeric)

# With special characters
text = "Python3!"
print(f"'Python3!'.isalnum() → {text.isalnum()}")  # False (! is not alphanumeric)

# Unicode characters
text = "Pythön3"
print(f"'Pythön3'.isalnum() → {text.isalnum()}")  # True (ö is a letter)

# Numbers only
text = "12345"
print(f"'12345'.isalnum() → {text.isalnum()}")  # True

# Letters only
text = "Python"
print(f"'Python'.isalnum() → {text.isalnum()}")  # True

# Empty string
text = ""
print(f"Empty string is alphanumeric? {text.isalnum()}")  # False (must have at least one character)


#=============================================================================
# str.isalpha()
#=============================================================================
"""
str.isalpha() returns True if all characters in the string are alphabetic
(letters) and there is at least one character.

Syntax: string.isalpha()
Parameters: None
Return: A boolean indicating whether all characters are alphabetic

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "Python"
print(f"'Python'.isalpha() → {text.isalpha()}")  # True

# With numbers
text = "Python3"
print(f"'Python3'.isalpha() → {text.isalpha()}")  # False (3 is not a letter)

# With spaces
text = "Python Programming"
print(f"'Python Programming'.isalpha() → {text.isalpha()}")  # False (space is not a letter)

# Unicode characters
text = "Pythön"
print(f"'Pythön'.isalpha() → {text.isalpha()}")  # True (ö is a letter)

# Empty string
text = ""
print(f"Empty string is alphabetic? {text.isalpha()}")  # False (must have at least one character)

# Special characters
text = "Python!"
print(f"'Python!'.isalpha() → {text.isalpha()}")  # False (! is not a letter)


#=============================================================================
# str.isascii()
#=============================================================================
"""
str.isascii() returns True if all characters in the string are ASCII characters.
Available in Python 3.7 and later.

Syntax: string.isascii()
Parameters: None
Return: A boolean indicating whether all characters are in the ASCII table

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "Hello123"
print(f"'Hello123'.isascii() → {text.isascii()}")  # True

# With non-ASCII characters
text = "Pythön"
print(f"'Pythön'.isascii() → {text.isascii()}")  # False (ö is not ASCII)

# Emoji
text = "Python 😀"
print(f"'Python 😀'.isascii() → {text.isascii()}")  # False (emoji is not ASCII)

# Special characters within ASCII range
text = "Python!@#$%^&*()"
print(f"'Python!@#$%^&*()'.isascii() → {text.isascii()}")  # True (all are ASCII)

# Empty string
text = ""
print(f"Empty string is ASCII? {text.isascii()}")  # True (trivially true)

# Control characters
text = "Hello\n"
print(f"String with newline is ASCII? {text.isascii()}")  # True (newline is ASCII)


#=============================================================================
# str.isdecimal()
#=============================================================================
"""
str.isdecimal() returns True if all characters in the string are decimal
characters and there is at least one character.

Decimal characters are those that can be used to form numbers in base 10,
specifically the digits 0-9.

Syntax: string.isdecimal()
Parameters: None
Return: A boolean indicating whether all characters are decimal

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "12345"
print(f"'12345'.isdecimal() → {text.isdecimal()}")  # True

# With letters
text = "12345a"
print(f"'12345a'.isdecimal() → {text.isdecimal()}")  # False

# With spaces
text = "12 345"
print(f"'12 345'.isdecimal() → {text.isdecimal()}")  # False

# Unicode digits that look like our digits
text = "１２３４５"  # Full-width digits
print(f"'１２３４５'.isdecimal() → {text.isdecimal()}")  # True

# Other Unicode numerals that aren't decimal
text = "①②③④⑤"  # Circled digits
print(f"'①②③④⑤'.isdecimal() → {text.isdecimal()}")  # False

# Vulgar fractions
text = "½"
print(f"'½'.isdecimal() → {text.isdecimal()}")  # False

# Superscript/subscript digits
text = "²³"
print(f"'²³'.isdecimal() → {text.isdecimal()}")  # False

# Roman numerals
text = "Ⅳ"
print(f"'Ⅳ'.isdecimal() → {text.isdecimal()}")  # False

# Empty string
text = ""
print(f"Empty string is decimal? {text.isdecimal()}")  # False (must have at least one character)

# Comparison with isdigit() and isnumeric()
text = "12345"  # Standard digits
print(f"'12345' - isdecimal: {text.isdecimal()}, isdigit: {text.isdigit()}, isnumeric: {text.isnumeric()}")

text = "²³"  # Superscript
print(f"'²³' - isdecimal: {text.isdecimal()}, isdigit: {text.isdigit()}, isnumeric: {text.isnumeric()}")

text = "½"  # Fraction
print(f"'½' - isdecimal: {text.isdecimal()}, isdigit: {text.isdigit()}, isnumeric: {text.isnumeric()}")


#=============================================================================
# str.isdigit()
#=============================================================================
"""
str.isdigit() returns True if all characters in the string are digits
and there is at least one character.

Digits include decimal characters and characters that need special
handling, like superscript/subscript digits.

Syntax: string.isdigit()
Parameters: None
Return: A boolean indicating whether all characters are digits

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "12345"
print(f"'12345'.isdigit() → {text.isdigit()}")  # True

# With letters
text = "12345a"
print(f"'12345a'.isdigit() → {text.isdigit()}")  # False

# With spaces
text = "12 345"
print(f"'12 345'.isdigit() → {text.isdigit()}")  # False

# Unicode digits that look like our digits
text = "１２３４５"  # Full-width digits
print(f"'１２３４５'.isdigit() → {text.isdigit()}")  # True

# Superscript/subscript digits
text = "²³"
print(f"'²³'.isdigit() → {text.isdigit()}")  # True

# Other Unicode numerals
text = "①②③④⑤"  # Circled digits
print(f"'①②③④⑤'.isdigit() → {text.isdigit()}")  # False

# Vulgar fractions
text = "½"
print(f"'½'.isdigit() → {text.isdigit()}")  # False

# Empty string
text = ""
print(f"Empty string is digit? {text.isdigit()}")  # False (must have at least one character)

# Comparison with isdecimal() and isnumeric() for different types of numbers
text = "12345"  # Standard digits
print(f"'12345' - isdecimal: {text.isdecimal()}, isdigit: {text.isdigit()}, isnumeric: {text.isnumeric()}")
# All True - standard digits are recognized by all three methods

text = "²³"  # Superscript
print(f"'²³' - isdecimal: {text.isdecimal()}, isdigit: {text.isdigit()}, isnumeric: {text.isnumeric()}")
# False, True, True - superscripts are digits and numeric but not decimal


#=============================================================================
# str.isidentifier()
#=============================================================================
"""
str.isidentifier() returns True if the string is a valid Python identifier.

A valid identifier:
- Can contain letters, numbers, and underscores
- Cannot start with a number
- Cannot be a Python keyword

Syntax: string.isidentifier()
Parameters: None
Return: A boolean indicating whether the string is a valid Python identifier

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "variable_name"
print(f"'variable_name'.isidentifier() → {text.isidentifier()}")  # True

# Starting with number
text = "1variable"
print(f"'1variable'.isidentifier() → {text.isidentifier()}")  # False

# With special characters
text = "variable-name"
print(f"'variable-name'.isidentifier() → {text.isidentifier()}")  # False

# With spaces
text = "variable name"
print(f"'variable name'.isidentifier() → {text.isidentifier()}")  # False

# Unicode characters
text = "variableñ"
print(f"'variableñ'.isidentifier() → {text.isidentifier()}")  # True (ñ is a letter)

# Starting with underscore
text = "_variable"
print(f"'_variable'.isidentifier() → {text.isidentifier()}")  # True

# Python keywords
import keyword
text = "if"
print(f"'if'.isidentifier() → {text.isidentifier()}")  # True (grammatically valid but reserved)
print(f"'if' is a Python keyword? {keyword.iskeyword(text)}")  # True

# Empty string
text = ""
print(f"Empty string is valid identifier? {text.isidentifier()}")  # False


#=============================================================================
# str.islower()
#=============================================================================
"""
str.islower() returns True if all cased characters in the string are lowercase
and there is at least one cased character.

Syntax: string.islower()
Parameters: None
Return: A boolean indicating whether all cased characters are lowercase

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "hello world"
print(f"'hello world'.islower() → {text.islower()}")  # True

# With some uppercase
text = "Hello world"
print(f"'Hello world'.islower() → {text.islower()}")  # False

# With digits
text = "hello123"
print(f"'hello123'.islower() → {text.islower()}")  # True (digits are not cased)

# With special characters
text = "hello!"
print(f"'hello!'.islower() → {text.islower()}")  # True (special chars are not cased)

# All uppercase
text = "HELLO"
print(f"'HELLO'.islower() → {text.islower()}")  # False

# Empty string
text = ""
print(f"Empty string is lowercase? {text.islower()}")  # False (no cased characters)

# Only special characters/digits
text = "123!@#"
print(f"'123!@#'.islower() → {text.islower()}")  # False (no cased characters)

# Unicode characters
text = "héllö"
print(f"'héllö'.islower() → {text.islower()}")  # True


#=============================================================================
# str.isnumeric()
#=============================================================================
"""
str.isnumeric() returns True if all characters in the string are numeric
and there is at least one character.

Numeric characters include digits, vulgar fractions, subscripts/superscripts,
roman numerals, and other characters that represent numbers.

Syntax: string.isnumeric()
Parameters: None
Return: A boolean indicating whether all characters are numeric

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "12345"
print(f"'12345'.isnumeric() → {text.isnumeric()}")  # True

# With letters
text = "12345a"
print(f"'12345a'.isnumeric() → {text.isnumeric()}")  # False

# With spaces
text = "12 345"
print(f"'12 345'.isnumeric() → {text.isnumeric()}")  # False

# Unicode digits
text = "１２３４５"  # Full-width digits
print(f"'１２３４５'.isnumeric() → {text.isnumeric()}")  # True

# Superscript/subscript
text = "²³"
print(f"'²³'.isnumeric() → {text.isnumeric()}")  # True

# Vulgar fractions
text = "½"
print(f"'½'.isnumeric() → {text.isnumeric()}")  # True

# Roman numerals
text = "Ⅳ"
print(f"'Ⅳ'.isnumeric() → {text.isnumeric()}")  # True

# Other numeric characters
text = "①②③④⑤"  # Circled digits
print(f"'①②③④⑤'.isnumeric() → {text.isnumeric()}")  # True

# Empty string
text = ""
print(f"Empty string is numeric? {text.isnumeric()}")  # False (must have at least one character)

# Comparing all three methods with different types of numeric characters
print("\nComparison of isdecimal, isdigit, and isnumeric:")
examples = [
    "12345",     # Standard digits
    "²³",        # Superscript
    "½",         # Fraction
    "Ⅳ",         # Roman numeral
    "①②③",       # Circled digits
]

for ex in examples:
    print(f"'{ex}' - isdecimal: {ex.isdecimal()}, isdigit: {ex.isdigit()}, isnumeric: {ex.isnumeric()}")


#=============================================================================
# str.isprintable()
#=============================================================================
"""
str.isprintable() returns True if all characters in the string are printable
or if the string is empty.

Printable characters are those which are not escape sequences like '\n', '\t'.

Syntax: string.isprintable()
Parameters: None
Return: A boolean indicating whether all characters are printable

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "Hello, World!"
print(f"'Hello, World!'.isprintable() → {text.isprintable()}")  # True

# With non-printable characters
text = "Hello\nWorld"  # Contains newline
print(f"String with newline {repr(text)}: {text.isprintable()}")  # False

# With tab
text = "Hello\tWorld"  # Contains tab
print(f"String with tab {repr(text)}: {text.isprintable()}")  # False

# With carriage return
text = "Hello\rWorld"  # Contains carriage return
print(f"String with carriage return {repr(text)}: {text.isprintable()}")  # False

# With form feed
text = "Hello\fWorld"  # Contains form feed
print(f"String with form feed {repr(text)}: {text.isprintable()}")  # False

# With Unicode printable characters
text = "Hello, 你好!"
print(f"String with Unicode {repr(text)}: {text.isprintable()}")  # True

# Empty string
text = ""
print(f"Empty string is printable? {text.isprintable()}")  # True (trivially true)

# Spaces
text = "   "
print(f"String with spaces {repr(text)}: {text.isprintable()}")  # True


#=============================================================================
# str.isspace()
#=============================================================================
"""
str.isspace() returns True if all characters in the string are whitespace
and there is at least one character.

Whitespace characters include space, tab, newline, etc.

Syntax: string.isspace()
Parameters: None
Return: A boolean indicating whether all characters are whitespace

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "   "  # Spaces
print(f"'   '.isspace() → {text.isspace()}")  # True

# Mixed with non-whitespace
text = "   a   "
print(f"'   a   '.isspace() → {text.isspace()}")  # False

# Different whitespace characters
text = " \t\n\r\f\v"  # Space, tab, newline, carriage return, form feed, vertical tab
print(f"{repr(text)}.isspace() → {text.isspace()}")  # True

# Empty string
text = ""
print(f"Empty string is space? {text.isspace()}")  # False (must have at least one character)

# Unicode whitespace
text = "\u2000\u2001"  # En Quad and Em Quad (Unicode whitespace)
print(f"{repr(text)}.isspace() → {text.isspace()}")  # True

# Visible separator
text = "_"  # Underscore
print(f"'_'.isspace() → {text.isspace()}")  # False (underscore is not whitespace)


#=============================================================================
# str.istitle()
#=============================================================================
"""
str.istitle() returns True if the string is titlecased and there is at least
one character.

A titlecased string has all words starting with an uppercase character, with
remaining characters being lowercase.

Syntax: string.istitle()
Parameters: None
Return: A boolean indicating whether the string is titlecased

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "Hello World"
print(f"'Hello World'.istitle() → {text.istitle()}")  # True

# Not titlecased
text = "Hello world"
print(f"'Hello world'.istitle() → {text.istitle()}")  # False

# All uppercase is not titlecased
text = "HELLO WORLD"
print(f"'HELLO WORLD'.istitle() → {text.istitle()}")  # False

# All lowercase is not titlecased
text = "hello world"
print(f"'hello world'.istitle() → {text.istitle()}")  # False

# With non-alphabetic characters
text = "Hello, World! 123"
print(f"'Hello, World! 123'.istitle() → {text.istitle()}")  # True

# Apostrophes in contractions
text = "It's A Nice Day"
print(f"'It's A Nice Day'.istitle() → {text.istitle()}")  # True

text = "I'm Fine"
print(f"'I'm Fine'.istitle() → {text.istitle()}")  # True

# Hyphenated words
text = "First-Class Package"
print(f"'First-Class Package'.istitle() → {text.istitle()}")  # True

# Empty string
text = ""
print(f"Empty string is title? {text.istitle()}")  # False (must have at least one character)

# Unicode characters
text = "Héllö Wörld"
print(f"'Héllö Wörld'.istitle() → {text.istitle()}")  # True


#=============================================================================
# str.isupper()
#=============================================================================
"""
str.isupper() returns True if all cased characters in the string are uppercase
and there is at least one cased character.

Syntax: string.isupper()
Parameters: None
Return: A boolean indicating whether all cased characters are uppercase

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "HELLO WORLD"
print(f"'HELLO WORLD'.isupper() → {text.isupper()}")  # True

# With some lowercase
text = "HELLO World"
print(f"'HELLO World'.isupper() → {text.isupper()}")  # False

# With digits
text = "HELLO123"
print(f"'HELLO123'.isupper() → {text.isupper()}")  # True (digits are not cased)

# With special characters
text = "HELLO!"
print(f"'HELLO!'.isupper() → {text.isupper()}")  # True (special chars are not cased)

# All lowercase
text = "hello"
print(f"'hello'.isupper() → {text.isupper()}")  # False

# Empty string
text = ""
print(f"Empty string is uppercase? {text.isupper()}")  # False (no cased characters)

# Only special characters/digits
text = "123!@#"
print(f"'123!@#'.isupper() → {text.isupper()}")  # False (no cased characters)

# Unicode characters
text = "HÉLLÖ"
print(f"'HÉLLÖ'.isupper() → {text.isupper()}")  # True


#=============================================================================
# str.join()
#=============================================================================
"""
str.join() returns a string which is the concatenation of the strings in an
iterable, with the original string as a separator.

Syntax: string.join(iterable)
Parameters:
    iterable: An iterable of strings to join
Return: A concatenated string

Time Complexity: O(n) where n is the total length of the resulting string
"""

# Basic usage
separator = ", "
iterable = ["apple", "banana", "cherry"]
print(f"', '.join(['apple', 'banana', 'cherry']) → '{separator.join(iterable)}'")  # 'apple, banana, cherry'

# Using different separators
print(f"Join with empty string: {('').join(iterable)}")  # 'applebananacherry'
# print(f"Join with newline: ,{('\\n').join(iterable)}")
print(f"Join with hyphen: {('-').join(iterable)}")  # 'apple-banana-cherry'

# Joining characters of a string
text = "Python"
print(f"Join characters with pipe: '{('|').join(text)}'")  # 'P|y|t|h|o|n'

# Joining mixed types (error)
mixed = ["apple", 123, "cherry"]
try:
    ", ".join(mixed)
except TypeError as e:
    print(f"Error when joining mixed types: {e}")

# Converting non-string items to strings
mixed_str = [str(item) for item in mixed]
print(f"Join after converting to strings: '{(', ').join(mixed_str)}'")  # 'apple, 123, cherry'

# Joining with dictionary keys
d = {"a": 1, "b": 2, "c": 3}
print(f"Join dictionary keys: '{('-').join(d)}'")  # 'a-b-c'

# Joining with dictionary values (error - must convert to strings)
try:
    "-".join(d.values())
except TypeError as e:
    print(f"Error when joining dict values: {e}")

# Joining with dictionary values (converted to strings)
print(f"Join dictionary values as strings: '{('-').join(str(v) for v in d.values())}'")  # '1-2-3'

# Joining an empty iterable
empty = []
print(f"Join empty iterable: '{('-').join(empty)}'")  # ''

# Joining a single item
single = ["apple"]
print(f"Join single item: '{('-').join(single)}'")  # 'apple'


#=============================================================================
# str.ljust()
#=============================================================================
"""
str.ljust() returns a left-justified string of specified width.

Syntax: string.ljust(width[, fillchar])
Parameters:
    width: The total width of the resulting string
    fillchar (optional): The character to fill the extra space (default is space)
Return: A left-justified string padded with specified character

Time Complexity: O(n) where n is the width
"""

# Basic usage
text = "Python"
print(f"'Python'.ljust(10) → '{text.ljust(10)}'")  # 'Python    '

# Using a different fill character
print(f"'Python'.ljust(10, '*') → '{text.ljust(10, '*')}'")  # 'Python****'

# If width is less than or equal to the string length, the original string is returned
print(f"'Python'.ljust(6) → '{text.ljust(6)}'")  # 'Python'
print(f"'Python'.ljust(4) → '{text.ljust(4)}'")  # 'Python'

# Empty string handling
empty = ""
print(f"Empty string ljust(5, '*') → '{empty.ljust(5, '*')}'")  # '*****'

# fillchar must be exactly one character
try:
    text.ljust(10, "**")
except TypeError as e:
    print(f"Error when using multiple characters as fillchar: {e}")


#=============================================================================
# str.lower()
#=============================================================================
"""
str.lower() returns a lowercase version of the string.

Syntax: string.lower()
Parameters: None
Return: A lowercase copy of the string

Time Complexity: O(n) where n is the length of the string
"""

# Basic usage
text = "Hello World"
print(f"'Hello World'.lower() → '{text.lower()}'")  # 'hello world'

# Already lowercase
text = "hello world"
print(f"Already lowercase: '{text.lower()}'")  # 'hello world'

# All uppercase
text = "HELLO WORLD"
print(f"All uppercase: '{text.lower()}'")  # 'hello world'