#!/usr/bin/env python3
"""
Python Built-in Functions Tutorial
----------------------------------
A comprehensive guide to the following Python built-in functions:
- format
- ascii
- chr
- ord
- bin
- hex
- oct

This file contains detailed explanations, examples, edge cases, and
exception handling for each function to help you master them and
become an advanced Python programmer.
"""

##############################################################################
#                              FORMAT FUNCTION                               #
##############################################################################
"""
format() - Returns a formatted representation of a value controlled by a format specifier.

Syntax:
    format(value[, format_spec])

Parameters:
    value       - The value to format
    format_spec - The specification for how the value should be formatted (optional)

Returns:
    A formatted string representation of the value
"""

def format_function_examples():
    """Examples demonstrating the format() built-in function."""
    
    print("=" * 30)
    print("FORMAT FUNCTION EXAMPLES")
    print("=" * 30)
    
    # Basic usage - default formatting
    print(f"Basic formatting of integer: {format(1234)}")
    
    # Number formatting with width and alignment
    # > means right align, < means left align, ^ means center
    print(f"Right align with width 10: '{format(1234, '>10')}'")
    print(f"Left align with width 10: '{format(1234, '<10')}'")
    print(f"Center align with width 10: '{format(1234, '^10')}'")
    
    # Fill character with alignment
    print(f"Right align with '-' fill: '{format(1234, '-<10')}'")
    print(f"Center align with '*' fill: '{format(1234, '*^10')}'")
    
    # Integer formats: binary, octal, hex
    print(f"Binary format: {format(42, 'b')}")
    print(f"Octal format: {format(42, 'o')}")
    print(f"Hexadecimal format (lowercase): {format(42, 'x')}")
    print(f"Hexadecimal format (uppercase): {format(42, 'X')}")
    
    # Numeric formatting - precision
    print(f"Float with 2 decimal places: {format(3.14159, '.2f')}")
    print(f"Float with 4 decimal places: {format(3.14159, '.4f')}")
    
    # Scientific notation
    print(f"Scientific notation (lowercase): {format(1234.5678, 'e')}")
    print(f"Scientific notation (uppercase): {format(1234.5678, 'E')}")
    print(f"Scientific with 2 decimal places: {format(1234.5678, '.2e')}")
    
    # Percentage
    print(f"Percentage format: {format(0.23456, '.2%')}")
    
    # Combining width, alignment, and precision
    print(f"Combined (width 12, 2 decimals, center): '{format(3.14159, '^12.2f')}'")
    
    # Grouping with commas
    print(f"With thousand separators: {format(1234567, ',d')}")
    print(f"Float with thousand separators: {format(1234567.89, ',.2f')}")
    
    # Sign control: + forces sign on positive numbers
    print(f"Always showing sign: {format(42, '+d')} and {format(-42, '+d')}")
    
    # Space for sign: space reserves space for sign (positive gets space, negative gets -)
    print(f"Space for sign: '{format(42, ' d')}' and '{format(-42, ' d')}'")
    
    # Using format() with strings
    print(f"String with width 10: '{format('hello', '10')}'")
    print(f"String right aligned: '{format('hello', '>10')}'")
    print(f"String with custom fill: '{format('hello', '_^10')}'")
    
    # FORMAT EXCEPTION CASES
    
    try:
        # Invalid format specification
        print(format(123, "invalid"))
    except ValueError as e:
        print(f"Exception with invalid format spec: {e}")
    
    try:
        # Using string format spec with non-string
        print(format(123, "s"))
    except ValueError as e:
        print(f"Exception using string format with number: {e}")

    # Type-specific formatting
    # Date formatting requires using the __format__ method of the object
    import datetime
    now = datetime.datetime.now()
    print(f"Date formatting: {format(now, '%Y-%m-%d %H:%M:%S')}")
    
    # Custom objects can implement __format__ method
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
        def __format__(self, format_spec):
            if not format_spec:
                return f"Point({self.x}, {self.y})"
            elif format_spec == 'c':  # Custom format for coordinates
                return f"({self.x}, {self.y})"
            else:
                return f"Point x={format(self.x, format_spec)}, y={format(self.y, format_spec)}"
    
    point = Point(3.5, 2.8)
    print(f"Custom object default format: {format(point)}")
    print(f"Custom object with format 'c': {format(point, 'c')}")
    print(f"Custom object with .1f format: {format(point, '.1f')}")


##############################################################################
#                              ASCII FUNCTION                                #
##############################################################################
# """
# ascii() - Returns a string containing a printable representation of an object,
#           escaping non-ASCII characters using \x, \u or \U escapes.

# Syntax:
#     ascii(object)

# Parameters:
#     object - Any Python object

# Returns:
#     A string with escaped non-ASCII characters
# """

def ascii_function_examples():
    """Examples demonstrating the ascii() built-in function."""
    
    print("\n" + "=" * 30)
    print("ASCII FUNCTION EXAMPLES")
    print("=" * 30)
    
    # Basic usage with ASCII characters
    ascii_string = "Hello, World!"
    print(f"ASCII on ASCII string: {ascii(ascii_string)}")
    
    # With non-ASCII characters
    non_ascii_string = "Hello, 世界!"  # 世界 means "world" in Chinese
    print(f"Original string: {non_ascii_string}")
    print(f"ASCII result: {ascii(non_ascii_string)}")
    
    # With emoji
    emoji_string = "Python is 🐍 awesome!"
    print(f"Original emoji string: {emoji_string}")
    print(f"ASCII result: {ascii(emoji_string)}")
    
    # With various special characters
    special_chars = "€£¥©®™½¼¾"
    print(f"Original special chars: {special_chars}")
    print(f"ASCII result: {ascii(special_chars)}")
    
    # With control characters
    control_chars = "Hello\nWorld\tTab\rReturn"
    print(f"ASCII with control chars: {ascii(control_chars)}")
    
    # With non-string objects
    print(f"ASCII on an integer: {ascii(42)}")
    print(f"ASCII on a float: {ascii(3.14159)}")
    print(f"ASCII on a list with non-ASCII: {ascii(['hello', '世界'])}")
    print(f"ASCII on a dict with non-ASCII: {ascii({'greeting': 'Olá'})}")
    
    # With custom objects
    class CustomObject:
        def __repr__(self):
            return "CustomObject(你好)"
    
    custom_obj = CustomObject()
    print(f"ASCII on custom object: {ascii(custom_obj)}")
    
    # Differences between ascii(), repr(), and str()
    test_string = "こんにちは"  # "Hello" in Japanese
    print(f"Original: {test_string}")
    print(f"ascii(): {ascii(test_string)}")
    print(f"repr(): {repr(test_string)}")
    print(f"str(): {str(test_string)}")


##############################################################################
#                               CHR FUNCTION                                 #
##############################################################################
"""
chr() - Returns a string representing a character whose Unicode code point is
        the integer passed as argument.

Syntax:
    chr(i)

Parameters:
    i - An integer representing a valid Unicode code point

Returns:
    A string representing the character with the given Unicode code point

Valid range: 0 to 0x10FFFF (0 to 1,114,111)
"""

def chr_function_examples():
    """Examples demonstrating the chr() built-in function."""
    
    print("\n" + "=" * 30)
    print("CHR FUNCTION EXAMPLES")
    print("=" * 30)
    
    # Basic ASCII characters
    print(f"chr(65) -> {chr(65)}  # Uppercase 'A'")
    print(f"chr(97) -> {chr(97)}  # Lowercase 'a'")
    print(f"chr(48) -> {chr(48)}  # Digit '0'")
    
    # Control characters
    print(f"chr(9) -> '\\t'  # Tab character")
    print(f"chr(10) -> '\\n'  # Newline character")
    print(f"chr(13) -> '\\r'  # Carriage return")
    print(f"chr(32) -> ' '  # Space character")
    
    # Extended ASCII (Latin-1) characters
    print(f"chr(169) -> {chr(169)}  # Copyright symbol ©")
    print(f"chr(174) -> {chr(174)}  # Registered trademark ®")
    print(f"chr(176) -> {chr(176)}  # Degree symbol °")
    
    # Unicode characters beyond ASCII
    print(f"chr(8364) -> {chr(8364)}  # Euro symbol €")
    print(f"chr(9731) -> {chr(9731)}  # Snowman ☃")
    print(f"chr(9829) -> {chr(9829)}  # Heart symbol ♥")
    
    # Emoji characters (high code points)
    print(f"chr(128512) -> {chr(128512)}  # Grinning face 😀")
    print(f"chr(128013) -> {chr(128013)}  # Snake 🐍")
    
    # Range examples
    print("\nRange of digits:")
    for i in range(48, 58):
        print(f"chr({i}) -> {chr(i)}")
    
    print("\nRange of uppercase letters:")
    for i in range(65, 71):  # Just showing A-F for brevity
        print(f"chr({i}) -> {chr(i)}")
    
    # Boundary cases
    print(f"\nMinimum valid code point: chr(0) -> '\\x00' (null character)")
    print(f"Maximum valid code point: chr(1114111) -> '{chr(1114111)}'")
    
    # CHR EXCEPTION CASES
    
    print("\nException cases:")
    try:
        # Negative code point
        print(chr(-1))
    except ValueError as e:
        print(f"chr(-1) raises: {e}")
    
    try:
        # Code point too large
        print(chr(1114112))  # 0x110000, just beyond the max
    except ValueError as e:
        print(f"chr(1114112) raises: {e}")
    
    try:
        # Non-integer argument
        print(chr(65.5))
    except TypeError as e:
        print(f"chr(65.5) raises: {e}")
    
    # Practical applications
    
    # 1. Creating a string from Unicode code points
    code_points = [80, 121, 116, 104, 111, 110]
    string_from_codes = ''.join(chr(cp) for cp in code_points)
    print(f"\nString from code points {code_points}: {string_from_codes}")
    
    # 2. Generate a full alphabet
    alphabet = ''.join(chr(i) for i in range(97, 123))
    print(f"Lowercase alphabet: {alphabet}")
    
    # 3. Working with Unicode blocks
    print("\nSome mathematical symbols:")
    for i in range(8704, 8710):
        print(f"chr({i}) -> {chr(i)}")


##############################################################################
#                               ORD FUNCTION                                 #
##############################################################################
"""
ord() - Returns the Unicode code point for a given character.
        This is the inverse of chr().

Syntax:
    ord(c)

Parameters:
    c - A string of length 1, representing a Unicode character

Returns:
    An integer representing the Unicode code point of the character
"""

def ord_function_examples():
    """Examples demonstrating the ord() built-in function."""
    
    print("\n" + "=" * 30)
    print("ORD FUNCTION EXAMPLES")
    print("=" * 30)
    
    # Basic ASCII characters
    print(f"ord('A') -> {ord('A')}  # Uppercase 'A'")
    print(f"ord('a') -> {ord('a')}  # Lowercase 'a'")
    print(f"ord('0') -> {ord('0')}  # Digit '0'")
    
    # Control characters
    # print(f"ord('\\t') -> {ord('\t')}  # Tab character")
    # print(f"ord('\\n') -> {ord('\n')}  # Newline character")
    # print(f"ord(' ') -> {ord(' ')}  # Space character")
    
    # Extended ASCII (Latin-1) characters
    print(f"ord('©') -> {ord('©')}  # Copyright symbol")
    print(f"ord('®') -> {ord('®')}  # Registered trademark")
    print(f"ord('°') -> {ord('°')}  # Degree symbol")
    
    # Unicode characters beyond ASCII
    print(f"ord('€') -> {ord('€')}  # Euro symbol")
    print(f"ord('☃') -> {ord('☃')}  # Snowman")
    print(f"ord('♥') -> {ord('♥')}  # Heart symbol")
    
    # Emoji (high code points)
    print(f"ord('😀') -> {ord('😀')}  # Grinning face")
    print(f"ord('🐍') -> {ord('🐍')}  # Snake")
    
    # Non-English letters
    print(f"ord('é') -> {ord('é')}  # e with acute accent")
    print(f"ord('ñ') -> {ord('ñ')}  # n with tilde")
    print(f"ord('ü') -> {ord('ü')}  # u with umlaut")
    print(f"ord('あ') -> {ord('あ')}  # Japanese Hiragana 'a'")
    print(f"ord('中') -> {ord('中')}  # Chinese character")
    
    # Verify the inverse relationship with chr()
    char = 'X'
    code_point = ord(char)
    original_char = chr(code_point)
    print(f"\nInverse relationship: ord('{char}') -> {code_point} -> chr({code_point}) -> '{original_char}'")
    
    # Full circle with non-ASCII character
    unicode_char = '🌟'
    code_point = ord(unicode_char)
    back_to_char = chr(code_point)
    print(f"With emoji: ord('{unicode_char}') -> {code_point} -> chr({code_point}) -> '{back_to_char}'")
    
    # ORD EXCEPTION CASES
    
    print("\nException cases:")
    try:
        # Empty string
        print(ord(''))
    except TypeError as e:
        print(f"ord('') raises: {e}")
    
    try:
        # String with multiple characters
        print(ord('ABC'))
    except TypeError as e:
        print(f"ord('ABC') raises: {e}")
    
    try:
        # Non-string argument
        print(ord(65))
    except TypeError as e:
        print(f"ord(65) raises: {e}")
    
    # Practical applications
    
    # 1. Converting a string to a list of code points
    text = "Python"
    code_points = [ord(c) for c in text]
    print(f"\nCode points for '{text}': {code_points}")
    
    # 2. Case conversion without using str methods
    # (for educational purposes - in practice, use str.upper() or str.lower())
    lowercase_a_to_z = range(ord('a'), ord('z')+1)
    char = 'h'
    if ord(char) in lowercase_a_to_z:
        # Convert to uppercase by shifting -32 in ASCII
        uppercase_char = chr(ord(char) - 32)
        print(f"Manual conversion: '{char}' -> '{uppercase_char}'")
    
    # 3. Simple character manipulation for ROT13 cipher
    def rot13_char(c):
        """Apply ROT13 cipher to a character."""
        if 'a' <= c <= 'z':
            # For lowercase: shift 13 places and wrap
            return chr((ord(c) - ord('a') + 13) % 26 + ord('a'))
        elif 'A' <= c <= 'Z':
            # For uppercase: shift 13 places and wrap
            return chr((ord(c) - ord('A') + 13) % 26 + ord('A'))
        else:
            # Non-alpha characters stay the same
            return c
    
    text = "Hello, World!"
    rot13_text = ''.join(rot13_char(c) for c in text)
    print(f"\nROT13 cipher example:")
    print(f"Original: {text}")
    print(f"ROT13: {rot13_text}")
    
    # 4. Check if a character is within a specific Unicode block
    def is_in_unicode_block(char, start, end):
        """Check if a character falls within a specific Unicode block."""
        code_point = ord(char)
        return start <= code_point <= end
    
    # CJK Unified Ideographs block: U+4E00 to U+9FFF
    cjk_examples = ['A', '你', '한', 'こ']
    for char in cjk_examples:
        is_cjk = is_in_unicode_block(char, 0x4E00, 0x9FFF)
        print(f"Is '{char}' a CJK character? {is_cjk}")


##############################################################################
#                               BIN FUNCTION                                 #
##############################################################################
"""
bin() - Converts an integer to a binary string prefixed with "0b".

Syntax:
    bin(x)

Parameters:
    x - An integer object

Returns:
    A string with the binary representation of x, prefixed with "0b"
"""

def bin_function_examples():
    """Examples demonstrating the bin() built-in function."""
    
    print("\n" + "=" * 30)
    print("BIN FUNCTION EXAMPLES")
    print("=" * 30)
    
    # Basic usage with positive integers
    print(f"bin(0) -> {bin(0)}")
    print(f"bin(1) -> {bin(1)}")
    print(f"bin(2) -> {bin(2)}")
    print(f"bin(10) -> {bin(10)}")
    print(f"bin(42) -> {bin(42)}")
    print(f"bin(255) -> {bin(255)}")
    print(f"bin(1024) -> {bin(1024)}")
    
    # Negative integers
    # Two's complement representation is used
    print(f"\nNegative numbers:")
    print(f"bin(-1) -> {bin(-1)}")
    print(f"bin(-10) -> {bin(-10)}")
    print(f"bin(-42) -> {bin(-42)}")
    
    # Understanding the representation of negative numbers:
    print("\nUnderstanding negative binary representation:")
    x = 10
    print(f"bin({x}) -> {bin(x)}")
    print(f"bin({-x}) -> {bin(-x)}")
    
    # This is because Python uses 2's complement for negative numbers,
    # but the output shows only the digits (with sign)
    
    # Objects with __index__ method
    class BinCompatible:
        def __init__(self, value):
            self.value = value
        
        def __index__(self):
            return self.value
    
    obj = BinCompatible(42)
    print(f"\nCustom object with __index__:")
    print(f"bin(BinCompatible(42)) -> {bin(obj)}")
    
    # Removing the "0b" prefix
    value = 127
    bin_with_prefix = bin(value)
    bin_without_prefix = bin(value)[2:]  # Remove first two characters
    print(f"\nRemoving '0b' prefix:")
    print(f"With prefix: {bin_with_prefix}")
    print(f"Without prefix: {bin_without_prefix}")
    
    # Ensuring specific width (padding with zeros)
    value = 7  # binary: 111
    padded_binary = bin(value)[2:].zfill(8)  # Pad to 8 bits
    print(f"\nPadding with zeros to specific width:")
    print(f"bin({value}) -> {bin(value)}")
    print(f"Padded to 8 bits: {padded_binary}")
    
    # Converting between binary strings and integers
    binary_string = "101010"  # Without 0b prefix
    value = int(binary_string, 2)
    print(f"\nConversion between binary string and int:")
    print(f"Binary string: {binary_string}")
    print(f"Converted to int: {value}")
    print(f"Back to binary: {bin(value)}")
    
    # BIN EXCEPTION CASES
    
    print("\nException cases:")
    try:
        # Non-integer value
        print(bin(3.14))
    except TypeError as e:
        print(f"bin(3.14) raises: {e}")
    
    try:
        # String value without __index__
        print(bin("101010"))
    except TypeError as e:
        print(f"bin('101010') raises: {e}")
    
    # Practical applications
    
    # 1. Binary digit counting
    value = 42  # binary: 101010
    binary_rep = bin(value)[2:]
    num_digits = len(binary_rep)
    num_ones = binary_rep.count('1')
    print(f"\nBinary digit analysis for {value} ({bin(value)}):")
    print(f"Number of binary digits: {num_digits}")
    print(f"Number of '1' bits: {num_ones}")
    
    # 2. Bit manipulation examples
    a = 42      # 101010 in binary
    b = 21      # 010101 in binary
    
    # Bitwise operations
    print(f"\nBit manipulation examples:")
    print(f"{a} ({bin(a)}) & {b} ({bin(b)}) = {a & b} ({bin(a & b)})  # AND")
    print(f"{a} ({bin(a)}) | {b} ({bin(b)}) = {a | b} ({bin(a | b)})  # OR")
    print(f"{a} ({bin(a)}) ^ {b} ({bin(b)}) = {a ^ b} ({bin(a ^ b)})  # XOR")
    print(f"~{a} ({bin(a)}) = {~a} ({bin(~a)})  # NOT")
    print(f"{a} ({bin(a)}) << 1 = {a << 1} ({bin(a << 1)})  # Left shift")
    print(f"{a} ({bin(a)}) >> 1 = {a >> 1} ({bin(a >> 1)})  # Right shift")
    
    # 3. Check if a number is a power of 2
    # A power of 2 in binary has only one '1' bit
    def is_power_of_two(n):
        """Check if a number is a power of 2 using binary representation."""
        if n <= 0:
            return False
        # Count the number of '1' bits
        return bin(n).count('1') == 1
    
    test_values = [1, 2, 3, 4, 8, 10, 16, 32, 100]
    print("\nChecking for powers of 2:")
    for val in test_values:
        print(f"{val} ({bin(val)}) is power of 2: {is_power_of_two(val)}")
    
    # 4. Binary to Gray code conversion
    # Gray code has property that adjacent numbers differ by only one bit
    def binary_to_gray(n):
        """Convert a number to its Gray code equivalent."""
        return n ^ (n >> 1)
    
    print("\nBinary to Gray code conversion:")
    for i in range(8):
        gray = binary_to_gray(i)
        print(f"{i} (bin: {bin(i)}) -> Gray: {gray} (bin: {bin(gray)})")


##############################################################################
#                               HEX FUNCTION                                 #
##############################################################################
"""
hex() - Converts an integer to a hexadecimal string prefixed with "0x".

Syntax:
    hex(x)

Parameters:
    x - An integer object

Returns:
    A string with the hexadecimal representation of x, prefixed with "0x"
"""

def hex_function_examples():
    """Examples demonstrating the hex() built-in function."""
    
    print("\n" + "=" * 30)
    print("HEX FUNCTION EXAMPLES")
    print("=" * 30)
    
    # Basic usage with positive integers
    print(f"hex(0) -> {hex(0)}")
    print(f"hex(1) -> {hex(1)}")
    print(f"hex(10) -> {hex(10)}")
    print(f"hex(15) -> {hex(15)}")
    print(f"hex(16) -> {hex(16)}")
    print(f"hex(255) -> {hex(255)}")
    print(f"hex(4096) -> {hex(4096)}")
    
    # Negative integers
    print(f"\nNegative numbers:")
    print(f"hex(-1) -> {hex(-1)}")
    print(f"hex(-10) -> {hex(-10)}")
    print(f"hex(-255) -> {hex(-255)}")
    
    # Objects with __index__ method
    class HexCompatible:
        def __init__(self, value):
            self.value = value
        
        def __index__(self):
            return self.value
    
    obj = HexCompatible(255)
    print(f"\nCustom object with __index__:")
    print(f"hex(HexCompatible(255)) -> {hex(obj)}")
    
    # Removing the "0x" prefix
    value = 255
    hex_with_prefix = hex(value)
    hex_without_prefix = hex(value)[2:]  # Remove first two characters
    print(f"\nRemoving '0x' prefix:")
    print(f"With prefix: {hex_with_prefix}")
    print(f"Without prefix: {hex_without_prefix}")
    
    # Ensuring specific width (padding with zeros)
    value = 15  # hex: f
    padded_hex = hex(value)[2:].zfill(4)  # Pad to 4 hex digits
    print(f"\nPadding with zeros to specific width:")
    print(f"hex({value}) -> {hex(value)}")
    print(f"Padded to 4 hex digits: {padded_hex}")
    
    # Converting between hex strings and integers
    hex_string = "1a3f"  # Without 0x prefix
    value = int(hex_string, 16)
    print(f"\nConversion between hex string and int:")
    print(f"Hex string: {hex_string}")
    print(f"Converted to int: {value}")
    print(f"Back to hex: {hex(value)}")
    
    # Upper and lowercase hex representation
    value = 171
    lower_hex = hex(value)
    upper_hex = "0x" + hex(value)[2:].upper()
    print(f"\nUpper and lowercase representations:")
    print(f"Default (lowercase): {lower_hex}")
    print(f"Uppercase: {upper_hex}")
    
    # HEX EXCEPTION CASES
    
    print("\nException cases:")
    try:
        # Non-integer value
        print(hex(3.14))
    except TypeError as e:
        print(f"hex(3.14) raises: {e}")
    
    try:
        # String value without __index__
        print(hex("abc"))
    except TypeError as e:
        print(f"hex('abc') raises: {e}")
    
    # Practical applications
    
    # 1. Color representation in web development
    red, green, blue = 255, 128, 64
    
    # Generate hex color code
    color_hex = f"#{red:02x}{green:02x}{blue:02x}"
    print(f"\nRGB to Hex color conversion:")
    print(f"RGB({red}, {green}, {blue}) -> Hex: {color_hex}")
    
    # Parse hex color code back to RGB
    color_code = "1a2b3c"
    r = int(color_code[0:2], 16)
    g = int(color_code[2:4], 16)
    b = int(color_code[4:6], 16)
    print(f"Hex #{color_code} -> RGB({r}, {g}, {b})")
    
    # 2. Memory addresses and debugging
    # id() returns the memory address in modern Python implementations
    some_object = [1, 2, 3]
    address = id(some_object)
    hex_address = hex(address)
    print(f"\nObject memory address:")
    print(f"Object: {some_object}")
    print(f"Memory address (decimal): {address}")
    print(f"Memory address (hex): {hex_address}")
    
    # 3. Formatting hex for output
    data_bytes = [0, 127, 255, 10, 31, 42]
    hex_bytes = [hex(b)[2:].zfill(2) for b in data_bytes]
    formatted_hex = ' '.join(hex_bytes).upper()
    print(f"\nFormatting byte data as hex:")
    print(f"Original bytes: {data_bytes}")
    print(f"Formatted hex: {formatted_hex}")
    
    # 4. Conversion between number bases
    decimal = 42
    binary = bin(decimal)
    hexadecimal = hex(decimal)
    octal = oct(decimal)
    print(f"\nNumber base conversion:")
    print(f"Decimal: {decimal}")
    print(f"Binary: {binary}")
    print(f"Hexadecimal: {hexadecimal}")
    print(f"Octal: {octal}")
    
    # Converting back to decimal
    print(f"\nConverting back to decimal:")
    print(f"From binary {binary}: {int(binary, 0)}")
    print(f"From hex {hexadecimal}: {int(hexadecimal, 0)}")
    print(f"From octal {octal}: {int(octal, 0)}")


##############################################################################
#                               OCT FUNCTION                                 #
##############################################################################
"""
oct() - Converts an integer to an octal string prefixed with "0o".

Syntax:
    oct(x)

Parameters:
    x - An integer object

Returns:
    A string with the octal representation of x, prefixed with "0o"
"""

def oct_function_examples():
    """Examples demonstrating the oct() built-in function."""
    
    print("\n" + "=" * 30)
    print("OCT FUNCTION EXAMPLES")
    print("=" * 30)
    
    # Basic usage with positive integers
    print(f"oct(0) -> {oct(0)}")
    print(f"oct(1) -> {oct(1)}")
    print(f"oct(7) -> {oct(7)}")
    print(f"oct(8) -> {oct(8)}")
    print(f"oct(10) -> {oct(10)}")
    print(f"oct(64) -> {oct(64)}")
    print(f"oct(100) -> {oct(100)}")
    
    # Negative integers
    print(f"\nNegative numbers:")
    print(f"oct(-1) -> {oct(-1)}")
    print(f"oct(-8) -> {oct(-8)}")
    print(f"oct(-100) -> {oct(-100)}")
    
    # Objects with __index__ method
    class OctCompatible:
        def __init__(self, value):
            self.value = value
        
        def __index__(self):
            return self.value
    
    obj = OctCompatible(64)
    print(f"\nCustom object with __index__:")
    print(f"oct(OctCompatible(64)) -> {oct(obj)}")
    
    # Removing the "0o" prefix
    value = 64
    oct_with_prefix = oct(value)
    oct_without_prefix = oct(value)[2:]  # Remove first two characters
    print(f"\nRemoving '0o' prefix:")
    print(f"With prefix: {oct_with_prefix}")
    print(f"Without prefix: {oct_without_prefix}")
    
    # Ensuring specific width (padding with zeros)
    value = 7  # octal: 7
    padded_oct = oct(value)[2:].zfill(3)  # Pad to 3 octal digits
    print(f"\nPadding with zeros to specific width:")
    print(f"oct({value}) -> {oct(value)}")
    print(f"Padded to 3 octal digits: {padded_oct}")
    
    # Converting between octal strings and integers
    octal_string = "644"  # Without 0o prefix
    value = int(octal_string, 8)
    print(f"\nConversion between octal string and int:")
    print(f"Octal string: {octal_string}")
    print(f"Converted to int: {value}")
    print(f"Back to octal: {oct(value)}")
    
    # OCT EXCEPTION CASES
    
    print("\nException cases:")
    try:
        # Non-integer value
        print(oct(3.14))
    except TypeError as e:
        print(f"oct(3.14) raises: {e}")
    
    try:
        # String value without __index__
        print(oct("644"))
    except TypeError as e:
        print(f"oct('644') raises: {e}")
    
    # Practical applications
    
    # 1. File permissions in Unix/Linux systems
    # Unix file permissions are commonly represented in octal
    # r (read) = 4, w (write) = 2, x (execute) = 1
    
    # Define permissions as octal
    user_perm = 7    # rwx (4+2+1)
    group_perm = 5   # r-x (4+0+1)
    others_perm = 0  # --- (0+0+0)
    
    # Combine into a complete permission set
    # In octal, each digit represents 3 bits (file permissions for user/group/others)
    permission = (user_perm << 6) | (group_perm << 3) | others_perm
    
    print(f"\nUnix file permissions:")
    print(f"User: {user_perm} (rwx)")
    print(f"Group: {group_perm} (r-x)")
    print(f"Others: {others_perm} (---)")
    print(f"Combined permission value: {permission} ({oct(permission)})")
    print(f"Typical chmod notation: {oct(permission)[2:]}")  # e.g., chmod 750
    
    # 2. Extracting individual permission components
    permission = 0o750  # rwxr-x---
    
    # Extract components
    user = (permission >> 6) & 0o7
    group = (permission >> 3) & 0o7
    others = permission & 0o7
    
    print(f"\nExtracting from permission {oct(permission)}:")
    print(f"User: {user} ({oct(user)[2:]})")
    print(f"Group: {group} ({oct(group)[2:]})")
    print(f"Others: {others} ({oct(others)[2:]})")
    
    # 3. Converting between different number bases
    decimal = 60
    octal_rep = oct(decimal)
    binary_rep = bin(decimal)
    hex_rep = hex(decimal)
    
    print(f"\nNumber base conversion for {decimal}:")
    print(f"Octal: {octal_rep}")
    print(f"Binary: {binary_rep}")
    print(f"Hexadecimal: {hex_rep}")
    
    # 4. Historical context - octal was more common in older computing systems
    print("\nOctal vs Hexadecimal for byte representation:")
    print("Octal requires 3 digits for 8 bits (byte): 0-7 (3 bits per digit)")
    print("Hexadecimal requires 2 digits for 8 bits: 0-F (4 bits per digit)")
    
    byte_value = 0b11010110  # 214 in decimal
    print(f"Byte value: {byte_value} (decimal)")
    print(f"Octal: {oct(byte_value)} (3 octal digits needed)")
    print(f"Hexadecimal: {hex(byte_value)} (2 hex digits needed)")


##############################################################################
#                                MAIN PROGRAM                                #
##############################################################################

if __name__ == "__main__":
    # Run all the examples
    format_function_examples()
    ascii_function_examples()
    chr_function_examples()
    ord_function_examples()
    bin_function_examples()
    hex_function_examples()
    oct_function_examples()