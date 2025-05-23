# -*- coding: utf-8 -*-
"""
Input and Output - Mastering Data Handling in Python for Advanced Applications

This script provides a comprehensive exploration of Python's input and output capabilities, essential for building robust and data-driven applications.
Designed for advanced Python developers, it delves into sophisticated formatting techniques, efficient file handling strategies, and structured data serialization using JSON.

This rigorous exploration will cover:

    - Fancier Output Formatting: Mastering f-strings, the `format()` method, manual formatting, and understanding legacy formatting techniques.
    - Reading and Writing Files: In-depth file object methods, binary vs. text modes, encoding nuances, and robust file I/O practices.
    - Saving structured data with JSON: Leveraging the `json` module for efficient serialization and deserialization of complex data structures.

Expect a focus on:

    - Advanced formatting techniques for precise and readable output.
    - Efficient and secure file handling practices.
    - Understanding encoding complexities and handling them gracefully.
    - Best practices for structured data serialization and deserialization using JSON.
    - Robust error handling strategies for all I/O operations.
    - Pythonic idioms and performance considerations for data handling.

Let's embark on this advanced journey to master Python's input and output mechanisms and elevate your data processing skills.
"""

################################################################################
# 7. Input and Output
################################################################################

print("\n--- 7. Input and Output ---\n")

# Python's input and output (I/O) capabilities are fundamental for interacting with external systems, users, and data storage.
# Mastering these techniques is crucial for building applications that can effectively process and present information.

################################################################################
# 7.1. Fancier Output Formatting
################################################################################

print("\n--- 7.1. Fancier Output Formatting ---\n")

# Python offers several powerful ways to format output, going beyond simple `print()` statements.
# We will explore formatted string literals (f-strings), the `str.format()` method, manual string formatting, and briefly touch upon older formatting techniques.

################################################################################
# 7.1.1. Formatted String Literals (f-strings)
################################################################################

print("\n--- 7.1.1. Formatted String Literals (f-strings) ---\n")

# Formatted string literals, or f-strings (introduced in Python 3.6), provide a concise and readable way to embed expressions inside string literals for formatting.
# They are prefixed with an 'f' or 'F' and use curly braces `{}` to enclose expressions that will be evaluated and inserted into the string.

# --- Basic f-string example ---
print("\n--- Basic f-string example ---")
name = "Alice"
age = 30
greeting = f"Hello, {name}! You are {age} years old." # Variable embedding
print(greeting)

# --- Expression evaluation within f-strings ---
print("\n--- Expression evaluation within f-strings ---")
x = 10
y = 7
result = f"The sum of {x} and {y} is {x + y}, and their product is {x * y}." # Inline expressions
print(result)

# --- Format specifiers within f-strings ---
print("\n--- Format specifiers within f-strings ---")
# Format specifiers control how values are formatted (e.g., precision for floats, alignment, padding).
pi_value = 3.1415926535
formatted_pi = f"Pi to 3 decimal places: {pi_value:.3f}" # .3f - float with 3 decimal places
print(formatted_pi)

integer_number = 42
formatted_integer = f"Binary representation of {integer_number} is {integer_number:b}, hexadecimal is {integer_number:x}, and decimal is {integer_number:d}." # b, x, d - binary, hexadecimal, decimal
print(formatted_integer)

aligned_string = "Python"
aligned_output = f"Left aligned: |{aligned_string:<10}|, Right aligned: |{aligned_string:>10}|, Centered: |{aligned_string:^10}|" # <, >, ^ - left, right, center alignment, 10 - width
print(aligned_output)

padded_number = 5
padded_output = f"Padded with zeros: {padded_number:05d}" # 05d - pad with zeros to width 5, decimal integer
print(padded_output)

# --- Escape sequences within f-strings ---
print("\n--- Escape sequences within f-strings ---")
newline_fstring = f"This string\nhas a newline character." # Standard escape sequences work within f-strings
print(newline_fstring)

# --- Raw f-strings (fr-strings) - for verbatim strings, escape sequences are not processed ---
raw_fstring = fr"Raw f-string: Newline is \\n, not a real newline." # 'r' prefix makes it a raw string
print(raw_fstring)

# --- Handling potential exceptions with f-strings ---
print("\n--- f-string Exception Handling ---")
try:
    undefined_variable # NameError - variable not defined
    fstring_error = f"Value: {undefined_variable}" # NameError will be raised *before* f-string formatting
except NameError as e:
    print(f"NameError in f-string: {e}")

try:
    invalid_format_specifier = f"{10:.x}" # ValueError - invalid format specifier for integer
except ValueError as e:
    print(f"ValueError in f-string format specifier: {e}")

# F-strings are generally efficient and highly recommended for modern Python development due to their readability and performance.

################################################################################
# 7.1.2. The String format() Method
################################################################################

print("\n--- 7.1.2. The String format() Method ---\n")

# The `str.format()` method provides another flexible way to format strings. It uses curly braces `{}` as placeholders, similar to f-strings, but formatting is controlled by method calls rather than f-string prefixes.

# --- Basic format() method example ---
print("\n--- Basic format() method example ---")
name_format = "Bob"
age_format = 28
greeting_format = "Hello, {}! You are {} years old.".format(name_format, age_format) # Positional arguments
print(greeting_format)

# --- Positional arguments in format() ---
print("\n--- Positional arguments in format() ---")
positional_format = "{0}, {1}, {2}".format("first", "second", "third") # Explicit positional indices
print(positional_format)

# --- Keyword arguments in format() ---
print("\n--- Keyword arguments in format() ---")
keyword_format = "Name: {name}, Age: {age}, City: {city}".format(name="Eve", age=35, city="Paris") # Keyword arguments
print(keyword_format)

# --- Mixing positional and keyword arguments in format() ---
print("\n--- Mixing positional and keyword arguments in format() ---")
mixed_format = "Value at index 0: {0}, Keyword value: {key}".format("positional_value", key="keyword_value")
print(mixed_format)

# --- Format specifiers with format() method ---
print("\n--- Format specifiers with format() method ---")
pi_value_format = 3.1415926535
formatted_pi_format = "Pi to 4 decimal places: {:.4f}".format(pi_value_format) # {:.4f} - format specifier within format() placeholders
print(formatted_pi_format)

aligned_string_format = "FormatMethod"
aligned_output_format = "Left: |{:<15}|, Right: |{:>15}|, Center: |{:^15}|".format(aligned_string_format, aligned_string_format, aligned_string_format)
print(aligned_output_format)

# --- Accessing attributes and items within format() ---
print("\n--- Accessing attributes and items within format() ---")
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

point_obj = Point(3, 5)
attribute_format = "Point coordinates: x={0.x}, y={0.y}".format(point_obj) # Accessing object attributes
print(attribute_format)

data_list_format = ['item1', 'item2', 'item3']
item_format = "First item: {0[0]}, Second item: {0[1]}".format(data_list_format) # Accessing list items by index
print(item_format)

data_dict_format = {'key1': 'value1', 'key2': 'value2'}
dict_format = "Value for key1: {0[key1]}, Value for key2: {0[key2]}".format(data_dict_format) # Accessing dictionary items by key
print(dict_format)

# --- Handling potential exceptions with format() method ---
print("\n--- format() method Exception Handling ---")
try:
    format_index_error = "Value at index {0}".format() # IndexError - missing positional argument
except IndexError as e:
    print(f"IndexError in format(): {e}")

try:
    format_key_error = "Value for key {key}".format() # KeyError - missing keyword argument
except KeyError as e:
    print(f"KeyError in format(): {e}")

try:
    invalid_format_specifier_format = "{:.x}".format(10) # ValueError - invalid format specifier
except ValueError as e:
    print(f"ValueError in format() format specifier: {e}")

# `str.format()` is versatile and powerful, offering similar formatting capabilities to f-strings, but with a slightly different syntax.
# It's still widely used, especially in codebases predating Python 3.6, and remains a valuable tool for string formatting.

################################################################################
# 7.1.3. Manual String Formatting
################################################################################

print("\n--- 7.1.3. Manual String Formatting ---\n")

# Manual string formatting involves building strings using concatenation, string methods, and type conversions, without relying on dedicated formatting features like f-strings or `format()`.
# While less concise and potentially less efficient for complex formatting, it provides fine-grained control and can be useful in specific scenarios.

# --- String concatenation for manual formatting ---
print("\n--- String concatenation for manual formatting ---")
name_manual = "Carlos"
age_manual = 45
manual_greeting = "Hello, " + name_manual + "! You are " + str(age_manual) + " years old." # String concatenation using '+' and type conversion (str())
print(manual_greeting)

# --- String methods for manual formatting ---
print("\n--- String methods for manual formatting ---")
price = 99.99
currency_symbol = "$"
manual_price_string = currency_symbol + str(price) # Concatenation with currency symbol
print(manual_price_string)

manual_aligned_string = "ManualAlign"
manual_left_aligned = manual_aligned_string.ljust(15) # Left justify with width 15
manual_right_aligned = manual_aligned_string.rjust(15) # Right justify with width 15
manual_centered_aligned = manual_aligned_string.center(15) # Center justify with width 15
print(f"Manual left aligned: |{manual_left_aligned}|, Manual right aligned: |{manual_right_aligned}|, Manual centered: |{manual_centered_aligned}|")

# --- Limitations of manual string formatting ---
print("\n--- Limitations of manual string formatting ---")
# - Verbosity: Can become verbose and less readable for complex formatting.
# - Error-prone: Manual type conversions and concatenation can introduce errors.
# - Less efficient: String concatenation can be less efficient than dedicated formatting methods in some cases (especially in older Python versions due to string immutability and repeated object creation).
# - Lack of format specifiers: No direct support for precision control, number formatting, etc., compared to f-strings and `format()`.

# Manual string formatting is generally discouraged for complex or frequent formatting tasks in modern Python.
# F-strings and `str.format()` offer more readable, maintainable, and often more efficient solutions.
# Manual formatting might be used for very basic string compositions or when performance is highly critical and string operations are minimized and carefully optimized.

################################################################################
# 7.1.4. Old string formatting (% operator)
################################################################################

print("\n--- 7.1.4. Old string formatting (% operator) ---\n")

# The `%` operator is the oldest string formatting method in Python. It's inspired by C's `printf`-style formatting.
# While still functional, it's generally considered less readable and more error-prone compared to f-strings and `str.format()`, and has security implications if used improperly.

# --- Basic % operator formatting ---
print("\n--- Basic % operator formatting ---")
name_percent = "Diana"
age_percent = 32
percent_greeting = "Hello, %s! You are %d years old." % (name_percent, age_percent) # %s - string, %d - integer, tuple of values after %
print(percent_greeting)

# --- Format specifiers with % operator ---
print("\n--- Format specifiers with % operator ---")
float_percent = 123.4567
formatted_float_percent = "Float with 2 decimal places: %.2f" % float_percent # %.2f - float with 2 decimal places
print(formatted_float_percent)

hex_percent = 255
formatted_hex_percent = "Hexadecimal: %x, Decimal: %d" % (hex_percent, hex_percent) # %x - hexadecimal, %d - decimal
print(formatted_hex_percent)

# --- Dictionary-based % formatting ---
print("\n--- Dictionary-based % formatting ---")
person_info_percent = {'name': 'Frank', 'city': 'Rome'}
dict_percent_format = "Name: %(name)s, City: %(city)s" % person_info_percent # %(key)s - access value from dictionary by key
print(dict_percent_format)

# --- Security concerns with % operator (Format String Vulnerabilities) ---
print("\n--- Security concerns with % operator ---")
# WARNING: Using % operator with *untrusted user input* can lead to format string vulnerabilities, potentially allowing arbitrary code execution in older Python versions.
# This is because format specifiers can be manipulated to access memory or execute code if the format string itself is controlled by an attacker.
# Example (demonstration of vulnerability - DO NOT USE IN PRODUCTION with untrusted input):
# user_input = "%x %x %x %x %s" # Malicious user input attempting to read memory or cause a crash.
# # vulnerable_string = user_input % ("safe_string",) # If 'user_input' is directly used as format string, it's vulnerable.
# # print(vulnerable_string) # Potential security risk!

# To mitigate this risk, *never* use user-provided strings directly as format strings with the % operator. Always use a *fixed, controlled format string* and pass user data as *arguments*.

# --- Deprecation trends of % operator ---
print("\n--- Deprecation trends of % operator ---")
# The % operator is considered a legacy formatting method in modern Python.
# F-strings and `str.format()` are the preferred and recommended methods for new code due to their readability, safety, and often better performance.
# While % operator might still be encountered in older codebases, it's best to migrate to f-strings or `format()` for maintainability and security reasons.

# --- Exception Handling with % operator ---
print("\n--- % operator Exception Handling ---")
try:
    percent_type_error = "%d" % "string" # TypeError - format specifier expects integer, but string provided
except TypeError as e:
    print(f"TypeError in % operator formatting: {e}")

try:
    percent_value_error = "%(name)s" % {} # KeyError - key 'name' not found in dictionary
except KeyError as e:
    print(f"KeyError in % operator dictionary formatting: {e}")

# In summary, while the `%` operator is functional, its limitations, security concerns, and reduced readability make f-strings and `str.format()` better choices for string formatting in modern Python development. Use `%` operator with caution and avoid it with untrusted input.

################################################################################
# 7.2. Reading and Writing Files
################################################################################

print("\n--- 7.2. Reading and Writing Files ---\n")

# File I/O is essential for interacting with files on disk. Python provides built-in functions and methods for reading from and writing to files.

# --- Opening Files - the `open()` function ---
print("\n--- Opening Files - the `open()` function ---")

# The `open()` function is used to open files. It returns a file object, which is then used for reading or writing.
# Syntax: `open(filename, mode='r', encoding=None, errors=None, ...)`

# Common modes:
# 'r' (read): Open for reading (default). Error if file does not exist.
# 'w' (write): Open for writing, truncating the file first. Creates the file if it does not exist.
# 'a' (append): Open for writing, appending to the end of the file if it exists. Creates the file if it does not exist.
# 'x' (exclusive creation): Open for exclusive creation, failing if the file already exists.
# 'b' (binary mode): Binary mode (for non-text files like images, executables).
# 't' (text mode): Text mode (default).
# '+' (update mode): Open for updating (reading and writing).

# Combining modes: 'rb' (read binary), 'wt' (write text - default if 't' is omitted), 'r+' (read and write), 'w+' (write and read, truncating first), 'a+' (append and read), 'x+' (exclusive create and read).

# Encoding: Specifies the character encoding for text files (e.g., 'utf-8', 'latin-1', 'ascii'). Important for handling non-ASCII characters correctly. Default encoding is platform-dependent (often UTF-8 on modern systems, but best to specify explicitly).
# Errors: Specifies how encoding errors should be handled (e.g., 'strict' - raise ValueError, 'ignore' - ignore errors, 'replace' - replace with replacement character).

# --- Opening file in read mode ('r') ---
print("\n--- Opening file in read mode ('r') ---")
try:
    file_read = open('example.txt', 'r', encoding='utf-8') # Open 'example.txt' in read mode with UTF-8 encoding
    print(f"File opened in read mode: {file_read}")
    file_read.close() # Remember to close the file after use (or use 'with open' for automatic closing - recommended)
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except IOError as e: # Catch broader I/O errors as well
    print(f"IOError during file open: {e}")

# --- Opening file in write mode ('w') ---
print("\n--- Opening file in write mode ('w') ---")
try:
    file_write = open('output.txt', 'w', encoding='utf-8') # Open 'output.txt' in write mode (creates or truncates)
    print(f"File opened in write mode: {file_write}")
    file_write.close()
except IOError as e:
    print(f"IOError during file open in write mode: {e}")

# --- Using 'with open(...)' for automatic file closing (recommended) ---
print("\n--- Using 'with open(...)') for automatic file closing ---")
# The 'with open(...)' statement ensures that the file is automatically closed even if errors occur within the block. This is best practice for file handling.
try:
    with open('another_output.txt', 'w', encoding='utf-8') as file_auto_close: # File object assigned to 'file_auto_close' within 'with' block
        print(f"File opened in write mode using 'with': {file_auto_close}")
        # File operations within this block
    # file_auto_close is automatically closed when exiting the 'with' block
    print("File automatically closed after 'with' block.")
except IOError as e:
    print(f"IOError during file operations with 'with open': {e}")

# --- Handling encoding errors during file opening ---
print("\n--- Handling encoding errors during file opening ---")
try:
    file_encoding_error = open('non_utf8_file.txt', 'r', encoding='utf-8', errors='strict') # Try to open a non-UTF-8 file as UTF-8 with strict error handling
    # If 'non_utf8_file.txt' is not actually UTF-8 encoded, this will raise UnicodeDecodeError (a subclass of ValueError and IOError)
    file_content = file_encoding_error.read() # Attempt to read content
    print(f"File content (if successfully read): {file_content}")
    file_encoding_error.close()
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError during file open (encoding issue): {e}")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")

################################################################################
# 7.2.1. Methods of File Objects
################################################################################

print("\n--- 7.2.1. Methods of File Objects ---\n")

# File objects returned by `open()` provide various methods for reading and writing data.

# --- Reading from files ---
print("\n--- Reading from files ---")

# 1. read([size]): Reads at most 'size' characters (in text mode) or bytes (in binary mode). If size is negative or None, reads the entire file. Returns an empty string ('') when end of file (EOF) is reached.
print("\n--- read() method ---")
try:
    with open('example.txt', 'r', encoding='utf-8') as file_read_method:
        content_all = file_read_method.read() # Read entire file content
        print(f"read() - Entire file content:\n---\n{content_all}\n---")

        file_read_method.seek(0) # Reset file pointer to beginning for next read
        content_limited = file_read_method.read(50) # Read first 50 characters
        print(f"read(50) - First 50 characters:\n---\n{content_limited}\n---")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except IOError as e:
    print(f"IOError during read(): {e}")

# 2. readline(): Reads one line from the file. A trailing newline character (\n) is kept at the end of the string, and is only omitted on the last line of the file if the file doesn’t end in a newline. Returns an empty string only when EOF is encountered immediately.
print("\n--- readline() method ---")
try:
    with open('example.txt', 'r', encoding='utf-8') as file_readline_method:
        line1 = file_readline_method.readline() # Read first line
        print(f"readline() - First line:\n---\n{line1.rstrip()}\n---") # rstrip() to remove trailing newline

        line2 = file_readline_method.readline() # Read second line
        print(f"readline() - Second line:\n---\n{line2.rstrip()}\n---")

        remaining_content = file_readline_method.read() # Read rest of the file after readline() calls
        print(f"read() after readline() - Remaining content:\n---\n{remaining_content}\n---") # Shows that readline reads line by line
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except IOError as e:
    print(f"IOError during readline(): {e}")

# 3. readlines(): Reads all lines from the file as a list of strings.  Similar to readline(), trailing newline characters are retained.
print("\n--- readlines() method ---")
try:
    with open('example.txt', 'r', encoding='utf-8') as file_readlines_method:
        all_lines = file_readlines_method.readlines() # Read all lines into a list
        print(f"readlines() - All lines as a list:\n---\n{all_lines}\n---") # List of lines with newlines
        for index, line in enumerate(all_lines):
            print(f"Line {index+1}: {line.rstrip()}") # Iterate and print lines without trailing newlines
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except IOError as e:
    print(f"IOError during readlines(): {e}")

# --- Writing to files ---
print("\n--- Writing to files ---")

# 1. write(string): Writes the contents of string to the file, returning the number of characters written. In text mode, string should be a str object. In binary mode, it should be bytes-like object.
print("\n--- write() method ---")
try:
    with open('output.txt', 'w', encoding='utf-8') as file_write_method: # 'w' mode truncates file if it exists
        num_chars_written1 = file_write_method.write("This is the first line.\n") # Write a string
        print(f"write() - Characters written for line 1: {num_chars_written1}")
        num_chars_written2 = file_write_method.write("This is the second line.\n") # Write another string
        print(f"write() - Characters written for line 2: {num_chars_written2}")
        # File is automatically flushed and closed when exiting 'with' block
    print("Data written to 'output.txt' using write().")
except IOError as e:
    print(f"IOError during write(): {e}")

# 2. writelines(list_of_strings): Write a list of strings to the file. No newline characters are added between lines; you need to add them explicitly if desired.
print("\n--- writelines() method ---")
try:
    lines_to_write = ["Line one from writelines.\n", "Line two from writelines.\n", "Line three.\n"]
    with open('output_lines.txt', 'w', encoding='utf-8') as file_writelines_method:
        file_writelines_method.writelines(lines_to_write) # Write a list of strings
    print("Lines written to 'output_lines.txt' using writelines().")
except IOError as e:
    print(f"IOError during writelines(): {e}")

# --- File positioning - tell() and seek() ---
print("\n--- File positioning - tell() and seek() ---")

# 1. tell(): Returns the current file position (number of bytes from the beginning of the file in binary mode, an opaque number in text mode).
# 2. seek(offset, whence=0): Change the file position. 'whence' argument:
#    0 (default): Seek from the beginning of the file.
#    1: Seek relative to the current position.
#    2: Seek relative to the end of the file.
#    In text mode, only seeking relative to the beginning is allowed (except seeking to offset 0), and offsets must be values returned by tell(), or zero. In binary mode, seeking is more flexible.

print("\n--- tell() and seek() examples ---")
try:
    with open('example.txt', 'r', encoding='utf-8') as file_position_methods:
        initial_position = file_position_methods.tell() # Get initial position (0)
        print(f"Initial file position (tell()): {initial_position}")

        first_line = file_position_methods.readline()
        position_after_readline = file_position_methods.tell() # Position after reading a line
        print(f"File position after readline() (tell()): {position_after_readline}")
        print(f"First line read: {first_line.rstrip()}")

        file_position_methods.seek(initial_position) # Seek back to the beginning
        print(f"File position after seek(initial_position) (tell()): {file_position_methods.tell()}")
        content_from_start = file_position_methods.read(20) # Read from the beginning again
        print(f"Content read after seek(initial_position):\n---\n{content_from_start}\n---")

        try:
            file_position_methods.seek(10, 1) # Seek forward 10 bytes from current position (in text mode, relative seek is restricted)
            print("Relative seek in text mode - might raise error.") # Depending on encoding and offset, this could raise ValueError in text mode
        except ValueError as e:
            print(f"ValueError during relative seek in text mode (as expected): {e}")

except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except IOError as e:
    print(f"IOError during file positioning methods: {e}")

# --- Closing files - close() method ---
print("\n--- Closing files - close() method ---")
# file_object.close(): Closes the file. It's important to close files to release system resources and flush any buffered output.
# As mentioned, 'with open(...)' handles automatic closing. Manual closing is needed if you don't use 'with'.

try:
    file_manual_close = open('temp_file.txt', 'w', encoding='utf-8')
    file_manual_close.write("Data to be written.")
    file_manual_close.close() # Manually close the file
    print("File manually closed using close().")
except IOError as e:
    print(f"IOError during manual file closing: {e}")

# --- Exception Handling during File Operations ---
# Always handle potential exceptions like FileNotFoundError, IOError, UnicodeDecodeError, etc., when working with files to create robust applications.
# Using 'with open(...)' is crucial for ensuring proper resource management and automatic file closing, even in error scenarios.

################################################################################
# 7.2.2. Saving structured data with json
################################################################################

print("\n--- 7.2.2. Saving structured data with json ---\n")

# The `json` module in Python allows you to easily serialize (convert Python objects to JSON strings) and deserialize (convert JSON strings back to Python objects) structured data.
# JSON (JavaScript Object Notation) is a lightweight data-interchange format, widely used for data exchange on the web and in configuration files.

# --- Importing the json module ---
print("\n--- Importing the json module ---")
import json

# --- Serializing Python objects to JSON strings - json.dumps() ---
print("\n--- Serializing Python objects to JSON strings - json.dumps() ---")

data_to_serialize = {
    'name': 'Expert User',
    'age': 40,
    'skills': ['Python', 'Data Science', 'Machine Learning'],
    'address': {
        'street': '123 Advanced Street',
        'city': 'Tech City'
    }
}

try:
    json_string_serialized = json.dumps(data_to_serialize) # Serialize Python dictionary to JSON string
    print(f"Serialized JSON string:\n---\n{json_string_serialized}\n---")

    json_string_indented = json.dumps(data_to_serialize, indent=4) # Serialize with indentation for readability
    print(f"Serialized JSON string (indented):\n---\n{json_string_indented}\n---")

    json_string_compact = json.dumps(data_to_serialize, separators=(',', ':')) # Compact JSON with custom separators
    print(f"Serialized JSON string (compact):\n---\n{json_string_compact}\n---")

    json_string_sorted_keys = json.dumps(data_to_serialize, sort_keys=True) # Serialize with keys sorted alphabetically
    print(f"Serialized JSON string (sorted keys):\n---\n{json_string_sorted_keys}\n---")

except TypeError as e:
    print(f"TypeError during json.dumps() - object not serializable: {e}")

# --- Deserializing JSON strings to Python objects - json.loads() ---
print("\n--- Deserializing JSON strings to Python objects - json.loads() ---")

json_string_to_deserialize = '{"name": "JSON Data", "value": 100, "items": ["a", "b", "c"]}'

try:
    python_object_deserialized = json.loads(json_string_to_deserialize) # Deserialize JSON string to Python dictionary
    print(f"Deserialized Python object:\n---\n{python_object_deserialized}\n---")
    print(f"Type of deserialized object: {type(python_object_deserialized)}")
    print(f"Accessing deserialized data: Name = {python_object_deserialized['name']}, Value = {python_object_deserialized['value']}")

except json.JSONDecodeError as e:
    print(f"json.JSONDecodeError during json.loads() - invalid JSON string: {e}")

# --- Reading JSON data from a file - json.load() ---
print("\n--- Reading JSON data from a file - json.load() ---")
try:
    with open('data.json', 'r', encoding='utf-8') as json_file_read: # Assume 'data.json' contains valid JSON
        loaded_json_data = json.load(json_file_read) # Deserialize JSON data from file to Python object
        print(f"JSON data loaded from file:\n---\n{loaded_json_data}\n---")
        print(f"Type of loaded JSON data: {type(loaded_json_data)}")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
except json.JSONDecodeError as e:
    print(f"json.JSONDecodeError during json.load() - invalid JSON in file: {e}")
except IOError as e:
    print(f"IOError during json.load() file operation: {e}")

# --- Writing JSON data to a file - json.dump() ---
print("\n--- Writing JSON data to a file - json.dump() ---")
python_data_to_dump = {'config': {'setting1': 'value1', 'setting2': 123}, 'status': 'active'}
try:
    with open('config.json', 'w', encoding='utf-8') as json_file_write:
        json.dump(python_data_to_dump, json_file_write, indent=4, sort_keys=True) # Serialize Python object to JSON and write to file with indentation and sorted keys
        print("JSON data written to 'config.json' using json.dump().")
except IOError as e:
    print(f"IOError during json.dump() file operation: {e}")

# --- Handling JSON serialization and deserialization errors ---
# Be prepared to handle TypeError during serialization (if objects are not JSON serializable) and json.JSONDecodeError during deserialization (if JSON string is invalid or file content is not valid JSON).
# Proper error handling is crucial when working with external data formats like JSON.

print("\n--- End of Input and Output ---\n")