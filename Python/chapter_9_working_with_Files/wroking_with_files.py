# Chapter 9: Working with Files: File I/O 📂 (Reading and Writing Data)

# 9.1 File Handling Basics: Opening and Closing Files 📂 (Opening the File Cabinet)

# Example 1: Opening a file in read mode ('r')
file = open('example1.txt', 'r')  # Open file for reading
# ... perform file operations ...
file.close()  # Close the file 🔒

# Example 2: Opening a file in write mode ('w')
file = open('example2.txt', 'w')  # Open file for writing (overwrites if file exists)
# ... perform file operations ...
file.close()  # Close the file 🔒

# Example 3: Opening a file in append mode ('a')
file = open('example3.txt', 'a')  # Open file for appending
# ... perform file operations ...
file.close()  # Close the file 🔒

# Example 4: Opening a file in binary read mode ('rb')
file = open('example4.bin', 'rb')  # Open binary file for reading
# ... perform file operations ...
file.close()  # Close the file 🔒

# Example 5: Opening a file in binary write mode ('wb')
file = open('example5.bin', 'wb')  # Open binary file for writing
# ... perform file operations ...
file.close()  # Close the file 🔒

# Example 6: Opening a file in read and write mode ('r+')
file = open('example6.txt', 'r+')  # Open file for reading and writing
# ... perform file operations ...
file.close()  # Close the file 🔒

# Example 7: Attempting to open a non-existent file in read mode
try:
    file = open('nonexistent.txt', 'r')  # Error if file doesn't exist 🚫
except FileNotFoundError as e:
    print(f"Error: {e}")
# No need to close the file as it wasn't opened

# Example 8: Opening a file with encoding specified
file = open('example8.txt', 'r', encoding='utf-8')  # Specify encoding
# ... perform file operations ...
file.close()  # Close the file 🔒

# Example 9: Forgetting to close a file (Not Recommended)
file = open('example9.txt', 'r')
# ... perform file operations ...
# file.close()  # Oops! File not closed 😱

# Example 10: Checking if a file is closed
file = open('example10.txt', 'r')
print(file.closed)  # Outputs: False
file.close()
print(file.closed)  # Outputs: True

# Example 11: Using absolute and relative file paths
file = open('/path/to/your/file.txt', 'r')  # Absolute path
file.close()
file = open('relative/path/to/your/file.txt', 'r')  # Relative path
file.close()

# Example 12: Handling exceptions when closing a file
try:
    file = open('example12.txt', 'r')
finally:
    file.close()  # Ensures file is closed even if an error occurs 🛡️

# Example 13: Opening multiple files
file1 = open('file1.txt', 'r')
file2 = open('file2.txt', 'w')
# ... perform operations on files ...
file1.close()
file2.close()

# Example 14: Misusing file modes (Common Mistake)
try:
    file = open('example14.txt', 'w')  # Open in write mode
    content = file.read()  # Error! Can't read in write mode 🚫
except io.UnsupportedOperation as e:
    print(f"Error: {e}")
finally:
    file.close()

# Example 15: Using os module to check if a file exists before opening
import os
if os.path.exists('example15.txt'):
    file = open('example15.txt', 'r')
    # ... perform file operations ...
    file.close()
else:
    print("File does not exist.")

# 9.2 Reading from Files: Extracting Data 📖 (Reading Documents)

# Example 1: Reading the entire file using read()
with open('example1.txt', 'r') as file:
    content = file.read()  # Reads entire file into memory
    print(content)

# Example 2: Reading line by line using readline()
with open('example2.txt', 'r') as file:
    line = file.readline()
    while line:
        print(line.strip())
        line = file.readline()

# Example 3: Reading all lines into a list using readlines()
with open('example3.txt', 'r') as file:
    lines = file.readlines()  # List of lines
    for line in lines:
        print(line.strip())

# Example 4: Iterating over the file object
with open('example4.txt', 'r') as file:
    for line in file:
        print(line.strip())

# Example 5: Reading a fixed number of characters
with open('example5.txt', 'r') as file:
    content = file.read(10)  # Reads first 10 characters
    print(content)

# Example 6: Handling large files efficiently
with open('largefile.txt', 'r') as file:
    for line in file:
        process(line)  # Replace with actual processing function

# Example 7: Reading binary files
with open('example7.bin', 'rb') as file:
    data = file.read()
    print(data)

# Example 8: Using try-except while reading files
try:
    with open('example8.txt', 'r') as file:
        content = file.read()
except FileNotFoundError as e:
    print(f"Error: {e}")

# Example 9: Reading CSV files line by line
with open('data.csv', 'r') as file:
    for line in file:
        fields = line.strip().split(',')
        print(fields)

# Example 10: Reading JSON data from a file
import json
with open('data.json', 'r') as file:
    data = json.load(file)  # Parses JSON into Python objects
    print(data)

# Example 11: Stripping whitespace from lines
with open('example11.txt', 'r') as file:
    for line in file:
        print(line.strip())  # Removes leading/trailing whitespace

# Example 12: Reading file using a specific encoding
with open('example12.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    print(content)

# Example 13: Handling UnicodeDecodeError
try:
    with open('example13.txt', 'r', encoding='utf-8') as file:
        content = file.read()
except UnicodeDecodeError as e:
    print(f"Encoding Error: {e}")

# Example 14: Using seek() to navigate file pointer
with open('example14.txt', 'r') as file:
    file.seek(5)  # Move to the 5th byte
    content = file.read()
    print(content)

# Example 15: Checking if file is readable
with open('example15.txt', 'r') as file:
    if file.readable():
        content = file.read()
        print(content)
    else:
        print("File is not readable")

# 9.3 Writing to Files: Storing Data ✍️ (Writing Documents)

# Example 1: Writing a single line to a file
with open('output1.txt', 'w') as file:
    file.write("Hello, World!\n")  # Writes text to file

# Example 2: Writing multiple lines using writelines()
lines = ["First line\n", "Second line\n", "Third line\n"]
with open('output2.txt', 'w') as file:
    file.writelines(lines)

# Example 3: Appending to a file
with open('output3.txt', 'a') as file:
    file.write("Appending a new line.\n")

# Example 4: Writing binary data to a file
with open('output4.bin', 'wb') as file:
    data = bytes([120, 3, 255, 0, 100])
    file.write(data)

# Example 5: Using write() without newline character
with open('output5.txt', 'w') as file:
    file.write("This is a line.")
    file.write("This is another line.")  # Continues on the same line

# Example 6: Writing user input to a file
user_input = input("Enter some text: ")
with open('output6.txt', 'w') as file:
    file.write(user_input + '\n')

# Example 7: Overwriting an existing file
with open('output7.txt', 'w') as file:
    file.write("This will overwrite any existing content.")

# Example 8: Writing numbers to a file
numbers = [1, 2, 3, 4, 5]
with open('output8.txt', 'w') as file:
    for num in numbers:
        file.write(f"{num}\n")

# Example 9: Handling exceptions during writing
try:
    with open('readonly.txt', 'w') as file:
        file.write("Attempting to write to a read-only file.")
except IOError as e:
    print(f"I/O Error: {e}")

# Example 10: Flushing output to the file
with open('output10.txt', 'w') as file:
    file.write("Data may not be immediately written.")
    file.flush()  # Flushes the internal buffer

# Example 11: Writing JSON data to a file
import json
data = {'name': 'Alice', 'age': 30}
with open('output11.json', 'w') as file:
    json.dump(data, file)

# Example 12: Using seek() to overwrite content
with open('output12.txt', 'w+') as file:
    file.write("Hello, World!")
    file.seek(7)
    file.write("Python")  # Overwrites 'World' with 'Python'

# Example 13: Writing to a file opened in read mode (Error)
try:
    with open('output13.txt', 'r') as file:
        file.write("Trying to write.")  # Error 🚫
except io.UnsupportedOperation as e:
    print(f"Error: {e}")

# Example 14: Checking if file is writable
with open('output14.txt', 'w') as file:
    if file.writable():
        file.write("The file is writable.")
    else:
        print("File is not writable.")

# Example 15: Writing formatted strings
value = 42
with open('output15.txt', 'w') as file:
    file.write(f"The answer is {value}.\n")

# 9.4 Context Managers (with statement): Safe File Handling 🛡️ (Automatic Closure)

# Example 1: Basic usage of 'with' statement
with open('example1.txt', 'r') as file:
    content = file.read()
    print(content)
# File is automatically closed here 🔒

# Example 2: No need to call close()
with open('example2.txt', 'w') as file:
    file.write("No need to manually close the file.")
# File is automatically closed here 🔒

# Example 3: Handling exceptions within 'with' block
try:
    with open('example3.txt', 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError as e:
    print(f"Error: {e}")
# File is closed even if an exception occurs 🛡️

# Example 4: Using multiple context managers
with open('input.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    data = infile.read()
    outfile.write(data)
# Both files are automatically closed 🔒

# Example 5: Custom context manager using 'contextlib'
from contextlib import contextmanager

@contextmanager
def open_file(name, mode):
    f = open(name, mode)
    try:
        yield f
    finally:
        f.close()
        print("File closed.")

with open_file('example5.txt', 'w') as file:
    file.write("Using a custom context manager.")
# "File closed." is printed after exiting the block

# Example 6: Nested 'with' statements
with open('example6.txt', 'w') as file:
    with open('example7.txt', 'r') as infile:
        data = infile.read()
        file.write(data)
# Both files are closed automatically 🔒

# Example 7: Using 'with' for resource management
class Sample:
    def __enter__(self):
        print("Entered.")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exited.")

with Sample() as sample:
    print("Inside with block.")
# "Entered.", "Inside with block.", "Exited." are printed

# Example 8: Suppressing exceptions in '__exit__'
class SuppressError:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True  # Suppress exception

with SuppressError():
    1 / 0  # Division by zero, exception suppressed 😶
print("Program continues.")

# Example 9: Context manager for timing operations
import time
@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed time: {end - start} seconds")

with timer():
    time.sleep(1)  # Simulate a long operation
# Prints elapsed time after operation

# Example 10: File operations with exception handling
with open('example10.txt', 'w') as file:
    file.write("Hello, World!")
    # An exception occurs here
    # File is still closed properly

# Example 11: Using 'with' with database connections (conceptual)
# with database.connect() as connection:
#     # Perform database operations
# Connection is closed automatically

# Example 12: Ensuring file closure in functions
def write_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(data)
        # No need to close the file
write_to_file('example12.txt', 'Data written from function.')

# Example 13: Incorrect use without 'with' (Common Mistake)
file = open('example13.txt', 'w')
file.write("Forgetting to close the file.")
# file.close()  # File remains open 😱

# Example 14: Using 'with' for thread locks
import threading
lock = threading.Lock()
with lock:
    # Thread-safe operations
    pass
# Lock is released automatically 🔓

# Example 15: Chaining context managers
from contextlib import ExitStack
with ExitStack() as stack:
    files = [stack.enter_context(open(f'file{i}.txt', 'w')) for i in range(3)]
    for file in files:
        file.write("Chained context managers.")
# All files are closed automatically 🔒

# End of Chapter 9 Examples