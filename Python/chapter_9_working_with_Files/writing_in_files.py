# Chapter 9: Working with Files: File I/O 📂 (Reading and Writing Data)

# 9.1 File Handling Basics: Opening and Closing Files 📂 (Opening the File Cabinet)

# Concept: Understanding how to open, read from, write to, and close files.
# Analogy: File handling is like working with a file cabinet 📂. You need to open a drawer (open a file),
# read or write documents (read/write data), and then close the drawer (close the file).

# Example 1: Opening a file for reading and closing it manually.
file = open("example1.txt", "r")  # Open the file "example1.txt" in read mode ("r")
# Perform file operations here...
file.close()  # Close the file to free up resources

# Example 2: Opening a file for writing (will create the file if it doesn't exist).
file = open("example2.txt", "w")  # Open the file "example2.txt" in write mode ("w")
file.write("Hello, World!")  # Write a string to the file
file.close()  # Close the file

# Example 3: Opening a file for appending data.
file = open("example3.txt", "a")  # Open the file "example3.txt" in append mode ("a")
file.write("Adding a new line.\n")  # Append a line to the file
file.close()  # Close the file

# Example 4: Opening a binary file for reading.
file = open("image.jpg", "rb")  # Open "image.jpg" in binary read mode ("rb")
# Perform binary file operations...
file.close()  # Close the file

# Example 5: Opening a file with read and write permissions.
file = open("example5.txt", "r+")  # Open "example5.txt" for reading and writing ("r+")
# Read or write data...
file.close()  # Close the file

# Example 6: Using try-except to handle file not found error.
try:
    file = open("nonexistent.txt", "r")  # Attempt to open a file that may not exist
except FileNotFoundError:
    print("The file does not exist.")
else:
    # If the file exists, perform operations
    file.close()  # Close the file

# Example 7: Checking if a file is closed.
file = open("example7.txt", "w")
print(file.closed)  # Output: False (file is open)
file.close()
print(file.closed)  # Output: True (file is closed)

# Example 8: Working with file encoding.
file = open("example8.txt", "r", encoding="utf-8")  # Specify the file encoding
# Read data...
file.close()

# Example 9: Writing data to a file using different encodings.
file = open("example9.txt", "w", encoding="utf-16")  # Write with UTF-16 encoding
file.write("Some text with special characters: ü, ö, ä")
file.close()

# Example 10: Handling IO errors.
try:
    file = open("example10.txt", "r")
except IOError:
    print("An I/O error occurred.")
else:
    file.close()

# Example 11: Using "with" statement for automatic closure (more on this later).
with open("example11.txt", "w") as file:
    file.write("Using 'with' for automatic file closure.")

# Example 12: Getting file information.
file = open("example12.txt", "w")
print("File name:", file.name)  # Output the name of the file
print("File mode:", file.mode)  # Output the mode in which the file is opened
file.close()

# Example 13: Truncating a file (clearing its contents).
file = open("example13.txt", "w")
file.write("This will be truncated.")
file.truncate(0)  # Clear the file by truncating it to 0 bytes
file.close()

# Example 14: Flushing the file buffer.
file = open("example14.txt", "w")
file.write("This line is buffered.")
file.flush()  # Force write the buffer to disk
file.close()

# Example 15: Using os module to check file existence before opening.
import os

if os.path.exists("example15.txt"):
    file = open("example15.txt", "r")
    # Read data...
    file.close()
else:
    print("File does not exist.")

# 9.2 Reading from Files: Extracting Data 📖 (Reading Documents)

# Concept: Methods for reading data from files.
# Analogy: Reading from a file is like reading documents 📖 from a file cabinet.

# Example 1: Reading the entire file content as a single string.
with open("read_example1.txt", "r") as file:
    content = file.read()  # Read the whole file
    print(content)  # Print the content

# Example 2: Reading the file line by line using readline().
with open("read_example2.txt", "r") as file:
    line1 = file.readline()  # Read the first line
    line2 = file.readline()  # Read the second line
    print(line1.strip())
    print(line2.strip())

# Example 3: Reading all lines into a list using readlines().
with open("read_example3.txt", "r") as file:
    lines = file.readlines()  # Read all lines into a list
    print(lines)  # Print the list of lines

# Example 4: Iterating over the file object directly.
with open("read_example4.txt", "r") as file:
    for line in file:
        print(line.strip())  # Print each line without extra newline

# Example 5: Reading a specific number of characters.
with open("read_example5.txt", "r") as file:
    partial_content = file.read(10)  # Read the first 10 characters
    print(partial_content)

# Example 6: Seeking to a position in the file.
with open("read_example6.txt", "r") as file:
    file.seek(5)  # Move to the 6th byte (offset from start)
    content = file.read()
    print(content)

# Example 7: Reading binary data.
with open("binary_file.bin", "rb") as file:
    binary_data = file.read()
    print(binary_data)

# Example 8: Using a loop to read large files efficiently.
with open("large_file.txt", "r") as file:
    for line in file:
        process(line)  # Hypothetical function to process each line

# Example 9: Reading file with specific encoding.
with open("read_example9.txt", "r", encoding="utf-8") as file:
    content = file.read()
    print(content)

# Example 10: Handling exceptions during file reading.
try:
    with open("read_example10.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found.")

# Example 11: Using contextlib to suppress exceptions.
from contextlib import suppress

with suppress(FileNotFoundError):
    with open("nonexistent_file.txt", "r") as file:
        content = file.read()

# Example 12: Reading CSV files.
with open("data.csv", "r") as file:
    for line in file:
        fields = line.split(",")  # Split each line by comma
        print(fields)

# Example 13: Reading JSON files.
import json

with open("data.json", "r") as file:
    data = json.load(file)  # Parse JSON data
    print(data)

# Example 14: Reading files using pathlib (alternative to open).
from pathlib import Path

file_path = Path("read_example14.txt")
content = file_path.read_text()
print(content)

# Example 15: Using iterator to read chunks of data.
def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function to read a file piece by piece."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

with open("large_file.txt", "r") as file:
    for piece in read_in_chunks(file):
        process(piece)  # Process each chunk

# 9.3 Writing to Files: Storing Data ✍️ (Writing Documents)

# Concept: Methods for writing data to files.
# Analogy: Writing to a file is like writing documents ✍️ and storing them in a file cabinet.

# Example 1: Writing a string to a new file.
with open("write_example1.txt", "w") as file:
    file.write("This is a new file.\n")  # Write a line to the file

# Example 2: Writing multiple lines using writelines().
lines = ["Line one.\n", "Line two.\n", "Line three.\n"]
with open("write_example2.txt", "w") as file:
    file.writelines(lines)  # Write multiple lines

# Example 3: Appending data to an existing file.
with open("write_example3.txt", "a") as file:
    file.write("Appending a new line.\n")

# Example 4: Writing data with automatic newline handling.
data = ["Item1", "Item2", "Item3"]
with open("write_example4.txt", "w") as file:
    for item in data:
        file.write(f"{item}\n")  # Write each item on a new line

# Example 5: Writing binary data to a file.
with open("binary_output.bin", "wb") as file:
    binary_data = bytes([120, 3, 255, 0, 100])
    file.write(binary_data)

# Example 6: Writing formatted strings.
name = "Alice"
age = 30
with open("write_example6.txt", "w") as file:
    file.write(f"Name: {name}, Age: {age}\n")

# Example 7: Using print function to write to a file.
with open("write_example7.txt", "w") as file:
    print("Using print function to write.", file=file)

# Example 8: Handling exceptions during file writing.
try:
    with open("write_example8.txt", "w") as file:
        file.write("Writing data safely.")
except IOError:
    print("An error occurred while writing to the file.")

# Example 9: Writing JSON data to a file.
import json

data = {"name": "Bob", "age": 25}
with open("write_example9.json", "w") as file:
    json.dump(data, file)

# Example 10: Writing CSV data.
import csv

rows = [["Name", "Age"], ["John", 28], ["Emma", 22]]
with open("write_example10.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(rows)

# Example 11: Writing with different encodings.
with open("write_example11.txt", "w", encoding="utf-16") as file:
    file.write("Text with UTF-16 encoding.")

# Example 12: Ensuring data is saved with flush().
with open("write_example12.txt", "w") as file:
    file.write("Data not yet saved.")
    file.flush()  # Ensure data is written to disk

# Example 13: Overwriting existing file content.
with open("write_example13.txt", "w") as file:
    file.write("This will overwrite any existing content.")

# Example 14: Writing large amounts of data efficiently.
large_data = ["Line {}\n".format(i) for i in range(1000000)]
with open("write_example14.txt", "w") as file:
    file.writelines(large_data)

# Example 15: Using pathlib to write text.
from pathlib import Path

file_path = Path("write_example15.txt")
file_path.write_text("Writing text using pathlib.")

# 9.4 Context Managers (with statement): Safe File Handling 🛡️ (Automatic Closure)

# Concept: Using with statement for automatic resource management (file closing).
# Analogy: with statement is like having a smart file cabinet 🛡️ that automatically closes the drawer when you're done.

# Example 1: Basic usage of 'with' statement.
with open("with_example1.txt", "r") as file:
    content = file.read()
    print(content)
# No need to call file.close(); it's automatic!

# Example 2: Handling exceptions within 'with' block.
try:
    with open("with_example2.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found.")

# Example 3: Writing to a file using 'with'.
with open("with_example3.txt", "w") as file:
    file.write("Writing safely with 'with'.")

# Example 4: Using multiple context managers.
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    data = infile.read()
    outfile.write(data)

# Example 5: Custom context manager using contextlib.
from contextlib import contextmanager

@contextmanager
def custom_open(file_name):
    f = open(file_name, "w")
    yield f
    f.close()

with custom_open("with_example5.txt") as file:
    file.write("Using a custom context manager.")

# Example 6: Context manager for database connection (hypothetical).
class DatabaseConnection:
    def __enter__(self):
        # Connect to the database
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the database connection
        pass

with DatabaseConnection() as db:
    # Perform database operations
    pass

# Example 7: Suppressing exceptions with context manager.
from contextlib import suppress

with suppress(FileNotFoundError):
    with open("nonexistent_file.txt", "r") as file:
        content = file.read()

# Example 8: Redirecting stdout using context manager.
import sys
from contextlib import redirect_stdout

with open("with_example8.txt", "w") as f:
    with redirect_stdout(f):
        print("This will be written to the file.")

# Example 9: Timing code execution with context manager.
import time

class Timer:
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()
        print(f"Elapsed time: {self.end - self.start} seconds")

with Timer():
    # Some time-consuming operations
    sum(range(1000000))

# Example 10: Using context manager for threading locks.
import threading

lock = threading.Lock()

with lock:
    # Thread-safe operations
    pass

# Example 11: Context manager to change working directory.
import os
from contextlib import contextmanager

@contextmanager
def change_dir(destination):
    original_dir = os.getcwd()
    os.chdir(destination)
    try:
        yield
    finally:
        os.chdir(original_dir)

with change_dir("/path/to/directory"):
    # Operations in the new directory
    pass

# Example 12: Ensuring file closure even if an error occurs.
with open("with_example12.txt", "w") as file:
    file.write("This will be written.")
    # An intentional error
    # x = 1 / 0  # Uncomment to test exception handling

# Example 13: Nested 'with' statements.
with open("with_input.txt", "r") as infile:
    with open("with_output.txt", "w") as outfile:
        data = infile.read()
        outfile.write(data)

# Example 14: Context manager for resource acquisition and release.
class ManagedResource:
    def __enter__(self):
        # Acquire resource
        print("Acquiring resource")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release resource
        print("Releasing resource")

with ManagedResource() as resource:
    # Use the resource
    pass

# Example 15: Using contextlib.closing to close objects that need closing.
from contextlib import closing
import urllib.request

with closing(urllib.request.urlopen('http://www.example.com')) as page:
    for line in page:
        print(line)