# Chapter 8: Handling Errors: Exceptions ⚠️
# Dealing with the Unexpected

# 8.1 What are Exceptions? Runtime Errors ⚠️ (Unexpected Events)

# Exceptions are errors that occur during the execution of a program.
# They disrupt the normal flow of the program's instructions.
# Understanding exceptions helps in writing robust code.

# Example 1: ZeroDivisionError
# Occurs when a number is divided by zero.
# result = 10 / 0  # ⚠️ ZeroDivisionError: division by zero

# Example 2: ValueError
# Occurs when a function receives an argument of the correct type but inappropriate value.
# number = int("abc")  # ⚠️ ValueError: invalid literal for int()

# Example 3: TypeError
# Occurs when an operation is applied to an object of inappropriate type.
# sum = '2' + 2  # ⚠️ TypeError: can only concatenate str (not "int") to str

# Example 4: IndexError
# Occurs when a sequence subscript is out of range.
my_list = [1, 2, 3]
# item = my_list[5]  # ⚠️ IndexError: list index out of range

# Example 5: KeyError
# Occurs when a dictionary key is not found.
my_dict = {'name': 'Alice'}
# value = my_dict['age']  # ⚠️ KeyError: 'age'

# Example 6: FileNotFoundError
# Occurs when a file operation fails (e.g., trying to open a non-existent file).
# file = open('non_existent_file.txt', 'r')  # ⚠️ FileNotFoundError

# Example 7: AttributeError
# Occurs when an attribute reference or assignment fails.
# text = "hello"
# text.uppercase()  # ⚠️ AttributeError: 'str' object has no attribute 'uppercase'

# Example 8: ImportError
# Occurs when an import statement fails to find the module definition or can't find a name in the module.
# import non_existent_module  # ⚠️ ImportError: No module named 'non_existent_module'

# Example 9: ModuleNotFoundError
# Specific type of ImportError when the module is not found.
# import nonexistent  # ⚠️ ModuleNotFoundError: No module named 'nonexistent'

# Example 10: NameError
# Occurs when a local or global name is not found.
# print(undefined_variable)  # ⚠️ NameError: name 'undefined_variable' is not defined

# Example 11: IndentationError
# Occurs when there is incorrect indentation.
# def my_function():
# print("Hello")  # ⚠️ IndentationError: expected an indented block

# Example 12: SyntaxError
# Occurs when the parser encounters a syntax error.
# eval('x === x')  # ⚠️ SyntaxError: invalid syntax

# Example 13: NotImplementedError
# Used in abstract classes as a placeholder for methods that need to be implemented in subclasses.
def abstract_method():
    raise NotImplementedError("This method should be overridden in subclasses")
# abstract_method()  # ⚠️ NotImplementedError: This method should be overridden in subclasses

# Example 14: MemoryError
# Occurs when an operation runs out of memory.
# large_list = [1] * (10 ** 10)  # ⚠️ MemoryError

# Example 15: OverflowError
# Occurs when the result of an arithmetic operation is too large to be represented.
import math
# result = math.exp(1000)  # ⚠️ OverflowError: math range error

# Example 16: UnicodeDecodeError
# Occurs when a Unicode-related encoding or decoding error happens.
# byte_str = b'\xff'
# text = byte_str.decode('utf-8')  # ⚠️ UnicodeDecodeError

# Example 17: AssertionError
# Occurs when an assert statement fails.
# assert 2 + 2 == 5, "Math is broken!"  # ⚠️ AssertionError: Math is broken!

# Example 18: StopIteration
# Raised by built-in function next() to indicate that there are no further items.
iterator = iter([1, 2, 3])
next(iterator)  # Output: 1
next(iterator)  # Output: 2
next(iterator)  # Output: 3
# next(iterator)  # ⚠️ StopIteration

# Example 19: KeyboardInterrupt
# Raised when the user hits the interrupt key (usually Ctrl+C).
# Run an infinite loop and interrupt manually:
# while True:
#     pass  # ⚠️ KeyboardInterrupt when interrupted

# Example 20: SystemExit
# Raised by sys.exit() function.
import sys
# sys.exit("Exiting the program")  # ⚠️ SystemExit: Exiting the program

# Remember, uncaught exceptions terminate the program and print a traceback.

# 8.2 try...except Blocks: Catching and Handling Exceptions 🎣 (Error Catch Nets)

# try...except blocks allow you to handle exceptions gracefully.

# Example 1: Basic try...except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero! ⚠️")  # Output: Cannot divide by zero! ⚠️

# Example 2: Catching multiple exceptions
try:
    value = int("abc")
except (ValueError, TypeError):
    print("Invalid value or type! ⚠️")  # Output: Invalid value or type! ⚠️

# Example 3: Using else clause
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero! ⚠️")
else:
    print(f"Result is {result}")  # Output: Result is 5.0

# Example 4: Using finally clause
try:
    file = open('sample.txt', 'r')
except FileNotFoundError:
    print("File not found! ⚠️")
finally:
    print("Execution complete.")  # Output: Execution complete.

# Example 5: Catching all exceptions
try:
    result = 10 / 0
except Exception as e:
    print(f"An error occurred: {e}")  # Output: An error occurred: division by zero

# Example 6: Handling specific exception and re-raising
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Caught division by zero!")
    raise  # Re-raise the exception

# Example 7: Multiple except blocks
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid input! Please enter a number.")
except ZeroDivisionError:
    print("Cannot divide by zero!")
else:
    print(f"Result is {result}")
finally:
    print("Process complete.")

# Example 8: Nested try...except blocks
try:
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("Inner exception caught")
        raise
except ZeroDivisionError:
    print("Outer exception caught")

# Example 9: Using pass in except block
try:
    result = 10 / 0
except ZeroDivisionError:
    pass  # Do nothing, silently handle

# Example 10: Using except without specifying exception (not recommended)
try:
    result = 10 / 0
except:
    print("An error occurred (unspecified exception caught)")  # Output: An error occurred

# Example 11: Accessing exception object
try:
    int("abc")
except ValueError as e:
    print(f"ValueError occurred: {e}")  # Output: ValueError occurred: invalid literal for int() with base 10: 'abc'

# Example 12: try...finally without except
try:
    print("Try block executed")
finally:
    print("Finally block executed")  # Output: Try block executed \n Finally block executed

# Example 13: Handling multiple exceptions separately
try:
    result = 10 / 0
except ZeroDivisionError:
    print("ZeroDivisionError caught!")
except ValueError:
    print("ValueError caught!")

# Example 14: Raising exception in except block
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Division by zero occurred!")
    raise ValueError("Custom message")  # Raises ValueError after handling ZeroDivisionError

# Example 15: Using except Exception to catch all exceptions
try:
    undefined_variable += 1
except Exception as e:
    print(f"Caught an exception: {e}")  # Output: Caught an exception: name 'undefined_variable' is not defined

# Example 16: Using try...except with file operations
try:
    with open('nonexistent_file.txt', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("File not found! Please check the file name.")

# Example 17: Combining else and finally
try:
    num = int("100")
except ValueError:
    print("Conversion failed!")
else:
    print(f"Conversion successful: {num}")  # Output: Conversion successful: 100
finally:
    print("Execution ended.")

# Example 18: Avoiding bare except clauses
try:
    result = 10 / 0
except ZeroDivisionError:
    print("ZeroDivisionError caught!")
except:
    print("Some other exception occurred!")  # This will not be executed in this example

# Example 19: Using assertions
try:
    x = -1
    assert x >= 0, "x must be non-negative"
except AssertionError as e:
    print(f"AssertionError: {e}")  # Output: AssertionError: x must be non-negative

# Example 20: Exception in finally block
try:
    result = 10 / 2
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("Finally block executed.")
    # raise ValueError("Exception in finally block")  # Uncommenting raises ValueError after finally

# 8.3 Raising Exceptions: Signaling Errors Manually 🚩 (Error Flags)

# You can raise exceptions using the raise statement.

# Example 1: Raising a built-in exception
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero!")  # Manually raise an exception
    return a / b

try:
    result = divide(10, 0)
except ZeroDivisionError as e:
    print(f"Error: {e}")  # Output: Error: Cannot divide by zero!

# Example 2: Raising a ValueError
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    print(f"Age is set to {age}")

try:
    set_age(-5)
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Age cannot be negative!

# Example 3: Raising a custom exception
class NegativeNumberError(Exception):
    pass

def square_root(x):
    if x < 0:
        raise NegativeNumberError("Cannot compute square root of negative number!")
    return x ** 0.5

try:
    result = square_root(-9)
except NegativeNumberError as e:
    print(f"Error: {e}")  # Output: Error: Cannot compute square root of negative number!

# Example 4: Raising an exception with arguments
class CustomError(Exception):
    def __init__(self, message, code):
        self.message = message
        self.code = code

def function():
    raise CustomError("An error occurred!", 500)

try:
    function()
except CustomError as e:
    print(f"Error: {e.message}, Code: {e.code}")  # Output: Error: An error occurred!, Code: 500

# Example 5: Raising NotImplementedError
def abstract_method():
    raise NotImplementedError("This method should be overridden.")

try:
    abstract_method()
except NotImplementedError as e:
    print(f"Error: {e}")  # Output: Error: This method should be overridden.

# Example 6: Re-raising caught exception
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Caught division by zero!")
    raise  # Re-raise the exception

# Example 7: Raising exception based on condition
def withdraw(balance, amount):
    if amount > balance:
        raise ValueError("Insufficient funds!")
    balance -= amount
    return balance

try:
    balance = withdraw(100, 150)
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Insufficient funds!

# Example 8: Using raise without specifying exception (re-raises last exception)
try:
    int('abc')
except ValueError:
    print("Initial exception caught.")
    raise  # Re-raises ValueError

# Example 9: Raising AssertionError
def check_positive(x):
    if x <= 0:
        raise AssertionError("Value must be positive!")
    print(f"Value is {x}")

try:
    check_positive(-10)
except AssertionError as e:
    print(f"AssertionError: {e}")  # Output: AssertionError: Value must be positive!

# Example 10: Chaining exceptions with raise from
try:
    try:
        int('abc')
    except ValueError as e:
        raise TypeError("Type conversion failed!") from e
except TypeError as e:
    print(f"Chained Error: {e}")  # Output: Chained Error: Type conversion failed!

# Example 11: Raising exception in __init__
class Person:
    def __init__(self, name):
        if not name:
            raise ValueError("Name cannot be empty!")
        self.name = name

try:
    person = Person('')
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Name cannot be empty!

# Example 12: Raising exception in property setter
class Temperature:
    def __init__(self, temperature):
        self._temperature = temperature

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero!")
        self._temperature = value

try:
    temp = Temperature(-300)
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Temperature cannot be below absolute zero!

# Example 13: Raising exception with custom message
def login(username):
    if username != 'admin':
        raise PermissionError(f"User '{username}' does not have access!")
    print("Access granted.")

try:
    login('guest')
except PermissionError as e:
    print(f"Error: {e}")  # Output: Error: User 'guest' does not have access!

# Example 14: Raising exception in context manager
class MyResource:
    def __enter__(self):
        print("Resource acquired.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Resource released.")
        if exc_type:
            print(f"Exception occurred: {exc_val}")

    def use(self):
        raise RuntimeError("Error using resource!")

try:
    with MyResource() as resource:
        resource.use()
except RuntimeError as e:
    print(f"Caught Exception: {e}")
# Output:
# Resource acquired.
# Resource released.
# Exception occurred: Error using resource!
# Caught Exception: Error using resource!

# Example 15: Raising exception in loop
for i in range(5):
    if i == 3:
        raise StopIteration("Stopped at i=3")
    print(i)
# Output:
# 0
# 1
# 2
# StopIteration: Stopped at i=3

# Example 16: Raising exception in recursive function
def factorial(n):
    if n < 0:
        raise ValueError("Negative values are not allowed!")
    elif n == 0:
        return 1
    else:
        return n * factorial(n - 1)

try:
    result = factorial(-5)
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Negative values are not allowed!

# Example 17: Raising exception in lambda function
check_even = lambda x: x % 2 == 0 or (_ for _ in ()).throw(ValueError("Number is not even!"))
try:
    check_even(3)
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Number is not even!

# Example 18: Raising exception in list comprehension
try:
    squares = [x ** 2 if x >= 0 else (_ for _ in ()).throw(ValueError("Negative number!")) for x in [1, -2, 3]]
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Negative number!

# Example 19: Raising exception with additional context
try:
    raise RuntimeError("Main error message") from KeyError("Cause of the error")
except RuntimeError as e:
    print(f"Error: {e}")
    print(f"Cause: {e.__cause__}")  # Output the original exception

# Example 20: Raising exception when subclassing built-in types
class PositiveInteger(int):
    def __new__(cls, value):
        if value < 0:
            raise ValueError("Value must be positive!")
        return int.__new__(cls, value)

try:
    num = PositiveInteger(-10)
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Value must be positive!

# This concludes the detailed examples for Handling Errors and Exceptions in Python.
# Each example is designed to provide a deep understanding of exceptions,
# how to catch and handle them, and how to raise exceptions intentionally.
# Remember to handle exceptions gracefully to build robust programs! 🛡️