# Chapter 8: Handling Errors: Exceptions ⚠️ (Dealing with the Unexpected)

# 8.1 What are Exceptions? Runtime Errors ⚠️ (Unexpected Events)

# Example 1: ZeroDivisionError
try:
    result = 10 / 0  # ❌ Division by zero
except ZeroDivisionError as e:
    print(f"Error: {e}")  # Outputs: division by zero

# Example 2: TypeError
try:
    result = 'hello' + 5  # ❌ Cannot add string and integer
except TypeError as e:
    print(f"Error: {e}")  # Outputs: can only concatenate str (not "int") to str

# Example 3: FileNotFoundError
try:
    with open('nonexistent_file.txt', 'r') as file:
        content = file.read()
except FileNotFoundError as e:
    print(f"Error: {e}")  # Outputs: [Errno 2] No such file or directory: 'nonexistent_file.txt'

# Example 4: ValueError
try:
    number = int('abc')  # ❌ Cannot convert string to integer
except ValueError as e:
    print(f"Error: {e}")  # Outputs: invalid literal for int() with base 10: 'abc'

# Example 5: IndexError
my_list = [1, 2, 3]
try:
    item = my_list[5]  # ❌ Index out of range
except IndexError as e:
    print(f"Error: {e}")  # Outputs: list index out of range

# Example 6: KeyError
my_dict = {'a': 1, 'b': 2}
try:
    value = my_dict['c']  # ❌ Key does not exist
except KeyError as e:
    print(f"Error: {e}")  # Outputs: 'c'

# Example 7: AttributeError
class MyClass:
    pass

obj = MyClass()
try:
    obj.attribute  # ❌ Attribute does not exist
except AttributeError as e:
    print(f"Error: {e}")  # Outputs: 'MyClass' object has no attribute 'attribute'

# Example 8: ImportError
try:
    import nonexistent_module  # ❌ Module does not exist
except ImportError as e:
    print(f"Error: {e}")  # Outputs: No module named 'nonexistent_module'

# Example 9: StopIteration
iterator = iter([1, 2, 3])
try:
    next(iterator)
    next(iterator)
    next(iterator)
    next(iterator)  # ❌ No more items
except StopIteration as e:
    print("Iterator has no more items.")  # Outputs message when iteration ends

# Example 10: OverflowError
try:
    result = 2 ** 10000  # ❌ Number too large
except OverflowError as e:
    print(f"Error: {e}")  # Outputs: (34, 'Result too large')

# Example 11: MemoryError
try:
    big_list = [1] * (10 ** 10)  # ⚠️ May cause MemoryError
except MemoryError as e:
    print("Memory Error: Unable to allocate sufficient memory.")  # Outputs memory error message

# Example 12: NameError
try:
    print(undefined_variable)  # ❌ Variable not defined
except NameError as e:
    print(f"Error: {e}")  # Outputs: name 'undefined_variable' is not defined

# Example 13: UnicodeEncodeError
try:
    'ß'.encode('ascii')  # ❌ Cannot encode character to ASCII
except UnicodeEncodeError as e:
    print(f"Error: {e}")  # Outputs encoding error message

# Example 14: SyntaxError
# Note: SyntaxError cannot be caught in a try-except block within the same code
# 📝 SyntaxError occurs during compilation, not at runtime

# Example 15: IndentationError
# Note: IndentationError occurs when incorrect indentation is used
# 📝 Like SyntaxError, it cannot be demonstrated in running code

# 8.2 try...except Blocks: Catching and Handling Exceptions 🎣 (Error Catch Nets)

# Example 1: Basic try-except
try:
    result = 10 / 0  # ❌ Division by zero
except ZeroDivisionError:
    print("Cannot divide by zero!")  # Handles the exception

# Example 2: Multiple except blocks
try:
    value = int('abc')  # ❌ ValueError
except ZeroDivisionError:
    print("Division by zero!")
except ValueError:
    print("Invalid integer!")  # Outputs: Invalid integer!

# Example 3: Catching multiple exceptions in one except block
try:
    result = 10 / 'a'  # ❌ TypeError
except (ZeroDivisionError, TypeError):
    print("An error occurred: Division or Type error")  # Outputs message for either exception

# Example 4: Using else with try-except
try:
    result = 10 / 2  # ✅ No exception
except ZeroDivisionError:
    print("Cannot divide by zero!")
else:
    print(f"Result is {result}")  # Outputs: Result is 5.0

# Example 5: Using finally block
try:
    file = open('sample.txt', 'r')
except FileNotFoundError:
    print("File not found.")
finally:
    print("Execution completed.")  # Always executes

# Example 6: Nested try-except
try:
    try:
        num = int('abc')  # ❌ ValueError
    except ValueError:
        print("Inner exception caught.")
        raise  # Re-raises the exception
except ValueError:
    print("Outer exception caught.")  # Handles re-raised exception

# Example 7: Exception with arguments
try:
    raise Exception("Custom error message")  # 🚩 Raising general exception
except Exception as e:
    print(f"Error: {e}")  # Outputs: Error: Custom error message

# Example 8: Catching all exceptions
try:
    result = 10 / 0  # ❌ ZeroDivisionError
except Exception as e:
    print(f"An exception occurred: {e}")  # Generic exception handling

# Example 9: Accessing exception attributes
try:
    value = int('abc')  # ❌ ValueError
except ValueError as e:
    print(f"Error args: {e.args}")  # Outputs: ('invalid literal for int() with base 10: \'abc\'',)

# Example 10: Using pass in except block
try:
    result = 10 / 0  # ❌ ZeroDivisionError
except ZeroDivisionError:
    pass  # 🚫 Does nothing, program continues silently

# Example 11: Finally without except
try:
    print("Trying...")
finally:
    print("Finally block executed.")  # Always executes

# Example 12: try-except-else-finally
try:
    print("Trying...")
    result = 10 / 2  # ✅ No exception
except ZeroDivisionError:
    print("Cannot divide by zero!")
else:
    print(f"Result: {result}")  # Outputs: Result: 5.0
finally:
    print("Cleaning up...")  # Always executes

# Example 13: Raising exception in except block
try:
    result = 10 / 0  # ❌ ZeroDivisionError
except ZeroDivisionError:
    print("Caught division by zero.")
    raise ValueError("Invalid division operation.")  # Raising different exception

# Example 14: Handling multiple exceptions separately
try:
    value = int('abc')  # ❌ ValueError
except TypeError:
    print("Type Error occurred.")
except ValueError:
    print("Value Error occurred.")  # Outputs: Value Error occurred.

# Example 15: Except block without exception type (not recommended)
try:
    result = 10 / 0  # ❌ ZeroDivisionError
except:
    print("An error occurred.")  # Catches any exception (not specific)

# 8.3 Raising Exceptions: Signaling Errors Manually 🚩 (Error Flags)

# Example 1: Raising a general exception
def check_positive(number):
    if number < 0:
        raise Exception("Number must be positive.")  # 🚩 Raising exception
    else:
        print(f"Number is {number}")

try:
    check_positive(-5)
except Exception as e:
    print(f"Error: {e}")  # Outputs: Error: Number must be positive.

# Example 2: Raising a TypeError
def add_numbers(a, b):
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers.")  # 🚩
    return a + b

try:
    result = add_numbers(5, '5')
except TypeError as e:
    print(f"Error: {e}")  # Outputs: Error: Both arguments must be integers.

# Example 3: Raising a ValueError with custom message
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative.")  # 🚩
    print(f"Age set to {age}")

try:
    set_age(-1)
except ValueError as e:
    print(f"Error: {e}")  # Outputs: Error: Age cannot be negative.

# Example 4: Custom exception class
class InvalidGradeError(Exception):
    # 📝 Custom exception for invalid grades
    pass

def set_grade(grade):
    if grade > 100:
        raise InvalidGradeError("Grade cannot exceed 100.")  # 🚩
    print(f"Grade set to {grade}")

try:
    set_grade(105)
except InvalidGradeError as e:
    print(f"Error: {e}")  # Outputs: Error: Grade cannot exceed 100.

# Example 5: Chaining exceptions
try:
    try:
        value = int('abc')  # ❌ ValueError
    except ValueError as e:
        raise TypeError("Conversion error.") from e  # Chaining exceptions
except TypeError as e:
    print(f"Error: {e}")  # Outputs: Error: Conversion error.
    print(f"Original exception: {e.__cause__}")  # Original exception details

# Example 6: Re-raising an exception
try:
    try:
        result = 10 / 0  # ❌ ZeroDivisionError
    except ZeroDivisionError:
        print("Caught division by zero.")
        raise  # Re-raising the exception
except ZeroDivisionError:
    print("Exception re-raised and caught again.")  # Handles re-raised exception

# Example 7: Using assert to raise AssertionError
def calculate_average(scores):
    assert len(scores) > 0, "List of scores cannot be empty."  # 🚩
    return sum(scores) / len(scores)

try:
    avg = calculate_average([])
except AssertionError as e:
    print(f"Error: {e}")  # Outputs: Error: List of scores cannot be empty.

# Example 8: Raising NotImplementedError
class BaseClass:
    def method(self):
        raise NotImplementedError("Subclass must implement this abstract method.")  # 🚩

class SubClass(BaseClass):
    pass

try:
    obj = SubClass()
    obj.method()
except NotImplementedError as e:
    print(f"Error: {e}")  # Outputs: Error: Subclass must implement this abstract method.

# Example 9: Raising KeyboardInterrupt (rarely done manually)
# Note: KeyboardInterrupt is usually raised when the user interrupts the program (e.g., Ctrl+C)
# Raising it manually is unconventional

# Example 10: Raising FileNotFoundError
def read_config(file_name):
    raise FileNotFoundError(f"Configuration file '{file_name}' not found.")  # 🚩

try:
    read_config('config.ini')
except FileNotFoundError as e:
    print(f"Error: {e}")  # Outputs error message for missing file

# Example 11: Custom exception with additional attributes
class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        super().__init__(f"Insufficient funds: Balance is {balance}, attempted withdrawal {amount}")
        self.balance = balance
        self.amount = amount

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)  # 🚩
    balance -= amount
    return balance

try:
    new_balance = withdraw(100, 150)
except InsufficientFundsError as e:
    print(f"Error: {e}")  # Outputs detailed error message

# Example 12: Raising StopIteration manually
def limited_generator():
    yield 1
    yield 2
    raise StopIteration("No more items.")  # 🚩

gen = limited_generator()
try:
    print(next(gen))  # Outputs: 1
    print(next(gen))  # Outputs: 2
    print(next(gen))  # Raises StopIteration
except StopIteration as e:
    print(f"Generator stopped: {e}")

# Example 13: Raising KeyboardInterrupt (not recommended)
# try:
#     raise KeyboardInterrupt("Interrupting program.")  # 🚩
# except KeyboardInterrupt as e:
#     print(f"Program interrupted: {e}")

# Example 14: Raising SystemExit to exit the program
# try:
#     raise SystemExit("Exiting program.")  # 🚩
# except SystemExit as e:
#     print(f"SystemExit: {e}")

# Example 15: Raising EOFError
def get_input():
    raise EOFError("No more input.")  # 🚩

try:
    get_input()
except EOFError as e:
    print(f"Error: {e}")  # Outputs: Error: No more input.