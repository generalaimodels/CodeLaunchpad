# Chapter 10: Decorators: Enhancing Functions 🎁 (Function Wrappers)

# 10.1 What are Decorators? Function Enhancers 🎁 (Function Wrappers)

# Concept: Understanding decorators as a way to modify or enhance functions.
# Decorators wrap a function, modifying its behavior without changing its code.

# Example 1: Basic decorator that prints messages before and after function execution.
def simple_decorator(func):
    def wrapper():
        print("Before the function execution.")
        func()  # Call the original function
        print("After the function execution.")
    return wrapper

@simple_decorator  # Applying the decorator to the function
def greet():
    print("Hello, World!")  # Original function behavior

greet()  # Calls the decorated function

# Example 2: Decorator that modifies the return value of a function.
def double_result(func):
    def wrapper():
        result = func()
        return result * 2  # Modify the result
    return wrapper

@double_result
def get_number():
    return 5

print(get_number())  # Output will be 10

# Example 3: Decorator that logs function execution.
def logger(func):
    def wrapper():
        print(f"Function '{func.__name__}' started.")
        func()
        print(f"Function '{func.__name__}' ended.")
    return wrapper

@logger
def process_data():
    print("Processing data...")

process_data()

# Example 4: Decorator that handles exceptions within a function.
def exception_handler(func):
    def wrapper():
        try:
            func()
        except Exception as e:
            print(f"An error occurred: {e}")
    return wrapper

@exception_handler
def divide_by_zero():
    result = 10 / 0  # This will raise a ZeroDivisionError

divide_by_zero()  # Error is handled by the decorator

# Example 5: Decorator that counts how many times a function is called.
def count_calls(func):
    def wrapper():
        wrapper.calls += 1  # Increment call count
        print(f"Call {wrapper.calls} of {func.__name__}")
        return func()
    wrapper.calls = 0
    return wrapper

@count_calls
def say_hi():
    print("Hi!")

say_hi()
say_hi()
say_hi()

# Example 6: Decorator that measures execution time of a function.
import time

def timer(func):
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        print(f"Executed '{func.__name__}' in {end_time - start_time} seconds.")
    return wrapper

@timer
def wait():
    time.sleep(1)

wait()

# Example 7: Decorator that only allows a function to run during daytime.
def daytime_only(func):
    def wrapper():
        current_hour = time.localtime().tm_hour
        if 6 <= current_hour < 18:
            return func()
        else:
            print("Function can only run during daytime hours.")
    return wrapper

@daytime_only
def daytime_activity():
    print("Doing daytime activity!")

daytime_activity()

# Example 8: Decorator that enforces type checking on function arguments.
def type_check(expected_type):
    def decorator(func):
        def wrapper(arg):
            if not isinstance(arg, expected_type):
                print(f"Wrong type! Expected {expected_type}.")
                return
            return func(arg)
        return wrapper
    return decorator

@type_check(int)
def square(number):
    return number * number

print(square(5))     # Correct type
print(square("5"))   # Incorrect type

# Example 9: Decorator that caches results of a function (simple memoization).
def cache(func):
    stored_results = {}
    def wrapper(n):
        if n in stored_results:
            print("Returning cached result.")
            return stored_results[n]
        result = func(n)
        stored_results[n] = result
        return result
    return wrapper

@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))  # Calculates and caches results

# Example 10: Decorator that adds authentication to a function.
def requires_authentication(func):
    def wrapper():
        authenticated = False  # Simulate authentication check
        if not authenticated:
            print("Access denied. Authentication required.")
            return
        return func()
    return wrapper

@requires_authentication
def secret_function():
    print("Secret function executed!")

secret_function()  # Access denied message

# Example 11: Decorator that repeats function execution multiple times.
def repeat_three_times(func):
    def wrapper():
        for _ in range(3):
            func()
    return wrapper

@repeat_three_times
def say_bye():
    print("Goodbye!")

say_bye()

# Example 12: Decorator that logs function arguments.
def log_arguments(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments were: {args}, {kwargs}")
        return func(*args, **kwargs)
    return wrapper

@log_arguments
def add(a, b):
    return a + b

print(add(3, 4))

# Example 13: Decorator that ensures a function only runs once.
def run_once(func):
    def wrapper():
        if not wrapper.has_run:
            wrapper.has_run = True
            return func()
    wrapper.has_run = False
    return wrapper

@run_once
def initialize():
    print("Initialization complete.")

initialize()
initialize()  # This time it won't run

# Example 14: Decorator that modifies function docstring.
def add_documentation(func):
    func.__doc__ += "\nAdditional documentation added by decorator."
    return func

@add_documentation
def documented_function():
    """Original documentation."""

print(documented_function.__doc__)

# Example 15: Decorator that checks for required permissions.
def requires_permission(permission):
    def decorator(func):
        def wrapper(*args, **kwargs):
            permissions = ["read"]  # Simulate current user permissions
            if permission not in permissions:
                print(f"Permission '{permission}' required.")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator

@requires_permission("write")
def write_data():
    print("Data written.")

write_data()

# 10.2 Decorators with Parameters (Customizable Wrappers)

# Concept: Creating decorators that can accept parameters, making them more flexible.

# Example 1: Decorator with parameter specifying the number of times to repeat a function.
def repeat(num_times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(4)  # Repeat the function 4 times
def wave(name):
    print(f"{name} waves 👋")

wave("Alice")

# Example 2: Decorator with logging level parameter.
def log(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[{level.upper()}] Executing '{func.__name__}'")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log("debug")
def multiply(a, b):
    return a * b

print(multiply(2, 3))

# Example 3: Decorator that delays function execution by a given amount of time.
def delay(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Waiting for {seconds} seconds...")
            time.sleep(seconds)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@delay(2)  # Delay execution by 2 seconds
def say_something():
    print("Delayed execution.")

say_something()

# Example 4: Decorator that takes user roles as parameters for access control.
def requires_role(role):
    def decorator(func):
        def wrapper(*args, **kwargs):
            user_role = "user"  # Simulate current user's role
            if user_role != role:
                print(f"Access denied. '{role}' role required.")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator

@requires_role("admin")
def access_admin_panel():
    print("Accessing admin panel.")

access_admin_panel()

# Example 5: Decorator with message parameter.
def custom_message(message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@custom_message("Welcome to the system!")
def login():
    print("User logged in.")

login()

# Example 6: Decorator that retries function execution upon failure.
def retry_on_failure(max_retries):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    retries += 1
                    print(f"Retry {retries}/{max_retries}")
            print("Max retries exceeded.")
        return wrapper
    return decorator

@retry_on_failure(3)
def unstable_function():
    import random
    if random.choice([True, False]):
        print("Function succeeded.")
    else:
        raise ValueError("Random failure.")

unstable_function()

# Example 7: Decorator enforcing maximum execution time.
def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import threading

            result = [None]
            def target():
                result[0] = func(*args, **kwargs)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                print("Function execution timed out.")
                return
            return result[0]
        return wrapper
    return decorator

@timeout(2)  # Function must complete within 2 seconds
def long_running_task():
    time.sleep(3)
    print("Task completed.")

long_running_task()

# Example 8: Decorator that limits function calls.
def call_limit(limit):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if wrapper.call_count >= limit:
                print("Call limit exceeded.")
                return
            wrapper.call_count += 1
            return func(*args, **kwargs)
        wrapper.call_count = 0
        return wrapper
    return decorator

@call_limit(2)
def limited_function():
    print("Function called.")

limited_function()
limited_function()
limited_function()  # Will not execute

# Example 9: Decorator that logs function execution to a file.
def log_to_file(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with open(filename, "a") as file:
                file.write(f"Function '{func.__name__}' executed.\n")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log_to_file("function_log.txt")
def perform_action():
    print("Action performed.")

perform_action()

# Example 10: Decorator that accepts multiple parameters.
def tag_wrapper(tag, css_class):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return f"<{tag} class='{css_class}'>{func(*args, **kwargs)}</{tag}>"
        return wrapper
    return decorator

@tag_wrapper("div", "container")
def get_content():
    return "Content inside div."

print(get_content())

# Example 11: Decorator that modifies function behavior based on parameter.
def conditional_decorator(active=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if active:
                print("Decorator is active.")
                return func(*args, **kwargs)
            else:
                print("Decorator is inactive.")
                return func(*args, **kwargs)
        return wrapper
    return decorator

@conditional_decorator(active=False)
def sample_function():
    print("Function executed.")

sample_function()

# Example 12: Decorator that enforces argument constraints.
def enforce_range(min_value, max_value):
    def decorator(func):
        def wrapper(value):
            if not (min_value <= value <= max_value):
                print(f"Value must be between {min_value} and {max_value}.")
                return
            return func(value)
        return wrapper
    return decorator

@enforce_range(1, 10)
def process_number(n):
    print(f"Processing number: {n}")

process_number(5)
process_number(15)  # Out of range

# Example 13: Decorator that formats function output.
def format_output(fmt):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return fmt.format(result)
        return wrapper
    return decorator

@format_output("Result is: {:.2f}")
def calculate_total(a, b):
    return a + b

print(calculate_total(5.678, 3.456))

# Example 14: Decorator that adds headers and footers to strings.
def add_header_footer(header, footer):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return f"{header}\n{func(*args, **kwargs)}\n{footer}"
        return wrapper
    return decorator

@add_header_footer("=== Start ===", "=== End ===")
def generate_report():
    return "Report content."

print(generate_report())

# Example 15: Decorator for input validation using regular expressions.
import re

def validate_input(pattern):
    def decorator(func):
        def wrapper(input_string):
            if not re.match(pattern, input_string):
                print("Invalid input format.")
                return
            return func(input_string)
        return wrapper
    return decorator

@validate_input(r"^\w+@\w+\.\w+$")  # Simple email pattern
def process_email(email):
    print(f"Processing email: {email}")

process_email("test@example.com")
process_email("invalid-email")  # Invalid format

# 10.3 Chaining Decorators (Layered Enhancements)

# Concept: Applying multiple decorators to a single function, creating layered enhancements.

# Example 1: Chaining two decorators.
def uppercase_decorator(func):
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

def split_decorator(func):
    def wrapper():
        result = func()
        return result.split()
    return wrapper

@split_decorator
@uppercase_decorator
def get_text():
    return "hello world"

print(get_text())  # Output: ['HELLO', 'WORLD']

# Example 2: Chaining decorators that check permissions and log actions.
def check_permission(func):
    def wrapper(*args, **kwargs):
        user_has_permission = True  # Simulated permission check
        if not user_has_permission:
            print("Permission denied.")
            return
        return func(*args, **kwargs)
    return wrapper

def log_action(func):
    def wrapper(*args, **kwargs):
        print(f"Action '{func.__name__}' is being executed.")
        return func(*args, **kwargs)
    return wrapper

@check_permission
@log_action
def delete_file():
    print("File deleted.")

delete_file()

# Example 3: Chaining decorators that format text.
def bold(func):
    def wrapper():
        return f"<b>{func()}</b>"
    return wrapper

def italic(func):
    def wrapper():
        return f"<i>{func()}</i>"
    return wrapper

def underline(func):
    def wrapper():
        return f"<u>{func()}</u>"
    return wrapper

@bold
@italic
@underline
def formatted_text():
    return "Decorated Text"

print(formatted_text())  # Output: <b><i><u>Decorated Text</u></i></b>

# Example 4: Chaining decorators with parameters.
def prefix(prefix_str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return prefix_str + func(*args, **kwargs)
        return wrapper
    return decorator

def suffix(suffix_str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs) + suffix_str
        return wrapper
    return decorator

@prefix("<<")
@suffix(">>")
def get_name():
    return "John Doe"

print(get_name())  # Output: <<John Doe>>

# Example 5: Mixing class-based and function-based decorators.
class CapsDecorator:
    def __init__(self, func):
        self.func = func
    def __call__(self):
        result = self.func()
        return result.capitalize()

def exclaim(func):
    def wrapper():
        return func() + "!"
    return wrapper

@CapsDecorator
@exclaim
def greet_person():
    return "hello"

print(greet_person())  # Output: Hello!

# Example 6: Chaining decorators that add logging and timing.
def log(func):
    def wrapper(*args, **kwargs):
        print(f"Executing '{func.__name__}'")
        return func(*args, **kwargs)
    return wrapper

def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start} seconds")
        return result
    return wrapper

@log
@time_it
def compute():
    time.sleep(1)
    return "Done"

print(compute())

# Example 7: Chaining decorators to enforce constraints and log activity.
def positive_input(func):
    def wrapper(n):
        if n < 0:
            print("Negative input not allowed.")
            return
        return func(n)
    return wrapper

def log_input(func):
    def wrapper(n):
        print(f"Input value: {n}")
        return func(n)
    return wrapper

@log_input
@positive_input
def square(n):
    return n * n

print(square(5))
print(square(-3))  # Negative input

# Example 8: Chaining decorators that authenticate and log function usage.
def authenticate(func):
    def wrapper(*args, **kwargs):
        user_authenticated = False  # Simulation
        if not user_authenticated:
            print("User not authenticated.")
            return
        return func(*args, **kwargs)
    return wrapper

def audit(func):
    def wrapper(*args, **kwargs):
        print(f"Audit log: '{func.__name__}' was called.")
        return func(*args, **kwargs)
    return wrapper

@authenticate
@audit
def sensitive_operation():
    print("Sensitive operation performed.")

sensitive_operation()

# Example 9: Chaining decorators that modify function inputs and outputs.
def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper

def add_ten(func):
    def wrapper(a, b):
        result = func(a, b)
        return result + 10
    return wrapper

@add_ten
@double_args
def add(a, b):
    return a + b

print(add(5, 3))  # ((5*2 + 3*2) + 10) = (10+6)+10=26

# Example 10: Chaining decorators with error handling and logging.
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
    return wrapper

def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling '{func.__name__}'")
        return func(*args, **kwargs)
    return wrapper

@error_handler
@logger
def divide(a, b):
    return a / b

print(divide(10, 2))
print(divide(10, 0))  # Will handle ZeroDivisionError

# Example 11: Chaining decorators to cache results and trace function calls.
def trace(func):
    def wrapper(*args, **kwargs):
        print(f"Trace: Called '{func.__name__}' with args={args}, kwargs={kwargs}")
        return func(*args, **kwargs)
    return wrapper

def memoize(func):
    cache = {}
    def wrapper(*args):
        if args in cache:
            print("Returning cached result.")
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize
@trace
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
print(factorial(5))  # Cached result

# Example 12: Chaining decorators with arguments.
def decorator_one(arg1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Decorator one argument: {arg1}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def decorator_two(arg2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Decorator two argument: {arg2}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@decorator_one("First")
@decorator_two("Second")
def function():
    print("Function executed.")

function()

# Example 13: Chaining decorators to enhance method functions in a class.
def method_decorator_one(func):
    def wrapper(self, *args, **kwargs):
        print("Method decorator one.")
        return func(self, *args, **kwargs)
    return wrapper

def method_decorator_two(func):
    def wrapper(self, *args, **kwargs):
        print("Method decorator two.")
        return func(self, *args, **kwargs)
    return wrapper

class MyClass:
    @method_decorator_one
    @method_decorator_two
    def my_method(self):
        print("Method in class.")

obj = MyClass()
obj.my_method()

# Example 14: Chaining decorators to add headers, footers, and format content.
def header(func):
    def wrapper(*args, **kwargs):
        return f"---HEADER---\n{func(*args, **kwargs)}"
    return wrapper

def footer(func):
    def wrapper(*args, **kwargs):
        return f"{func(*args, **kwargs)}\n---FOOTER---"
    return wrapper

def make_bold(func):
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

@footer
@header
@make_bold
def display_message():
    return "Decorated message."

print(display_message())

# Example 15: Chaining decorators to enforce constraints and process results.
def enforce_non_negative(func):
    def wrapper(n):
        if n < 0:
            print("Negative input not allowed.")
            return
        return func(n)
    return wrapper

def square_result(func):
    def wrapper(n):
        result = func(n)
        return result * result
    return wrapper

@enforce_non_negative
@square_result
def increment(n):
    return n + 1

print(increment(3))   # (3+1)^2 = 16
print(increment(-2))  # Negative input not allowed