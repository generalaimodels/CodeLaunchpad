# Chapter 10: Decorators: Enhancing Functions 🎁 (Function Wrappers)

# 10.1 What are Decorators? Function Enhancers 🎁 (Function Wrappers)

# Example 1: Basic decorator that prints before and after a function
def my_decorator(func):
    def wrapper():
        print("Before the function execution")  # 🎁 Before
        func()
        print("After the function execution")   # 🎁 After
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")  # 👋

say_hello()  # Calls the decorated function

# Example 2: Decorating a function without using @ syntax
def my_decorator(func):
    def wrapper():
        print("Decorator is adding functionality")  # 🎁
        func()
    return wrapper

def greet():
    print("Greetings!")  # 🙌

greet = my_decorator(greet)  # Manually decorating
greet()

# Example 3: Passing arguments to the function being decorated
def my_decorator(func):
    def wrapper(name):
        print(f"Adding decoration for {name}")  # 🎁
        func(name)
    return wrapper

@my_decorator
def say_hi(name):
    print(f"Hi, {name}!")  # 😊

say_hi("Alice")

# Example 4: Preserving function metadata using functools.wraps
import functools

def my_decorator(func):
    @functools.wraps(func)  # Preserves metadata
    def wrapper(*args, **kwargs):
        """Wrapper function"""
        print("Function is being called")  # 🎁
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def add(a, b):
    """Adds two numbers"""
    return a + b  # ➕

result = add(3, 5)
print(f"Result is {result}")
print(f"Function name: {add.__name__}")  # Correct function name
print(f"Docstring: {add.__doc__}")       # Correct docstring

# Example 5: Decorating functions with arbitrary arguments (*args, **kwargs)
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Arguments were:", args, kwargs)  # 🎁
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def multiply(a, b):
    return a * b  # ✖️

product = multiply(4, 5)
print(f"Product is {product}")

# Example 6: A logging decorator
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Logging: Calling function {func.__name__}")  # 📝
        return func(*args, **kwargs)
    return wrapper

@logger
def divide(a, b):
    return a / b  # ➗

result = divide(10, 2)
print(f"Division result: {result}")

# Example 7: Timing function execution
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time} seconds")  # ⏰
        return result
    return wrapper

@timer
def compute_square(n):
    return [i*i for i in range(n)]  # 🧮

squares = compute_square(10000)

# Example 8: Access control decorator
def require_permission(func):
    def wrapper(*args, **kwargs):
        user_has_permission = False  # 🚫
        if user_has_permission:
            return func(*args, **kwargs)
        else:
            print("Permission denied")  # ❌
    return wrapper

@require_permission
def delete_file(filename):
    print(f"{filename} has been deleted")  # 🗑️

delete_file("important_file.txt")

# Example 9: Input validation decorator
def validate_input(func):
    def wrapper(x):
        if x < 0:
            raise ValueError("Negative value not allowed!")  # ⚠️
        return func(x)
    return wrapper

@validate_input
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)  # 🔁

try:
    print(factorial(5))
    print(factorial(-1))  # This will raise an exception
except ValueError as e:
    print(e)

# Example 10: Memoization decorator (caching results)
def memoize(func):
    cache = {}  # 🗄️
    def wrapper(n):
        if n in cache:
            return cache[n]
        result = func(n)
        cache[n] = result
        return result
    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)  # 🔁

print(fibonacci(10))  # Cached computations

# Example 11: Counting function calls
def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"Call {wrapper.calls} of {func.__name__}")  # 🔢
        return func(*args, **kwargs)
    wrapper.calls = 0  # Initialize count
    return wrapper

@count_calls
def greet(name):
    print(f"Hello, {name}!")  # 👋

greet("Alice")
greet("Bob")
greet("Charlie")

# Example 12: Debugging decorator
def debug(func):
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]               # 🛠️
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")    # 📋
        return value
    return wrapper

@debug
def say_full_name(first_name, last_name):
    full_name = f"{first_name} {last_name}"
    print(f"Full name: {full_name}")  # 😊
    return full_name

full_name = say_full_name("John", last_name="Doe")

# Example 13: Decorating methods in a class
def method_decorator(method):
    def wrapper(self, *args, **kwargs):
        print(f"Method {method.__name__} called")  # 🏷️
        return method(self, *args, **kwargs)
    return wrapper

class MyClass:
    @method_decorator
    def instance_method(self):
        print("This is an instance method")  # 👤

obj = MyClass()
obj.instance_method()

# Example 14: Decorator for singleton pattern
def singleton(cls):
    instances = {}  # 📦
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
            print(f"Created new instance of {cls.__name__}")  # 🆕
        else:
            print(f"Using existing instance of {cls.__name__}")  # ♻️
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Initializing database connection")  # 🔌

db1 = DatabaseConnection()
db2 = DatabaseConnection()

# Example 15: Decorator that modifies the return value
def make_uppercase(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        uppercase_result = result.upper()
        print(f"Converted to uppercase: {uppercase_result}")  # 🔠
        return uppercase_result
    return wrapper

@make_uppercase
def get_message():
    return "Hello, world!"  # 🌍

message = get_message()

# 10.2 Decorators with Parameters (Customizable Wrappers)

# Example 1: Decorator with parameter to repeat function execution
def repeat(num_times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)  # Repeats the function 3 times
def greet(name):
    print(f"Hello, {name}!")  # 👋

greet("Alice")

# Example 2: Decorator with parameter to set logging level
def log(level):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[{level.upper()}] {func.__name__} is called")  # 📝
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log("debug")  # Sets the logging level to DEBUG
def process_data(data):
    print(f"Processing data: {data}")  # 💾

process_data("Sample Data")

# Example 3: Decorator with parameters for authentication roles
def require_role(role):
    def decorator(func):
        def wrapper(user_role, *args, **kwargs):
            if user_role == role:
                return func(*args, **kwargs)
            else:
                print(f"Access denied for role: {user_role}")  # 🚫
        return wrapper
    return decorator

@require_role("admin")
def delete_user(user_id):
    print(f"User {user_id} deleted")  # 🗑️

delete_user("user", 123)  # Access denied
delete_user("admin", 456)  # User deleted

# Example 4: Decorator with parameters to enforce type checking
def type_check(expected_type):
    def decorator(func):
        def wrapper(arg):
            if not isinstance(arg, expected_type):
                raise TypeError(f"Expected type {expected_type.__name__}")  # ⚠️
            return func(arg)
        return wrapper
    return decorator

@type_check(int)
def square(n):
    return n * n  # 🔢

try:
    print(square(5))
    print(square("5"))  # Raises TypeError
except TypeError as e:
    print(e)

# Example 5: Decorator with optional parameters
def greet(greeting="Hello"):
    def decorator(func):
        def wrapper(name):
            print(f"{greeting}, {name}!")  # 👋
            return func(name)
        return wrapper
    return decorator

@greet("Hi")
def say_name(name):
    print(f"Your name is {name}")  # 😊

say_name("Alice")

# Example 6: Decorator with keyword-only arguments
def tag(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            tagged_result = f"<{name}>{result}</{name}>"  # 🏷️
            return tagged_result
        return wrapper
    return decorator

@tag(name="div")
def get_text():
    return "This is some text"  # 📝

print(get_text())

# Example 7: Decorator with multiple parameters
def authenticate(user, password):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if user == "admin" and password == "secret":
                print("Authentication successful")  # 🔑
                return func(*args, **kwargs)
            else:
                print("Authentication failed")  # 🚫
        return wrapper
    return decorator

@authenticate(user="admin", password="secret")
def protected_resource():
    print("Accessing protected resource")  # 🔐

protected_resource()

# Example 8: Parameterized decorator with functools.wraps
import functools

def repeat(num_times):
    def decorator(func):
        @functools.wraps(func)  # Preserving metadata
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(2)
def greet():
    """Greet function"""
    print("Hello!")  # 👋

greet()
print(f"Function name: {greet.__name__}")  # Correct name
print(f"Docstring: {greet.__doc__}")       # Correct docstring

# Example 9: Decorator with regular function parameters
def emphasize(func=None, *, style="bold"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if style == "bold":
                decorated_result = f"<b>{result}</b>"  # 🖋️
            elif style == "italic":
                decorated_result = f"<i>{result}</i>"
            else:
                decorated_result = result
            return decorated_result
        return wrapper
    if func:
        return decorator(func)
    return decorator

@emphasize(style="italic")
def get_quote():
    return "To be or not to be"  # 🎭

print(get_quote())

# Example 10: Decorator with parameters that take functions as arguments
def apply_operation(operation):
    def decorator(func):
        def wrapper(x, y):
            result = func(x, y)
            return operation(result)  # 🧮
        return wrapper
    return decorator

import math

@apply_operation(math.sqrt)
def add(a, b):
    return a + b  # ➕

print(add(9, 16))  # Applies sqrt to the sum

# Example 11: Creating a decorator for retrying failed operations
def retry(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt +1} failed: {e}")  # 🔄
            print("All attempts failed")
        return wrapper
    return decorator

@retry(3)
def fragile_operation():
    import random
    if random.choice([True, False]):
        print("Operation succeeded")  # ✅
    else:
        raise ValueError("Operation failed")  # ❌

fragile_operation()

# Example 12: Decorator with dynamic parameters
def conditional_decorator(condition):
    def decorator(func):
        if not condition:
            return func  # No decoration needed
        def wrapper(*args, **kwargs):
            print("Function is decorated")  # 🎁
            return func(*args, **kwargs)
        return wrapper
    return decorator

@conditional_decorator(condition=True)
def foo():
    print("Executing foo")  # 🚀

foo()

# Example 13: Using class methods as decorators with parameters
class Repeat:
    def __init__(self, times):
        self.times = times  # 🔢
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            for _ in range(self.times):
                func(*args, **kwargs)
        return wrapper

@Repeat(times=4)
def greet(name):
    print(f"Hello, {name}!")  # 👋

greet("Bob")

# Example 14: Decorator with parameters that cache results
def cache(max_size=100):
    def decorator(func):
        cached = {}
        def wrapper(*args):
            if args in cached:
                print("Returning cached result")  # 🗄️
                return cached[args]
            result = func(*args)
            if len(cached) >= max_size:
                cached.popitem()
            cached[args] = result
            return result
        return wrapper
    return decorator

@cache(max_size=2)
def compute_square(n):
    print(f"Computing square of {n}")  # 📐
    return n * n

compute_square(2)
compute_square(3)
compute_square(2)  # Returns cached result

# Example 15: Decorator with parameter to customize exception handling
def handle_exceptions(handler):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler(e)  # 🎣
        return wrapper
    return decorator

def log_exception(e):
    print(f"Exception occurred: {e}")  # 📝

@handle_exceptions(log_exception)
def risky_operation():
    raise ValueError("An error happened")  # ❗️

risky_operation()

# 10.3 Chaining Decorators (Layered Enhancements)

# Example 1: Chaining two simple decorators
def decorator_one(func):
    def wrapper(*args, **kwargs):
        print("Decorator One Before")  # 🎁1
        result = func(*args, **kwargs)
        print("Decorator One After")   # 🎁1
        return result
    return wrapper

def decorator_two(func):
    def wrapper(*args, **kwargs):
        print("Decorator Two Before")  # 🎁2
        result = func(*args, **kwargs)
        print("Decorator Two After")   # 🎁2
        return result
    return wrapper

@decorator_one
@decorator_two
def say_hello():
    print("Hello!")  # 👋

say_hello()

# Example 2: Order of decorator application
def uppercase_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()  # 🔠
    return wrapper

def remove_spaces_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.replace(" ", "")  # 🔤
    return wrapper

@remove_spaces_decorator
@uppercase_decorator
def get_text():
    return "Hello World"  # 🌍

print(get_text())  # Output: HELLOWORLD

# Example 3: Chaining decorators with parameters
def star_decorator(func):
    def wrapper(*args, **kwargs):
        print("*" * 30)  # ⭐
        func(*args, **kwargs)
        print("*" * 30)
    return wrapper

def hash_decorator(func):
    def wrapper(*args, **kwargs):
        print("#" * 30)  # #
        func(*args, **kwargs)
        print("#" * 30)
    return wrapper

@star_decorator
@hash_decorator
def printed_message(msg):
    print(msg)  # 📢

printed_message("Chained Decorators!")

# Example 4: Combining class and function decorators
def function_decorator(func):
    def wrapper(*args, **kwargs):
        print("Function decorator")  # 🎀
        return func(*args, **kwargs)
    return wrapper

class ClassDecorator:
    def __init__(self, func):
        self.func = func  # 🧩
    def __call__(self, *args, **kwargs):
        print("Class decorator")  # 🏷️
        return self.func(*args, **kwargs)

@function_decorator
@ClassDecorator
def say_goodbye():
    print("Goodbye!")  # 👋

say_goodbye()

# Example 5: Chaining decorators that modify function arguments
def add_decorator(func):
    def wrapper(a, b):
        return func(a + 1, b + 1)  # ➕
    return wrapper

def multiply_decorator(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)  # ✖️
    return wrapper

@add_decorator
@multiply_decorator
def display_numbers(a, b):
    print(f"a = {a}, b = {b}")  # 🔢

display_numbers(1, 2)

# Example 6: Chaining decorators that handle exceptions
def exception_handler_one(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError:
            print("Handled ValueError")  # 🎣1
    return wrapper

def exception_handler_two(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            print("Handled ZeroDivisionError")  # 🎣2
    return wrapper

@exception_handler_one
@exception_handler_two
def risky_calc(a, b):
    return a / b  # ➗

risky_calc(10, 0)  # ZeroDivisionError handled
risky_calc("10", 2)  # ValueError handled

# Example 7: Chaining decorators affecting return values
def square_result(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result * result  # 🔳
    return wrapper

def increment_result(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result + 1  # ➕
    return wrapper

@increment_result
@square_result
def get_number():
    return 2  # 🔢

print(get_number())  # ((2)^2) +1 = 5

# Example 8: Chaining decorators with functools.wraps
import functools

def decorator_a(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator A")  # 🅰️
        return func(*args, **kwargs)
    return wrapper

def decorator_b(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator B")  # 🅱️
        return func(*args, **kwargs)
    return wrapper

@decorator_a
@decorator_b
def function():
    """This is the function's docstring"""  # 📝
    print("Function is called")  # 📞

function()
print(f"Function name: {function.__name__}")
print(f"Docstring: {function.__doc__}")

# Example 9: Chaining decorators on class methods
def decorator_one(func):
    def wrapper(*args, **kwargs):
        print("Decorator One")  # 1️⃣
        return func(*args, **kwargs)
    return wrapper

def decorator_two(func):
    def wrapper(*args, **kwargs):
        print("Decorator Two")  # 2️⃣
        return func(*args, **kwargs)
    return wrapper

class MyClass:
    @decorator_one
    @decorator_two
    def method(self):
        print("Method called")  # 🔧

obj = MyClass()
obj.method()

# Example 10: Chaining decorators for login required and admin required
def login_required(func):
    def wrapper(user, *args, **kwargs):
        if user["is_authenticated"]:
            return func(user, *args, **kwargs)
        else:
            print("Login required")  # 🔑
    return wrapper

def admin_required(func):
    def wrapper(user, *args, **kwargs):
        if user["is_admin"]:
            return func(user, *args, **kwargs)
        else:
            print("Admin access required")  # 👮
    return wrapper

@login_required
@admin_required
def perform_admin_task(user):
    print("Admin task performed")  # 🛠️

user = {"is_authenticated": True, "is_admin": False}
perform_admin_task(user)  # Should print "Admin access required"

user["is_admin"] = True
perform_admin_task(user)  # Should perform admin task

# Example 11: Chaining decorators affecting flow control
def decorator_a(func):
    def wrapper(*args, **kwargs):
        print("Start A")  # 🅰️
        result = func(*args, **kwargs)
        print("End A")
        return result
    return wrapper

def decorator_b(func):
    def wrapper(*args, **kwargs):
        print("Start B")  # 🅱️
        result = func(*args, **kwargs)
        print("End B")
        return result
    return wrapper

@decorator_a
@decorator_b
def compute():
    print("Computing...")  # 🖥️

compute()

# Example 12: Chaining memoization and timing decorators
def memoize(func):
    cache = {}  # 🗄️
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        else:
            print("Using cached result")  # ♻️
        return cache[args]
    return wrapper

def timeit(func):
    def wrapper(*args):
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        print(f"Time: {end_time - start_time} seconds")  # ⏰
        return result
    return wrapper

@memoize
@timeit
def slow_function(n):
    time.sleep(1)  # Simulate slow computation
    return n * n  # 🔢

print(slow_function(5))
print(slow_function(5))  # Cached result

# Example 13: Chaining decorators for input validation and output formatting
def validate_input(func):
    def wrapper(n):
        if n < 0:
            raise ValueError("Negative number not allowed")  # ⚠️
        return func(n)
    return wrapper

def format_output(func):
    def wrapper(n):
        result = func(n)
        return f"Result is: {result}"  # 📝
    return wrapper

@format_output
@validate_input
def compute_square(n):
    return n * n  # 📐

print(compute_square(4))

# Example 14: Chaining decorators in a Flask-like application
def route(url):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Routing to {url}")  # 🌐
            return func(*args, **kwargs)
        return wrapper
    return decorator

def authorize(role):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Authorizing role: {role}")  # 🔐
            return func(*args, **kwargs)
        return wrapper
    return decorator

@route("/home")
@authorize("user")
def home_page():
    print("Welcome to the home page")  # 🏠

home_page()

# Example 15: Chaining decorators that manage resources
def open_connection(func):
    def wrapper(*args, **kwargs):
        print("Opening connection")  # 🔌
        result = func(*args, **kwargs)
        print("Closing connection")  # 🔌
        return result
    return wrapper

def transaction(func):
    def wrapper(*args, **kwargs):
        print("Starting transaction")  # 💳
        result = func(*args, **kwargs)
        print("Ending transaction")    # 💳
        return result
    return wrapper

@open_connection
@transaction
def execute_query():
    print("Executing query")  # 📝

execute_query()

# End of Chapter 10 Examples