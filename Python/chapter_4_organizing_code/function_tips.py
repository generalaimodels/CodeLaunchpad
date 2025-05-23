# Chapter 4: Organizing Code: Functions 📦 (Reusable Code Blocks)

# 4.1 What are Functions? Code Packages 📦 (Named Actions)

# Example 1:
def say_hello():
    # 📝 This function prints a greeting message.
    print("Hello!")

say_hello()  # 📞 Calling the function

# Example 2:
def display_number():
    # 📝 This function displays a number.
    print(42)

display_number()  # 📞 Function call

# Example 3:
def greet():
    # 📝 Prints a simple greeting.
    print("Hi there!")

greet()  # 📞 Function invocation

# Example 4:
def print_line():
    # 📝 This function prints a line separator.
    print("-" * 30)

print_line()  # 📞 Calling the function

# Example 5:
def show_date():
    # 📝 Displays the current date.
    from datetime import date
    print("Today's date is:", date.today())

show_date()  # 📞 Function call

# Example 6:
def welcome_message():
    # 📝 Welcomes the user.
    print("Welcome to the program!")

welcome_message()  # 📞 Invoke the function

# Example 7:
def farewell():
    # 📝 Bids farewell to the user.
    print("Goodbye!")

farewell()  # 📞 Function call

# Example 8:
def show_pi():
    # 📝 Displays the value of pi.
    import math
    print("The value of pi is:", math.pi)

show_pi()  # 📞 Calling the function

# Example 9:
def empty_function():
    # 📝 Does nothing.
    pass  # 🚫 No operation

empty_function()  # 📞 Function invocation

# Example 10:
def print_smiley():
    # 😊 Prints a smiley face.
    print(":)")

print_smiley()  # 📞 Call the function

# Example 11:
def display_message():
    # 📝 Displays a custom message.
    print("Learning Python is fun!")

display_message()  # 📞 Function call

# Example 12:
def show_version():
    # 📝 Shows the Python version.
    import sys
    print("Python version:", sys.version)

show_version()  # 📞 Invoke the function

# Example 13:
def list_numbers():
    # 📝 Prints numbers from 1 to 5.
    for i in range(1, 6):
        print(i, end=' ')
    print()  # Newline

list_numbers()  # 📞 Calling the function

# Example 14:
def print_alphabet():
    # 📝 Prints the alphabet.
    import string
    print(string.ascii_lowercase)

print_alphabet()  # 📞 Function call

# Example 15:
def announce_end():
    # 📝 Announces the end of examples.
    print("End of function examples.")

announce_end()  # 📞 Call the function

# 4.2 Function Parameters and Arguments (Inputs to Functions)

# Example 1:
def greet_person(name):
    # 📝 Greets the person by name.
    print(f"Hello, {name}!")

greet_person("Alice")  # 📞 Positional argument

# Example 2:
def add_numbers(a, b):
    # 📝 Adds two numbers.
    result = a + b
    print(f"{a} + {b} = {result}")

add_numbers(5, 7)  # 📞 Positional arguments

# Example 3:
def subtract_numbers(a, b):
    # 📝 Subtracts b from a.
    result = a - b
    print(f"{a} - {b} = {result}")

subtract_numbers(a=10, b=3)  # 📞 Keyword arguments

# Example 4:
def multiply_numbers(a, b=2):
    # 📝 Multiplies two numbers with default b=2.
    result = a * b
    print(f"{a} * {b} = {result}")

multiply_numbers(5)  # 📞 Uses default b

# Example 5:
def divide_numbers(a=10, b=2):
    # 📝 Divides a by b with default values.
    result = a / b
    print(f"{a} / {b} = {result}")

divide_numbers()  # 📞 Uses default a and b

# Example 6:
def calculate_power(base, exponent=2):
    # 📝 Calculates base raised to exponent.
    result = base ** exponent
    print(f"{base}^{exponent} = {result}")

calculate_power(3)  # 📞 Default exponent

# Example 7:
def introduce_person(first_name, last_name):
    # 📝 Introduces a person.
    print(f"My name is {first_name} {last_name}.")

introduce_person(last_name="Smith", first_name="John")  # 📞 Keyword arguments

# Example 8:
def print_colors(*colors):
    # 📝 Prints all given colors.
    print("Colors:", colors)

print_colors("Red", "Green", "Blue")  # 📞 Variable arguments

# Example 9:
def show_profile(name, age, **info):
    # 📝 Displays a profile with additional info.
    print(f"Name: {name}, Age: {age}, Info: {info}")

show_profile("Anna", 28, city="New York", hobby="Photography")  # 📞 Keyword variable arguments

# Example 10:
def calculate_total(*numbers):
    # 📝 Calculates the sum of all numbers.
    total = sum(numbers)
    print("Total sum:", total)

calculate_total(1, 2, 3, 4, 5)  # 📞 Variable arguments

# Example 11:
def greet_full_name(first, middle='', last=''):
    # 📝 Greets with full name.
    print(f"Hello, {first} {middle} {last}".strip())

greet_full_name("Emily")  # 📞 Only first name

# Example 12:
def make_sandwich(bread, *fillings):
    # 📝 Describes the sandwich.
    print(f"Bread: {bread}, Fillings: {fillings}")

make_sandwich("Wheat", "Ham", "Cheese")  # 📞 Variable arguments

# Example 13:
def build_profile(first, last, **user_info):
    # 📝 Builds a user profile dictionary.
    profile = {'first_name': first, 'last_name': last}
    profile.update(user_info)
    print("Profile:", profile)

build_profile('Albert', 'Einstein', location='Princeton', field='Physics')  # 📞 Keyword variable arguments

# Example 14:
def order_pizza(size, *toppings):
    # 📝 Summarizes the pizza order.
    print(f"Ordering {size} pizza with toppings:")
    for topping in toppings:
        print(f"- {topping}")

order_pizza('Large', 'Pepperoni', 'Olives', 'Mushrooms')  # 📞 Variable arguments

# Example 15:
def print_pet_info(animal_type, pet_name):
    # 📝 Prints information about a pet.
    print(f"I have a {animal_type} named {pet_name}.")

print_pet_info(pet_name='Buddy', animal_type='dog')  # 📞 Keyword arguments

# 4.3 Return Values: Function Outputs 📤 (Results of Actions)

# Example 1:
def sum_two_numbers(a, b):
    # 📝 Returns the sum of a and b.
    return a + b  # 📤 Return result

result = sum_two_numbers(10, 15)
print("Sum:", result)  # Outputs: 25

# Example 2:
def get_full_name(first_name, last_name):
    # 📝 Returns the full name.
    return f"{first_name} {last_name}"

full_name = get_full_name("Jane", "Doe")
print("Full Name:", full_name)  # Outputs: Jane Doe

# Example 3:
def is_even(number):
    # 📝 Checks if a number is even.
    return number % 2 == 0

print("Is 4 even?", is_even(4))  # Outputs: True

# Example 4:
def get_max(a, b):
    # 📝 Returns the maximum of a and b.
    return max(a, b)

maximum = get_max(8, 12)
print("Max:", maximum)  # Outputs: 12

# Example 5:
def list_movies():
    # 📝 Returns a list of movies.
    return ["The Matrix", "Inception", "Interstellar"]

movies = list_movies()
print("Movies:", movies)  # Outputs: [...]

# Example 6:
def compute_area(radius):
    # 📝 Calculates area of a circle.
    area = 3.1416 * radius ** 2
    return area

circle_area = compute_area(5)
print("Area:", circle_area)  # Outputs: Area value

# Example 7:
def get_user_info():
    # 📝 Returns user info as a dictionary.
    return {"username": "admin", "password": "1234"}

user = get_user_info()
print("User Info:", user)  # Outputs: {...}

# Example 8:
def no_return_function():
    # 📝 Function without return statement.
    print("This function returns None by default.")

result = no_return_function()
print("Result:", result)  # Outputs: None

# Example 9:
def divide(a, b):
    # 📝 Divides a by b, handles division by zero.
    if b == 0:
        return None  # 📤 Return None if division by zero
    return a / b

division_result = divide(10, 0)
print("Division Result:", division_result)  # Outputs: None

# Example 10:
def get_person_age(name):
    # 📝 Returns age based on name (dummy data).
    ages = {'Alice': 30, 'Bob': 25}
    return ages.get(name, 0)

age = get_person_age('Alice')
print("Age:", age)  # Outputs: 30

# Example 11:
def filter_even_numbers(numbers):
    # 📝 Returns even numbers from a list.
    return [num for num in numbers if num % 2 == 0]

even_numbers = filter_even_numbers([1, 2, 3, 4, 5, 6])
print("Even Numbers:", even_numbers)  # Outputs: [2, 4, 6]

# Example 12:
def factorial(n):
    # 📝 Computes factorial using recursion.
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

fact_result = factorial(5)
print("Factorial:", fact_result)  # Outputs: 120

# Example 13:
def make_uppercase(text):
    # 📝 Converts text to uppercase.
    return text.upper()

uppercase_text = make_uppercase("hello")
print("Uppercase:", uppercase_text)  # Outputs: HELLO

# Example 14:
def check_prime(num):
    # 📝 Checks if a number is prime.
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

is_prime = check_prime(7)
print("Is 7 prime?", is_prime)  # Outputs: True

# Example 15:
def get_temperature():
    # 📝 Simulates getting temperature, returns None if unavailable.
    import random
    if random.choice([True, False]):
        return random.uniform(20.0, 30.0)
    else:
        return None  # 📤 No temperature available

temperature = get_temperature()
print("Temperature:", temperature)  # Outputs: Value or None