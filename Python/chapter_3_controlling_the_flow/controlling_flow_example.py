# Chapter 3: Controlling the Flow 🚦 - Decision Making and Loops

# 3.1 Conditional Statements: Making Decisions 🤔 (If-Else Branches)

# Example 1:
age = 18
if age >= 18:
    # ✅ Age is 18 or older
    print("You are an adult.")
else:
    # ❌ Age is less than 18
    print("You are a minor.")

# Example 2:
score = 85
if score >= 90:
    # 🎉 Excellent score
    print("Grade: A")
elif score >= 80:
    # 👍 Good score
    print("Grade: B")
elif score >= 70:
    # 😊 Fair score
    print("Grade: C")
else:
    # 😕 Needs improvement
    print("Grade: D")

# Example 3:
num = 0
if num > 0:
    # ➕ Positive number
    print("Positive number")
elif num == 0:
    # 🅾️ Zero
    print("Zero")
else:
    # ➖ Negative number
    print("Negative number")

# Example 4:
password = "secret123"
if password == "admin":
    # 🔑 Correct password
    print("Access granted.")
else:
    # 🚫 Incorrect password
    print("Access denied.")

# Example 5:
# ⚠️ Be careful with indentation; it's crucial in Python!
temperature = 30
if temperature > 25:
    print("It's hot outside.")
    print("Stay hydrated!")  # 🥤
# Missing indentation would cause an IndentationError

# Example 6:
is_member = True
discount = 0
if is_member:
    discount = 10  # 💰 Member discount
else:
    discount = 0
print(f"Discount: {discount}%")

# Example 7:
language = "Python"
if language == "Python":
    # 🐍 Python selected
    print("You are using Python.")
elif language == "Java":
    # ☕ Java selected
    print("You are using Java.")
else:
    # 🤷 Other language
    print("Unknown language.")

# Example 8:
balance = 1500
withdrawal = 500
if withdrawal <= balance:
    balance -= withdrawal  # 💸 Withdrawal successful
    print(f"Withdrawal successful. New balance: {balance}")
else:
    # ❌ Insufficient funds
    print("Insufficient balance.")

# Example 9:
# ⚠️ Comparing floating-point numbers can be tricky due to precision.
x = 0.1 + 0.2
if abs(x - 0.3) < 1e-9:
    print("x is approximately 0.3")
else:
    print("x is not 0.3")

# Example 10:
permissions = ["read", "write"]
if "execute" in permissions:
    # 🏃 Execution permission granted
    print("You can execute files.")
else:
    # ⛔ No execution permission
    print("Execution permission denied.")

# Example 11:
# ⚠️ Logical operators
a = True
b = False
if a and b:
    print("Both are True")
else:
    print("At least one is False")

# Example 12:
user_input = ""
if user_input:
    print("Input received")
else:
    # ⌨️ No input provided
    print("No input provided")

# Example 13:
day = "Saturday"
if day == "Saturday" or day == "Sunday":
    # 🎉 It's the weekend
    print("It's the weekend!")
else:
    # 🏢 Weekday work
    print("It's a weekday.")

# Example 14:
# ⚠️ Beware of assignment (=) vs equality (==)
value = 10
if value == 10:
    print("Value is 10")
# Using '=' instead of '==' would cause a SyntaxError

# Example 15:
items_in_cart = 3
if items_in_cart:
    # 🛒 Cart is not empty
    print(f"You have {items_in_cart} items in your cart.")
else:
    # 🛍️ Cart is empty
    print("Your cart is empty.")

# 3.2 Loops: Repeating Actions 🔄 (Repetitive Tasks)

# Example 1: For loop with a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    # 🍎 Iterating over fruits
    print(f"I like {fruit}")

# Example 2: For loop with range()
for i in range(5):
    # 🔢 Counting from 0 to 4
    print(i)

# Example 3: While loop
count = 0
while count < 3:
    # ⌛ Looping while count is less than 3
    print(f"Count is {count}")
    count += 1

# Example 4: For loop with else
for number in range(1, 4):
    print(number)
else:
    # ✅ Loop completed without break
    print("Loop ended normally.")

# Example 5: While loop with else
n = 0
while n < 5:
    print(n)
    n += 1
else:
    # ✅ While loop ended
    print("While loop finished.")

# Example 6: Nested loops
for i in range(1, 4):
    for j in range(1, 3):
        # 🔄 Nested loop
        print(f"i={i}, j={j}")

# Example 7: Iterating over a dictionary
person = {'name': 'Alice', 'age': 25}
for key in person:
    # 🗝️ Iterating keys
    print(f"{key}: {person[key]}")

# Example 8: Iterating over a string
for char in "Hello":
    # 🔠 Iterating characters
    print(char)

# Example 9: Infinite loop (be cautious!)
# ⚠️ Uncommenting the below lines will create an infinite loop
# while True:
#     print("This will run forever.")

# Example 10: Using enumerate()
colors = ["red", "green", "blue"]
for index, color in enumerate(colors):
    # 📝 Accessing index and value
    print(f"{index}: {color}")

# Example 11: Looping with step in range()
for i in range(0, 10, 2):
    # 🔢 Counting with step of 2
    print(i)

# Example 12: Reverse iteration
for i in reversed(range(5)):
    # 🔄 Reverse counting
    print(i)

# Example 13: Looping over a set
unique_numbers = {1, 2, 3}
for num in unique_numbers:
    # 🔢 Iterating over set
    print(num)

# Example 14: Sum of first n numbers
n = 5
total = 0
for i in range(1, n+1):
    total += i  # ➕ Adding numbers
print(f"Sum is {total}")

# Example 15: Multiplication table
num = 3
for i in range(1, 11):
    product = num * i
    # ✖️ Multiplying
    print(f"{num} x {i} = {product}")

# 3.3 Loop Control: break and continue (Emergency Exits & Skips)

# Example 1: Using break in a for loop
for i in range(10):
    if i == 5:
        # 🛑 Breaks the loop when i is 5
        break
    print(i)

# Example 2: Using continue in a for loop
for i in range(5):
    if i == 2:
        # ⏭️ Skips the iteration when i is 2
        continue
    print(i)

# Example 3: Using break in a while loop
n = 0
while n < 10:
    if n == 6:
        # 🛑 Break when n is 6
        break
    print(n)
    n += 1

# Example 4: Using continue in a while loop
n = 0
while n < 5:
    n += 1
    if n == 3:
        # ⏭️ Skip when n is 3
        continue
    print(n)

# Example 5: Break with nested loops
for i in range(3):
    for j in range(3):
        if j == 1:
            # 🛑 Breaks inner loop when j is 1
            break
        print(f"i={i}, j={j}")

# Example 6: Continue with nested loops
for i in range(3):
    for j in range(3):
        if j == 1:
            # ⏭️ Skips to next iteration of inner loop
            continue
        print(f"i={i}, j={j}")

# Example 7: Using pass in loops
for i in range(5):
    pass  # 🤷 Does nothing
# Loop completes without any operation

# Example 8: Finding prime numbers
for num in range(2, 10):
    for i in range(2, num):
        if num % i == 0:
            # ❌ Not a prime number
            break
    else:
        # ✅ Prime number found
        print(f"{num} is a prime number")

# Example 9: Searching for an item
items = ["apple", "banana", "cherry"]
for item in items:
    if item == "banana":
        # 🍌 Item found
        print("Banana found!")
        break
else:
    # 🔍 Item not found
    print("Banana not found.")

# Example 10: Skipping even numbers
for num in range(1, 6):
    if num % 2 == 0:
        # ⏭️ Skip even numbers
        continue
    print(num)  # Prints only odd numbers

# Example 11: Break in infinite loop
counter = 0
while True:
    if counter >= 3:
        # 🛑 Break after 3 iterations
        break
    print(counter)
    counter += 1

# Example 12: Continue in infinite loop
# ⚠️ Be cautious with continue in loops to avoid infinite loops
counter = 0
while counter < 5:
    counter += 1
    if counter == 2:
        continue  # ⏭️ Skips printing when counter is 2
    print(counter)

# Example 13: Using break in a loop over a file
# Assuming 'sample.txt' exists
# try:
#     with open('sample.txt', 'r') as file:
#         for line in file:
#             if 'STOP' in line:
#                 # 🛑 Break if 'STOP' is found
#                 break
#             print(line.strip())
# except FileNotFoundError:
#     # ❌ File does not exist
#     print("File not found.")

# Example 14: Using continue to skip negative numbers
numbers = [1, -2, 3, -4, 5]
for num in numbers:
    if num < 0:
        # ⏭️ Skip negative numbers
        continue
    print(num)

# Example 15: Break with function inside loop
def is_special(num):
    # 🧐 Checks if number is special
    return num == 42

for i in range(100):
    if is_special(i):
        # 🛑 Found the special number
        print(f"Found special number: {i}")
        break