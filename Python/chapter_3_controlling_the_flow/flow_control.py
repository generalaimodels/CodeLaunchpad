# Chapter 3: Controlling the Flow 🚦 - Decision Making and Loops

# =========================================
# 3.1 Conditional Statements: Making Decisions 🤔 (If-Else Branches)
# =========================================

# Example 1: Simple if statement
age = 18
if age >= 18:
    print("You're an adult!")  # 🎉 Prints if condition is True

# Example 2: if-else statement
temperature = 30
if temperature > 25:
    print("It's hot outside.")  # 🌞
else:
    print("It's not that hot.")  # 🌤️

# Example 3: if-elif-else statement
score = 85
if score >= 90:
    print("Grade: A")  # 🅰️
elif score >= 80:
    print("Grade: B")  # 🅱️
elif score >= 70:
    print("Grade: C")  # 🅾️
else:
    print("Grade: D or F")  # ❌

# Example 4: Nested if statements
num = 10
if num > 0:
    print("Positive number")  # ➕
    if num % 2 == 0:
        print("Even number")  # ✨
    else:
        print("Odd number")  # 🔢

# Example 5: Checking multiple conditions with logical AND
username = "admin"
password = "1234"
if username == "admin" and password == "1234":
    print("Access granted")  # ✅
else:
    print("Access denied")  # ❌

# Example 6: Checking multiple conditions with logical OR
day = "Saturday"
if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")  # 🎉
else:
    print("It's a weekday.")  # 📅

# Example 7: Using not operator
logged_in = False
if not logged_in:
    print("Please log in.")  # 🔐

# Example 8: Ternary conditional operator
age = 16
message = "Eligible to vote" if age >= 18 else "Not eligible to vote"
print(message)  # 🗳️

# Example 9: Comparing strings
word = "Python"
if word == "Python":
    print("Match found!")  # 🐍

# Example 10: Comparing lists
list_a = [1, 2, 3]
list_b = [1, 2, 3]
if list_a == list_b:
    print("Lists are equal")  # 📋

# Example 11: Checking for membership with 'in'
fruits = ["apple", "banana", "cherry"]
if "banana" in fruits:
    print("Banana is in the list")  # 🍌

# Example 12: Handling None
data = None
if data is None:
    print("No data available")  # 🚫

# Example 13: Using isinstance()
value = 10
if isinstance(value, int):
    print("Value is an integer")  # 🔢

# Example 14: Combining conditions
x = 5
if 0 < x < 10:
    print("x is between 0 and 10")  # 🎯

# Example 15: Avoiding common mistake - using '=' instead of '=='
# Correct usage:
if x == 5:
    print("x equals 5")  # ✅
# Incorrect (would cause SyntaxError):
# if x = 5:
#     print("x equals 5")  # ❌

# =========================================
# 3.2 Loops: Repeating Actions 🔄 (Repetitive Tasks)
# =========================================

# Example 1: Simple for loop over a list
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(num)  # Prints numbers from 1 to 5

# Example 2: For loop with range()
for i in range(5):
    print(i)  # Prints 0 to 4

# Example 3: While loop with counter
count = 0
while count < 5:
    print("Count is:", count)  # Shows count from 0 to 4
    count += 1

# Example 4: Iterating over a string
for char in "Hello":
    print(char)  # Prints each character in "Hello"

# Example 5: Looping through dictionary keys
student_scores = {"Alice": 90, "Bob": 85, "Charlie": 95}
for student in student_scores:
    print(student)  # Prints student names

# Example 6: Looping through dictionary items
for student, score in student_scores.items():
    print(f"{student} scored {score}")  # Shows name and score

# Example 7: Using else with a for loop
for num in range(3):
    print(num)
else:
    print("Loop finished")  # Executes after the loop completes

# Example 8: Using else with a while loop
n = 3
while n > 0:
    print(n)
    n -= 1
else:
    print("Countdown complete!")  # 🚀

# Example 9: Infinite loop (use with caution!)
# Uncomment to run (it will run indefinitely)
# while True:
#     print("This will run forever...")  # ♾️

# Example 10: Nested for loops
for i in range(1, 3):
    for j in range(1, 3):
        print(f"i={i}, j={j}")  # Combines i and j

# Example 11: Iterating over a list with index
colors = ["red", "green", "blue"]
for index, color in enumerate(colors):
    print(f"Color {index}: {color}")  # Shows index and color

# Example 12: Using pass inside loops
for i in range(5):
    pass  # Does nothing, placeholder

# Example 13: While loop with break condition
n = 10
while n > 0:
    print(n)
    if n == 5:
        print("Breaking loop")
        break  # Exit loop when n is 5
    n -= 1

# Example 14: For loop with range and step
for i in range(0, 10, 2):
    print(i)  # Prints even numbers from 0 to 8

# Example 15: Looping over a set
unique_numbers = {1, 2, 3, 2, 1}
for number in unique_numbers:
    print(number)  # Prints unique numbers

# =========================================
# 3.3 Loop Control: break and continue (Emergency Exits & Skips)
# =========================================

# Example 1: Using break in a for loop
for letter in "Python":
    if letter == "h":
        break  # Exit loop when letter is 'h'
    print(letter)  # Prints 'P', 'y', 't'

# Example 2: Using continue in a for loop
for letter in "Python":
    if letter == "h":
        continue  # Skip 'h'
    print(letter)  # Prints all letters except 'h'

# Example 3: Using break in a while loop
i = 1
while i <= 10:
    if i == 5:
        break  # Exit loop when i is 5
    print(i)
    i += 1  # Prints 1 to 4

# Example 4: Using continue in a while loop
i = 0
while i < 5:
    i += 1
    if i == 3:
        continue  # Skip when i is 3
    print(i)  # Prints 1, 2, 4, 5

# Example 5: Break out of nested loops using flag
found = False
for i in range(3):
    for j in range(3):
        if i * j == 4:
            found = True
            break  # Break inner loop
    if found:
        break  # Break outer loop
print("Exited loops")  # 🚪

# Example 6: Using else with loops and break
for n in range(2, 6):
    for x in range(2, n):
        if n % x == 0:
            print(f"{n} equals {x} * {n//x}")
            break  # Not a prime number
    else:
        print(f"{n} is a prime number")  # 🔢

# Example 7: Skipping even numbers
for num in range(10):
    if num % 2 == 0:
        continue  # Skip even numbers
    print(num)  # Prints odd numbers

# Example 8: Exiting loop when input is correct
# Uncomment to run (requires user input)
# while True:
#     password = input("Enter password: ")
#     if password == "secret":
#         print("Access granted")
#         break  # Correct password, exit loop
#     else:
#         print("Try again")

# Example 9: Using break with infinite loop
i = 0
while True:
    print(i)
    i += 1
    if i >= 3:
        break  # Exit after i reaches 3

# Example 10: Using continue to skip letters
for letter in "Hello World":
    if letter == "o":
        continue  # Skip 'o'
    print(letter)

# Example 11: Break when item found in list
animals = ["cat", "dog", "rabbit"]
for animal in animals:
    if animal == "dog":
        print("Found the dog!")
        break  # Stop searching
    print(animal)

# Example 12: Continue in nested loops
for i in range(3):
    for j in range(3):
        if j == 1:
            continue  # Skip when j is 1
        print(f"i={i}, j={j}")

# Example 13: Break out of loop based on condition
for num in range(2, 10):
    if num % 5 == 0:
        break  # Exit loop when num is multiple of 5
    print(num)

# Example 14: Continue to next iteration
for num in range(5):
    if num == 2:
        continue  # Skip number 2
    print(num)

# Example 15: Using pass (does nothing, but syntactically needed)
for letter in "Python":
    if letter == "h":
        pass  # Placeholder, does nothing
    print(letter)

# End of Chapter 3 Examples