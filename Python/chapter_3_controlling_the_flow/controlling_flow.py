# developer_chapter_3_control_flow.py

# Chapter 3: "Controlling the Flow 🚦 - Decision Making and Loops"

# Hello Developer bhai! Chapter 3 mein aapka swagat hai.
# Iss chapter mein, hum seekhenge ki code ka flow kaise control karna hai.
# Imagine you are driving a car 🚗. You need to decide when to turn, when to stop, and how to keep going in circles if needed.
# Code mein bhi aise hi decisions aur repetitions hote hain. Let's learn about them!

# ---

# ### 3.1 Conditional Statements: Making Decisions 🤔 (If-Else Branches)

# **Concept:** Conditional statements are all about making decisions in your code.
# "Agar aisa hai, toh yeh karo, warna woh karo." - That's the basic idea.

# **Analogy:** Sophisticated Traffic Management System 🚦 at a Multi-Way Intersection 🌐

# Imagine a busy traffic intersection. Traffic lights decide who goes and when.
# `if`, `elif`, `else` are like these traffic lights for your code.

# *   **`if` condition (Primary Route Prioritization 🚦):**
#     "Agar red light hai, toh gaadi roko." -  First check, most important condition.

# *   **`elif` conditions (Secondary Route Evaluations 🚦🚦):**
#     "Agar red light nahi hai, lekin yellow light hai, toh slow down." -  If the first condition is false, check another one. You can have many `elif`s.

# *   **`else` condition (Default Route Handling 🚦):**
#     "Agar red aur yellow dono nahi hai, toh go ahead!" -  If none of the above conditions are true, do this by default.

# **Explanation Breakdown (Technical Precision):**

# *   **`if` statement - Predicate-Based Execution:**
#     The `if` statement checks a condition. If the condition is `True`, it runs the code inside the `if` block.

#     ```python
transaction_amount = 1500
account_balance = 2000
if transaction_amount <= account_balance:  # Condition: Is transaction amount less than or equal to account balance?
    print("Transaction approved. ✅")  # If True, print this message
    account_balance -= transaction_amount  # And update account balance
    print(f"Remaining balance: {account_balance}")
#     ```

# *   **`elif` (else if) statement - Sequential Condition Cascading:**
#     `elif` is used after `if` to check another condition if the `if` condition was `False`.
#     You can have multiple `elif`s to check many conditions in order.

#     ```python
http_status_code = 404
if http_status_code == 200:
    print("Success: Request successful! 👍")
elif http_status_code == 404:  # If the first 'if' was False, check this condition
    print("Error: Page not found ⚠️")
elif http_status_code == 500:  # If both 'if' and previous 'elif' were False, check this
    print("Error: Server problem ❌")
#     ```

# *   **`else` statement - Contingency Execution Path:**
#     `else` is used at the end. If none of the `if` or `elif` conditions were `True`, the code inside `else` block runs.
#     It's like a default action if nothing else matches.

#     ```python
user_role = "guest"
if user_role == "admin":
    print("Admin access granted. 👑")
elif user_role == "editor":
    print("Editor access granted. ✍️")
else:  # If user_role is neither "admin" nor "editor"
    print("Guest access only. Limited features. 🔒")  # Default action for guests
#     ```

# *   **Indentation - Syntactic Block Delimitation 📏:**
#     Python uses indentation (spaces) to decide which code belongs inside `if`, `elif`, `else` blocks.
#     Correct indentation is VERY important. If indentation is wrong, your code won't work!

#     ```python
#     if True:
#         print("This line is inside the if block.") # Indented - part of 'if'
#     print("This line is outside the if block.")     # Not indented - after 'if'
#     ```

# **Structure & Visual (Enhanced Flowchart):**

# ```mermaid
# graph LR
#     A[Start] --> B{Condition 1? 🤔};
#     B -- Yes ✅ --> C[Code Block 1 🧱 (if block)];
#     B -- No ❌ --> D{Condition 2? 🤔 (elif?)};
#     D -- Yes ✅ --> E[Code Block 2 🧱 (elif block)];
#     D -- No ❌ --> F[Else Block 🧱 (else block)];
#     C --> G[End 🏁];
#     E --> G;
#     F --> G;
#
#     style B fill:#f9f,stroke:#333,stroke-width:2px
#     style D fill:#f9f,stroke:#333,stroke-width:2px
#     style C fill:#ccf,stroke:#333,stroke-width:2px
#     style E fill:#ccf,stroke:#333,stroke-width:2px
#     style F fill:#ccf,stroke:#333,stroke-width:2px
#     style G fill:#cfc,stroke:#333,stroke-width:2px
# ```

# **Example - Checking Number Sign:**
number = int(input("Enter a number: ")) # Get number input from user

if number > 0:
    print(f"{number} is a positive number 👍")
elif number == 0:
    print(f"{number} is zero 😐")
else:
    print(f"{number} is a negative number 👎")

# **Summary:** `if`, `elif`, `else` help your code make decisions based on conditions.
# They are like traffic lights controlling the flow of your program. Indentation is key!

# ---

# ### 3.2 Loops: Repeating Actions 🔄 (Repetitive Tasks)

# **Concept:** Loops are for doing the same thing again and again.
# "Yeh kaam baar baar karo jab tak..." -  Repeat actions until a condition is met or for each item in a list.

# **Analogy:** Automated Robotic Assembly Line 🤖🏭

# Imagine a factory assembly line. Robots do the same task on each item moving on the line.
# `for` and `while` loops are like these assembly lines in code.

# *   **`for` loop (Item-Based Processing Line 📦➡️⚙️➡️📦):**
#     "Har ek item ke liye yeh karo." -  Do something for each item in a list, tuple, string, etc.

# *   **`while` loop (Condition-Driven Quality Control Loop 🔄):**
#     "Jab tak condition true hai, yeh karte raho." - Keep doing something as long as a condition is true.

# **Explanation Breakdown (Technical Precision):**

# *   **`for` loop - Sequence Iteration Protocol:**
#     `for` loop is used to iterate over a sequence (like a list, string, tuple, range).
#     It goes through each item in the sequence one by one and executes the code block for each item.

#     ```python
data_points = [25, 67, 89, 42, 95]
for point in data_points:  # For each 'point' in 'data_points' list
    processed_value = point * 2  # Process each point (multiply by 2)
    print(f"Value: {point}, Processed: {processed_value}")
#     ```

# *   **`while` loop - Condition-Controlled Repetition Cycle:**
#     `while` loop keeps running as long as a condition is `True`.
#     Make sure the condition eventually becomes `False`, otherwise, it will run forever (infinite loop!).

#     ```python
counter = 0
while counter < 5:  # While 'counter' is less than 5
    print(f"Counter is: {counter}")
    counter += 1  # Increment counter (important to avoid infinite loop!)
#     ```

# *   **`range()` function for `for` loop:**
#     `range()` is often used with `for` loop to repeat something a specific number of times.
#     `range(start, stop, step)` generates a sequence of numbers.

#     ```python
for i in range(1, 6):  # Numbers from 1 to 5 (6 is not included)
    print(f"Iteration number: {i}")
#     ```

# **Visuals (Enhanced Flowcharts):**

# *   **`for` loop Visual:**

#     ```mermaid
#     graph LR
#         A[Sequence: [Item 1 📦, Item 2 📦, Item 3 📦, ...]] --> B{Next Item? 🤔};
#         B -- Yes ✅ --> C[Process Item ⚙️ (Code Block)];
#         C --> B;
#         B -- No ❌ --> D[Loop End 🏁];
#         style B fill:#f9f,stroke:#333,stroke-width:2px
#         style C fill:#ccf,stroke:#333,stroke-width:2px
#         style D fill:#cfc,stroke:#333,stroke-width:2px
#     ```

# *   **`while` loop Visual:**

#     ```mermaid
#     graph LR
#         A[Start 🏁] --> B{Condition? 🤔};
#         B -- Yes ✅ --> C[Code Block 🧱];
#         C --> B;
#         B -- No ❌ --> D[Loop End 🏁];
#         style B fill:#f9f,stroke:#333,stroke-width:2px
#         style C fill:#ccf,stroke:#333,stroke-width:2px
#         style D fill:#cfc,stroke:#333,stroke-width:2px
#     ```

# **Example - Sum of numbers using for loop:**
numbers_list = [1, 2, 3, 4, 5]
sum_of_numbers = 0
for num in numbers_list:
    sum_of_numbers += num # Add each number to the sum
print(f"Sum of numbers in list: {sum_of_numbers}")

# **Example - Countdown using while loop:**
countdown = 5
print("Countdown starts!")
while countdown > 0:
    print(countdown)
    countdown -= 1 # Decrease countdown
print("Blast off! 🚀")

# **Summary:** Loops (`for` and `while`) are for repeating code.
# `for` loop for items in a sequence, `while` loop for as long as a condition is true.
# Be careful with `while` loops to avoid infinite loops!

# ---

# ### 3.3 Loop Control: `break` and `continue` (Emergency Exits & Iteration Management)

# **Concept:** `break` and `continue` are special commands to control loops from inside.
# They let you change the normal flow of a loop.

# **Analogy:** Advanced Conveyor Belt System with Exception Handling 🏭🛑⏭️

# Back to the factory assembly line, but now with more control.

# *   **`break` - Emergency Stop Protocol 🛑:**
#     "Agar emergency hai, assembly line band karo!" -  Stop the loop completely, right now.

# *   **`continue` - Faulty Item Bypass Mechanism ⏭️:**
#     "Agar item kharab hai, is item ko skip karo, lekin line chalti rahe." - Skip the current step in the loop and go to the next one.

# **Explanation Breakdown (Technical Precision):**

# *   **`break`:  Unconditional Loop Termination 🚪 -  Exception Handling Paradigm:**
#     `break` immediately exits the loop. The loop stops right there, and the code after the loop continues.
#     Used when you find what you are looking for or something goes wrong and you need to stop the loop.

#     ```python
search_target = 78
data_series = [23, 45, 67, 78, 90, 12, 34]
found_index = -1

for index, value in enumerate(data_series):
    if value == search_target:
        found_index = index
        print(f"Target {search_target} found at index {found_index}")
        break  # Exit the loop as soon as target is found!
    print(f"Searching... checked value: {value}") # Will stop printing after finding 78

if found_index == -1:
    print(f"Target {search_target} not found.")
#     ```

# *   **`continue`: Conditional Iteration Bypass ⏭️ -  Data Filtering and Selective Processing:**
#     `continue` skips the rest of the current iteration and goes to the next iteration of the loop.
#     Used when you want to skip processing for some items but continue with the rest of the loop.

#     ```python
data_values = [10, -5, 20, 0, 30, -2, 15]
positive_values_sum = 0

for value in data_values:
    if value <= 0:  # Check for non-positive values
        print(f"Skipping non-positive value: {value}")
        continue  # Skip to the next iteration if value is not positive
    positive_values_sum += value
    print(f"Adding positive value: {value}")

print(f"Sum of positive values: {positive_values_sum}")
#     ```

# **Example - Break in while loop:**
count = 0
while True: # Intentionally infinite loop (until break)
    print(f"Count is: {count}")
    count += 1
    if count >= 3:
        print("Count reached 3, breaking out of loop!")
        break # Exit the while loop when count is 3

# **Example - Continue in for loop:**
for i in range(5):
    if i == 2:
        print("Skipping iteration 2 using continue")
        continue # Skip iteration when i is 2
    print(f"Processing iteration: {i}")

# **Summary:** `break` stops the loop completely. `continue` skips the current iteration and goes to the next.
# These are powerful tools to control loops in more complex ways.

# ---

# **Congratulations, Developer bhai!** You have completed Chapter 3 on Control Flow! 🚦
# Now you know how to make decisions in your code using `if`, `elif`, `else` and how to repeat actions using `for` and `while` loops.
# You also learned how to control loops with `break` and `continue`.
# Keep practicing these, and you will be controlling your code like a boss! 😎