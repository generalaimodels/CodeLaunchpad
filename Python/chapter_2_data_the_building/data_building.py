# developer_chapter_2_data_fundamentals.py

# Chapter 2: "Data: The Building Blocks 🧱 of Python"

# Okay, Developer bhai, let's talk about 'Data' in Python.
# Think of data as basic ingredients, like atta, daal, chawal for cooking.
# In programming also, data is super important. This chapter is all about understanding data.
# It's like learning about bricks and cement before building a house. Got it?

# ---

# ### 2.1 Variables: Containers for Information 📦 (Labeled Boxes)

# **Concept:** Variables are like boxes to store information in your computer's memory.
# Think of them as 'data vaults' to keep your important stuff safe.

# **Analogy:** High-Capacity Data Vaults 🏦 with Customizable Labels 🏷️.

# **Explanation:**

# Variables are names we give to memory locations where we store data.
# It's how Python remembers things. Let's break it down:

# *   **Variable Naming Rules (Vault Labeling Conventions 🏷️):**
#     Think of these as rules for naming your boxes. If you name them properly, you won't get confused, right?

#     *   **Alphanumeric characters:**  Use letters (a-z, A-Z) and numbers (0-9).  Like 'myBox1'.
#     *   **Underscores:** Use underscore '_' to separate words, like 'user_name'.  Makes it readable, yaar.
#     *   **Cannot start with a digit:**  Name cannot start with a number, like '1box' is wrong, but 'box1' is okay.
#     *   **Case-sensitive:** 'name' and 'Name' are different boxes.  Python is very particular about upper and lower case.
#     *   **Avoid reserved keywords:**  Don't use words Python already uses, like 'if', 'for', 'print'. These are like special words in Python's dictionary.

#     **Example of Valid and Invalid Variable Names:**

#     | Valid Names       | Invalid Names     | Reason for Invalidity             |
#     |-------------------|--------------------|-------------------------------------|
#     | `my_variable`     | `1st_variable`    | Starts with a digit                  |
#     | `variableName`    | `variable-name`   | Hyphen '-' not allowed               |
#     | `_private_var`    | `for`             | Reserved keyword                     |
#     | `userName123`     | `my var`          | Space ' ' not allowed                |

# *   **Assignment Operator `=` (Data Deposition Mechanism 📥):**
#     This '=' sign is used to put values inside your variable boxes.  It's like putting things into the vault.

#     **How it works:**
#     1.  **Right-hand side (RHS):** Python first looks at what's on the right side of '='. It calculates the value.
#     2.  **Left-hand side (LHS):** Then, it takes that value and puts it into the box named on the left side of '='.

#     **It's not like maths equation!** In programming, `=` means 'store this value in this variable'.

#     **Visual Representation:**

#     ```
#     Value on RHS ----> 📥 Assignment Operator (=) ----> [Variable on LHS] 📦
#     ```

# *   **Dynamic Typing - Intelligent Vault System 🐍🧠:**
#     Python is smart, yaar! You don't need to tell Python what type of data will go in the box beforehand.
#     Python figures it out automatically based on the value you are putting in.

#     **Example:**
#     If you put a number, Python knows it's a number. If you put text, it knows it's text.
#     It's like a vault that adjusts itself to store any type of valuable without you telling it in advance.

#     **Contrast with other languages (just for info, no need to worry now):**
#     In some languages (like Java, C++), you have to say, "This box is for numbers only!" beforehand. That's called 'static typing'.
#     Python's 'dynamic typing' is more flexible and faster for coding, but you need to be a bit careful sometimes.

# **Example:**

name = "Alice"      # 📦 [name] stores text "Alice" 💬
age = 30           # 📦 [age] stores number 30 🔢
is_student = False  # 📦 [is_student] stores True/False value ✅/❌
salary = 75000.50  # 📦 [salary] stores decimal number 75000.50 🔢.decimal

# **Visual Representation:**

# ```
# [name] 📦---------> "Alice"  💬 (String - text data)
#        🏷️ "Name of User"

# [age]  📦---------> 30       🔢 (Integer - whole number data)
#        🏷️ "User's Age"

# [is_student] 📦---------> False ✅/❌ (Boolean - True/False data)
#              🏷️ "Is User a Student?"

# [salary] 📦---------> 75000.50 🔢.decimal (Float - decimal number data)
#          🏷️ "Annual Salary"
# ```

# **Summary:** Variables are names for memory locations to store data. Naming rules are important.
# `=` operator assigns values to variables. Python is dynamically typed, so types are figured out automatically.

# ---

# ### 2.2 Data Types: Kinds of Information 📊 (Different Box Types)

# **Concept:** Data types are like different types of boxes for different kinds of stuff.
# Some boxes for numbers, some for text, some for True/False, etc.

# **Analogy:** Specialized Container Classes 🗄️ in your Data Vault System.

# **Explanation:**

# Data types tell Python what kind of data we are dealing with.
# It's important because Python treats different types of data differently.
# Like, you can add numbers, but you can't 'add' text in the same way, right?

# Here are the basic data types in Python:

# *   **Integers (int): Whole numbers (..., -2, -1, 0, 1, 2, ...). 🔢 Counting Blocks:**
#     These are whole numbers, no decimal point.  Like counting things - 1 apple, 2 apples, etc.

#     *Example:* `count = 150`, `index = -5`, `quantity = 0`

# *   **Floating-point numbers (float): Numbers with decimal points (3.14, -0.5, 2.0). 🔢.decimal Measuring Liquids:**
#     Numbers with decimal points.  Like measuring liquids or money with paise.

#     *Example:* `price = 99.99`, `temperature = 25.5`, `ratio = 0.75`

# *   **Strings (str): Textual data, sequences of characters ("hello", "Python"). 💬 Words and Sentences:**
#     Text, words, sentences.  Anything you write in quotes.

#     *Example:* `message = "Welcome to Python"`, `name = 'Developer'`, `symbol = "$" `

# *   **Booleans (bool): True or False values. Logical states. ✅/❌ Switches - On or Off:**
#     Only two values: `True` or `False`.  Used for logic, yes/no answers, on/off switches.

#     *Example:* `is_valid = True`, `is_completed = False`, `has_error = True`

# *   **Lists (list): Ordered collections of items, mutable (changeable). `[1, 2, "apple"]`. 📜 Shopping Lists:**
#     List of items in order. You can change them - add, remove, change items.  Like a shopping list that you can update.
#     Use square brackets `[]`.

#     *Example:* `numbers = [10, 20, 30]`, `items = ["pen", "paper", "book"]`, `mixed_list = [1, "hello", True, 3.14]`

# *   **Tuples (tuple): Ordered collections of items, immutable (unchangeable). `(1, 2, "apple")`. 🔒📜 Fixed Records:**
#     Similar to lists, but you cannot change them once created.  Fixed, like a record that should not be altered.
#     Use parentheses `()`.

#     *Example:* `coordinates = (10, 20)`, `rgb_color = (255, 0, 0)`, `fixed_data = ("name", "version", 1.0)`

# *   **Dictionaries (dict): Key-value pairs. `{"name": "Alice", "age": 30}`. 📒 Dictionaries - Word and Definition:**
#     Store data in pairs of 'key' and 'value'.  Like a dictionary where you look up a 'word' (key) to find its 'meaning' (value).
#     Use curly braces `{}`.

#     *Example:* `student = {"name": "Alice", "age": 20, "major": "Computer Science"}`, `config = {"host": "localhost", "port": 8080}`

# *   **Sets (set): Unordered collections of unique items. `{1, 2, 3}`. 🎒 Bag of Unique Items:**
#     Collection of items, but no duplicates allowed and order doesn't matter. Like a bag where you only keep unique things.
#     Use curly braces `{}` or `set()`.

#     *Example:* `unique_numbers = {1, 2, 3, 3, 4}`, `tags = {"python", "programming", "developer"}`

# *   **NoneType (None): Represents the absence of a value. ∅ Empty Box:**
#     Special type meaning 'no value'.  Like an empty box or when something is missing.

#     *Example:* `result = None`, `user_address = None` (if address is not available).

# **Visual Representation:**

# ```
# Data Types Table:

# | Data Type      | Description                                  | Analogy                  | Emoji(s)       | Example                     |
# |----------------|----------------------------------------------|--------------------------|----------------|-----------------------------|
# | Integer (int)  | Whole numbers                                | Counting Blocks          | 🔢              | `10, -5, 0`                  |
# | Float (float)  | Numbers with decimal points                  | Measuring Liquids        | 🔢.decimal      | `3.14, -0.5, 2.0`            |
# | String (str) | Textual data                                 | Words and Sentences      | 💬              | `"hello", 'Python'`          |
# | Boolean (bool) | True or False values                         | Switches - On/Off        | ✅/❌          | `True, False`               |
# | List (list)    | Ordered, mutable collection                  | Shopping Lists           | 📜              | `[1, 2, "apple"]`           |
# | Tuple (tuple)  | Ordered, immutable collection                | Fixed Records            | 🔒📜            | `(1, 2, "apple")`           |
# | Dictionary (dict)| Key-value pairs                             | Dictionaries             | 📒              | `{"key": "value"}`          |
# | Set (set)      | Unordered, unique items                      | Bag of Unique Items      | 🎒              | `{1, 2, 3}`                 |
# | NoneType (None)| Absence of a value                           | Empty Box                | ∅              | `None`                      |
# ```

# **Summary:** Python has different types of data like numbers, text, True/False, lists, tuples, dictionaries, sets, and None.
# Each type is used for different purposes. Understanding data types is key to working with data in Python.

# ---

# ### 2.3 Operators: Performing Actions ➕➖✖️➗ (Action Verbs)

# **Concept:** Operators are like tools to work with data. They tell Python what to DO with the data.
# Like adding, subtracting, comparing, etc.

# **Analogy:** Construction Tools 🛠️ in your Programming Workshop.

# **Explanation:**

# Operators are special symbols that perform operations on values (operands).
# Think of them as verbs in sentences, they tell you what action to perform.

# Let's look at different types of operators:

# *   **Arithmetic Operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`. ➕➖✖️➗ (Math Tools):**
#     These are for basic maths calculations.

#     *   `+` (Addition): Adds two numbers.  *Example:* `x + y`
#     *   `-` (Subtraction): Subtracts one number from another. *Example:* `x - y`
#     *   `*` (Multiplication): Multiplies two numbers. *Example:* `x * y`
#     *   `/` (Division): Divides one number by another. Gives a float result (with decimal). *Example:* `x / y`
#     *   `//` (Floor Division): Divides and gives only the whole number part (integer). *Example:* `x // y`
#     *   `%` (Modulo): Gives the remainder after division. *Example:* `x % y`
#     *   `**` (Exponentiation): Raises a number to a power. *Example:* `x ** y` (x to the power of y)

# *   **Comparison Operators: `==`, `!=`, `>`, `<`, `>=`, `<=`. ⚖️ (Comparison Tools - Is it equal? Bigger?):**
#     These compare two values and give a True or False answer.

#     *   `==` (Equal to): Checks if two values are the same. *Example:* `x == y`
#     *   `!=` (Not equal to): Checks if two values are different. *Example:* `x != y`
#     *   `>` (Greater than): Checks if the first value is bigger. *Example:* `x > y`
#     *   `<` (Less than): Checks if the first value is smaller. *Example:* `x < y`
#     *   `>=` (Greater than or equal to): Checks if the first value is bigger or equal. *Example:* `x >= y`
#     *   `<=` (Less than or equal to): Checks if the first value is smaller or equal. *Example:* `x <= y`

# *   **Assignment Operators: `=`, `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`. ✍️ (Value Update Tools):**
#     `=` is basic assignment. Others are shortcuts to update variables.

#     *   `=` (Assignment): Assigns a value to a variable. *Example:* `x = 10`
#     *   `+=` (Add and assign): Adds and updates the variable. *Example:* `x += 5` (same as `x = x + 5`)
#     *   `-=` (Subtract and assign): Subtracts and updates. *Example:* `x -= 3` (same as `x = x - 3`)
#     *   `*=`, `/=`, `//=`, `%=`, `**=` : Same idea for other operations.

# *   **Logical Operators: `and`, `or`, `not`. AND, OR, NOT gates 🚪 (Logic Tools - Combining Conditions):**
#     Used to combine True/False conditions. Like logic gates in electronics.

#     *   `and` (Logical AND): True only if BOTH conditions are True. *Example:* `condition1 and condition2`
#     *   `or` (Logical OR): True if AT LEAST ONE condition is True. *Example:* `condition1 or condition2`
#     *   `not` (Logical NOT): Reverses the condition. True becomes False, False becomes True. *Example:* `not condition`

# *   **Membership Operators: `in`, `not in`. ∈, ∉ (Checking inside a group):**
#     Check if something is present in a list, tuple, string, etc.

#     *   `in`: True if value is found in the sequence. *Example:* `value in sequence`
#     *   `not in`: True if value is NOT found in the sequence. *Example:* `value not in sequence`

# *   **Identity Operators: `is`, `is not`. 🆔 (Checking if it's the SAME object):**
#     Check if two variables are actually the SAME object in memory.  Less common, but good to know.
#     `==` checks if values are equal, `is` checks if they are the *same thing* in memory.

#     *   `is`: True if both variables point to the same object. *Example:* `variable1 is variable2`
#     *   `is not`: True if they are NOT the same object. *Example:* `variable1 is not variable2`

# **Example:**

x = 10
y = 5

# Arithmetic Operators
sum_xy = x + y        # Addition: 10 + 5 = 15
diff_xy = x - y       # Subtraction: 10 - 5 = 5
product_xy = x * y    # Multiplication: 10 * 5 = 50
quotient_xy = x / y   # Division: 10 / 5 = 2.0 (float)
floor_div = x // y    # Floor Division: 10 // 5 = 2 (integer)
remainder = x % y     # Modulo: 10 % 5 = 0 (remainder is 0)
exponent = x ** y      # Exponentiation: 10 ** 5 = 100000

# Comparison Operators
is_equal = (x == y)   # Equal to: 10 == 5 is False
is_not_equal = (x != y)# Not equal to: 10 != 5 is True
is_greater = (x > y)   # Greater than: 10 > 5 is True
is_less = (x < y)      # Less than: 10 < 5 is False
is_greater_equal = (x >= y) # Greater than or equal to: 10 >= 5 is True
is_less_equal = (x <= y)    # Less than or equal to: 10 <= 5 is False

# Assignment Operators
x += 5                # Add and assign: x becomes 15 (x = x + 5)
y -= 2                # Subtract and assign: y becomes 3 (y = y - 2)

# Logical Operators
is_both_true = True and True   # Logical AND: True and True is True
is_either_true = True or False # Logical OR: True or False is True
is_not_true = not False        # Logical NOT: not False is True

# Membership Operators
numbers = [1, 2, 3, 4, 5]
is_3_in_numbers = 3 in numbers     # Membership: 3 is in [1, 2, 3, 4, 5] - True
is_6_not_in_numbers = 6 not in numbers # Membership: 6 is not in [1, 2, 3, 4, 5] - True

# Identity Operators
list1 = [1, 2, 3]
list2 = list1         # list2 points to the same list as list1
list3 = [1, 2, 3]     # list3 is a NEW list, even if it has same values

is_list1_list2 = (list1 is list2)  # Identity: list1 and list2 are the same object - True
is_list1_list3 = (list1 is list3)  # Identity: list1 and list3 are DIFFERENT objects - False (even values are same)
is_list1_equal_list3 = (list1 == list3) # Equality: list1 and list3 have same VALUE - True

# **Summary:** Operators are tools to perform actions on data.
# We have arithmetic, comparison, assignment, logical, membership, and identity operators.
# Each type does different things. Mastering operators is like learning to use different tools in your toolkit.

# ---

# **Congratulations, Developer!** You have finished Chapter 2 on Data Fundamentals! 🏗️
# You now know about variables, data types, and operators – the basic building blocks of Python.
# Keep practicing these concepts. Next chapter, we'll see how to use these blocks to build something more interesting.
# All the best! 👍