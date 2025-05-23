Okay boss, let's get into Chapter 3: Control Flow - Making Decisions. This is where your code starts to get smart, making choices based on different conditions. We'll cover everything in detail, so you can confidently use conditional statements.

**Chapter 3: Control Flow - Making Decisions**

**1. Conditional Statements: `if`, `else if`, `else`**

Think of conditional statements as decision-making tools for your code. They allow you to execute different blocks of code based on whether a condition is true or false.

*   **`if` statement:** Executes a block of code *only if* a condition is true.

    ```javascript
    if (condition) {
        // Code to execute if condition is true
    }
    ```

    *   The `condition` inside the parentheses is an expression that evaluates to either `true` or `false`.

    *Example*

    ```javascript
      let isLoggedIn = true;
        if(isLoggedIn) {
            console.log("User is logged in");
        }
    ```
    Output :
    ```
    User is logged in
    ```

*   **`else` statement:** Executes a block of code if the `if` condition is *false*.

    ```javascript
    if (condition) {
        // Code to execute if condition is true
    } else {
        // Code to execute if condition is false
    }
    ```

*   *Example*

    ```javascript
    let temperature = 10;
    if(temperature > 20) {
        console.log("It is hot")
    } else {
        console.log("It is cold")
    }
    ```
    Output:
    ```
    It is cold
    ```
*   **`else if` statement:** Allows you to check multiple conditions in a sequence. If the `if` condition is false, it checks the `else if` condition and so on.

    ```javascript
    if (condition1) {
        // Code to execute if condition1 is true
    } else if (condition2) {
        // Code to execute if condition1 is false and condition2 is true
    } else {
        // Code to execute if all conditions are false
    }
    ```

    *Example*

    ```javascript
    let score = 85;
    if (score >= 90) {
      console.log("Grade A");
    } else if (score >= 80) {
        console.log("Grade B");
    } else if (score >= 70) {
      console.log("Grade C");
    } else {
        console.log("Grade D");
    }

    ```
    Output:
    ```
    Grade B
    ```

**Important Points:**

*   You can have multiple `else if` statements.
*   The `else` statement is optional; you don't always need it.
*   Once one condition is found to be true, the corresponding block of code is executed, and the rest of the conditions are skipped.

**2. Switch Statements**

The `switch` statement is another way to make decisions. It's often used when you have a variable that can have multiple possible values, and you need to execute a different block of code for each value.

```javascript
switch (expression) {
    case value1:
        // Code to execute if expression === value1
        break;
    case value2:
        // Code to execute if expression === value2
        break;
    ...
    default:
        // Code to execute if no case matches
}
```

*   The `expression` inside the parentheses is evaluated once.
*   The value of the `expression` is compared to the `value` in each `case`.
*   If a match is found, the corresponding block of code is executed.
*   The `break` statement is used to exit the `switch` statement after a match. Without `break`, the code will continue to execute the next `case`.
*   The `default` case is executed if no other cases matches

*Example*

```javascript
let day = "Monday";

switch (day) {
    case "Monday":
        console.log("It's the start of the week");
        break;
    case "Friday":
        console.log("Weekend is near")
        break;
    case "Saturday":
    case "Sunday":
        console.log("It's weekend");
        break;
    default:
        console.log("It's a normal working day");
}
```

Output:
```
It's the start of the week
```

**Important Points:**

*   `switch` statements are good when you have many conditions to check against one expression.
*   You can also use `switch` statement without default statement
*   `break` statements are crucial in `switch` statements, otherwise the execution will "fall through" to the next `case`.
*  `switch` uses strict equality `===` for comparison.
*   You can group multiple `case` statements together to execute the same code block.
    *  In the above example, if the value of `day` is either "Saturday" or "Sunday", then the statement `"It's weekend"` will get printed.

**3. Example:**

Let's look at the example you provided:

```javascript
let age = 20;
if (age >= 18) {
    console.log("Eligible to vote");
} else {
    console.log("Not eligible to vote");
}
```

*   Here, we declare a variable `age` and set its value to `20`.
*   The `if` statement checks if `age` is greater than or equal to `18`. Because `20 >= 18` is true, the code inside the `if` block is executed, and the console displays "Eligible to vote".
*   If the value of age was less than `18`, then "Not eligible to vote" would have printed.

**Expected Outcome:**

You should now be able to:

*   Use `if`, `else if`, and `else` statements to create decision-making logic in your code.
*   Use `switch` statements for multiple cases.
*   Write code that reacts to different conditions.

That's all for Chapter 3, boss. This is a very important concept for writing actual logic in your program. Remember, practice is very important. Feel free to ask any questions. We are always here to help you out. Ready for next chapter? 🚀
