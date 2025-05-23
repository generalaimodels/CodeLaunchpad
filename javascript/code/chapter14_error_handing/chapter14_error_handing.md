Okay boss, let's get into Chapter 14: Error Handling. This is super important because even the best code can have errors. Knowing how to handle errors gracefully makes your code more robust, user-friendly, and easier to debug. We'll learn about `try...catch`, how to throw custom errors, and some basic debugging techniques.

**Chapter 14: Error Handling**

**1. `try`, `catch`, and `finally` Blocks**

*   **Error Handling:** Error handling is the process of anticipating and responding to errors that may occur during the execution of your code.
*   **`try...catch` Block:** This is the primary mechanism for handling errors in JavaScript.
    *   The code that might throw an error is placed inside the `try` block.
    *   If an error occurs within the `try` block, the execution jumps to the `catch` block, where you can handle the error.

    ```javascript
    try {
      // Code that may throw an error
      let result = 10 / 0; // This will cause an error
      console.log("Result:" , result) // This code will not execute
    } catch (error) {
      // Code to handle the error
      console.error("An error occurred:", error);
    }
    ```

*   **Explanation:**
    *   If an error occurs in `try` block (e.g., dividing by zero), the rest of the code within the `try` block is skipped.
    *   The `catch` block will be executed and the error object is available as an argument (`error`) to the callback function.
    *  If no error occurred, then `catch` block will not be executed.

*   **`finally` Block:** The `finally` block is optional and is used to execute code whether or not an error occurred.
    *   It is typically used to perform cleanup actions (like releasing resources).
    *   The code inside `finally` will be always executed whether there is error or not.

    ```javascript
      try {
        // Code that may throw an error
        let result = 10 / 2;
        console.log("Result:" , result)
      } catch (error) {
        console.error("An error occurred:", error);
      } finally {
        console.log("Finally block executed"); // This will be executed even if error occur or not.
      }
      ```
     *  Here the code inside `try` block will get executed without throwing error.
     *   So `catch` block will not be executed.
     *   But the code inside `finally` block will get executed.

*   `try` block must have either `catch` block or `finally` block.
*   You can have both `catch` and `finally` block.

**2. Throwing Custom Errors**

*   You can create and throw your own errors using the `throw` keyword.
*   This is useful when you need to signal specific types of errors in your code.

    ```javascript
    function validateAge(age) {
      if (age < 0) {
        throw new Error("Age cannot be negative.");
      }
      if (age > 150) {
        throw new Error("Age cannot be greater than 150.")
      }
      console.log("Age is valid: ", age);
    }

    try {
      validateAge(-5); // Throwing custom error
    } catch(error) {
      console.error("Error:", error) // Output:  Error: Error: Age cannot be negative.
    }
    try {
      validateAge(160) // Throwing custom error
    } catch(error) {
      console.error("Error:", error) // Output:  Error: Error: Age cannot be greater than 150.
    }

    try {
      validateAge(30)
    } catch(error) {
      console.error("Error:", error)
    }
    ```

*   In the above example, if the age is invalid then `throw` statement will throw an error.
*   This error can be handled in the `catch` block.
*   If there is no error, then `catch` block will not execute.

**3. Debugging JavaScript Code**

Debugging is the process of finding and fixing errors in your code.

*   **`console.log()`:** The most common method for debugging.
    *   You can insert `console.log()` statements at various points in your code to check variable values, program flow, and verify output.

        ```javascript
            function add(a,b) {
              console.log("value of a: ", a)
              console.log("value of b: ", b)
              let sum = a + b;
              console.log("sum: ", sum)
              return sum;
            }
            add(5,10)
        ```

*   **Browser Developer Tools:** Modern browsers have powerful developer tools that include a debugger.
    *   You can set breakpoints in your code, step through each line, inspect variable values, etc.
    *   To access developer tool:
         *  **Chrome:** Right-click on page and select `Inspect` or `Inspect Element`. Go to `Sources` tab.
         *  **Firefox:** Right-click on the page and select `Inspect Element` and go to `Debugger` tab.

    *   **Breakpoints:** Set breakpoints at specific lines in your code and then run the program. When the execution reaches the breakpoint, the code will pause, and you can inspect variables, and go line by line.

**Basic debugging steps:**

1.  **Identify the Issue:** Understand the error, reproduce it and look for error messages.
2.  **Isolate the Problem:** Try to isolate the part of the code that's causing the error.
3.  **Use Debugging Tools:** Use `console.log()` or browser debugger to inspect variables and program flow.
4.  **Fix the Error:** Make changes and re-run the code.
5.  **Test Thoroughly:** Once the error is fixed, test your code to be sure it works.

**Expected Outcome:**

You should now be able to:

*   Use `try`, `catch`, and `finally` blocks to handle errors.
*   Create and throw custom errors when needed.
*   Use debugging tools and techniques to find and fix errors.
*  Write more reliable and robust code that handles errors gracefully.

That’s all for Chapter 14, boss! Error handling and debugging are essential skills for any developer. Practice and try out the tools to be comfortable with this concept. Let me know when you have any questions. We are now moving towards object oriented programming. Let's go! 🚀
