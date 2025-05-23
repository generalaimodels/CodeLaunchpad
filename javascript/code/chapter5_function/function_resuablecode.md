Okay boss, let's jump into Chapter 5: Functions - Reusable Code Blocks. Functions are a cornerstone of programming; they help you organize your code, make it reusable, and easier to understand. We'll cover everything in detail, so you'll become a function expert!

**Chapter 5: Functions - Reusable Code Blocks**

**1. Defining Functions (Using `function` keyword)**

Think of a function as a mini-program inside your program. You give it a name, and it does a specific job. You can "call" this mini-program whenever you need it.

```javascript
function functionName(parameters) {
    // Code to be executed
    // return value (optional)
}
```

*   **`function` keyword:** It tells JavaScript that you're defining a function.
*   **`functionName`:** The name you give to your function. Follow the same naming conventions as variables (camelCase recommended).
*   **`parameters` (optional):**  Variables that are passed into the function. They act as placeholders for values you'll give when you call the function.
*   **`{ ... }`:** The code block of the function—the set of instructions that will be executed.
*   **`return` (optional):** If the function needs to give a value back to where it was called, use the `return` keyword followed by the value.

*Example*

```javascript
function greet(name) {
    console.log("Hello, " + name + "!");
}

greet("Priya"); // Calling the function
greet("Rohan");
```

*   Output
    ```
    Hello, Priya!
    Hello, Rohan!
    ```
*   Here we have defined a `greet` function which takes one argument `name`. When we call this function by passing different name argument, the function will print `"Hello, <name>!"`.

**2. Function Parameters and Arguments**

*   **Parameters:** These are the variables listed in the function definition within the parentheses. They are like placeholders for the actual values that will be passed when the function is called.
*   **Arguments:** These are the actual values passed to the function when you call it. They fill the placeholders of parameters.

```javascript
function add(a, b) { // a and b are parameters
    return a + b;
}

let sum = add(10, 20); // 10 and 20 are arguments
console.log(sum); // output: 30
```

*   In the `add` function, `a` and `b` are parameters.
*   When we call `add(10, 20)`, `10` is the argument that corresponds to parameter `a`, and `20` is the argument for parameter `b`.

**Key points about parameters and arguments**

*   Functions can have zero or more parameters.
*   The order of arguments matters. The first argument will map to the first parameter, and so on.
*   If you pass fewer arguments than the function expects, the remaining parameters will be set to `undefined`.
*   If you pass more arguments than the function expects, the extra arguments are ignored

**3. Returning Values from Functions**

Functions can return values using the `return` keyword. The `return` statement does two things:

1.  It specifies the value that the function will return.
2.  It immediately stops the function's execution and returns control to where the function was called.

```javascript
function multiply(x, y) {
    return x * y;
}

let result = multiply(5, 4);
console.log(result); // output: 20
```

*   The `multiply` function calculates the product of `x` and `y` and returns it.
*   The returned value is stored in the variable `result`.
*   If function does not return a value, it will return `undefined` by default.

**4. Function Expressions and Anonymous Functions**

Besides defining functions using the `function` keyword, you can also create functions as expressions.

*   **Function Expression:** Creating a function and assigning it to a variable

    ```javascript
    let add = function(a, b) { // function expression assigned to variable add
        return a + b;
    }

    console.log(add(5, 10));
    ```

*   Here, we are creating a function with parameter `a`, `b` and returning their sum. This function is assigned to the variable `add`.
*   **Anonymous Function:** A function without a name. Often used with function expressions or as callbacks (we'll discuss callbacks later).

    ```javascript
    setTimeout(function() {
        console.log("This is an anonymous function");
    }, 1000);
    ```
    * Here we are passing anonymous function as an argument to the setTimeout function.

**5. Example (from your instructions):**

```javascript
function add(a, b) {
    return a + b;
}

console.log(add(5, 3)); // output: 8
```

*   This defines a function named `add` with parameters `a` and `b`.
*   It returns the sum of `a` and `b`.
*   When we call `add(5, 3)`, it returns `8`, which is then printed to the console.

**Expected Outcome:**

By the end of this chapter, you should be able to:

*   Define your own functions using the `function` keyword.
*   Understand how to use function parameters and arguments.
*   Return values from your functions.
*   Create function expressions and use anonymous functions.
*  Write modular and reusable code.

That's it for Chapter 5, boss! You're now equipped with the power of functions! They are essential for writing clean and maintainable code. Practice defining and using different types of functions. Any doubts? Feel free to ask, we are here to help. Next, we will start intermediate javascript. Are you ready? 🚀
