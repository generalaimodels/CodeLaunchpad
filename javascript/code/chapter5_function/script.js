// Chapter 5: Functions - Reusable Code Blocks 🧱

// 🚀 Welcome to the world of Functions! Think of them as magic boxes 📦.
// You put something in (optional), they do some work, and maybe give you something back! 🎁

// --------------------------------------------------------------------------
// 1. Defining Functions (Using `function` keyword) 🏷️
// --------------------------------------------------------------------------

// Imagine you have a robot 🤖 that does a specific task, like saying "Hello!".
// You need to give it a name and instructions. That's what defining a function is!

// Syntax:
// function functionName(parameters) {
//     // ⚙️ Code to be executed (instructions for the robot)
//     // return value; (optional - what the robot gives back) 🎁
// }

// - `function` keyword:  🔑 This word tells JavaScript "Hey! I'm creating a function!".
// - `functionName`:  🏷️  Give your function a name, like 'greetRobot' or 'calculateSum'.
//                     Use camelCase (like variable names) - e.g., `myFunctionName`.
// - `(parameters)`:  🧽  Optional placeholders for information you give to the function.
//                     Think of them as input slots. You can have zero, one, or many!
// - `{ ... }`:  📦  This is the function's BODY!  All the code that runs when you use the function goes here.
// - `return`:  🎁  Optional! If you want your function to give back a result, use `return`.
//             If you don't use `return`, the function does its job but doesn't give anything back explicitly (it returns `undefined` implicitly).

// Example: Let's create a function to say "Hello" to someone! 👋

function sayHello(name) { // 'name' is a parameter - a placeholder for the person's name
    console.log("Hello, " + name + "!"); // 👋  Instructions: Print a greeting!
}

// Let's use our function!  We "call" or "invoke" the function by its name:
sayHello("Alice"); //  Calling sayHello with the argument "Alice"
sayHello("Bob");   //  Calling sayHello with the argument "Bob"
sayHello("Coder"); //  Calling sayHello with the argument "Coder"

// Output:
// Hello, Alice!
// Hello, Bob!
// Hello, Coder!

// Explanation:
// - We defined a function called `sayHello`.
// - It takes one parameter: `name`.
// - Inside the function, it uses `console.log` to display a greeting message, including the `name`.
// - When we call `sayHello("Alice")`, "Alice" becomes the value for the `name` parameter inside the function.
// - The function then executes its code, printing "Hello, Alice!".
// - The same happens for "Bob" and "Coder".

// --------------------------------------------------------------------------
// 2. Function Parameters and Arguments 🧽 ➡️ 📦
// --------------------------------------------------------------------------

// Let's dive deeper into Parameters and Arguments. They are like INPUT and VALUES.

// - Parameters (Placeholders): 🧽
//   - Variables listed inside the parentheses `()` in the function DEFINITION.
//   - They are like labels or names for the inputs the function expects.
//   - Think of them as empty slots waiting to be filled with values.

// - Arguments (Actual Values): ➡️📦
//   - The actual values you PASS to the function when you CALL it.
//   - They fill in the parameter slots.

// Example:  `add` function

function add(number1, number2) { // `number1` and `number2` are PARAMETERS (placeholders) 🧽🧽
    return number1 + number2;       // Calculate the sum and RETURN it 🎁
}

let sumResult = add(5, 10); // 5 and 10 are ARGUMENTS (actual values) ➡️📦📦
console.log("Sum:", sumResult); // Output: Sum: 15

// Visual Representation:

// Function Definition:  function add(number1, number2)  🧽🧽
// Function Call:      add(5, 10)                        ➡️📦📦

//  Parameter `number1` gets the Argument `5`.
//  Parameter `number2` gets the Argument `10`.

// Key Points about Parameters and Arguments:

// 🔑 1. Zero or More Parameters:
function doSomething() { // No parameters!
    console.log("Doing something!");
}
doSomething(); // Calling with no arguments

function greetPerson(firstName, lastName) { // Two parameters
    console.log("Greetings,", firstName, lastName + "!");
}
greetPerson("John", "Doe"); // Calling with two arguments

// 🔑 2. Order Matters!  Like positions in a line. 🧍🧍‍♀️🧍‍♂️
function subtract(firstNumber, secondNumber) {
    return firstNumber - secondNumber;
}
console.log("Subtraction:", subtract(10, 3)); // 10 is firstNumber, 3 is secondNumber. Output: 7
console.log("Subtraction:", subtract(3, 10)); // 3 is firstNumber, 10 is secondNumber. Output: -7

// 🔑 3. Fewer Arguments than Parameters?  Missing arguments become `undefined`. 🤔
function power(base, exponent) {
    console.log("Base:", base);       // Let's see the base value
    console.log("Exponent:", exponent); // Let's see the exponent value
    return Math.pow(base, exponent);
}
console.log("Power:", power(2,4)); // Only provided one argument (base = 2, exponent = undefined)
// Output:
// Base: 2
// Exponent: undefined
// Power: NaN (Not a Number) - because you can't raise to the power of `undefined` in a meaningful way.

// 🔑 4. More Arguments than Parameters? Extra arguments are IGNORED.  🤫
function multiplyTwoNumbers(x, y) {
    console.log("First Number:", x);
    console.log("Second Number:", y);
    return x * y;
}
console.log("Multiplication:", multiplyTwoNumbers(4, 5, 6, 7)); // Passed 4 arguments, but function expects only 2.
// Output:
// First Number: 4
// Second Number: 5
// Multiplication: 20
// Arguments 6 and 7 are ignored in this case.

// --------------------------------------------------------------------------
// 3. Returning Values from Functions 🎁
// --------------------------------------------------------------------------

// Functions can give back results!  Think of it like a vending machine. 🍫 You put money in (arguments),
// it does some processing (function code), and gives you a snack back (returned value). 🎁

// The `return` keyword is the key! 🔑

// 1. Specifies the value to be returned:  `return value;`  🎁
// 2. Stops function execution IMMEDIATELY:  Once `return` is hit, the function stops and goes back to where it was called. 🏃‍♂️💨

function calculateArea(length, width) {
    let area = length * width; // Calculate the area
    return area;              // Return the calculated area 🎁
    console.log("This line will NEVER be executed!"); // Code after `return` is unreachable! 🚫
}

let rectangleArea = calculateArea(10, 5); // Call the function and store the returned value
console.log("Area of rectangle:", rectangleArea); // Output: Area of rectangle: 50

// What if a function doesn't have a `return` statement? 🤔

function sayGreeting(personName) {
    console.log("Good day,", personName + "!"); // Just prints a greeting, no `return`
}

let greetingMessage = sayGreeting("Eve");
console.log("Function return value:", greetingMessage); // Output: Function return value: undefined

// If a function doesn't explicitly `return` a value, it implicitly returns `undefined`.

// --------------------------------------------------------------------------
// 4. Function Expressions and Anonymous Functions 🎭
// --------------------------------------------------------------------------

// There are different ways to create functions in JavaScript! 🎭

// ➡️ Function Declaration (what we've seen so far): Using the `function` keyword and giving it a name.
//    function functionName() { ... }

// ➡️ Function Expression:  Creating a function and assigning it to a VARIABLE. 🏷️

// Example of Function Expression:
let multiply = function(num1, num2) { //  Function is created *without* a name (anonymous function) and assigned to `multiply`
    return num1 * num2;
}; // Semicolon is needed here because it's a variable assignment statement.

console.log("Product:", multiply(7, 3)); //  We can call the function using the variable name `multiply`
// Output: Product: 21

// Anonymous Function:  A function WITHOUT a name. 👻  Often used in function expressions or as callbacks.

// Example of Anonymous Function in `setTimeout` (we'll learn about `setTimeout` later):
setTimeout(function() { //  Anonymous function here - no name after `function` keyword! 👻
    console.log("Delayed message after 1 second!");
}, 1000); // 1000 milliseconds = 1 second

// Explanation:
// - We are passing an anonymous function as an argument to `setTimeout`.
// - `setTimeout` will execute this function after 1000 milliseconds (1 second).
// - Anonymous functions are handy when you need a function for a short, specific task and don't need to reuse it elsewhere with a name.

// --------------------------------------------------------------------------
// 5. Example (from your instructions): ➕
// --------------------------------------------------------------------------

function add(a, b) { // Function definition with parameters `a` and `b`
    return a + b;      // Returns the sum of `a` and `b`
}

console.log(add(5, 3)); // Function call with arguments 5 and 3. Output: 8

// Breakdown:
// 1. `function add(a, b)`:  Defines a function named `add` that takes two parameters, `a` and `b`.
// 2. `return a + b;`:  Inside the function, it calculates the sum of `a` and `b` and returns the result.
// 3. `console.log(add(5, 3));`:
//    - `add(5, 3)`:  Calls the `add` function with arguments `5` and `3`.
//    - The function executes, `a` becomes 5, `b` becomes 3, and it returns `5 + 3 = 8`.
//    - `console.log(8)`:  Prints the returned value `8` to the console.

// --------------------------------------------------------------------------
// 🎉 Congratulations! You've completed Chapter 5: Functions! 🎉
// --------------------------------------------------------------------------

// You are now equipped to:
// ✅ Define functions using `function` keyword.
// ✅ Understand parameters and arguments.
// ✅ Return values from functions.
// ✅ Create function expressions and use anonymous functions.
// ✅ Write reusable and organized code! 🚀

// Practice makes perfect! 🏋️‍♀️ Try writing different functions for various tasks.
// If you have any questions, don't hesitate to ask!  We are here to help you on your coding journey! 🌟

// Ready for Intermediate JavaScript? Let's level up! ⬆️ Are you ready to proceed?  👍 or 👎 ?

// 🚀 Chapter 5: Functions - Reusable Code Blocks (Advanced Edition!) 🌟

// Boss, you're ready to level up your function game! 💪
// We're going beyond the basics and exploring more powerful function concepts.
// Get ready for functions that are even MORE reusable and flexible! ✨

// --------------------------------------------------------------------------
// 6. Function Scope and Closures 🌐🔒
// --------------------------------------------------------------------------

// Imagine functions as little neighborhoods 🏘️. Variables declared inside a function are like residents who live only in that neighborhood.
// This is called "scope" - where variables are accessible.

// - Scope:  Defines the visibility or accessibility of variables and functions.
//   - Local Scope (Function Scope): Variables declared inside a function are local to that function.
//     They can only be accessed within the function itself. 🔒
//   - Global Scope: Variables declared outside any function are global.
//     They can be accessed from anywhere in your code. 🌍

// Example: Scope in action!

let globalVar = "I'm global! 🌍"; // Global variable

function myFunction() {
    let localVar = "I'm local! 🏘️"; // Local variable (function scope)
    console.log("Inside myFunction:");
    console.log("Global variable:", globalVar); // Accessible here (global scope)
    console.log("Local variable:", localVar);  // Accessible here (local scope)
}

myFunction();

console.log("\nOutside myFunction:");
console.log("Global variable:", globalVar); // Accessible here (global scope)
// console.log("Local variable:", localVar); // ❌ Error! localVar is not defined here (local scope of myFunction)

// Output:
// Inside myFunction:
// Global variable: I'm global! 🌍
// Local variable: I'm local! 🏘️
//
// Outside myFunction:
// Global variable: I'm global! 🌍
// Error! localVar is not defined

// Explanation:
// - `globalVar` is declared outside any function, so it's global.  Everyone can access it! 🌍
// - `localVar` is declared inside `myFunction`, so it's local to `myFunction`. Only residents of `myFunction`'s neighborhood know about it! 🏘️
// - Trying to access `localVar` outside `myFunction` results in an error because it's out of scope. 🔒

// 🔑 Closures: Functions with Memory! 🧠
// A closure is a function that "remembers" its surrounding environment (lexical environment) even after the outer function has finished executing.
// It's like a function carrying a backpack 🎒 of variables from its birthplace!

function outerFunction(outerVar) {
    function innerFunction(innerVar) {
        console.log("Outer variable:", outerVar); // innerFunction "closes over" outerVar
        console.log("Inner variable:", innerVar);
    }
    return innerFunction; // Return the inner function!
}

let myInnerFunc = outerFunction("Hello from outer"); // Call outerFunction and get back innerFunction
myInnerFunc("Hello from inner"); // Call innerFunction

let anotherInnerFunc = outerFunction("Greetings from outer space!");
anotherInnerFunc("Greetings from inner earth!");

// Output:
// Outer variable: Hello from outer
// Inner variable: Hello from inner
// Outer variable: Greetings from outer space!
// Inner variable: Greetings from inner earth!

// Explanation of Closure Magic: ✨
// 1. `outerFunction("Hello from outer")` is called.
// 2. `outerVar` inside `outerFunction` becomes "Hello from outer".
// 3. `innerFunction` is defined inside `outerFunction`. Importantly, `innerFunction` *encloses* `outerVar` from its surrounding scope.  This is the closure! 🎒
// 4. `outerFunction` returns `innerFunction`.
// 5. `myInnerFunc` now holds the *returned* `innerFunction`.  Crucially, `myInnerFunc` *still remembers* `outerVar` ("Hello from outer") from its creation environment! 🧠
// 6. When `myInnerFunc("Hello from inner")` is called:
//    - `innerVar` becomes "Hello from inner".
//    - `innerFunction` can still access `outerVar` ("Hello from outer") because of the closure.
// 7. The same happens for `anotherInnerFunc`, but it remembers "Greetings from outer space!" from *its* `outerFunction` call.

// Each time `outerFunction` is called, it creates a *new* closure.  Each returned `innerFunction` has its own private "backpack" 🎒 of remembered variables.

// --------------------------------------------------------------------------