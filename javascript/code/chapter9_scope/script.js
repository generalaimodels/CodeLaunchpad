/**
 * 🚀 Chapter 9: Scope and Closures in JavaScript 🕵️‍♂️🔒
 *
 * Scope and closures are fundamental concepts in JavaScript that govern variable accessibility 🌐
 * and memory management 🧠. Mastering them is crucial for writing robust and maintainable code.
 *
 *  🌟 Key Concepts:
 *  1.  🌐 Global vs. Local Scope:  Understanding where variables are accessible based on their declaration location.
 *  2.  🧱 Block Scope (`let`, `const` vs. `var`):  Exploring the impact of different variable declarations on scope within blocks.
 *  3.  🔒 Closures:  Delving into functions remembering their lexical environment, even after the outer function has executed.
 *
 * Let's demystify these concepts with detailed code examples and analogies! 🚀
 */

/**
 * ------------------------------------------------------------------------------------------
 * 1. 🌐 Global vs. Local Scope: Variable Visibility Zones 🗺️
 * ------------------------------------------------------------------------------------------
 *
 * Scope defines the context in which variables are accessible. It's like visibility zones for variables.
 *
 * a) Global Scope: Variables declared outside any function or block ({}) live in the global scope.
 *    - Accessible from anywhere in the JavaScript code: inside functions, outside functions, everywhere! 🌍
 *    - Think of it as a public space where everyone can access and see everything. 🏞️
 *    - **Caution**: Excessive global variables can lead to naming collisions and make code harder to manage and debug. 🐞
 *
 * b) Local Scope (Function Scope): Variables declared inside a function have local scope.
 *    - Accessible only within that function and any functions nested inside it (lexical scope). 🏠
 *    - Think of it as a private space within a house, only accessible to those inside. 🚪
 *    - Helps in encapsulation and prevents naming conflicts, promoting modularity. 🧩
 *
 *  🌳 Scope Hierarchy (Global as root, functions as branches):
 *
 *      Global Scope 🌍
 *      └── Function Scope 1 🏠
 *          └── Function Scope 2 🏠 (nested)
 *      └── Function Scope 3 🏠
 */

// 🍎 Example 1: Global Scope Variable
let globalVar = "I am global"; // Declared outside any function - Global Scope 🌐
console.log("Global variable outside function:", globalVar); // Output: I am global - Accessible here

function myFunctionGlobalScope() {
    console.log("Global variable inside function:", globalVar); // Output: I am global - Accessible inside too!
}

myFunctionGlobalScope(); // Calling the function
console.log("Global variable again outside function:", globalVar); // Output: I am global - Still accessible


// 🍌 Example 2: Local Scope Variable
function myFunctionLocalScope() {
    let localVar = "I am local"; // Declared inside the function - Local Scope 🏠
    console.log("Local variable inside function:", localVar); // Output: I am local - Accessible here
}

myFunctionLocalScope(); // Calling the function - local variable is accessible during function execution

// console.log("Local variable outside function:", localVar); // ❌ Error! localVar is not defined - Not accessible outside its function scope!

/**
 * ⚠️ Important Notes about Scope:
 * - Global variables are accessible everywhere, but minimize their use for better code organization. 🌐
 * - Local variables are confined to their function, promoting encapsulation and reducing naming conflicts. 🏠
 * - Scope helps in managing variable lifecycle and accessibility, essential for modular and maintainable code. 🧩
 */

/**
 * ------------------------------------------------------------------------------------------
 * 2. 🧱 Block Scope (`let`, `const` vs. `var`): Scope Declaration Deep Dive 🧱
 * ------------------------------------------------------------------------------------------
 *
 * Variable declarations using `var`, `let`, and `const` have different scoping behaviors, especially within blocks (`{}`).
 * Blocks are code enclosed in curly braces, like in `if` statements, `for` loops, etc. { ... }
 *
 * a) `var`: Function-Scoped or Global-Scoped ⚙️
 *    - If declared inside a function, `var` has function scope (like local scope discussed above). 🏠
 *    - If declared outside any function, `var` has global scope. 🌍
 *    - **Hoisting**: `var` declarations are hoisted to the top of their scope (function or global). This means you can *use* a `var` variable before its declaration line in code, but it will be `undefined` until the declaration line is actually executed.  👻 (More on hoisting later if needed).
 *    - **Redeclaration**: `var` allows redeclaration within the same scope (within a function or global). This can lead to confusion and bugs. 🐛
 *
 * b) `let` and `const`: Block-Scoped 🧱
 *    - Introduced in ES6 (ECMAScript 2015) to provide block scope. 🎉
 *    - Variables declared with `let` or `const` are only accessible within the block they are defined in and any nested blocks. {}
 *    - **No Hoisting (Temporal Dead Zone):** `let` and `const` declarations are also hoisted, but they are not initialized. Accessing them before the declaration line results in a `ReferenceError`, creating a "Temporal Dead Zone" (TDZ). 🚧  This is actually a good thing as it prevents accidental usage before declaration.
 *    - **No Redeclaration**: `let` and `const` do not allow redeclaration within the same scope (block, function, or global). This helps prevent accidental variable overwrites and naming conflicts, leading to more predictable code. ✅
 *    - `const` is for constants: Once assigned, its value cannot be reassigned (for primitive values, object bindings are constant, but object properties can be changed).  🔒
 *
 *  🌳 Block Scope Visualization (`let` and `const` create block boundaries):
 *
 *      Function Scope (for 'var' if inside function) 🏠  OR  Global Scope (for 'var' outside function) 🌍
 *      └── Block 1 {} (for 'let' and 'const') 🧱
 *          └── Block 2 {} (nested - for 'let' and 'const') 🧱
 *      └── Block 3 {} (for 'let' and 'const') 🧱
 */

console.log("\n--- Block Scope Exploration ---");

// 🍎 Example 1: `var` - Function Scope (or Global if outside function) - No Block Scope for 'var'
var var_x = 10; // Global scope 'var' 🌍

function testVarScope() {
    var var_x = 20; // Function scope 'var' - *shadows* the global 'var_x' within this function 🏠
    console.log("Inside testVarScope function - var_x:", var_x); // Output: 20 - Function-scoped 'var_x'
}

testVarScope(); // Call the function
console.log("Outside function - global var_x:", var_x); // Output: 10 - Global 'var_x' is unchanged. Function scope 'var_x' did not overwrite it in global scope.

if (true) {
    var var_y = 30; // 'var' inside block - but 'var' is NOT block-scoped! It's function-scoped (or global if outside function).
}
console.log("var_y after if block:", var_y); // Output: 30 - 'var_y' is accessible outside the if block!  ❌  'var' ignores block scope rules (when outside a function).

// 🍌 Example 2: `let` and `const` - Block Scope Enforcement 🧱
let let_a = 100;  // Global scope 'let' 🌐
const const_PI = 3.14; // Global scope 'const' 🌐

if (true) {
    let let_a = 200; // Block scope 'let' - only within this if block 🧱 - *shadows* global 'let_a' within this block.
    const const_PI = 3.14159; // Block scope 'const' - only within this if block 🧱 - *shadows* global 'const_PI' within this block.
    console.log("Inside if block - let_a:", let_a); // Output: 200 - Block-scoped 'let_a'
    console.log("Inside if block - const_PI:", const_PI); // Output: 3.14159 - Block-scoped 'const_PI'
}

console.log("Outside if block - global let_a:", let_a); // Output: 100 - Global 'let_a' is unchanged. Block scope 'let_a' did not overwrite it in global scope.
console.log("Outside if block - global const_PI:", const_PI); // Output: 3.14 - Global 'const_PI' is unchanged. Block scope 'const_PI' did not overwrite it in global scope.

// console.log("const_PI = 3.142"); // ❌ Error! Assignment to constant variable. - 'const' cannot be reassigned after initial assignment.

/**
 * ✅ Key Differences Summary (`var` vs. `let` vs. `const`):
 * | Feature         | `var`                   | `let`                      | `const`                    |
 * |-----------------|-------------------------|---------------------------|----------------------------|
 * | Scope           | Function or Global       | Block                      | Block                      |
 * | Hoisting        | Yes (initialized to undefined) | Yes (Temporal Dead Zone) | Yes (Temporal Dead Zone) |
 * | Redeclaration   | Yes (within function)   | No                         | No                         |
 * | Reassignable    | Yes                     | Yes                        | No (after initialization)   |
 *
 *  💡 Best Practices:
 *  - Prefer `let` and `const` over `var` in modern JavaScript for block scope and predictable behavior. 👍
 *  - Use `const` by default for variables that should not be reassigned. 🔒
 *  - Use `let` for variables that may need to be reassigned within their scope. 🛠️
 *  - Avoid `var` unless you have specific legacy code reasons or fully understand its function scope behavior. 🙅‍♂️
 */

/**
 * ------------------------------------------------------------------------------------------
 * 3. 🔒 Understanding Closures and Their Uses: Remembering the Environment 🧠
 * ------------------------------------------------------------------------------------------
 *
 * A closure is a function bundled together with its lexical environment.  Lexical environment is essentially
 * the scope in which the function was declared.  This environment consists of any variables that were in-scope
 * at the time the function was created.
 *
 * In simpler terms:  A closure is a function that "remembers" variables from its surrounding scope, even after
 * that outer scope has finished executing. It's like a function carrying a backpack 🎒 of variables it needs.
 *
 * Key Characteristics of Closures:
 * - Inner function: Closures are typically created by defining a function inside another function (outer function).
 * - Lexical Environment Capture: The inner function "closes over" or captures variables from the outer function's scope.
 * - Persistence: Even after the outer function completes, the inner function (closure) retains access to the captured variables. 🕰️
 *
 *  🌳 Closure Formation Process:
 *
 *      Outer Function Execution 🚀
 *      ├── Variable Declaration (in outer scope) 📦
 *      ├── Inner Function Definition (closure formed here, capturing outer scope) 🔒
 *      └── Outer Function Returns Inner Function 🎁
 *
 *      Later, Inner Function (Closure) is Called 📞
 *      └── Closure Accesses and Uses Captured Variables from Outer Scope 🔑
 */

console.log("\n--- Closures Demystified ---");

// 🍎 Example 1: Basic Closure - Inner function remembers outer variable
function outerFunctionClosureBasic() {
    let outerVarClosure = "I am from outer function scope"; // Variable in outer scope 📦

    function innerFunctionClosureBasic() { // Inner function - closure is formed here 🔒
        console.log("Closure inner function accessing outerVarClosure:", outerVarClosure); // Accessing 'outerVarClosure' - closure in action! 🔑
    }

    return innerFunctionClosureBasic; // Return the inner function - the closure 🎁
}

const closureFuncBasic = outerFunctionClosureBasic(); // Call outer function, get back the inner function (closure)
closureFuncBasic(); // Output: Closure inner function accessing outerVarClosure: I am from outer function scope - Closure remembers 'outerVarClosure'! 🕰️


// 🍌 Example 2: Closure with Counter - Data Privacy and State Preservation 🛡️
function createCounterClosure() {
    let countClosure = 0; // Private variable within createCounterClosure scope 🤫

    return { // Returning an object with methods (functions that are closures) 🎁
        incrementClosure: function() { // Closure 1 - remembers and modifies 'countClosure' 🔒
            countClosure++;
        },
        getCountClosure: function() { // Closure 2 - remembers and accesses 'countClosure' 🔒
            return countClosure;
        }
    };
}

const counterClosureInstance = createCounterClosure(); // Get the counter object with closure methods
counterClosureInstance.incrementClosure(); // Increment the counter using closure method
counterClosureInstance.incrementClosure(); // Increment again
console.log("Counter value from closure:", counterClosureInstance.getCountClosure()); // Output: 2 - Closure remembers and maintains 'countClosure' state! 🕰️
// console.log("Direct access to countClosure:", counterClosureInstance.countClosure); // ❌ undefined - 'countClosure' is private, not directly accessible from outside! Data privacy achieved! 🛡️

/**
 * ✅ Use Cases of Closures:
 * 1. Data Privacy & Encapsulation:  Create private variables and methods, as shown in the counter example. 🛡️
 * 2. State Preservation:  Maintain state across function calls, useful for counters, memoization, etc. 🕰️
 * 3. Function Factories: Create functions dynamically with pre-set configurations (partial application, currying - advanced topics). 🏭
 * 4. Event Handlers and Callbacks: Closures are commonly used in event handlers and callbacks to access relevant data from their creation scope. 🖱️
 *
 * 🔑 Key Takeaways about Closures:
 * - Functions remember their lexical scope (environment). 🧠
 * - Closures enable data privacy and stateful behavior. 🛡️🕰️
 * - They are a powerful feature of JavaScript for creating modular and maintainable code. 🧩
 */

/**
 * 🎉 Congratulations! 🎉 You've now conquered Scope and Closures! 🏆
 *
 * You can now:
 * - Differentiate between global and local scope. 🌐 🏠
 * - Understand block scope and the differences between `var`, `let`, and `const`. 🧱 ⚙️ 🎉
 * - Explain closures and how they work, including their use cases. 🔒 🧠 🛡️
 * - Identify and use closures to solve problems and create more robust code. 🛠️
 *
 * Scope and closures are indeed more intricate concepts, but with practice and these explanations,
 * you're well-equipped to handle them!  Keep experimenting and coding! 🚀✨
 *
 *  Next stop:  ➡️ DOM Manipulation! 🖱️ Let's move to browser environments and interactive web pages! 🌐💻
 */