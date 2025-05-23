// 🚀 Chapter 19: Best Practices and Performance in JavaScript 🌟

// 1. Writing Clean Code 🧼
// -----------------------
// Clean Code: Code that is easy to read, understand, and modify. 📖🧠🛠️
// It's like a well-organized room 🗂️ vs. a messy one 🗑️.

// Key Principles of Clean Code: 🔑
//  - 🏷️ Meaningful Names: Use descriptive names for variables, functions, classes.  Names should tell what things are. 🏷️➡️🗣️
//      - ❌ Bad: `let d;`  // What is 'd'? 🤔
//      - ✅ Good: `let daysSinceLastLogin;` // Clear and descriptive! 📅🕰️
//      - ❌ Bad Function: `function process() { ... }` // What does 'process' do? 🤷‍♀️
//      - ✅ Good Function: `function calculateTotalPrice() { ... }` // Explains the function's purpose! 💰➕

// Example 1a: Meaningful Variable Names 🏷️
// ---------------------------------------
// ❌ Unclear names:
let a_1a = 10; // ❌ 'a' - what does it represent?
let b_1a = 20; // ❌ 'b' - what does it represent?
let c_1a = a_1a + b_1a; // ❌ 'c' - what is this?

// ✅ Clear names:
let itemPrice_1a = 10; // ✅ 'itemPrice' - price of an item 💰
let quantity_1a = 20; // ✅ 'quantity' - number of items 🔢
let totalPrice_1a = itemPrice_1a * quantity_1a; // ✅ 'totalPrice' - total cost 💰✖️🔢

// Example 1b: Meaningful Function Names 🏷️
// ---------------------------------------
// ❌ Unclear function name:
function fn_1b(x) { // ❌ 'fn' - doesn't say what it does
    return x * 2;
}

// ✅ Clear function name:
function doubleNumber_1b(number) { // ✅ 'doubleNumber' - clearly doubles a number 🔢*2
    return number * 2;
}

// ------------------------------------------------------------------------

//  - 📏 Consistent Formatting: Use consistent indentation, spacing, line breaks.  Visual consistency improves readability. 📏➡️👁️
//      - ❌ Inconsistent:
/* ❌
if(condition){
  // code
}else {
// more code
}
*/
//      - ✅ Consistent:
/* ✅
if (condition) {
    // code
} else {
    // more code
}
*/

// Example 2a: Consistent Indentation 📏
// -----------------------------------
// ❌ Inconsistent indentation:
/* ❌
function calculateArea_2a(length, width){
if(length > 0) {
return length * width;
}
  return 0;
}
*/

// ✅ Consistent indentation:
function calculateArea_2a(length, width) { // ✅ Consistent indentation makes code structure clear
    if (length > 0) {
        return length * width;
    }
    return 0;
}

// Example 2b: Consistent Spacing 📏
// --------------------------------
// ❌ Inconsistent spacing:
/* ❌
let x_2b=5;let y_2b = 10;
function sum_2b(a,b){return a+b;}
*/

// ✅ Consistent spacing:
let x_2b = 5; // ✅ Space after variable declaration
let y_2b = 10; // ✅ Space before and after operators
function sum_2b(a, b) { // ✅ Space around function parameters and braces
    return a + b; // ✅ Space around return value and operator
}

// ------------------------------------------------------------------------

//  - ✂️ Avoid Long Functions: Break down large functions into smaller, single-responsibility functions.  Small functions are easier to understand and test. ✂️➡️🧩
//      - ❌ Long Function (doing too much):
/* ❌
function processOrder_long() {
  // ... many lines to validate order ...
  // ... many lines to calculate price ...
  // ... many lines to update inventory ...
  // ... many lines to send email ...
}
*/
//      - ✅ Smaller Functions (each does one thing):
/* ✅
function validateOrder() { ... }
function calculatePrice() { ... }
function updateInventory() { ... }
function sendConfirmationEmail() { ... }

function processOrder_clean() { // Orchestrates smaller functions
  if (validateOrder()) {
    const price = calculatePrice();
    updateInventory();
    sendConfirmationEmail();
  }
}
*/

// Example 3a: Refactoring Long Function - Before ✂️
// ------------------------------------------------
// ❌ Long function doing multiple tasks:
function processUserData_3a(userData) { // ❌ Long function - processes user data, validates, saves, sends email
    // ... many lines to validate user data (name, email, etc.) ...
    let isValid = true; // Assume valid initially
    if (!userData.name) isValid = false; // Example validation
    if (!userData.email) isValid = false;

    if (isValid) {
        // ... many lines to save user data to database ...
        // Simulate saving to database
        console.log("Saving user data:", userData);

        // ... many lines to send welcome email ...
        // Simulate sending email
        console.log("Sending welcome email to:", userData.email);
        return true; // Success
    } else {
        console.error("Invalid user data:", userData);
        return false; // Failure
    }
}

// Example 3b: Refactoring Long Function - After ✂️➡️🧩
// -----------------------------------------------
// ✅ Smaller functions with single responsibilities:
function validateUserData_3b(userData) { // ✅ Function to validate user data - single responsibility
    if (!userData.name || !userData.email) {
        console.error("Invalid user data:", userData);
        return false;
    }
    return true;
}

function saveDataToDatabase_3b(userData) { // ✅ Function to save data - single responsibility
    console.log("Saving user data to database:", userData); // Simulate database save
    return true; // Assume success
}

function sendWelcomeEmail_3b(userData) { // ✅ Function to send email - single responsibility
    console.log("Sending welcome email to:", userData.email); // Simulate email sending
    return true; // Assume success
}

function processUserDataRefactored_3b(userData) { // ✅ Orchestrating function - uses smaller functions
    if (validateUserData_3b(userData)) {
        if (saveDataToDatabase_3b(userData)) {
            sendWelcomeEmail_3b(userData);
            return true; // Overall success
        }
    }
    return false; // Overall failure
}

// ------------------------------------------------------------------------

//  - 💬 Comments: Write clear, concise comments for complex logic.  Explain 'why', not just 'what'. 💬➡️❓
//      - ❌ Over-commenting obvious code:
/* ❌
let age = 25; // Declare age variable
age = age + 1; // Increment age by 1
*/
//      - ✅ Commenting complex logic:
/* ✅
// Calculate discount percentage based on customer loyalty level.
// Loyalty levels are: 1-Bronze, 2-Silver, 3-Gold.
// Gold level gets 15% discount, Silver 10%, Bronze 5%.
function calculateDiscount(loyaltyLevel) { ... }
*/

// Example 4a: When to Add Comments - Complex Logic 💬
// --------------------------------------------------
function calculateTax_4a(price, taxRate, isTaxExempt) {
    if (isTaxExempt) {
        return 0; // No tax for tax-exempt customers. Obvious - no comment needed.
    }

    // Apply a special discounted tax rate for prices over $100.
    // This is to promote larger purchases.
    if (price > 100) { // Complex logic - comment explains 'why'
        taxRate *= 0.9; // Reduce tax rate by 10% for prices > $100
    }

    return price * taxRate;
}

// Example 4b: When NOT to Add Comments - Obvious Code 💬🚫
// -----------------------------------------------------
function addNumbers_4b(a, b) {
    // Return the sum of a and b. ❌ Unnecessary comment - code is self-explanatory.
    return a + b;
}

// ------------------------------------------------------------------------

//  - ♻️ DRY (Don't Repeat Yourself): Avoid code duplication.  Reusable code is easier to maintain and update. ♻️➡️🔄
//      - ❌ Code Duplication:
/* ❌
function calculateAreaRectangle() {
  // ... code to calculate rectangle area ...
}
function calculatePerimeterRectangle() {
  // ... almost same code structure as area calculation, duplicated logic ...
}
*/
//      - ✅ Reusable Function:
/* ✅
function calculateRectangleProperty(length, width, propertyType) {
  if (propertyType === 'area') {
    // ... calculate area ...
  } else if (propertyType === 'perimeter') {
    // ... calculate perimeter ...
  }
}
*/

// Example 5a: Code Duplication - Before DRY 👯‍♀️
// --------------------------------------------
// ❌ Duplicated code for formatting names:
function formatFullName_5a(firstName, lastName) { // ❌ Function 1 to format full name
    const formattedName = firstName.trim() + " " + lastName.trim(); // Formatting logic
    return formattedName;
}

function formatUserName_5a(firstName, lastName) { // ❌ Function 2 - similar formatting logic duplicated
    const formattedName = firstName.trim() + "_" + lastName.trim(); // Slightly different formatting
    return formattedName;
}

// Example 5b: Code Reusability - After DRY ♻️🔄
// -------------------------------------------
// ✅ Reusable function for formatting names:
function formatName_5b(firstName, lastName, formatType) { // ✅ Reusable function - formats name based on type
    const trimmedFirstName = firstName.trim();
    const trimmedLastName = lastName.trim();
    let separator = " "; // Default separator (space)

    if (formatType === "username") {
        separator = "_"; // Underscore for username
    } else if (formatType === "initials") {
        return trimmedFirstName.charAt(0) + "." + trimmedLastName.charAt(0) + "."; // Initials format
    }

    return trimmedFirstName + separator + trimmedLastName;
}

// Using the reusable function:
const fullName_5b = formatName_5b("  John  ", "  Doe  ", "full"); // Format as full name
const userName_5b = formatName_5b("  John  ", "  Doe  ", "username"); // Format as username
const initials_5b = formatName_5b("  John  ", "  Doe  ", "initials"); // Format as initials

// ------------------------------------------------------------------------

//  - 😙 KISS (Keep It Simple, Stupid): Prefer simple, straightforward solutions.  Simpler code is easier to understand, debug, and maintain. 😙➡️🧘‍♂️
//      - ❌ Overly Complex Solution:
/* ❌
function complexCalculation() {
  // ... many nested loops, complex conditions, hard to follow ...
}
*/
//      - ✅ Simple Solution:
/* ✅
function simpleCalculation() {
  // ... straightforward logic, easy to understand ...
}
*/

// Example 6a: Complex vs Simple Logic - Before KISS 🤯
// --------------------------------------------------
// ❌ Over-complicated way to check if array contains a value:
function checkValueExistsComplex_6a(array, value) { // ❌ Complex way to check if value exists in array
    let exists = false;
    for (let i = 0; i < array.length; i++) {
        if (array[i] === value) {
            exists = true;
            break; // Break loop once found
        }
    }
    return exists;
}

// Example 6b: Simple and Direct Logic - After KISS 😙🧘‍♂️
// -----------------------------------------------------
// ✅ Simple and direct way using built-in method:
function checkValueExistsSimple_6b(array, value) { // ✅ Simple, direct way using includes()
    return array.includes(value); // Use array.includes() - simple and clear
}

// ------------------------------------------------------------------------

//  - 🔒 Use `const` and `let` over `var`: For better scoping and predictability.  `const` for constants, `let` for variables that change. 🔒➡️✅
//      - ❌ Using `var` (function-scoped, can lead to issues):
/* ❌
var globalVar = "I am global";
function exampleVar() {
  var functionVar = "I am function-scoped";
  if (true) {
    var blockVar = "I am also function-scoped (due to var)";
  }
  console.log(blockVar); // Works - blockVar is function-scoped
}
console.log(globalVar); // Works - globalVar is global
// console.log(functionVar); // Error - functionVar is function-scoped
*/
//      - ✅ Using `const` and `let` (block-scoped, more predictable):
/* ✅
const constantVar = "I am constant"; // Value cannot be reassigned
let blockScopedVar = "I am block-scoped";
function exampleLetConst() {
  let functionLet = "I am function-scoped (let)";
  if (true) {
    const blockConst = "I am block-scoped (const)";
    let blockLet = "I am block-scoped (let)";
    // blockConst = "trying to reassign"; // Error - const cannot be reassigned
  }
  // console.log(blockConst); // Error - blockConst is block-scoped
  // console.log(blockLet); // Error - blockLet is block-scoped
  console.log(functionLet); // Works - functionLet is function-scoped
}
console.log(constantVar); // Works - constantVar is global-like (module-scoped in modules)
// console.log(blockScopedVar); // Works - blockScopedVar is global-like (module-scoped in modules)
*/

// Example 7a: `var` vs `let` in Loops 🔄
// ------------------------------------
// ❌ `var` in loop - closure issue:
function varLoop_7a() {
    for (var i = 0; i < 3; i++) { // var - function-scoped
        setTimeout(function() {
            console.log("var i:", i); // Will print 3, 3, 3 (due to closure over 'var i')
        }, 100);
    }
}

// Example 7b: `let` in Loops - Block Scoping Solves Closure Issue ✅🔒
// ----------------------------------------------------------------
// ✅ `let` in loop - block-scoped, each iteration has its own 'i':
function letLoop_7b() {
    for (let i = 0; i < 3; i++) { // let - block-scoped
        setTimeout(function() {
            console.log("let i:", i); // Will print 0, 1, 2 (each setTimeout has its own 'let i')
        }, 100);
    }
}

// Example 7c: Using `const` when Value Doesn't Change 🔒
// -----------------------------------------------------
// ✅ Using `const` for values that should not be reassigned:
const PI_7c = 3.14159; // ✅ 'const' for constants - prevents accidental reassignment
// PI_7c = 3.14; // Error - Assignment to constant variable.

// ------------------------------------------------------------------------

//  - 🚦 Use Strict Mode: Enable strict mode for improved code quality and error detection.  Catches common mistakes and enforces stricter rules. 🚦✅
//      - Add `"use strict";` at the beginning of JavaScript files or functions. 🚦

// Example 8a: Strict Mode - Error for Undeclared Variables 🚦
// --------------------------------------------------------
"use strict"; // Enable strict mode globally for this file 🚦

function strictModeExample_8a() {
    // "use strict"; // You can also enable strict mode only within a function 🚦

    // x_8a = 10; // ❌ Error in strict mode: x_8a is not defined (undeclared variable) - uncomment to see error.

    let y_8a = 20; // ✅ Declared with 'let' - no error
    console.log("y_8a in strict mode:", y_8a);
}

// Example 8b: Strict Mode - Prevents Accidental Globals 🚦
// ------------------------------------------------------
"use strict"; // Strict mode enabled

function strictModeGlobal_8b() {
    function innerFunction() {
        // z_8b = 30; // ❌ Error in strict mode: z_8b is not defined (accidental global) - uncomment to see error.
        let z_8b = 30; // ✅ Declared with 'let' - no error, function-scoped
        console.log("z_8b in strict mode:", z_8b);
    }
    innerFunction();
}

// ------------------------------------------------------------------------

// 2. Performance Optimization 🚀
// ----------------------------
// Improving code speed and efficiency.  Faster apps = happier users! 🚀😊

// Optimization Techniques: 🛠️
//  - 📉 Avoid Unnecessary DOM Manipulations: DOM operations are slow.  Batch updates instead of many small ones. 📉➡️📦
//      - ❌ Bad: Updating DOM in each loop iteration.
//      - ✅ Good: Build DOM changes in memory, then update DOM once.

// Example 9a: Minimizing DOM Manipulation - Bad Way ❌📉
// ----------------------------------------------------
function badDOMManipulation_9a() {
    const container_9a = document.getElementById('container_9a'); // Assume <div id="container_9a"> exists in HTML
    for (let i = 0; i < 1000; i++) {
        const div_9a = document.createElement('div'); // Create element in each loop
        div_9a.textContent = `Item ${i}`;
        container_9a.appendChild(div_9a); // ❌ DOM manipulation in each iteration - SLOW
    }
}

// Example 9b: Minimizing DOM Manipulation - Good Way ✅🚀
// -----------------------------------------------------
function goodDOMManipulation_9b() {
    const container_9b = document.getElementById('container_9b'); // Assume <div id="container_9b"> exists in HTML
    let content_9b = ""; // Build HTML string in memory
    for (let i = 0; i < 1000; i++) {
        content_9b += `<div>Item ${i}</div>`; // Build HTML string - FAST in memory
    }
    container_9b.innerHTML = content_9b; // ✅ Single DOM manipulation - FASTER
}

// ------------------------------------------------------------------------

//  - 🧮 Use Efficient Algorithms: Choose algorithms that scale well.  O(n log n) is generally better than O(n^2) for large datasets. 🧮➡️🚀
//      - Example: Sorting - Merge Sort (efficient) vs. Bubble Sort (inefficient for large arrays).

// Example 10a: Inefficient Algorithm - Bubble Sort 🐌
// -------------------------------------------------
// ❌ Bubble Sort - Simple but slow for large arrays (O(n^2))
function bubbleSort_10a(arr) { // 🐌 Bubble Sort - inefficient for large arrays
    const n = arr.length;
    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap arr[j] and arr[j+1]
                const temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}

// Example 10b: Efficient Algorithm - Native Sort (usually optimized - QuickSort or MergeSort) 🚀
// -----------------------------------------------------------------------------------------
// ✅ Native Sort - Generally efficient (implementation varies, often QuickSort or MergeSort - O(n log n) average)
function efficientSort_10b(arr) { // 🚀 Efficient Native Sort - usually O(n log n)
    return arr.sort((a, b) => a - b); // Use built-in sort - optimized
}

// ------------------------------------------------------------------------

//  - 📉 Minimize Loop Iterations: Optimize loop conditions, avoid unnecessary loops.  Less looping = faster code. 📉🔄
//      - ❌ Unnecessary Loop: Looping through entire array when you can stop earlier.
//      - ✅ Optimized Loop: Break loop when condition is met.

// Example 11a: Unnecessary Loop Iterations - Inefficient ❌🐌
// --------------------------------------------------------
function findFirstEvenNumberInefficient_11a(arr) { // ❌ Inefficient - loops through entire array even after finding first even number
    let firstEven = undefined;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] % 2 === 0) {
            firstEven = arr[i];
            // Continue looping unnecessarily even after finding first even number
        }
    }
    return firstEven;
}

// Example 11b: Optimized Loop Iterations - Efficient ✅🚀
// -----------------------------------------------------
function findFirstEvenNumberEfficient_11b(arr) { // ✅ Efficient - breaks loop once first even number is found
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] % 2 === 0) {
            return arr[i]; // ✅ Return immediately when first even number is found - efficient
        }
    }
    return undefined; // No even number found
}

// ------------------------------------------------------------------------

//  - 🗄️ Use Caching: Store frequently accessed data to avoid recalculations.  Cache = faster access. 🗄️➡️🚀
//      - Example: Caching API responses, calculation results.

// Example 12a: Without Caching - Redundant Calculations ❌🐌
// --------------------------------------------------------
function calculateFactorialNoCache_12a(n) { // ❌ No Caching - recalculates factorial every time
    if (n === 0) return 1;
    return n * calculateFactorialNoCache_12a(n - 1); // Recalculates for same 'n' repeatedly
}

// Example 12b: With Caching - Memoization ✅🚀🗄️
// ---------------------------------------------
const factorialCache_12b = {}; // Cache object 🗄️
function calculateFactorialWithCache_12b(n) { // ✅ With Caching (Memoization) - stores and reuses results
    if (n === 0) return 1;
    if (factorialCache_12b[n]) { // Check if result is cached 🗄️✅
        return factorialCache_12b[n]; // Return cached result - FAST 🚀
    }
    const result = n * calculateFactorialWithCache_12b(n - 1); // Calculate only if not cached
    factorialCache_12b[n] = result; // Store result in cache 🗄️
    return result;
}

// ------------------------------------------------------------------------

//  - ⚙️ Use `let` and `const` Properly: Avoid unnecessary variable declarations.  Declare only when needed, use `const` when value doesn't change. ⚙️🔒
//      - ✅ `const` for values that don't change - slight performance benefit in some engines.
//      - ✅ `let` when value needs to be reassigned.
//      - ❌ Avoid unnecessary variable declarations - keep code clean.

// Example 13a: Proper Use of `const` and `let` ✅🔒
// --------------------------------------------------
function calculateCircleArea_13a(radius) {
    const PI_13a = Math.PI; // ✅ 'const' for PI - constant value
    let area_13a; // ✅ 'let' for area - will be calculated and assigned
    if (radius > 0) {
        area_13a = PI_13a * radius * radius;
    } else {
        area_13a = 0;
    }
    return area_13a;
}

// Example 13b: Avoiding Unnecessary Variables ❌🧼➡️✅
// ----------------------------------------------------
// ❌ Unnecessary variable:
function getFullNameVerbose_13b(firstName, lastName) {
    let fullName_13b; // ❌ Unnecessary variable declaration
    fullName_13b = firstName + " " + lastName;
    return fullName_13b;
}

// ✅ More concise - directly return expression:
function getFullNameConcise_13b(firstName, lastName) {
    return firstName + " " + lastName; // ✅ Directly return - cleaner and slightly more efficient
}

// ------------------------------------------------------------------------

//  - ⏳ Debounce and Throttle: Limit rate of event handler execution (e.g., scroll, resize).  Prevent excessive function calls. ⏳🚫🌊
//      - Debounce: Delay function execution until after a pause. (Wait for user to stop typing before searching).
//      - Throttle: Limit function execution rate to once per interval. (Fire function at most every 100ms during scroll).

// Example 14a: Debounce - Search Input ⏳
// -------------------------------------
// Debounce function (simplified example):
function debounce_14a(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
            func.apply(this, args);
        }, delay);
    };
}

// Example usage of debounce for search input:
function handleSearchInput_14a(query) {
    console.log("Searching for:", query); // Simulate API call or heavy operation
}

const debouncedSearch_14a = debounce_14a(handleSearchInput_14a, 300); // Debounce search function

// Example 14b: Throttle - Scroll Event ⏳
// -------------------------------------
// Throttle function (simplified example):
function throttle_14b(func, interval) {
    let lastCallTime = 0;
    return function(...args) {
        const now = Date.now();
        if (now - lastCallTime >= interval) {
            lastCallTime = now;
            func.apply(this, args);
        }
    };
}

// Example usage of throttle for scroll event:
function handleScroll_14b() {
    console.log("Handling scroll event"); // Simulate expensive operation on scroll
}

const throttledScroll_14b = throttle_14b(handleScroll_14b, 100); // Throttle scroll handler

// ------------------------------------------------------------------------

//  - 🌍 Avoid Global Variables: Global variables can cause naming conflicts and are harder to manage.  Prefer local scope. 🌍🚫
//      - ❌ Global Variables: Can be accidentally overwritten, hard to track.
//      - ✅ Local Variables: Scope is limited, less prone to conflicts.

// Example 15a: Global Variable - Potential Conflicts ❌🌍
// -----------------------------------------------------
// ❌ Global variable - potential naming conflicts, harder to track
var counter_15a = 0; // Global variable 🌍

function incrementCounter_15a() {
    counter_15a++; // Modifies global variable
    console.log("Global counter:", counter_15a);
}

function resetCounter_15a() {
    counter_15a = 0; // Modifies global variable
    console.log("Global counter reset:", counter_15a);
}

// Example 15b: Local Variable - Better Scoping ✅🧘‍♂️
// --------------------------------------------------
function createCounter_15b() { // Factory function to create counters with local scope
    let counterValue_15b = 0; // Local variable - scoped within createCounter

    return {
        increment: function() {
            counterValue_15b++; // Modifies local variable
            console.log("Local counter:", counterValue_15b);
        },
        reset: function() {
            counterValue_15b = 0; // Modifies local variable
            console.log("Local counter reset:", counterValue_15b);
        }
    };
}

const counter1_15b = createCounter_15b(); // Create counter instance 1
const counter2_15b = createCounter_15b(); // Create counter instance 2 - independent scope

// ------------------------------------------------------------------------

//  - 🖼️ Optimize Images: Compress images, use appropriate formats (WebP, JPEG, PNG).  Smaller images = faster loading. 🖼️➡️🚀
//  - ⏳ Lazy Loading: Load resources (images, scripts) only when needed (e.g., when they come into viewport).  Load only what's necessary initially. ⏳➡️🚀
//  - 📦 Code Splitting: Divide code into smaller bundles, load only necessary code initially.  Faster initial load, load other parts on demand. 📦➡️🚀
//  - 🧵 Use Web Workers: For computationally intensive tasks, run in background threads to avoid blocking UI.  Keep UI responsive, offload heavy tasks to workers. 🧵➡️🚀

//  - 📊 Profiling Tools: Use browser developer tools (Performance tab) to identify bottlenecks.  Find and fix performance issues effectively. 📊🛠️

// ------------------------------------------------------------------------

// 3. Best Coding Techniques 🛠️
// ----------------------------
// Practices for robust and efficient development. 🛠️✅

//  - 🤝 Code Review: Review your code and have others review it.  Catch errors, improve code quality, share knowledge. 🤝🧐
//  - 🗂️ Version Control (Git): Use Git to track changes, collaborate, revert versions.  Essential for team work and managing code evolution. 🗂️🌳
//  - 🧪 Write Test Cases: Write unit tests to verify code correctness.  Ensure code works as expected, prevent regressions. 🧪✅
//  - 🧹 Use a Linter (ESLint): Automatically check code for errors and style issues.  Maintain code consistency, catch potential problems early. 🧹✅
//  - 📚 Learn and Update: Keep learning new technologies, best practices.  Stay current, improve skills, adopt better techniques. 📚🔄

// ------------------------------------------------------------------------

// 4. Example: Use of `use strict` for Clean Code 🚦🧼
// --------------------------------------------------
// (Example already covered in section 1.7 - Example 8a & 8b)
// `use strict` helps enforce cleaner code by:
//  - Preventing accidental global variables. 🌍🚫
//  - Throwing errors for certain "unsafe" actions. ⚠️
//  - Making debugging easier by catching mistakes early. 🐞➡️✅

// ------------------------------------------------------------------------

// ✅ Expected Outcome: 🎉
// ----------------------
// You should now be able to:
//  - ✅ Write cleaner, more readable, maintainable code. 🧼✅
//  - ✅ Optimize JavaScript code for better performance. 🚀✅
//  - ✅ Apply best coding techniques for robust development. 🛠️✅
//  - ✅ Write effective and efficient JavaScript code! 🚀💻✅

// 🎉 Congratulations! You've completed Chapter 19 and the entire JavaScript Journey! 🎉
// Keep practicing, keep learning, and build amazing things with JavaScript! 🚀💻🌟