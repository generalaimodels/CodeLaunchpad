// 🚀 Chapter 19: Best Practices and Performance in JavaScript 🌟

// 1. Writing Clean Code 🧼
// -----------------------
// Clean Code: Code that is easy to read, understand, and modify. 📖🧠🛠️
// It's like a well-organized toolbox 🧰 vs. a messy drawer 🗑️.

// Key Principles of Clean Code: 🔑
//  - 🏷️ Meaningful Names: Use descriptive names for variables, functions, classes.  Names should tell what things are. 🏷️➡️🗣️
//      - ❌ Bad: `let x = 5;`  // What is 'x'? Age? Count? Index? 🤔
//      - ✅ Good: `let itemCount = 5;` // Clear, it represents the count of items 🔢
//      - ❌ Bad Function: `function fn(d) { /* ... */ }` // 'fn' and 'd' are vague 🤷‍♀️
//      - ✅ Good Function: `function calculateDiscountPercentage(discountValue) { /* ... */ }` // Purpose is clear! 💰📉

// Example 1a: Variable Names - Clarity is Key 🏷️
// ---------------------------------------------
// ❌ Ambiguous names:
let a_1a = "John"; // ❌ 'a' - could be anything: address, author, ...
let arr_1a = [1, 2, 3]; // ❌ 'arr' - array of what? numbers? users?

// ✅ Descriptive names:
let userName_1a = "John"; // ✅ 'userName' - clearly a user's name 🧑‍💼
let itemIds_1a = [1, 2, 3]; // ✅ 'itemIds' - array of item identifiers 🆔[]

// Example 1b: Function Names - Action and Intent 🏷️
// ------------------------------------------------
// ❌ Vague function names:
function process_1b(data) { // ❌ 'process' - process what? how?
    // ... code that processes user data ...
}
function handle_1b(event) { // ❌ 'handle' - handle which event? what action?
    // ... code that handles click event ...
}

// ✅ Specific function names:
function processUserData_1b(userData) { // ✅ 'processUserData' - processes user-related data 🧑‍ ডাটা
    // ... code to validate and store user data ...
}
function handleClickEvent_1b(clickEvent) { // ✅ 'handleClickEvent' - handles a click event 🖱️👂
    // ... code to respond to a mouse click ...
}

// ------------------------------------------------------------------------

//  - 📏 Consistent Formatting: Use consistent indentation, spacing, line breaks.  Visual consistency improves readability. 📏➡️👁️
//      - ❌ Inconsistent Indentation:
/* ❌
function example_2a(){
if(true){
  console.log("Hello")
 }
}
*/
//      - ✅ Consistent Indentation:
/* ✅
function example_2a() {
    if (true) {
        console.log("Hello");
    }
}
*/

// Example 2a: Indentation and Block Structure 📏
// ---------------------------------------------
// ❌ Messy, inconsistent indentation:
/* ❌
function calculatePrice_2a(basePrice, taxRate){
if (basePrice > 100){
return basePrice * (1 + taxRate);
}else{
return basePrice;
}
}
*/

// ✅ Clean, consistent indentation:
function calculatePrice_2a(basePrice, taxRate) { // ✅ Function definition at column 0
    if (basePrice > 100) { // ✅ 'if' block indented
        return basePrice * (1 + taxRate); // ✅ Code inside 'if' indented further
    } else { // ✅ 'else' aligned with 'if'
        return basePrice; // ✅ Code inside 'else' indented
    }
}

// Example 2b: Spacing and Operator Consistency 📏
// ----------------------------------------------
// ❌ Cramped, inconsistent spacing:
/* ❌
let x_2b=10;let y_2b=20;function sum_2b(a,b){return a+b;}
*/

// ✅ Properly spaced code:
let x_2b = 10; // ✅ Space after variable name and '='
let y_2b = 20; // ✅ Consistent spacing throughout
function sum_2b(a, b) { // ✅ Space before and after function parameters
    return a + b; // ✅ Space around operators like '+'
}

// ------------------------------------------------------------------------

//  - ✂️ Avoid Long Functions: Break down large functions into smaller, single-responsibility functions.  Small functions are easier to understand and test. ✂️➡️🧩
//      - ❌ Monolithic Function (does too much):
/* ❌
function handleOrder_long() {
  // ... validate customer info (name, address, payment) ...
  // ... calculate total price (items, discounts, shipping) ...
  // ... update inventory (reduce stock levels) ...
  // ... send confirmation email to customer ...
  // ... log order details for analytics ...
  // (function is hundreds of lines long)
}
*/
//      - ✅ Decomposed Functions (single responsibility):
/* ✅
function validateCustomerInfo() { ... } // Validates customer details
function calculateOrderTotal() { ... } // Calculates total order price
function updateInventoryStock() { ... } // Updates product stock
function sendOrderConfirmationEmail() { ... } // Sends email confirmation
function logOrderAnalytics() { ... } // Logs order for analytics

function handleOrder_clean() { // Orchestrates the process - clean and readable
  if (validateCustomerInfo()) {
    const total = calculateOrderTotal();
    updateInventoryStock();
    sendOrderConfirmationEmail();
    logOrderAnalytics();
    // ... (handle order flow using smaller functions) ...
  }
}
*/

// Example 3a: Long Function Refactoring - Before ✂️
// -------------------------------------------------
// ❌ Single long function for user registration:
function registerUser_3a(name, email, password, address, profilePic) { // ❌ Long function - many steps in one
    // ... validate name, email, password format/strength ... (validation block)
    let isValid = true; // Assume valid at first
    if (!name || name.length < 3) isValid = false; // Example validation
    if (!email || !email.includes('@')) isValid = false;

    if (isValid) {
        // ... hash password for security ... (security block)
        const hashedPassword = /* ... hashing logic ... */;

        // ... upload profile picture to cloud storage ... (upload block)
        const profilePicUrl = /* ... cloud upload logic ... */;

        // ... save user data (name, email, hashedPassword, profilePicUrl, address) to database ... (database block)
        const userData = { name, email, hashedPassword, profilePicUrl, address };
        /* ... database save operation ... */
        console.log("User registered:", userData);

        // ... send welcome email to user ... (email block)
        /* ... email sending logic ... */
        console.log("Welcome email sent to:", email);

        return true; // Registration success
    } else {
        console.error("Registration failed due to validation errors.");
        return false; // Registration failure
    }
}

// Example 3b: Function Decomposition - After ✂️➡️🧩
// ------------------------------------------------
// ✅ Smaller functions, each with a clear responsibility:
function validateRegistrationData_3b(name, email, password) { // ✅ Validates input data
    if (!name || name.length < 3) return false;
    if (!email || !email.includes('@')) return false;
    return true;
}

function hashPassword_3b(password) { // ✅ Hashes password securely
    // ... secure password hashing algorithm ...
    return /* ... hashed password ... */;
}

function uploadProfilePicture_3b(profilePic) { // ✅ Uploads profile picture
    // ... cloud storage upload logic ...
    return /* ... profile picture URL ... */;
}

function saveUserToDatabase_3b(userData) { // ✅ Saves user data to DB
    /* ... database save operation ... */
    console.log("User data saved:", userData);
    return true;
}

function sendWelcomeEmailToUser_3b(email) { // ✅ Sends welcome email
    /* ... email sending logic ... */
    console.log("Welcome email sent to:", email);
    return true;
}

function registerUserRefactored_3b(name, email, password, address, profilePic) { // ✅ Orchestrates registration
    if (!validateRegistrationData_3b(name, email, password)) {
        console.error("Validation failed.");
        return false;
    }
    const hashedPassword = hashPassword_3b(password);
    const profilePicUrl = uploadProfilePicture_3b(profilePic);
    const userData = { name, email, hashedPassword, profilePicUrl, address };

    if (saveUserToDatabase_3b(userData)) {
        sendWelcomeEmailToUser_3b(email);
        return true; // Overall registration success
    } else {
        console.error("Database save failed.");
        return false; // Overall registration failure
    }
}

// ------------------------------------------------------------------------

//  - 💬 Comments: Write clear, concise comments for complex logic.  Explain 'why' (intent), not just 'what' (code). 💬➡️❓
//      - ❌ Over-commenting the obvious:
/* ❌
let count = 0; // Initialize count to 0
count = count + 1; // Increment count by 1
*/
//      - ✅ Explaining non-obvious logic:
/* ✅
// Calculate the final price after applying both promotional and loyalty discounts.
// Promotional discounts take precedence over loyalty discounts if both are applicable.
function calculateFinalPrice(basePrice, promoDiscount, loyaltyDiscount) { ... }
*/

// Example 4a: Explain Complex Algorithms with Comments 💬
// ----------------------------------------------------
function calculateFibonacci_4a(n) {
    // Fibonacci sequence: each number is the sum of the two preceding ones.
    // Sequence starts 0, 1, 1, 2, 3, 5, 8, ...
    if (n <= 0) return 0; // Base case: 0th Fibonacci number is 0
    if (n === 1) return 1; // Base case: 1st Fibonacci number is 1

    // Recursive calculation: Fn = F(n-1) + F(n-2)
    return calculateFibonacci_4a(n - 1) + calculateFibonacci_4a(n - 2);
}

// Example 4b: Avoid Redundant Comments for Self-Explanatory Code 💬🚫
// ---------------------------------------------------------------
function multiplyByTwo_4b(number) {
    // Multiply the input number by 2 and return the result. ❌ Unnecessary comment, code is clear.
    return number * 2;
}

// ------------------------------------------------------------------------

//  - ♻️ DRY (Don't Repeat Yourself): Avoid code duplication.  Reusable code is easier to maintain and update. ♻️➡️🔄
//      - ❌ Duplicated Validation Logic:
/* ❌
function validateEmail_v1(email) {
  // ... complex regex for email validation ...
}
function validateEmailForAdmin_v2(email) {
  // ... almost identical complex regex as v1, duplicated logic ...
}
*/
//      - ✅ Reusable Validation Function:
/* ✅
function validateEmail(email, options = {}) { // Reusable function with options
  const regex = /* ... complex email regex ... * /;
  // ... validation logic using regex and options ...
}
function validateEmail_v1_DRY(email) {
  return validateEmail(email); // Reuse the common validation
}
function validateEmailForAdmin_v2_DRY(email) {
  return validateEmail(email, { isAdminContext: true }); // Reuse with specific option
}
*/

// Example 5a: Duplicated Logic - Before DRY 👯‍♀️
// ---------------------------------------------
// ❌ Duplicated formatting code in multiple functions:
function formatOrderDate_5a(date) { // ❌ Function 1 - format date for orders
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0'); // Months are 0-indexed
    const year = date.getFullYear();
    return `${year}-${month}-${day}`; // YYYY-MM-DD format
}

function formatPaymentDate_5a(date) { // ❌ Function 2 - almost same formatting logic duplicated
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const year = date.getFullYear();
    return `${year}-${month}-${day}`; // Same YYYY-MM-DD format
}

// Example 5b: Reusable Function - After DRY ♻️🔄
// --------------------------------------------
// ✅ Reusable date formatting function:
function formatDate_5b(date) { // ✅ Reusable function - date formatting logic
    const day = date.getDate().toString().padStart(2, '0');
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const year = date.getFullYear();
    return `${year}-${month}-${day}`; // YYYY-MM-DD format
}

function formatOrderDate_5b_DRY(date) { // ✅ Function 1 - using reusable formatter
    return formatDate_5b(date); // Reuse formatDate for order dates
}

function formatPaymentDate_5b_DRY(date) { // ✅ Function 2 - using same reusable formatter
    return formatDate_5b(date); // Reuse formatDate for payment dates
}

// ------------------------------------------------------------------------

//  - 😙 KISS (Keep It Simple, Stupid): Prefer simple, straightforward solutions.  Simpler code is easier to understand, debug, and maintain. 😙➡️🧘‍♂️
//      - ❌ Over-engineered Solution (complex, hard to read):
/* ❌
function complexValueProcessor() {
  // ... uses multiple design patterns, abstract classes,
  //     and convoluted logic for a simple task ...
}
*/
//      - ✅ Simple, Direct Solution (easy to grasp):
/* ✅
function simpleValueProcessor() {
  // ... straightforward, linear logic to achieve the same task ...
}
*/

// Example 6a: Complex vs Simple Approach - Before KISS 🤯
// ---------------------------------------------------
// ❌ Overly complex way to find max value in array:
function findMaxValueComplex_6a(numbers) { // ❌ Complex - manually iterates and compares
    if (!numbers || numbers.length === 0) return undefined; // Handle empty array

    let maxValue = -Infinity; // Initialize to negative infinity
    for (let i = 0; i < numbers.length; i++) {
        let currentNumber = numbers[i];
        if (currentNumber > maxValue) {
            maxValue = currentNumber; // Update max if current is greater
        }
    }
    return maxValue;
}

// Example 6b: Simple and Direct Approach - After KISS 😙🧘‍♂️
// ------------------------------------------------------
// ✅ Simple, direct way using built-in Math.max():
function findMaxValueSimple_6b(numbers) { // ✅ Simple - uses Math.max for direct approach
    if (!numbers || numbers.length === 0) return undefined; // Handle empty array
    return Math.max(...numbers); // Use Math.max with spread operator - concise and clear
}

// ------------------------------------------------------------------------

//  - 🔒 Use `const` and `let` over `var`: For better scoping and predictability.  `const` for constants, `let` for variables that change. 🔒➡️✅
//      - ❌ `var` - function-scoped, hoisting issues, less predictable:
/* ❌
function example_var() {
  var message = "Hello";
  if (true) {
    var message = "World"; // Oops, overwrites outer 'message' due to var's scope
  }
  console.log(message); // Output: "World" (unexpected)
}
*/
//      - ✅ `let` and `const` - block-scoped, more predictable, less error-prone:
/* ✅
function example_let_const() {
  let message = "Hello"; // 'let' - block-scoped
  if (true) {
    let message = "World"; // This 'message' is block-scoped, separate from outer one
    console.log("Inner message:", message); // Output: "Inner message: World"
  }
  console.log("Outer message:", message); // Output: "Outer message: Hello" (as expected)
  const PI = 3.14159; // 'const' - constant, cannot be reassigned
  // PI = 3.14; // Error: Assignment to constant variable.
}
*/

// Example 7a: Scope with `var` - Function Scope 🔍
// -----------------------------------------------
function varScopeExample_7a() {
    var functionScopedVar = "Function scope"; // 'var' - function-scoped
    if (true) {
        var functionScopedVar = "Reassigned in block"; // Overwrites outer 'functionScopedVar'
    }
    console.log("varScopeExample_7a:", functionScopedVar); // Output: "varScopeExample_7a: Reassigned in block" (unexpected)
}

// Example 7b: Scope with `let` - Block Scope ✅🔒
// ----------------------------------------------
function letScopeExample_7b() {
    let blockScopedVar = "Block scope"; // 'let' - block-scoped
    if (true) {
        let blockScopedVar = "Inner block scope"; // Separate variable in this block
        console.log("Inside block letScopeExample_7b:", blockScopedVar); // Output: "Inside block letScopeExample_7b: Inner block scope"
    }
    console.log("Outside block letScopeExample_7b:", blockScopedVar); // Output: "Outside block letScopeExample_7b: Block scope" (expected)
}

// ------------------------------------------------------------------------

//  - 🚦 Use Strict Mode: Enable strict mode for improved code quality and error detection.  Catches common mistakes and enforces stricter rules. 🚦✅
//      - Add `"use strict";` at the top of JavaScript files or functions. 🚦

// Example 8a: Strict Mode - Preventing Accidental Globals 🚦
// -------------------------------------------------------
"use strict"; // Enable strict mode for this example 🚦

function strictModeGlobalVar_8a() {
    // accidentalGlobal = "Oops, global"; // ❌ Error in strict mode: Assignment to undeclared variable
    let intentionalLocal = "Local variable"; // ✅ Declared with 'let' - no error
    console.log("strictModeGlobalVar_8a - Local:", intentionalLocal);
}

// Example 8b: Strict Mode - Forbidding `with` Statement (security risk) 🚦
// --------------------------------------------------------------------
"use strict"; // Strict mode enabled

function strictModeWithStatement_8b() {
    // with ({ x: 10 }) { // ❌ Error in strict mode: Strict mode code may not include a with statement
    //     console.log(x); // 'with' creates scope ambiguity, disallowed in strict mode
    // }
    console.log("strictModeWithStatement_8b - 'with' is disallowed in strict mode.");
}

// ------------------------------------------------------------------------

// 2. Performance Optimization 🚀
// ----------------------------
// Performance Optimization: Improving code speed and efficiency.  Faster apps = happier users! 🚀😊

// Optimization Techniques: 🛠️
//  - 📉 Avoid Unnecessary DOM Manipulations: DOM operations are slow.  Batch updates instead of many small ones. 📉➡️📦
//      - ❌ Bad: Updating DOM in every loop iteration.
//      - ✅ Good: Construct DOM changes in memory, then update DOM once.

// Example 9a: DOM Manipulation - Inefficient Multiple Updates ❌🐌
// ------------------------------------------------------------
function inefficientDOMUpdates_9a() {
    const list_9a = document.getElementById('listContainer_9a'); // Assume <ul id="listContainer_9a"> exists
    for (let i = 0; i < 100; i++) {
        const listItem_9a = document.createElement('li'); // Create <li> element
        listItem_9a.textContent = `Item ${i}`;
        list_9a.appendChild(listItem_9a); // ❌ DOM manipulation on each loop iteration - SLOW
    }
}

// Example 9b: DOM Manipulation - Efficient Single Update ✅🚀
// ---------------------------------------------------------
function efficientDOMUpdates_9b() {
    const list_9b = document.getElementById('listContainer_9b'); // Assume <ul id="listContainer_9b"> exists
    let listHTML_9b = ''; // Build HTML string in memory
    for (let i = 0; i < 100; i++) {
        listHTML_9b += `<li>Item ${i}</li>`; // Append <li> to string - FAST in memory
    }
    list_9b.innerHTML = listHTML_9b; // ✅ Single DOM update using innerHTML - FASTER
}

// ------------------------------------------------------------------------

//  - 🧮 Use Efficient Algorithms: Choose algorithms that scale well.  O(n log n) is generally better than O(n^2) for large datasets. 🧮➡️🚀
//      - Example: Searching - Binary Search (efficient for sorted data) vs. Linear Search (inefficient for large sorted data).

// Example 10a: Inefficient Search - Linear Search 🐌
// ------------------------------------------------
// ❌ Linear Search - O(n) - checks each element one by one
function linearSearch_10a(arr, target) { // 🐌 Linear Search - checks every element
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === target) {
            return i; // Found at index i
        }
    }
    return -1; // Not found
}

// Example 10b: Efficient Search - Binary Search 🚀
// ---------------------------------------------
// ✅ Binary Search - O(log n) - much faster for sorted arrays
function binarySearch_10b(arr, target) { // 🚀 Binary Search - efficient for sorted arrays
    let left = 0, right = arr.length - 1;

    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) {
            return mid; // Found at index mid
        } else if (arr[mid] < target) {
            left = mid + 1; // Search in right half
        } else {
            right = mid - 1; // Search in left half
        }
    }
    return -1; // Not found
}

// ------------------------------------------------------------------------

//  - 📉 Minimize Loop Iterations: Optimize loop conditions, avoid unnecessary loops.  Less looping = faster code. 📉🔄
//      - ❌ Unnecessary Full Loop: Looping through entire array when you can stop early.
//      - ✅ Early Exit Loop: Break or return from loop when result is found.

// Example 11a: Loop - Unnecessary Full Iteration ❌🐌
// ---------------------------------------------------
function checkArrayContains_Inefficient_11a(arr, value) { // ❌ Inefficient - loops through entire array
    let found = false;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === value) {
            found = true;
            // Continues to loop even after finding the value
        }
    }
    return found;
}

// Example 11b: Loop - Efficient Early Exit ✅🚀
// ----------------------------------------
function checkArrayContains_Efficient_11b(arr, value) { // ✅ Efficient - returns as soon as value is found
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] === value) {
            return true; // ✅ Return 'true' immediately - early exit
        }
    }
    return false; // Not found after checking all elements
}

// ------------------------------------------------------------------------

//  - 🗄️ Use Caching (Memoization): Store results of expensive function calls to reuse them.  Cache = faster subsequent calls. 🗄️➡️🚀
//      - Example: Caching results of complex calculations or API responses.

// Example 12a: Function without Caching - Redundant Computation ❌🐌
// ---------------------------------------------------------------
function expensiveCalculation_NoCache_12a(input) { // ❌ No Cache - always re-calculates
    console.log("Performing expensive calculation for:", input); // Simulate expensive operation
    // ... complex computation based on input ...
    return input * 2; // Simplified expensive operation
}

// Example 12b: Function with Caching (Memoization) ✅🚀🗄️
// --------------------------------------------------------
const cache_12b = {}; // Cache object to store results 🗄️
function expensiveCalculation_WithCache_12b(input) { // ✅ With Cache - reuses stored results
    if (cache_12b[input] !== undefined) { // Check if result is in cache 🗄️✅
        console.log("Cache hit for:", input, ", returning cached result.");
        return cache_12b[input]; // Return cached result - FAST 🚀
    }

    console.log("Performing expensive calculation for:", input); // Only calculate if not in cache
    const result = input * 2; // Perform calculation
    cache_12b[input] = result; // Store result in cache 🗄️
    return result;
}

// ------------------------------------------------------------------------

//  - ⚙️ Proper `let` and `const` Usage: `const` for constants, `let` for variables that change.  Minor performance benefits, mostly about code clarity. ⚙️🔒
//      - ✅ `const` signals value immutability - slight optimization in some engines.
//      - ✅ `let` for variables that need reassignment within scope.

// Example 13a: Using `const` for Constants ✅🔒
// ---------------------------------------------
function calculateCircleCircumference_13a(radius) {
    const PI_13a = Math.PI; // ✅ 'const' - PI is a mathematical constant
    return 2 * PI_13a * radius;
}

// Example 13b: Using `let` for Variables that Change ✅🔒
// ------------------------------------------------------
function incrementValue_13b(startValue) {
    let currentValue_13b = startValue; // ✅ 'let' - value will be incremented
    currentValue_13b++; // Increment the variable
    return currentValue_13b;
}

// ------------------------------------------------------------------------

//  - ⏳ Debounce and Throttle: Limit rate of event handler execution.  Improve responsiveness, reduce unnecessary computations. ⏳🚫🌊
//      - Debounce: Delay execution until no more events in a delay period (e.g., for search input).
//      - Throttle: Execute at most once in a given interval (e.g., for scroll events).

// Example 14a: Debounce for Input Events ⏳
// ---------------------------------------
// Debounce function (simplified example):
function debounce_14a(func, delay) { // Debounce HOF - delays function execution
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId); // Clear previous timeout
        timeoutId = setTimeout(() => { // Set new timeout
            func.apply(this, args); // Execute function after delay
        }, delay);
    };
}

function handleInputChange_14a(inputValue) { // Function to be debounced
    console.log("Debounced Input Change:", inputValue); // Simulate action on input change
}

const debouncedHandler_14a = debounce_14a(handleInputChange_14a, 500); // Debounced version of handler

// Example 14b: Throttle for Scroll Events ⏳
// ---------------------------------------
// Throttle function (simplified example):
function throttle_14b(func, interval) { // Throttle HOF - limits function call rate
    let lastCallTime = 0;
    return function(...args) {
        const now = Date.now();
        if (now - lastCallTime >= interval) { // Check if interval has passed
            lastCallTime = now;
            func.apply(this, args); // Execute function if interval passed
        }
    };
}

function handleScrollEvent_14b() { // Function to be throttled
    console.log("Throttled Scroll Event"); // Simulate action on scroll
}

const throttledHandler_14b = throttle_14b(handleScrollEvent_14b, 200); // Throttled version of handler

// ------------------------------------------------------------------------

//  - 🌍 Minimize Global Variables: Global scope pollution can lead to conflicts and hard-to-debug issues.  Use local scope as much as possible. 🌍🚫
//      - ❌ Global Variables: Accessible everywhere, potential for accidental modifications.
//      - ✅ Local Variables: Scope limited to function or block, better encapsulation.

// Example 15a: Global Variable - Scope Issues ❌🌍
// ---------------------------------------------
var globalCounter_15a = 0; // ❌ Global variable - widely accessible

function incrementGlobalCounter_15a() {
    globalCounter_15a++; // Modifies global counter
    console.log("Global Counter Value:", globalCounter_15a);
}

function resetGlobalCounter_15a() {
    globalCounter_15a = 0; // Resets global counter
    console.log("Global Counter Reset to:", globalCounter_15a);
}

// Example 15b: Local Variable - Encapsulated Scope ✅🧘‍♂️
// ------------------------------------------------------
function createLocalCounter_15b() { // Factory function to create counters with local scope
    let localCounterValue_15b = 0; // ✅ Local variable - encapsulated

    return {
        increment: function() {
            localCounterValue_15b++; // Modifies local counter
            console.log("Local Counter Value:", localCounterValue_15b);
        },
        reset: function() {
            localCounterValue_15b = 0; // Resets local counter
            console.log("Local Counter Reset to:", localCounterValue_15b);
        }
    };
}

const counterInstance_15b = createLocalCounter_15b(); // Create counter instance with local scope

// ------------------------------------------------------------------------

//  - 🖼️ Optimize Images: Compress images (e.g., using tools like TinyPNG), use appropriate formats (WebP for better compression). 🖼️➡️🚀
//  - ⏳ Lazy Loading: Load images and other resources only when they are about to be visible in the viewport. ⏳➡️🚀
//  - 📦 Code Splitting: Break your JavaScript application into smaller chunks (bundles) that can be loaded on demand. 📦➡️🚀
//  - 🧵 Web Workers for Heavy Tasks: Offload CPU-intensive tasks to Web Workers to keep the main UI thread responsive. 🧵➡️🚀
//  - 📊 Profiling Tools: Use browser's developer tools (Performance tab, Lighthouse) to identify performance bottlenecks and optimize effectively. 📊🛠️

// ------------------------------------------------------------------------

// 3. Best Coding Techniques 🛠️
// ----------------------------
// Practices for robust, collaborative, and maintainable development. 🛠️✅

//  - 🤝 Code Review: Regularly review code (your own and others') for quality, errors, and best practices. 🤝🧐
//  - 🗂️ Version Control (Git): Use Git for tracking changes, collaboration, branching, and easy rollback. 🗂️🌳
//  - 🧪 Write Unit Tests: Create automated tests to verify functionality and prevent regressions. 🧪✅
//  - 🧹 Use a Linter (ESLint, JSHint): Employ linters for automated code quality checks and style enforcement. 🧹✅
//  - 📚 Continuous Learning: Stay updated with new JavaScript features, best practices, and performance techniques. 📚🔄

// ------------------------------------------------------------------------

// 4. Example: `use strict` for Clean Code Reinforcement 🚦🧼
// -------------------------------------------------------
// (Refer to section 1.7 - Example 8a & 8b for detailed `use strict` examples)
// `use strict` is a cornerstone of clean coding because it:
//  - Enforces stricter parsing and error handling, highlighting potential issues early. 🚦
//  - Prevents common JavaScript "mistakes" and bad practices, leading to more robust code. ✅
//  - Improves code readability and maintainability by promoting cleaner syntax and preventing unexpected behaviors. 🧼

// ------------------------------------------------------------------------

// ✅ Expected Outcome: 🎉
// ----------------------
// You should now be able to:
//  - ✅ Write cleaner, more readable, and maintainable JavaScript code. 🧼✅
//  - ✅ Apply various performance optimization techniques to improve efficiency. 🚀✅
//  - ✅ Understand and implement best coding practices for robust development. 🛠️✅
//  - ✅ Approach JavaScript development with a focus on quality and performance! 🚀💻✅

// 🎉 Congratulations! You've completed Chapter 19 and the entire JavaScript Mastery Journey! 🎉
// Keep practicing, keep applying these best practices, and build amazing, efficient JavaScript applications! 🚀💻🌟🏆