// 🚀 Chapter 17: Advanced Topics in JavaScript 🌟

// 1. Higher-Order Functions (HOF) ⚙️
// -----------------------------------
// Functions that operate on other functions, either by:
//    a) Taking functions as arguments ➡️ 🎁
//    b) Returning a function as a result 🎁 ➡️

// They enhance code flexibility and reusability. ✨

// a. Built-in Higher-Order Functions (Array Methods) 🛠️📦
// -------------------------------------------------------

// Example 1: map() - Transforming array elements 🔄➡️🆕
// ----------------------------------------------------
const numbers_map_1 = [1, 2, 3, 4, 5]; // Original array 🔢
// 🌳 Visualizing map():
// [1, 2, 3, 4, 5]  --- map(function) --->  [transformed_1, transformed_2, ...]

// Using anonymous function Anonymous function inside map 👤➡️🔄
const squaredNumbers_map_1 = numbers_map_1.map(function(number) {
    return number * number; // Squaring each number 🔢 * 🔢
});
console.log("Squared Numbers (map - Example 1):", squaredNumbers_map_1); // Output: [1, 4, 9, 16, 25] 🆕🔢[]

// Using arrow function (concise syntax)  => 🏹
const numbersTimes2_map_1 = numbers_map_1.map((number) => number * 2); // Multiply each by 2 🔢 * 2
console.log("Numbers Times 2 (map - Example 1):", numbersTimes2_map_1);  // Output: [2, 4, 6, 8, 10] 🆕🔢[]

// Example 2: map() - More transformations 🔄➡️🆕
// -----------------------------------------------
const names_map_2 = ["Alice", "Bob", "Charlie"]; // Array of names 🧑‍🤝‍🧑
// 🌳 Visualizing map() for names:
// ["Alice", "Bob", "Charlie"] --- map(function) --->  ["Hello, Alice!", "Hello, Bob!", ...]

const greetings_map_2 = names_map_2.map(name => `Hello, ${name}! 👋`); // Creating greetings for each name
console.log("Greetings (map - Example 2):", greetings_map_2); // Output: ["Hello, Alice! 👋", "Hello, Bob! 👋", "Hello, Charlie! 👋"] 🆕💬[]

const nameLengths_map_2 = names_map_2.map(name => name.length); // Getting length of each name
console.log("Name Lengths (map - Example 2):", nameLengths_map_2); // Output: [5, 3, 7] 🆕🔢[]

// -----------------------------------------------------------------------

// Example 3: filter() - Selecting array elements based on condition 🔎➡️🆕
// -----------------------------------------------------------------------
const numbers_filter_3 = [1, 2, 3, 4, 5, 6]; // Number array 🔢
// 🌳 Visualizing filter():
// [1, 2, 3, 4, 5, 6] --- filter(condition_function) --->  [elements_that_pass_condition]

// Using anonymous function to filter even numbers 🔍 % 2 === 0
const evenNumbers_filter_3 = numbers_filter_3.filter(function(number) {
    return number % 2 === 0; // Condition: is number even? ❓
});
console.log("Even Numbers (filter - Example 3):", evenNumbers_filter_3); // Output: [2, 4, 6] 🆕🔢[] (only even numbers)

// Using arrow function to filter odd numbers 🔍 % 2 !== 0 => 🏹
const oddNumbers_filter_3 = numbers_filter_3.filter((number) => number % 2 !== 0); // Condition: is number odd? ❓
console.log("Odd Numbers (filter - Example 3):", oddNumbers_filter_3);   // Output: [1, 3, 5] 🆕🔢[] (only odd numbers)

// Example 4: filter() - Filtering strings based on length 🔎➡️🆕
// -----------------------------------------------------------
const words_filter_4 = ["apple", "banana", "kiwi", "orange", "grape"]; // Array of words 📝
// 🌳 Visualizing filter() for words:
// ["apple", "banana", ...] --- filter(condition_function) --->  [words_that_pass_condition]

const longWords_filter_4 = words_filter_4.filter(word => word.length > 5); // Condition: word length > 5 📏 > 5
console.log("Long Words (filter - Example 4):", longWords_filter_4); // Output: ["banana", "orange", "grape"] 🆕📝[] (words longer than 5 chars)

const startsWithA_filter_4 = words_filter_4.filter(word => word.startsWith('a') || word.startsWith('A')); // Condition: starts with 'a' or 'A'  startsWith('a')
console.log("Words starting with 'a' (filter - Example 4):", startsWithA_filter_4); // Output: ["apple"] 🆕📝[] (words starting with 'a')

// -----------------------------------------------------------------------

// Example 5: reduce() - Accumulating array elements to a single value ➕➡️📦
// -----------------------------------------------------------------------
const numbers_reduce_5 = [1, 2, 3, 4, 5]; // Number array 🔢
// 🌳 Visualizing reduce():
// [1, 2, 3, 4, 5] --- reduce(reducer_function, initial_value) --->  single_accumulated_value 📦

// Using anonymous function to sum all numbers ➕🔢
const sum_reduce_5 = numbers_reduce_5.reduce(function(accumulator, currentValue) {
    return accumulator + currentValue; // Accumulate sum ➕
}, 0); // Initial accumulator value is 0 📦=0
console.log("Sum (reduce - Example 5):", sum_reduce_5); // Output: 15 📦=15 (sum of all numbers)

// Using arrow function to calculate product of all numbers ✖️🔢 => 🏹
const product_reduce_5 = numbers_reduce_5.reduce((acc, curr) => acc * curr, 1); // Accumulate product ✖️, initial value 1 📦=1
console.log("Product (reduce - Example 5):", product_reduce_5); // Output: 120 📦=120 (product of all numbers)

// Example 6: reduce() -  More complex accumulation 📦➡️📦
// -------------------------------------------------------
const items_reduce_6 = [{ price: 10 }, { price: 20 }, { price: 30 }]; // Array of item objects 🛍️[]
// 🌳 Visualizing reduce() for item prices:
// [{price: 10}, ...] --- reduce(reducer_function, initial_value) --->  total_price 📦

const totalPrice_reduce_6 = items_reduce_6.reduce((acc, item) => acc + item.price, 0); // Summing up prices ➕💰
console.log("Total Price (reduce - Example 6):", totalPrice_reduce_6); // Output: 60 📦=60 (total price of items)

const maxPrice_reduce_6 = items_reduce_6.reduce((acc, item) => Math.max(acc, item.price), 0); // Finding max price 📈💰
console.log("Max Price (reduce - Example 6):", maxPrice_reduce_6); // Output: 30 📦=30 (maximum price)

// -----------------------------------------------------------------------

// b. Custom Higher-Order Functions 🛠️⚙️
// ------------------------------------
// Creating your own HOFs for reusable operations. 🔄⚙️

// Example 7: operation() - Generic operation HOF ⚙️
// -------------------------------------------------
// HOF 'operation' takes array and a function (fn) and applies fn to each element.
function operation_7(arr, fn) { // HOF definition: takes array and function
    const result = []; // Initialize result array 🆕[]
    for (let i = 0; i < arr.length; i++) { // Loop through input array 🔄
        result.push(fn(arr[i])); // Apply function 'fn' to each element and push to result ⚙️➡️🆕[]
    }
    return result; // Return the new array with operated values 🎁🆕[]
}

// Helper functions to be used with 'operation' 🛠️⚙️
function square_7(x) { return x * x; } // Function to square a number 🔢*🔢
function cube_7(x) { return x * x * x; }  // Function to cube a number 🔢*🔢*🔢

const numbers_7 = [1, 2, 3, 4, 5]; // Input number array 🔢[]

const squaredNumbers_7 = operation_7(numbers_7, square_7); // Using 'operation' with 'square' function ⚙️➡️🔢*🔢
console.log("Squared Numbers (Custom HOF - Example 7):", squaredNumbers_7); // Output: [1, 4, 9, 16, 25] 🆕🔢[]

const cubedNumbers_7 = operation_7(numbers_7, cube_7); // Using 'operation' with 'cube' function ⚙️➡️🔢*🔢*🔢
console.log("Cubed Numbers (Custom HOF - Example 7):", cubedNumbers_7);  // Output: [1, 8, 27, 64, 125] 🆕🔢[]

// Using anonymous function directly as callback 👤⚙️
const multipliedBy2_7 = operation_7(numbers_7, (x) => x * 2); // Using anonymous function inside HOF 👤➡️⚙️➡️🔢*2
console.log("Multiplied by 2 (Custom HOF - Example 7):", multipliedBy2_7); // Output: [2, 4, 6, 8, 10] 🆕🔢[]

// Example 8: createMultiplier() - HOF returning a function 🎁➡️⚙️
// --------------------------------------------------------------
// HOF 'createMultiplier' returns a function that multiplies by a given factor.
function createMultiplier_8(factor) { // HOF definition: takes factor
    return function(number) { // Returns a function that takes 'number' 🎁➡️⚙️
        return number * factor; // Returned function multiplies 'number' by 'factor' 🔢*factor
    };
}

const multiplyBy3_8 = createMultiplier_8(3); // Creating a multiplier function for factor 3 ⚙️ factor=3
const multiplyBy10_8 = createMultiplier_8(10); // Creating a multiplier function for factor 10 ⚙️ factor=10

console.log("Multiply by 3 (Custom HOF - Example 8):", multiplyBy3_8(5)); // Output: 15 (5 * 3) ⚙️(5)➡️15
console.log("Multiply by 10 (Custom HOF - Example 8):", multiplyBy10_8(5)); // Output: 50 (5 * 10) ⚙️(5)➡️50

// ------------------------------------------------------------------------

// 2. Currying 🍛
// -------------
// Transforming a function with multiple arguments into a sequence of functions,
// each taking a single argument. ➡️ 🍛➡️🍛➡️🍛

// Example 9: add(a, b, c) - Normal vs Curried 🔄🍛
// ------------------------------------------------
// Normal function taking three arguments:
function add_9(a, b, c) { // Normal add function (3 args) 🔢+🔢+🔢
    return a + b + c; // Returns sum of a, b, and c ➕
}
console.log("Normal add(1, 2, 3) - Example 9:", add_9(1, 2, 3)); // Output: 6 (1+2+3)

// Curried version of add(a, b, c): 🍛➡️🍛➡️🍛
function addCurried_9(a) { // First curried function (takes 'a') 🍛(a)
    return function(b) { // Returns a function that takes 'b' 🎁➡️🍛(b)
        return function(c) { // Returns a function that takes 'c' 🎁➡️🍛(c)
            return a + b + c; // Finally returns sum a+b+c ➕
        };
    };
}
console.log("Curried addCurried(1)(2)(3) - Example 9:", addCurried_9(1)(2)(3)); // Output: 6 (1+2+3) 🍛(1)🍛(2)🍛(3)

// Partial application with curried function 🧩
const add1_9 = addCurried_9(1); // Create function 'add1' by fixing 'a' as 1 🍛(1)➡️⚙️(b,c)
console.log("Partial add1(2)(3) - Example 9:", add1_9(2)(3));    // Output: 6 (1+2+3) ⚙️(2)🍛(3)

const add1And2_9 = add1_9(2); // Create function 'add1And2' by fixing 'b' as 2 in 'add1' ⚙️(2)➡️⚙️(c)
console.log("Partial add1And2(3) - Example 9:", add1And2_9(3));   // Output: 6 (1+2+3) ⚙️(3)

// Example 10: power(exponent, base) - Currying for reusability 🔄🍛
// -------------------------------------------------------------
// Normal power function:
function power_10(exponent, base) { // Normal power function (exponent, base) base^exponent
    return Math.pow(base, exponent); // Returns base raised to the power of exponent base^exponent
}
console.log("Normal power(2, 4) - Example 10:", power_10(2, 4)); // Output: 16 (4^2)

// Curried power function (curry by exponent first): 🍛➡️🎁
function powerCurried_10(exponent) { // Curried power function (takes exponent first) 🍛(exponent)
    return function(base) { // Returns a function that takes base 🎁➡️⚙️(base)
        return Math.pow(base, exponent); // Returns base^exponent ⚙️ base^exponent
    };
}

const square_10 = powerCurried_10(2); // Create 'square' function by fixing exponent as 2 🍛(2)➡️⚙️(base)
const cube_10 = powerCurried_10(3); // Create 'cube' function by fixing exponent as 3 🍛(3)➡️⚙️(base)

console.log("Curried square(4) - Example 10:", square_10(4)); // Output: 16 (4^2) ⚙️(4)
console.log("Curried cube(4) - Example 10:", cube_10(4));  // Output: 64 (4^3) ⚙️(4)

// 📝 Currying is useful for:
//   - Creating specialized functions from general ones. ⚙️➡️⚙️'
//   - Improving code readability and reusability. 📖♻️
//   - Enabling function composition. 🔗⚙️⚙️⚙️

// ------------------------------------------------------------------------

// 3. Pure Functions 😇
// ------------------
// Functions that:
//    a) Always return the same output for the same input. ➡️ === ➡️
//    b) Have no side effects (no external state modification). 🛡️

// Example 11: Pure vs Impure - add(a, b) 😇 vs addImpure(a) 😈
// -----------------------------------------------------------
// Pure function (no side effects, same input -> same output) 😇
function addPure_11(a, b) { // Pure add function 😇
    return a + b; // Returns sum of a and b ➕
}
console.log("Pure addPure(5, 10) - Example 11:", addPure_11(5, 10)); // Output: 15 😇➡️15
console.log("Pure addPure(5, 10) - Example 11 (again):", addPure_11(5, 10)); // Output: 15 (still 15) 😇➡️15 (same input, same output)

// Impure function (has side effects, output depends on external state) 😈
let x_11 = 10; // External variable 🌍
function addImpure_11(a) { // Impure add function 😈
    x_11 = x_11 + a; // Side effect: modifies external variable 'x' 💥🌍
    return x_11; // Returns updated value of x 🎁
}
console.log("Impure addImpure(5) - Example 11:", addImpure_11(5)); // Output: 15 😈➡️15 (x becomes 15)
console.log("Impure addImpure(5) - Example 11 (again):", addImpure_11(5)); // Output: 20 😈➡️20 (x becomes 20, different output for same input!)

// Example 12: Pure vs Impure - array operations 😇 vs 😈
// -----------------------------------------------------
// Pure function (creating a new array - no mutation of input) 😇
function pureArrayPush_12(arr, element) { // Pure array push function 😇
    return [...arr, element]; // Returns a new array with element added (spread operator) 🆕[]➕
}
const originalArray_12 = [1, 2, 3]; // Original array 🔢[]
const newArrayPure_12 = pureArrayPush_12(originalArray_12, 4); // Creating new array using pure function 🆕[]
console.log("Pure originalArray - Example 12:", originalArray_12); // Output: [1, 2, 3] (original array unchanged) 😇[]➡️[]
console.log("Pure newArrayPure - Example 12:", newArrayPure_12); // Output: [1, 2, 3, 4] (new array created) 🆕[]

// Impure function (modifying the original array - mutation) 😈
function impureArrayPush_12(arr, element) { // Impure array push function 😈
    arr.push(element); // Side effect: modifies the original array directly 💥[]
    return arr; // Returns the modified array (same array object!) 🎁[]
}
const originalArrayImpure_12 = [1, 2, 3]; // Original array 🔢[]
const newArrayImpure_12 = impureArrayPush_12(originalArrayImpure_12, 4); // Modifying original array using impure function 😈[]💥
console.log("Impure originalArrayImpure - Example 12:", originalArrayImpure_12); // Output: [1, 2, 3, 4] (original array is changed!) 😈[]💥➡️[]💥
console.log("Impure newArrayImpure - Example 12:", newArrayImpure_12); // Output: [1, 2, 3, 4] (same modified array) 😈[]💥

// 📝 Benefits of Pure Functions: 🎉
//   - Predictable: Same input, same output always. 🔮
//   - Testable: Easy to unit test in isolation. ✅
//   - Easier to reason about and debug. 🤔🐞➡️✅
//   - Referential transparency. 🔗🔄

// ------------------------------------------------------------------------

// 4. Advanced Regular Expressions (RegEx) 🔤🔍
// ------------------------------------------
// Powerful tool for pattern matching in strings. 🔤🔍
// Used in string methods: match(), test(), replace(), search(), etc. 🔤.methods()

// Basic RegEx Syntax Recap: 📝
//   /pattern/ - RegEx literal 📝
//   ^ - Start of string 🚀
//   $ - End of string 🏁
//   . - Any character (except newline) 🔣
//   * - Zero or more occurrences 0️⃣+
//   + - One or more occurrences 1️⃣+
//   ? - Zero or one occurrence 0️⃣|1️⃣
//   [] - Character set (e.g., [a-z] lowercase) 🔤{}
//   \d - Digit [0-9] 🔢
//   \w - Alphanumeric [a-zA-Z0-9_] 🔤🔢_
//   | - OR operator ❗|❗

// Example 13: RegEx test() - Basic patterns 🔤🔍✅
// ------------------------------------------------
let text_13 = "The quick brown fox jumps over the lazy dog"; // Sample text 🔤
let email_13 = "test@test.com"; // Sample email 📧
let mobile_13 = "+91-9999999999"; // Sample mobile 📱
let password_13 = "Password@123"; // Sample password 🔑
let url_13 = "https://www.example.com"; // Sample URL 🌐

// Check if text contains "fox" 🦊🔍
console.log("RegEx test(/fox/) - Example 13:", /fox/.test(text_13)); // Output: true ✅🦊

// Check if text starts with "The" 🚀🔍
console.log("RegEx test(/^The/) - Example 13:", /^The/.test(text_13)); // Output: true ✅🚀The

// Check if text ends with "dog" 🏁🔍
console.log("RegEx test(/dog$/) - Example 13:", /dog$/.test(text_13)); // Output: true ✅🏁dog

// Check for lowercase letters (at least one) 🔤🔍[a-z]+
console.log("RegEx test(/[a-z]+/) - Example 13:", /[a-z]+/.test(text_13)); // Output: true ✅🔤[a-z]+

// Example 14: RegEx - Validating email, mobile, password, URL 📧📱🔑🌐🔍✅
// ---------------------------------------------------------------------

// Validate email format 📧🔍
const emailRegex_14 = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/; // Complex email regex 📧🔍
console.log("RegEx email validation - Example 14:", emailRegex_14.test(email_13)); // Output: true ✅📧

// Validate mobile number (starts with +country-code-10digits) 📱🔍
const mobileRegex_14 = /^\+\d{2}-\d{10}$/; // Mobile number regex 📱🔍
console.log("RegEx mobile validation - Example 14:", mobileRegex_14.test(mobile_13)); // Output: true ✅📱

// Validate password (min 8 chars, uppercase, lowercase, number, special char) 🔑🔍
const passwordRegex_14 = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/; // Complex password regex 🔑🔍
console.log("RegEx password validation - Example 14:", passwordRegex_14.test(password_13)); // Output: true ✅🔑

// Validate URL format (http/https, domain, top level domain) 🌐🔍
const urlRegex_14 = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/; // URL regex 🌐🔍
console.log("RegEx URL validation - Example 14:", urlRegex_14.test(url_13)); // Output: true ✅🌐

// 📝 RegEx are extremely powerful for:
//   - Data validation (email, phone, etc.) ✅
//   - Text parsing and extraction. 🗂️➡️📦
//   - Search and replace operations. 🔍🔄

// ------------------------------------------------------------------------

// 5. Web Workers 🧵
// ----------------
// Running JavaScript code in background threads. 🧵
// Non-blocking UI for heavy computations. 🚀 UI 🚫🧱

// Key Concepts: 🔑
//   Worker Thread: Background thread for worker code. 🧵
//   Worker Object: Create new web worker. ⚙️
//   postMessage(): Send messages to/from worker. 📤📥
//   onmessage Event: Receive messages in worker/main thread. 👂

// Basic Steps: 👣
// 1. Create worker file (e.g., worker.js) 📄🧵
// 2. Create Worker object in main thread. ⚙️🧵
// 3. Send message to worker (postMessage()). 📤🧵
// 4. Listen for messages from worker (onmessage). 👂🧵

// Example 15: Basic Web Worker - worker.js (separate file) 📄🧵
// ---------------------------------------------------------
// (Create a file named 'worker.js' in the same directory)
/*  worker.js content:
    onmessage = function(event) { // Listener for messages from main thread 👂📥
        const data = event.data; // Data received from main thread 📦
        console.log('Worker: Message received', data); // Log received message 📥
        // Perform heavy computation (example: doubling the data) 🏋️‍♂️
        const result = data * 2; // Double the data 🔢*2
        postMessage(result); // Send result back to main thread 📤🎁
        console.log('Worker: Result sent back', result); // Log sent result 📤
    };
*/

// Example 16: Basic Web Worker - main script (in HTML file) 📄⚙️🧵📤📥👂
// ------------------------------------------------------------------
// In your main JavaScript file (e.g., app.js or in <script> tag)

// Create a new worker object, pointing to 'worker.js' file ⚙️🧵📄
const myWorker_16 = new Worker('worker.js'); // Create worker from 'worker.js' ⚙️🧵📄

// Send a message to the worker thread using postMessage 📤🧵
myWorker_16.postMessage(5); // Send number 5 to worker 📤🎁

// Listen for messages from the worker thread using onmessage 👂📥
myWorker_16.onmessage = function(event) { // Listener for messages from worker 👂📥
    const result_16 = event.data; // Data received from worker 📦
    console.log("Main thread: Result from worker:", result_16); // Output: Result from worker: 10 📥🎁
};

// Send another message to worker 📤🎁
myWorker_16.postMessage(15); // Send number 15 to worker 📤🎁
// (Worker will process and send back result, which will be caught by onmessage again) 👂📥🎁

// 📝 Important Web Worker Notes: ⚠️
//   - No direct DOM access from worker threads (security). 🛡️🚫DOM
//   - Useful for CPU-intensive tasks, network operations. 🏋️‍♂️🌐
//   - Web Workers need to be served via HTTP/HTTPS (server required, not just local file). 🌐 server 💻
//   - CORS restrictions apply when loading worker scripts. 🌐 CORS 🚫

// 🎉 Congratulations! You've completed Chapter 17 on Advanced Topics! 🎉
// Practice these advanced concepts to become a JavaScript master! 🚀💻💪