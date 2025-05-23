// Chapter 2 | Deep Dive into Functions & Scope

// 1. Execution Context & Hoisting

// JavaScript engine creates an execution context before executing code.
// It has two phases: Creation and Execution.

// Creation Phase:
// - Memory allocated for variables and functions.
// - Variables declared with 'var' are initialized with 'undefined'.
// - Variables declared with 'let' and 'const' are allocated memory but NOT initialized (Temporal Dead Zone).
// - Function declarations are fully hoisted (definition stored in memory).

// Execution Phase:
// - Code is executed line by line.
// - Values are assigned to variables.

// Hoisting Examples:
console.log("--- 1. Execution Context & Hoisting ---");

// var hoisting
console.log(myVar); // undefined (hoisted and initialized with undefined)
var myVar = 10;
console.log(myVar); // 10

// let/const hoisting (Temporal Dead Zone - TDZ)
// console.log(myLet); // ReferenceError: Cannot access 'myLet' before initialization
let myLet = 20;
console.log(myLet); // 20

// console.log(myConst); // ReferenceError: Cannot access 'myConst' before initialization
const myConst = 30;
console.log(myConst); // 30

// Function declaration hoisting
hoistedFunc(); // "Hoisted function called!" (fully hoisted)
function hoistedFunc() {
    console.log("Hoisted function called!");
}

// Function expression hoisting (variable hoisted, function assignment is not)
// notHoistedFunc(); // TypeError: notHoistedFunc is not a function
var notHoistedFunc = function() {
    console.log("Function expression called!");
};
notHoistedFunc(); // "Function expression called!"

// 2. Call Stack & Stack Frames

// The Call Stack manages execution contexts.
// When a function is called, a new stack frame (containing its execution context) is pushed onto the stack.
// When a function returns, its frame is popped off the stack.
// JavaScript is single-threaded, so there's only one call stack.

console.log("\n--- 2. Call Stack & Stack Frames ---");

function first() {
    console.log("Entering first()");
    second();
    console.log("Exiting first()");
}

function second() {
    console.log("Entering second()");
    third();
    console.log("Exiting second()");
}

function third() {
    console.log("Entering third()");
    // Base case for visualization
    console.log("Executing third()");
    // throw new Error("Simulating error"); // Uncomment to see stack trace
    console.log("Exiting third()");
}

console.log("Starting execution");
first();
console.log("Execution finished");

// Stack Overflow: Occurs with excessive recursion without a proper base case.
// function recursiveOverflow() {
//     recursiveOverflow();
// }
// try {
//     recursiveOverflow();
// } catch (e) {
//     console.error("Caught Stack Overflow:", e.message);
// }

// 3. Function Invocation Patterns

console.log("\n--- 3. Function Invocation Patterns ---");

const obj = {
    value: 100,
    method: function() {
        console.log("Method Invocation:", this.value); // 'this' refers to 'obj'
    },
    arrowMethod: () => {
        // Arrow functions inherit 'this' from the surrounding lexical scope
        // In this case, the global scope (or undefined in strict mode)
        console.log("Arrow Method 'this':", this);
    }
};

// a) Method Invocation: Called on an object. 'this' is the object.
obj.method(); // 100
obj.arrowMethod(); // Window/global object or undefined

// b) Constructor Invocation: Using the 'new' keyword. 'this' is the newly created object.
function ConstructorFunc(val) {
    this.value = val;
    console.log("Constructor Invocation: new object created with value:", this.value);
}
const instance = new ConstructorFunc(200); // Creates a new object, sets 'this' to it.

// c) Direct Call (Function Invocation): Simple function call.
function directCallFunc() {
    // 'use strict'; // Uncommenting this makes 'this' undefined
    console.log("Direct Call 'this':", this); // In non-strict mode, 'this' is global object (window/global)
}
directCallFunc();

// d) call / apply / bind: Explicitly setting 'this'.

const otherObj = { value: 300 };

// call: Invokes the function immediately with a specified 'this' value and arguments provided individually.
directCallFunc.call(otherObj); // Direct Call 'this': { value: 300 }

function greet(greeting, punctuation) {
    console.log(`${greeting}, ${this.name}${punctuation}`);
}
const person1 = { name: "Alice" };
const person2 = { name: "Bob" };

greet.call(person1, "Hello", "!"); // Hello, Alice!

// apply: Similar to call, but arguments are provided as an array.
greet.apply(person2, ["Hi", "?"]); // Hi, Bob?

// bind: Creates a *new* function with a permanently bound 'this' value (and optionally, bound arguments). Does not invoke immediately.
const greetAlice = greet.bind(person1);
greetAlice("Good morning", "."); // Good morning, Alice.

const greetBobFormal = greet.bind(person2, "Good day"); // Partially applying arguments
greetBobFormal("!!"); // Good day, Bob!!


// 4. Closures & Lexical Scope

// Lexical Scope: Scope is determined by the physical placement of code (where functions are declared), not where they are called. Inner functions have access to outer function scopes.
// Closure: A function bundled with references to its surrounding state (the lexical environment). It gives access to an outer function's scope from an inner function, even after the outer function has returned.

console.log("\n--- 4. Closures & Lexical Scope ---");

function outerFunction(outerVar) {
    const outerConst = "I am outer";
    return function innerFunction(innerVar) {
        // innerFunction closes over outerVar and outerConst
        console.log(`Outer Var: ${outerVar}, Outer Const: ${outerConst}, Inner Var: ${innerVar}`);
    };
}

const closureInstance1 = outerFunction("OuterVal1");
const closureInstance2 = outerFunction("OuterVal2");

closureInstance1("InnerVal1"); // Outer Var: OuterVal1, Outer Const: I am outer, Inner Var: InnerVal1
closureInstance2("InnerVal2"); // Outer Var: OuterVal2, Outer Const: I am outer, Inner Var: InnerVal2

// Classic example: Counter
function createCounter() {
    let count = 0; // Private variable due to closure
    return function() {
        count++;
        console.log("Count:", count);
        return count;
    };
}

const counter1 = createCounter();
const counter2 = createCounter(); // Each has its own scope and 'count' variable

counter1(); // Count: 1
counter1(); // Count: 2
counter2(); // Count: 1 (independent)

// Potential Issue: Closures can keep objects in memory longer than expected if not managed carefully, potentially leading to memory leaks if they hold references to large data structures that are no longer needed otherwise.

// 5. IIFE, Strict Mode, Tail Call Optimization

console.log("\n--- 5. IIFE, Strict Mode, Tail Call Optimization ---");

// a) IIFE (Immediately Invoked Function Expression)
// Creates a scope to avoid polluting the global namespace and execute code immediately.

(function() {
    var iifeVar = "I am inside an IIFE";
    console.log("IIFE executed:", iifeVar);
})();
// console.log(iifeVar); // ReferenceError: iifeVar is not defined (it's scoped to the IIFE)

const result = (function(x, y) {
    return x + y;
})(5, 3);
console.log("IIFE result:", result); // 8

// b) Strict Mode
// Enables stricter error checking and disables some problematic features.
// Can be applied globally (top of script) or per function.

function strictModeExample() {
    'use strict';
    // undeclaredVariable = 10; // ReferenceError: undeclaredVariable is not defined
    let objWithSetter = { set prop(value) { /* do nothing */ } };
    // objWithSetter.prop = 10; // Throws TypeError in strict mode if setter doesn't exist or isn't functional

    function checkThis() {
        console.log("Strict mode 'this' in function:", this); // undefined
    }
    checkThis();
}
strictModeExample();

// c) Tail Call Optimization (TCO)
// An optimization where a function's last action is calling another function (or itself recursively).
// If supported, the engine can reuse the current stack frame instead of creating a new one, preventing stack overflow for deep recursion.
// Note: TCO support is *not* consistently implemented across JavaScript engines, especially in non-strict mode or with complex tail calls. It's unreliable in practice for cross-browser/environment code.

// Example of a tail-recursive function (potentially optimizable)
function factorialTailRecursive(n, accumulator = 1) {
    'use strict'; // TCO often requires strict mode
    if (n === 0) {
        return accumulator;
    }
    // The recursive call is the VERY LAST operation.
    return factorialTailRecursive(n - 1, n * accumulator);
}

console.log("Tail Recursive Factorial (5):", factorialTailRecursive(5)); // 120
// console.log("Tail Recursive Factorial (20000):", factorialTailRecursive(20000)); // Might work if TCO is active, otherwise RangeError (Stack Overflow)

// Non-tail-recursive version (cannot be optimized)
function factorialNonTail(n) {
    if (n === 0) {
        return 1;
    }
    // Multiplication happens *after* the recursive call returns.
    return n * factorialNonTail(n - 1);
}
console.log("Non-Tail Recursive Factorial (5):", factorialNonTail(5)); // 120


// 6. Currying & Partial Application

console.log("\n--- 6. Currying & Partial Application ---");

// a) Currying
// Transforming a function f(a, b, c) into f(a)(b)(c). Each function takes one argument.

function multiplyThree(a) {
    return function(b) {
        return function(c) {
            return a * b * c;
        };
    };
}

const curriedMultiply = multiplyThree(2);
const curriedMultiplyBy3 = curriedMultiply(3);
const finalResult = curriedMultiplyBy3(4);
console.log("Curried multiplication (2)(3)(4):", finalResult); // 24
console.log("Direct curried call:", multiplyThree(2)(3)(4)); // 24

// Generic curry helper (simplified example)
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        } else {
            return function(...args2) {
                return curried.apply(this, args.concat(args2));
            }
        }
    };
}

function sumThree(a, b, c) {
    return a + b + c;
}
const curriedSum = curry(sumThree);
console.log("Generic Curry Sum (1)(2)(3):", curriedSum(1)(2)(3)); // 6
console.log("Generic Curry Sum (1, 2)(3):", curriedSum(1, 2)(3)); // 6


// b) Partial Application
// Creating a new function by pre-filling some of the arguments of an existing function.

function add(a, b, c) {
    return a + b + c;
}

// Using bind for partial application
const add5 = add.bind(null, 5); // 'this' is irrelevant here, pre-fill 'a' with 5
console.log("Partial Application with bind (add5(10, 20)):", add5(10, 20)); // 35

// Using closures for partial application
function partialAdd(a) {
    return function(b, c) {
        return add(a, b, c);
    };
}
const add10 = partialAdd(10);
console.log("Partial Application with closure (add10(20, 30)):", add10(20, 30)); // 60

// Difference: Currying always produces nested functions taking one argument each. Partial application produces a function that takes the *remaining* arguments.