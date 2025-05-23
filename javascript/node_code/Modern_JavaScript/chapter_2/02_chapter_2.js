// Chapter 2 | Deep Dive into Functions & Scope

// 1. Execution Context & Hoisting

// Example 1: Variable Hoisting
console.log(a); // undefined (declaration hoisted, initialization not)
var a = 10;

// Example 2: Function Hoisting
foo(); // "Function Declaration"
function foo() {
    console.log("Function Declaration");
}

// Example 3: Function Expression Hoisting
try {
    bar(); // TypeError: bar is not a function
} catch (e) {
    console.log(e.message);
}
var bar = function() {
    console.log("Function Expression");
};

// Example 4: Let/Const Hoisting (Temporal Dead Zone)
try {
    console.log(b); // ReferenceError
} catch (e) {
    console.log(e.message);
}
let b = 20;

// Example 5: Multiple Execution Contexts
function outer() {
    var x = 1;
    function inner() {
        var y = 2;
        console.log(x, y);
    }
    inner();
}
outer(); // 1 2

// 2. Call Stack & Stack Frames

// Example 1: Simple Call Stack
function first() {
    second();
}
function second() {
    third();
}
function third() {
    console.log("End of stack");
}
first(); // "End of stack"

// Example 2: Stack Overflow
function recurse() {
    recurse();
}
try {
    recurse();
} catch (e) {
    console.log("Stack Overflow");
}

// Example 3: Stack Frame with Arguments
function sum(a, b) {
    return a + b;
}
console.log(sum(2, 3)); // 5

// Example 4: Nested Calls
function a1() {
    b1();
}
function b1() {
    c1();
}
function c1() {
    console.log("Nested Call Stack");
}
a1(); // "Nested Call Stack"

// Example 5: Stack Unwinding
function f1() {
    try {
        throw new Error("Error ");
    } catch (e) {
        console.log("Caught:", e.message);
    }
}
f1(); // "Caught: Error"

// 3. Function Invocation Patterns

// Method Invocation
const obj = {
    x: 42,
    getX: function() {
        return this.x;
    }
};
console.log(obj.getX()); // 42

// Constructor Invocation
function Person(name) {
    this.name = name;
}
const p = new Person("Alice");
console.log(p.name); // "Alice"

// Direct Call
function greet() {
    return "Hello";
}
console.log(greet()); // "Hello"

// call / apply / bind

// Example 1: call
function showName() {
    console.log(this.name);
}
const user1 = { name: "Bob" };
showName.call(user1); // "Bob"

// Example 2: apply
function sumArgs(a, b) {
    return a + b;
}
console.log(sumArgs.apply(null, [5, 7])); // 12

// Example 3: bind
const boundShowName = showName.bind(user1);
boundShowName(); // "Bob"

// Example 4: call with arguments
function introduce(greeting) {
    console.log(greeting, this.name);
}
introduce.call(user1, "Hi"); // "Hi Bob"

// Example 5: apply with multiple arguments
function multiply(a, b, c) {
    return a * b * c;
}
console.log(multiply.apply(null, [2, 3, 4])); // 24

// 4. Closures & Lexical Scope

// Example 1: Basic Closure
function makeAdder(x) {
    return function(y) {
        return x + y;
    };
}
const add5 = makeAdder(5);
console.log(add5(3)); // 8

// Example 2: Private Variables
function Counter() {
    let count = 0;
    return {
        inc: function() { count++; },
        get: function() { return count; }
    };
}
const c = Counter();
c.inc();
console.log(c.get()); // 1

// Example 3: Loop with Closure (var)
var funcs = [];
for (var i = 0; i < 3; i++) {
    funcs.push(function() { return i; });
}
console.log(funcs[0]()); // 3

// Example 4: Loop with Closure (let)
let funcs2 = [];
for (let j = 0; j < 3; j++) {
    funcs2.push(function() { return j; });
}
console.log(funcs2[0]()); // 0

// Example 5: Lexical Scope
function outer2() {
    let x = 10;
    function inner2() {
        console.log(x);
    }
    inner2();
}
outer2(); // 10

// 5. IIFE, Strict Mode, Tail Call Optimization

// Example 1: IIFE
(function() {
    console.log("IIFE executed");
})();

// Example 2: IIFE with Parameters
(function(msg) {
    console.log(msg);
})("Hello from IIFE");

// Example 3: Strict Mode
(function() {
    'use strict';
    try {
        undeclaredVar = 10; // ReferenceError
    } catch (e) {
        console.log("Strict mode error");
    }
})();

// Example 4: Tail Call Optimization (ES6+)
function factorial(n, acc = 1) {
    if (n <= 1) return acc;
    return factorial(n - 1, n * acc); // Tail call
}
console.log(factorial(5)); // 120

// Example 5: IIFE for Module Pattern
const Module = (function() {
    let privateVar = 0;
    return {
        inc: function() { privateVar++; },
        get: function() { return privateVar; }
    };
})();
Module.inc();
console.log(Module.get()); // 1

// 6. Currying & Partial Application

// Example 1: Currying
function multiplyCurried(a) {
    return function(b) {
        return a * b;
    };
}
const double = multiplyCurried(2);
console.log(double(5)); // 10

// Example 2: Partial Application with bind
function greet2(greeting, name) {
    return `${greeting}, ${name}`;
}
const sayHelloTo = greet2.bind(null, "Hello");
console.log(sayHelloTo("Bob")); // "Hello, Bob"

// Example 3: Generic Curry Function
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        } else {
            return function(...next) {
                return curried.apply(this, args.concat(next));
            };
        }
    };
}
function add(a, b, c) { return a + b + c; }
const curriedAdd = curry(add);
console.log(curriedAdd(1)(2)(3)); // 6

// Example 4: Partial Application Manually
function partial(fn, ...fixedArgs) {
    return function(...restArgs) {
        return fn(...fixedArgs, ...restArgs);
    };
}
function sum3(a, b, c) { return a + b + c; }
const add1 = partial(sum3, 1);
console.log(add1(2, 3)); // 6

// Example 5: Currying with Arrow Functions
const curriedSum = a => b => c => a + b + c;
console.log(curriedSum(1)(2)(3)); // 6