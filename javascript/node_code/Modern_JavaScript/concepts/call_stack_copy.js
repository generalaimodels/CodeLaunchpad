// ==========================
// Call Stack in JavaScript
// ==========================

// The call stack is a core part of the JavaScript runtime. It is a LIFO (Last-In, First-Out) stack
// that keeps track of function invocations. When a function is called, a stack frame is pushed onto
// the stack. When the function returns, its frame is popped off. Only synchronous code is managed
// by the call stack; asynchronous code is handled by the event loop and callback queue.

// --------------------------
// 1. Basic Call Stack Example
// --------------------------

function first() {
    console.log('Inside first()');
    second();
    console.log('Exiting first()');
}

function second() {
    console.log('Inside second()');
    third();
    console.log('Exiting second()');
}

function third() {
    console.log('Inside third()');
}

first();

/*
Call Stack Trace:
1. first() is called -> pushed to stack
2. first() calls second() -> second() pushed
3. second() calls third() -> third() pushed
4. third() returns -> popped
5. second() resumes, then returns -> popped
6. first() resumes, then returns -> popped
*/

// Output:
// Inside first()
// Inside second()
// Inside third()
// Exiting second()
// Exiting first()

// --------------------------
// 2. Stack Overflow Example
// --------------------------

// If the call stack grows too large (e.g., due to infinite recursion), a stack overflow occurs.

function stackOverflow() {
    return stackOverflow();
}

// Uncommenting the following line will cause a "RangeError: Maximum call stack size exceeded"
// stackOverflow();

// --------------------------
// 3. Call Stack and Error Stack Trace
// --------------------------

function a() {
    b();
}
function b() {
    c();
}
function c() {
    throw new Error('Stack trace demo');
}

try {
    a();
} catch (e) {
    console.log('Error stack trace:\n', e.stack);
}

// The stack trace shows the sequence of function calls leading to the error.

// --------------------------
// 4. Call Stack and Synchronous Execution
// --------------------------

function sync1() {
    console.log('sync1 start');
    sync2();
    console.log('sync1 end');
}
function sync2() {
    console.log('sync2');
}

sync1();

// Output:
// sync1 start
// sync2
// sync1 end

// --------------------------
// 5. Call Stack and Asynchronous Code
// --------------------------

// Asynchronous code (e.g., setTimeout, Promises) does NOT execute on the call stack immediately.
// Instead, the callback is scheduled and executed after the current call stack is empty.

console.log('A');

setTimeout(function timeoutCallback() {
    console.log('B (from setTimeout)');
}, 0);

console.log('C');

// Output:
// A
// C
// B (from setTimeout)

// The call stack executes 'A', then 'C'. The setTimeout callback is pushed to the callback queue
// and only executed after the stack is empty.

// --------------------------
// 6. Nested Function Calls and Stack Frames
// --------------------------

function outer(x) {
    function inner(y) {
        return x + y;
    }
    return inner(10);
}

console.log('outer(5):', outer(5)); // 15

// Each function invocation creates a new stack frame with its own scope.

// --------------------------
// 7. Exception Handling and Stack Unwinding
// --------------------------

function f1() {
    f2();
}
function f2() {
    f3();
}
function f3() {
    throw new Error('Exception in f3');
}

try {
    f1();
} catch (e) {
    console.log('Caught:', e.message);
}

// When an exception is thrown, the stack unwinds: frames are popped until a catch block is found.

// --------------------------
// 8. Call Stack and Recursion
// --------------------------

function factorial(n) {
    if (n === 0) return 1;
    return n * factorial(n - 1);
}

console.log('factorial(5):', factorial(5)); // 120

// Each recursive call adds a new frame to the stack until the base case is reached.

// --------------------------
// 9. Call Stack Limitations and Best Practices
// --------------------------

// - Deep recursion can cause stack overflow. Use iteration or tail recursion (if supported).
// - Avoid blocking the call stack with long-running synchronous code.
// - Asynchronous code helps keep the call stack clear for UI responsiveness.

// Example: Iterative vs Recursive Fibonacci

function fibRecursive(n) {
    if (n <= 1) return n;
    return fibRecursive(n - 1) + fibRecursive(n - 2);
}

function fibIterative(n) {
    let a = 0, b = 1;
    for (let i = 0; i < n; i++) {
        [a, b] = [b, a + b];
    }
    return a;
}

console.log('fibRecursive(5):', fibRecursive(5)); // 5
console.log('fibIterative(5):', fibIterative(5)); // 5

// Iterative approach is safer for large n due to stack limitations.

// --------------------------
// 10. Call Stack and Function Context
// --------------------------

function contextDemo() {
    console.log('this:', this);
}

contextDemo(); // In non-strict mode, 'this' is the global object; in strict mode, undefined.

// --------------------------
// 11. Call Stack and Anonymous Functions
// --------------------------

setTimeout(function () {
    // This function is pushed to the stack when executed by the event loop.
    console.log('Anonymous function executed');
}, 0);

// --------------------------
// 12. Call Stack and Arrow Functions
// --------------------------

const arrow = () => {
    console.log('Arrow function on stack');
};
arrow();

// --------------------------
// 13. Call Stack and Function Expressions
// --------------------------

const expr = function namedExpr() {
    console.log('Function expression on stack');
};
expr();

// --------------------------
// 14. Call Stack and Global Execution Context
// --------------------------

// The global code is the first frame on the call stack when a script runs.

console.log('Global execution context');

// ==========================
// Summary
// ==========================
// - The call stack is a LIFO structure managing synchronous function calls.
// - Each function call creates a stack frame; returns or exceptions pop frames.
// - Stack overflow occurs with excessive recursion.
// - Asynchronous code is not executed on the call stack immediately.
// - Understanding the call stack is crucial for debugging, performance, and writing robust code.
