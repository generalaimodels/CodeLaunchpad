// ==========================
// 0. Utility helpers (indentation, banner printer)
// ==========================
function indent(str, level = 2) {
    return str.split('\n').map(line => ' '.repeat(level) + line).join('\n');
}
function banner(title) {
    const line = '='.repeat(60);
    console.log('\n' + line);
    console.log(`= ${title}`);
    console.log(line);
}

// ==========================
// 1. Basic LIFO behavior
// ==========================
banner('1. Basic LIFO behavior');

function lifoExample1() {
    function a() { console.log('Example 1: a'); }
    function b() { console.log('Example 1: b'); }
    function c() { console.log('Example 1: c'); }
    a(); b(); c();
}
lifoExample1();

function lifoExample2() {
    function foo() { console.log('Example 2: foo'); }
    function bar() { foo(); console.log('Example 2: bar'); }
    bar();
}
lifoExample2();

function lifoExample3() {
    function first() { console.log('Example 3: first'); }
    function second() { first(); console.log('Example 3: second'); }
    function third() { second(); console.log('Example 3: third'); }
    third();
}
lifoExample3();

function lifoExample4() {
    let stack = [];
    function push(val) { stack.push(val); }
    function pop() { return stack.pop(); }
    push(1); push(2); push(3);
    console.log('Example 4:', pop(), pop(), pop());
}
lifoExample4();

function lifoExample5() {
    function logStack(msg) {
        console.log('Example 5:', msg, new Error().stack.split('\n')[2].trim());
    }
    function x() { logStack('x'); }
    function y() { x(); logStack('y'); }
    function z() { y(); logStack('z'); }
    z();
}
lifoExample5();

// ==========================
// 2. Nested invocation chain
// ==========================
banner('2. Nested invocation chain');

function nestedExample1() {
    function a() { console.log('Example 1: a'); }
    function b() { a(); console.log('Example 1: b'); }
    function c() { b(); console.log('Example 1: c'); }
    c();
}
nestedExample1();

function nestedExample2() {
    function outer() {
        function inner() {
            console.log('Example 2: inner');
        }
        inner();
        console.log('Example 2: outer');
    }
    outer();
}
nestedExample2();

function nestedExample3() {
    function f1() { console.log('Example 3: f1'); }
    function f2() { f1(); console.log('Example 3: f2'); }
    function f3() { f2(); console.log('Example 3: f3'); }
    function f4() { f3(); console.log('Example 3: f4'); }
    f4();
}
nestedExample3();

function nestedExample4() {
    function callChain(n) {
        if (n === 0) {
            console.log('Example 4: base');
            return;
        }
        callChain(n - 1);
        console.log('Example 4:', n);
    }
    callChain(3);
}
nestedExample4();

function nestedExample5() {
    function alpha() {
        function beta() {
            function gamma() {
                console.log('Example 5: gamma');
            }
            gamma();
            console.log('Example 5: beta');
        }
        beta();
        console.log('Example 5: alpha');
    }
    alpha();
}
nestedExample5();

// ==========================
// 3. Recursion & unwinding order
// ==========================
banner('3. Recursion & unwinding order');

function recursionExample1() {
    function recurse(n) {
        if (n === 0) {
            console.log('Example 1: base');
            return;
        }
        console.log('Example 1: down', n);
        recurse(n - 1);
        console.log('Example 1: up', n);
    }
    recurse(3);
}
recursionExample1();

function recursionExample2() {
    function factorial(n) {
        if (n === 0) return 1;
        return n * factorial(n - 1);
    }
    console.log('Example 2: factorial(4) =', factorial(4));
}
recursionExample2();

function recursionExample3() {
    function fib(n) {
        if (n <= 1) return n;
        return fib(n - 1) + fib(n - 2);
    }
    console.log('Example 3: fib(5) =', fib(5));
}
recursionExample3();

function recursionExample4() {
    function printReverse(arr, i = 0) {
        if (i === arr.length) return;
        printReverse(arr, i + 1);
        console.log('Example 4:', arr[i]);
    }
    printReverse([1, 2, 3]);
}
recursionExample4();

function recursionExample5() {
    function sum(n) {
        if (n === 0) return 0;
        return n + sum(n - 1);
    }
    console.log('Example 5: sum(5) =', sum(5));
}
recursionExample5();

// ==========================
// 4. Stack‐overflow demonstration (commented out for safety)
// ==========================
banner('4. Stack‐overflow demonstration (commented out for safety)');

// Example 1: Infinite recursion (commented out)
// function overflow1() { overflow1(); }
// overflow1();

// Example 2: Large recursion depth (commented out)
// function overflow2(n) { if (n > 0) overflow2(n - 1); }
// overflow2(1e6);

// Example 3: Stack overflow with mutual recursion (commented out)
// function a() { b(); }
// function b() { a(); }
// a();

// Example 4: Stack overflow with array fill (commented out)
// function fillStack(arr) { arr.push(1); fillStack(arr); }
// fillStack([]);

// Example 5: Stack overflow with setTimeout (safe, will not overflow)
let safeDepth = 0;
function safeRecursion() {
    if (safeDepth < 5) {
        safeDepth++;
        setTimeout(safeRecursion, 0);
    } else {
        console.log('Example 5: Safe recursion with setTimeout, depth:', safeDepth);
    }
}
safeRecursion();

// ==========================
// 5. Asynchronous callbacks vs. synchronous stack
// ==========================
banner('5. Asynchronous callbacks vs. synchronous stack');

function asyncExample1() {
    function sync() { console.log('Example 1: sync'); }
    function async() { setTimeout(() => console.log('Example 1: async'), 0); }
    sync();
    async();
}
asyncExample1();

function asyncExample2() {
    function foo() { console.log('Example 2: foo'); }
    function bar() { setImmediate(() => console.log('Example 2: bar (async)')); }
    foo();
    bar();
}
asyncExample2();

function asyncExample3() {
    function a() { console.log('Example 3: a'); }
    function b() { process.nextTick(() => console.log('Example 3: b (microtask)')); }
    a();
    b();
}
asyncExample3();

function asyncExample4() {
    function first() { console.log('Example 4: first'); }
    function second() { Promise.resolve().then(() => console.log('Example 4: second (promise)')); }
    first();
    second();
}
asyncExample4();

function asyncExample5() {
    function syncStack() { console.log('Example 5: syncStack'); }
    function asyncStack() { setTimeout(() => console.log('Example 5: asyncStack'), 10); }
    syncStack();
    asyncStack();
}
asyncExample5();

// ==========================
// 6. Error stack traces & automatic unwinding
// ==========================
banner('6. Error stack traces & automatic unwinding');

function errorExample1() {
    function thrower() { throw new Error('Example 1: error'); }
    try { thrower(); } catch (e) { console.log(e.stack.split('\n')[0]); }
}
errorExample1();

function errorExample2() {
    function a() { b(); }
    function b() { c(); }
    function c() { throw new Error('Example 2: stack trace'); }
    try { a(); } catch (e) { console.log(e.stack.split('\n')[1]); }
}
errorExample2();

function errorExample3() {
    function foo() { throw new Error('Example 3: foo error'); }
    try { foo(); } catch (e) { console.log('Example 3:', e.message); }
}
errorExample3();

function errorExample4() {
    function bar() { throw new Error('Example 4: bar error'); }
    try { bar(); } catch (e) { console.log('Example 4:', e.stack.split('\n')[2]); }
}
errorExample4();

function errorExample5() {
    function deep(n) { if (n === 0) throw new Error('Example 5: deep error'); else deep(n - 1); }
    try { deep(3); } catch (e) { console.log('Example 5:', e.stack.split('\n')[1]); }
}
errorExample5();

// ==========================
// 7. try / finally & guaranteed unwinding
// ==========================
banner('7. try / finally & guaranteed unwinding');

function finallyExample1() {
    try {
        console.log('Example 1: try');
    } finally {
        console.log('Example 1: finally');
    }
}
finallyExample1();

function finallyExample2() {
    try {
        throw new Error('Example 2: error');
    } finally {
        console.log('Example 2: finally after throw');
    }
}
try { finallyExample2(); } catch {}

function finallyExample3() {
    try {
        console.log('Example 3: try');
        return;
    } finally {
        console.log('Example 3: finally after return');
    }
}
finallyExample3();

function finallyExample4() {
    function f() {
        try {
            throw new Error('Example 4: error');
        } finally {
            console.log('Example 4: finally in function');
        }
    }
    try { f(); } catch {}
}
finallyExample4();

function finallyExample5() {
    let cleaned = false;
    function cleanup() { cleaned = true; }
    try {
        throw new Error('Example 5: error');
    } finally {
        cleanup();
    }
    console.log('Example 5: cleaned =', cleaned);
}
try { finallyExample5(); } catch {}

// ==========================
// 8. Tail Call Optimization (TCO) note + polyfill emulation
// ==========================
banner('8. Tail Call Optimization (TCO) note + polyfill emulation');

// Example 1: TCO not supported in most JS engines (demonstration)
function tcoExample1(n, acc = 1) {
    if (n === 0) return acc;
    return tcoExample1(n - 1, acc * n);
}
console.log('Example 1: factorial(5) =', tcoExample1(5));

// Example 2: TCO emulation with loop
function tcoExample2(n, acc = 1) {
    while (n > 0) {
        acc *= n;
        n--;
    }
    return acc;
}
console.log('Example 2: factorial(5) =', tcoExample2(5));

// Example 3: TCO for Fibonacci (inefficient, but for illustration)
function tcoFib(n, a = 0, b = 1) {
    if (n === 0) return a;
    return tcoFib(n - 1, b, a + b);
}
console.log('Example 3: fib(6) =', tcoFib(6));

// Example 4: TCO polyfill using trampoline
function trampoline(fn) {
    while (typeof fn === 'function') fn = fn();
    return fn;
}
function tcoTramp(n, acc = 1) {
    if (n === 0) return acc;
    return () => tcoTramp(n - 1, acc * n);
}
console.log('Example 4: factorial(5) =', trampoline(() => tcoTramp(5)));

// Example 5: TCO with accumulator for sum
function tcoSum(n, acc = 0) {
    if (n === 0) return acc;
    return tcoSum(n - 1, acc + n);
}
console.log('Example 5: sum(5) =', tcoSum(5));