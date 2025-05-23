// Chapter 1 | Modern JavaScript Syntax & Patterns

// 1. ES6+ Language Enhancements

// 1.1 let / const vs. var

// var: function-scoped, hoisted, allows redeclaration
function varExample() {
    console.log(a); // undefined (hoisted)
    var a = 10;
    console.log(a) // 10
    if (true) {
        var a = 20; // same variable!
        console.log(a); // 20
    }
    console.log(a); // 20
}
varExample();

// let: block-scoped, not hoisted to usable state, no redeclaration in same scope
function letExample() {
    // console.log(b); // ReferenceError: Cannot access 'b' before initialization
    let b = 10;
    if (true) {
        let b = 20; // different variable (block scope)
        console.log(b); // 20
    }
    console.log(b); // 10
}
letExample();

// const: block-scoped, must be initialized, cannot be reassigned
function constExample() {
    const c = 30;
    // c = 40; // TypeError: Assignment to constant variable
    const obj = { x: 1 };
    obj.x = 2; // Allowed: object is mutable
    // obj = {}; // TypeError: Assignment to constant variable
    console.log(obj.x); // 2
}
constExample();

// 1.2 Arrow functions & lexical this

// Regular function: 'this' depends on how function is called
const obj1 = {
    value: 42,
    getValue: function() {
        return this.value;
    }
};
console.log(obj1.getValue()); // 42

// Arrow function: 'this' is lexically bound (to enclosing scope)
const obj2 = {
    value: 99,
    getValue: () => {
        // 'this' here is NOT obj2, but the enclosing scope (likely global)
        return this.value;
    }
};
console.log(obj2.getValue()); // undefined (in strict mode, globalThis.value)

// Arrow function in callbacks
function Timer() {
    this.seconds = 0;
    setInterval(() => {
        this.seconds++;
        // 'this' refers to Timer instance
    }, 1000);
}
// new Timer();

// 2. Destructuring & Rest/Spread

// 2.1 Object/array destructuring

// Array destructuring
const arr = [1, 2, 3, 4];
const [first, second, ...restArr] = arr;
console.log(first, second, restArr); // 1 2 [3,4]

// Skipping elements
const [, , third] = arr;
console.log(third); // 3

// Swapping variables
let x = 5, y = 10;
[x, y] = [y, x];
console.log(x, y); // 10 5

// Object destructuring
const user = { id: 1, name: "Alice", age: 25 };
const { name, age } = user;
console.log(name, age); // Alice 25

// Renaming and default values
const { id: userId, email = "noemail@example.com" } = user;
console.log(userId, email); // 1 'noemail@example.com'

// Nested destructuring
const nested = { a: { b: 2 } };
const { a: { b } } = nested;
console.log(b); // 2

// 2.2 Rest parameters & Spread operator

// Rest parameters (function arguments)
function sum(...nums) {
    return nums.reduce((acc, n) => acc + n, 0);
}
console.log(sum(1, 2, 3, 4)); // 10

// Spread operator (arrays)
const arr1 = [1, 2];
const arr2 = [3, 4];
const combined = [...arr1, ...arr2];
console.log(combined); // [1,2,3,4]

// Spread operator (objects)
const objA = { foo: 1, bar: 2 };
const objB = { bar: 3, baz: 4 };
const merged = { ...objA, ...objB }; // bar: 3 overwrites bar: 2
console.log(merged); // { foo:1, bar:3, baz:4 }

// 3. Template Literals & Tagged Templates

// Template literals
const person = "Bob";
const greeting = `Hello, ${person}!`;
console.log(greeting); // Hello, Bob!

// Multiline strings
const multiline = `Line 1
Line 2`;
console.log(multiline);

// Tagged templates
function tag(strings, ...values) {
    // strings: array of string literals
    // values: interpolated values
    return strings[0] + values.map((v, i) => `[${v}]` + strings[i + 1]).join('');
}
const tagged = tag`Sum: ${1 + 2}, Product: ${2 * 3}`;
console.log(tagged); // Sum: [3], Product: [6]

// 4. Default & Rest Parameters

function multiply(a, b = 2) {
    return a * b;
}
console.log(multiply(5)); // 10

function logAll(first, ...others) {
    console.log(first, others);
}
logAll(1, 2, 3, 4); // 1 [2,3,4]

// 5. Optional Chaining (?.) & Nullish Coalescing (??)

const deepObj = { a: { b: { c: 42 } } };
console.log(deepObj.a?.b?.c); // 42
console.log(deepObj.x?.y?.z); // undefined

// Optional chaining with function calls
const maybeFunc = null;
console.log(maybeFunc?.()); // undefined

// Nullish coalescing: returns right-hand side only if left is null or undefined
console.log(null ?? "default"); // "default"
console.log(0 ?? "default"); // 0 (0 is not null/undefined)
console.log(undefined ?? "fallback"); // "fallback"

// 6. Modules (ESM)

// In a real .js file, use the following syntax (not executable in this snippet):
// --- moduleA.js ---
// export const foo = 123;
// export function bar() { return "bar"; }
// export default function baz() { return "baz"; }

// --- moduleB.js ---
// import baz, { foo, bar } from './moduleA.js';
// import * as all from './moduleA.js';

// Dynamic import (returns a promise)
// import('./moduleA.js').then(module => {
//     console.log(module.foo);
// });

// Example (pseudo-code, not executable here):
/*
export const PI = 3.14;
export default function area(r) { return PI * r * r; }

import area, { PI } from './circle.js';
console.log(area(2), PI);
*/

// Note: In Node.js, use "type": "module" in package.json or .mjs extension for ESM.
// Dynamic import() is asynchronous and returns a promise.

// Exceptions & Edge Cases:
// - let/const are not hoisted to usable state (temporal dead zone).
// - Arrow functions cannot be used as constructors (no 'new').
// - Spread/rest with non-iterables throws TypeError.
// - Optional chaining short-circuits on null/undefined only.
// - Nullish coalescing (??) differs from ||: 0, '', false are not treated as nullish.
// - ESM imports/exports are static; dynamic import() is runtime and async.