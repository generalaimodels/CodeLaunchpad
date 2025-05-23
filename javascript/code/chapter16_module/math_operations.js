// a) Named Exports 🏷️
// --------------------
// You can export multiple items from a module using their names. 🏷️🏷️🏷️

// Example 1: math_operations.js ➕➖✖️➗
// --------------------------------------
// This module provides basic math functions and constants.

// Exporting functions and constants: 🚀
export function add(a, b) { // ➕ Function to add two numbers
    return a + b;
}

export function subtract(a, b) { // ➖ Function to subtract two numbers
    return a - b;
}

export const PI = 3.14159; // 🥧 Constant for Pi

// Tree-like structure of named exports in math_operations.js: 🌳📦
// math_operations.js
// ├── export add       🏷️➕
// ├── export subtract  🏷️➖
// └── export PI        🏷️🥧
