// 🚀 Chapter 16: Modules and Imports in JavaScript 📜

// 📚 1. What are Modules? 📦
// --------------------------
// Modules are like separate compartments 🗄️ in your JavaScript code.
// Each module is a file 📄 containing JavaScript code.
// They help organize your code into logical, manageable pieces. 🧩

// Benefits of using Modules: 🎉
//  - 🏢 Organization: Keep your codebase structured and easy to navigate.  Imagine a well-organized library 📚 instead of a messy pile of books.
//  - ♻️ Reusability: Use code from one module in many parts of your application. Write once, use everywhere! 🔄
//  - 🛠️ Maintainability: Easier to fix bugs and update code when it's in modules.  Like fixing a single part of a machine ⚙️ instead of the whole thing.
//  - 🛡️ Namespace: Avoid naming conflicts. Modules create their own scope, preventing variables from clashing. Think of it as having separate rooms 🚪 in a house, each with its own set of items.

//  Visualizing Modules: 🌳
//  Project
//  ├── module1.js 📦
//  ├── module2.js 📦
//  └── main.js    📦 (Imports and uses module1 & module2)

// ------------------------------------------------------------------------

// 📤 2. Exporting Modules from One File 📂
// --------------------------------------
// To make parts of a module (functions, variables, classes) usable in other modules, you need to 'export' them. 📤
// Think of exporting as making items in your compartment 📦 available to others.

// There are two main types of exports:
//  a) Named Exports 🏷️
//  b) Default Exports ⭐️

// ------------------------------------------------------------------------

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

// Example 2: shapes.js 📐📏
// -----------------------
// This module defines classes for different shapes.

export class Circle { // ⚪ Class for Circle
    constructor(radius) {
        this.radius = radius;
    }
    getArea() {
        return PI * this.radius * this.radius; // Using PI from same module or another imported module (if exported)
    }
}

export class Square { // ⬜ Class for Square
    constructor(side) {
        this.side = side;
    }
    getArea() {
        return this.side * this.side;
    }
}

// Tree-like structure of named exports in shapes.js: 🌳📦
// shapes.js
// ├── export Circle 🏷️⚪
// └── export Square 🏷️⬜

// 📝 Key points about Named Exports:
//  - You can export as many named items as you want. 🔢
//  - When importing, you MUST use the exact same names. 🏷️➡️🏷️

// ------------------------------------------------------------------------

// b) Default Exports ⭐️
// --------------------
// Each module can have ONE default export. ⭐️
// Typically used for the main functionality of the module. 🥇

// Example 1: greeting.js 👋
// -----------------------
// This module provides a default greeting function.

function greet(name) { // 👋 Function to greet someone
    return `Hello, ${name}! 👋`;
}

export default greet; // ⭐️ Exporting 'greet' function as default

// Tree-like structure of default export in greeting.js: 🌳📦
// greeting.js
// └── export default greet ⭐️👋

// Example 2: calculator.js 🧮
// -------------------------
// This module provides a calculator object as the default export.

const calculator = { // 🧮 Calculator object with methods
    sum: (a, b) => a + b,
    multiply: (a, b) => a * b,
};

export default calculator; // ⭐️ Exporting 'calculator' object as default

// Tree-like structure of default export in calculator.js: 🌳📦
// calculator.js
// └── export default calculator ⭐️🧮

// 📝 Key points about Default Exports:
//  - Only ONE default export per module allowed. ☝️
//  - When importing, you can choose any name you want to represent the default export. 🏷️➡️ произвольное_имя (any_name)

// ------------------------------------------------------------------------

// 📥 3. Importing Modules into Another File 📂➡️📂
// ----------------------------------------
// To use exported items from a module, you need to 'import' them in another file. 📥
// Think of importing as bringing items from another compartment 📦 into your current one.

//  a) Importing Named Exports 🏷️📥
//  b) Importing Default Exports ⭐️📥
//  c) Importing Both Named and Default Exports 🏷️⭐️📥
//  d) Importing all as a namespace 🌐📥

// ------------------------------------------------------------------------

// a) Importing Named Exports 🏷️📥
// -----------------------------
// Use curly braces `{}` to import named exports. 🏷️➡️🏷️

// Example 1: Using math_operations.js in app.js ➕➖🥧➡️📂
// -------------------------------------------------------
// Assuming we have 'math_operations.js' with named exports (add, subtract, PI)

// app.js
import { add, subtract, PI } from './math_operations.js'; // 📥 Importing named exports

console.log("Addition:", add(5, 3)); // Output: Addition: 8 ➕
console.log("Subtraction:", subtract(10, 4)); // Output: Subtraction: 6 ➖
console.log("Pi Value:", PI); // Output: Pi Value: 3.14159 🥧

// Example 2: Using shapes.js in main.js 📐⬜⚪➡️📂
// -------------------------------------------------
// Assuming we have 'shapes.js' with named exports (Circle, Square)

// main.js
import { Circle, Square } from './shapes.js'; // 📥 Importing named exports

const myCircle = new Circle(5); // ⚪ Creating a Circle object
console.log("Circle Area:", myCircle.getArea()); // Output: Circle Area: 78.53975

const mySquare = new Square(7); // ⬜ Creating a Square object
console.log("Square Area:", mySquare.getArea()); // Output: Square Area: 49

// Renaming Imports with 'as' ➡️🏷️🆕🏷️
// You can rename imported items using the 'as' keyword.

// renamed_main.js
import { add as sum, PI as constantPi } from './math_operations.js'; // 📥 Importing and renaming

console.log("Sum:", sum(2, 7)); // Output: Sum: 9 ➕ (using renamed 'sum' for 'add')
console.log("Constant Pi:", constantPi); // Output: Constant Pi: 3.14159 🥧 (using renamed 'constantPi' for 'PI')

// 📝 Key points about Importing Named Exports:
//  - Import names MUST match the exported names (unless you rename with 'as'). 🏷️➡️🏷️
//  - Use curly braces `{}` for named imports. 📦➡️{}

// ------------------------------------------------------------------------

// b) Importing Default Exports ⭐️📥
// ------------------------------
// Import default exports without curly braces. ⭐️➡️🏷️ (any name)

// Example 1: Using greeting.js in user_app.js 👋➡️📂
// ----------------------------------------------------
// Assuming we have 'greeting.js' with a default export (greet)

// user_app.js
import sayHello from './greeting.js'; // 📥 Importing default export (renamed to 'sayHello')

console.log(sayHello("Alice")); // Output: Hello, Alice! 👋 (using imported 'sayHello' which is 'greet')
console.log(sayHello("Bob"));   // Output: Hello, Bob! 👋

// Example 2: Using calculator.js in calculation_app.js 🧮➡️📂
// ----------------------------------------------------------
// Assuming we have 'calculator.js' with a default export (calculator)

// calculation_app.js
import myCalculator from './calculator.js'; // 📥 Importing default export (renamed to 'myCalculator')

console.log("Calculator Sum:", myCalculator.sum(10, 5)); // Output: Calculator Sum: 15 ➕ (using imported 'myCalculator' which is 'calculator')
console.log("Calculator Multiply:", myCalculator.multiply(3, 4)); // Output: Calculator Multiply: 12 ✖️

// 📝 Key points about Importing Default Exports:
//  - No curly braces `{}` needed for default imports. 📦➡️🏷️ (any name)
//  - You can choose any name for the imported default export. 🏷️ произвольное_имя (any_name)

// ------------------------------------------------------------------------

// c) Importing Both Named and Default Exports 🏷️⭐️📥
// ---------------------------------------------
// You can import both named and default exports in a single import statement. 🤝

// Example 1: combined_module.js 📦 (hypothetical module with both types of exports)
// -----------------------------------------------------------------------
// Let's imagine 'combined_module.js' exports a default function 'mainFunction' and named constants 'VALUE_A', 'VALUE_B'.

// combined_module.js (hypothetical)
// export default function mainFunction() {
//     return "This is the default function.";
// }
// export const VALUE_A = 100;
// export const VALUE_B = 200;

// Using combined_module.js in app_combined.js 📦➡️📂
// --------------------------------------------------

// app_combined.js
// import defaultExport, { VALUE_A, VALUE_B } from './combined_module.js'; // 📥 Importing both default and named

// console.log("Default Export:", defaultExport()); // Output: Default Export: This is the default function. ⭐️
// console.log("Value A:", VALUE_A); // Output: Value A: 100 🏷️
// console.log("Value B:", VALUE_B); // Output: Value B: 200 🏷️

// 📝 Key points about Importing Both:
//  - Default export comes first, then named exports in curly braces. ⭐️, {}
//  - Order is important: `import default, { named1, named2 } from 'module';` 📦➡️⭐️, {}

// ------------------------------------------------------------------------

// d) Importing all as a namespace 🌐📥
// ----------------------------------
// Import all exports as a single namespace object. 🌐📦➡️📂

// Example 1: Importing all from math_operations.js 🌐➕➖🥧➡️📂
// ----------------------------------------------------------

// namespace_app.js
import * as MathUtils from './math_operations.js'; // 📥 Importing all as namespace 'MathUtils'

console.log("Namespace Add:", MathUtils.add(7, 2)); // Output: Namespace Add: 9 🌐➕
console.log("Namespace Subtract:", MathUtils.subtract(15, 5)); // Output: Namespace Subtract: 10 🌐➖
console.log("Namespace PI:", MathUtils.PI); // Output: Namespace PI: 3.14159 🌐🥧

const circleArea = new MathUtils.Circle(3).getArea(); // Using Circle class from namespace
console.log("Namespace Circle Area:", circleArea); // Output: Namespace Circle Area: 28.27431

// 📝 Key points about Namespace Imports:
//  - Use `import * as namespaceName from 'module';` 🌐📦➡️📂
//  - Access exports using dot notation: `namespaceName.exportName`. 🌐.🏷️
//  - Useful for modules with many exports. 📦➡️🌐

// ------------------------------------------------------------------------

// ⚙️ 4. Usage of Module Bundlers (like Webpack or Parcel - Optional) 📦➡️ 📦📦📦
// ------------------------------------------------------------------------
// Module bundlers are tools that bundle all your modules and their dependencies into one or more files. 📦➡️📦📦📦
// Useful for larger projects to optimize loading and manage dependencies.

// Benefits of Module Bundlers: 🎉
//  - 📦 Bundling: Combine many module files into fewer files (often a single bundle). Reduces HTTP requests and speeds up loading. 🚄
//  - 🔗 Dependency Management: Automatically handle dependencies between modules. Makes sure everything is in the right place. 🧩
//  - 🛠️ Code Transformation: Can transform your code (e.g., convert modern JavaScript to work in older browsers, using Babel). 🔄➡️👴👵🖥️
//  - 🚀 Optimization: Minify (reduce file size), compress, and optimize code for better performance. 💨

// Popular Module Bundlers: 🌟
//  - Webpack: Very powerful and configurable. 🏋️‍♂️
//  - Parcel: Zero-configuration, easy to use. 🚀💨
//  - Rollup: Optimized for library development. 📚

//  Bundling Process Visualization: 📦➡️📦📦📦
//  Module Files (module1.js, module2.js, ...) ➡️ Module Bundler (Webpack/Parcel/Rollup) ➡️ Bundled File (bundle.js)

//  For small projects, module bundlers might be optional. For larger, complex applications, they are highly recommended. 👍

// ------------------------------------------------------------------------

// ⚠️ Important Notes: 📝
// ----------------------
//  - `<script type="module">`: To use modules in HTML, you MUST add `type="module"` to your `<script>` tag. 🏷️<script type="module">
//     Example:
//     ```html
//     <script src="app.js" type="module"></script>
//     ```

//  - Server or Module Bundler: Modules often need to be served by a web server (like when you open HTML files through 'live-server' or similar) or processed by a module bundler. 🌐 server 💻 or 📦 bundler

// ------------------------------------------------------------------------

// ✅ Expected Outcome: 🎉
// ----------------------
// You should now be able to:
//  - ✅ Understand what modules are and their benefits. 📦🎉
//  - ✅ Export elements using named and default exports. 📤🏷️⭐️
//  - ✅ Import elements using named and default imports. 📥🏷️⭐️
//  - ✅ Understand the basics of module bundlers. 📦⚙️
//  - ✅ Organize code better with modules for cleaner, maintainable code. 🏢🛠️✨

// 🎉 Congratulations! You've completed Chapter 16 on Modules and Imports! 🎉
// Practice using modules in your projects to solidify your understanding. 🚀 Keep coding! 💻💪