// Chapter 2: Variables, Data Types, and Operators

console.log("🚀 Chapter 2: Variables, Data Types, and Operators 🚀");

// --------------------------------------------------------------
// 1. Understanding Variables: Declaring and Initializing (var, let, const)
// --------------------------------------------------------------

console.log("\n--- 1. Variables ---");

// Using 'var' (Older way - avoid in modern JS!)
var oldVariable = "I'm an old variable"; // Function scope, not recommended.
console.log("var:", oldVariable); // Output: I'm an old variable

// Using 'let' (Preferred for variables that can change)
let age = 30; // Declaration and initialization
console.log("let:", age);   // Output: 30
age = 31;     // Reassigning a value
console.log("let (reassigned):", age);  // Output: 31

// Using 'const' (For constants, values that shouldn't change)
const PI = 3.14159; // Declaration and initialization. Value cannot be changed
console.log("const:", PI); // Output: 3.14159

// Trying to reassign a const throws an error
// PI = 3.14; // This line would cause an error

// Example of block scope
function exampleScope() {
  if (true) {
    let blockScoped = "I'm only available in this block";
     console.log("Inside block:", blockScoped)
  }
  // console.log(blockScoped); // This would cause an error.
}
exampleScope()

// Example of function scope
function functionScope(){
    var functionScoped = "I'm only available in this function";
    console.log("Inside function", functionScoped); // accessable
}
functionScope()
//console.log(functionScoped) // error

// Table explaining var, let and const
console.table([
  { keyword: "var", scope: "Function scope", reassignable: "Yes", example: "var x = 10; x = 20;" },
  { keyword: "let", scope: "Block scope", reassignable: "Yes", example: "let y = 15; y = 25;" },
  { keyword: "const", scope: "Block scope", reassignable: "No", example: "const PI = 3.14;" },
]);


// --------------------------------------------------------------
// 2. JavaScript Data Types
// --------------------------------------------------------------

console.log("\n--- 2. Data Types ---");

// a. Primitive Data Types
console.log("\n--- Primitive Data Types ---");
let myNumber = 42;
let myString = "Hello, World!";
let myBoolean = true;
let myNull = null;
let myUndefined; // Declared but not initialized
let mySymbol = Symbol("mySymbol");

console.log("Number:", myNumber, typeof myNumber);          // Output: Number: 42 'number'
console.log("String:", myString, typeof myString);         // Output: String: Hello, World! 'string'
console.log("Boolean:", myBoolean, typeof myBoolean);        // Output: Boolean: true 'boolean'
console.log("Null:", myNull, typeof myNull);              // Output: Null: null 'object' (Note: typeof null is an old error in JS)
console.log("Undefined:", myUndefined, typeof myUndefined);  // Output: Undefined: undefined 'undefined'
console.log("Symbol:", mySymbol, typeof mySymbol);          //Output: Symbol(mySymbol) 'symbol'

// Table explaining primitive data types
console.table([
  { type: "Number", description: "Represents numeric values", example: "10, 3.14, -5" },
  { type: "String", description: "Represents text", example: '"hello", \'JavaScript\'' },
  { type: "Boolean", description: "true or false", example: "true, false" },
  { type: "Null", description: "Intentional absence of a value", example: "null" },
  { type: "Undefined", description: "Variable declared but no value assigned", example: "let x;" },
    { type: "Symbol", description: "Unique identifier", example: "Symbol('mySymbol')" },
]);

// b. Non-Primitive Data Types
console.log("\n--- Non-Primitive Data Types ---");

let myObject = { name: "John", age: 30 };
let myArray = [1, 2, 3, 4];

console.log("Object:", myObject, typeof myObject);      // Output: Object: { name: 'John', age: 30 } 'object'
console.log("Array:", myArray, typeof myArray);         // Output: Array: [ 1, 2, 3, 4 ] 'object'

// Table explaining non-primitive data types
console.table([
  { type: "Object", description: "Collection of key-value pairs", example: '{name: "John", age: 30}' },
  { type: "Array", description: "Ordered list of values", example: '[1, 2, 3, 4], ["apple", "banana"]' },
]);


// --------------------------------------------------------------
// 3. Operators
// --------------------------------------------------------------

console.log("\n--- 3. Operators ---");

// Arithmetic Operators
let a = 10;
let b = 3;

console.log("Addition:", a + b);      // Output: 13
console.log("Subtraction:", a - b);   // Output: 7
console.log("Multiplication:", a * b);  // Output: 30
console.log("Division:", a / b);      // Output: 3.3333333333333335
console.log("Modulo:", a % b);       // Output: 1
console.log("Exponentiation:", a ** b);  // Output: 1000

// Assignment Operators
let x = 5;
x += 5; // x = x + 5
console.log("Compound Assignment (+=):", x);  // Output: 10

let y = 10;
y -= 3; // y= y-3
console.log("Compound Assignment (-=):", y); // Output: 7


// Comparison Operators
let p = 5;
let q = "5";
console.log("Equal to (==):", p == q);     // Output: true (checks only value)
console.log("Strict Equal to (===):", p === q);   // Output: false (checks value and type)
console.log("Not equal to (!=):", p != 10);     // Output: true
console.log("Strict not equal to (!==):", p !== q);   // Output: true
console.log("Greater than (>):", 10 > 5);    // Output: true
console.log("Less than (<):", 5 < 10);      // Output: true
console.log("Greater than or equal to (>=):", 10 >= 10);   // Output: true
console.log("Less than or equal to (<=):", 5 <= 10);     // Output: true

// Logical Operators
let condition1 = true;
let condition2 = false;
console.log("Logical AND (&&):", condition1 && true);   // Output: true
console.log("Logical OR (||):", condition1 || condition2); // Output: true
console.log("Logical NOT (!):", !condition1);    // Output: false

//Table explaining operators
console.table([
    {operator: "+", type: "Arithmetic", description: "Addition", example: "5 + 3"},
    {operator: "-", type: "Arithmetic", description: "Subtraction", example: "10 - 4"},
    {operator: "*", type: "Arithmetic", description: "Multiplication", example: "6 * 7"},
    {operator: "/", type: "Arithmetic", description: "Division", example: "10 / 2"},
    {operator: "%", type: "Arithmetic", description: "Modulo", example: "10 % 3"},
    {operator: "**", type: "Arithmetic", description: "Exponentiation", example: "2 ** 3"},
    {operator: "=", type: "Assignment", description: "Assigns value", example: "age = 25"},
    {operator: "+=, -=, *=, /=, %=", type: "Assignment", description: "Compound assignment", example: "x += 5"},
    {operator: "==", type: "Comparison", description: "Equal to", example: "5 == '5'"},
    {operator: "===", type: "Comparison", description: "Strict equal to", example: "5 === '5'"},
    {operator: "!=", type: "Comparison", description: "Not equal to", example: "5 != 10"},
    {operator: "!==", type: "Comparison", description: "Strict not equal to", example: "5 !== '5'"},
    {operator: ">", type: "Comparison", description: "Greater than", example: "10 > 5"},
    {operator: "<", type: "Comparison", description: "Less than", example: "5 < 10"},
    {operator: ">=", type: "Comparison", description: "Greater than or equal to", example: "10 >= 10"},
    {operator: "<=", type: "Comparison", description: "Less than or equal to", example: "5 <= 10"},
    {operator: "&&", type: "Logical", description: "Logical AND", example: "true && true"},
    {operator: "||", type: "Logical", description: "Logical OR", example: "true || false"},
    {operator: "!", type: "Logical", description: "Logical NOT", example: "!true"}
]);


// --------------------------------------------------------------
// 4. Type Conversion
// --------------------------------------------------------------

console.log("\n--- 4. Type Conversion ---");

// Implicit Type Conversion (Coercion)
let num = 5;
let str = "10";
console.log("Implicit Conversion (+):", num + str); // Output: "510" (Number 5 is converted to String)
console.log("Implicit Conversion (-):", num - str); // Output: -5 (String 10 is converted to Number)

// Explicit Type Conversion
let stringNum = "25";
let convertedNum = Number(stringNum); // Using Number()
console.log("Explicit Conversion (Number):", convertedNum + 5); // Output: 30

let numberToString = String(42); // Using String()
console.log("Explicit Conversion (String):", numberToString, typeof numberToString); //Output: 42 String

let stringBoolean = "hello";
let convertedBoolean = Boolean(stringBoolean) // Using Boolean()
console.log("Explicit Conversion (Boolean):",convertedBoolean, typeof convertedBoolean) // Output: true boolean

let decimalNumber = "10.5"
let convertedInteger = parseInt(decimalNumber) //Using parseInt()
console.log("Explicit Conversion (parseInt):", convertedInteger)  //Output: 10

let convertedFloatNumber = parseFloat(decimalNumber) //Using parseFloat()
console.log("Explicit Conversion (parseFloat):", convertedFloatNumber) //Output: 10.5


// Examples of different explicit type conversion
console.table([
  {type : "Number()", Description: "Converts value to number", examples: "Number(\"10\"), Number(true), Number(\"hello\")"},
   {type : "String()", Description: "Converts value to string", examples: "String(10), String(true)"},
    {type : "Boolean()", Description: "Converts value to boolean", examples: "Boolean(1), Boolean(0), Boolean(\"hello\"), Boolean(\"\")"},
      {type : "parseInt()", Description: "Parses string to integer", examples: "parseInt(\"10.5\")"},
        {type : "parseFloat()", Description: "Parses string to float", examples: "parseFloat(\"10.5\")"},
]);



// Example from the original request
let ageEx = 30;
let nameEx = "Raju";
console.log("Example Output:", ageEx + 5); // Output: 35 (Addition performs a numeric operation)


console.log("\n🎉 Chapter 2 Complete! You're doing great! 🎉");