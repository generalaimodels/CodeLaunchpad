Okay boss, let's move on to the intermediate level with Chapter 6: Working with Strings. Strings are fundamental for handling text in any programming language, and JavaScript provides lots of powerful tools for string manipulation. We'll explore these in detail, ensuring you become a string master!

**Chapter 6: Working with Strings**

**1. String Methods**

JavaScript provides a number of built-in methods (functions) that allow you to perform operations on strings. Here are some of the most useful ones:

| Method         | Description                                                                  | Example                       |
| :------------- | :--------------------------------------------------------------------------- | :---------------------------- |
| `length`       | Returns the number of characters in the string                               | `"hello".length` (Result: `5`) |
| `substring()`  | Extracts a part of a string (specify start and end indices)                 | `"hello".substring(1, 4)` (Result: `"ell"`) |
| `slice()`    | Extracts a section of a string and returns it as a new string             | `"hello".slice(1, 4)` (Result: `"ell"`) |
| `indexOf()`    | Returns the index of the first occurrence of a substring in a string        | `"hello".indexOf("l")` (Result: `2`) |
| `lastIndexOf()`| Returns the index of the last occurrence of a substring in a string           | `"hello".lastIndexOf("l")` (Result: `3`) |
| `toUpperCase()` | Converts the string to uppercase                                             | `"hello".toUpperCase()` (Result: `"HELLO"`) |
| `toLowerCase()` | Converts the string to lowercase                                             | `"HELLO".toLowerCase()` (Result: `"hello"`) |
| `trim()`       | Removes whitespace from both ends of a string                               | `"  hello  ".trim()` (Result: `"hello"`) |
| `split()`      | Splits a string into an array of substrings based on a separator            | `"apple,banana,mango".split(",")` (Result: `["apple", "banana", "mango"]`) |
| `replace()`    | Replaces a substring with another substring (only replaces the first match) | `"hello world".replace("world", "JavaScript")` (Result: `"hello JavaScript"`) |
| `replaceAll()`    | Replaces all occurrences of a substring with another substring             | `"hello world world".replaceAll("world", "JavaScript")` (Result: `"hello JavaScript JavaScript"`) |
| `startsWith()`| Checks if a string starts with a specified substring                          | `"hello".startsWith("he")` (Result: `true`) |
| `endsWith()`   | Checks if a string ends with a specified substring                             | `"hello".endsWith("lo")` (Result: `true`) |
| `includes()`   | Checks if a string contains a specified substring                          | `"hello".includes("el")` (Result: `true`) |
| `charAt()`    | Returns the character at a specified index                               | `"hello".charAt(1)` (Result: `"e"`) |
|`concat()`    | Joins two or more strings                             | `"hello".concat(" world")` (Result: `"hello world"`)  |

**Important Notes:**

*   String methods do not modify the original string. Instead, they return a *new* string with the changes.
*   Indices in JavaScript start from `0`.
*  The second parameter of `substring` and `slice` methods are optional. If only one index is given, then the string from that index to end of string will be extracted.
*  If the substring is not found by `indexOf()`, `lastIndexOf()` , then it returns -1.
* `replace()` method by default replace only the first occurrence of substring. To replace all occurrences, you can use `replaceAll()` method.
* String methods are case-sensitive.

*Examples*

```javascript
let message = "  Hello, World!   ";

console.log(message.length);         // Output: 17
console.log(message.trim());        // Output: "Hello, World!"
console.log(message.toUpperCase()); // Output: "  HELLO, WORLD!   "
console.log(message.substring(2, 7)); //Output: " Hell"
console.log(message.split(","));     // Output: ["  Hello", " World!   "]
console.log(message.indexOf("World")); // Output: 9
console.log(message.replace("World", "Universe")) //Output: "  Hello, Universe!   "

```

**2. Template Literals (String Interpolation)**

Template literals, introduced in ES6, provide a more convenient way to create strings, especially when you need to embed variables or expressions within a string.

*   Template literals are enclosed by backticks (`` ` ``).
*   Use `${variable}` or `${expression}` to insert variables and expressions inside the string.

```javascript
let name = "Raju";
let age = 30;

// Using string concatenation (old way)
let message1 = "My name is " + name + " and I am " + age + " years old.";
console.log(message1); //Output: My name is Raju and I am 30 years old.

// Using template literals
let message2 = `My name is ${name} and I am ${age} years old.`;
console.log(message2); //Output: My name is Raju and I am 30 years old.

let sum = 5 + 10
let message3 = `Sum of 5 and 10 is ${sum}`
console.log(message3) // Output: Sum of 5 and 10 is 15
```

*   Template literals make string formatting much cleaner and easier to read compared to the old string concatenation method.

**3. Example (from your instructions):**

```javascript
let message = "  Hello, World!   ";
console.log(message.trim()); // Output: "Hello, World!"
```

*   Here we have a string with extra spaces at beginning and the end.
*   The `trim()` method is used to remove whitespace from both ends of the string.

**Expected Outcome:**

You should now be comfortable with:

*   Using various string methods to manipulate text (like finding substrings, converting case, trimming, etc).
*   Formatting strings using template literals.
*   Selecting the right string method for the given task.

That's all for Chapter 6, boss! You are now a string handling expert. Practice with different string methods and template literals to become confident. Any doubts? Just let me know. We are ready for arrays next! Let's go! 🚀
