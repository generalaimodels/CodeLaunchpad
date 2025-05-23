// Hello Boss! 👋 Let's dive deep into the world of JavaScript Strings! 🚀
// As the world's best coder, I'm here to make sure you become a STRING MASTER! 👑

// Chapter 6: Working with Strings - Let's EXPLORE! 🗺️

// -------------------- 1. String Methods --------------------
// Strings are like LEGO bricks for text! 🧱 JavaScript gives us AMAZING tools (methods)
// to play and build with these bricks. Think of methods as special ACTIONS you can perform on strings. ✨

// Let's break down each method with examples and super clear explanations! 👇

// ===== 1.1. `length` =====
// Concept:  📏  Counting Characters!
// What it does?  Tells you how many characters (letters, numbers, spaces, symbols) are in your string.
// Example:  Imagine you have a word "CODE". `length` will tell you it has 4 letters.

let textLength = "JavaScript";
let lengthCount = textLength.length;

console.log("String:", textLength); // Output: String: JavaScript
console.log("Length:", lengthCount);  // Output: Length: 10
// Explanation: "JavaScript" has 10 characters (including capital 'J' and lowercase letters).

// ===== 1.2. `substring(startIndex, endIndex)` =====
// Concept: ✂️  Cutting out a piece of string! (like slicing a cake 🍰)
// What it does? Extracts a part of the string from the `startIndex` up to (but NOT including) `endIndex`.
//          Think of indices as positions of characters, starting from 0.
// Example:  "HELLO".substring(1, 4)  ->  Start at position 1 ('E'), go up to position 4 ('O'), but don't include 'O'.  Result: "ELL"

let textSubstring = "WonderfulDay";
let slicedSubstring = textSubstring.substring(3, 8);

console.log("Original String:", textSubstring);    // Output: Original String: WonderfulDay
console.log("Substring (3, 8):", slicedSubstring); // Output: Substring (3, 8): derfu
// Explanation:
// "WonderfulDay"
//  0123456789...  (Indices)
//  ---^^^^^---   (Positions 3 to 7 are extracted: 'd', 'e', 'r', 'f', 'u')

// 📝 Note: If you only give one index `substring(startIndex)`, it takes from that index to the END of the string!
let substringToEnd = textSubstring.substring(5);
console.log("Substring (5 to end):", substringToEnd); // Output: Substring (5 to end): fulDay

// ===== 1.3. `slice(startIndex, endIndex)` =====
// Concept: 🔪  Similar to `substring`, also cuts a piece!
// What it does?  Almost the SAME as `substring` for positive indices. Extracts a section from `startIndex` to `endIndex` (exclusive).
// Key Difference: `slice()` can handle NEGATIVE indices!  🤯 Negative index counts from the END of the string.
// Example: "HELLO".slice(1, 4) -> "ELL" (Same as substring for positive indices)
//          "HELLO".slice(-3, -1) -> Counts from end: -1 is 'O', -2 is 'L', -3 is 'L'. Extracts from 'L' to 'L' (exclusive of -1 'O'). Result: "LL"

let textSlice = "AmazingJourney";
let slicedSection = textSlice.slice(2, 7);
console.log("Slice (2, 7):", slicedSection);   // Output: Slice (2, 7): azing

let negativeSlice = textSlice.slice(-5, -1);
console.log("Slice (-5, -1):", negativeSlice); // Output: Slice (-5, -2): rne
// Explanation of negative slice:
// "AmazingJourney"
// ...-------5-4-3-2-1  (Negative Indices from end)
//      ^^^^^         (Positions -5 to -3 are extracted: 'r', 'n', 'e')

// ===== 1.4. `indexOf(substring)` =====
// Concept: 🔍  Finding the FIRST location! (like searching for a word in a book 📖)
// What it does?  Finds the index of the FIRST time a `substring` appears in the string.
// Returns the index (position) if found, otherwise returns -1 (meaning not found).
// Example: "BANANA".indexOf("ANA") ->  "ANA" starts at index 1 in "BANANA". Result: 1

let textIndex = "HappyLearning";
let indexFirstL = textIndex.indexOf("L");

console.log("String:", textIndex);              // Output: String: HappyLearning
console.log("Index of 'L':", indexFirstL);      // Output: Index of 'L': 5
// Explanation: The FIRST 'L' in "HappyLearning" is at index 5 (starting from 0).

let indexNotFound = textIndex.indexOf("Z");
console.log("Index of 'Z' (not found):", indexNotFound); // Output: Index of 'Z' (not found): -1

// ===== 1.5. `lastIndexOf(substring)` =====
// Concept: 🔎 Finding the LAST location! (searching from the END of the book 📖)
// What it does? Finds the index of the LAST time a `substring` appears in the string.
// Returns the index (position) of the LAST occurrence, or -1 if not found.
// Example: "BANANA".lastIndexOf("ANA") -> The LAST "ANA" starts at index 3. Result: 3

let textLastIndex = "BananaBandana";
let indexLastNa = textLastIndex.lastIndexOf("na");

console.log("String:", textLastIndex);               // Output: String: BananaBandana
console.log("Last Index of 'na':", indexLastNa);    // Output: Last Index of 'na': 8
// Explanation: The LAST "na" in "BananaBandana" is at index 8.

// ===== 1.6. `toUpperCase()` =====
// Concept:  ⬆️  SHOUTING! (making everything CAPITAL letters)
// What it does? Converts the ENTIRE string to uppercase.
// Example: "hello".toUpperCase() -> "HELLO"

let textUpper = "lowercase";
let upperCaseText = textUpper.toUpperCase();

console.log("Original:", textUpper);       // Output: Original: lowercase
console.log("Uppercase:", upperCaseText);  // Output: Uppercase: LOWERCASE

// ===== 1.7. `toLowerCase()` =====
// Concept:  ⬇️  Whispering! (making everything lowercase letters)
// What it does? Converts the ENTIRE string to lowercase.
// Example: "HELLO".toLowerCase() -> "hello"

let textLower = "UPPERCASE";
let lowerCaseText = textLower.toLowerCase();

console.log("Original:", textLower);       // Output: Original: UPPERCASE
console.log("Lowercase:", lowerCaseText);  // Output: Lowercase: uppercase

// ===== 1.8. `trim()` =====
// Concept: ✨ Cleaning up extra spaces! (like tidying up your room 🧹)
// What it does? Removes whitespace (spaces, tabs, newlines) from the BEGINNING and END of a string.
// Whitespace in the MIDDLE of the string is NOT removed.
// Example: "  hello  ".trim() -> "hello"

let textWithSpaces = "   Spaces around!   ";
let trimmedText = textWithSpaces.trim();

console.log("Original with spaces:", textWithSpaces); // Output: Original with spaces:    Spaces around!
console.log("Trimmed:", trimmedText);              // Output: Trimmed: Spaces around!

// ===== 1.9. `split(separator)` =====
// Concept:  쪼개기! (Korean for splitting/dividing)  Breaking string into pieces! 🧩
// What it does? Splits a string into an ARRAY of substrings based on a `separator`.
// The `separator` is what you use to decide where to split the string (e.g., comma, space, etc.).
// Example: "apple,banana,mango".split(",") -> Splits at each comma. Result: ["apple", "banana", "mango"]

let textToSplit = "one,two,three,four";
let splitArray = textToSplit.split(",");

console.log("Original String:", textToSplit);    // Output: Original String: one,two,three,four
console.log("Split Array:", splitArray);       // Output: Split Array: [ 'one', 'two', 'three', 'four' ]
// Explanation: The string is split wherever a comma "," is found, creating an array of parts.

// ===== 1.10. `replace(substringToReplace, newSubstring)` =====
// Concept: 🔄  Replacing ONE thing with another! (like swapping ingredients in a recipe 🍲)
// What it does? Replaces the FIRST occurrence of `substringToReplace` with `newSubstring`.
// IMPORTANT: Only the FIRST match is replaced!
// Example: "hello world world".replace("world", "JavaScript") -> Only the FIRST "world" is replaced. Result: "hello JavaScript world"

let textReplaceFirst = "replace this word once";
let replacedTextFirst = textReplaceFirst.replace("word", "phrase");

console.log("Original:", textReplaceFirst);      // Output: Original: replace this word once
console.log("Replaced (first only):", replacedTextFirst); // Output: Replaced (first only): replace this phrase once

// ===== 1.11. `replaceAll(substringToReplace, newSubstring)` =====
// Concept: 🔁  Replacing ALL things! (like fixing ALL broken windows in a house 🏠)
// What it does? Replaces ALL occurrences of `substringToReplace` with `newSubstring`.
// This is different from `replace()` which only replaces the first one.
// Example: "hello world world".replaceAll("world", "JavaScript") -> BOTH "world"s are replaced. Result: "hello JavaScript JavaScript"

let textReplaceAll = "replace all words words";
let replacedTextAll = textReplaceAll.replaceAll("words", "phrases");

console.log("Original:", textReplaceAll);        // Output: Original: replace all words words
console.log("Replaced (all):", replacedTextAll);   // Output: Replaced (all): replace all phrases phrases

// ===== 1.12. `startsWith(substring)` =====
// Concept:  ✅  Checking the BEGINNING! (like verifying if a sentence starts correctly)
// What it does? Checks if the string STARTS with the specified `substring`.
// Returns `true` if it starts with the substring, `false` otherwise.
// Example: "hello".startsWith("he") -> "hello" DOES start with "he". Result: true

let textStartsWith = "StartHere";
let startsWithCheck = textStartsWith.startsWith("Start");

console.log("String:", textStartsWith);             // Output: String: StartHere
console.log("Starts with 'Start':", startsWithCheck); // Output: Starts with 'Start': true
console.log("Starts with 'start' (case-sensitive):", textStartsWith.startsWith("start")); // Output: Starts with 'start' (case-sensitive): false (Case matters!)

// ===== 1.13. `endsWith(substring)` =====
// Concept:  ✅  Checking the END! (like verifying if a sentence ends correctly)
// What it does? Checks if the string ENDS with the specified `substring`.
// Returns `true` if it ends with the substring, `false` otherwise.
// Example: "hello".endsWith("lo") -> "hello" DOES end with "lo". Result: true

let textEndsWith = "EndingNow";
let endsWithCheck = textEndsWith.endsWith("Now");

console.log("String:", textEndsWith);            // Output: String: EndingNow
console.log("Ends with 'Now':", endsWithCheck);  // Output: Ends with 'Now': true
console.log("Ends with 'now' (case-sensitive):", textEndsWith.endsWith("now")); // Output: Ends with 'now' (case-sensitive): false (Case matters!)

// ===== 1.14. `includes(substring)` =====
// Concept:  ✅  Checking if it's INSIDE! (like checking if a word is in a paragraph)
// What it does? Checks if the string CONTAINS the specified `substring` ANYWHERE within it.
// Returns `true` if the substring is found, `false` otherwise.
// Example: "hello".includes("el") -> "hello" DOES contain "el". Result: true

let textIncludes = "ContainingText";
let includesCheck = textIncludes.includes("Text");

console.log("String:", textIncludes);             // Output: String: ContainingText
console.log("Includes 'Text':", includesCheck);    // Output: Includes 'Text': true
console.log("Includes 'text' (case-sensitive):", textIncludes.includes("text")); // Output: Includes 'text' (case-sensitive): false (Case matters!)

// ===== 1.15. `charAt(index)` =====
// Concept:  📍  Getting a character at a specific position! (like finding a specific house number on a street 🏘️)
// What it does? Returns the character at the specified `index`. Remember, indices start from 0!
// Example: "hello".charAt(1) -> Character at index 1 is 'e'. Result: "e"

let textCharAt = "CharacterAt";
let charAtIndex = textCharAt.charAt(4);

console.log("String:", textCharAt);           // Output: String: CharacterAt
console.log("Character at index 4:", charAtIndex); // Output: Character at index 4: a

// ===== 1.16. `concat(string1, string2, ...)` =====
// Concept:  🔗  Joining strings together! (like linking train cars 🚂)
// What it does? Joins (concatenates) two or more strings together.
// Example: "hello".concat(" ", "world") -> Joins "hello", " ", and "world". Result: "hello world"

let textConcat1 = "Hello";
let textConcat2 = " ";
let textConcat3 = "World";
let combinedText = textConcat1.concat(textConcat2, textConcat3, "!");

console.log("String 1:", textConcat1);        // Output: String 1: Hello
console.log("String 2:", textConcat2);        // Output: String 2:
console.log("String 3:", textConcat3);        // Output: String 3: World
console.log("Concatenated:", combinedText);     // Output: Concatenated: Hello World!

// -------------------- 2. Template Literals (String Interpolation) --------------------
// Forget the old way of joining strings with '+'! ❌ Template literals are the MODERN and COOL way! 😎

// Concept:  `` Backticks and ${} magic! ✨
// How it works?  Enclose your string in BACKTICKS `` (not single or double quotes!).
// Use ${variableOrExpression} to put variables or even calculations directly inside your string!

let playerName = "Alice";
let playerScore = 150;

// Old way (String Concatenation - can be messy!)
let messageOld = "Player name: " + playerName + ", Score: " + playerScore;
console.log("Old way:", messageOld); // Output: Old way: Player name: Alice, Score: 150

// Template Literals - CLEAN and EASY! ✅
let messageNew = `Player name: ${playerName}, Score: ${playerScore}`;
console.log("Template Literal:", messageNew); // Output: Template Literal: Player name: Alice, Score: 150

// You can even do calculations inside ${}! ➕➖✖️➗
let itemPrice = 25;
let quantity = 3;
let totalPriceMessage = `Total cost: ${itemPrice * quantity} dollars`;
console.log("Calculation in template literal:", totalPriceMessage); // Output: Calculation in template literal: Total cost: 75 dollars

// Template literals are GREAT for:
//  ✅  Readability: Easier to see the structure of your string.
//  ✅  Embedding variables:  Cleanly insert variable values.
//  ✅  Multi-line strings:  Template literals can span multiple lines without extra effort!

let multiLineString = `
This is a
multi-line
string using
template literals!
`;
console.log("Multi-line string:", multiLineString);
// Output:
// Multi-line string:
// This is a
// multi-line
// string using
// template literals!


// -------------------- 3. Example (from your instructions) --------------------

let messyMessage = "  Hello, World!   ";
console.log("Original messy message:", messyMessage); // Output: Original messy message:   Hello, World!
let cleanMessage = messyMessage.trim();
console.log("Cleaned message (trim()):", cleanMessage);  // Output: Cleaned message (trim()): Hello, World!

// Explanation:
// 1. `messyMessage` has extra spaces at the beginning and end. 😫
// 2. `messyMessage.trim()` is called.  ✨ The `trim()` method MAGICALLY removes those extra spaces!
// 3. `cleanMessage` now holds the trimmed string: "Hello, World!". 😊

// -------------------- 🎉 Chapter 6 Complete! 🎉 --------------------

// You are now officially a JavaScript String Expert! 🏆
// Practice these methods, experiment with template literals, and you'll be handling text like a PRO! 💪

// Any questions, Boss?  Just ask!  We're ready to conquer ARRAYS next!  🚀  Let's GO! 💨