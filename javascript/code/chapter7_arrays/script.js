// Hello Boss! 👋 Ready to become an ARRAY NINJA? 🥷 Let's master JavaScript Arrays together! 🚀
// I'm here to make arrays crystal clear for EVERYONE! ✨

// Chapter 7: Arrays - Ordered Collections - Let's ORGANIZE our data! 🗂️
 
// -------------------- 1. Creating and Accessing Array Elements --------------------
// Imagine arrays as CONTAINERS 📦 or LISTS 📝 that hold items in a specific order.
// These items can be ANYTHING: numbers, words, true/false, even other arrays! 🤯

// ===== 1.1. Creating Arrays =====
// We use SQUARE BRACKETS `[]` to create arrays. Think of it like building a box! 🧰

// Example 1: Array of Fruits 🍎🍌🥭 (Strings - words!)
let fruits = ["apple", "banana", "mango"];
console.log("Fruits Array:", fruits); // Output: Fruits Array: [ 'apple', 'banana', 'mango' ]
// Explanation: We created an array named `fruits` and put three strings (fruit names) inside.

// Example 2: Array of Numbers 🔢 (Integers!)
let numbers = [1, 2, 3, 4, 5];
console.log("Numbers Array:", numbers); // Output: Numbers Array: [ 1, 2, 3, 4, 5 ]
// Explanation:  An array `numbers` holding numerical values.

// Example 3: Mixed Data Types 🌈 (Anything goes!)
let mixed = [1, "hello", true, null];
console.log("Mixed Array:", mixed); // Output: Mixed Array: [ 1, 'hello', true, null ]
// Explanation: JavaScript arrays are flexible! We can mix numbers, strings, booleans, null - all in one array!

// Example 4: Empty Array 🫙 (Starting from scratch!)
let emptyArray = [];
console.log("Empty Array:", emptyArray); // Output: Empty Array: []
// Explanation:  An array with nothing inside, like an empty box ready to be filled.

// ===== 1.2. Accessing Array Elements using Indices =====
// Think of INDICES as ADDRESSES 📍 or POSITIONS of items in the array.
// Indices start from 0, like counting floors in a building from the ground floor! 🏢

// Let's use our `colors` array:
let colors = ["red", "green", "blue"];
// Indices:          0      1       2

console.log("First color (index 0):", colors[0]); // Output: First color (index 0): red
// Explanation: `colors[0]` accesses the element at index 0, which is "red". It's like saying "Give me the item at the first position".

console.log("Third color (index 2):", colors[2]); // Output: Third color (index 2): blue
// Explanation: `colors[2]` gets the element at index 2, which is "blue" (remember, we start counting from 0!).

// ===== 1.3. Modifying Array Elements =====
// Arrays are MUTABLE, meaning you can CHANGE their contents after they are created! ✏️

console.log("Original colors array:", colors); // Output: Original colors array: [ 'red', 'green', 'blue' ]

colors[1] = "yellow"; // Change the element at index 1 (originally "green") to "yellow"
console.log("Colors array after modification:", colors); // Output: Colors array after modification: [ 'red', 'yellow', 'blue' ]
// Explanation: We used `colors[1] = "yellow"` to REPLACE "green" with "yellow" at index 1.

// ===== 1.4. Important Points about Arrays =====

// 📝 Point 1: Mixed Data Types Allowed 🌈
// As we saw in the `mixed` array, JavaScript arrays can hold different types of data together.
// This is very flexible but be mindful of data types when you process array elements!

// 📝 Point 2: Accessing Non-existent Index ❓ -> `undefined`
console.log("Accessing index 5 in colors array (which has only 3 elements):", colors[5]); // Output: Accessing index 5 in colors array (which has only 3 elements): undefined
// Explanation: If you try to access an index that is OUTSIDE the array's bounds (like index 5 in an array of length 3), JavaScript returns `undefined`. It means "nothing is there at that position".

// -------------------- 2. Array Methods --------------------
// Array methods are like POWER TOOLS 🛠️ for working with arrays! They let you do cool things like:
// - Add items, remove items, find items, change order, and much more! ✨

// Let's explore each method with examples and clear explanations! 👇

// ===== 2.1. `push(element1, element2, ...)` =====
// Concept:  ➡️  Adding to the END! (like joining a queue at the back 🚶‍♀️🚶‍♂️)
// What it does? Adds one or more `element`s to the END of the array.
// Returns: The NEW LENGTH of the array after adding elements.
// Modifies: The ORIGINAL ARRAY is changed (mutated).

let fruitsPush = ["apple", "banana"];
console.log("Original fruits array:", fruitsPush); // Output: Original fruits array: [ 'apple', 'banana' ]

let newLengthPush = fruitsPush.push("orange", "grape"); // Add "orange" and "grape" to the end
console.log("Fruits array after push:", fruitsPush);     // Output: Fruits array after push: [ 'apple', 'banana', 'orange', 'grape' ]
console.log("New length after push:", newLengthPush);    // Output: New length after push: 4
// Explanation: `push()` added "orange" and "grape" to the end of `fruitsPush`. The array is now longer, and `push()` told us the new length (4).

// ===== 2.2. `pop()` =====
// Concept:  ⬅️  Removing from the END! (like the last person leaving a queue 👋)
// What it does? Removes the LAST element from the array.
// Returns: The REMOVED ELEMENT.
// Modifies: The ORIGINAL ARRAY is changed (mutated).

let fruitsPop = ["apple", "banana", "orange"];
console.log("Original fruits array:", fruitsPop); // Output: Original fruits array: [ 'apple', 'banana', 'orange' ]

let removedFruitPop = fruitsPop.pop(); // Remove the last element ("orange")
console.log("Fruits array after pop:", fruitsPop);    // Output: Fruits array after pop: [ 'apple', 'banana' ]
console.log("Removed fruit (pop):", removedFruitPop);   // Output: Removed fruit (pop): orange
// Explanation: `pop()` removed "orange" from the end of `fruitsPop`. The array is now shorter, and `pop()` returned the removed fruit "orange".

// ===== 2.3. `shift()` =====
// Concept:  ⬆️  Removing from the BEGINNING! (like the first person leaving a queue 🚶💨)
// What it does? Removes the FIRST element from the array.
// Returns: The REMOVED ELEMENT.
// Modifies: The ORIGINAL ARRAY is changed (mutated).

let fruitsShift = ["apple", "banana", "orange"];
console.log("Original fruits array:", fruitsShift); // Output: Original fruits array: [ 'apple', 'banana', 'orange' ]

let removedFruitShift = fruitsShift.shift(); // Remove the first element ("apple")
console.log("Fruits array after shift:", fruitsShift);   // Output: Fruits array after shift: [ 'banana', 'orange' ]
console.log("Removed fruit (shift):", removedFruitShift);  // Output: Removed fruit (shift): apple
// Explanation: `shift()` removed "apple" from the BEGINNING of `fruitsShift`. The array is shorter, and `shift()` returned the removed fruit "apple".

// ===== 2.4. `unshift(element1, element2, ...)` =====
// Concept:  ⬇️  Adding to the BEGINNING! (like cutting in line at the front 😠... but in a good way for arrays! 😉)
// What it does? Adds one or more `element`s to the BEGINNING of the array.
// Returns: The NEW LENGTH of the array after adding elements.
// Modifies: The ORIGINAL ARRAY is changed (mutated).

let fruitsUnshift = ["banana", "orange"];
console.log("Original fruits array:", fruitsUnshift); // Output: Original fruits array: [ 'banana', 'orange' ]

let newLengthUnshift = fruitsUnshift.unshift("apple", "mango"); // Add "apple" and "mango" to the beginning
console.log("Fruits array after unshift:", fruitsUnshift);    // Output: Fruits array after unshift: [ 'apple', 'mango', 'banana', 'orange' ]
console.log("New length after unshift:", newLengthUnshift);   // Output: New length after unshift: 4
// Explanation: `unshift()` added "apple" and "mango" to the FRONT of `fruitsUnshift`. The array is now longer, and `unshift()` returned the new length (4).

// ===== 2.5. `splice(startIndex, deleteCount, item1, item2, ...)` =====
// Concept: ✂️  Powerful CUTTING and PASTING! (like a surgeon operating on an array 👨‍⚕️)
// What it does?  Can REMOVE elements and/or ADD new elements ANYWHERE in the array.
// `startIndex`:  Position to start changing the array (index).
// `deleteCount`: Number of elements to REMOVE from `startIndex`. If 0, no elements are removed.
// `item1, item2, ...`: Optional. New elements to ADD at `startIndex`.
// Returns: An array containing the DELETED elements (if any). If no elements deleted, returns an empty array.
// Modifies: The ORIGINAL ARRAY is changed (mutated).

let numbersSplice = [1, 2, 3, 4, 5];
console.log("Original numbers array:", numbersSplice); // Output: Original numbers array: [ 1, 2, 3, 4, 5 ]

// Example 2.5.1: Removing elements using splice
let removedElementsSplice = numbersSplice.splice(2, 2); // Start at index 2, delete 2 elements (3 and 4)
console.log("Numbers array after splice (remove):", numbersSplice);      // Output: Numbers array after splice (remove): [ 1, 2, 5 ]
console.log("Removed elements (splice):", removedElementsSplice);    // Output: Removed elements (splice): [ 3, 4 ]

// Example 2.5.2: Inserting elements using splice
numbersSplice.splice(1, 0, 10, 20); // Start at index 1, delete 0 elements, insert 10 and 20
console.log("Numbers array after splice (insert):", numbersSplice);      // Output: Numbers array after splice (insert): [ 1, 10, 20, 2, 5 ]
console.log("Removed elements (splice - insert only):", numbersSplice.splice(1, 0, 10, 20)); // Output: Removed elements (splice - insert only): [] (Oops! This line is wrong, it MODIFIES array again and also logs removed - should just log array after insert before this line)
console.log("Numbers array after splice (second insert - fixed logging):", numbersSplice); // Output: Numbers array after splice (second insert - fixed logging): [ 1, 10, 20, 2, 5, 10, 20 ] (Previous line was wrong, array is further modified)


// Example 2.5.3: Replacing elements using splice (remove and insert at once)
let numbersReplace = [1, 2, 3, 4, 5];
let replacedElementsSplice = numbersReplace.splice(2, 2, 30, 40); // Start at index 2, delete 2 elements (3 and 4), insert 30 and 40
console.log("Numbers array after splice (replace):", numbersReplace);     // Output: Numbers array after splice (replace): [ 1, 2, 30, 40, 5 ]
console.log("Replaced elements (splice):", replacedElementsSplice);   // Output: Replaced elements (splice): [ 3, 4 ]

// ===== 2.6. `slice(startIndex, endIndex)` =====
// Concept: 🔪  Extracting a SLICE of the array! (like cutting a piece of pizza 🍕)
// What it does?  Creates a NEW ARRAY containing a section of the original array.
// `startIndex`:  Start index of the slice (inclusive).
// `endIndex`:  End index of the slice (exclusive). Elements up to, but NOT including, `endIndex` are included.
// Returns: A NEW ARRAY containing the extracted section.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let numbersSlice = [10, 20, 30, 40, 50];
console.log("Original numbers array:", numbersSlice); // Output: Original numbers array: [ 10, 20, 30, 40, 50 ]

let slicedArray = numbersSlice.slice(1, 4); // Slice from index 1 up to (but not including) index 4
console.log("Sliced array (slice(1, 4)):", slicedArray);    // Output: Sliced array (slice(1, 4)): [ 20, 30, 40 ]
console.log("Original numbers array (unchanged):", numbersSlice); // Output: Original numbers array (unchanged): [ 10, 20, 30, 40, 50 ]
// Explanation: `slice(1, 4)` created a NEW array `slicedArray` containing elements from index 1, 2, and 3 of `numbersSlice`. The original `numbersSlice` remains unchanged.

// 📝 Note: If you only give one index `slice(startIndex)`, it takes from that index to the END of the array!
let sliceToEnd = numbersSlice.slice(3);
console.log("Slice to end (slice(3)):", sliceToEnd); // Output: Slice to end (slice(3)): [ 40, 50 ]

// ===== 2.7. `concat(array1, array2, ...)` =====
// Concept: 🔗  Joining arrays together! (like linking train carriages 🚂)
// What it does?  Creates a NEW ARRAY by joining the original array with other arrays and/or values.
// Returns: A NEW ARRAY that is the combination of the original and other arrays/values.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let array1Concat = [1, 2, 3];
let array2Concat = [4, 5, 6];
let array3Concat = [7, 8];

console.log("Array 1:", array1Concat); // Output: Array 1: [ 1, 2, 3 ]
console.log("Array 2:", array2Concat); // Output: Array 2: [ 4, 5, 6 ]
console.log("Array 3:", array3Concat); // Output: Array 3: [ 7, 8 ]

let combinedArrayConcat = array1Concat.concat(array2Concat, array3Concat, [9, 10]); // Join array1, array2, array3, and a new array [9, 10]
console.log("Combined array (concat):", combinedArrayConcat); // Output: Combined array (concat): [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
console.log("Original array 1 (unchanged):", array1Concat);    // Output: Original array 1 (unchanged): [ 1, 2, 3 ]
// Explanation: `concat()` created a NEW array `combinedArrayConcat` by joining all the arrays passed to it. The original `array1Concat` remains unchanged.

// ===== 2.8. `indexOf(element, startIndex)` =====
// Concept: 🔍  Finding the FIRST position of an element! (like searching for a specific book in a shelf 📚)
// What it does?  Searches the array for the FIRST occurrence of `element`.
// `element`: The value to search for.
// `startIndex`: Optional. Index to start searching from (default is 0).
// Returns: The INDEX of the FIRST occurrence of `element` if found. Returns -1 if not found.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let numbersIndexOf = [10, 20, 30, 20, 40];
console.log("Numbers array:", numbersIndexOf); // Output: Numbers array: [ 10, 20, 30, 20, 40 ]

let index20First = numbersIndexOf.indexOf(20); // Find the first index of 20
console.log("First index of 20:", index20First); // Output: First index of 20: 1
// Explanation: The FIRST time 20 appears in `numbersIndexOf` is at index 1.

let index20Start2 = numbersIndexOf.indexOf(20, 2); // Find index of 20, starting search from index 2
console.log("Index of 20 starting from index 2:", index20Start2); // Output: Index of 20 starting from index 2: 3
// Explanation: Starting the search from index 2, the FIRST time 20 is found is at index 3.

let index50NotFound = numbersIndexOf.indexOf(50); // Search for 50 (which is not in the array)
console.log("Index of 50 (not found):", index50NotFound); // Output: Index of 50 (not found): -1
// Explanation: 50 is not in `numbersIndexOf`, so `indexOf()` returns -1.

// ===== 2.9. `lastIndexOf(element, startIndex)` =====
// Concept: 🔎 Finding the LAST position of an element! (searching for a book from the END of the shelf 📚)
// What it does?  Searches the array for the LAST occurrence of `element`, searching backwards.
// `element`: The value to search for.
// `startIndex`: Optional. Index to start searching backwards from (default is array's last index).
// Returns: The INDEX of the LAST occurrence of `element` if found. Returns -1 if not found.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let numbersLastIndexOf = [10, 20, 30, 20, 40];
console.log("Numbers array:", numbersLastIndexOf); // Output: Numbers array: [ 10, 20, 30, 20, 40 ]

let index20Last = numbersLastIndexOf.lastIndexOf(20); // Find the last index of 20
console.log("Last index of 20:", index20Last); // Output: Last index of 20: 3
// Explanation: The LAST time 20 appears in `numbersLastIndexOf` is at index 3.

let index20Start2Last = numbersLastIndexOf.lastIndexOf(20, 2); // Find last index of 20, searching backwards from index 2
console.log("Last index of 20 searching from index 2 backwards:", index20Start2Last); // Output: Last index of 20 searching from index 2 backwards: 1
// Explanation: Searching backwards from index 2, the LAST time 20 is found is at index 1.

let index50NotFoundLast = numbersLastIndexOf.lastIndexOf(50); // Search for 50 (not found)
console.log("Last index of 50 (not found):", index50NotFoundLast); // Output: Last index of 50 (not found): -1

// ===== 2.10. `includes(element, startIndex)` =====
// Concept: ✅  Checking if an element EXISTS in the array! (like checking if you have a key in your pocket 🔑)
// What it does?  Checks if the array CONTAINS `element`.
// `element`: The value to search for.
// `startIndex`: Optional. Index to start searching from (default is 0).
// Returns: `true` if `element` is found in the array, `false` otherwise.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let fruitsIncludes = ["apple", "banana", "orange"];
console.log("Fruits array:", fruitsIncludes); // Output: Fruits array: [ 'apple', 'banana', 'orange' ]

let includesBanana = fruitsIncludes.includes("banana"); // Check if "banana" is in the array
console.log("Includes 'banana':", includesBanana); // Output: Includes 'banana': true
// Explanation: "banana" is in `fruitsIncludes`, so `includes()` returns `true`.

let includesGrape = fruitsIncludes.includes("grape"); // Check for "grape" (not in array)
console.log("Includes 'grape':", includesGrape); // Output: Includes 'grape': false
// Explanation: "grape" is not in `fruitsIncludes`, so `includes()` returns `false`.

let includesAppleStart1 = fruitsIncludes.includes("apple", 1); // Check for "apple" starting search from index 1
console.log("Includes 'apple' starting from index 1:", includesAppleStart1); // Output: Includes 'apple' starting from index 1: false
// Explanation: Starting the search from index 1, "apple" is not found (because it's at index 0), so it returns `false`.

// ===== 2.11. `join(separator)` =====
// Concept: 🧵  Stitching array elements into a STRING! (like sewing beads together on a thread 🪡)
// What it does?  Creates a NEW STRING by concatenating all elements of the array, separated by a `separator`.
// `separator`: Optional. String to separate elements. Default is comma "," if not provided.
// Returns: A NEW STRING containing all array elements joined together.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let fruitsJoin = ["apple", "banana", "mango"];
console.log("Fruits array:", fruitsJoin); // Output: Fruits array: [ 'apple', 'banana', 'mango' ]

let joinedStringComma = fruitsJoin.join(); // Join with default comma separator
console.log("Joined string (comma):", joinedStringComma); // Output: Joined string (comma): apple,banana,mango

let joinedStringSpace = fruitsJoin.join(" "); // Join with space separator
console.log("Joined string (space):", joinedStringSpace); // Output: Joined string (space): apple banana mango

let joinedStringDash = fruitsJoin.join(" - "); // Join with " - " separator
console.log("Joined string (dash):", joinedStringDash); // Output: Joined string (dash): apple - banana - mango

// ===== 2.12. `reverse()` =====
// Concept: 🔄  Flipping the array order! (like reversing a line of people ↩️)
// What it does?  Reverses the order of elements in the array IN PLACE.
// Returns: The REVERSED ARRAY (which is the same as the original array, but reversed).
// Modifies: The ORIGINAL ARRAY IS changed (mutated).

let numbersReverse = [1, 2, 3, 4, 5];
console.log("Original numbers array:", numbersReverse); // Output: Original numbers array: [ 1, 2, 3, 4, 5 ]

let reversedArrayReverse = numbersReverse.reverse(); // Reverse the array
console.log("Reversed array (reverse):", reversedArrayReverse); // Output: Reversed array (reverse): [ 5, 4, 3, 2, 1 ]
console.log("Original numbers array (now reversed):", numbersReverse); // Output: Original numbers array (now reversed): [ 5, 4, 3, 2, 1 ]
// Explanation: `reverse()` changed the order of elements in `numbersReverse` directly. Both `reversedArrayReverse` and `numbersReverse` now point to the same reversed array.

// ===== 2.13. `sort(compareFunction)` =====
// Concept: 🗂️  Putting elements in ORDER! (like sorting books on a shelf alphabetically 📚)
// What it does?  Sorts the elements of the array IN PLACE.
// By default, `sort()` sorts elements as STRINGS (lexicographically/dictionary order).
// For numbers, you need to provide a `compareFunction` to sort numerically.
// Returns: The SORTED ARRAY (which is the same as the original array, but sorted).
// Modifies: The ORIGINAL ARRAY IS changed (mutated).

// Example 2.13.1: Sorting strings (default - alphabetical order)
let fruitsSortString = ["banana", "apple", "orange"];
console.log("Original fruits array:", fruitsSortString); // Output: Original fruits array: [ 'banana', 'apple', 'orange' ]

fruitsSortString.sort(); // Sort strings alphabetically
console.log("Sorted fruits array (strings):", fruitsSortString); // Output: Sorted fruits array (strings): [ 'apple', 'banana', 'orange' ]

// Example 2.13.2: Sorting numbers (default - WRONG for numbers)
let numbersSortDefault = [3, 1, 10, 2];
console.log("Original numbers array:", numbersSortDefault); // Output: Original numbers array: [ 3, 1, 10, 2 ]

numbersSortDefault.sort(); // Sort numbers (incorrectly as strings by default!)
console.log("Sorted numbers array (default - WRONG):", numbersSortDefault); // Output: Sorted numbers array (default - WRONG): [ 1, 10, 2, 3 ]
// Explanation: Default `sort()` treats numbers as strings, so "10" comes before "2" because "1" comes before "2" in dictionary order.

// Example 2.13.3: Sorting numbers CORRECTLY (using compare function)
let numbersSortCorrect = [3, 1, 10, 2];
console.log("Original numbers array:", numbersSortCorrect); // Output: Original numbers array: [ 3, 1, 10, 2 ]

numbersSortCorrect.sort((a, b) => a - b); // Sort numbers in ascending order (smallest to largest)
console.log("Sorted numbers array (correct - ascending):", numbersSortCorrect); // Output: Sorted numbers array (correct - ascending): [ 1, 2, 3, 10 ]
// Explanation: `(a, b) => a - b` is a compare function.
// - If `a - b` is negative, `a` comes before `b`.
// - If `a - b` is positive, `b` comes before `a`.
// - If `a - b` is zero, the order doesn't change.
// For descending order (largest to smallest): `numbersSortCorrect.sort((a, b) => b - a);`

// ===== 2.14. `forEach(callbackFunction)` =====
// Concept: 🚶‍♀️🚶‍♂️  Looping through each element! (like visiting each person in a group one by one)
// What it does?  Executes a `callbackFunction` ONCE for EACH element in the array.
// `callbackFunction`: A function that will be called for each element. It usually takes three arguments:
//   - `currentValue`: The current element being processed.
//   - `index`: The index of the current element.
//   - `array`: The array `forEach()` was called on.
// Returns: `undefined`. `forEach()` does NOT create a new array or return any specific value. It's mainly for performing actions on each element.
// Modifies: The ORIGINAL ARRAY is NOT directly changed by `forEach()` itself, but the `callbackFunction` *can* modify the array if you code it to do so. (Be careful with this!).

let numbersForEach = [10, 20, 30];
console.log("Numbers array:", numbersForEach); // Output: Numbers array: [ 10, 20, 30 ]

numbersForEach.forEach((number, index) => { // For each number in the array...
    console.log(`Element at index ${index} is: ${number}`); // ...log its index and value
});
// Output:
// Element at index 0 is: 10
// Element at index 1 is: 20
// Element at index 2 is: 30

// Example of modifying array inside forEach (use with CAUTION!):
let numbersForEachModify = [1, 2, 3];
numbersForEachModify.forEach((number, index, array) => {
    array[index] = number * 2; // Double each number IN PLACE in the original array
});
console.log("Numbers array after forEach modification:", numbersForEachModify); // Output: Numbers array after forEach modification: [ 2, 4, 6 ]
// ⚠️ Modifying the array inside `forEach` can sometimes lead to unexpected behavior, especially in more complex scenarios. Be mindful!

// ===== 2.15. `map(callbackFunction)` =====
// Concept: 🗺️  TRANSFORMING each element and creating a NEW ARRAY! (like making a new map from an old one, with changes)
// What it does?  Creates a NEW ARRAY by calling a `callbackFunction` on EVERY element in the original array.
// `callbackFunction`:  Same arguments as `forEach` (`currentValue`, `index`, `array`).
// Returns: A NEW ARRAY containing the results of calling `callbackFunction` on each element.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let numbersMap = [1, 2, 3];
console.log("Numbers array:", numbersMap); // Output: Numbers array: [ 1, 2, 3 ]

let doubledNumbersMap = numbersMap.map((number) => { // For each number, return its double
    return number * 2;
});
console.log("Doubled numbers array (map):", doubledNumbersMap); // Output: Doubled numbers array (map): [ 2, 4, 6 ]
console.log("Original numbers array (unchanged):", numbersMap);    // Output: Original numbers array (unchanged): [ 1, 2, 3 ]
// Explanation: `map()` created a NEW array `doubledNumbersMap` where each element is the result of doubling the corresponding element in `numbersMap`. The original `numbersMap` is untouched.

// Concise way with arrow function (if callback is a single expression):
let tripledNumbersMapConcise = numbersMap.map(number => number * 3); // Even shorter syntax!
console.log("Tripled numbers array (map - concise):", tripledNumbersMapConcise); // Output: Tripled numbers array (map - concise): [ 3, 6, 9 ]

// ===== 2.16. `filter(callbackFunction)` =====
// Concept: 🔍  SELECTING elements based on a CONDITION! (like filtering out only the red apples from a basket 🍎➡️🧺)
// What it does?  Creates a NEW ARRAY containing only the elements from the original array that pass a test implemented by the `callbackFunction`.
// `callbackFunction`: Same arguments as `forEach` (`currentValue`, `index`, `array`).
//  It MUST return `true` if the element should be INCLUDED in the new array, and `false` if it should be EXCLUDED.
// Returns: A NEW ARRAY containing only the elements that passed the test (for which `callbackFunction` returned `true`).
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let numbersFilter = [1, 2, 3, 4, 5, 6];
console.log("Numbers array:", numbersFilter); // Output: Numbers array: [ 1, 2, 3, 4, 5, 6 ]

let evenNumbersFilter = numbersFilter.filter((number) => { // For each number, check if it's even
    return number % 2 === 0; // Return true if even, false if odd
});
console.log("Even numbers array (filter):", evenNumbersFilter); // Output: Even numbers array (filter): [ 2, 4, 6 ]
console.log("Original numbers array (unchanged):", numbersFilter);  // Output: Original numbers array (unchanged): [ 1, 2, 3, 4, 5, 6 ]
// Explanation: `filter()` created a NEW array `evenNumbersFilter` containing only the even numbers from `numbersFilter`.  The original array is unchanged.

// Concise way:
let oddNumbersFilterConcise = numbersFilter.filter(number => number % 2 !== 0); // Shorter syntax for filtering odd numbers
console.log("Odd numbers array (filter - concise):", oddNumbersFilterConcise); // Output: Odd numbers array (filter - concise): [ 1, 3, 5 ]

// ===== 2.17. `reduce(callbackFunction, initialValue)` =====
// Concept: 🧮  REDUCING the array to a SINGLE VALUE! (like summing up all items in a shopping cart to get the total price 🛒➡️💰)
// What it does?  Applies a `callbackFunction` (reducer) to each element of the array, to reduce it to a single value.
// `callbackFunction`:  Takes two main arguments (and optionally index and array):
//   - `accumulator`: The accumulated value from previous calls to the callback. It's like a running total.
//   - `currentValue`: The current element being processed.
// `initialValue`: Optional. Value to use as the first argument to the first call of the `callbackFunction`.
//                 If `initialValue` is NOT provided, the first element of the array is used as the initial accumulator, and the callback starts from the SECOND element. (Use with caution for empty arrays!).
// Returns: The SINGLE VALUE that results from the reduction.
// Modifies: The ORIGINAL ARRAY is NOT changed (non-mutating).

let numbersReduce = [1, 2, 3, 4];
console.log("Numbers array:", numbersReduce); // Output: Numbers array: [ 1, 2, 3, 4 ]

// Example 2.17.1: Summing up all numbers using reduce
let sumReduce = numbersReduce.reduce((accumulator, currentValue) => {
    console.log(`Accumulator: ${accumulator}, Current Value: ${currentValue}`); // Log accumulator and current value in each step
    return accumulator + currentValue; // Add current value to the accumulator for the next step
}, 0); // Initial value of accumulator is 0

console.log("Sum of numbers (reduce):", sumReduce); // Output: Sum of numbers (reduce): 10
// Output (with console.log inside reduce):
// Accumulator: 0, Current Value: 1
// Accumulator: 1, Current Value: 2
// Accumulator: 3, Current Value: 3
// Accumulator: 6, Current Value: 4
// Sum of numbers (reduce): 10
// Explanation:
// 1. Initial `accumulator` is 0 (provided as the second argument to `reduce`). `currentValue` is the first element (1). `accumulator + currentValue = 1`. New `accumulator` is 1.
// 2. `accumulator` is now 1, `currentValue` is 2. `accumulator + currentValue = 3`. New `accumulator` is 3.
// 3. `accumulator` is 3, `currentValue` is 3. `accumulator + currentValue = 6`. New `accumulator` is 6.
// 4. `accumulator` is 6, `currentValue` is 4. `accumulator + currentValue = 10`. New `accumulator` is 10.
// 5. Reduction is complete. `reduce()` returns the final `accumulator` value, which is 10.

// Concise way:
let productReduceConcise = numbersReduce.reduce((acc, curr) => acc * curr, 1); // Calculate product, initial value is 1 (for multiplication)
console.log("Product of numbers (reduce - concise):", productReduceConcise); // Output: Product of numbers (reduce - concise): 24

// -------------------- 3. Multidimensional Arrays --------------------
// Arrays can be nested INSIDE other arrays, like boxes within boxes! 🎁📦 This creates MULTIDIMENSIONAL ARRAYS.
// Think of them as TABLES or MATRICES 📊.

let matrix = [
    [1, 2, 3],  // Row 0
    [4, 5, 6],  // Row 1
    [7, 8, 9]   // Row 2
];
// This is a 2-dimensional array (2D array) - like a grid with rows and columns.

console.log("Matrix:", matrix);
// Output:
// Matrix: [
//   [ 1, 2, 3 ],
//   [ 4, 5, 6 ],
//   [ 7, 8, 9 ]
// ]

// Accessing elements in a multidimensional array:
// Use MULTIPLE INDICES!  `matrix[rowIndex][columnIndex]`

console.log("Element at row 0, column 0:", matrix[0][0]); // Output: Element at row 0, column 0: 1 (First row, first column)
console.log("Element at row 1, column 2:", matrix[1][2]); // Output: Element at row 1, column 2: 6 (Second row, third column)
console.log("Element at row 2, column 1:", matrix[2][1]); // Output: Element at row 2, column 1: 8 (Third row, second column)

// -------------------- 4. Example (from your instructions) --------------------

let instructionNumbers = [1, 2, 3, 4, 5];
console.log("Initial instruction numbers array:", instructionNumbers); // Output: Initial instruction numbers array: [ 1, 2, 3, 4, 5 ]

instructionNumbers.push(6); // Add 6 to the end using push()
console.log("Instruction numbers array after push(6):", instructionNumbers); // Output: Instruction numbers array after push(6): [ 1, 2, 3, 4, 5, 6 ]





// Explanation:
// 1. We start with an array `instructionNumbers` containing [1, 2, 3, 4, 5].
// 2. We use `instructionNumbers.push(6)` to add the number 6 to the END of the array.
// 3. `push()` modifies the original array. Now `instructionNumbers` is [1, 2, 3, 4, 5, 6].

// -------------------- 🎉 Chapter 7 Complete! 🎉 --------------------

// You are now a JavaScript Array MASTER! 🏆 You can create, access, and manipulate arrays like a PRO! 💪
// Keep practicing these methods and you'll be handling lists of data with ease! 🚀

// Any questions, Boss?  Don't hesitate to ask!  We're ready to conquer OBJECTS next!  🚀 Let's GO! 💨

// Quick Reference Table of Array Methods (for your SUPER BRAIN! 🧠)

/*
| Method          | Description                                                | Modifies Original Array? | Returns                                        |
|-----------------|------------------------------------------------------------|--------------------------|------------------------------------------------|
| `push()`        | Add to end                                                 | YES                      | New length                                     |
| `pop()`         | Remove from end                                            | YES                      | Removed element                                |
| `shift()`       | Remove from beginning                                      | YES                      | Removed element                                |
| `unshift()`     | Add to beginning                                           | YES                      | New length                                     |
| `splice()`      | Add/remove elements anywhere                              | YES                      | Array of deleted elements                      |
| `slice()`       | Extract section                                            | NO                       | New array (extracted section)                  |
| `concat()`      | Join arrays                                                | NO                       | New array (joined arrays)                      |
| `indexOf()`     | Find first index of element                               | NO                       | First index (or -1 if not found)              |
| `lastIndexOf()` | Find last index of element                                | NO                       | Last index (or -1 if not found)               |
| `includes()`    | Check if element exists                                    | NO                       | `true` or `false`                               |
| `join()`        | Convert array to string                                     | NO                       | New string (joined elements)                   |
| `reverse()`     | Reverse array order                                        | YES                      | Reversed array (same as original, reversed)    |
| `sort()`        | Sort array elements                                        | YES                      | Sorted array (same as original, sorted)       |
| `forEach()`     | Loop through elements                                     | NO (but callback can)     | `undefined`                                    |
| `map()`         | Transform elements, create new array                       | NO                       | New array (transformed elements)             |
| `filter()`      | Select elements based on condition, create new array       | NO                       | New array (filtered elements)                |
| `reduce()`      | Reduce array to single value                               | NO                       | Single value (result of reduction)            |
*/


