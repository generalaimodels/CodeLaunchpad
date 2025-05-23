// JavaScript Array Methods - Deep Dive Explanation! 🚀

// This file provides detailed code examples and explanations for each JavaScript array method
// listed in the table. Let's become Array Method Masters! 👑

// -------------------- Array Methods Table Explanation --------------------
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

// Let's explore each method with code examples and detailed comments! 👇

// ===== 1. `push(element1, element2, ...)` =====
// Description: Adds one or more elements to the END of an array. ➡️📦
// Modifies Original Array? YES ✅ (It changes the original array directly!)
// Returns: New length of the array after adding the elements. 📏
// Analogy: Imagine adding items to the back of a line. 🚶‍♀️🚶‍♂️

let fruitsPush = ["apple", "banana"]; // Initial array
console.log("Original fruitsPush array:", fruitsPush); // Output: [ 'apple', 'banana' ]

let newLengthPush = fruitsPush.push("orange", "grape"); // Add "orange" and "grape" to the end
console.log("fruitsPush array after push():", fruitsPush);   // Output: [ 'apple', 'banana', 'orange', 'grape' ] (Modified!)
console.log("Return value of push() (new length):", newLengthPush); // Output: 4

// ===== 2. `pop()` =====
// Description: Removes the LAST element from an array. ⬅️📦
// Modifies Original Array? YES ✅ (Changes the original array!)
// Returns: The removed element. 📤
// Analogy: Removing the last item from a stack. 🥞

let fruitsPop = ["apple", "banana", "orange"]; // Initial array
console.log("Original fruitsPop array:", fruitsPop); // Output: [ 'apple', 'banana', 'orange' ]

let removedFruitPop = fruitsPop.pop(); // Remove the last element ("orange")
console.log("fruitsPop array after pop():", fruitsPop);    // Output: [ 'apple', 'banana' ] (Modified!)
console.log("Return value of pop() (removed element):", removedFruitPop); // Output: orange

// ===== 3. `shift()` =====
// Description: Removes the FIRST element from an array. ⬆️📦
// Modifies Original Array? YES ✅ (Changes the original array!)
// Returns: The removed element. 📤
// Analogy: Removing the first person in a line. 🚶💨

let fruitsShift = ["apple", "banana", "orange"]; // Initial array
console.log("Original fruitsShift array:", fruitsShift); // Output: [ 'apple', 'banana', 'orange' ]

let removedFruitShift = fruitsShift.shift(); // Remove the first element ("apple")
console.log("fruitsShift array after shift():", fruitsShift);   // Output: [ 'banana', 'orange' ] (Modified!)
console.log("Return value of shift() (removed element):", removedFruitShift); // Output: apple

// ===== 4. `unshift(element1, element2, ...)` =====
// Description: Adds one or more elements to the BEGINNING of an array. ⬇️📦
// Modifies Original Array? YES ✅ (Changes the original array!)
// Returns: New length of the array after adding the elements. 📏
// Analogy: Adding items to the front of a line. 🚶‍♀️🚶‍♂️➡️

let fruitsUnshift = ["banana", "orange"]; // Initial array
console.log("Original fruitsUnshift array:", fruitsUnshift); // Output: [ 'banana', 'orange' ]

let newLengthUnshift = fruitsUnshift.unshift("apple", "mango"); // Add "apple" and "mango" to the beginning
console.log("fruitsUnshift array after unshift():", fruitsUnshift);    // Output: [ 'apple', 'mango', 'banana', 'orange' ] (Modified!)
console.log("Return value of unshift() (new length):", newLengthUnshift); // Output: 4

// ===== 5. `splice(startIndex, deleteCount, item1, item2, ...)` =====
// Description: Adds or removes elements from an array at ANY position. ✂️📦 (Versatile method!)
// Modifies Original Array? YES ✅ (Changes the original array!)
// Returns: An array containing the deleted elements (if any). [] if no elements were deleted. 📤
// Analogy: Like surgery on an array - you can cut out parts and insert new ones! 👨‍⚕️

let numbersSplice = [1, 2, 3, 4, 5]; // Initial array
console.log("Original numbersSplice array:", numbersSplice); // Output: [ 1, 2, 3, 4, 5 ]

let removedElementsSplice = numbersSplice.splice(2, 2, 6, 7); // Start at index 2, delete 2 elements (3, 4), and insert 6 and 7
console.log("numbersSplice array after splice():", numbersSplice);      // Output: [ 1, 2, 6, 7, 5 ] (Modified!)
console.log("Return value of splice() (deleted elements):", removedElementsSplice); // Output: [ 3, 4 ]

// ===== 6. `slice(startIndex, endIndex)` =====
// Description: Extracts a section of an array and returns a NEW array. 🔪📦 (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array remains unchanged!)
// Returns: A NEW array containing the extracted section. 📦✨
// Analogy: Taking a slice of cake 🍰 - you get a piece, but the original cake is still there!

let numbersSlice = [1, 2, 3, 4, 5]; // Initial array
console.log("Original numbersSlice array:", numbersSlice); // Output: [ 1, 2, 3, 4, 5 ]

let slicedArray = numbersSlice.slice(1, 4); // Extract from index 1 up to (but not including) index 4
console.log("slicedArray (returned by slice()):", slicedArray);    // Output: [ 2, 3, 4 ] (New array!)
console.log("numbersSlice array after slice():", numbersSlice);   // Output: [ 1, 2, 3, 4, 5 ] (Unchanged!)

// ===== 7. `concat(array1, array2, ...)` =====
// Description: Joins two or more arrays to create a NEW array. 🔗📦 (Non-destructive!)
// Modifies Original Array? NO ❌ (Original arrays are not changed!)
// Returns: A NEW array containing all joined arrays. 📦➕📦➡️📦✨
// Analogy: Combining train carriages 🚂 - you create a longer train, but the original carriages are still separate.

let array1Concat = [1, 2]; // Initial arrays
let array2Concat = [3, 4];
console.log("array1Concat:", array1Concat); // Output: [ 1, 2 ]
console.log("array2Concat:", array2Concat); // Output: [ 3, 4 ]

let combinedArrayConcat = array1Concat.concat(array2Concat, [5, 6]); // Join array1, array2, and [5, 6]
console.log("combinedArrayConcat (returned by concat()):", combinedArrayConcat); // Output: [ 1, 2, 3, 4, 5, 6 ] (New array!)
console.log("array1Concat after concat():", array1Concat);    // Output: [ 1, 2 ] (Unchanged!)
console.log("array2Concat after concat():", array2Concat);    // Output: [ 3, 4 ] (Unchanged!)

// ===== 8. `indexOf(element, startIndex)` =====
// Description: Returns the FIRST index at which a given element can be found in the array, or -1 if it is not present. 🔍📦 (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array is not changed!)
// Returns: First index of the element (number) or -1 if not found. 📍 or -1
// Analogy: Searching for a specific book on a shelf 📚 - you find its position (index) or realize it's not there (-1).

let numbersIndexOf = [10, 20, 30, 20]; // Initial array
console.log("numbersIndexOf array:", numbersIndexOf); // Output: [ 10, 20, 30, 20 ]

let index20First = numbersIndexOf.indexOf(20); // Find the first index of 20
console.log("Return value of indexOf(20) (first index):", index20First); // Output: 1

let index40NotFound = numbersIndexOf.indexOf(40); // Search for 40 (not present)
console.log("Return value of indexOf(40) (not found):", index40NotFound); // Output: -1

// ===== 9. `lastIndexOf(element, startIndex)` =====
// Description: Returns the LAST index at which a given element can be found in the array, or -1 if it is not present. 🔎📦 (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array is not changed!)
// Returns: Last index of the element (number) or -1 if not found. 📍 or -1
// Analogy: Searching for the last occurrence of a word in a book. 📖

let numbersLastIndexOf = [10, 20, 30, 20]; // Initial array
console.log("numbersLastIndexOf array:", numbersLastIndexOf); // Output: [ 10, 20, 30, 20 ]

let index20Last = numbersLastIndexOf.lastIndexOf(20); // Find the last index of 20
console.log("Return value of lastIndexOf(20) (last index):", index20Last); // Output: 3

let index40NotFoundLast = numbersLastIndexOf.lastIndexOf(40); // Search for 40 (not present)
console.log("Return value of lastIndexOf(40) (not found):", index40NotFoundLast); // Output: -1

// ===== 10. `includes(element, startIndex)` =====
// Description: Determines whether an array includes a certain element, returning true or false. ✅📦 (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array is not changed!)
// Returns: `true` if the element is found, `false` otherwise.  boolean (true/false)
// Analogy: Checking if you have a specific item in your bag. 🎒

let fruitsIncludes = ["apple", "banana", "orange"]; // Initial array
console.log("fruitsIncludes array:", fruitsIncludes); // Output: [ 'apple', 'banana', 'orange' ]

let includesBanana = fruitsIncludes.includes("banana"); // Check if "banana" is present
console.log("Return value of includes('banana'):", includesBanana); // Output: true

let includesGrape = fruitsIncludes.includes("grape"); // Check if "grape" is present (not present)
console.log("Return value of includes('grape'):", includesGrape); // Output: false

// ===== 11. `join(separator)` =====
// Description: Creates and returns a new string by concatenating all of the elements in an array, separated by a specified separator string. 🧵📦➡️📜 (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array is not changed!)
// Returns: A NEW string formed by joining array elements. 📜✨
// Analogy: Stringing beads 🪡 - you connect array elements into a single string using a separator.

let fruitsJoin = ["apple", "banana", "mango"]; // Initial array
console.log("fruitsJoin array:", fruitsJoin); // Output: [ 'apple', 'banana', 'mango' ]

let joinedStringComma = fruitsJoin.join(); // Join with default comma separator
console.log("Return value of join() (comma separator):", joinedStringComma); // Output: apple,banana,mango

let joinedStringSpace = fruitsJoin.join("📦 "); // Join with space separator
console.log("Return value of join(' ') (space separator):", joinedStringSpace); // Output: apple banana mango

// ===== 12. `reverse()` =====
// Description: Reverses the order of the elements in an array IN PLACE. 🔄📦
// Modifies Original Array? YES ✅ (Changes the original array!)
// Returns: The reversed array. (It's the same array instance as the original, just reversed). 📦↩️
// Analogy: Flipping a stack of cards. 🃏

let numbersReverse = [1, 2, 3, 4, 5]; // Initial array
console.log("Original numbersReverse array:", numbersReverse); // Output: [ 1, 2, 3, 4, 5 ]

let reversedArrayReverse = numbersReverse.reverse(); // Reverse the array
console.log("numbersReverse array after reverse():", numbersReverse);   // Output: [ 5, 4, 3, 2, 1 ] (Modified!)
console.log("Return value of reverse() (reversed array - same instance):", reversedArrayReverse); // Output: [ 5, 4, 3, 2, 1 ] (Same array!)

// ===== 13. `sort(compareFunction)` =====
// Description: Sorts the elements of an array IN PLACE. 🗂️📦
// Modifies Original Array? YES ✅ (Changes the original array!)
// Returns: The sorted array. (Same array instance as original, just sorted). 📦✔️
// Analogy: Alphabetizing books on a shelf 📚. By default sorts as strings! For numbers, use a compare function.

let fruitsSort = ["banana", "apple", "orange"]; // Initial array (strings)
console.log("Original fruitsSort array:", fruitsSort); // Output: [ 'banana', 'apple', 'orange' ]

fruitsSort.sort(); // Sort strings alphabetically (default sort)
console.log("fruitsSort array after sort() (strings):", fruitsSort); // Output: [ 'apple', 'banana', 'orange' ] (Modified!)

let numbersSort = [3, 1, 10, 2]; // Initial array (numbers)
console.log("Original numbersSort array:", numbersSort); // Output: [ 3, 1, 10, 2 ]

numbersSort.sort((a, b) => a - b); // Sort numbers numerically (ascending order) using compare function
console.log("numbersSort array after sort((a, b) => a - b) (numbers):", numbersSort); // Output: [ 1, 2, 3, 10 ] (Modified!)

// ===== 14. `forEach(callbackFunction)` =====
// Description: Executes a provided function ONCE for each array element. 🚶‍♀️📦 (For side effects, not for creating new arrays!)
// Modifies Original Array? NO (directly by forEach itself) ❌, but the callback function CAN modify the array! ⚠️ Be careful!
// Returns: `undefined`.  It doesn't return a new array or value.  void
// Analogy:  Going through each item in a list and performing an action on each (e.g., printing each item). 📝

let numbersForEach = [10, 20, 30]; // Initial array
console.log("numbersForEach array:", numbersForEach); // Output: [ 10, 20, 30 ]

numbersForEach.forEach((number, index) => { // For each element...
    console.log(`forEach: Element at index ${index} is ${number}`); // ...log the element and its index
});
// Output (forEach doesn't return anything, it just performs actions):
// forEach: Element at index 0 is 10
// forEach: Element at index 1 is 20
// forEach: Element at index 2 is 30

// ===== 15. `map(callbackFunction)` =====
// Description: Creates a NEW array by calling a provided function on EVERY element in the calling array. 🗺️📦➡️📦✨ (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array is not changed!)
// Returns: A NEW array with the results of calling the callback function on each element. 📦✨
// Analogy: Transforming each ingredient in a recipe and creating a new dish. 🍳➡️🍲

let numbersMap = [1, 2, 3]; // Initial array
console.log("numbersMap array:", numbersMap); // Output: [ 1, 2, 3 ]

let doubledNumbersMap = numbersMap.map((number) => { // For each element...
    return number * 2; // ...double it and return the doubled value
});
console.log("doubledNumbersMap (returned by map()):", doubledNumbersMap); // Output: [ 2, 4, 6 ] (New array!)
console.log("numbersMap array after map():", numbersMap);    // Output: [ 1, 2, 3 ] (Unchanged!)

// ===== 16. `filter(callbackFunction)` =====
// Description: Creates a NEW array with all elements that pass the test implemented by the provided function. 🔍📦➡️📦✨ (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array is not changed!)
// Returns: A NEW array containing only the elements that pass the test (for which the callback returns `true`). 📦✨
// Analogy: Filtering coffee beans ☕ - you select only the good beans based on a condition. 🫘➡️☕

let numbersFilter = [1, 2, 3, 4, 5, 6]; // Initial array
console.log("numbersFilter array:", numbersFilter); // Output: [ 1, 2, 3, 4, 5, 6 ]

let evenNumbersFilter = numbersFilter.filter((number) => { // For each element...
    return number % 2 === 0; // ...check if it's even, return true if even, false if odd
});
console.log("evenNumbersFilter (returned by filter()):", evenNumbersFilter); // Output: [ 2, 4, 6 ] (New array with only even numbers!)
console.log("numbersFilter array after filter():", numbersFilter);  // Output: [ 1, 2, 3, 4, 5, 6 ] (Unchanged!)

// ===== 17. `reduce(callbackFunction, initialValue)` =====
// Description: Executes a reducer function (provided callback) on each element of the array, resulting in a single output value. 🧮📦➡️Value✨ (Non-destructive!)
// Modifies Original Array? NO ❌ (Original array is not changed!)
// Returns: A SINGLE value that results from the reduction. 🔢✨
// Analogy:  Calculating the total sum of items in a shopping cart 🛒 - you reduce the list of prices to a single total price. 💰

let numbersReduce = [1, 2, 3, 4]; // Initial array
console.log("numbersReduce array:", numbersReduce); // Output: [ 1, 2, 3, 4 ]

let sumReduce = numbersReduce.reduce((accumulator, currentValue) => { // Reducer function
    return accumulator + currentValue; // Accumulate the sum
}, 0); // Initial value of the accumulator is 0
console.log("sumReduce (returned by reduce()):", sumReduce); // Output: 10 (Single value - the sum!)
console.log("numbersReduce array after reduce():", numbersReduce); // Output: [ 1, 2, 3, 4 ] (Unchanged!)

// -------------------- 🎉 Congratulations! You are now an Array Method Expert! 🎉 --------------------
// Keep practicing and experimenting with these methods to solidify your understanding! 💪
// You've mastered Chapter 7 and are ready for even more JavaScript adventures! 🚀 ✨