// 🚀 Chapter 18: Testing in JavaScript 🧪

// 1. Basics of Unit Testing 🎯
// --------------------------
// Unit Testing: Testing individual units (functions, modules) of code in isolation. 🎯
// Think of it as checking each LEGO brick 🧱 individually before building a castle 🏰.

// Purpose of Unit Testing: ✅
//  - ✅ Verify Correctness: Ensure each part of code works as expected. Like checking if each engine part ⚙️ of a car 🚗 works correctly.
//  - 🐛 Catch Bugs Early: Find and fix issues early in development. Like finding a crack 🔍 in a building's foundation 🏗️ early on.
//  - 📈 Improve Code Quality: Write modular, testable, and better code. Like designing modular circuits 💡 for easier testing and maintenance.
//  - 🛡️ Prevent Regressions: Ensure new changes don't break existing functionality. Like making sure adding a new room 🚪 to a house 🏠 doesn't break the plumbing 🚰.

// Key Principles of Unit Testing: 🔑
//  - 🔬 Test Small Units: Focus on testing individual functions or small modules. Test a single gear ⚙️, not the whole machine 🚂.
//  -  изолированность Test in Isolation: Keep tests independent and avoid external dependencies (DB, API). Test a battery 🔋 alone, not within the whole device 📱.
//  - 💯 Test All Scenarios: Cover various inputs (valid, invalid, edge cases). Test all weather conditions ☀️🌧️❄️ for a jacket 🧥.
//  - 📝 Write Clear Test Cases: Tests should be easy to understand and maintain. Write test instructions 📃 clearly for anyone to follow.
//  - 🤖 Automate Tests: Tests should run automatically for efficiency. Set up automated assembly line 🏭 for checking products.

// Visualizing Unit Testing: 🌳
// Project
// ├── module1.js 📦
// ├── module2.js 📦
// └── test
//     ├── module1.test.js 🧪 (Tests for module1.js)
//     └── module2.test.js 🧪 (Tests for module2.js)

// ------------------------------------------------------------------------

// 2. Testing with frameworks like Jest (Optional) 🛠️ (Optional)
// ------------------------------------------------------------
// Testing Frameworks: Tools providing structure for writing, running, and managing tests. 🛠️
// Like having a toolset 🧰 for building and fixing things, instead of just using your hands.

// Jest: Popular JavaScript testing framework (especially for React). Comes with: 🌟
//  - 🏃 Test Runner: Executes tests and shows results. 🏁
//  - Assertion Library: Functions to check if expectations are met (e.g., `expect(value).toBe(expectedValue)` in Jest). 🧐
//  - Mocking Capabilities: Simulate dependencies for isolated testing. 🎭

// For this chapter, we'll focus on basic JavaScript for unit testing, without external frameworks. 🧘‍♂️
// This helps understand the core concepts of testing first. 🧱➡️🏰

// ------------------------------------------------------------------------

// 3. Writing Test Cases 📝
// -----------------------
// Test Case: Checks behavior of a specific unit of code. 📝
// Like a single step-by-step instruction 📜 to verify a feature.

// Structure of a Test Case: 🏗️
//  - Input (Setup): Prepare the data or conditions for the code unit. ⚙️ Set the stage 🎬.
//  - Execution (Action): Run the function or code to be tested. ▶️ Perform the action 🏋️‍♀️.
//  - Assertion (Verification): Check if the actual output matches the expected output. 🧐 Compare result with expectation 📊.

// Basic Test Functions (without framework): 🛠️
// -----------------------------------------

function test(testName, callback) { // Function to create a test case 🧪
    try {
        callback(); // Execute the test logic (assertion inside) ▶️
        console.log(`%c✅ ${testName}: Passed`, 'color: green'); // Test passed! 🎉 (Green color in console)
    } catch (error) {
        console.log(`%c❌ ${testName}: Failed - ${error}`, 'color: red'); // Test failed! 💔 (Red color in console)
    }
}

function assertEqual(actual, expected) { // Function to assert equality (for primitive types) 🧐===🧐
    if (actual !== expected) {
        throw new Error(`Expected ${expected}, but got ${actual}`); // Error if not equal 💔
    }
}

function assertArrayEqual(actual, expected) { // Function to assert array equality 🧐[]===🧐[]
    if (!Array.isArray(actual) || !Array.isArray(expected)) { // Check if both are arrays isArray()
        throw new Error(`Both arguments must be Arrays`); // Error if not arrays 💔
    }
    if (actual.length !== expected.length) { // Check array length 📏
        throw new Error(`Expected Array of length ${expected.length}, but got array of length ${actual.length}`); // Length mismatch error 💔
    }
    for (let i = 0; i < actual.length; i++) { // Loop through array elements 🔄
        if (actual[i] !== expected[i]) { // Check each element equality 🧐===🧐
            throw new Error(`Expected array with element ${expected[i]} at index ${i} but got ${actual[i]}`); // Element mismatch error 💔
        }
    }
}

// Example Functions to Test: 🧪 functions()
// ---------------------------------------

function add(a, b) { // Simple addition function ➕
    return a + b;
}

function multiply(a, b) { // Simple multiplication function ✖️
    return a * b;
}

function reverseString(str) { // Function to reverse a string 🔄🔤
    return str.split("").reverse().join(""); // Split, reverse, join 🔄🔤
}

function isPalindrome(str) { // Function to check if string is palindrome Palindrome? 🤔
    const reversedStr = str.split("").reverse().join(""); // Reverse the string 🔄🔤
    return str.toLowerCase() === reversedStr.toLowerCase(); // Compare original and reversed (case-insensitive) 🧐🔤===🔄🔤
}

function removeDuplicates(arr) { // Function to remove duplicates from array ✂️👯‍♀️
    return [...new Set(arr)]; // Using Set to remove duplicates and spread to array ✂️👯‍♀️➡️[]
}

// Writing Test Cases for above functions: 📝🧪
// -----------------------------------------

test("Test add function with positive numbers", function() { // Test case for add function with +ve numbers ➕🔢
    assertEqual(add(5, 10), 15); // Assert add(5, 10) equals 15 ✅
});

test("Test add function with negative numbers", function() { // Test case for add function with -ve numbers ➖🔢
    assertEqual(add(-5, -10), -15); // Assert add(-5, -10) equals -15 ✅
});

test("Test add function with zero", function() { // Test case for add function with zero 0️⃣
    assertEqual(add(5, 0), 5); // Assert add(5, 0) equals 5 ✅
});

test("Test multiply function with positive numbers", function() { // Test case for multiply function with +ve numbers ✖️🔢
    assertEqual(multiply(5, 10), 50); // Assert multiply(5, 10) equals 50 ✅
});

test("Test multiply function with negative numbers", function() { // Test case for multiply function with -ve numbers ✖️➖🔢
    assertEqual(multiply(-5, 10), -50); // Assert multiply(-5, 10) equals -50 ✅
});

test("Test multiply function with zero", function() { // Test case for multiply function with zero ✖️0️⃣
    assertEqual(multiply(5, 0), 0); // Assert multiply(5, 0) equals 0 ✅
});

test("Test reverseString function", function() { // Test case for reverseString function 🔄🔤
    assertEqual(reverseString("hello"), "olleh"); // Assert reverseString("hello") equals "olleh" ✅
});

test("Test reverseString function with empty string", function() { // Test case for reverseString function with empty string 🔄""
    assertEqual(reverseString(""), ""); // Assert reverseString("") equals "" ✅
});

test("Test isPalindrome function with palindrome string", function() { // Test case for isPalindrome function with palindrome Palindrome✅
    assertEqual(isPalindrome("madam"), true); // Assert isPalindrome("madam") equals true ✅
});

test("Test isPalindrome function with non-palindrome string", function() { // Test case for isPalindrome function with non-palindrome Palindrome❌
    assertEqual(isPalindrome("hello"), false); // Assert isPalindrome("hello") equals false ✅
});

test("Test isPalindrome function with mixed case palindrome", function() { // Test case for isPalindrome function with mixed case Palindrome✅
    assertEqual(isPalindrome("RaceCar"), true); // Assert isPalindrome("RaceCar") equals true ✅
});

test("Test removeDuplicates array with duplicate elements", function() { // Test case for removeDuplicates function with duplicates ✂️👯‍♀️
    assertArrayEqual(removeDuplicates([1, 2, 2, 3, 4, 4, 5]), [1, 2, 3, 4, 5]); // Assert removeDuplicates([..]) equals [1, 2, 3, 4, 5] ✅[]
});

test("Test removeDuplicates array with unique elements", function() { // Test case for removeDuplicates function with unique elements ✂️👯‍♀️->👤
    assertArrayEqual(removeDuplicates([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]); // Assert removeDuplicates([..]) equals [1, 2, 3, 4, 5] ✅[]
});

test("Test removeDuplicates array with empty array", function() { // Test case for removeDuplicates function with empty array ✂️[]
    assertArrayEqual(removeDuplicates([]), []); // Assert removeDuplicates([]) equals [] ✅[]
});

// ------------------------------------------------------------------------

// 4. Debugging Testing 🐞
// ----------------------
// When a test fails, find out WHY! 🐞➡️✅

// Debugging Steps: 👣
// 1. Examine Test Output: Check the error message. It shows expected vs actual. 🔍 Read the test failure message carefully.
// 2. Use console.log(): Print intermediate values in functions and tests. 📝 Add console.log() to see what's happening inside.
// 3. Check Test Input: Verify if test input is set up correctly. 🧐 Double check the input data you are providing to the function in the test.
// 4. Verify Logic: Review function's code for bugs. 🤔 Step-by-step code review of the function itself.

// Example of Debugging (Let's make a test fail intentionally): 💔
// ----------------------------------------------------------

// Let's assume our 'add' function is WRONGLY implemented as multiply: 😈
/*
function add(a, b) { // BUG: It's actually multiplying, not adding! 😈✖️
  return a * b; // Oops! 🤫 Should be 'a + b'
}
*/
// (Uncomment above function to simulate a bug)

test("Test add function with positive numbers - DEBUG EXAMPLE", function() { // Test case for 'add' function (expecting it to fail) 💔
    assertEqual(add(5, 10), 15); // Assertion: add(5, 10) should be 15 (but it's not due to the bug) ❌
});
// Run this test. It will FAIL. ❌
// Output will show: "Expected 15, but got 50"
// Debugging Process: 🐞
// 1. Examine Output: "Expected 15, but got 50". Hmm, 5 * 10 = 50... 🤔
// 2. Use console.log(): Add logs in 'add' function:
/*
function add(a, b) {
  console.log("add function called with:", a, b); // Log inputs 📝
  const result = a * b; // Still doing multiply 😈✖️
  console.log("add function result:", result); // Log result 📝
  return result;
}
*/
// Run test again. Output will show in console:
// "add function called with: 5 10"
// "add function result: 50"
// 3. Verify Logic: Looking at the 'add' function code, we see 'return a * b;' - AHA! BUG FOUND! 💡 It should be 'return a + b;'.
// 4. Fix the Bug: Change 'return a * b;' to 'return a + b;' in 'add' function. ✅
// 5. Re-run Test: Now the test will PASS! 🎉✅

// 📝 Debugging is a skill. Practice makes perfect! 🧘‍♂️🛠️

// ------------------------------------------------------------------------

// ✅ Expected Outcome: 🎉
// ----------------------
// You should now be able to:
//  - ✅ Understand basics of unit testing and its importance. 🎯✅
//  - ✅ Write basic unit tests for JavaScript functions. 📝🧪
//  - ✅ Use assertions to verify test results. 🧐✅
//  - ✅ Debug failing tests using output and console.log(). 🐞✅
//  - ✅ Write more reliable code by testing. 🛡️✅

// 🎉 Congratulations! You've completed Chapter 18 on Testing! 🎉
// Testing is crucial for writing robust applications. Keep testing your code! 🚀💻🧪💪