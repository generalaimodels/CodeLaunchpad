Okay boss, let's dive into Chapter 18: Testing. Writing tests is a crucial part of software development. It helps ensure your code is working as expected and prevents bugs from creeping in. We'll cover the basics of unit testing, how to write test cases, and some tips for debugging tests. Let's get started!

**Chapter 18: Testing**

**1. Basics of Unit Testing**

*   **Unit Testing:** It's a type of testing where individual units (functions, components, modules) of code are tested in isolation to verify their correctness.
*   **Purpose of Unit Testing:**
    *   **Verify correctness:** Ensures that each unit of code is functioning correctly.
    *   **Catch bugs early:** Helps identify bugs and issues early in the development cycle, making them easier to fix.
    *   **Improve code quality:** Encourages you to write more modular and testable code.
    *   **Prevent regressions:** Ensures that changes in code don't introduce new bugs in previously working code.
*   **Key Principles of Unit Testing:**
    *   **Test small units:** Focus on testing individual functions or modules.
    *   **Test in isolation:** Keep tests isolated from each other and from external dependencies.
    *   **Test all scenarios:** Cover different types of input and expected output.
    *   **Write clear test cases:** Test cases should be clear and easy to understand.
    *   **Automate tests:** Test should be automated to run and check results efficiently.

**2. Testing with frameworks like Jest (Optional)**

*   **Testing Frameworks:** Frameworks provide a structure and tools to help you write, run, and manage tests. They provide features like test runners, assertion libraries, and mocking capabilities.
*   **Jest:** Jest is a popular testing framework for JavaScript (especially with React), which comes with many features like test runner, assertion library and mocking functions.
*   Although testing frameworks like Jest are helpful, we'll focus on writing unit tests using basic JavaScript for this chapter.

**3. Writing Test Cases**

*   A test case checks the behavior of a specific unit of code.
*   It includes:
    *   **Input:** Setting up the input for the unit of code.
    *   **Execution:** Calling the function or code to be tested.
    *   **Assertion:** Verifying that the actual output matches the expected output.

**Basic structure of a test case (without using framework)**

```javascript
function test(testName, callback) {
  try {
      callback();
      console.log(`%c✅ ${testName}: Passed`, 'color: green'); // Test passed
  } catch (error) {
     console.log(`%c❌ ${testName}: Failed - ${error}`, 'color: red'); // Test failed
  }
}

function assertEqual(actual, expected) {
    if(actual !== expected){
        throw new Error(`Expected ${expected}, but got ${actual}`)
    }
}

function assertArrayEqual(actual, expected) {
    if(actual.length !== expected.length){
         throw new Error(`Expected Array of length ${expected.length}, but got array of length ${actual.length}`)
    }
     for(let i = 0; i < actual.length; i++) {
          if(actual[i] !== expected[i]){
              throw new Error(`Expected array with element ${expected[i]} at index ${i} but got ${actual[i]}`)
          }
     }
}
```

*   `test` function is used to create test case
*   `assertEqual()` function is used to assert that actual value is equal to the expected value
*  `assertArrayEqual()` function is used to assert the array value. If actual value not equal to the expected value then it will throw error.
*  We will use this test and assertion function to write tests.

*Example*

Let's write a simple function and then write test cases for it.

```javascript
function add(a, b) {
  return a + b;
}

function multiply(a, b) {
    return a * b;
}

function reverseString(str){
    return str.split("").reverse().join("");
}
function isPalindrome(str) {
    const reversedStr = str.split("").reverse().join("");
    return str === reversedStr;
}
function removeDuplicates(arr) {
  return [...new Set(arr)]
}

test("Test add function with positive number", function(){
  assertEqual(add(5, 10), 15);
})

test("Test add function with negative number", function(){
  assertEqual(add(-5, -10), -15);
})

test("Test multiply function with positive number", function(){
    assertEqual(multiply(5, 10), 50)
})
test("Test multiply function with negative number", function(){
    assertEqual(multiply(-5, 10), -50)
})

test("Test reverse string function", function(){
    assertEqual(reverseString("hello"), "olleh")
})

test("Test isPalindrome function with palindrome string", function(){
    assertEqual(isPalindrome("madam"), true)
})

test("Test isPalindrome function with non palindrome string", function(){
    assertEqual(isPalindrome("hello"), false)
})

test("Test remove duplicate array with duplicate element", function(){
  assertArrayEqual(removeDuplicates([1,2,2,3,4,4,5]), [1,2,3,4,5])
})

test("Test remove duplicate array with unique element", function(){
    assertArrayEqual(removeDuplicates([1,2,3,4,5]), [1,2,3,4,5])
})
```

**4. Debugging Testing**

*   When a test fails, you need to debug the test to understand why the output didn't match.

*   **Debugging Steps:**
    1.  **Examine the test output:** Check the error message from the failed test. It will show what the expected output was and what the actual output was.
    2.  **Use `console.log()`:** Insert `console.log()` in your functions to observe intermediate values and code flow.
    3. **Check the test input:** Sometimes test fail because the input data is wrong or there was a mistake in setting up the input value.
    4. **Verify the Logic:** Sometimes, your function might have some bug and is giving an incorrect result.

*   For example, let's assume that we have following code and test.

    ```javascript
        function add(a, b) {
          return a * b;
        }

        test("Test add function with positive number", function(){
          assertEqual(add(5, 10), 15);
        })
    ```

     *   The above test will fail. It says that it expects `15` but got `50`.
     *  This shows that there is bug in the `add` function, as it is doing `multiply` instead of `add`.
     * You can use debugging technique to identify and fix such bugs.

**Example (from your instructions):**

* The example mentioned in your instruction is about writing unit tests using basic JavaScript, which we have covered above in detail.

**Expected Outcome:**

You should now be able to:

*   Understand the basics of unit testing and its importance.
*   Write basic unit tests for your functions.
*  Use assertion to verify result of the test
*   Debug your code using test results and debugging techniques.
*  Write more reliable and robust code by writing and running tests.

That’s all for Chapter 18, boss! Testing is a crucial part of the development process. Practice writing tests for your functions. Always start by writing test before writing your function code. Any doubts? Just let me know. We are moving to last chapter, which will be best practices and performance, Let's go! 🚀
