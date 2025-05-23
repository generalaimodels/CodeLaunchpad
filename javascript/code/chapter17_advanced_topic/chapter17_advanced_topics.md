Okay boss, let's tackle Chapter 17: Advanced Topics. This chapter will cover some of the more sophisticated and powerful features of JavaScript. We'll go into detail with each concept, so you'll be comfortable with these advanced techniques and can use them to write more elegant and efficient code.

**Chapter 17: Advanced Topics**

**1. Higher-Order Functions**

*   **Higher-Order Functions:** A higher-order function is a function that either:
    *   Takes one or more functions as arguments, or
    *   Returns a function as its result.
    *   They are a core feature of functional programming and make code more flexible and reusable.
*   JavaScript has several built-in higher-order functions for arrays (`map`, `filter`, `reduce`, `forEach`, `sort`, etc.).

**a. Examples with `map()`, `filter()`, and `reduce()`:**

*   **`map()`:** Creates a *new* array by applying a function to each element of the original array.

    ```javascript
    const numbers = [1, 2, 3, 4, 5];
    const squaredNumbers = numbers.map(function(number) {
        return number * number;
    });
    console.log(squaredNumbers); // Output: [1, 4, 9, 16, 25]

    const numbersTimes2 = numbers.map((number) => number * 2); // Using arrow function
    console.log(numbersTimes2);  // Output: [2,4,6,8,10]
    ```

*   **`filter()`:** Creates a *new* array with only the elements that pass a specific condition (defined by a function).

    ```javascript
     const numbers = [1, 2, 3, 4, 5];
    const evenNumbers = numbers.filter(function(number) {
        return number % 2 === 0;
    });
    console.log(evenNumbers); // Output: [2, 4]

    const oddNumbers = numbers.filter((number) => number%2 !== 0) // Using arrow function
    console.log(oddNumbers)   // Output: [1, 3, 5]
    ```

*   **`reduce()`:** Applies a function to each element of the array and accumulates a single result.

    ```javascript
    const numbers = [1, 2, 3, 4, 5];
    const sum = numbers.reduce(function(accumulator, currentValue) {
        return accumulator + currentValue;
    }, 0); //Initial value of accumulator is 0
    console.log(sum); // Output: 15

    const product = numbers.reduce((acc, curr) => acc*curr, 1) //Initial value of accumulator is 1
    console.log(product) // Output: 120
    ```

**b. Custom Higher-Order Functions**

You can also create your own higher-order functions.

```javascript
function operation(arr, fn) {
  const result = [];
    for(let i=0; i<arr.length; i++) {
      result.push(fn(arr[i]))
    }
    return result
}

function square(x){
  return x*x
}

function cube(x) {
  return x*x*x
}

const numbers = [1,2,3,4,5]

const squaredNumbers = operation(numbers, square)
console.log(squaredNumbers) // Output: [1,4,9,16,25]

const cubedNumbers = operation(numbers, cube)
console.log(cubedNumbers)  // Output: [1, 8, 27, 64, 125]

const multipliedBy2 = operation(numbers, (x) => x * 2)
console.log(multipliedBy2) // Output: [2, 4, 6, 8, 10]
```

*   The `operation` function takes an array and a callback function as an argument, and return a new array by applying the given callback function for each element.

**2. Currying**

*   **Currying:** It's a functional programming technique where a function that takes multiple arguments is transformed into a sequence of functions, each taking a single argument.
*   Currying allows you to create more specific functions from more general ones.

    ```javascript
    //Normal function
    function add(a,b,c) {
       return a + b + c;
    }

    console.log(add(1,2,3)) // Output: 6

    // Curried version
    function addCurried(a) {
      return function (b) {
        return function (c) {
          return a + b + c;
        };
      };
    }

    console.log(addCurried(1)(2)(3)); // Output: 6
    const add1 = addCurried(1)
    console.log(add1(2)(3))    // Output: 6
    const add1And2 = add1(2);
    console.log(add1And2(3))   // Output: 6
    ```
*   `addCurried` return a function which takes `b` as an argument and again return a function which takes `c` as an argument.
*   This is equivalent to the normal `add` function.

*   Currying is useful in cases where we want to use same arguments again and again with different arguments
    ```javascript
      function power(exponent){
        return function(base){
           return Math.pow(base, exponent)
        }
      }
       const square = power(2)
       const cube = power(3)
      console.log(square(4)); // Output: 16
      console.log(cube(4))  // Output: 64
    ```

**3. Pure Functions**

*   **Pure Function:** A pure function is a function that:
    *   Always returns the same output for the same input.
    *   Has no side effects (does not modify external variables or state).
*   Pure functions are predictable, testable, and easier to reason about.

    ```javascript
    // Pure function (no side effects)
    function add(a, b) {
        return a + b;
    }

    console.log(add(5, 10)) //Output: 15
    console.log(add(5, 10)) //Output: 15
    ```

    ```javascript
    // Impure function (has side effects)
    let x = 10;
    function addImpure(a){
        x = x + a;
        return x;
    }

    console.log(addImpure(5)) //Output: 15
    console.log(addImpure(5)) // Output: 20
    ```

*   `add` function is a pure function. For the same input, it will give same output.
*   `addImpure` function is impure. As it is updating an external variable x. For the same input, we are getting a different output.

**4. Advanced Regular Expressions**

*   **Regular Expressions (RegEx):** A powerful tool for pattern matching in strings.
*   JavaScript uses RegEx in string methods like `match`, `test`, `replace`, `search`, and more.
*   **Basic RegEx:**
    *   `/pattern/`: Regular expression literal (e.g., `/abc/`)
    *   `^`: Start of the string
    *   `$`: End of the string
    *   `.`: Any character (except newline)
    *   `*`: Zero or more occurrences of the previous character
    *   `+`: One or more occurrences of the previous character
    *   `?`: Zero or one occurrence of the previous character
    *   `[]`: Character set (e.g., `[a-z]` means any lowercase letter)
    *   `\d`: Any digit
    *   `\w`: Any alphanumeric character
    *  `|`: Or operator

*   **Examples:**

    ```javascript
    let text = "The quick brown fox jumps over the lazy dog";
    let email = "test@test.com";
    let mobile = "+91-9999999999";
    let password = "Password@123"
    let url = "https://www.example.com";

    //Check if the string contain "fox"
    console.log(/fox/.test(text)); // Output: true

    //Check if the string start with "The"
    console.log(/^The/.test(text)); // Output: true

    // Check if the string ends with "dog"
    console.log(/dog$/.test(text)); // Output: true

    // Match if string contains any lowercase letters from a-z
    console.log(/[a-z]+/.test(text)) // Output: true

    // Check for valid email address format
    console.log(/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/.test(email)) // Output: true

    //Check for valid mobile number starting with country code
    console.log(/^\+\d{2}-\d{10}$/.test(mobile)); //Output: true

    //Check for valid password which should have one uppercase, one lowercase, one special character and one number
    console.log(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/.test(password)) //Output: true

    //Match for valid URL format
     console.log(/^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/.test(url)) //Output: true
    ```
   *  Regular expressions are very powerful and can be used for different types of pattern matching.

**5. Web Workers**

*   **Web Workers:** A way to run JavaScript code in background threads (separate from the main browser UI thread).
    *   Web workers are used to run computationally intensive tasks without blocking the UI, making web applications more responsive.
*   **Key Concepts:**
    *   **Worker Thread:** The background thread where the worker code runs.
    *   **`Worker` Object:** Used to create a new web worker.
    *   **`postMessage()`:** Used to send messages to and from the worker.
    *   **`onmessage` Event:** The event listener within the worker that receives messages.
*  As Web Workers run on separate thread, it is not directly able to manipulate DOM.

**Basic Steps:**

1.  **Create a worker file** (e.g., `worker.js`):
    ```javascript
    // worker.js
    onmessage = function(event) {
        const data = event.data;
        // Perform some heavy computation
        let result = data * 2;
        postMessage(result); // send the result back to the main thread
    };
    ```
2.  **Create a worker object**
    ```javascript
         const myWorker = new Worker('worker.js');
    ```
3. **Send message to the worker using `postMessage()`**
    ```javascript
       myWorker.postMessage(5)
    ```

4.  **Listen for messages from the worker** using `onmessage`

     ```javascript
        myWorker.onmessage = function(event) {
            const result = event.data;
            console.log("Result from worker:", result); // Output: Result from worker: 10
        };
     ```

**Important Notes**

*   Web workers don't have direct access to the DOM
*   They are useful for heavy computation, network tasks.
*  Web Workers need to be run in a server or else, they will give CORS error.

**Expected Outcome:**

You should now be able to:

*   Understand and use higher-order functions (especially `map`, `filter`, `reduce`).
*   Apply currying to create more specialized functions.
*   Write pure functions and understand their benefits.
*   Use advanced regular expressions for complex pattern matching.
*  Implement basic web worker and understand it's usage.

That’s all for Chapter 17, boss! These are advanced concepts, so take your time, practice a lot, and don't hesitate to ask questions. We're moving towards testing and best practices next! Let's go! 🚀
