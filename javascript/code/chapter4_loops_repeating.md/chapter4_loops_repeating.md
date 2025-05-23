Okay boss, let's dive into Chapter 4: Loops - Repeating Actions. Loops are super important in programming because they allow you to execute a block of code multiple times, saving you from writing the same code repeatedly. We'll make sure you understand all types of loops and how to control them.

**Chapter 4: Loops - Repeating Actions**

**1. `for` Loops**

The `for` loop is one of the most common loops. It's perfect when you know how many times you want to repeat a block of code.

```javascript
for (initialization; condition; increment/decrement) {
    // Code to execute repeatedly
}
```

*   **Initialization:** This runs *only once* at the beginning of the loop. It's usually used to declare and initialize a counter variable (e.g., `let i = 0`).
*   **Condition:** This is checked before *each* iteration. If it's `true`, the loop continues; if it's `false`, the loop stops. (e.g., `i < 5`).
*   **Increment/Decrement:** This updates the counter variable after *each* iteration (e.g., `i++` which is `i = i+1`).

*Example*

```javascript
for (let i = 0; i < 5; i++) {
    console.log(i);
}
```

*   This loop will print numbers from `0` to `4` in the console.
*   **Explanation:**
    1.  `let i = 0`: `i` is initialized to `0`.
    2.  `i < 5`: The loop checks if `i` is less than `5`. Initially, it's true so loop start.
    3.  `console.log(i)`: The value of `i` (which is `0`) is printed.
    4.  `i++`: `i` is incremented to `1`.
    5.  The loop goes back to the condition check (`i < 5`).
    6.  Steps 2-5 are repeated until `i` is no longer less than 5 (when i = 5).

| Iteration | i | condition (i<5) | Output |
|---|---|---|---|
| 1 | 0 | true | 0 |
| 2 | 1 | true | 1 |
| 3 | 2 | true | 2 |
| 4 | 3 | true | 3 |
| 5 | 4 | true | 4 |
| 6 | 5 | false | Loop stops|

**2. `while` Loops**

The `while` loop is used when you don't know in advance how many times you'll need to repeat a block of code. It continues to loop as long as the condition is true.

```javascript
while (condition) {
    // Code to execute repeatedly
    // Update the condition inside the loop
}
```

*   The `condition` is checked *before* each iteration.
*   It's very important to update the condition inside the loop, or it might lead to an infinite loop.

*Example*

```javascript
let counter = 0;
while (counter < 3) {
    console.log("Count is: " + counter);
    counter++;
}
```

*   This will print:
    ```
    Count is: 0
    Count is: 1
    Count is: 2
    ```
*   **Explanation:**
    1.  `let counter = 0`: counter is initialized to 0
    2.  `counter < 3`: The loop checks if `counter` is less than `3`. Initially, it's true.
    3.  `console.log("Count is: " + counter)`: Prints the value of `counter`.
    4.  `counter++`: Increments the counter.
    5.  The loop goes back to step 2.
    6.  Steps 2-5 repeats until counter is not less than 3(counter = 3)

| Iteration | counter | condition (counter<3) | Output |
|---|---|---|---|
| 1 | 0 | true | Count is: 0 |
| 2 | 1 | true | Count is: 1 |
| 3 | 2 | true | Count is: 2 |
| 4 | 3 | false | Loop stops|

**3. `do...while` Loops**

The `do...while` loop is similar to the `while` loop, but with one key difference: the code block is executed *at least once* before the condition is checked.

```javascript
do {
    // Code to execute repeatedly
    // Update the condition inside the loop
} while (condition);
```

*   The code block runs, then the condition is checked.
*   The loop repeats as long as the condition is true.

*Example*

```javascript
let num = 5;
do {
    console.log("Number is: " + num);
    num--;
} while (num > 0);
```

*   This will print:

    ```
    Number is: 5
    Number is: 4
    Number is: 3
    Number is: 2
    Number is: 1
    ```
* **Explanation**
    1.  `let num = 5`: variable num is initialized to `5`
    2.  `console.log("Number is: " + num)`: Prints the value of `num`.
    3. `num--`: `num` is decremented.
    4. `num > 0`: The loop check if num is greater than `0`. If true then it goes to Step 2.
    5. Steps 2-4 repeated till the condition `num > 0` is false.

| Iteration | num | Output | condition (num>0) |
|---|---|---|---|
| 1 | 5 | Number is: 5 | true |
| 2 | 4 | Number is: 4 | true |
| 3 | 3 | Number is: 3 | true |
| 4 | 2 | Number is: 2 | true |
| 5 | 1 | Number is: 1 | true |
| 6 | 0 |  | false |

**4. Loop Control: `break` and `continue`**

These keywords allow you to control the flow within a loop.

*   **`break`:** Immediately exits the loop, regardless of the condition.

    ```javascript
    for (let i = 0; i < 10; i++) {
        if (i === 5) {
            break; // Exit the loop when i is 5
        }
        console.log(i);
    }
    ```

    *   This will print numbers from `0` to `4`. when `i = 5`, the break statement will be executed and the loop will be stopped.

*   **`continue`:** Skips the current iteration and jumps to the next iteration.

    ```javascript
    for (let i = 0; i < 5; i++) {
        if (i === 2) {
            continue; // Skip when i is 2
        }
        console.log(i);
    }
    ```

    *   This will print `0`, `1`, `3`, and `4`. When `i = 2`, continue statement skips the console statement and starts the next iteration.

**Example (from your instructions):**

```javascript
for(let i = 0; i < 5; i++) {
    console.log(i);
}
```

*   This `for` loop prints numbers from `0` to `4` as explained above.

**Expected Outcome:**

You should now be able to:

*   Use `for`, `while`, and `do...while` loops to repeat code blocks.
*   Understand when to use each type of loop.
*   Control the flow of loops using `break` and `continue`.
*   Write efficient code that can handle repetitive tasks.

That wraps up Chapter 4, boss. Remember, practice makes perfect, so try writing different kinds of loops. Any doubts, let me know. We are moving towards very important concepts. Next up is functions. Are you ready? 🚀
