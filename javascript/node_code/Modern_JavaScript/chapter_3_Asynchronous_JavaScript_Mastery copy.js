// -----------------------------------------------------------------------------
// Chapter 3 | Asynchronous JavaScript Mastery
// -----------------------------------------------------------------------------
// This file provides a comprehensive exploration of asynchronous JavaScript concepts,
// designed for developers seeking a deep understanding.
// -----------------------------------------------------------------------------

console.log("Chapter 3: Asynchronous JavaScript Mastery");
console.log("========================================");

// -----------------------------------------------------------------------------
// 1. Event Loop, Task Queue & Microtasks
// -----------------------------------------------------------------------------
console.log("\n--- 1. Event Loop, Task Queue & Microtasks ---");

/**
 * ## Overview
 * JavaScript is single-threaded, meaning it executes one piece of code at a time.
 * Asynchronous operations (like I/O, timers, network requests) allow the main
 * thread to continue execution without blocking. The Event Loop mechanism
 * orchestrates this process.
 *
 * ## Core Components
 * 1.  **Call Stack:** A LIFO (Last-In, First-Out) stack where function calls are pushed
 *     when invoked and popped when they return. Synchronous code execution happens here.
 * 2.  **Web APIs / Node APIs:** Environments (Browser/Node.js) provide APIs for
 *     asynchronous operations (e.g., `setTimeout`, `fetch`, `fs.readFile`). When
 *     called, these operations are handed off to the environment to handle.
 * 3.  **Task Queue (Callback Queue / Macrotask Queue):** A FIFO (First-In, First-Out)
 *     queue where callbacks associated with completed asynchronous operations handled
 *     by Web/Node APIs are placed (e.g., the callback function for `setTimeout`).
 * 4.  **Microtask Queue:** Another FIFO queue, but with higher priority than the
 *     Task Queue. Callbacks associated with Promises (`.then()`, `.catch()`,
 *     `.finally()`) and `queueMicrotask()` are placed here.
 * 5.  **Event Loop:** Continuously monitors the Call Stack and the queues. Its core
 *     logic is:
 *     a. If the Call Stack is empty, check the Microtask Queue.
 *     b. If the Microtask Queue has tasks, execute *all* of them sequentially,
 *        adding any new microtasks generated during this process to the queue
 *        and executing them as well, until the Microtask Queue is empty.
 *     c. If the Call Stack and Microtask Queue are empty, check the Task Queue.
 *     d. If the Task Queue has a task, dequeue the oldest one and push its
 *        associated callback function onto the Call Stack for execution.
 *     e. Repeat the cycle.
 *
 * ## Execution Order Priority
 * 1. Synchronous Code (Call Stack)
 * 2. Microtasks (Microtask Queue) - All microtasks are processed before any task.
 * 3. Tasks (Task Queue / Macrotask Queue) - Only one task is processed per loop cycle.
 */

console.log("Sync: Start"); // 1. Added to Call Stack, logged, popped.

setTimeout(() => {
    console.log("Task Queue: setTimeout callback (Macrotask)"); // 5. Pushed to Call Stack by Event Loop
}, 0); // 2. Handed off to Web/Node API, timer starts. After 0ms (approx), callback added to Task Queue.

Promise.resolve().then(() => {
    console.log("Microtask Queue: Promise.then callback 1"); // 4. Pushed to Call Stack by Event Loop (after sync code, before macrotask)
    Promise.resolve().then(() => {
        console.log("Microtask Queue: Nested Promise.then callback 2"); // 4.1 Added during microtask processing
    });
});

queueMicrotask(() => {
    console.log("Microtask Queue: queueMicrotask callback"); // 4.2 Added during microtask processing (or directly)
});

// Example showing microtask priority
Promise.resolve().then(() => console.log("Microtask Queue: Another Promise.then")); // 4.3

console.log("Sync: End"); // 3. Added to Call Stack, logged, popped.

// Expected Output Order (Illustrative):
// Sync: Start
// Sync: End
// Microtask Queue: Promise.then callback 1
// Microtask Queue: queueMicrotask callback
// Microtask Queue: Another Promise.then
// Microtask Queue: Nested Promise.then callback 2
// Task Queue: setTimeout callback (Macrotask)


// -----------------------------------------------------------------------------
// 2. Callbacks vs. Promises
// -----------------------------------------------------------------------------
console.log("\n--- 2. Callbacks vs. Promises ---");

/**
 * ## Callbacks
 * A callback is a function passed as an argument to another function, intended to be
 * executed ("called back") at a later time, typically after an asynchronous
 * operation completes.
 *
 * ### Pros:
 * - Simple concept for basic async operations.
 * - Fundamental pattern in early JavaScript and Node.js APIs.
 *
 * ### Cons:
 * - **Callback Hell (Pyramid of Doom):** Nested callbacks for sequential async
 *   operations lead to deeply indented, hard-to-read, and hard-to-maintain code.
 * - **Inversion of Control:** You pass your callback function to the asynchronous
 *   function, trusting it to call your function correctly (once, not too early/late,
 *   with correct arguments, handling errors appropriately). This relinquishes control.
 * - **Error Handling:** Error handling often relies on conventions (like Node.js's
 *   `error-first` pattern: `callback(err, data)`), which can be inconsistent or cumbersome.
 */

console.log("Callback Example: Simulating Async Operation");

function asyncOperationWithCallback(data, delay, callback) {
    console.log(`Callback: Starting operation with data "${data}"...`);
    setTimeout(() => {
        const success = Math.random() > 0.3; // Simulate potential failure
        if (success) {
            const result = `Processed: ${data}`;
            console.log(`Callback: Operation successful for "${data}"`);
            callback(null, result); // Node.js error-first pattern: null error, then data
        } else {
            const error = new Error(`Callback: Operation failed for "${data}"`);
            console.error(error.message);
            callback(error, null); // Pass error, null data
        }
    }, delay);
}

// Simulating sequential operations leading to "Callback Hell"
/*
asyncOperationWithCallback("Data A", 50, (errA, resultA) => {
    if (errA) {
        console.error("Callback Hell: Error A:", errA.message);
        return;
    }
    console.log("Callback Hell: Result A:", resultA);
    asyncOperationWithCallback("Data B", 70, (errB, resultB) => {
        if (errB) {
            console.error("Callback Hell: Error B:", errB.message);
            return;
        }
        console.log("Callback Hell: Result B:", resultB);
        asyncOperationWithCallback("Data C", 60, (errC, resultC) => {
            if (errC) {
                console.error("Callback Hell: Error C:", errC.message);
                return;
            }
            console.log("Callback Hell: Result C:", resultC);
            console.log("Callback Hell: All operations complete.");
            // ... potentially more nesting ...
        });
    });
});
*/
// Note: The above callback hell example is commented out to avoid excessive async logs
// during the main script execution. Run it separately if needed.

/**
 * ## Promises
 * A Promise represents the eventual result of an asynchronous operation. It acts as a
 * placeholder for a value that is not yet known. A Promise can be in one of three states:
 * 1.  **Pending:** Initial state; neither fulfilled nor rejected.
 * 2.  **Fulfilled (Resolved):** The operation completed successfully, and the Promise
 *     has a resulting value.
 * 3.  **Rejected:** The operation failed, and the Promise has a reason (an Error object).
 *
 * ### Pros:
 * - **Avoids Callback Hell:** Enables chaining (`.then()`) for sequential operations,
 *   leading to flatter, more readable code.
 * - **Improved Control:** You attach handlers (`.then()`, `.catch()`) to the Promise
 *   object, rather than passing control away via a callback.
 * - **Standardized Error Handling:** `.catch()` provides a dedicated and consistent way
 *   to handle errors for the entire chain (or specific parts).
 * - **Composability:** Promises can be easily combined (e.g., `Promise.all()`).
 *
 * ### Cons:
 * - Slightly more complex initial concept than simple callbacks.
 * - Cannot be cancelled natively (though patterns like AbortController exist).
 */

console.log("\nPromise Example: Simulating Async Operation");

function asyncOperationWithPromise(data, delay) {
    console.log(`Promise: Starting operation with data "${data}"...`);
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            const success = Math.random() > 0.3; // Simulate potential failure
            if (success) {
                const result = `Processed: ${data}`;
                console.log(`Promise: Operation successful for "${data}"`);
                resolve(result); // Fulfill the promise with the result
            } else {
                const error = new Error(`Promise: Operation failed for "${data}"`);
                console.error(error.message);
                reject(error); // Reject the promise with an error
            }
        }, delay);
    });
}

// Example of using the Promise-based function

asyncOperationWithPromise("Data P1", 80)
    .then(result => {
        console.log("Promise Usage: Result P1:", result);
        // Returning a value here wraps it in a resolved Promise implicitly
        return "Next step data";
    })
    .then(nextData => {
        console.log("Promise Usage: Received from previous step:", nextData);
        return asyncOperationWithPromise("Data P2", 60); // Chain another async operation
    })
    .then(resultP2 => {
        console.log("Promise Usage: Result P2:", resultP2);
    })
    .catch(error => {
        // Catches any rejection from the preceding chain
        console.error("Promise Usage: An error occurred in the chain:", error.message);
    })
    .finally(() => {
        // Executes regardless of whether the promise was fulfilled or rejected
        console.log("Promise Usage: Chain finished (finally).");
    });

// Note: The above promise example is commented out to avoid excessive async logs.


// -----------------------------------------------------------------------------
// 3. Promise API & Chaining
// -----------------------------------------------------------------------------
console.log("\n--- 3. Promise API & Chaining ---");

/**
 * ## Instance Methods
 * These methods are called on a Promise instance.
 *
 * 1.  **`promise.then(onFulfilled, onRejected)`:**
 *     - Attaches callbacks for the fulfilled and rejected states of the Promise.
 *     - `onFulfilled`: Executed if the Promise is fulfilled. Receives the fulfillment value.
 *     - `onRejected`: Executed if the Promise is rejected. Receives the rejection reason (error).
 *     - **Returns a new Promise:** This is crucial for chaining.
 *       - If `onFulfilled` or `onRejected` returns a value, the new Promise is fulfilled with that value.
 *       - If they throw an error, the new Promise is rejected with that error.
 *       - If they return a Promise, the new Promise adopts the state of the returned Promise.
 *
 * 2.  **`promise.catch(onRejected)`:**
 *     - Syntactic sugar for `promise.then(null, onRejected)`.
 *     - Attaches a callback specifically for the rejected state.
 *     - Also returns a new Promise, allowing further chaining (e.g., to recover from errors).
 *
 * 3.  **`promise.finally(onFinally)`:**
 *     - Attaches a callback that executes when the Promise settles (either fulfilled or rejected).
 *     - Does not receive any arguments (fulfillment value or rejection reason).
 *     - Returns a new Promise that typically settles with the same state and value/reason
 *       as the original Promise. Useful for cleanup tasks (e.g., hiding a loader).
 *       - If `onFinally` throws an error or returns a rejected Promise, the new Promise
 *         will be rejected with that error/reason, overriding the original settlement.
 */

console.log("Promise Chaining Example:");

function step(num, delay, shouldFail = false) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            console.log(`Step ${num} executing...`);
            if (shouldFail) {
                reject(new Error(`Step ${num} failed!`));
            } else {
                resolve(`Result from Step ${num}`);
            }
        }, delay);
    });
}


step(1, 50)
    .then(result1 => {
        console.log("Chain:", result1); // Result from Step 1
        // The return value is passed to the next .then()
        return step(2, 70);
    })
    .then(result2 => {
        console.log("Chain:", result2); // Result from Step 2
        // Simulate an error within a .then() handler
        // throw new Error("Error after Step 2");
        return step(3, 60, true); // Simulate a failing async step
    })
    .then(result3 => {
        // This block will be skipped if the previous step rejects or throws
        console.log("Chain:", result3);
    })
    .catch(error => {
        console.error("Chain: Caught Error:", error.message);
        // Can recover from the error by returning a non-Promise value
        // or another Promise. If nothing is returned, the chain continues
        // with 'undefined' wrapped in a resolved Promise.
        // return "Recovered from error";
        // Re-throwing or returning a rejected Promise propagates the error
        // throw new Error("Propagated error");
    })
    .then(finalResult => {
        // This runs if the catch block doesn't re-throw or return a rejected Promise
        console.log("Chain: After catch block. Result:", finalResult);
    })
    .finally(() => {
        console.log("Chain: Finally block executed.");
    });

// Note: The above promise chaining example is commented out.

/**
 * ## Static Methods
 * These methods are called directly on the `Promise` constructor.
 *
 * 1.  **`Promise.resolve(value)`:**
 *     - Returns a Promise object that is resolved with the given value.
 *     - If the value is a thenable (an object with a `.then` method), the returned
 *       Promise will "follow" that thenable, adopting its eventual state.
 *     - If the value is already a Promise, it's returned as is.
 *
 * 2.  **`Promise.reject(reason)`:**
 *     - Returns a Promise object that is rejected with the given reason (usually an Error).
 *
 * 3.  **`Promise.all(iterable)`:**
 *     - Takes an iterable (e.g., an Array) of Promises.
 *     - Returns a single Promise that fulfills when *all* Promises in the iterable
 *       have fulfilled. The fulfillment value is an array of the fulfillment values
 *       (in the same order as the input iterable).
 *     - Rejects immediately if *any* of the Promises in the iterable reject. The
 *       rejection reason is the reason of the first Promise that rejected.
 *
 * 4.  **`Promise.race(iterable)`:**
 *     - Takes an iterable of Promises.
 *     - Returns a single Promise that settles (fulfills or rejects) as soon as
 *       *the first* Promise in the iterable settles. The resulting Promise adopts
 *       the state and value/reason of that first settled Promise.
 *
 * 5.  **`Promise.allSettled(iterable)`:**
 *     - Takes an iterable of Promises.
 *     - Returns a single Promise that fulfills when *all* Promises in the iterable
 *       have settled (either fulfilled or rejected).
 *     - The fulfillment value is an array of objects, each describing the outcome
 *       of a Promise in the iterable:
 *       - `{ status: 'fulfilled', value: V }`
 *       - `{ status: 'rejected', reason: R }`
 *     - Useful when you need to know the result of every Promise, regardless of success or failure.
 *
 * 6.  **`Promise.any(iterable)`:**
 *     - Takes an iterable of Promises.
 *     - Returns a single Promise that fulfills as soon as *any* of the Promises in
 *       the iterable fulfills. The fulfillment value is the value of the first
 *       Promise that fulfilled.
 *     - Rejects only if *all* Promises in the iterable reject. The rejection reason
 *       is an `AggregateError` containing an array of all rejection reasons.
 */

console.log("\nPromise Static Methods Examples:");

const p1 = Promise.resolve("P1 Resolved");
const p2 = new Promise(resolve => setTimeout(() => resolve("P2 Resolved after 100ms"), 100));
const p3 = new Promise((_, reject) => setTimeout(() => reject(new Error("P3 Rejected after 50ms")), 50));
const p4 = Promise.reject("P4 Rejected Immediately");
const p5 = new Promise(resolve => setTimeout(() => resolve("P5 Resolved after 150ms"), 150));

// Promise.all example (will reject because p3 rejects first)

Promise.all([p1, p2, p5])
    .then(results => console.log("Promise.all (success):", results)) // [ 'P1 Resolved', 'P2 Resolved after 100ms', 'P5 Resolved after 150ms' ]
    .catch(err => console.error("Promise.all (rejected):", err.message)); // Not hit in this case

Promise.all([p1, p2, p3, p5]) // Add p3 which rejects
    .then(results => console.log("Promise.all (success with reject):", results)) // Not hit
    .catch(err => console.error("Promise.all (rejected with reject):", err.message)); // P3 Rejected after 50ms


// Promise.race example (will reject because p3 rejects fastest)

Promise.race([p2, p3, p5])
    .then(result => console.log("Promise.race (fulfilled):", result)) // Not hit
    .catch(err => console.error("Promise.race (rejected):", err.message)); // P3 Rejected after 50ms

Promise.race([p1, p2, p5]) // p1 resolves immediately
    .then(result => console.log("Promise.race (fulfilled first):", result)) // P1 Resolved
    .catch(err => console.error("Promise.race (rejected first):", err.message)); // Not hit


// Promise.allSettled example (always fulfills with status objects)

Promise.allSettled([p1, p2, p3, p5])
    .then(results => {
        console.log("Promise.allSettled results:");
        results.forEach(r => {
            if (r.status === 'fulfilled') {
                console.log(`  - Fulfilled: ${r.value}`);
            } else {
                console.log(`  - Rejected: ${r.reason.message}`);
            }
        });
    });
    // Output (order depends slightly on timing, but structure is consistent):
    // Promise.allSettled results:
    //   - Fulfilled: P1 Resolved
    //   - Rejected: P3 Rejected after 50ms
    //   - Fulfilled: P2 Resolved after 100ms
    //   - Fulfilled: P5 Resolved after 150ms

// Promise.any example (fulfills with the first fulfilled promise)

Promise.any([p3, p4, p2, p5]) // p2 fulfills first among successful ones
    .then(result => console.log("Promise.any (fulfilled):", result)) // P2 Resolved after 100ms
    .catch(err => console.error("Promise.any (rejected):", err)); // Not hit

Promise.any([p3, p4]) // All reject
    .then(result => console.log("Promise.any (all reject - fulfilled):", result)) // Not hit
    .catch(err => {
        console.error("Promise.any (all reject - rejected):", err instanceof AggregateError, err.errors.map(e => e?.message || e));
        // Output: Promise.any (all reject - rejected): true [ 'P3 Rejected after 50ms', 'P4 Rejected Immediately' ]
    });

// Note: Static method examples are commented out.

// -----------------------------------------------------------------------------
// 4. `async` / `await` Internals
// -----------------------------------------------------------------------------
console.log("\n--- 4. `async` / `await` Internals ---");

/**
 * ## Overview
 * `async` and `await` are keywords introduced in ES2017 (ES8) that provide
 * syntactic sugar on top of Promises, making asynchronous code look and behave
 * more like synchronous code.
 *
 * ## `async` Keyword
 * - Used to declare an asynchronous function (`async function myFunction() { ... }`).
 * - An `async` function *always* implicitly returns a Promise.
 * - If the function explicitly returns a value, that value becomes the fulfillment
 *   value of the returned Promise.
 * - If the function throws an error, the returned Promise is rejected with that error.
 *
 * ## `await` Keyword
 * - Can *only* be used inside an `async` function (or at the top level of modules).
 * - Pauses the execution of the `async` function until the Promise it's applied to settles.
 * - If the Promise fulfills, `await` returns the fulfillment value.
 * - If the Promise rejects, `await` throws the rejection reason (error). This allows
 *   using standard `try...catch` blocks for error handling.
 *
 * ## Internals (Simplified View)
 * Conceptually, `async/await` can be thought of as being built upon Promises and
 * potentially Generators (though the exact engine implementation might differ).
 * When an `await` is encountered:
 * 1. The `async` function's execution pauses.
 * 2. The state of the function (variables, etc.) is saved.
 * 3. Control is returned to the caller (event loop), allowing other code to run.
 * 4. When the awaited Promise settles:
 *    - If fulfilled, the `async` function resumes execution after the `await`,
 *      with the fulfillment value being the result of the `await` expression.
 *    - If rejected, an error is thrown *at the point of the `await`* within the
 *      `async` function, which can be caught by a `try...catch`.
 *
 * ## Error Handling
 * Use standard `try...catch...finally` blocks within `async` functions to handle
 * errors from awaited Promises or synchronous code within the function.
 */

console.log("async/await Example:");

// Re-using the Promise-based function from before
// function asyncOperationWithPromise(data, delay) { ... }

async function processDataAsync() {
    console.log("Async/Await: Starting process...");
    try {
        // await pauses execution until the promise resolves/rejects
        const result1 = await asyncOperationWithPromise("Async Data 1", 80);
        console.log("Async/Await: Result 1:", result1);

        // The resolved value is directly available
        const result2 = await asyncOperationWithPromise("Async Data 2", 60);
        console.log("Async/Await: Result 2:", result2);

        // Simulating a failing operation
        const result3 = await asyncOperationWithPromise("Async Data 3 (Fail)", 50); // This would throw
        console.log("Async/Await: Result 3:", result3); // This line wouldn't execute if above fails

        // Simulating synchronous error
        if (Math.random() > 0.5) {
            throw new Error("Async/Await: Simulated synchronous error!");
        }

        console.log("Async/Await: All operations successful.");
        return "Process Completed Successfully"; // This becomes the fulfillment value of the promise returned by processDataAsync
    } catch (error) {
        console.error("Async/Await: Error caught:", error.message);
        // The function will return a rejected promise if an error is caught and not handled/re-thrown
        // return "Process Failed"; // Can return a value to indicate failure gracefully
        throw error; // Re-throw to propagate the rejection
    } finally {
        console.log("Async/Await: Process finished (finally block).");
    }
}

// Calling the async function and handling its returned Promise

processDataAsync()
    .then(finalResult => {
        console.log("Async/Await Caller: Success -", finalResult);
    })
    .catch(error => {
        console.error("Async/Await Caller: Failure -", error.message);
    });

// Note: The above async/await example is commented out.

// Top-level await (supported in ES Modules and some environments like modern Node.js)
try {
    console.log("Top-Level Await: Starting...");
    const result = await asyncOperationWithPromise("Top-Level Data", 40);
    console.log("Top-Level Await: Result:", result);
} catch (error) {
    console.error("Top-Level Await: Error:", error.message);
}


// -----------------------------------------------------------------------------
// 5. Generators & Co Routines
// -----------------------------------------------------------------------------
console.log("\n--- 5. Generators & Co Routines ---");

/**
 * ## Generators (`function*`, `yield`)
 * Introduced in ES6 (ES2015), generator functions are a special type of function
 * that can be paused and resumed, allowing their execution context (variable bindings)
 * to be saved across pauses.
 *
 * - **Declaration:** Use `function* functionName() { ... }` or `function*() { ... }`.
 * - **`yield` Keyword:** Pauses the generator function's execution and returns a value
 *   to the caller. When resumed, execution continues from the point immediately
 *   after the `yield`. `yield` can also receive a value back when resumed.
 * - **Iterator Protocol:** When a generator function is called, it doesn't execute
 *   its body immediately. Instead, it returns a special *iterator* object (Generator object).
 * - **`iterator.next(value)`:** Calling `.next()` on the iterator resumes the generator's
 *   execution until the next `yield`, `return`, or `throw`.
 *   - It returns an object like `{ value: yieldedValue, done: false }`.
 *   - `value`: The value yielded by the `yield` expression (or the return value).
 *   - `done`: `false` if the generator yielded, `true` if it returned or finished.
 *   - The optional `value` passed to `.next(value)` becomes the result of the `yield`
 *     expression *inside* the generator function upon resumption.
 *
 * ## Co-routines (Using Generators for Async)
 * Before `async/await`, generators were used with helper functions (co-routine runners)
 * to manage asynchronous operations in a synchronous-looking style. The basic idea:
 * 1. The generator function `yield`s a Promise (or other representation of an async operation).
 * 2. The co-routine runner receives the Promise.
 * 3. The runner waits for the Promise to settle using `.then()`.
 * 4. When the Promise fulfills, the runner calls `iterator.next(fulfillmentValue)` to resume the generator, passing the result back.
 * 5. If the Promise rejects, the runner calls `iterator.throw(rejectionReason)` to throw the error inside the generator, allowing `try...catch` within the generator.
 *
 * `async/await` essentially standardizes and simplifies this co-routine pattern.
 */

console.log("Generator Example:");

function* idGenerator() {
    let id = 1;
    console.log("Generator: Starting");
    yield id++; // Pause 1
    console.log("Generator: Resumed after 1");
    yield id++; // Pause 2
    console.log("Generator: Resumed after 2");
    yield id++; // Pause 3
    console.log("Generator: Resumed after 3");
    return "Finished"; // Finish
}

const gen = idGenerator(); // Creates the generator object, doesn't run code yet

console.log("Generator Caller: Calling next() 1");
let result = gen.next(); // Starts execution, pauses at first yield
console.log("Generator Caller: Received:", result); // { value: 1, done: false }

console.log("Generator Caller: Calling next() 2");
result = gen.next(); // Resumes, pauses at second yield
console.log("Generator Caller: Received:", result); // { value: 2, done: false }

// Passing value back into generator
function* valueReceiver() {
    console.log("Receiver Gen: Started");
    const receivedValue = yield "Ready for value"; // Pause 1, yield string
    console.log("Receiver Gen: Received:", receivedValue);
    const receivedValue2 = yield `Got ${receivedValue}, send another`; // Pause 2
    console.log("Receiver Gen: Received 2:", receivedValue2);
    return "Receiver Done";
}

const receiverGen = valueReceiver();
console.log("Receiver Caller: next() 1");
console.log(receiverGen.next()); // { value: 'Ready for value', done: false }
console.log("Receiver Caller: next('Hello')");
console.log(receiverGen.next("Hello")); // Resumes, logs "Received: Hello", yields `Got Hello, send another` -> { value: 'Got Hello, send another', done: false }
console.log("Receiver Caller: next('World')");
console.log(receiverGen.next("World")); // Resumes, logs "Received 2: World", returns -> { value: 'Receiver Done', done: true }
console.log("Receiver Caller: next() after done");
console.log(receiverGen.next()); // { value: undefined, done: true }


console.log("\nCo-routine Example (Simplified Runner):");

// Simple co-routine runner function
function runCoroutine(generatorFunction) {
    const iterator = generatorFunction(); // Get the iterator

    function handleNext(value) {
        const result = iterator.next(value); // Resume generator, pass value in

        if (result.done) {
            // Generator finished, resolve the main promise with the return value
            return Promise.resolve(result.value);
        } else {
            // Generator yielded a value (expecting a Promise)
            // Ensure the yielded value is a Promise
            return Promise.resolve(result.value)
                .then(
                    // If yielded promise resolves, pass the value back into generator
                    res => handleNext(res),
                    // If yielded promise rejects, throw the error back into generator
                    err => handleThrow(err)
                );
        }
    }

    function handleThrow(err) {
        try {
            const result = iterator.throw(err); // Throw error into generator

            if (result.done) {
                 // Generator finished (maybe caught the error and returned)
                return Promise.resolve(result.value);
            } else {
                 // Generator caught the error and yielded something else
                return Promise.resolve(result.value)
                    .then(res => handleNext(res), e => handleThrow(e));
            }
        } catch (uncaughtError) {
             // Generator didn't catch the thrown error, reject the main promise
            return Promise.reject(uncaughtError);
        }
    }


    // Start the process
    return handleNext();
}

// Generator function designed for the co-routine runner
function* asyncTaskGenerator() {
    console.log("Co-routine Gen: Starting...");
    try {
        const resultA = yield asyncOperationWithPromise("Co-routine Data A", 80); // yield a Promise
        console.log("Co-routine Gen: Result A:", resultA);

        const resultB = yield asyncOperationWithPromise("Co-routine Data B", 60); // yield another Promise
        console.log("Co-routine Gen: Result B:", resultB);

        // Yielding a failing promise
        // const resultC = yield asyncOperationWithPromise("Co-routine Data C (Fail)", 50);
        // console.log("Co-routine Gen: Result C:", resultC); // Skipped if above fails

        return "Co-routine Generator Finished Successfully";
    } catch (error) {
        console.error("Co-routine Gen: Caught error:", error.message);
        return "Co-routine Generator Finished with Error";
    } finally {
        console.log("Co-routine Gen: Finally block.");
    }
}

// Execute the generator using the co-routine runner
runCoroutine(asyncTaskGenerator)
    .then(finalResult => {
        console.log("Co-routine Runner: Success -", finalResult);
    })
    .catch(error => {
        console.error("Co-routine Runner: Failure -", error.message);
    });

// Note: The above co-routine example is commented out.


// -----------------------------------------------------------------------------
// 6. Cancellation Patterns & AbortController
// -----------------------------------------------------------------------------
console.log("\n--- 6. Cancellation Patterns & AbortController ---");

/**
 * ## The Need for Cancellation
 * Often, asynchronous operations (like fetching data, long computations) might
 * become irrelevant before they complete (e.g., user navigates away, types a new
 * search query). Continuing these operations wastes resources (CPU, network, memory)
 * and can lead to unexpected behavior if their results arrive later.
 *
 * ## Older Patterns (Less Standard)
 * - **Flags:** Using a boolean variable (`isCancelled`) checked within the async
 *   operation's callbacks or loops. Requires manual implementation and passing the
 *   flag around.
 * - **Custom Promise Wrappers:** Creating Promises with an added `.cancel()` method.
 *   This is non-standard and breaks Promise interoperability.
 *
 * ## `AbortController` and `AbortSignal` (Standard API)
 * This is the modern, standard way to signal cancellation requests for DOM requests
 * (like `fetch`) and other async APIs that support it.
 *
 * 1.  **`AbortController`:**
 *     - An object used to manage the cancellation signal.
 *     - `const controller = new AbortController();`
 *     - **`controller.signal`:** Returns an `AbortSignal` object associated with this controller.
 *       This signal object is passed to the asynchronous operation.
 *     - **`controller.abort(reason)`:** Signals abortion to any operations listening
 *       to the associated `AbortSignal`. The optional `reason` can be any value
 *       (often an Error) indicating why the abortion occurred.
 *
 * 2.  **`AbortSignal`:**
 *     - Represents the signal itself. Cannot be triggered directly.
 *     - **`signal.aborted`:** A boolean indicating if `controller.abort()` has been called.
 *     - **`signal.reason`:** The reason passed to `controller.abort()`, if any (available in newer environments).
 *     - **`signal.onabort`:** An event handler property that gets called when `abort()` is invoked.
 *     - **`signal.addEventListener('abort', listener)`:** Standard event listener mechanism.
 *
 * ## Usage with `fetch`
 * The `fetch` API natively supports `AbortSignal`. Pass the signal as an option:
 * ```javascript
 * const controller = new AbortController();
 * const signal = controller.signal;
 *
 * fetch(url, { signal })
 *   .then(response => { ... })
 *   .catch(error => {
 *     if (error.name === 'AbortError') {
 *       console.log('Fetch aborted!');
 *     } else {
 *       console.error('Fetch error:', error);
 *     }
 *   });
 *
 * // To cancel:
 * // setTimeout(() => controller.abort(), 1000);
 * ```
 * When `controller.abort()` is called, the `fetch` promise rejects with a `DOMException` named `'AbortError'`.
 *
 * ## Implementing Cancellation in Custom Functions
 * For your own Promise-based functions, you can accept an `AbortSignal` as an argument
 * and check its `aborted` status or listen for the `abort` event.
 */

console.log("AbortController Example:");

function cancellableAsyncTask(data, delay, signal) {
    return new Promise((resolve, reject) => {
        console.log(`Cancellable Task: Starting for "${data}" (will take ${delay}ms)`);

        // Check if already aborted before starting timer
        if (signal?.aborted) {
            console.warn(`Cancellable Task: Aborted before start for "${data}".`);
            // Reject with the reason provided to abort() if available, else a generic AbortError
            return reject(signal.reason || new DOMException('Operation aborted', 'AbortError'));
        }

        const timeoutId = setTimeout(() => {
            console.log(`Cancellable Task: Completed successfully for "${data}".`);
            resolve(`Processed: ${data}`);
            // Clean up listener if task completes normally
            signal?.removeEventListener('abort', abortListener);
        }, delay);

        // Listener function for the abort event
        const abortListener = () => {
            clearTimeout(timeoutId); // Crucial: Stop the scheduled work
            console.warn(`Cancellable Task: Abort signal received for "${data}".`);
            // Reject with the reason provided to abort() if available, else a generic AbortError
            reject(signal.reason || new DOMException('Operation aborted', 'AbortError'));
        };

        // Attach the listener to the signal
        signal?.addEventListener('abort', abortListener, { once: true }); // {once: true} automatically removes listener after firing
    });
}

const controller = new AbortController();
const signal = controller.signal;
const customAbortReason = new Error("User cancelled the operation");

// Start the cancellable task
/*
cancellableAsyncTask("Task Data 1", 150, signal)
    .then(result => console.log("Cancellable Task Caller: Success -", result))
    .catch(error => {
        if (error.name === 'AbortError') {
            console.error("Cancellable Task Caller: Aborted -", error.message);
            if (error === customAbortReason) {
                console.log("(Custom abort reason detected)");
            }
        } else {
            console.error("Cancellable Task Caller: Other Error -", error);
        }
    });
*/

// Start another task that will definitely be cancelled
/*
const controller2 = new AbortController();
cancellableAsyncTask("Task Data 2 (to be cancelled)", 200, controller2.signal)
    .then(result => console.log("Cancellable Task 2 Caller: Success -", result))
    .catch(error => {
         if (error.name === 'AbortError') {
            console.error("Cancellable Task 2 Caller: Aborted -", error.message);
         } else {
             console.error("Cancellable Task 2 Caller: Other Error -", error);
         }
    });
*/

// Abort the first task after 70ms (before it completes)
// setTimeout(() => {
//     console.log("Main: Aborting Task 1...");
//     controller.abort(customAbortReason);
// }, 70);

// Abort the second task immediately
// console.log("Main: Aborting Task 2 immediately...");
// controller2.abort();

// Note: The above AbortController examples are commented out.


// -----------------------------------------------------------------------------
// 7. Throttling, Debouncing & Rate Limiting
// -----------------------------------------------------------------------------
console.log("\n--- 7. Throttling, Debouncing & Rate Limiting ---");

/**
 * These are techniques to control how often a function is allowed to execute,
 * especially useful for event handlers attached to frequent events (scroll, resize, keypress)
 * or for limiting API calls.
 *
 * ## Debouncing
 * - **Goal:** Group multiple sequential calls to a function into a single call after
 *   a certain period of inactivity (a "quiet" period).
 * - **Analogy:** Waiting for someone to finish typing before suggesting auto-completions.
 * - **Mechanism:** When the debounced function is called, a timer is set (or reset).
 *   The actual function execution is delayed until the timer expires *without*
 *   being reset by another call.
 * - **Use Cases:** Search input suggestions, validating form fields after user stops typing,
 *   saving drafts automatically.
 *
 * ## Throttling
 * - **Goal:** Ensure a function is executed at most once within a specified time interval.
 * - **Analogy:** A turnstile allowing only one person through every few seconds.
 * - **Mechanism:** When the throttled function is called, it executes immediately
 *   (usually, depending on implementation) and then enters a "cooldown" period.
 *   Any calls during the cooldown are ignored (or queued for execution *after* the cooldown).
 * - **Use Cases:** Handling scroll or resize events to update UI without excessive re-renders,
 *   rate-limiting button clicks to prevent accidental double submissions.
 *
 * ## Rate Limiting
 * - **Goal:** Restrict the number of times an action (typically an API request) can
 *   be performed within a specific time window.
 * - **Difference from Throttling:** Throttling focuses on the minimum time *between*
 *   executions, while rate limiting focuses on the maximum *number* of executions
 *   within a window.
 * - **Mechanism:** Often implemented server-side, but can be simulated client-side.
 *   Tracks the number of calls within the current window. If the limit is exceeded,
 *   subsequent calls are blocked or delayed until the window resets.
 * - **Use Cases:** Preventing API abuse, ensuring fair usage of shared resources.
 */

console.log("Debouncing Example:");

function debounce(func, wait) {
    let timeoutId = null;

    return function(...args) {
        const context = this; // Preserve context (`this`)

        clearTimeout(timeoutId); // Clear the previous timer on every call

        timeoutId = setTimeout(() => {
            timeoutId = null; // Clear the id *before* executing
            func.apply(context, args); // Execute the original function
        }, wait);
    };
}

function handleInput(event) {
    // In a real scenario, event might be used, e.g., event.target.value
    console.log(`Debounced: Input changed! Value: ${event?.target?.value || '(simulated)'}`);
    // Perform search suggestion fetch, validation, etc.
}

const debouncedHandleInput = debounce(handleInput, 300); // Wait 300ms of inactivity

// Simulate rapid input events
/*
console.log("Simulating rapid input for debouncing (expect one log after ~300ms delay)...");
debouncedHandleInput({ target: { value: 't' } });
setTimeout(() => debouncedHandleInput({ target: { value: 'te' } }), 50);
setTimeout(() => debouncedHandleInput({ target: { value: 'tes' } }), 100);
setTimeout(() => debouncedHandleInput({ target: { value: 'test' } }), 200); // This call resets the timer
// Expected: handleInput logs "test" about 300ms after the last call (at t=500ms)
*/

console.log("\nThrottling Example:");

function throttle(func, limit) {
    let inThrottle = false;
    let lastResult;
    let lastArgs; // Store the last arguments if called during cooldown

    return function(...args) {
        const context = this;
        lastArgs = args; // Always update lastArgs

        if (!inThrottle) {
            inThrottle = true; // Enter cooldown

            // Execute immediately
            lastResult = func.apply(context, lastArgs);

            setTimeout(() => {
                inThrottle = false; // End cooldown
                // Optional: Check if it was called again during cooldown and execute with latest args
                // if (lastArgs) {
                //    lastResult = func.apply(context, lastArgs);
                //    lastArgs = null; // Clear stored args after execution
                // }
            }, limit);
        }
        // Always return the result of the *last* execution (or undefined initially)
        return lastResult;
    };
}


// Alternative Throttling (Leading and Trailing edge option)
function throttleAdvanced(func, wait, options = {}) {
  let context, args, result;
  let timeout = null;
  let previous = 0;
  if (!options) options = {};

  const later = function() {
    previous = options.leading === false ? 0 : Date.now();
    timeout = null;
    result = func.apply(context, args);
    if (!timeout) context = args = null; // Clean up references
  };

  const throttled = function(...funcArgs) {
    const now = Date.now();
    if (!previous && options.leading === false) previous = now;
    const remaining = wait - (now - previous);
    context = this;
    args = funcArgs;

    if (remaining <= 0 || remaining > wait) {
      if (timeout) {
        clearTimeout(timeout);
        timeout = null;
      }
      previous = now;
      result = func.apply(context, args);
      if (!timeout) context = args = null;
    } else if (!timeout && options.trailing !== false) {
      // Schedule trailing edge execution
      timeout = setTimeout(later, remaining);
    }
    return result;
  };

  throttled.cancel = function() {
      clearTimeout(timeout);
      previous = 0;
      timeout = context = args = null;
  };

  return throttled;
}


function handleScroll(event) {
    // In a real scenario, event might be used, e.g., window.scrollY
    console.log(`Throttled: Scroll event! Position: ${event?.position || '(simulated)'}`);
    // Update UI based on scroll position
}

// Throttle to execute at most once every 500ms
const throttledHandleScroll = throttleAdvanced(handleScroll, 500, {leading: true, trailing: true});

// Simulate rapid scroll events
/*
console.log("Simulating rapid scroll for throttling (expect logs roughly every 500ms)...");
let scrollPos = 0;
const scrollInterval = setInterval(() => {
    scrollPos += 20;
    throttledHandleScroll({ position: scrollPos });
}, 100); // Fire event every 100ms

// Stop simulation after 2 seconds
setTimeout(() => {
    clearInterval(scrollInterval);
    // Make one final call after stopping to ensure trailing edge (if configured) fires
    throttledHandleScroll({ position: scrollPos + 10});
    console.log("Scroll simulation stopped.");
    // Optionally cancel if needed: throttledHandleScroll.cancel();
}, 2100);
*/
// Note: Debounce/Throttle simulations are commented out.

console.log("\nRate Limiting (Conceptual):");
// Rate limiting is typically complex and often managed server-side.
// Client-side simulation might involve:
// - A queue for pending requests.
// - A counter for requests made within the current time window.
// - `setTimeout` or `setInterval` to reset the counter periodically.
// - Logic to delay or drop requests exceeding the limit.
// Due to complexity, a full implementation is omitted here, but the concept
// involves tracking calls over time and preventing excess calls within that time.


// -----------------------------------------------------------------------------
// 8. Worker Threads & Web Workers
// -----------------------------------------------------------------------------