// ==========================
// Web APIs / Node APIs in JavaScript
// ==========================

// JavaScript engines (like V8) are embedded in environments such as browsers and Node.js.
// These environments provide APIs for asynchronous operations, which are not part of the JavaScript language itself.
// When such APIs are called, the operation is delegated to the environment, freeing the call stack for other tasks.

// --------------------------
// 1. Web APIs in Browsers
// --------------------------

// Common Web APIs include:
// - setTimeout / setInterval
// - DOM Events (addEventListener)
// - fetch / XMLHttpRequest
// - Geolocation
// - WebSockets
// - Web Storage (localStorage, sessionStorage)
// - Canvas, Audio, Video, etc.

// Example: setTimeout (Timer API)
console.log('Web API: setTimeout start');
setTimeout(() => {
    console.log('Web API: setTimeout callback');
}, 100);
console.log('Web API: setTimeout end');

// Example: fetch (Network API)
console.log('Web API: fetch start');
fetch('https://jsonplaceholder.typicode.com/todos/1')
    .then(response => response.json())
    .then(data => {
        console.log('Web API: fetch data', data);
    })
    .catch(error => {
        console.log('Web API: fetch error', error);
    });
console.log('Web API: fetch end');

// Example: addEventListener (Event API)
// (Uncomment and run in browser)
/*
document.body.addEventListener('click', function onClick(event) {
    console.log('Web API: DOM event - body clicked', event);
});
*/

// --------------------------
// 2. Node.js APIs
// --------------------------

// Node.js provides its own set of asynchronous APIs, such as:
// - fs (File System): fs.readFile, fs.writeFile
// - net, http, https (Networking)
// - timers: setTimeout, setImmediate, process.nextTick
// - child_process
// - crypto

// Example: fs.readFile (File System API)
const fs = require ? require('fs') : null; // For Node.js environments

if (fs) {
    fs.readFile(__filename, 'utf8', (err, data) => {
        if (err) {
            console.log('Node API: fs.readFile error', err);
        } else {
            console.log('Node API: fs.readFile success, file length:', data.length);
        }
    });
}

// Example: setImmediate (Timer API, Node.js only)
if (typeof setImmediate === 'function') {
    setImmediate(() => {
        console.log('Node API: setImmediate callback');
    });
}

// Example: process.nextTick (Microtask API, Node.js only)
if (typeof process !== 'undefined' && typeof process.nextTick === 'function') {
    process.nextTick(() => {
        console.log('Node API: process.nextTick callback');
    });
}

// --------------------------
// 3. How Asynchronous APIs Work
// --------------------------

// When an async API is called:
// 1. The call stack invokes the API function.
// 2. The environment (browser/Node) takes over the operation (e.g., timer, I/O, network).
// 3. The call stack is cleared for other code.
// 4. When the operation completes, the environment schedules the callback (macro/microtask queue).
// 5. The event loop pushes the callback onto the call stack when it's free.

// Example: setTimeout vs. synchronous code
console.log('Before setTimeout');
setTimeout(() => {
    console.log('Inside setTimeout callback');
}, 0);
console.log('After setTimeout');

// Output order:
// Before setTimeout
// After setTimeout
// Inside setTimeout callback

// --------------------------
// 4. Exceptions and Edge Cases
// --------------------------

// a) setTimeout minimum delay
setTimeout(() => {
    console.log('setTimeout with 0ms delay (minimum enforced by environment)');
}, 0);

// Browsers enforce a minimum delay (often 4ms) for nested timeouts after 5+ nested calls.

// b) fetch is not available in Node.js < v18 without a polyfill
if (typeof fetch === 'undefined') {
    console.log('fetch is not available in this environment');
}

// c) fs.readFile is asynchronous; fs.readFileSync is synchronous and blocks the event loop
if (fs) {
    try {
        const data = fs.readFileSync(__filename, 'utf8');
        console.log('Node API: fs.readFileSync success, file length:', data.length);
    } catch (err) {
        console.log('Node API: fs.readFileSync error', err);
    }
}

// d) Uncaught exceptions in async callbacks are not caught by surrounding try/catch
try {
    setTimeout(() => {
        throw new Error('Async error in setTimeout');
    }, 10);
} catch (e) {
    // This will NOT catch the error above
    console.log('Caught error:', e.message);
}

// To handle async errors, use error-first callbacks or .catch for Promises

// --------------------------
// 5. Multiple Asynchronous APIs and Order of Execution
// --------------------------

console.log('Order: start');

setTimeout(() => {
    console.log('Order: setTimeout');
}, 0);

if (typeof Promise !== 'undefined') {
    Promise.resolve().then(() => {
        console.log('Order: Promise microtask');
    });
}

if (typeof process !== 'undefined' && typeof process.nextTick === 'function') {
    process.nextTick(() => {
        console.log('Order: process.nextTick microtask');
    });
}

if (typeof setImmediate === 'function') {
    setImmediate(() => {
        console.log('Order: setImmediate');
    });
}

console.log('Order: end');

// Typical output in Node.js:
// Order: start
// Order: end
// Order: process.nextTick microtask
// Order: Promise microtask
// Order: setTimeout
// Order: setImmediate

// In browsers, process.nextTick and setImmediate are not available.

// --------------------------
// 6. Custom Asynchronous API Example
// --------------------------

// Simulate an async API using setTimeout
function customAsyncOperation(data, callback) {
    setTimeout(() => {
        callback(null, `Processed: ${data}`);
    }, 50);
}

customAsyncOperation('input', (err, result) => {
    if (err) {
        console.log('Custom API error:', err);
    } else {
        console.log('Custom API result:', result);
    }
});

// --------------------------
// 7. Summary of Web/Node APIs
// --------------------------

// - Web APIs and Node APIs provide asynchronous capabilities to JavaScript.
// - These APIs are not part of the JS language, but are provided by the environment.
// - Asynchronous operations are handled outside the call stack, enabling non-blocking code.
// - Callbacks, Promises, and async/await are used to handle results from these APIs.
// - Error handling in async code requires special care (error-first callbacks, .catch, try/catch in async functions).
// - Understanding the interaction between the call stack, event loop, and environment APIs is essential for robust, performant JavaScript.