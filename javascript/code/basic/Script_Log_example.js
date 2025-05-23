/**
 * ============================================================
 *  Mastering Console Methods in JavaScript (Node.js & Browser)
 *  ------------------------------------------------------------
 *  This file demonstrates the usage of all major console methods,
 *  with professional code style, detailed explanations, and
 *  multiple real-world examples for each method.
 * ============================================================
 */

// 1. console.log()
//    - Used for general output of information.
console.log("Example 1: Logging a simple message.");
console.log("User %s has %d points.", "Alice", 1500);

console.log("\n");

// 2. console.warn()
//    - Used to output warnings.
console.warn("Example 1: This is a warning message.");
console.warn("Low disk space: only %dMB left.", 200);

console.log("\n");

// 3. console.dir()
//    - Displays an interactive list of the properties of a specified JavaScript object.
const user = { name: "Bob", age: 30, skills: ["JS", "Python"] };
console.dir(user, { depth: null });
// console.dir(document.body, { showHidden: true }); // In browser

console.log("\n");

// 4. console.time() / console.timeEnd() / console.timeLog()
//    - Used to measure the time taken by a block of code.
console.time("process");
for (let i = 0; i < 1e6; i++) {} // Simulate workload
console.timeLog("process", "Halfway done...");
for (let i = 0; i < 1e6; i++) {}
console.timeEnd("process");

console.time("fetchData");
setTimeout(() => {
    console.timeEnd("fetchData");
}, 500);

console.log("\n");

// 5. console.trace()
//    - Prints a stack trace to the console.
function a() { b(); }
function b() { c(); }
function c() { console.trace("Trace Example 1:"); }
a();

function foo() { bar(); }
function bar() { console.trace("Trace Example 2:"); }
foo();

console.log("\n");

// 6. console.assert()
//    - Writes an error message if the assertion is false.
console.assert(1 === 2, "Example 1: 1 is not equal to 2");
console.assert(Array.isArray([]), "Example 2: This will not log because assertion is true");

console.log("\n");

// 7. console.clear()
//    - Clears the console (works in browser, limited in Node.js).
// console.clear(); // Uncomment to clear console

console.log("After clear (if supported).");
console.log("\n");

// 8. console.count() / console.countReset()
//    - Logs the number of times that this particular call to count() has been called.
console.count("apples");
console.count("apples");
console.count("oranges");
console.countReset("apples");
console.count("apples");

console.count("login");
console.count("login");
console.countReset("login");
console.count("login");

console.log("\n");

// 9. console.group() / console.groupEnd() / console.groupCollapsed()
//    - Groups together console messages.
console.group("User Details");
console.log("Name: Alice");
console.log("Age: 25");
console.groupEnd();

console.groupCollapsed("Collapsed Group Example");
console.log("This is inside a collapsed group.");
console.groupEnd();

console.log("\n");

// 10. console.table()
//     - Displays tabular data as a table.
const people = [
    { name: "Alice", age: 25 },
    { name: "Bob", age: 30 }
];
console.table(people);

const scores = { Alice: 90, Bob: 85, Carol: 95 };
console.table(scores);

console.log("\n");

// 11. console.debug()
//     - Outputs a message at the "debug" log level.
console.debug("Debug Example 1: This is a debug message.");
console.debug("Debug Example 2: Variable x =", 42);

console.log("\n");

// 12. console.info()
//     - Informational messages.
console.info("Info Example 1: Server started on port 3000.");
console.info("Info Example 2: User logged in.");

console.log("\n");

// 13. console.dirxml()
//     - Displays an XML/HTML element representation (browser only).
// console.dirxml(document.body); // Uncomment in browser
// console.dirxml(document.querySelectorAll("div")); // Uncomment in browser

console.log("\n");

// 14. console.error()
//     - Outputs an error message.
console.error("Error Example 1: Something went wrong!");
console.error(new Error("Error Example 2: Custom error object."));

console.log("\n");

// 15. console._stdoutErrorHandler, console._stderrErrorHandler, console._ignoreErrors
//     - Internal/private methods, not for public use. Demonstration for educational purposes only.
if (console._stdoutErrorHandler) {
    console._stdoutErrorHandler(new Error("Simulated stdout error"));
}
if (console._stderrErrorHandler) {
    console._stderrErrorHandler(new Error("Simulated stderr error"));
}
// if (console._ignoreErrors) {
//     console._ignoreErrors(() => { throw new Error("Ignored error"); });
// }

console.log("\n");

// 16. console._times
//     - Internal object for tracking timers. Not for direct use, but can be inspected.
if (console._times) {
    console.log("Current timers:", console._times);
}

console.log("\n");

// 17. console.Console
//     - Custom Console instance (Node.js only).
const { Console } = require('console');
const fs = require('fs');
const output = fs.createWriteStream('./stdout.log');
const errorOutput = fs.createWriteStream('./stderr.log');
const myConsole = new Console({ stdout: output, stderr: errorOutput });
myConsole.log("This will be written to stdout.log");
myConsole.error("This will be written to stderr.log");

const memoryConsole = new Console(process.stdout, process.stderr);
memoryConsole.log("Logging to process stdout");
memoryConsole.error("Logging to process stderr");

console.log("\n");

// 18. console.profile() / console.profileEnd()
//     - Starts and ends a CPU profile (browser only).
// console.profile("Profile Example 1");
// for (let i = 0; i < 1e6; i++) {}
// console.profileEnd("Profile Example 1");

// console.profile("Profile Example 2");
// setTimeout(() => {
//     console.profileEnd("Profile Example 2");
// }, 1000);

console.log("\n");

// 19. console.timeStamp()
//     - Adds a timestamp to the browser's performance timeline (browser only).
// console.timeStamp("TimeStamp Example 1");
// setTimeout(() => {
//     console.timeStamp("TimeStamp Example 2");
// }, 500);

console.log("\n");

// 20. console.context()
//     - Used in some browsers to switch the context of the console (not standard).
// console.context(document); // Uncomment in supported browsers

console.log("\n");

// 21. console.createTask()
//     - Non-standard, used in some environments for async task tracking.
// if (console.createTask) {
//     const task = console.createTask("Async Task Example");
//     task.run(() => {
//         console.log("Task running...");
//     });
// }

console.log("\n");

// 22. console._stdout, console._stderr
//     - Internal streams for stdout and stderr (Node.js).
if (console._stdout) {
    console._stdout.write("Direct write to stdout\n");
}
if (console._stderr) {
    console._stderr.write("Direct write to stderr\n");
}

/**
 * ============================================================
 *  End of Console Methods Mastery File
 *  ------------------------------------------------------------
 *  This file is a comprehensive reference for all console methods,
 *  with real-world examples and professional code style.
 * ============================================================
 */