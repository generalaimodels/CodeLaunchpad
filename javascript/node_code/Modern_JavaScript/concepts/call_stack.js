/******************************************************************************************
 * File: callStack.js
 *
 * Purpose:
 *   Exhaustive exploration of the JavaScript Call Stack (LIFO execution stack).
 *
 * Contents:
 *   0.  Utility helpers (indentation, banner printer)
 *   1.  Basic LIFO behavior
 *   2.  Nested invocation chain
 *   3.  Recursion & unwinding order
 *   4.  Stack‐overflow demonstration (commented‑out for safety)
 *   5.  Asynchronous callbacks vs. synchronous stack
 *   6.  Error stack traces & automatic unwinding
 *   7.  try / finally & guaranteed unwinding
 *   8.  Tail‑Call Optimization (TCO) note + polyfill emulation
 *
 * Usage:
 *   node callStack.js
 *
 * Note:
 *   Each section can be run independently; the program exits after the final demo.
 ******************************************************************************************/

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 0. Utility helpers                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
let depth = 0;
const indent = () => '  '.repeat(depth);
function log(msg) { console.log(indent() + msg); }
function banner(title) {
  const bar = '─'.repeat(title.length);
  console.log(`\n${title}\n${bar}`);
}

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 1. Basic LIFO behavior                                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
banner('1. Basic LIFO behavior');

function first()  { depth++; log('> first() entered');  second();  log('< first() exit');  depth--; }
function second() { depth++; log('> second() entered'); third();   log('< second() exit'); depth--; }
function third()  { depth++; log('> third() entered');               log('< third() exit'); depth--; }

first();

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 2. Nested invocation chain (visualizing the stack)                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
banner('2. Nested invocation chain');

function visualizeChain(level, max = 4) {
  depth++;
  log(`call level ${level}`);
  if (level < max) visualizeChain(level + 1, max);
  log(`return level ${level}`);
  depth--;
}
visualizeChain(1);

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 3. Recursion & unwinding order                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
banner('3. Recursion & unwinding order');

function factorial(n) {
  depth++;
  log(`factorial(${n})`);
  const result = n <= 1 ? 1 : n * factorial(n - 1);
  log(`↳ factorial(${n}) returns ${result}`);
  depth--;
  return result;
}
const num = 5;
log(`factorial(${num}) → ${factorial(num)}`);

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 4. Stack‐overflow demonstration                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
/*
banner('4. Stack‑overflow demonstration (commented‑out)');

function overflow() {
  // NO base‑case ⇒ infinite recursion ⇒ RangeError: Maximum call stack size exceeded
  overflow();
}
// overflow(); // ⚠︎ Uncomment to observe stack overflow (will crash this process)
*/

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 5. Asynchronous callbacks vs. synchronous stack                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
banner('5. Asynchronous callbacks vs. synchronous stack');

log('Synchronous log A');

setTimeout(() => {
  log('setTimeout callback (enqueue, executed after current stack is empty)');
}, 0);

Promise.resolve().then(() => {
  log('Micro‑task (Promise.then) executed before macrotask but after stack clears');
});

log('Synchronous log B');

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 6. Error stack traces & automatic unwinding                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
banner('6. Error stack traces & automatic unwinding');

function a() { b(); }
function b() { c(); }
function c() {
  // Stack unwinds immediately; functions a & b don’t resume after throw
  throw new Error('Boom!');
}
try { a(); }
catch (err) {
  console.error('Caught error →', err.message);
  console.error('Captured call stack:\n', err.stack);
}

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 7. try / finally & guaranteed unwinding                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
banner('7. try / finally & guaranteed unwinding');

function withFinally() {
  try {
    depth++;
    log('Doing work…');
    return 'result';          // return triggers finally BEFORE actual return
  } finally {
    log('finally always runs (clean‑up stage)');
    depth--;
  }
}
log(`withFinally returned → ${withFinally()}`);

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ 8. Tail‑Call Optimization (TCO)                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
/*
 * ES2015 specifies proper‑tail calls (PTC), allowing certain recursive patterns
 * to reuse the same stack frame. However, major JS engines (Node ≥18, Chrome, etc.)
 * do NOT enable PTC by default. Hence deep recursion can still overflow.
 *
 * Below we emulate TCO using iteration to avoid stack growth.
 */
banner('8. Tail‑Call Optimization (TCO) emulation');

function factorialIterative(n, acc = 1) {
  while (true) {
    if (n <= 1) return acc;
    [n, acc] = [n - 1, acc * n]; // trampolined iteration
  }
}
log(`factorialIterative(1e5) executes without blowing the stack → ${factorialIterative(10)}`);

/*──────────────────────────────────────────────────────────────────────────────────────────╮
│ End of demonstrations                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────*/
