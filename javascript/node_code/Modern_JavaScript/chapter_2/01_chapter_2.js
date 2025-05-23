/***************************************************************************************************
*  Chapter 2 | Deep Dive into Functions & Scope
*  A single self-contained JavaScript file covering:
*     1. Execution Context & Hoisting
*     2. Call Stack & Stack Frames
*     3. Function Invocation Patterns  (method, constructor, direct call, call / apply / bind)
*     4. Closures & Lexical Scope
*     5. IIFE, Strict Mode, Tail-Call Optimization
*     6. Currying & Partial Application
*  Each topic contains a brief technical discussion followed by 5 distinct examples illustrating
*  edge-cases & real-world scenarios.  Run in Node ≥14 or any modern browser console.
***************************************************************************************************/



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| 1. EXECUTION CONTEXT & HOISTING
|   • At runtime, JavaScript creates an Execution Context (EC) for: global code, every function call,
|     and eval.  Each EC has:
|       – Variable Environment (VE)   – contains var/let/const bindings.
|       – LexicalEnvironment (LE)     – scope chain reference.
|       – ThisBinding                 – value of `this`.
|   • Hoisting: during Creation Phase, declarations are moved to the top of their scope; initial-
|     isation differs between var (undefined), let/const (TDZ), and function declarations (whole
|     body hoisted).  Function expressions obey variable semantics.
|
|   NOTE: Examples intentionally produce console output AND errors (caught) to clarify behaviour.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/* Example 1 — var vs let hoisting */
console.log('E1-1:', typeof fooVar);         // undefined (var hoisted, initialised)
try { console.log(fooLet); } catch (e) {     // ReferenceError (TDZ)
  console.log('E1-1:', e.message);
}
var  fooVar = 42;
let  fooLet = 99;

/* Example 2 — function declaration vs expression */
hoisted();                                    // Works: full body hoisted
function hoisted() { console.log('E1-2: ok'); }

try { notHoisted(); } catch (e) {             // TypeError: not a function
  console.log('E1-2:', e.message);
}
var notHoisted = function () {};

/* Example 3 — overlapping names (var + function) */
console.log('E1-3:', typeof mix);             // function (function wins in hoisting priority)
var mix = 1;                                  // Reassigns after creation phase
function mix() {}

/* Example 4 — block scope with let/const */
{
  console.log('E1-4 block start');
  let x = 'block';
  {
    console.log('E1-4 inner sees:', x);       // lexical capture
  }
}
try { console.log(x); } catch (e) {           // ReferenceError
  console.log('E1-4:', e.message);
}

/* Example 5 — implicit global (sloppy mode) */
function implicitGlobal() {
  y = 'leaked';                               // NO var/let/const → becomes global (non-strict)
}
implicitGlobal();
console.log('E1-5:', globalThis.y);           // 'leaked'
delete globalThis.y;                          // clean-up



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| 2. CALL STACK & STACK FRAMES
|   • Each function call pushes a Stack Frame (activation object) containing arguments, locals,
|     return address, and EC metadata.  Recursion depth = stack depth.
|   • JavaScript engines optimise tail-calls (in strict mode & compliant environments).
|   • Understanding stack traces aids in debugging and preventing stack overflows.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

function tracer(tag) {
  console.log('Stack:', tag, '\n', new Error().stack.split('\n').slice(2).join('\n'));
}

/* Example 1 — simple nested calls */
function A() { tracer('E2-1:A'); }
function B() { A(); }
function C() { B(); }
C();

/* Example 2 — recursion depth vs stack overflow */
function recurse(n) {
  if (n === 0) return 'done';
  return recurse(n - 1);
}
try { recurse(1e5); } catch (e) {
  console.log('E2-2:', e.name);              // RangeError: Maximum call stack size exceeded
}

/* Example 3 — asynchronous calls produce NEW call stacks */
function asyncDemo() {
  setTimeout(() => tracer('E2-3:async callback'), 0);
}
asyncDemo();

/* Example 4 — error stack trace inspection */
function level1() { level2(); }
function level2() { level3(); }
function level3() { throw new Error('E2-4 custom'); }
try { level1(); } catch (e) {
  console.log('E2-4 Trace:\n', e.stack);
}

/* Example 5 — manual stack unwinding with try/finally */
function safeInvoker(fn) {
  try   { fn(); }
  finally { console.log('E2-5: stack unwound properly'); }
}
safeInvoker(() => console.log('E2-5: inside fn'));



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| 3. FUNCTION INVOCATION PATTERNS
|   A. Direct Call             f()
|   B. Method Call             obj.f()
|   C. Constructor Call        new F()
|   D. call / apply / bind     f.call(obj, ...); f.apply(obj, args); const g = f.bind(obj, ...)
|   `this` binding rules vary accordingly.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/* Shared helper to log this value */
function showThis(label) { console.log(label, '→', this); }

/*------------------------------- 3A: Direct Call -----------------------------------------------*/
function direct() { showThis('E3-A direct'); }
direct();                                     // this → global (undefined in strict)

/*------------------------------- 3B: Method Call ------------------------------------------------*/
const obj = { id: 'Obj#1', method: showThis };
obj.method('E3-B method');                    // this → obj

/*------------------------------- 3C: Constructor Call -------------------------------------------*/
function Person(name) { this.name = name; }
const alice = new Person('Alice');
console.log('E3-C constructor:', alice.name); // 'Alice'

/*------------------------------- 3D: call / apply / bind ----------------------------------------*/
function greet(greeting, punctuation) {
  console.log(`${greeting}, ${this.name}${punctuation}`);
}
const bob = { name: 'Bob' };

/* Example 1 — call */
greet.call(bob, 'Hi', '!');                   // E3-D-1

/* Example 2 — apply */
greet.apply(bob, ['Hello', '!!']);            // E3-D-2

/* Example 3 — bind (permanent) */
const heyBob = greet.bind(bob, 'Hey');
heyBob('?');                                  // E3-D-3

/* Example 4 — losing context then restoring */
const lost = obj.method;
lost('E3-D-4 lost');                          // this → global
lost.call(obj, 'E3-D-4 restored');            // fixed

/* Example 5 — constructor WITH bind (new beats bind) */
const BoundPerson = Person.bind({ custom: true }, 'IgnoredName');
const charlie = new BoundPerson('Charlie');
console.log('E3-D-5:', charlie.name);         // 'Charlie', not 'IgnoredName'



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| 4. CLOSURES & LEXICAL SCOPE
|   • Closure: function that captures variables from its defining (lexical) scope even after that
|     scope has finished executing.  Enables data-encapsulation, factories, and more.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/* Example 1 — counter factory */
function makeCounter() {
  let count = 0;
  return () => ++count;
}
const c1 = makeCounter(), c2 = makeCounter();
console.log('E4-1:', c1(), c1(), c2());       // 1 2 1

/* Example 2 — private state in class-like pattern */
function Stack() {
  const items = [];
  return {
    push(x) { items.push(x); },
    pop()  { return items.pop(); },
    get size() { return items.length; }
  };
}
const s = Stack();
s.push(10); s.push(20);
console.log('E4-2 size:', s.size, 'pop:', s.pop());

/* Example 3 — loop with let vs var */
const funcsVar = [], funcsLet = [];
for (var i = 0; i < 3; i++) funcsVar.push(() => i);
for (let j = 0; j < 3; j++) funcsLet.push(() => j);
console.log('E4-3 var:', funcsVar.map(f => f()));  // 3,3,3
console.log('E4-3 let:', funcsLet.map(f => f()));  // 0,1,2

/* Example 4 — memoization */
function memoize(fn) {
  const cache = new Map();
  return function (arg) {
    if (cache.has(arg)) return cache.get(arg);
    const result = fn(arg);
    cache.set(arg, result);
    return result;
  };
}
const fib = memoize(n => n < 2 ? n : fib(n-1) + fib(n-2));
console.log('E4-4 fib(20):', fib(20));

/* Example 5 — event handler capturing DOM element (browser) */
function highlight(id) {
  const el = { id };                          // stand-in for document.getElementById(id)
  return () => console.log(`E4-5 highlight #${el.id}`);
}
const clickHandler = highlight('btn-save');
clickHandler();

// E4-5 highlight #btn-save

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| 5. IIFE, STRICT MODE, TAIL CALL OPTIMIZATION
|   • IIFE (Immediately Invoked Function Expression) isolates scope.
|   • 'use strict' enables stricter parsing: silent errors thrown, eliminates `this` coercion,
|     forbids implicit globals, etc.
|   • Proper Tail Calls (PTC) reuse stack frame; only applicable in strict mode & engines supporting
|     ES2015 tail-call (Safari).  Example shows conceptual benefit even if not optimised.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/* Example 1 — Classic IIFE */
(function () {
  const secret = 123;
  console.log('E5-1 IIFE secret:', secret);  
})(); // ==> E5-1 IIFE secret: 123
// console.log(secret); // ReferenceError

/* Example 2 — Module pattern using IIFE */
const Logger = (function () {
  const logs = [];
  return {
    log: msg => logs.push(msg),
    dump: () => [...logs]
  };
})();
Logger.log('boot');
console.log('E5-2:', Logger.dump());

/* Example 3 — Strict mode impact on this */
(function () {
  'use strict';
  function strictThis() { console.log('E5-3 this:', this); }
  strictThis();                              // undefined, not global
})();

/* Example 4 — Prevent implicit globals */
(function () {
  'use strict';
  try { undeclared = 5; } catch (e) {
    console.log('E5-4:', e.message);         // undeclared is not defined
  }
})();

/* Example 5 — Tail-recursive factorial (conceptual) */
(function () {
  'use strict';
  function factTR(n, acc = 1) {
    if (n === 0) return acc;
    return factTR(n-1, n * acc);             // Tail position
  }
  console.log('E5-5 factorial 5:', factTR(5));
})();



/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| 6. CURRYING & PARTIAL APPLICATION
|   • Currying: transform f(a,b,c) → f(a)(b)(c) — returns unary chain.
|   • Partial Application: fix a subset of arguments now, return function that accepts rest.
|   • Benefits: configurability, higher-order utilities, point-free style.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

/* Example 1 — generic curry helper (ES2020) */
function curry(fn, arity = fn.length) {
  return function curried(...args) {
    return args.length >= arity
      ? fn.apply(this, args)
      : (...more) => curried.apply(this, args.concat(more));
  };
}
function volume(l, w, h) { return l * w * h; }
const curriedVolume = curry(volume);
console.log('E6-1:', curriedVolume(2)(3)(4)); // 24

/* Example 2 — partial application via bind */
function compute(a, b, c) { console.log(a,b,c); return a + b * c; }
const add5Times = compute.bind(null, 5);      // fixes `a`
console.log('E6-2:', add5Times(2, 10));       // 25

/* Example 3 — lodash style _.partial implementation */
function partial(fn, ...preset) {
  return (...later) => fn(...preset, ...later);
}
const greetHi = partial(greet, 'Hi');         // reuses greet from earlier
greetHi.call({ name: 'Dana' }, '!');          // E6-3

/* Example 4 — Currying for validation pipeline */
const isBetween = curry((min, max, x) => x >= min && x <= max);
const isAdultAge = isBetween(18)(65);
console.log('E6-4 age 30 adult?', isAdultAge(30));

/* Example 5 — Partial for logging */
function log(level, timestamp, message) {
  console.log(`[${level}] ${timestamp}: ${message}`);
}
const infoLog = partial(log, 'INFO', new Date().toISOString());
infoLog('Server started');                    // E6-5



/***************************************************************************************************
* End of Chapter 2 single-file deep dive.  All topics illustrated with 5 scenario-rich examples.
***************************************************************************************************/