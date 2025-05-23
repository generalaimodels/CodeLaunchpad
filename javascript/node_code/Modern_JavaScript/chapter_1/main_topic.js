// 1. Promises vs Async/Await

// Example 1: Basic Promise
function fetchDataPromise() {
    return new Promise((resolve, reject) => {
        setTimeout(() => resolve("Data fetched"), 100);
    });
}
fetchDataPromise().then(console.log);

// Example 2: Async/Await equivalent
async function fetchDataAsync() {
    const data = await fetchDataPromise();
    console.log(data);
}
fetchDataAsync();

// Example 3: Promise with error handling
function fetchWithErrorPromise() {
    return new Promise((_, reject) => {
        setTimeout(() => reject(new Error("Failed")), 100);
    });
}
fetchWithErrorPromise().catch(err => console.error(err.message));

// Example 4: Async/Await with try/catch
async function fetchWithErrorAsync() {
    try {
        await fetchWithErrorPromise();
    } catch (err) {
        console.error(err.message);
    }
}
fetchWithErrorAsync();

// Example 5: Chaining Promises vs Sequential Async/Await
function step1() { return Promise.resolve(1); }
function step2(val) { return Promise.resolve(val + 1); }
function step3(val) { return Promise.resolve(val + 1); }

// Promise chaining
step1().then(step2).then(step3).then(console.log);

// Async/Await sequential
async function runSteps() {
    let val = await step1();
    val = await step2(val);
    val = await step3(val);
    console.log(val);
}
runSteps();


// 2. Higher-Order Functions

// Example 1: Function as argument
function applyOperation(arr, op) {
    return arr.map(op);
}
console.log(applyOperation([1,2,3], x => x * 2));

// Example 2: Function returning function
function multiplier(factor) {
    return x => x * factor;
}
const double = multiplier(2);
console.log(double(5));

// Example 3: Array filter with custom predicate
function filterArray(arr, predicate) {
    return arr.filter(predicate);
}
console.log(filterArray([1,2,3,4], x => x % 2 === 0));

// Example 4: Custom forEach
function forEach(arr, fn) {
    for (let i = 0; i < arr.length; i++) fn(arr[i], i, arr);
}
forEach([1,2,3], x => console.log(x));

// Example 5: Compose two functions
function compose(f, g) {
    return function(x) {
        return f(g(x));
    };
}
const add1 = x => x + 1;
const square = x => x * x;
const add1ThenSquare = compose(square, add1);
console.log(add1ThenSquare(3));


// 3. Closures and their Applications

// Example 1: Private variable
function counter() {
    let count = 0;
    return () => ++count;
}
const inc = counter();
console.log(inc(), inc());

// Example 2: Function factory
function greeter(greeting) {
    return function(name) {
        return `${greeting}, ${name}`;
    };
}
const hello = greeter("Hello");
console.log(hello("World"));

// Example 3: Partial application
function partial(fn, ...args) {
    return (...rest) => fn(...args, ...rest);
}
function sum(a, b, c) { return a + b + c; }
const add5 = partial(sum, 2, 3);
console.log(add5(4));

// Example 4: Event handler with closure
function makeHandler(msg) {
    return function() {
        console.log(msg);
    };
}
document.body.addEventListener('click', makeHandler("Clicked!"));

// Example 5: Loop with closure
let funcs = [];
for (let i = 0; i < 3; i++) {
    funcs.push(() => console.log(i));
}
funcs.forEach(fn => fn());


// 4. Memoization Techniques

// Example 1: Simple memoization
function memoize(fn) {
    const cache = {};
    return function(...args) {
        const key = JSON.stringify(args);
        if (cache[key]) return cache[key];
        return cache[key] = fn(...args);
    };
}
const fib = memoize(n => n < 2 ? n : fib(n-1) + fib(n-2));
console.log(fib(10));

// Example 2: Memoize with Map
function memoizeMap(fn) {
    const cache = new Map();
    return function(...args) {
        const key = args.toString();
        if (cache.has(key)) return cache.get(key);
        const val = fn(...args);
        cache.set(key, val);
        return val;
    };
}
const squareMemo = memoizeMap(x => x * x);
console.log(squareMemo(5));

// Example 3: Memoize async function
function memoizeAsync(fn) {
    const cache = {};
    return async function(...args) {
        const key = JSON.stringify(args);
        if (cache[key]) return cache[key];
        return cache[key] = await fn(...args);
    };
}
const fetchMemo = memoizeAsync(async x => x * 2);
fetchMemo(3).then(console.log);

// Example 4: Memoize with WeakMap for object keys
function memoizeObj(fn) {
    const cache = new WeakMap();
    return function(obj) {
        if (cache.has(obj)) return cache.get(obj);
        const val = fn(obj);
        cache.set(obj, val);
        return val;
    };
}
const getId = memoizeObj(obj => obj.id);
console.log(getId({id: 1}));

// Example 5: Memoize with limited cache size (LRU)
function memoizeLRU(fn, limit = 3) {
    const cache = new Map();
    return function(...args) {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            const val = cache.get(key);
            cache.delete(key);
            cache.set(key, val);
            return val;
        }
        const val = fn(...args);
        cache.set(key, val);
        if (cache.size > limit) cache.delete(cache.keys().next().value);
        return val;
    };
}
const add = memoizeLRU((a, b) => a + b);
console.log(add(1,2), add(2,3), add(3,4), add(1,2)); // LRU in action


// 5. Currying in JavaScript

// Example 1: Basic currying
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) return fn(...args);
        return (...next) => curried(...args, ...next);
    };
}
function sum3(a, b, c) { return a + b + c; }
const curriedSum = curry(sum3);
console.log(curriedSum(1)(2)(3));

// Example 2: Currying with arrow functions
const multiply = a => b => c => a * b * c;
console.log(multiply(2)(3)(4));

// Example 3: Currying for partial application
const curriedAdd = curry((a, b, c) => a + b + c);
const add2 = curriedAdd(2);
console.log(add2(3,4));

// Example 4: Currying for event handler
function handleEvent(type) {
    return function(selector) {
        return function(callback) {
            document.querySelector(selector).addEventListener(type, callback);
        };
    };
}
// handleEvent('click')('#btn')(e => console.log('Clicked', e));

// Example 5: Infinite currying
function infiniteCurry(a) {
    return function(b) {
        if (b) return infiniteCurry(a + b);
        return a;
    };
}
console.log(infiniteCurry(1)(2)(3)());


// 6. Event Loop and its Mechanics

// Example 1: setTimeout vs Promise
console.log('A');
setTimeout(() => console.log('B'), 0);
Promise.resolve().then(() => console.log('C'));
console.log('D');

// Example 2: Microtasks vs Macrotasks
setTimeout(() => console.log('setTimeout'), 0);
Promise.resolve().then(() => console.log('promise'));
queueMicrotask(() => console.log('microtask'));

// Example 3: Blocking the event loop
// Uncomment to see blocking
// while(true) {} // Blocks everything

// Example 4: process.nextTick (Node.js only)
// process.nextTick(() => console.log('nextTick'));

// Example 5: Order of execution
console.log(1);
setTimeout(() => console.log(2), 0);
Promise.resolve().then(() => console.log(3));
console.log(4);


// 7. Prototype Chain and Inheritance

// Example 1: Basic prototype chain
function Animal(name) { this.name = name; }
Animal.prototype.speak = function() { return `${this.name} makes a noise.`; };
const dog = new Animal('Dog');
console.log(dog.speak());

// Example 2: Inheritance with constructor functions
function Dog(name) { Animal.call(this, name); }
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;
Dog.prototype.bark = function() { return `${this.name} barks.`; };
const d = new Dog('Rex');
console.log(d.speak(), d.bark());

// Example 3: ES6 class inheritance
class Person {
    constructor(name) { this.name = name; }
    greet() { return `Hello, ${this.name}`; }
}
class Student extends Person {
    study() { return `${this.name} is studying.`; }
}
const s = new Student('Alice');
console.log(s.greet(), s.study());

// Example 4: Object.create for inheritance
const proto = { greet() { return 'Hi'; } };
const obj = Object.create(proto);
console.log(obj.greet());

// Example 5: Prototype chain lookup
const a = { x: 1 };
const b = Object.create(a);
b.y = 2;
console.log(b.x, b.y);


// 8. Function Composition with pipe and compose

// Example 1: Compose (right-to-left)
function compose(...fns) {
    return x => fns.reduceRight((v, f) => f(v), x);
}
const add21 = x => x + 2;
const mul3 = x => x * 3;
const composed = compose(mul3, add21);
console.log(composed(4)); // (4+2)*3

// Example 2: Pipe (left-to-right)
function pipe(...fns) {
    return x => fns.reduce((v, f) => f(v), x);
}
const piped = pipe(add2, mul3);
console.log(piped(4)); // (4+2)*3

// Example 3: Compose with multiple functions
const f1 = x => x + 1;
const f2 = x => x * 2;
const f3 = x => x - 3;
const composed2 = compose(f3, f2, f1);
console.log(composed2(5));

// Example 4: Pipe with async functions
async function asyncAdd(x) { return x + 1; }
async function asyncMul(x) { return x * 2; }
function asyncPipe(...fns) {
    return x => fns.reduce((p, f) => p.then(f), Promise.resolve(x));
}
asyncPipe(asyncAdd, asyncMul)(3).then(console.log);

// Example 5: Compose with error handling
function safeCompose(...fns) {
    return x => fns.reduceRight((v, f) => {
        try { return f(v); } catch { return v; }
    }, x);
}
const errorFn = x => { throw new Error(); };
console.log(safeCompose(f1, errorFn, f2)(2));


// 9. Lazy Evaluation and Generators

// Example 1: Basic generator
function* genNumbers() {
    yield 1;
    yield 2;
    yield 3;
}
const g = genNumbers();
console.log(g.next().value, g.next().value, g.next().value);

// Example 2: Infinite sequence
function* infiniteSeq() {
    let i = 0;
    while (true) yield i++;
}
const inf = infiniteSeq();
console.log(inf.next().value, inf.next().value, inf.next().value);

// Example 3: Lazy map
function* lazyMap(iter, fn) {
    for (const val of iter) yield fn(val);
}
const arr = [1,2,3];
const mapped = lazyMap(arr, x => x * 2);
console.log([...mapped]);

// Example 4: Lazy filter
function* lazyFilter(iter, pred) {
    for (const val of iter) if (pred(val)) yield val;
}
const filtered = lazyFilter(arr, x => x % 2 === 1);
console.log([...filtered]);

// Example 5: Composing lazy operations
function* lazyTake(iter, n) {
    let i = 0;
    for (const val of iter) {
        if (i++ >= n) break;
        yield val;
    }
}
const composedLazy = lazyTake(lazyMap(infiniteSeq(), x => x * 3), 5);
console.log([...composedLazy]);


// 10. Web Workers and Multithreading

// Example 1: Basic Web Worker (main.js)
// const worker = new Worker('worker.js');
// worker.postMessage('Hello');
// worker.onmessage = e => console.log(e.data);

// worker.js
// onmessage = e => postMessage('Received: ' + e.data);

// Example 2: Inline worker (Blob)
const code = `
    onmessage = function(e) {
        postMessage(e.data * 2);
    }
`;
const blob = new Blob([code], { type: 'application/javascript' });
const worker = new Worker(URL.createObjectURL(blob));
worker.onmessage = e => console.log('Worker result:', e.data);
worker.postMessage(21);

// Example 3: Transferable objects
const ab = new ArrayBuffer(8);
const worker2 = new Worker(URL.createObjectURL(new Blob([`
    onmessage = function(e) {
        postMessage(e.data, [e.data]);
    }
`], { type: 'application/javascript' })));
worker2.postMessage(ab, [ab]);

// Example 4: SharedWorker (shared.js)
// const shared = new SharedWorker('shared.js');
// shared.port.onmessage = e => console.log(e.data);
// shared.port.postMessage('Ping');

// shared.js
// onconnect = function(e) {
//     const port = e.ports[0];
//     port.onmessage = function(e) { port.postMessage('Pong'); };
// };

// Example 5: OffscreenCanvas in worker
// main.js
// const worker = new Worker('canvasWorker.js');
// const canvas = document.createElement('canvas');
// const offscreen = canvas.transferControlToOffscreen();
// worker.postMessage({canvas: offscreen}, [offscreen]);

// canvasWorker.js
// onmessage = function(e) {
//     const ctx = e.data.canvas.getContext('2d');
//     ctx.fillRect(0, 0, 100, 100);
// };