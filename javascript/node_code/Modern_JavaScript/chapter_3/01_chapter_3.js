// Asynchronous JavaScript Mastery

// 1. Event Loop, Task Queue & Microtasks

// Example 1: setTimeout vs Promise
console.log('A');
setTimeout(() => console.log('B'), 0);
Promise.resolve().then(() => console.log('C'));
console.log('D');
// Output: A D C B

// Example 2: process.nextTick (Node.js) vs Promise
console.log('E');
process.nextTick(() => console.log('F'));
Promise.resolve().then(() => console.log('G'));
console.log('H');
// Output: E H F G

// Example 3: Multiple Promises and setTimeout
setTimeout(() => console.log('I'), 0);
Promise.resolve().then(() => console.log('J'));
Promise.resolve().then(() => {
  console.log('K');
  setTimeout(() => console.log('L'), 0);
});
// Output: J K I L (J and K order may vary, but before I and L)

// Example 4: Microtask inside Macrotask
setTimeout(() => {
  console.log('M');
  Promise.resolve().then(() => console.log('N'));
}, 0);
// Output: M N

// Example 5: Nested setTimeout
setTimeout(() => {
  console.log('O');
  setTimeout(() => console.log('P'), 0);
}, 0);
// Output: O P

// 2. Callbacks vs. Promises

// Example 1: Callback Hell
function fetchDataCallback(cb) {
  setTimeout(() => cb(null, 'data1'), 100);
}
fetchDataCallback((err, data) => {
  if (!err) {
    fetchDataCallback((err2, data2) => {
      if (!err2) {
        console.log('Callback:', data, data2);
      }
    });
  }
});

// Example 2: Promises for Sequential Operations
function fetchDataPromise() {
  return new Promise(res => setTimeout(() => res('data1'), 100));
}
fetchDataPromise().then(data => fetchDataPromise().then(data2 => console.log('Promise:', data, data2)));

// Example 3: Error Handling in Callbacks
function errorCallback(cb) {
  setTimeout(() => cb(new Error('fail')), 100);
}
errorCallback((err, data) => {
  if (err) 
    console.error('Callback Error:', err.message);
});

// Example 4: Error Handling in Promises
function errorPromise() {
  return new Promise((_, rej) => setTimeout(() => rej(new Error('fail')), 100));
}
errorPromise().catch(err => console.error('Promise Error:', err.message));

// Example 5: Callback to Promise Conversion
function callbackToPromise(fn) {
  return new Promise((res, rej) => fn((err, data) => err ? rej(err) : res(data)));
}
callbackToPromise(cb => setTimeout(() => cb(null, 'done'), 100))
  .then(console.log);

// 3. Promise API & Chaining

// Example 1: Promise.all
Promise.all([
  Promise.resolve(1),
  Promise.resolve(2),
  Promise.resolve(3)
]).then(console.log); // [1,2,3]

// Example 2: Promise.race
Promise.race([
  new Promise(res => setTimeout(() => res('first'), 50)),
  new Promise(res => setTimeout(() => res('second'), 100))
]).then(console.log); // 'first'

// Example 3: Promise.allSettled
Promise.allSettled([
  Promise.resolve('ok'),
  Promise.reject('fail')
]).then(console.log); // [{status: 'fulfilled', ...}, {status: 'rejected', ...}]

// Example 4: Promise.finally
Promise.resolve('done').finally(() => console.log('Cleanup')).then(console.log);

// Example 5: Chaining with Error Propagation
Promise.resolve(10).then(x => x * 2).then(x => { throw new Error('fail'); }).catch(err => console.error('Caught:', err.message));

// 4. async / await Internals

// Example 1: Basic async/await
async function asyncFunc1() {
  const res = await Promise.resolve('async1');
  console.log(res);
}
asyncFunc1();

// Example 2: Error Handling
async function asyncFunc2() {
  try {
    await Promise.reject(new Error('fail'));
  } catch (e) {
    console.error('Caught:', e.message);
  }
}
asyncFunc2();

// Example 3: Sequential vs Parallel
async function sequential() {
  const a = await Promise.resolve(1);
  const b = await Promise.resolve(2);
  console.log('Sequential:', a, b);
}
async function parallel() {
  const [a, b] = await Promise.all([Promise.resolve(1), Promise.resolve(2)]);
  console.log('Parallel:', a, b);
}
sequential();
parallel();

// Example 4: Awaiting non-Promise
async function awaitNonPromise() {
  const val = await 42;
  console.log('Awaited:', val);
}
awaitNonPromise();

// Example 5: async returns Promise
async function returnsPromise() {
  return 'value';
}
returnsPromise().then(console.log);

// 5. Generators & Co Routines

// Example 1: Basic Generator
function* gen1() {
  yield 1;
  yield 2;
  return 3;
}
const g1 = gen1();
console.log(g1.next(), g1.next(), g1.next());

// Example 2: Generator for Async Flow (manual)
function* asyncGen() {
  const a = yield Promise.resolve(1);
  const b = yield Promise.resolve(a + 1);
  return b;
}
function runGen(gen) {
  const it = gen();
  function step(val) {
    const { value, done } = it.next(val);
    if (done) return value;
    return value.then(step);
  }
  return step();
}
runGen(asyncGen).then(console.log);

// Example 3: Infinite Generator
function* infinite() {
  let i = 0;
  while (true) yield i++;
}
const inf = infinite();
console.log(inf.next().value, inf.next().value);

// Example 4: Delegating Generators
function* genA() { yield 1; yield 2; }
function* genB() { yield* genA(); yield 3; }
for (const v of genB()) console.log('Delegated:', v);

// Example 5: Generator Exception Handling
function* genErr() {
  try {
    yield 1;
    throw new Error('fail');
  } catch (e) {
    yield e.message;
  }
}
const ge = genErr();
console.log(ge.next(), ge.next(), ge.next());

// 6. Cancellation Patterns & AbortController

// Example 1: AbortController with fetch
const controller1 = new AbortController();
fetch('https://jsonplaceholder.typicode.com/todos/1', { signal: controller1.signal })
  .then(res => res.json())
  .then(console.log)
  .catch(err => console.error('Aborted:', err.name));
controller1.abort();

// Example 2: Manual Cancellation with Flag
function cancellableTimeout(ms) {
  let cancelled = false;
  const promise = new Promise((res, rej) => {
    const id = setTimeout(() => cancelled ? rej('Cancelled') : res('Done'), ms);
    if (cancelled) clearTimeout(id);
  });
  return {
    promise,
    cancel: () => { cancelled = true; }
  };
}
const c1 = cancellableTimeout(100);
c1.cancel();
c1.promise.catch(console.error);

// Example 3: AbortController with EventListener
const controller2 = new AbortController();
document.body.addEventListener('click', () => console.log('clicked'), { signal: controller2.signal });
controller2.abort();

// Example 4: Promise.race for Cancellation
function cancellablePromise(promise, signal) {
  return Promise.race([
    promise,
    new Promise((_, rej) => signal.addEventListener('abort', () => rej('Aborted')))
  ]);
}
const controller3 = new AbortController();
cancellablePromise(new Promise(res => setTimeout(res, 100)), controller3.signal)
  .catch(console.error);
controller3.abort();

// Example 5: Observable-style Cancellation
function observable(subscriber) {
  let active = true;
  setTimeout(() => { if (active) subscriber('data'); }, 100);
  return () => { active = false; };
}
const unsubscribe = observable(console.log);
unsubscribe();

// 7. Throttling, Debouncing & Rate Limiting

// Example 1: Throttle Function
function throttle(fn, wait) {
  let last = 0;
  return function(...args) {
    const now = Date.now();
    if (now - last >= wait) {
      last = now;
      fn.apply(this, args);
    }
  };
}
const throttled = throttle(() => console.log('Throttled!'), 1000);
setInterval(throttled, 200);

// Example 2: Debounce Function
function debounce(fn, delay) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}
const debounced = debounce(() => console.log('Debounced!'), 1000);
for (let i = 0; i < 5; i++) debounced();

// Example 3: Rate Limiter (Token Bucket)
class RateLimiter {
  constructor(limit, interval) {
    this.limit = limit;
    this.tokens = limit;
    setInterval(() => this.tokens = this.limit, interval);
  }
  try() {
    if (this.tokens > 0) {
      this.tokens--;
      return true;
    }
    return false;
  }
}
const limiter = new RateLimiter(2, 1000);
setInterval(() => console.log('Allowed:', limiter.try()), 300);

// Example 4: Leading/Trailing Debounce
function debounceLeading(fn, delay) {
  let timer, called = false;
  return function(...args) {
    if (!called) {
      fn.apply(this, args);
      called = true;
    }
    clearTimeout(timer);
    timer = setTimeout(() => called = false, delay);
  };
}
const debouncedLead = debounceLeading(() => console.log('Leading!'), 1000);
for (let i = 0; i < 5; i++) debouncedLead();

// Example 5: Async Throttle
function asyncThrottle(fn, wait) {
  let last = 0, pending;
  return async function(...args) {
    const now = Date.now();
    if (now - last >= wait) {
      last = now;
      return fn.apply(this, args);
    } else if (!pending) {
      pending = new Promise(res => setTimeout(() => {
        last = Date.now();
        pending = null;
        res(fn.apply(this, args));
      }, wait - (now - last)));
      return pending;
    }
    return pending;
  };
}
const asyncThrottled = asyncThrottle(async () => console.log('Async Throttled!'), 1000);
setInterval(asyncThrottled, 200);

// 8. Worker Threads & Web Workers

// Example 1: Web Worker (browser)
if (typeof Worker !== 'undefined') {
  const worker = new Worker(URL.createObjectURL(new Blob([`
    onmessage = e => postMessage(e.data * 2);
  `])));
  worker.onmessage = e => console.log('Worker:', e.data);
  worker.postMessage(21);
}

// Example 2: Worker Thread (Node.js)
const { Worker: NodeWorker, isMainThread, parentPort } = require('worker_threads');
if (isMainThread) {
  const worker = new NodeWorker(__filename);
  worker.on('message', msg => console.log('Node Worker:', msg));
  worker.postMessage(10);
} else {
  parentPort.on('message', msg => parentPort.postMessage(msg * 3));
}

// Example 3: SharedArrayBuffer (browser/Node.js)
if (typeof SharedArrayBuffer !== 'undefined') {
  const sab = new SharedArrayBuffer(4);
  const arr = new Int32Array(sab);
  arr[0] = 42;
  // Pass sab to worker for shared memory
}

// Example 4: Terminating Worker
if (typeof Worker !== 'undefined') {
  const worker = new Worker(URL.createObjectURL(new Blob([`onmessage = () => {}`])));
  worker.terminate();
}

// Example 5: Transferable Objects
if (typeof Worker !== 'undefined') {
  const worker = new Worker(URL.createObjectURL(new Blob([`
    onmessage = e => postMessage(e.data, [e.data.buffer]);
  `])));
  const ab = new ArrayBuffer(8);
  worker.postMessage(ab, [ab]);
}