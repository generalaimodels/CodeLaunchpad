// Chapter 3 | Asynchronous JavaScript Mastery

// 1. Event Loop, Task Queue & Microtasks

// The event loop is the mechanism that allows JavaScript to perform non-blocking operations
// by offloading operations to the system kernel whenever possible.
// JavaScript has a single-threaded event loop, but can handle concurrency via the event loop, task queue, and microtasks.

// Example: Event Loop, Task Queue, Microtasks
console.log('1. Start');

setTimeout(() => {
  console.log('4. setTimeout (macrotask)');
}, 0);

Promise.resolve().then(() => {
  console.log('3. Promise.then (microtask)');
});

console.log('2. End');

// Output order:
// 1. Start
// 2. End
// 3. Promise.then (microtask)
// 4. setTimeout (macrotask)

// Explanation:
// 1. Synchronous code runs first.
// 2. Microtasks (Promise callbacks) run after the current stack, before macrotasks (setTimeout, setInterval).
// 3. Macrotasks run after microtasks are cleared.

// 2. Callbacks vs. Promises

// Callbacks: Functions passed as arguments to be executed later.
// Drawbacks: Callback hell, error handling is cumbersome.

function fetchDataCallback(url, callback) {
  setTimeout(() => {
    if (url === 'bad') return callback(new Error('Invalid URL'));
    callback(null, { data: 'result from ' + url });
  }, 100);
}

fetchDataCallback('good', (err, data) => {
  if (err) {
    console.error('Callback Error:', err);
    return;
  }
  console.log('Callback Data:', data);
});

// Promises: Represent a value that may be available now, later, or never.
// Advantages: Chaining, better error handling, avoids callback hell.

function fetchDataPromise(url) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (url === 'bad') return reject(new Error('Invalid URL'));
      resolve({ data: 'result from ' + url });
    }, 100);
  });
}

fetchDataPromise('good')
  .then(data => {
    console.log('Promise Data:', data);
    return fetchDataPromise('bad');
  })
  .catch(err => {
    console.error('Promise Error:', err);
  });

// 3. Promise API & Chaining

// Promise.resolve, Promise.reject, Promise.all, Promise.race, Promise.allSettled, Promise.any

// Chaining: Each .then returns a new promise, allowing chaining.

Promise.resolve(1)
  .then(val => {
    console.log('Chain 1:', val);
    return val + 1;
  })
  .then(val => {
    console.log('Chain 2:', val);
    return Promise.reject('Error in chain');
  })
  .catch(err => {
    console.error('Chain Error:', err);
    return 42;
  })
  .then(val => {
    console.log('Chain 3:', val);
  });

// Promise.all: Waits for all promises to resolve, rejects if any fail.
Promise.all([
  Promise.resolve('A'),
  Promise.resolve('B'),
  Promise.resolve('C')
]).then(results => {
  console.log('Promise.all:', results);
});

// Promise.race: Resolves/rejects as soon as one promise settles.
Promise.race([
  new Promise(res => setTimeout(() => res('First'), 50)),
  new Promise(res => setTimeout(() => res('Second'), 100))
]).then(result => {
  console.log('Promise.race:', result);
});

// Promise.allSettled: Waits for all to settle, never rejects.
Promise.allSettled([
  Promise.resolve('OK'),
  Promise.reject('Fail')
]).then(results => {
  console.log('Promise.allSettled:', results);
});

// Promise.any: Resolves as soon as any promise resolves, rejects if all reject.
Promise.any([
  Promise.reject('fail1'),
  Promise.resolve('success'),
  Promise.reject('fail2')
]).then(result => {
  console.log('Promise.any:', result);
}).catch(err => {
  console.error('Promise.any Error:', err);
});

// 4. async / await Internals

// async functions always return a promise.
// await pauses execution until the promise resolves or rejects.

async function asyncExample() {
  try {
    const a = await fetchDataPromise('good');
    console.log('async/await a:', a);
    const b = await fetchDataPromise('bad');
    console.log('async/await b:', b);
  } catch (err) {
    console.error('async/await Error:', err);
  }
}
asyncExample();

// Internals: async functions are syntactic sugar over promise chains.

function asyncSugar() {
  return fetchDataPromise('good')
    .then(a => {
      console.log('asyncSugar a:', a);
      return fetchDataPromise('bad');
    })
    .then(b => {
      console.log('asyncSugar b:', b);
    })
    .catch(err => {
      console.error('asyncSugar Error:', err);
    });
}
asyncSugar();

// 5. Generators & Co Routines

// Generators: Functions that can pause and resume execution (yield).
// Useful for lazy sequences, async control flow (with libraries like co).

function* numberGenerator() {
  yield 1;
  yield 2;
  yield 3;
}

const gen = numberGenerator();
console.log('Generator:', gen.next().value); // 1
console.log('Generator:', gen.next().value); // 2
console.log('Generator:', gen.next().value); // 3
console.log('Generator:', gen.next().done);  // true

// Async Generators: Use 'async function*' and 'for await...of'

async function* asyncNumberGenerator() {
  yield 1;
  await new Promise(res => setTimeout(res, 50));
  yield 2;
}

(async () => {
  for await (const num of asyncNumberGenerator()) {
    console.log('Async Generator:', num);
  }
})();

// Co Routines: Using generators to write async code (pre-async/await era)

function co(generatorFunc) {
  return function(...args) {
    const gen = generatorFunc(...args);
    return new Promise((resolve, reject) => {
      function step(nextF, arg) {
        let next;
        try {
          next = nextF.call(gen, arg);
        } catch (e) {
          return reject(e);
        }
        if (next.done) return resolve(next.value);
        Promise.resolve(next.value).then(
          v => step(gen.next, v),
          e => step(gen.throw, e)
        );
      }
      step(gen.next);
    });
  };
}

const coExample = co(function* () {
  const a = yield fetchDataPromise('good');
  console.log('co a:', a);
  try {
    const b = yield fetchDataPromise('bad');
    console.log('co b:', b);
  } catch (e) {
    console.error('co Error:', e);
  }
});
coExample();

// 6. Cancellation Patterns & AbortController

// Promises cannot be cancelled natively. Use patterns like AbortController.

function fetchWithAbort(url, signal) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      if (signal.aborted) {
        return reject(new Error('Aborted'));
      }
      resolve('Fetched: ' + url);
    }, 100);
    signal.addEventListener('abort', () => {
      clearTimeout(timeout);
      reject(new Error('Aborted by signal'));
    });
  });
}

const controller = new AbortController();
const { signal } = controller;

fetchWithAbort('resource', signal)
  .then(res => console.log('AbortController:', res))
  .catch(err => console.error('AbortController Error:', err));

setTimeout(() => controller.abort(), 50); // Aborts before fetch completes

// 7. Throttling, Debouncing & Rate Limiting

// Throttling: Ensures a function is called at most once in a specified period.

function throttle(fn, delay) {
  let last = 0;
  return function(...args) {
    const now = Date.now();
    if (now - last >= delay) {
      last = now;
      fn.apply(this, args);
    }
  };
}

let throttled = throttle(() => console.log('Throttled:', Date.now()), 200);
setInterval(throttled, 50); // Will log at most once every 200ms

// Debouncing: Ensures a function is called only after a specified period of inactivity.

function debounce(fn, delay) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

let debounced = debounce(() => console.log('Debounced:', Date.now()), 200);
for (let i = 0; i < 5; i++) setTimeout(debounced, i * 50); // Only last call runs

// Rate Limiting: Restricts the number of calls in a time window.

function rateLimit(fn, maxCalls, interval) {
  let calls = 0;
  let queue = [];
  setInterval(() => {
    calls = 0;
    while (queue.length && calls < maxCalls) {
      calls++;
      queue.shift()();
    }
  }, interval);
  return function(...args) {
    if (calls < maxCalls) {
      calls++;
      fn.apply(this, args);
    } else {
      queue.push(() => fn.apply(this, args));
    }
  };
}

let rateLimited = rateLimit(() => console.log('RateLimited:', Date.now()), 2, 500);
for (let i = 0; i < 6; i++) setTimeout(rateLimited, i * 100);

// 8. Worker Threads & Web Workers

// Web Workers (Browser): Run scripts in background threads.

if (typeof Worker !== 'undefined') {
  // In browser environment
  const workerScript = `
    self.onmessage = function(e) {
      self.postMessage('Worker received: ' + e.data);
    }
  `;
  const blob = new Blob([workerScript], { type: 'application/javascript' });
  const worker = new Worker(URL.createObjectURL(blob));
  worker.onmessage = e => console.log('Web Worker:', e.data);
  worker.postMessage('Hello from main thread');
}

// Worker Threads (Node.js): For CPU-intensive tasks in Node.js

// Uncomment below for Node.js >= v10.5.0
/*
const { Worker, isMainThread, parentPort } = require('worker_threads');
if (isMainThread) {
  const worker = new Worker(__filename);
  worker.on('message', msg => console.log('Worker Thread:', msg));
  worker.postMessage('Hello from main thread');
} else {
  parentPort.on('message', msg => {
    parentPort.postMessage('Worker received: ' + msg);
  });
}
*/

// Note: Web Workers and Worker Threads cannot access the DOM or main thread variables directly.
// Communication is via message passing (postMessage/onmessage).

// --- End of Chapter 3 ---