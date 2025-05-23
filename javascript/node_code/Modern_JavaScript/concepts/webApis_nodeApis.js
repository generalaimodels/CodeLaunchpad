/**********************************************************************************************
 * File: webApis_nodeApis.js
 *
 * Purpose:
 *   Exhaustive, example‑driven exploration of JavaScript “Web APIs / Node APIs” that enable
 *   asynchronous, non‑blocking operations outside the Call Stack. Each demo is self‑contained
 *   and can be executed with:  node webApis_nodeApis.js  (requires Node 18+ for built‑in fetch)
 *
 * Contents:
 *   0.  Utilities (indent, logging, banner)
 *   1.  Timers            – setTimeout / clearTimeout
 *   2.  Repeating timers  – setInterval / clearInterval + drift correction
 *   3.  Micro‑ vs. Macro‑tasks – queueMicrotask, process.nextTick, setImmediate
 *   4.  fetch() & AbortController – HTTP, streaming, cancellation, error handling
 *   5.  File‑system I/O   – fs.readFile (async) vs. fs.readFileSync (blocking)
 *   6.  EventEmitter      – custom async events + once() helper
 *   7.  Node Streams      – readable stream back‑pressure & async iteration
 *   8.  Worker Threads    – true parallelism, off‑main‑thread CPU work
 *   9.  Exception paths   – unhandledRejection & uncaughtException hooks
 *  10.  Cleanup & graceful shutdown
 **********************************************************************************************/

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 0. Utilities                                                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
let depth = 0;
const indent = () => '  '.repeat(depth);
const log = (msg, ...args) => console.log(indent() + msg, ...args);
const banner = (t) => {
  console.log('\n' + '='.repeat(t.length));
  console.log(t);
  console.log('='.repeat(t.length));
};

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 1. Timers – setTimeout / clearTimeout                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('1. Timers – setTimeout / clearTimeout');

depth++;
const timeoutId = setTimeout(() => {
  log('This message appears after ≥ 100 ms (macrotask).');
}, 100);

setTimeout(() => {
  log('Cancelling the first timeout before it fires.');
  clearTimeout(timeoutId);
}, 50);
depth--;

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 2. Repeating timers – setInterval / clearInterval + drift correction                     │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('2. Repeating timers – setInterval / clearInterval + drift correction');

depth++;
let counter = 0;
const every = 200; // ms
let expected = performance.now() + every;
const intervalId = setInterval(() => {
  const drift = performance.now() - expected;
  log(`tick #${++counter} – drift: ${drift.toFixed(1)} ms`);
  expected += every;
  if (counter === 5) {
    log('Stopping interval.');
    clearInterval(intervalId);
  }
}, every);
depth--;

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 3. Micro‑ vs. Macro‑tasks – queueMicrotask, process.nextTick, setImmediate               │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('3. Micro‑ vs. Macro‑tasks – queueMicrotask / nextTick / setImmediate');

depth++;
log('Synchronous A');

queueMicrotask(() => log('microtask ‑ queueMicrotask (after current stack, before macrotasks)'));

process.nextTick(() => log('microtask ‑ process.nextTick (Node‑only, before other microtasks)'));

setImmediate(() => log('macrotask ‑ setImmediate (Node‑only, after I/O)'));

setTimeout(() => log('macrotask ‑ setTimeout 0'), 0);

log('Synchronous B');
depth--;

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 4. fetch() & AbortController – HTTP, streaming, cancellation, error handling              │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('4. fetch() & AbortController – HTTP, streaming, cancellation, error handling');

depth++;
(async () => {
  const controller = new AbortController();
  const { signal } = controller;

  // Cancel the request if it exceeds 300 ms
  const watchdog = setTimeout(() => {
    controller.abort();
  }, 300);

  try {
    const res = await fetch('https://jsonplaceholder.typicode.com/todos/1', { signal });
    clearTimeout(watchdog);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const body = await res.json(); // body parsing is also asynchronous
    log('Fetched JSON:', body);
  } catch (err) {
    if (err.name === 'AbortError') {
      log('Request aborted (timeout hit).');
    } else {
      log('Fetch failed:', err.message);
    }
  }
})();
depth--;

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 5. File‑system I/O – fs.readFile (async) vs. fs.readFileSync (blocking)                   │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('5. File‑system I/O – fs.readFile (async) vs. fs.readFileSync (blocking)');

depth++;
import { readFile, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const demoPath = join(__dirname, 'demo.txt');
readFile(demoPath, 'utf8', (err, data) => {
  if (err) return log('Async read error:', err.code);
  log('Async fs.readFile →', data.trim());
});

try {
  const b = readFileSync(demoPath, 'utf8');
  log('Sync  fs.readFileSync →', b.trim());
} catch (e) {
  log('Sync read error:', e.code);
}
depth--;

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 6. EventEmitter – custom async events + once() helper                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('6. EventEmitter – custom async events + once() helper');

depth++;
import { EventEmitter, once } from 'node:events';

const bus = new EventEmitter();

(async () => {
  // Listener registered before emit
  bus.on('data', (payload) => log('on:data   received', payload));

  // Asynchronously wait for a single event
  const waiter = once(bus, 'data').then(([payload]) =>
    log('once:data received', payload)
  );

  setTimeout(() => bus.emit('data', { ok: true }), 100);
  await waiter;
  depth--;
})();

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 7. Node Streams – readable stream back‑pressure & async iteration                         │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('7. Node Streams – readable stream back‑pressure & async iteration');

depth++;
import { Readable } from 'node:stream';

const makeCounterStream = (n) =>
  new Readable({
    objectMode: true,
    read() {
      if (n === 0) this.push(null);
      else this.push(n--);
    }
  });

for await (const num of makeCounterStream(5)) {
  log('Stream chunk:', num);
}
depth--;

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 8. Worker Threads – true parallelism, off‑main‑thread CPU work                           │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('8. Worker Threads – true parallelism, off‑main‑thread CPU work');

depth++;
import { Worker, isMainThread, parentPort, threadId } from 'node:worker_threads';

const fibonacci = (n) => (n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2));

if (isMainThread) {
  const worker = new Worker(new URL(import.meta.url), { workerData: 40 });
  worker.on('message', (msg) => log(`[main] fib(40) = ${msg}`));
  worker.on('error', (err) => log('[main] worker error:', err));
  worker.on('exit', (code) => {
    log('[main] worker exited with', code);
    depth--;
  });
} else {
  // Worker context
  import { workerData } from 'node:worker_threads';
  const result = fibonacci(workerData);
  parentPort.postMessage(result);
}
 
/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 9. Exception paths – unhandledRejection & uncaughtException hooks                         │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('9. Exception paths – unhandledRejection & uncaughtException hooks');

process.on('unhandledRejection', (reason) => {
  log('‼️  Unhandled Promise rejection caught globally →', reason);
});
process.on('uncaughtException', (err) => {
  log('‼️  Uncaught Exception caught globally →', err.message);
});

/* Trigger a global rejection */
Promise.reject(new Error('Global rejection demo'));

/*──────────────────────────────────────────────────────────────────────────────────────────────
│ 10. Cleanup & graceful shutdown                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────*/
banner('10. Cleanup & graceful shutdown');

process.on('beforeExit', (code) => log('[beforeExit] EventLoop empty, code', code));
process.on('exit', (code) => log('[exit]  Final exit, code', code));

/* Keep process alive long enough for all async demos */
setTimeout(() => {
  log('All demos completed.');
}, 1500);