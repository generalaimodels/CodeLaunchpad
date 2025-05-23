// ==========================
// 0. Utilities (indent, logging, banner)
// ==========================
const util = require('util');
const colors = require('colors/safe');

function indent(str, level = 2) {
    return str.split('\n').map(line => ' '.repeat(level) + line).join('\n');
}

function log(...args) {
    console.log(colors.cyan('[LOG]'), ...args);
}

function error(...args) {
    console.error(colors.red('[ERR]'), ...args);
}

function banner(title) {
    const line = '='.repeat(60);
    console.log('\n' + colors.yellow(line));
    console.log(colors.yellow.bold(`= ${title}`));
    console.log(colors.yellow(line));
}

// ==========================
// 1. Timers – setTimeout / clearTimeout
// ==========================
banner('1. Timers – setTimeout / clearTimeout');

// Example 1: Basic setTimeout
setTimeout(() => log('Example 1: Timeout after 500ms'), 500);

// Example 2: Passing arguments to callback
setTimeout((msg) => log('Example 2:', msg), 300, 'Hello from setTimeout');

// Example 3: clearTimeout cancels timer
const t3 = setTimeout(() => error('Example 3: Should not run'), 200);
clearTimeout(t3);

// Example 4: setTimeout with 0ms (executes after current call stack)
setTimeout(() => log('Example 4: setTimeout 0ms'), 0);

// Example 5: Nested setTimeouts (recursive delay)
let count = 0;
function repeatTimeout() {
    if (++count < 3) {
        log(`Example 5: repeatTimeout count=${count}`);
        setTimeout(repeatTimeout, 100);
    }
}
setTimeout(repeatTimeout, 100);

// ==========================
// 2. Repeating timers – setInterval / clearInterval + drift correction
// ==========================
banner('2. Repeating timers – setInterval / clearInterval + drift correction');

// Example 1: Basic setInterval
let i1 = 0;
const int1 = setInterval(() => {
    log(`Example 1: setInterval count=${++i1}`);
    if (i1 >= 3) clearInterval(int1);
}, 200);

// Example 2: clearInterval before first run
const int2 = setInterval(() => error('Example 2: Should not run'), 100);
clearInterval(int2);

// Example 3: setInterval with arguments
const int3 = setInterval((msg) => log('Example 3:', msg), 150, 'Interval arg');
setTimeout(() => clearInterval(int3), 500);

// Example 4: setInterval drift demonstration
let driftCount = 0, start = Date.now();
const int4 = setInterval(() => {
    const now = Date.now();
    log(`Example 4: Drift = ${now - start - driftCount * 100}ms`);
    if (++driftCount >= 3) clearInterval(int4);
}, 100);

// Example 5: Drift-corrected interval
function setDriftlessInterval(fn, interval, times = 3) {
    let count = 0, expected = Date.now() + interval;
    function step() {
        fn(++count);
        if (count >= times) return;
        expected += interval;
        setTimeout(step, Math.max(0, expected - Date.now()));
    }
    setTimeout(step, interval);
}
setDriftlessInterval((n) => log(`Example 5: Driftless interval #${n}`), 120);

// ==========================
// 3. Micro vs. Macro tasks – queueMicrotask, process.nextTick, setImmediate
// ==========================
banner('3. Micro vs. Macro tasks – queueMicrotask, process.nextTick, setImmediate');

// Example 1: setTimeout (macro) vs. queueMicrotask (micro)
setTimeout(() => log('Example 1: setTimeout (macro)'), 0);
queueMicrotask(() => log('Example 1: queueMicrotask (micro)'));

// Example 2: process.nextTick (micro, Node.js only)
process.nextTick(() => log('Example 2: process.nextTick'));

// Example 3: setImmediate (macro, Node.js only)
setImmediate(() => log('Example 3: setImmediate'));

// Example 4: Execution order
log('Example 4: Synchronous');
queueMicrotask(() => log('Example 4: queueMicrotask'));
process.nextTick(() => log('Example 4: nextTick'));
setTimeout(() => log('Example 4: setTimeout'), 0);
setImmediate(() => log('Example 4: setImmediate'));

// Example 5: Microtask inside macrotask
setTimeout(() => {
    log('Example 5: setTimeout');
    queueMicrotask(() => log('Example 5: queueMicrotask inside setTimeout'));
}, 0);

// ==========================
// 4. fetch() & AbortController – HTTP, streaming, cancellation, error handling
// ==========================
banner('4. fetch() & AbortController – HTTP, streaming, cancellation, error handling');
const fetch = require('node-fetch');

// Example 1: Basic fetch
fetch('https://jsonplaceholder.typicode.com/posts/1')
    .then(res => res.json())
    .then(data => log('Example 1:', data.title));

// Example 2: Fetch with AbortController (cancellation)
const ac2 = new AbortController();
fetch('https://jsonplaceholder.typicode.com/posts/2', { signal: ac2.signal })
    .then(res => res.json())
    .then(data => log('Example 2:', data.title))
    .catch(err => error('Example 2:', err.name));
ac2.abort(); // Cancel immediately

// Example 3: Streaming response
fetch('https://jsonplaceholder.typicode.com/posts')
    .then(res => {
        log('Example 3: Streaming response');
        return res.body.getReader().read();
    })
    .then(({ value, done }) => log('Example 3: First chunk length', value && value.length));

// Example 4: Error handling (404)
fetch('https://jsonplaceholder.typicode.com/404')
    .then(res => {
        if (!res.ok) throw new Error('Not Found');
        return res.text();
    })
    .catch(err => error('Example 4:', err.message));

// Example 5: Timeout with AbortController
function fetchWithTimeout(url, ms) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), ms);
    return fetch(url, { signal: ac.signal })
        .finally(() => clearTimeout(timer));
}
fetchWithTimeout('https://jsonplaceholder.typicode.com/posts/3', 100)
    .then(res => res.json())
    .then(data => log('Example 5:', data.title))
    .catch(err => error('Example 5:', err.name));

// ==========================
// 5. File system I/O – fs.readFile (async) vs. fs.readFileSync (blocking)
// ==========================
banner('5. File system I/O – fs.readFile (async) vs. fs.readFileSync (blocking)');
const fs = require('fs');
const path = require('path');
const tmpFile = path.join(__dirname, 'tmp.txt');
fs.writeFileSync(tmpFile, 'Hello\nWorld\nNode.js\nAsync\nSync');

// Example 1: Async readFile
fs.readFile(tmpFile, 'utf8', (err, data) => {
    if (err) return error('Example 1:', err);
    log('Example 1:', data.split('\n')[0]);
});

// Example 2: Sync readFileSync
try {
    const data2 = fs.readFileSync(tmpFile, 'utf8');
    log('Example 2:', data2.split('\n')[1]);
} catch (e) {
    error('Example 2:', e);
}

// Example 3: Async error (file not found)
fs.readFile('notfound.txt', 'utf8', (err) => {
    if (err) error('Example 3:', err.code);
});

// Example 4: Sync error (file not found)
try {
    fs.readFileSync('notfound.txt', 'utf8');
} catch (e) {
    error('Example 4:', e.code);
}

// Example 5: Async readFile with Promise
fs.promises.readFile(tmpFile, 'utf8')
    .then(data => log('Example 5:', data.split('\n')[2]));

// ==========================
// 6. EventEmitter – custom async events + once() helper
// ==========================
banner('6. EventEmitter – custom async events + once() helper');
const { EventEmitter, once } = require('events');

class MyEmitter extends EventEmitter {}
const emitter = new MyEmitter();

// Example 1: Basic event
emitter.on('greet', name => log('Example 1: Hello', name));
emitter.emit('greet', 'Alice');

// Example 2: Async event handler
emitter.on('async', async (msg) => {
    await new Promise(r => setTimeout(r, 100));
    log('Example 2:', msg);
});
emitter.emit('async', 'Async event!');

// Example 3: once() helper (Promise)
once(emitter, 'ready').then(([msg]) => log('Example 3:', msg));
emitter.emit('ready', 'Ready event fired');

// Example 4: Remove listener
function tempListener() { log('Example 4: Should not run'); }
emitter.on('temp', tempListener);
emitter.removeListener('temp', tempListener);
emitter.emit('temp');

// Example 5: Error event
emitter.on('error', err => error('Example 5:', err.message));
emitter.emit('error', new Error('Something went wrong'));

// ==========================
// 7. Node Streams – readable stream back pressure & async iteration
// ==========================
banner('7. Node Streams – readable stream back pressure & async iteration');
const { Readable } = require('stream');

// Example 1: Basic Readable stream
const readable1 = Readable.from(['A', 'B', 'C']);
readable1.on('data', chunk => log('Example 1:', chunk.toString()));

// Example 2: Back pressure (pause/resume)
const readable2 = Readable.from(['1', '2', '3', '4']);
readable2.on('data', chunk => {
    log('Example 2:', chunk.toString());
    readable2.pause();
    setTimeout(() => readable2.resume(), 100);
});

// Example 3: Async iteration
(async () => {
    const readable3 = Readable.from(['X', 'Y', 'Z']);
    for await (const chunk of readable3) {
        log('Example 3:', chunk.toString());
    }
})();

// Example 4: Custom Readable stream
class NumberStream extends Readable {
    constructor(max) {
        super();
        this.current = 1;
        this.max = max;
    }
    _read() {
        if (this.current > this.max) this.push(null);
        else this.push(String(this.current++));
    }
}
const ns = new NumberStream(3);
ns.on('data', chunk => log('Example 4:', chunk.toString()));

// Example 5: Handling 'end' event
const readable5 = Readable.from(['end', 'of', 'stream']);
readable5.on('end', () => log('Example 5: Stream ended'));
readable5.resume();

// ==========================
// 8. Worker Threads – true parallelism, off main thread CPU work
// ==========================
banner('8. Worker Threads – true parallelism, off main thread CPU work');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

if (isMainThread) {
    // Example 1: Basic worker
    new Worker(__filename, { workerData: { ex: 1 } });

    // Example 2: Passing data and receiving result
    const w2 = new Worker(__filename, { workerData: { ex: 2, num: 5 } });
    w2.on('message', msg => log('Example 2:', msg));

    // Example 3: Error handling
    const w3 = new Worker(__filename, { workerData: { ex: 3 } });
    w3.on('error', err => error('Example 3:', err.message));

    // Example 4: Multiple workers (parallel)
    for (let i = 0; i < 2; ++i) {
        const w = new Worker(__filename, { workerData: { ex: 4, id: i } });
        w.on('message', msg => log('Example 4:', msg));
    }

    // Example 5: Worker termination
    const w5 = new Worker(__filename, { workerData: { ex: 5 } });
    w5.terminate().then(() => log('Example 5: Worker terminated'));
} else {
    switch (workerData.ex) {
        case 1:
            parentPort.postMessage('Example 1: Hello from worker');
            break;
        case 2:
            parentPort.postMessage(`Factorial: ${factorial(workerData.num)}`);
            break;
        case 3:
            throw new Error('Example 3: Worker error');
        case 4:
            parentPort.postMessage(`Example 4: Worker ${workerData.id} running`);
            break;
        case 5:
            setTimeout(() => parentPort.postMessage('Should not run'), 1000);
            break;
    }
}
function factorial(n) { return n <= 1 ? 1 : n * factorial(n - 1); }

// ==========================
// 9. Exception paths – unhandledRejection & uncaughtException hooks
// ==========================
banner('9. Exception paths – unhandledRejection & uncaughtException hooks');

// Example 1: unhandledRejection
process.on('unhandledRejection', (reason, promise) => {
    error('Example 1: UnhandledRejection', reason.message);
});
Promise.reject(new Error('Example 1: Promise rejected'));

// Example 2: uncaughtException
process.on('uncaughtException', err => {
    error('Example 2: UncaughtException', err.message);
});
setTimeout(() => { throw new Error('Example 2: Thrown error'); }, 100);

// Example 3: handled rejection (no event)
Promise.reject(new Error('Example 3: Handled')).catch(() => {});

// Example 4: async function unhandled rejection
(async () => {
    throw new Error('Example 4: Async unhandled');
})();

// Example 5: Remove listener
function tempHandler() {}
process.on('unhandledRejection', tempHandler);
process.removeListener('unhandledRejection', tempHandler);

// ==========================
// 10. Cleanup & graceful shutdown
// ==========================
banner('10. Cleanup & graceful shutdown');

let shuttingDown = false;
function cleanup() {
    if (shuttingDown) return;
    shuttingDown = true;
    log('Cleanup: Closing resources...');
    try { fs.unlinkSync(tmpFile); } catch {}
    setTimeout(() => {
        log('Cleanup: Done. Exiting.');
        process.exit(0);
    }, 100);
}

// Example 1: SIGINT (Ctrl+C)
process.on('SIGINT', () => {
    log('Example 1: SIGINT received');
    cleanup();
});

// Example 2: SIGTERM
process.on('SIGTERM', () => {
    log('Example 2: SIGTERM received');
    cleanup();
});

// Example 3: beforeExit
process.on('beforeExit', (code) => {
    log('Example 3: beforeExit', code);
});

// Example 4: exit
process.on('exit', (code) => {
    log('Example 4: exit', code);
});

// Example 5: Custom shutdown trigger
setTimeout(() => {
    log('Example 5: Custom shutdown');
    cleanup();
}, 2000);