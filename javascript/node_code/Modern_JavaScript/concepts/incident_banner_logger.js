/* ======================================================================== *
 *  TOPIC 0 – UTILITIES (indent, logging, banner)                           *
 * ======================================================================== */
const util = require('util');
let currentIndent = 0;
const INDENT_STR = '  ';
function indent(level = 1) { currentIndent += level; }
function outdent(level = 1) { currentIndent = Math.max(0, currentIndent - level); }
function log(...args) {
  const prefix = INDENT_STR.repeat(currentIndent);
  console.log(prefix + args.map(a => (typeof a === 'string' ? a : util.inspect(a, { colors: true, depth: 4 }))).join(' '));
}
function banner(title) {
  console.log('\n' + '='.repeat(80));
  console.log(`>>> ${title}`);
  console.log('='.repeat(80));
}

/* 0A – simple log */
function util0A() { banner('0A – simple logging'); log('Hello, world'); }

/* 0B – timestamped log */
function util0B() { banner('0B – timestamped'); log(new Date().toISOString(), 'Event occurred'); }

/* 0C – indentation demo */
function util0C() {
  banner('0C – indentation');
  log('Level 0');
  indent(); log('Level 1'); indent(); log('Level 2'); outdent(2);
}

/* 0D – colored inspect */
function util0D() { banner('0D – colored objects'); log({ foo: 'bar', arr: [1, 2, 3] }); }

/* 0E – banner reuse */
function util0E() { banner('0E – done with utilities'); }

/* ======================================================================== *
 *  TOPIC 1 – TIMERS (setTimeout / clearTimeout)                            *
 * ======================================================================== */

/* 1A – basic timeout */
function timer1A() { banner('1A – basic setTimeout'); setTimeout(() => log('1A fired after 300ms'), 300); }

/* 1B – zero‑delay timeout (macro‑task) */
function timer1B() { banner('1B – 0ms timeout'); setTimeout(() => log('1B executed last'), 0); log('1B executed first'); }

/* 1C – cancel timeout */
function timer1C() {
  banner('1C – cancel timeout');
  const id = setTimeout(() => log('1C should not appear'), 200);
  clearTimeout(id);
  log('1C cancelled before firing');
}

/* 1D – capture closure variables */
function timer1D() {
  banner('1D – closure loop');
  for (let i = 1; i <= 3; i++) setTimeout(() => log(`1D – loop i=${i}`), i * 100);
}

/* 1E – timeout returning Promise */
function timer1E() {
  banner('1E – timeout as Promise');
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  sleep(150).then(() => log('1E – awaited 150ms'));
}

/* ======================================================================== *
 *  TOPIC 2 – REPEATING TIMERS (setInterval / clearInterval + drift)        *
 * ======================================================================== */

/* 2A – basic interval */
function interval2A() {
  banner('2A – interval');
  let n = 0;
  const id = setInterval(() => { log(`2A tick ${++n}`); if (n === 3) clearInterval(id); }, 200);
}

/* 2B – self‑clearing interval via counter */
function interval2B() {
  banner('2B – auto stop');
  let n = 5;
  const id = setInterval(() => {
    log(`2B – countdown ${n--}`);
    if (n < 0) clearInterval(id);
  }, 100);
}

/* 2C – drift‑corrected scheduler */
function interval2C() {
  banner('2C – drift‑corrected');
  const interval = 100;
  let expected = Date.now() + interval, n = 0;
  const step = () => {
    const dt = Date.now() - expected;
    log(`2C tick ${++n}, drift=${dt}ms`);
    expected += interval;
    if (n < 5) setTimeout(step, Math.max(0, interval - dt));
  };
  setTimeout(step, interval);
}

/* 2D – dynamic interval change */
function interval2D() {
  banner('2D – dynamic interval');
  let id, delay = 100;
  const tick = () => {
    log(`2D delay=${delay}`); delay += 100;
    clearInterval(id); id = setInterval(tick, delay);
    if (delay > 300) clearInterval(id);
  };
  id = setInterval(tick, delay);
}

/* 2E – using setTimeout recursively (better) */
function interval2E() {
  banner('2E – recursive timeouts');
  let n = 0;
  const recur = () => {
    if (++n > 3) return;
    log(`2E recur ${n}`); setTimeout(recur, 150);
  };
  recur();
}

/* ======================================================================== *
 *  TOPIC 3 – MICRO VS. MACRO TASKS                                         *
 * ======================================================================== */

/* 3A – process.nextTick beats everything */
function micro3A() {
  banner('3A – nextTick');
  process.nextTick(() => log('3A nextTick'));
  Promise.resolve().then(() => log('3A promise microtask'));
  setTimeout(() => log('3A timeout'), 0);
  log('3A sync');
}

/* 3B – queueMicrotask */
function micro3B() {
  banner('3B – queueMicrotask');
  queueMicrotask(() => log('3B microtask'));
  log('3B sync code');
}

/* 3C – setImmediate vs setTimeout 0 */
function micro3C() {
  banner('3C – immediate vs timeout');
  setImmediate(() => log('3C immediate'));
  setTimeout(() => log('3C timeout'), 0);
}

/* 3D – microtasks inside I/O callback */
function micro3D() {
  banner('3D – microtasks in I/O');
  require('fs').readFile(__filename, () => {
    log('3D fs callback');
    Promise.resolve().then(() => log('3D microtask in I/O'));
  });
}

/* 3E – starvation demo */
function micro3E() {
  banner('3E – nextTick starvation (stop after 5)');
  let n = 0;
  function spam() {
    if (++n > 5) return; // guard
    log(`3E tick ${n}`);
    process.nextTick(spam);
  }
  spam();
}

/* ======================================================================== *
 *  TOPIC 4 – fetch() & AbortController                                     *
 * ======================================================================== */
const { AbortController } = require('abort-controller');

/* 4A – simple GET */
async function fetch4A() {
  banner('4A – simple GET');
  const res = await fetch('https://jsonplaceholder.typicode.com/todos/1');
  log('4A status', res.status);
  log('4A json', await res.json());
}

/* 4B – POST with JSON body */
async function fetch4B() {
  banner('4B – POST');
  const res = await fetch('https://httpbin.org/post', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hello: 'world' })
  });
  log('4B code', res.status);
}

/* 4C – streaming response */
async function fetch4C() {
  banner('4C – streaming');
  const res = await fetch('https://jsonplaceholder.typicode.com/posts');
  let bytes = 0;
  for await (const chunk of res.body) bytes += chunk.length;
  log('4C bytes', bytes);
}

/* 4D – abort fetch */
async function fetch4D() {
  banner('4D – abort');
  const ac = new AbortController();
  const p = fetch('https://httpbin.org/delay/3', { signal: ac.signal }).catch(err => log('4D error', err.type));
  setTimeout(() => ac.abort(), 500);
  await p;
}

/* 4E – error handling */
async function fetch4E() {
  banner('4E – error code');
  const res = await fetch('https://httpbin.org/status/404');
  if (!res.ok) log('4E failed', res.status);
}

/* ======================================================================== *
 *  TOPIC 5 – FILE SYSTEM I/O                                               *
 * ======================================================================== */
const fs = require('fs');
const fsp = fs.promises;

/* 5A – async readFile */
function fs5A() {
  banner('5A – async readFile');
  fs.readFile(__filename, 'utf8', (e, d) => log('5A length', d.length));
}

/* 5B – sync readFile */
function fs5B() {
  banner('5B – sync readFile');
  const data = fs.readFileSync(__filename, 'utf8');
  log('5B chars', data.length);
}

/* 5C – promise readFile */
async function fs5C() {
  banner('5C – promise readFile');
  const d = await fsp.readFile(__filename);
  log('5C bytes', d.length);
}

/* 5D – stream read */
function fs5D() {
  banner('5D – stream read');
  let bytes = 0;
  fs.createReadStream(__filename).on('data', chunk => bytes += chunk.length).on('end', () => log('5D bytes', bytes));
}

/* 5E – compare sync vs async timing */
async function fs5E() {
  banner('5E – timing');
  console.time('sync');
  fs.readFileSync(__filename);
  console.timeEnd('sync');
  console.time('async');
  await fsp.readFile(__filename);
  console.timeEnd('async');
}

/* ======================================================================== *
 *  TOPIC 6 – EventEmitter                                                  *
 * ======================================================================== */
const { EventEmitter } = require('events');

/* 6A – basic emitter */
function evt6A() {
  banner('6A – basic');
  const ee = new EventEmitter();
  ee.on('hi', msg => log('6A got', msg));
  ee.emit('hi', 'Event 1');
}

/* 6B – once helper */
function evt6B() {
  banner('6B – once');
  const ee = new EventEmitter();
  ee.once('only', () => log('6B fired once'));
  ee.emit('only'); ee.emit('only');
}

/* 6C – async iteration over events */
async function evt6C() {
  banner('6C – async iterator');
  const ee = new EventEmitter();
  (async () => {
    for await (const [msg] of on(ee, 'msg')) log('6C got', msg);
  })();
  const { on } = require('events');
  ee.emit('msg', 'first'); ee.emit('msg', 'second');
  setTimeout(() => ee.removeAllListeners('msg'), 200);
}

/* 6D – error events */
function evt6D() {
  banner('6D – error handling');
  const ee = new EventEmitter();
  ee.on('error', err => log('6D caught', err.message));
  ee.emit('error', new Error('boom'));
}

/* 6E – wild card like behavior */
function evt6E() {
  banner('6E – pattern listeners');
  const ee = new EventEmitter();
  const any = (evt, ...a) => log('6E any', evt, a);
  const origEmit = ee.emit;
  ee.emit = function (evt, ...a) { any(evt, ...a); return origEmit.call(this, evt, ...a); };
  ee.emit('test', 42);
}

/* ======================================================================== *
 *  TOPIC 7 – NODE STREAMS                                                  *
 * ======================================================================== */
const { Readable, Transform } = require('stream');

