/***************************************************************************************************
 *  Node.js Core Module : `events`
 *  Author  : <Your Name Here>
 *  Purpose : Deep-dive, hands-on tour of the EventEmitter pattern through 10 progressively richer
 *            examples.  Every public API surface (from the most-used to the least-used) is covered,
 *            each snippet is isolated in its own function to avoid side-effects and comes with the
 *            expected output.  Execute this file with `node events-tour.js` to observe the logs.
 *
 *  Table of Contents
 *  ─────────────────
 *   1.   Basic “on / emit”                           – .on(), .emit()
 *   2.   “once” vs. “on”                            – .once()
 *   3.   Listener ordering                          – .prependListener(), .prependOnceListener()
 *   4.   Aliases & removal                          – .addListener() alias, .removeListener(), .off()
 *   5.   Nuking everything                          – .removeAllListeners()
 *   6.   Resource-safety                            – .setMaxListeners(), .getMaxListeners()
 *   7.   Introspection & debugging                  – .listeners(), .rawListeners(), .eventNames()
 *   8.   Counting listeners                         – EventEmitter.listenerCount()
 *   9.   Promise-style one-shot                     – events.once(emitter, eventName[, opts])
 *  10.   Async-iterator consumption                 – events.on(emitter, eventName[, opts])
 *
 *  Node.js Version: ≥ 16.x (older versions might lack `events.once/on` helpers)
 ***************************************************************************************************/

'use strict';
const events = require('events');
const { EventEmitter, once: eventsOnce, on: eventsOn } = events;

/***************************************************************************************************
 * Helper to run snippets isolated and pretty-print output separators
 **************************************************************************************************/
function runExample(title, fn) {
  console.log('\n' + '─'.repeat(80));
  console.log(`Example: ${title}`);
  console.log('─'.repeat(80));
  fn();
}

/***************************************************************************************************
 * 1) BASIC “on / emit”
 *    ──────────────────
 *    The canonical way to subscribe with `.on()` (alias `.addListener()`) and fire with `.emit()`.
 **************************************************************************************************/
runExample('1) Basic .on() and .emit()', () => {
  const clock = new EventEmitter();

  clock.on('tick', (time) => console.log(`Tick @ ${time}`));

  ['10:00', '10:01', '10:02'].forEach(t => clock.emit('tick', t));

  // Expected output:
  // Tick @ 10:00
  // Tick @ 10:01
  // Tick @ 10:02
});

/***************************************************************************************************
 * 2) “once” vs. “on”
 *    ───────────────
 *    `.once()` registers an auto-removing listener.  Perfect for one-shot initialization.
 **************************************************************************************************/
runExample('2) .once() fires exactly once', () => {
  const db = new EventEmitter();

  db.once('connected', () => console.log('DB connected (fired once)'));
  db.on('connected', () => console.log('DB connected (fires every time)'));

  db.emit('connected');
  db.emit('connected');

  // Expected output:
  // DB connected (fired once)
  // DB connected (fires every time)
  // DB connected (fires every time)
});

/***************************************************************************************************
 * 3) Listener ordering with prepend variants
 *    ───────────────────────────────────────
 *    Normal `.on()` pushes to the END of the listener array.
 *    `.prependListener()` inserts at the BEGINNING.
 *    Same philosophy for `.prependOnceListener()`.
 **************************************************************************************************/
runExample('3) .prependListener() & .prependOnceListener()', () => {
  const ee = new EventEmitter();

  ee.on('boot', () => console.log('#3'));
  ee.prependListener('boot', () => console.log('#2'));
  ee.prependOnceListener('boot', () => console.log('#1-(once)'));

  ee.emit('boot');
  ee.emit('boot');

  // Expected output:
  // #1-(once)
  // #2
  // #3
  // #2
  // #3
});

/***************************************************************************************************
 * 4) Aliases & removal
 *    ─────────────────
 *    `.addListener()` ≡ `.on()`
 *    `.removeListener()` ≡ `.off()`  (since Node 10)
 **************************************************************************************************/
runExample('4) addListener alias + removing with off()', () => {
  const bus = new EventEmitter();

  const listener = (msg) => console.log(`Received: ${msg}`);
  bus.addListener('message', listener); // alias of .on()

  bus.emit('message', 'hello 📨');

  bus.off('message', listener);         // alias of .removeListener()
  bus.emit('message', 'this will NOT show');

  // Expected output:
  // Received: hello 📨
});

/***************************************************************************************************
 * 5) removeAllListeners
 *    ──────────────────
 *    Quickly detach every listener for one event or for all events.
 **************************************************************************************************/
runExample('5) .removeAllListeners()', () => {
  const chat = new EventEmitter();

  chat.on('msg', (m) => console.log(`msg #1: ${m}`));
  chat.on('msg', (m) => console.log(`msg #2: ${m}`));

  chat.emit('msg', '👋 before purge');

  chat.removeAllListeners('msg'); // pass event name or omit for ALL events
  chat.emit('msg', '👋 after purge (nothing happens)');

  // Expected output:
  // msg #1: 👋 before purge
  // msg #2: 👋 before purge
});

/***************************************************************************************************
 * 6) setMaxListeners / getMaxListeners
 *    ─────────────────────────────────
 *    Avoid memory leaks by capping the number of listeners.
 **************************************************************************************************/
runExample('6) .setMaxListeners() & .getMaxListeners()', () => {
  const stock = new EventEmitter();

  console.log('Default cap:', stock.getMaxListeners());  // 10

  stock.setMaxListeners(2);
  console.log('New cap   :', stock.getMaxListeners());    // 2

  // Adding 3rd listener triggers a warning
  stock.on('price', () => {});
  stock.on('price', () => {});
  stock.on('price', () => {}); // exceeds cap

  // Expected output:
  // Default cap: 10
  // New cap   : 2
  // (Plus a ProcessWarning about possible memory leak)
});

/***************************************************************************************************
 * 7) listeners / rawListeners / eventNames
 *    ─────────────────────────────────────
 *    Introspection tools to debug & inspect internal state.
 **************************************************************************************************/
runExample('7) .listeners(), .rawListeners(), .eventNames()', () => {
  const srv = new EventEmitter();
  const noop = () => {};

  srv.on('boot', noop);
  srv.prependOnceListener('boot', () => console.log('boot'));
  srv.on('shutdown', noop);

  console.log('eventNames →', srv.eventNames()); // ['boot', 'shutdown']
  console.log('listeners(boot).length →', srv.listeners('boot').length);     // logical count
  console.log('rawListeners(boot).length →', srv.rawListeners('boot').length); // includes wrappers

  // Expected output (numbers may differ if you change listeners):
  // eventNames → [ 'boot', 'shutdown' ]
  // listeners(boot).length → 2
  // rawListeners(boot).length → 2
  // (plus 'boot' from the console.log inside listener)
  srv.emit('boot');
});

/***************************************************************************************************
 * 8) Static listenerCount
 *    ────────────────────
 *    Legacy helper (still useful) to count listeners without an instance method.
 **************************************************************************************************/
runExample('8) EventEmitter.listenerCount()', () => {
  const api = new EventEmitter();
  const f = () => {};

  api.on('req', f);
  api.on('req', f);

  console.log('Listener count:', EventEmitter.listenerCount(api, 'req')); // 2

  // Expected output:
  // Listener count: 2
});

/***************************************************************************************************
 * 9) Promise-style one-shot with events.once()
 *    ─────────────────────────────────────────
 *    No manual `.once()` handler, just `await events.once(...)`.
 **************************************************************************************************/
runExample('9) events.once() returns a Promise', async () => {
  const power = new EventEmitter();

  setTimeout(() => power.emit('ready', 42), 300); // simulate async

  const [code] = await eventsOnce(power, 'ready');
  console.log('Power-up code:', code);

  // Expected output (after 300 ms delay):
  // Power-up code: 42
});

/***************************************************************************************************
 * 10) Async-iterator consumption with events.on()
 *     ────────────────────────────────────────────
 *     Turn an event stream into an async iterator (`for await...of`) – extremely elegant!
 **************************************************************************************************/
runExample('10) events.on() async iterator', async () => {
  const ticker = new EventEmitter();

  // Producer: emits a tick every 100ms then stops
  let count = 0;
  const id = setInterval(() => {
    ticker.emit('tick', ++count);
    if (count === 3) {
      clearInterval(id);
      ticker.emit('end'); // we’ll use this to break the loop
    }
  }, 100);

  // Consumer: async iterator
  for await (const n of eventsOn(ticker, 'tick')) {
    console.log('Tick:', n);
    if (n === 3) break; // graceful exit
  }

  console.log('Loop completed');

  // Expected output:
  // Tick: 1
  // Tick: 2
  // Tick: 3
  // Loop completed
});
/***************************************************************************************************
 * End of file
 ***************************************************************************************************/