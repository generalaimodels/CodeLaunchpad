/***************************************************************************************************
* Title        : “Node.js ‘assert’ Module ‒ Definitive Hands-on Cheat-Sheet”
* Description  : One single file that demonstrates EVERY public API exposed by Node.js `assert`
*                (common → least-used) through 10 concise, **independent** examples.
* Audience     : Next-gen developers wanting bullet-proof confidence in assertion testing.
* Coding style : Modern ES-2023, 2-space indent, const/let, arrow-functions, strict mode.                                 
***************************************************************************************************/
'use strict';
const assert = require('assert'); // core module – no install needed

/***************************************************************************************************
* Example-1  ─ Basic Truthiness & Immediate Failure
* Covers      : assert(value), assert.ok(value), assert.fail(message)
***************************************************************************************************/
(() => {
  console.log('\nEX-1 ► Basic truthiness');

  assert(true);                      // passes silently
  assert.ok(42, '42 is truthy');     // alias of assert()

  try {
    assert.fail('Manual failure for demo');  // always throws
  } catch (err) {
    console.log(' ➜ expected:', err.message); // Expected output: Manual failure for demo
  }
})();

/***************************************************************************************************
* Example-2  ─ Shallow Equality vs. Strict Equality
* Covers      : equal, notEqual, strictEqual, notStrictEqual
***************************************************************************************************/
(() => {
  console.log('\nEX-2 ► Shallow equality');

  assert.equal(1, '1');             // loose (==) comparison – passes
  assert.notEqual(1, 2);            // passes

  try {                             // strict (===) comparison – will fail
    assert.strictEqual(1, '1');
  } catch (err) {
    console.log(' ➜ strictEqual failed:', err.message);
  }

  assert.notStrictEqual(1, 2);      // passes
})();

/***************************************************************************************************
* Example-3  ─ Deep Equality (Objects, Arrays, Maps, Sets…)
* Covers      : deepEqual, notDeepEqual, deepStrictEqual, notDeepStrictEqual
***************************************************************************************************/
(() => {
  console.log('\nEX-3 ► Deep equality');

  const objA = { id: 5, tags: ['js', 'assert'] };
  const objB = { id: 5, tags: ['js', 'assert'] };

  assert.deepEqual(objA, objB);              // structural, allows prototypes & type coercion
  assert.notDeepEqual(objA, { id: '5' });    // fails when structure is different enough

  class Cat { constructor(name) { this.name = name; } }
  const kitty1 = new Cat('Fluffy');
  const kitty2 = { name: 'Fluffy' };         // plain object, not instance of Cat

  try {
    assert.deepStrictEqual(kitty1, kitty2);  // fails – different prototypes
  } catch (err) {
    console.log(' ➜ deepStrictEqual failed:', err.message);
  }

  assert.notDeepStrictEqual(kitty1, kitty2); // passes
})();

/***************************************************************************************************
* Example-4  ─ Pattern Matching Utilities
* Covers      : match, doesNotMatch
***************************************************************************************************/
(() => {
  console.log('\nEX-4 ► RegExp matchers');

  const email = 'test@example.com';
  assert.match(email, /^[\w.-]+@[\w.-]+\.\w+$/);

  try {
    assert.doesNotMatch(email, /@example/);  // will throw because it actually matches
  } catch (err) {
    console.log(' ➜ doesNotMatch failed:', err.message);
  }
})();

/***************************************************************************************************
* Example-5  ─ Exception Assertions (sync)
* Covers      : throws, doesNotThrow, ifError
***************************************************************************************************/
(() => {
  console.log('\nEX-5 ► Synchronous exceptions');

  const willThrow = () => { throw new TypeError('Boom'); };
  const wontThrow  = () => 42;

  assert.throws(willThrow, { name: 'TypeError', message: 'Boom' });

  assert.doesNotThrow(wontThrow);

  try {
    assert.ifError(new Error('Unexpected error object')); // throws if value is truthy
  } catch (err) {
    console.log(' ➜ ifError caught:', err.message);
  }
})();

/***************************************************************************************************
* Example-6  ─ Promise-aware Assertions
* Covers      : rejects, doesNotReject
***************************************************************************************************/
(() => {
  console.log('\nEX-6 ► Asynchronous promises');

  const asyncReject = () => Promise.reject(new RangeError('Out of range'));
  const asyncResolve = () => Promise.resolve(100);

  (async () => {
    await assert.rejects(asyncReject, RangeError);     // passes
    await assert.doesNotReject(asyncResolve);          // passes
  })();
})();

/***************************************************************************************************
* Example-7  ─ Custom AssertionError
* Covers      : assert.AssertionError
***************************************************************************************************/
(() => {
  console.log('\nEX-7 ► Custom AssertionError');

  try {
    throw new assert.AssertionError({
      actual: 3, expected: 2, operator: '===',
      message: '3 is not strictly equal to 2',
      stackStartFn: (() => {})                       // hide current wrapper in stack trace
    });
  } catch (err) {
    if (err instanceof assert.AssertionError) {
      console.log(' ➜ custom AssertionError:', err.message);
    }
  }
})();

/***************************************************************************************************
* Example-8  ─ Manual fail with extra context
* Covers      : fail(actual, expected, message, operator)
***************************************************************************************************/
(() => {
  console.log('\nEX-8 ► Custom fail() details');

  const a = 10, b = 20;
  if (a < b) {
    // Everything fine
  } else {
    // We’ll never reach here, but shows full signature
    assert.fail(a, b, 'a should be less than b', '<');
  }
  console.log(' ➜ no output because condition met');
})();

/***************************************************************************************************
* Example-9  ─ CallTracker (least-used but very handy)
* Covers      : assert.CallTracker, tracker.calls(), tracker.verify()
***************************************************************************************************/
(() => {
  console.log('\nEX-9 ► CallTracker');

  const tracker = new assert.CallTracker();

  const cb = tracker.calls((x) => x * 2, 2); // must be invoked exactly 2 times
  cb(3);
  cb(4);

  try {
    tracker.verify();                        // passes; no exception
    console.log(' ➜ CallTracker verified OK');
  } catch (err) {
    console.error(err);
  }
})();

/***************************************************************************************************
* Example-10 ─ Combining multiple assertions in a test helper
* Demonstrates how everything can work together in a mini test case.
***************************************************************************************************/
(() => {
  console.log('\nEX-10 ► Mini integration test');

  const add = (x, y) => x + y;

  const testAdd = () => {
    assert.strictEqual(add(2, 3), 5);
    assert.notStrictEqual(add(-1, 1), 3);
    assert.deepStrictEqual({ sum: add(1, 1) }, { sum: 2 });
    assert.match(String(add(10, 5)), /^\d+$/);
  };

  assert.doesNotThrow(testAdd);
  console.log(' ➜ All mini-tests passed 🎉');
})();

/***************************************************************************************************
* End of cheat-sheet – you now master every single API in `assert` 🎯
***************************************************************************************************/