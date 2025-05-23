/********************************************************************************************
*  Title  : Node.js `assert` Module – 10 Exhaustive Examples (All APIs Covered)             *
*  Author : “Best-Coder-GPT” 🤖                                                             *
*  Target : Next-gen devs who want rock-solid confidence with the *full* `assert` arsenal. *
*                                                                                          *
*  How to use this file                                                                     *
*  ─────────────────────                                                                     *
*  1. `node assert_demo.js` (or whatever you call this file)                                *
*  2. Read the inline comments + expected output.                                           *
*  3. Tweak / play / break things to feel the behaviour.                                    *
*                                                                                          *
*  Coverage checklist ✔                                                                     *
*  ─────────────────────                                                                     *
*  • assert(value[, message]) / assert.ok() / assert.AssertionError                         *
*  • assert.equal / notEqual / strictEqual / notStrictEqual                                 *
*  • assert.deepEqual / notDeepEqual / deepStrictEqual / notDeepStrictEqual                 *
*  • assert.match / doesNotMatch                                                            *
*  • assert.throws / doesNotThrow                                                           *
*  • assert.rejects / doesNotReject                                                         *
*  • assert.ifError                                                                         *
*  • assert.fail                                                                            *
*  • assert.callTracker()                                                                   *
*  • assert.strict (the *strict mode* variant; identical API surface)                       *
********************************************************************************************/


const assert = require('assert'); // Classic mode
const strictAssert = require('assert/strict'); // Always-strict variant (ESM-friendly too)

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 1 – Basic Truthiness Checks (assert(), assert.ok())                              */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 1 ─ Truthiness');
assert(true);           // passes silently
assert.ok(1);           // passes silently (non-zero is truthy)
// Expected: no output, no throw

try {
  assert(false, '0️⃣  This will throw');
} catch (err) {
  console.log('Expected: ', err.message); // 0️⃣  This will throw
}

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 2 – Shallow Equality vs. Strict Equality                                         */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 2 ─ Shallow Equality');
assert.equal(5, '5');               // loose (==) comparison ➜ passes
assert.notEqual(5, 6);              // passes

try { assert.equal(5, 6); } catch (e) { console.log('equal failed as expected'); }

assert.strictEqual(5, 5);           // strict (===) comparison ➜ passes
assert.notStrictEqual(5, '5');      // passes

try { assert.strictEqual(5, '5'); } catch (e) { console.log('strictEqual failed as expected'); }

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 3 – Deep (Recursive) Equality                                                    */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 3 ─ Deep Equality');
const objA = { x: 1, y: [2, 3] };
const objB = { x: 1, y: [2, 3] };
assert.deepEqual(objA, objB);                 // Uses == for primitives inside ➜ passes
assert.notDeepEqual(objA, { x: 2 });          // passes

// deepStrictEqual checks prototypes + uses === internally
assert.deepStrictEqual(objA, objB);
const withBuffer = { buf: Buffer.from('42') };
try { assert.deepStrictEqual(withBuffer, { buf: Buffer.from('42').toString() }); }
catch (e) { console.log('deepStrictEqual failed as expected'); }

assert.notDeepStrictEqual({ a: 1 }, { a: '1' });

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 4 – String Pattern Matching                                                      */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 4 ─ Pattern Matching');
assert.match('hello-world', /hello/);
assert.doesNotMatch('hello-world', /bye/);

try { assert.match('dev', /^prod/); } catch (e) { console.log('match failed as expected'); }

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 5 – Sync Error Expectations (throws / doesNotThrow)                              */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 5 ─ throws / doesNotThrow');
function willThrow() { throw new TypeError('💥'); }
function willNotThrow() { return 42; }

assert.throws(willThrow, { name: 'TypeError', message: '💥' });
assert.doesNotThrow(willNotThrow);

try { assert.doesNotThrow(willThrow); } catch (e) { console.log('doesNotThrow failed ✔'); }

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 6 – Async Error Expectations (rejects / doesNotReject)                           */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 6 ─ rejects / doesNotReject');
const asyncOk = () => Promise.resolve('👍');
const asyncFail = () => Promise.reject(new RangeError('Out of range'));

(async () => {
  await assert.rejects(asyncFail, RangeError);           // passes
  await assert.doesNotReject(asyncOk);                   // passes

  try {
    await assert.doesNotReject(asyncFail);
  } catch (e) {
    console.log('doesNotReject failed ✔');
  }
})();

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 7 – ifError: shortcut for “err must be falsy”                                     */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 7 ─ ifError');
function nodeStyleCallback(err, data) {
  assert.ifError(err); // throws if err is truthy
  console.log('Callback received:', data);
}
nodeStyleCallback(null, '✅  All good');

try { nodeStyleCallback(new Error('Boom')); }
catch (e) { console.log('ifError triggered as designed'); }

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 8 – assert.fail & custom AssertionError                                          */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 8 ─ fail & custom AssertionError');
// Manual failure (always throws):
try {
  assert.fail('Forced failure');
} catch (e) {
  console.log('fail()  message:', e.message);
}

// Creating/throwing a tailored AssertionError:
try {
  throw new assert.AssertionError({
    message   : 'Custom failure',
    expected  : 1,
    actual    : 2,
    operator  : '===',
    code      : 'ERR_CUSTOM'
  });
} catch (e) {
  console.log('Custom AssertionError code: ', e.code); // ERR_CUSTOM
}

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 9 – callTracker(): ensure fn gets invoked N times                                 */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 9 ─ callTracker');
const tracker = assert.callTracker;
const mustCallTwice = tracker.calls((x) => console.log('Tracked call arg =', x), 2);

mustCallTwice('first');
setImmediate(() => mustCallTwice('second')); // second call async

// Let tracker verify on process exit:
process.on('exit', () => {
  tracker.verify(); // will throw if call counts do not match
});

/*───────────────────────────────────────────────────────────────────────────────────────────*/
/* Example 10 – strictAssert variant (ES/Strict-only)                                       */
/*───────────────────────────────────────────────────────────────────────────────────────────*/
console.log('\nExample 10 ─ assert/strict');
try {
  strictAssert.equal(1, '1'); // In strict mode, .equal -> ===, so this *fails*
} catch (e) {
  console.log('strictAssert.equal failed as expected');
}

strictAssert.deepStrictEqual({ a: 1 }, { a: 1 }); // passes

/********************************************************************************************
*  End of file – you now wield the *entire* Node.js assert API like a pro! 🚀               *
********************************************************************************************/