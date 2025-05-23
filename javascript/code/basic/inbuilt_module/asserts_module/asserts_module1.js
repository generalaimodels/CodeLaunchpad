/**
 * JavaScript Assertion Testing with 'assert' Module
 * 
 * This file demonstrates the usage of Node.js's built-in 'assert' module.
 * Each example covers a different method, from most common to least used,
 * with clear code, comments, and expected outputs.
 * 
 * To run: `node <filename>.js`
 */

const assert = require('assert');

// 1. assert(value[, message])
// Throws AssertionError if value is falsy.
function testAssertBasic() {
    assert(1 === 1, '1 should equal 1'); // Passes, no output
    // assert(1 === 2, '1 should equal 2'); // Throws: AssertionError: 1 should equal 2
}
testAssertBasic();

// 2. assert.strictEqual(actual, expected[, message])
// Tests strict equality (===)
function testStrictEqual() {
    assert.strictEqual(5, 5); // Passes
    // assert.strictEqual(5, '5'); // Throws: AssertionError
}
testStrictEqual();

// 3. assert.notStrictEqual(actual, expected[, message])
// Tests strict inequality (!==)
function testNotStrictEqual() {
    assert.notStrictEqual(5, '5'); // Passes
    // assert.notStrictEqual(5, 5); // Throws: AssertionError
}
testNotStrictEqual();

// 4. assert.deepStrictEqual(actual, expected[, message])
// Tests deep equality (objects, arrays, etc.)
function testDeepStrictEqual() {
    assert.deepStrictEqual({a: 1}, {a: 1}); // Passes
    // assert.deepStrictEqual({a: 1}, {a: '1'}); // Throws: AssertionError
}
testDeepStrictEqual();

// 5. assert.notDeepStrictEqual(actual, expected[, message])
// Tests deep inequality
function testNotDeepStrictEqual() {
    assert.notDeepStrictEqual({a: 1}, {a: 2}); // Passes
    // assert.notDeepStrictEqual({a: 1}, {a: 1}); // Throws: AssertionError
}
testNotDeepStrictEqual();

// 6. assert.equal(actual, expected[, message])
// Tests loose equality (==)
function testEqual() {
    assert.equal(5, '5'); // Passes
    // assert.equal(5, 6); // Throws: AssertionError
}
testEqual();

// 7. assert.notEqual(actual, expected[, message])
// Tests loose inequality (!=)
function testNotEqual() {
    assert.notEqual(5, 6); // Passes
    // assert.notEqual(5, '5'); // Throws: AssertionError
}
testNotEqual();

// 8. assert.deepEqual(actual, expected[, message])
// Tests deep loose equality (deprecated, but still used)
function testDeepEqual() {
    assert.deepEqual({a: 1}, {a: '1'}); // Passes (loose equality)
    // assert.deepEqual({a: 1}, {a: 2}); // Throws: AssertionError
}
testDeepEqual();

// 9. assert.notDeepEqual(actual, expected[, message])
// Tests deep loose inequality (deprecated)
function testNotDeepEqual() {
    assert.notDeepEqual({a: 1}, {a: 2}); // Passes
    // assert.notDeepEqual({a: 1}, {a: '1'}); // Throws: AssertionError
}
testNotDeepEqual();

// 10. assert.fail([message])
// Throws AssertionError with provided message
function testFail() {
    // assert.fail('This is a failure'); // Throws: AssertionError: This is a failure
}
testFail();

// 11. assert.throws(fn[, error][, message])
// Expects function to throw error
function testThrows() {
    assert.throws(
        () => { throw new TypeError('Wrong value'); },
        TypeError
    ); // Passes

    // assert.throws(
    //     () => { return 1; },
    //     Error
    // ); // Throws: AssertionError (function did not throw)
}
testThrows();

// 12. assert.doesNotThrow(fn[, error][, message])
// Expects function NOT to throw error
function testDoesNotThrow() {
    assert.doesNotThrow(
        () => { return 1; }
    ); // Passes

    // assert.doesNotThrow(
    //     () => { throw new Error('Oops'); }
    // ); // Throws: AssertionError
}
testDoesNotThrow();

// 13. assert.ifError(value)
// Throws if value is truthy (commonly used for callbacks)
function testIfError() {
    assert.ifError(null); // Passes
    // assert.ifError('error'); // Throws: AssertionError: error
}
testIfError();

// 14. assert.match(string, regexp[, message])
// Asserts string matches regular expression
function testMatch() {
    assert.match('hello world', /world/); // Passes
    // assert.match('hello', /world/); // Throws: AssertionError
}
testMatch();

// 15. assert.doesNotMatch(string, regexp[, message])
// Asserts string does NOT match regular expression
function testDoesNotMatch() {
    assert.doesNotMatch('hello', /world/); // Passes
    // assert.doesNotMatch('hello world', /world/); // Throws: AssertionError
}
testDoesNotMatch();

// 16. assert.ok(value[, message])
// Asserts value is truthy (alias for assert)
function testOk() {
    assert.ok(true); // Passes
    // assert.ok(false, 'Should be truthy'); // Throws: AssertionError: Should be truthy
}
testOk();

// 17. assert.rejects(asyncFn[, error][, message])
// Expects async function to reject
async function testRejects() {
    await assert.rejects(
        async () => { throw new Error('fail'); },
        Error
    ); // Passes

    // await assert.rejects(
    //     async () => { return 1; },
    //     Error
    // ); // Throws: AssertionError
}
testRejects();

// 18. assert.doesNotReject(asyncFn[, error][, message])
// Expects async function NOT to reject
async function testDoesNotReject() {
    await assert.doesNotReject(
        async () => { return 1; }
    ); // Passes

    // await assert.doesNotReject(
    //     async () => { throw new Error('fail'); }
    // ); // Throws: AssertionError
}
testDoesNotReject();

/**
 * Summary:
 * - All major and minor methods of 'assert' module are covered.
 * - Each example is self-contained and demonstrates expected behavior.
 * - Uncomment lines to see assertion failures in action.
 */