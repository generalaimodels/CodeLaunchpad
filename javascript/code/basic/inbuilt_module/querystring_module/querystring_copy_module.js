/**
 * Node.js 'querystring' Module - Comprehensive Examples
 * 
 * The 'querystring' module provides utilities for parsing and formatting URL query strings.
 * This file demonstrates all major and minor methods, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const querystring = require('querystring');

// 1. querystring.parse(): Parse a query string into an object
const obj1 = querystring.parse('foo=bar&abc=xyz&abc=123');
console.log('1. parse:', obj1);
// Output: { foo: 'bar', abc: [ 'xyz', '123' ] }

// 2. querystring.stringify(): Stringify an object into a query string
const str2 = querystring.stringify({ foo: 'bar', baz: ['qux', 'quux'], corge: '' });
console.log('2. stringify:', str2);
// Output: 'foo=bar&baz=qux&baz=quux&corge='

// 3. querystring.escape(): Custom escaping of a string (overridable)
console.log('3. escape:', querystring.escape('a b&c=d'));
// Output: 'a%20b%26c%3Dd'

// 4. querystring.unescape(): Custom unescaping of a string (overridable)
console.log('4. unescape:', querystring.unescape('a%20b%26c%3Dd'));
// Output: 'a b&c=d'

// 5. parse() with custom separator and equals
const obj5 = querystring.parse('foo:bar;abc:xyz', ';', ':');
console.log('5. parse (custom sep/eq):', obj5);
// Output: { foo: 'bar', abc: 'xyz' }

// 6. stringify() with custom separator and equals
const str6 = querystring.stringify({ foo: 'bar', abc: 'xyz' }, ';', ':');
console.log('6. stringify (custom sep/eq):', str6);
// Output: 'foo:bar;abc:xyz'

// 7. Edge Case: parse() with repeated keys
const obj7 = querystring.parse('a=1&a=2&a=3');
console.log('7. parse (repeated keys):', obj7);
// Output: { a: [ '1', '2', '3' ] }

// 8. Edge Case: stringify() with array values
const str8 = querystring.stringify({ a: [1, 2, 3] });
console.log('8. stringify (array):', str8);
// Output: 'a=1&a=2&a=3'

// 9. Exception Handling: parse() with non-string input
try {
    querystring.parse({ not: 'a string' });
} catch (err) {
    console.log('9. Exception caught:', err.message);
    // Output: input must be a string
}

// 10. Overriding escape/unescape for custom encoding
querystring.escape = function(str) { return str.replace(/ /g, '+'); };
querystring.unescape = function(str) { return str.replace(/\+/g, ' '); };
const str10 = querystring.stringify({ hello: 'world test' });
console.log('10. Custom escape:', str10);
// Output: 'hello=world+test'
const obj10 = querystring.parse('hello=world+test');
console.log('10. Custom unescape:', obj10);
// Output: { hello: 'world test' }

/**
 * Additional Notes:
 * - All major and minor methods of 'querystring' module are covered.
 * - Both default and custom separator/equals usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js querystring module.
 * - For new code, prefer the WHATWG URLSearchParams API for URL query handling.
 */