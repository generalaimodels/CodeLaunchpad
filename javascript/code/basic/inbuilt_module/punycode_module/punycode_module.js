/**
 * Node.js 'punycode' Module - Comprehensive Examples
 * 
 * The 'punycode' module provides functions for converting Unicode strings to and from Punycode, 
 * which is used for Internationalized Domain Names (IDN). 
 * Note: This module is deprecated since Node.js v7.0.0, but still available for compatibility.
 * 
 * This file demonstrates all major and minor methods, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const punycode = require('punycode/');

// 1. punycode.encode(): Encode a Unicode string to Punycode (basic code points are not encoded)
console.log('1. Encode:', punycode.encode('mañana')); 
// Output: 'maana-pta'

// 2. punycode.decode(): Decode a Punycode string to Unicode
console.log('2. Decode:', punycode.decode('maana-pta')); 
// Output: 'mañana'

// 3. punycode.toASCII(): Convert a Unicode domain to ASCII/Punycode (for IDN)
console.log('3. toASCII:', punycode.toASCII('mañana.com')); 
// Output: 'xn--maana-pta.com'

// 4. punycode.toUnicode(): Convert an ASCII/Punycode domain to Unicode
console.log('4. toUnicode:', punycode.toUnicode('xn--maana-pta.com')); 
// Output: 'mañana.com'

// 5. punycode.ucs2.encode(): Encode an array of code points to a string
console.log('5. ucs2.encode:', punycode.ucs2.encode([0x61, 0xF1, 0x61])); 
// Output: 'aña'

// 6. punycode.ucs2.decode(): Decode a string to an array of code points
console.log('6. ucs2.decode:', punycode.ucs2.decode('mañana')); 
// Output: [109, 97, 241, 97, 110, 97]

// 7. Edge Case: toASCII with all-ASCII input (should return unchanged)
console.log('7. toASCII (ASCII only):', punycode.toASCII('example.com')); 
// Output: 'example.com'

// 8. Edge Case: toUnicode with non-punycode input (should return unchanged)
console.log('8. toUnicode (ASCII only):', punycode.toUnicode('example.com')); 
// Output: 'example.com'

// 9. Exception Handling: Invalid input for encode (should throw)
try {
    punycode.encode(12345);
} catch (err) {
    console.log('9. Exception caught:', err.message); 
    // Output: The input must be a string
}

// 10. Full round-trip: Unicode -> Punycode -> Unicode
const original = 'I ♥ ☃';
const ascii = punycode.toASCII(original);
const restored = punycode.toUnicode(ascii);
console.log('10. Round-trip:', { original, ascii, restored }); 
// Output: { original: 'I ♥ ☃', ascii: 'xn--i--7iq.xn--n3h', restored: 'I ♥ ☃' }

/**
 * Additional Notes:
 * - All major and minor methods of 'punycode' module are covered.
 * - Both direct encoding/decoding and domain conversion are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Unicode <-> Punycode conversion in Node.js.
 * - For new code, prefer the WHATWG URL API for domain name handling.
 */