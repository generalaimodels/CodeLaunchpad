/**
 * Node.js Buffer Module: Comprehensive Examples
 * 
 * This file demonstrates all major and minor methods of the Node.js Buffer class.
 * Each example is self-contained, with clear code, comments, and expected output.
 * 
 * To run: `node <filename>.js`
 */

const { Buffer } = require('buffer');

// 1. Buffer.alloc(size[, fill[, encoding]])
// Allocates a buffer of specified size, optionally filled.
function exampleAlloc() {
    const buf = Buffer.alloc(5, 'a'); // Fills with ASCII 'a' (0x61)
    console.log(buf); // <Buffer 61 61 61 61 61>
    // Expected output: <Buffer 61 61 61 61 61>
}
exampleAlloc();

// 2. Buffer.allocUnsafe(size)
// Allocates a buffer of specified size, memory is not initialized (may contain old data).
function exampleAllocUnsafe() {
    const buf = Buffer.allocUnsafe(5);
    console.log(buf); // <Buffer ...> (contents are uninitialized, may vary)
    // Expected output: <Buffer ...> (random/old memory)
}
exampleAllocUnsafe();

// 3. Buffer.from(array), Buffer.from(string[, encoding]), Buffer.from(buffer)
// Creates a buffer from array, string, or another buffer.
function exampleFrom() {
    const buf1 = Buffer.from([1, 2, 3]);
    console.log(buf1); // <Buffer 01 02 03>
    const buf2 = Buffer.from('hello', 'utf8');
    console.log(buf2); // <Buffer 68 65 6c 6c 6f>
    const buf3 = Buffer.from(buf2);
    console.log(buf3); // <Buffer 68 65 6c 6c 6f>
    // Expected output: <Buffer 01 02 03> <Buffer 68 65 6c 6c 6f> <Buffer 68 65 6c 6c 6f>
}
exampleFrom();

// 4. buf.toString([encoding[, start[, end]]])
// Converts buffer to string with optional encoding and range.
function exampleToString() {
    const buf = Buffer.from('hello world');
    console.log(buf.toString('utf8', 0, 5)); // hello
    // Expected output: hello
}
exampleToString();

// 5. buf.write(string[, offset[, length]][, encoding])
// Writes string to buffer, returns bytes written.
function exampleWrite() {
    const buf = Buffer.alloc(11);
    const bytes = buf.write('hello world', 0, 'utf8');
    console.log(buf.toString()); // hello world
    console.log(bytes); // 11
    // Expected output: hello world, 11
}
exampleWrite();

// 6. buf.slice([start[, end]])
// Returns a new buffer referencing the same memory.
function exampleSlice() {
    const buf = Buffer.from('buffer example');
    const slice = buf.slice(0, 6);
    console.log(slice.toString()); // buffer
    // Expected output: buffer
}
exampleSlice();

// 7. buf.copy(target[, targetStart[, sourceStart[, sourceEnd]]])
// Copies data from one buffer to another.
function exampleCopy() {
    const src = Buffer.from('12345');
    const dest = Buffer.alloc(5);
    src.copy(dest, 0, 1, 4); // Copy '234'
    console.log(dest.toString()); // 234
    // Expected output: 234 (with null bytes at end)
}
exampleCopy();

// 8. Buffer.concat(list[, totalLength])
// Concatenates multiple buffers.
function exampleConcat() {
    const buf1 = Buffer.from('Hello, ');
    const buf2 = Buffer.from('World!');
    const buf3 = Buffer.concat([buf1, buf2]);
    console.log(buf3.toString()); // Hello, World!
    // Expected output: Hello, World!
}
exampleConcat();

// 9. buf.equals(otherBuffer)
// Checks if two buffers have the same bytes.
function exampleEquals() {
    const buf1 = Buffer.from('abc');
    const buf2 = Buffer.from('abc');
    const buf3 = Buffer.from('def');
    console.log(buf1.equals(buf2)); // true
    console.log(buf1.equals(buf3)); // false
    // Expected output: true, false
}
exampleEquals();

// 10. buf.compare(target[, targetStart[, targetEnd[, sourceStart[, sourceEnd]]]])
// Compares two buffers, returns -1, 0, or 1.
function exampleCompare() {
    const buf1 = Buffer.from('123');
    const buf2 = Buffer.from('124');
    console.log(buf1.compare(buf2)); // -1 (buf1 < buf2)
    console.log(buf2.compare(buf1)); // 1  (buf2 > buf1)
    console.log(buf1.compare(Buffer.from('123'))); // 0 (equal)
    // Expected output: -1, 1, 0
}
exampleCompare();

// 11. buf.fill(value[, offset[, end]][, encoding])
// Fills buffer with specified value.
function exampleFill() {
    const buf = Buffer.alloc(5);
    buf.fill('A');
    console.log(buf.toString()); // AAAAA
    // Expected output: AAAAA
}
exampleFill();

// 12. buf.includes(value[, byteOffset][, encoding])
// Checks if buffer includes a value.
function exampleIncludes() {
    const buf = Buffer.from('nodejs');
    console.log(buf.includes('js')); // true
    console.log(buf.includes('python')); // false
    // Expected output: true, false
}
exampleIncludes();

// 13. buf.indexOf(value[, byteOffset][, encoding])
// Returns first index of value, or -1 if not found.
function exampleIndexOf() {
    const buf = Buffer.from('banana');
    console.log(buf.indexOf('a')); // 1
    console.log(buf.indexOf('z')); // -1
    // Expected output: 1, -1
}
exampleIndexOf();

// 14. buf.lastIndexOf(value[, byteOffset][, encoding])
// Returns last index of value, or -1 if not found.
function exampleLastIndexOf() {
    const buf = Buffer.from('banana');
    console.log(buf.lastIndexOf('a')); // 5
    // Expected output: 5
}
exampleLastIndexOf();

// 15. buf.readUInt8(offset)
// Reads unsigned 8-bit integer at offset.
function exampleReadUInt8() {
    const buf = Buffer.from([0x10, 0x20, 0x30]);
    console.log(buf.readUInt8(1)); // 32
    // Expected output: 32
}
exampleReadUInt8();

// 16. buf.writeUInt8(value, offset)
// Writes unsigned 8-bit integer at offset.
function exampleWriteUInt8() {
    const buf = Buffer.alloc(3);
    buf.writeUInt8(255, 1);
    console.log(buf); // <Buffer 00 ff 00>
    // Expected output: <Buffer 00 ff 00>
}
exampleWriteUInt8();

// 17. buf.byteLength
// Returns the length of the buffer in bytes.
function exampleByteLength() {
    const buf = Buffer.from('hello');
    console.log(buf.byteLength); // 5
    // Expected output: 5
}
exampleByteLength();

// 18. Buffer.isBuffer(obj)
// Checks if the object is a Buffer.
function exampleIsBuffer() {
    const buf = Buffer.from('test');
    console.log(Buffer.isBuffer(buf)); // true
    console.log(Buffer.isBuffer({})); // false
    // Expected output: true, false
}
exampleIsBuffer();

// 19. Buffer.isEncoding(encoding)
// Checks if encoding is a valid Buffer encoding.
function exampleIsEncoding() {
    console.log(Buffer.isEncoding('utf8')); // true
    console.log(Buffer.isEncoding('foo')); // false
    // Expected output: true, false
}
exampleIsEncoding();

// 20. buf.toJSON()
// Returns a JSON representation of the buffer.
function exampleToJSON() {
    const buf = Buffer.from([1, 2, 3]);
    console.log(JSON.stringify(buf)); // {"type":"Buffer","data":[1,2,3]}
    // Expected output: {"type":"Buffer","data":[1,2,3]}
}
exampleToJSON();

/**
 * Summary:
 * - All major and minor Buffer methods are covered.
 * - Each example is self-contained and demonstrates expected behavior.
 * - Uncomment lines or modify values to see different results.
 */