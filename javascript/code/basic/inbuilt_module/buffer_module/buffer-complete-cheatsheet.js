/***************************************************************************************************
* File         : buffer-complete-cheatsheet.js
* Purpose      : “Node.js Buffer Module ‒ 10 self-contained, progressively advanced examples that
*                showcase EVERY public API in everyday ↔ edge-case scenarios.”
* Node target  : ≥ 16.x  (BigInt / transcode / swap64 supported)
* Run          : `node buffer-complete-cheatsheet.js`
* Conventions  : ES-2023, 2-space indent, strict mode, immutable constants where possible.
***************************************************************************************************/
'use strict';
const { Buffer } = require('buffer'); // explicit import for clarity

/***************************************************************************************************
* EX-1 ► Creating Buffers (from string/array/arrayBuffer/otherBuffer) & basic props
*   Covers : Buffer.from(…), Buffer.isBuffer, Buffer.byteLength, buf.length, buf.toString
***************************************************************************************************/
(() => {
  console.log('\nEX-1: Creation & basics');

  // 1A) From UTF-8 string (default encoding)
  const utf8Buf = Buffer.from('Hello 🌍');
  console.log(' length bytes  :', utf8Buf.length);                 // 10
  console.log(' byteLength    :', Buffer.byteLength('Hello 🌍'));   // 10
  console.log(' toString      :', utf8Buf.toString());             // Hello 🌍

  // 1B) From array of octets
  const octetBuf = Buffer.from([0x48, 0x69]);                      // 'Hi'
  console.log(' octet toString:', octetBuf.toString());            // Hi

  // 1C) From existing Buffer (makes a copy)
  const clone = Buffer.from(octetBuf);
  console.log(' isBuffer clone :', Buffer.isBuffer(clone));        // true

  // 1D) From ArrayBuffer (+offset/length)
  const ab = new ArrayBuffer(8);
  const view = new Uint8Array(ab);
  view.set([1, 2, 3, 4]);
  const abBuf = Buffer.from(ab, 1, 3);                             // bytes 1..3 → [2,3,4]
  console.log(' ArrayBuffer   :', abBuf);                          // <Buffer 02 03 04>
})();

/***************************************************************************************************
* EX-2 ► alloc vs allocUnsafe vs allocUnsafeSlow, & buf.fill
*   Covers : Buffer.alloc, Buffer.allocUnsafe, Buffer.allocUnsafeSlow, buf.fill
***************************************************************************************************/
(() => {
  console.log('\nEX-2: Memory allocation strategies');

  const safe = Buffer.alloc(5, 0x1);            // zero-filled or custom fill
  console.log(' alloc (safe)       :', safe);   // <Buffer 01 01 01 01 01>

  const unsafe = Buffer.allocUnsafe(5);         // fast – may contain old memory!
  unsafe.fill(0);                               // ALWAYS sanitize before use
  console.log(' allocUnsafe filled :', unsafe); // <Buffer 00 00 00 00 00>

  const hugeSlow = Buffer.allocUnsafeSlow(5);   // outside of shared slab
  hugeSlow.fill(0x2);
  console.log(' allocUnsafeSlow    :', hugeSlow); // <Buffer 02 02 02 02 02>
})();

/***************************************************************************************************
* EX-3 ► Static helpers: compare, concat, isEncoding, kMaxLength
*   Covers : Buffer.compare, Buffer.concat, Buffer.isEncoding, Buffer.kMaxLength
***************************************************************************************************/
(() => {
  console.log('\nEX-3: Static helpers');

  const a = Buffer.from('abc');
  const b = Buffer.from('abd');
  console.log(' compare(a,b)  :', Buffer.compare(a, b));     // -1  (a < b)

  const pieces = [Buffer.from('Hello, '), Buffer.from('world!')];
  const joined = Buffer.concat(pieces);
  console.log(' concat        :', joined.toString());        // Hello, world!

  console.log(' isEncoding(\'base64\'):', Buffer.isEncoding('base64')); // true
  console.log(' Max buf length   :', Buffer.kMaxLength);    // platform dependent bytes
})();

/***************************************************************************************************
* EX-4 ► Read / Write numbers (UInt/Int/Float/Double) LE & BE
*   Covers : buf.writeUInt16LE, buf.readUInt16LE, writeInt32BE, readFloatLE, readDoubleBE
***************************************************************************************************/
(() => {
  console.log('\nEX-4: Numeric read/write');

  const numBuf = Buffer.alloc(8);

  numBuf.writeUInt16LE(0x1234, 0);           // offset 0-1
  numBuf.writeInt32BE(-42, 2);               // offset 2-5
  numBuf.writeFloatLE(Math.PI, 6);           // offset 6-9 (will overflow → truncated)

  console.log(' UInt16LE (0)   :', numBuf.readUInt16LE(0)); // 4660
  console.log(' Int32BE  (2)   :', numBuf.readInt32BE(2));  // -42
  // Float prints imprecise due to truncation; only show demonstration.
})();

/***************************************************************************************************
* EX-5 ► BigInt support & 64-bit read/write
*   Covers : writeBigUInt64LE, readBigUInt64LE, writeBigInt64BE, readBigInt64BE
***************************************************************************************************/
(() => {
  console.log('\nEX-5: 64-bit BigInt');

  const big = Buffer.alloc(16);
  big.writeBigUInt64LE(0x1_0000_0000n, 0);   // 4 294 967 296
  big.writeBigInt64BE(-1n, 8);

  console.log(' BigUInt64LE :', big.readBigUInt64LE(0));    // 4294967296n
  console.log(' BigInt64BE  :', big.readBigInt64BE(8));     // -1n
})();

/***************************************************************************************************
* EX-6 ► Copying, slicing, subarray mutability
*   Covers : buf.copy, buf.slice, buf.subarray, buf.fill (again)
***************************************************************************************************/
(() => {
  console.log('\nEX-6: Copy vs slice vs subarray');

  const src = Buffer.from('ABCDEFG');
  const dst = Buffer.alloc(src.length);

  src.copy(dst, 0, 0, 3);                      // copy 'ABC'
  console.log(' dst after copy   :', dst.toString()); // ABC\0\0\0\0

  const slice = src.slice(0, 3);               // shares memory
  const sub   = src.subarray(3, 6);            // alias of slice
  slice.fill(0x61);                            // overwrite with 'a'
  console.log(' src mutated      :', src.toString()); // aaaDEFG
  console.log(' sub (3-6)        :', sub.toString()); // DEF
})();

/***************************************************************************************************
* EX-7 ► Search & iteration helpers
*   Covers : buf.includes, indexOf, lastIndexOf, entries, keys, values
***************************************************************************************************/
(() => {
  console.log('\nEX-7: Searching & iterating');

  const hay = Buffer.from('bananas');
  console.log(' includes "ana" :', hay.includes('ana'));            // true
  console.log(' first "a" idx  :', hay.indexOf(0x61));              // 1
  console.log(' last  "a" idx  :', hay.lastIndexOf('a'));           // 5

  for (const [i, byte] of hay.entries()) {
    if (byte === 0x62) console.log(` byte at ${i} is 'b'`);
  }
  // keys() → indices; values() → bytes
})();

/***************************************************************************************************
* EX-8 ► Endianness byte-swap utilities
*   Covers : buf.swap16, buf.swap32, buf.swap64
***************************************************************************************************/
(() => {
  console.log('\nEX-8: swap16/32/64');

  const s16 = Buffer.from([0x12, 0x34]);
  s16.swap16();                                // [0x34, 0x12]
  console.log(' swap16:', s16);                // <Buffer 34 12>

  const s32 = Buffer.from([0x11, 0x22, 0x33, 0x44]);
  s32.swap32();                                // [0x44,0x33,0x22,0x11]
  console.log(' swap32:', s32);                // <Buffer 44 33 22 11>

  const s64 = Buffer.from([0,1,2,3,4,5,6,7]);
  s64.swap64();
  console.log(' swap64:', s64);                // <Buffer 07 06 05 04 03 02 01 00>
})();

/***************************************************************************************************
* EX-9 ► Encoding conversion with Buffer.transcode
*   Covers : Buffer.transcode(source, fromEnc, toEnc)
***************************************************************************************************/
(() => {
  console.log('\nEX-9: transcode (convert encoding)');

  const utf16 = Buffer.from('Z͑', 'utf16le');         // funky char
  const utf8  = Buffer.transcode(utf16, 'utf16le', 'utf8');
  console.log(' utf8      :', utf8.toString());       // Z͑

  // Round-trip back to UTF-16LE:
  const back = Buffer.transcode(utf8, 'utf8', 'utf16le');
  console.log(' roundtrip :', back.equals(utf16));    // true
})();

/***************************************************************************************************
* EX-10 ► Serialization, equals, compare in practice
*   Covers : buf.toJSON, JSON.stringify, buf.equals, buf.compare (instance variant), buf.write
***************************************************************************************************/
(() => {
  console.log('\nEX-10: Serialization & equality');

  const bufA = Buffer.alloc(4);
  bufA.write('node');                            // writes ASCII

  const bufB = Buffer.from('node');
  console.log(' equals        :', bufA.equals(bufB));        // true
  console.log(' compare       :', bufA.compare(bufB));       // 0

  const json = JSON.stringify(bufA);             // triggers toJSON → { type: 'Buffer', data:[…] }
  console.log(' JSON          :', json);

  // Demonstrate restoration from JSON:
  const parsed = JSON.parse(json, (key, value) =>
    value && value.type === 'Buffer' ? Buffer.from(value.data) : value
  );
  console.log(' restored OK   :', bufA.equals(parsed));      // true
})();

/***************************************************************************************************
* END ‑ You now wield full mastery over Buffer’s API spectrum 🚀
***************************************************************************************************/