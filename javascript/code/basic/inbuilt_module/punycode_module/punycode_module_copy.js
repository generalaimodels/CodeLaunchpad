/***************************************************************************************************
 *  Node.js  (built-in, deprecated)  MODULE  :  `punycode`
 *
 *  Author   : <Your Name Here>
 *  File     : punycode-tour.js             –>   run with `node punycode-tour.js`
 *
 *  Mission  : 10 short, self-contained exercises that demonstrate every *public* surface exposed by
 *              the `punycode` module:
 *
 *                 • punycode.encode()         – Unicode → Punycode (single label)
 *                 • punycode.decode()         – Punycode → Unicode
 *                 • punycode.toASCII()        – Unicode domain  → IDNA ASCII (“xn--…” labels)
 *                 • punycode.toUnicode()      – IDNA ASCII      → Unicode domain
 *                 • punycode.ucs2.decode()    – surrogate-aware string → code-point array
 *                 • punycode.ucs2.encode()    – code-point array → string
 *                 • punycode.version          – library version string
 *
 *  Each example prints its own header, expected console output is embedded right above the snippet
 *  so coders can verify their machine matches the intent (values identical except obvious machine
 *  differences such as emoji code-points still valid).
 *
 *  NOTE:  `punycode` was promoted to a *legacy* module since Node v7.0.0 – modern code bases should
 *         prefer `URL`, `Intl`, or the WHATWG `punycode.js` polyfill, but the API is still handy
 *         when you need raw IDNA plumbing without external deps.
 ***************************************************************************************************/
'use strict';
const punycode = require('punycode');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Tiny sequential runner – no dependencies
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const DEMOS = [];
function demo(title, fn) { DEMOS.push({ title, fn }); }
(async () => {
  for (const { title, fn } of DEMOS) {
    console.log('\n' + '═'.repeat(88));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(88));
    await fn();
  }
})();

/***************************************************************************************************
 * 1)  encode() – plain “mañana”  →  “maana-pta”
 **************************************************************************************************/
demo('1) punycode.encode()', () => {
  const src  = 'mañana';                       // Unicode (ñ = U+00F1)
  const p    = punycode.encode(src);
  console.log(`"${src}"  →  "${p}"`);
  /* Expected output:
     "mañana"  →  "maana-pta"
  */
});

/***************************************************************************************************
 * 2)  decode() – round-trip back to Unicode
 **************************************************************************************************/
demo('2) punycode.decode()', () => {
  const label   = 'maana-pta';
  const unicode = punycode.decode(label);
  console.log(`"${label}"  →  "${unicode}"`);
  /* Expected output:
     "maana-pta"  →  "mañana"
  */
});

/***************************************************************************************************
 * 3)  toASCII() – full domain with multiple labels
 **************************************************************************************************/
demo('3) punycode.toASCII() (domain → IDNA)', () => {
  const domain    = 'español.example.com';
  const asciiForm = punycode.toASCII(domain);
  console.log(`"${domain}"  →  "${asciiForm}"`);
  /* Expected output:
     "español.example.com"  →  "xn--espaol-zwa.example.com"
  */
});

/***************************************************************************************************
 * 4)  toUnicode() – reverse conversion (IDNA → Unicode)
 **************************************************************************************************/
demo('4) punycode.toUnicode() (IDNA → domain)', () => {
  const idna  = 'xn--espaol-zwa.example.com';
  const uni   = punycode.toUnicode(idna);
  console.log(`"${idna}"  →  "${uni}"`);
  /* Expected output:
     "xn--espaol-zwa.example.com"  →  "español.example.com"
  */
});

/***************************************************************************************************
 * 5)  Edge-case toASCII(): already ASCII stays untouched (idempotency)
 **************************************************************************************************/
demo('5) toASCII() idempotency check', () => {
  const d1 = 'nodejs.org';
  const d2 = punycode.toASCII(d1);
  console.log('Same reference?', d1 === d2);          // true
  /* Expected output:
     Same reference? true
  */
});

/***************************************************************************************************
 * 6)  ucs2.decode() – break a string with surrogate pairs into raw code points
 **************************************************************************************************/
demo('6) ucs2.decode() (emoji string → code points)', () => {
  const input = '✨🚀';
  const cps   = punycode.ucs2.decode(input);          // [0x2728, 0x1F680]
  console.log(`${JSON.stringify(input)}  → `, cps.map(c => 'U+' + c.toString(16)));
  /* Expected output:
     "✨🚀"  →  [ 'U+2728', 'U+1f680' ]
  */
});

/***************************************************************************************************
 * 7)  ucs2.encode() – build same string back from code points
 **************************************************************************************************/
demo('7) ucs2.encode() (code points → string)', () => {
  const points = [0x2728, 0x1F680];
  const str    = punycode.ucs2.encode(points);
  console.log(points, '→', str);
  /* Expected output:
     [ 10024, 128640 ] → ✨🚀
  */
});

/***************************************************************************************************
 * 8)  encode() multi-label helper – slugify arbitrary string for URLs
 *      (illustrates how encode() only accepts a *single* label, so we must split)
 **************************************************************************************************/
demo('8) encode() helper – slugify for URL path segments', () => {
  function slugify(unicode) {
    return unicode
      .toLowerCase()
      .split(/\s+/)          // words
      .map(w => (/^[\x00-\x7F]+$/.test(w) ? w : punycode.encode(w)))
      .join('-');
  }
  const title = 'Привет мир hello 世界';
  console.log(`Slug  →  ${slugify(title)}`);
  /* Expected output (sample):
     Slug  →  privet-mir-hello--28-oxa2asqa-aoa
     (Exact punycode of Cyrillic & CJK words may vary)
  */
});

/***************************************************************************************************
 * 9)  Version string – quick sanity check
 **************************************************************************************************/
demo('9) punycode.version', () => {
  console.log('punycode version →', punycode.version);
  /* Expected output (Node 20):
     punycode version → 2.3.0
  */
});

/***************************************************************************************************
 * 10)  Robust domain canonicaliser – mix of Unicode, Punycode & ASCII
 *       Demonstrates  toASCII() / toUnicode() interplay + error catching
 **************************************************************************************************/
demo('10) Canonicalise hostnames (error handling)', () => {
  const samples = [
    'mañana.COM',
    'xn--maana-pta.com',
    'sub.пример.рф',
    'bad_domain_§.com'     // illegal char, will throw
  ];

  for (const host of samples) {
    try {
      const ascii = punycode.toASCII(host.toLowerCase());
      const uni   = punycode.toUnicode(ascii);
      console.log(`${host}  →  ASCII: ${ascii}  →  Unicode: ${uni}`);
    } catch (e) {
      console.log(`${host}  →  Error: ${e.message}`);
    }
  }
  /* Expected output (exact punycode will match RFC 3492):
     mañana.COM          →  ASCII: xn--maana-pta.com  →  Unicode: mañana.com
     xn--maana-pta.com   →  ASCII: xn--maana-pta.com  →  Unicode: mañana.com
     sub.пример.рф       →  ASCII: sub.xn--e1afmkfd.xn--p1ai  →  Unicode: sub.пример.рф
     bad_domain_§.com    →  Error: Domain name contains illegal characters
  */
});

/***************************************************************************************************
 *  End of file – happy hacking!
 ***************************************************************************************************/