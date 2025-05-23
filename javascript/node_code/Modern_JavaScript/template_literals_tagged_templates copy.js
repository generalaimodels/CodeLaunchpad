/**************************************************************************
 *  TEMPLATE LITERALS & TAGGED TEMPLATES – COMPLETE REFERENCE IN ONE FILE *
 *  Author: <your‑name> (Top‑Ranked Coder)                                 *
 *  Language: ECMAScript 2023                                             *
 **************************************************************************/

/*──────────────────────────────────────────────────────────────────────────
  1. BASIC TEMPLATE LITERALS
──────────────────────────────────────────────────────────────────────────*/

const name = 'Ada';
const language = 'JavaScript';

// 1.1  String interpolation (automatic toString())
const greet = `Hello, ${name}! Welcome to ${language}.`;
console.log(greet); // → Hello, Ada! Welcome to JavaScript.

// 1.2  Multi‑line string (no \n, preserves formatting)
const multiLine = `
Line 1
    Line 2 (indented with 4 spaces)
Line 3`;
console.log(multiLine);

// 1.3  Embed any expression
const a = 5, b = 7;
console.log(`Sum = ${a + b}, Random = ${Math.random().toFixed(2)}`);

// 1.4  Back‑tick escape (use \`)
console.log(`Back‑tick inside template: \``);

/*──────────────────────────────────────────────────────────────────────────
  2. NESTING, CONDITIONALS & FUNCTIONS
──────────────────────────────────────────────────────────────────────────*/

function upper(s) { return s.toUpperCase(); }

const condition = true;
const nested = `Status: ${
  condition ? `✅  ${upper('Success')}` : `❌  ${upper('Failure')}`
}`;
console.log(nested);

/*──────────────────────────────────────────────────────────────────────────
  3. TAGGED TEMPLATES – MECHANICS
──────────────────────────────────────────────────────────────────────────*/

/*
   Signature:
       tag(strings: TemplateStringsArray, ...substitutions: any[]) : any
   The engine calls tag() with:
     – strings: an array of literal sections
     – substitutions: runtime values between literals
*/

function debugTag(strings, ...values) {
  console.log('strings:', strings);          // Raw literal chunks
  console.log('values :', values);           // Interpolated values
  return strings.reduce((out, str, i) =>
           out + str + (values[i] ?? ''), '');
}

const tagged = debugTag`User ${name} likes ${language}.`;
console.log('Reconstructed:', tagged);

/*──────────────────────────────────────────────────────────────────────────
  4. BUILT‑IN TAG: String.raw (shows backslash escapes literally)
──────────────────────────────────────────────────────────────────────────*/

console.log(String.raw`C:\Users\${name}\nNewFolder`);
// → C:\Users\Ada\nNewFolder  (note: \n not converted)

/*──────────────────────────────────────────────────────────────────────────
  5. PRACTICAL TAGGED‑TEMPLATE USE‑CASES
──────────────────────────────────────────────────────────────────────────*/

/* 5.1 HTML escaping to mitigate XSS */
function safeHTML(strings, ...values) {
  const escape = (s) => String(s)
     .replace(/&/g, '&amp;').replace(/</g, '&lt;')
     .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
     .replace(/'/g, '&#39;');
  return strings.reduce((out, str, i) =>
           out + str + (i < values.length ? escape(values[i]) : ''), '');
}
const userInput = '<script>alert("XSS")</script>';
console.log(safeHTML`<p>User says: ${userInput}</p>`);

/* 5.2  SQL‑builder (auto‑parameterization) */
function sql(strings, ...vals) {
  const text  = strings.reduce((sql, part, i) => sql + part + (i < vals.length ? `$${i+1}` : ''), '');
  return { text, params: vals };
}
const minAge = 18, city = 'Paris';
const query = sql`SELECT * FROM users WHERE age >= ${minAge} AND city = ${city}`;
console.log(query); // {text:"SELECT * … age >= $1 AND city = $2", params:[18,"Paris"]}

/* 5.3  i18n / Localization */
const dict = { en:{greet:'Hello'}, fr:{greet:'Bonjour'} };
function i18n(lang) {
  return (strings, ...vals) => strings.reduce(
    (out, str, i) => out + str.replace(/%\w+%/g, m => dict[lang][m.slice(1,-1)]) +
                     (vals[i] ?? ''), '');
}
console.log(i18n('fr')`%greet% ${name}!`); // → Bonjour Ada!

/*──────────────────────────────────────────────────────────────────────────
  6. RAW TEMPLATES INSIDE CUSTOM TAG
──────────────────────────────────────────────────────────────────────────*/

function logRaw(strings, ...vals) {
  console.log('raw:', strings.raw); // unprocessed (includes \n, \t, etc.)
  return strings.raw.join('|');
}
console.log(logRaw`Line1\nLine2`);

/*──────────────────────────────────────────────────────────────────────────
  7. EXCEPTIONS & EDGE CASES
──────────────────────────────────────────────────────────────────────────*/

try {
  const notAFunction = 42;
  notAFunction`oops`;      // Engine tries to call 42 → TypeError
} catch (e) {
  console.error('Expected error:', e.message);
}

// Undefined interpolation returns "undefined" string
console.log(`Value: ${undefined}`);          // → "Value: undefined"

// Template literal ENFORCES actual string coercion
const sym = Symbol('id');
try {
  `${sym}`;                                 // Throws TypeError (cannot convert Symbol)
} catch (e) {
  console.error('Symbol interpolation error:', e.message);
}

/*──────────────────────────────────────────────────────────────────────────
  8. PERFORMANCE NOTES (optional micro‑benchmark)
──────────────────────────────────────────────────────────────────────────*/
function bench(label, fn) {
  const t0 = performance.now();
  fn();
  console.log(label, (performance.now() - t0).toFixed(2), 'ms');
}
bench('1e6 simple concatenations', () => {
  let s = '';
  for (let i = 0; i < 1e6; ++i) s = s + 'a';
});
bench('1e6 template literals', () => {
  let s = '';
  for (let i = 0; i < 1e6; ++i) s = `${s}a`;
});

/*──────────────────────────────────────────────────────────────────────────
  9. SUMMARY (console output acts as runnable documentation)
──────────────────────────────────────────────────────────────────────────*/
// All theory demonstrated live above. Run node, Deno, or browser console.
// End of file.