/**
 * File: template_literals_tagged_templates.js
 * Description: Comprehensive exploration of Template Literals and Tagged Templates in JavaScript.
 * Author: Top Coder Global #1
 * Date: 2024-06-10
 */

"use strict";

/* =========================================================================
   1. BASIC TEMPLATE LITERALS
   ========================================================================= */

// Single-line interpolation
const user = "Alice";
const age = 30;
const greeting = `Hello, ${user}! You are ${age} years old.`;
console.log(greeting); // Hello, Alice! You are 30 years old.

// Expression interpolation
const a = 5, b = 10;
console.log(`Sum of ${a} and ${b} is ${a + b}.`); // Sum of 5 and 10 is 15.

// Multiline strings
const poem = `
  Roses are red,
  Violets are blue,
  JS template literals,
  Make coding feel new.
`;
console.log(poem);

// Escaping backticks and dollar sign
const tricky = `Use \`backticks\` and \${dollar signs} literally.`;
console.log(tricky);

/* =========================================================================
   2. ADVANCED USAGE: NESTED AND CONDITIONAL EXPRESSIONS
   ========================================================================= */

function renderUserProfile({ name, score }) {
  return `
    Profile:
      Name: ${name}
      Score: ${score}
      Status: ${score >= 50 ? "Pass" : "Fail"}
  `;
}
console.log(renderUserProfile({ name: "Bob", score: 72 }));

// Nested template literal inside interpolation
const items = ["apple", "banana", "cherry"];
const listMarkup = `
  <ul>
    ${items.map(item => `<li>${item}</li>`).join("\n    ")}
  </ul>
`;
console.log(listMarkup);

/* =========================================================================
   3. TAGGED TEMPLATES
   ========================================================================= */

/**
 * A tag function receives:
 *   - strings: Array of literal segments.
 *   - values: Interpolated values.
 * It returns a processed string.
 */
function simpleTag(strings, ...values) {
  // Demonstrate arguments
  console.log("strings:", strings);
  console.log("values: ", values);
  // Reconstruct with uppercase values
  return strings.reduce((result, str, i) => {
    const val = values[i] !== undefined ? String(values[i]).toUpperCase() : "";
    return result + str + val;
  }, "");
}

const result1 = simpleTag`Name: ${user}, Age: ${age}`;
console.log("Tagged output:", result1); // Name: ALICE, Age: 30

/* =========================================================================
   4. RAW STRINGS AND ESCAPING
   ========================================================================= */

function showRaw(strings, ...values) {
  console.log("raw strings:", strings.raw);
  return strings.raw.reduce((acc, str, i) => acc + str + (values[i] || ""), "");
}

// Demonstrate that \n is not processed
const rawResult = showRaw`Line1\nLine2\tEnd`;
console.log("Raw output:", rawResult); // Line1\nLine2\tEnd

/* =========================================================================
   5. SANITIZATION EXAMPLE (SAFE HTML)
   ========================================================================= */

function escapeHTML(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function htmlTag(strings, ...values) {
  return strings.reduce((out, str, i) => {
    const val = values[i] !== undefined ? escapeHTML(String(values[i])) : "";
    return out + str + val;
  }, "");
}

const userInput = `<script>alert("XSS")</script>`;
const safe = htmlTag`User says: ${userInput}`;
console.log(safe); // User says: &lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;

/* =========================================================================
   6. CUSTOM FORMATTING: CURRENCY
   ========================================================================= */

function currencyTag(strings, ...values) {
  return strings.reduce((acc, str, i) => {
    const val = values[i] !== undefined
      ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(values[i])
      : "";
    return acc + str + val;
  }, "");
}

const price = 1234.5;
console.log(currencyTag`Total price: ${price}`); // Total price: $1,234.50

/* =========================================================================
   7. ERROR HANDLING & EXCEPTIONS
   ========================================================================= */

// If you call a tag without template literal, it's invalid:
try {
  simpleTag("Not", "a", "template"); 
} catch (err) {
  console.error("Tagging error:", err.message);
}

// If placeholders exceed supplied values, values[i] becomes undefined
// const strIncomplete = simpleTag`X=${42}, Y=${ }`; // values: [42, ""] 
// console.log(strIncomplete);

/* =========================================================================
   8. DYNAMIC TAG CREATION WITH CLOSURES
   ========================================================================= */

function makeTag(prefix) {
  return (strings, ...values) => {
    const body = strings.reduce((acc, str, i) => acc + str + (values[i] || ""), "");
    return `${prefix}: ${body}`;
  };
}

const infoTag = makeTag("INFO");
console.log(infoTag`Server started at ${new Date().toISOString()}`);

/* =========================================================================
   SUMMARY:
     - `Template Literals`: backtick-delimited, support interpolation, multiline.
     - `Tagged Templates`: custom processing of template literal, receive raw strings and values.
     - Applications: sanitization, formatting, localization, logging, DSLs.
   ========================================================================= */