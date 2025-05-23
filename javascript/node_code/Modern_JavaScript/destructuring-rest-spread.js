/**
 * File: destructuring-rest-spread.js
 * Description: Comprehensive coverage of:
 *   1. Array / Object Destructuring
 *   2. Rest Parameters & Spread Operator
 * All code in one file. No extraneous content.
 */

/* ============================================================================
 * 1. Array Destructuring
 * ============================================================================ */
// Basic extraction
const rgb = [255, 200, 100];
const [red, green, blue] = rgb;
console.log({ red, green, blue }); // {red:255,green:200,blue:100}

// Skipping elements & defaults
const scores = [90, , 85];
const [first = 0, second = 50, third = 0] = scores;
console.log({ first, second, third }); // {first:90,second:50,third:85}

// Rest element
const letters = ['a', 'b', 'c', 'd'];
const [head, ...tail] = letters;
console.log(head, tail); // a [ 'b', 'c', 'd' ]

// Nested destructuring
const matrix = [[1,2], [3,4]];
const [[a11, a12], [a21, a22]] = matrix;
console.log({ a11, a12, a21, a22 });

// Swapping variables without temp
let x = 1, y = 2;
[x, y] = [y, x];
console.log({ x, y }); // x:2, y:1

// Exception: cannot destructure null/undefined
try {
  const { foo } = null;
} catch (e) {
  console.log('Error:', e.name); // TypeError
}

/* ============================================================================
 * 2. Object Destructuring
 * ============================================================================ */
const user = {
  id: 42,
  name: 'Alice',
  address: {
    city: 'Wonderland',
    zip: '12345'
  }
};

// Basic extraction
const { id, name } = user;
console.log({ id, name });

// Renaming & defaults
const { name: userName, age = 30 } = user;
console.log({ userName, age });

// Nested destructuring
const {
  address: { city, zip: postalCode }
} = user;
console.log({ city, postalCode });

// Rest properties
const { address, ...metadata } = user;
console.log({ address, metadata });

// Exception: destructuring missing object
try {
  const { foo } = undefined;
} catch (e) {
  console.log('Error:', e.name); // TypeError
}

/* ============================================================================
 * 3. Rest Parameters
 * ============================================================================ */
// Function with fixed and rest args
function join(separator, ...items) {
  return items.join(separator);
}
console.log(join('-', 'a', 'b', 'c')); // "a-b-c"

// Rest must be last parameter; only one allowed
// function bad(...a, b) {} // SyntaxError

// Use case: sum arbitrary numbers
const sum = (...nums) => nums.reduce((s, n) => s + n, 0);
console.log(sum(1, 2, 3, 4)); // 10

/* ============================================================================
 * 4. Spread Operator
 * ============================================================================ */
// --- Arrays ---
// Copying (shallow)
const arr1 = [1, 2, { v: 3 }];
const arr2 = [...arr1];
arr2[0] = 100;
arr2[2].v = 999;
console.log(arr1, arr2); 
// arr1: [1,2,{v:999}] arr2: [100,2,{v:999}]

// Concatenation
const a = [0];
const b = [1,2,3];
const c = [...a, ...b];
console.log(c); // [0,1,2,3]

// Function invocation
function max(...nums) {
  return Math.max(...nums);
}
console.log(max(...[5,10,3])); // 10

// Exception: spread on non-iterable
try {
  const bad = [...123];
} catch (e) {
  console.log('Error:', e.name); // TypeError
}

// --- Objects ---
// Shallow copy & merge
const defaults = { a: 1, b: 2 };
const overrides = { b: 20, c: 30 };
const merged = { ...defaults, ...overrides };
console.log(merged); // {a:1, b:20, c:30}

// Adding properties
const extended = { ...user, isActive: true };
console.log(extended);

// Exception: preserving prototype chain
const proto = { hello() { return 'hi'; } };
const objClone = { ...Object.create(proto), x: 1 };
console.log(typeof objClone.hello); // undefined

// Note: spread in objects is a shallow copy; nested objects remain by reference

/* ============================================================================
 * End of file
 * ============================================================================ */