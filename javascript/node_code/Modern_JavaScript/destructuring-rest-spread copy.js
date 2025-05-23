/**
 * ES6+ DESTRUCTURING  &  REST/SPREAD – FULL‑SPECTRUM GUIDE
 * =======================================================
 * 2.1 Object / Array Destructuring
 * 2.2 Rest parameters            ( ...rest in function params )
 * 2.3 Spread operator            ( ...spread in literals / calls )
 *
 * Run via:  node destructuring_rest_spread.js
 * -------------------------------------------------------
 * Each topic is exhaustively demonstrated, covers quirks,
 * caveats, and self‑verifies with assertions or explicit
 * error handling.  No external dependencies.
 */

'use strict';

/* ──────────────────────────────────────────────────────────
 * 2.1 OBJECT  &  ARRAY  DESTRUCTURING
 * ────────────────────────────────────────────────────────── */
(() => {
  console.log('\n=== 2.1 Object & Array Destructuring ===========================');

  /* 2.1.1 BASIC ARRAY DESTRUCTURING --------------------------------- */
  const arr = [10, 20, 30];
  const [a1, a2, a3] = arr;
  console.assert(a1 === 10 && a3 === 30);

  /* Skipping elements + default values */
  const [ , , third = 'default', fourth = 40 ] = [ , , 3 ];
  console.assert(third === 3 && fourth === 40);

  /* 2.1.2 BASIC OBJECT DESTRUCTURING -------------------------------- */
  const obj = {x: 1, y: 2, z: 3};
  const {x, z} = obj;
  console.assert(x === 1 && z === 3);

  /* Renaming & default */
  const {missing: m = 'fallback'} = obj;
  console.assert(m === 'fallback');

  /* 2.1.3 NESTED & DEEP DESTRUCTURING -------------------------------- */
  const deep = {
    id: 7,
    meta: {
      tags: ['js', 'es6'],
      coords: {lat: 42, lng: 13}
    }
  };
  const {
    meta: {
      tags: [firstTag],
      coords: {lat}
    }
  } = deep;
  console.assert(firstTag === 'js' && lat === 42);

  /* 2.1.4 REST PATTERN IN DESTRUCTURING ------------------------------ */
  const [head, ...tail] = ['h', 't1', 't2'];
  console.assert(head === 'h' && tail.join() === 't1,t2');

  const {x: keepX, ...others} = {x: 10, y: 20, z: 30};
  console.assert(keepX === 10 && others.y === 20 && !('x' in others));

  /* 2.1.5 FUNCTION PARAMETER DESTRUCTURING --------------------------- */
  const toStr = ({id, title = 'Untitled'}) => `#${id}:${title}`;
  console.assert(toStr({id: 5, title: 'ES6'}) === '#5:ES6');

  /* 2.1.6 ASSIGN TO EXISTING VARIABLES ------------------------------- */
  let p, q;
  ({p, q} = {p: 1, q: 2});
  console.assert(p === 1 && q === 2);

  /* 2.1.7 DESTRUCTURING FAILS ON null / undefined -------------------- */
  try {
    const {foo} = null;                 // TypeError
  } catch (e) {
    console.log('Expected TypeError (null destructuring):', e.message);
  }

  /* 2.1.8 DEFAULT VALUE LAZINESS & ORDER ----------------------------- */
  let counter = 0;
  function incr() { counter++; return 100; }
  const [val = incr()] = [undefined];    // default invoked
  const [noCall = incr()] = [10];        // default skipped
  console.assert(counter === 1 && val === 100 && noCall === 10);

  /* 2.1.9 HOLES & UNDEFINED IN ARRAY DESTRUCTURING ------------------- */
  const [hole1] = [ , 99]; // leading hole
  console.assert(hole1 === undefined);

  /* 2.1.10 COMPUTED PROPERTY NAMES ----------------------------------- */
  const key = 'dynamic';
  const {[key]: dynVal} = {dynamic: 321};
  console.assert(dynVal === 321);
})();

/* ──────────────────────────────────────────────────────────
 * 2.2 REST PARAMETERS
 * ────────────────────────────────────────────────────────── */
(() => {
  console.log('\n=== 2.2 Rest Parameters =======================================');

  /* 2.2.1 BASIC USAGE ------------------------------------------------- */
  function sum(...nums) { return nums.reduce((a,b)=>a+b,0); }
  console.assert(sum(1,2,3) === 6 && sum() === 0);

  /* 2.2.2 REST AFTER FORMAL PARAMETERS ------------------------------- */
  const prepend = (prefix, ...strings) => strings.map(s => prefix + s);
  console.assert(prepend('>', 'a','b').join() === '>a,>b');

  /* 2.2.3 REST PARAMETER IS A TRUE ARRAY ----------------------------- */
  (function (...args) {
    console.assert(Array.isArray(args) && args.pop() === 3);
  })(1,2,3);

  /* 2.2.4 CANNOT HAVE MULTIPLE REST PARAMETERS ----------------------- */
  try { eval('function bad(...a, ...b){}'); }
  catch (e) { console.log('Expected SyntaxError (multiple rest):', e.message); }

  /* 2.2.5 PARAMS AFTER REST DISALLOWED ------------------------------- */
  try { eval('function bad(...a, b){}'); }
  catch (e) { console.log('Expected SyntaxError (param after rest):', e.message); }
})();

/* ──────────────────────────────────────────────────────────
 * 2.3 SPREAD OPERATOR
 * ────────────────────────────────────────────────────────── */
(() => {
  console.log('\n=== 2.3 Spread Operator =======================================');

  /* 2.3.1 CALL SPREAD ------------------------------------------------- */
  const maxVal = Math.max(...[3, 7, 5]);
  console.assert(maxVal === 7);

  /* 2.3.2 ARRAY LITERALS --------------------------------------------- */
  const merged = [0, ...['a','b'], 3];
  console.assert(merged.join() === '0,a,b,3');

  /* Shallow copy caveat */
  const arr1 = [[1], [2]];
  const arr2 = [...arr1];
  arr2[0][0] = 99;
  console.assert(arr1[0][0] === 99, 'spread is shallow');

  /* 2.3.3 OBJECT LITERALS (ES2018) ----------------------------------- */
  const o1 = {x:1, nested:{d:4}};
  const o2 = {...o1, y:2};
  o2.nested.d = 99;
  console.assert(o1.nested.d === 99, 'object spread is shallow');

  /* Property precedence */
  const clash = {a:1, ...{a:2}};
  console.assert(clash.a === 2);

  /* 2.3.4 SPREAD NON‑ITERABLE → TYPE ERROR IN ARRAY CONTEXT ---------- */
  try { const bad = [...123]; }
  catch (e) { console.log('Expected TypeError (spread non‑iterable):', e.message); }

  /* 2.3.5 SPREAD WITH STRINGS → CHARACTER ARRAY ---------------------- */
  const chars = [...'🔥ES6'];
  console.assert(chars.length === 3 && chars[0] === '🔥');

  /* 2.3.6 USING SPREAD FOR IMMUTABLE UPDATES ------------------------- */
  const state = {count:0};
  const next = {...state, count: state.count + 1};
  console.assert(next.count === 1 && state.count === 0);
})();

/* ──────────────────────────────────────────────────────────
 * 2.4 INTERPLAY – DESTRUCTURING  ×  REST/SPREAD
 * ────────────────────────────────────────────────────────── */
(() => {
  console.log('\n=== 2.4 Interplay (Rest Pattern & Spread) =====================');

  /* 2.4.1 CLONE‑MODIFY PATTERN --------------------------------------- */
  const user = {id:1, role:'user', prefs:{theme:'dark'}};
  const admin = {...user, role:'admin'};        // spread
  console.assert(admin.role === 'admin' && user.role === 'user');

  /* 2.4.2 OMISSION PATTERN ------------------------------------------- */
  const {role, ...stripped} = admin;            // rest pattern
  console.assert(!('role' in stripped) && stripped.id === 1);

  /* 2.4.3 RECOMPOSE AFTER PARTIALS ----------------------------------- */
  const full = {...stripped, role};
  console.assert(JSON.stringify(full) === JSON.stringify(admin));

  /* 2.4.4 CURRY‑LIKE GATHER & SPREAD --------------------------------- */
  const logArgs = (...args) => console.log('log:', ...args);
  logArgs('a', 'b', 'c'); // rest gathers, spread prints

  /* 2.4.5 ARRAY SLICING WITHOUT MUTATION ----------------------------- */
  const orig = [1,2,3,4,5];
  const [, , ...tail] = orig;            // rest
  const clone = [...tail];               // spread
  console.assert(clone.join() === '3,4,5');
})();

/* EOF – Single file exhaustively covering Destructuring & Rest/Spread */