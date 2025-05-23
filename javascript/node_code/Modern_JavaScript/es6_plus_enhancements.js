/**
 * ES6+ LANGUAGE ENHANCEMENTS – COMPREHENSIVE WALK‑THROUGH
 * =======================================================
 * 1. let / const  vs.  var
 * 2. Arrow functions & lexical `this`
 *
 * Execute via:  node es6_plus_enhancements.js
 * -------------------------------------------------------
 * Every concept is self‑contained, thoroughly commented,
 * showcases corner‑cases, and self‑verifies via assertions
 * or explicit error handling where appropriate.
 */

'use strict';

/* ──────────────────────────────
 * SECTION 1 ─ let / const vs var
 * ────────────────────────────── */
(() => {
  console.log('\n=== 1. let / const vs var ======================================');

  /*
   * 1.1 SCOPE
   * --------------------------------------------------------
   * var   – function‑scoped (or globally scoped when declared at top level)
   * let   – block‑scoped
   * const – block‑scoped (additionally: binding is read‑only)
   */

  function scopeDemo() {
    if (true) {
      var  x = 'varScoped';   // function scope
      let  y = 'letScoped';   // block scope
      const z = 'constScoped';// block scope
    }
    console.assert(x === 'varScoped',   'var should be visible outside block');
    try {
      // Both lines throw ReferenceError (y, z not defined here)
      console.log(y, z);
    } catch (e) {
      console.log('Expected Error (block scope):', e.message);
    }
  }
  scopeDemo();

  /*
   * 1.2 HOISTING & TEMPORAL DEAD ZONE (TDZ)
   * --------------------------------------------------------
   * - var declarations are hoisted + initialized to `undefined`.
   * - let/const declarations are hoisted BUT NOT initialized;
   *   accessing them before the declaration triggers a ReferenceError.
   */
  function hoistDemo() {
    console.assert(a === undefined, 'var a should be hoisted → undefined');
    var a = 10;

    try {
      console.log(b); // TDZ for `let`
    } catch (e) {
      console.log('Expected TDZ Error for let:', e.message);
    }
    let b = 20;

    try {
      console.log(c); // TDZ for `const`
    } catch (e) {
      console.log('Expected TDZ Error for const:', e.message);
    }
    const c = 30;
  }
  hoistDemo();

  /*
   * 1.3 REDECLARATION RULES
   * --------------------------------------------------------
   * - var can be redeclared within the same scope.
   * - let/const cannot be redeclared in the same scope.
   */
  function redeclarationDemo() {
    var v = 1;
    var v = 2; // OK

    let l = 1;
    try {
      let l = 2; // Allowed (shadowing in nested block)
    } catch(_) {}

    try {
      eval('let l = 3;'); // Same scope → error
    } catch (e) {
      console.log('Expected Redeclaration Error for let:', e.message);
    }

    const c1 = 1;
    try {
      const c1 = 2; // Same scope → error
    } catch (e) {
      console.log('Expected Redeclaration Error for const:', e.message);
    }
  }
  redeclarationDemo();

  /*
   * 1.4 CONST ≠ IMMUTABLE OBJECT
   * --------------------------------------------------------
   * const protects the binding (reference) – not the value.
   */
  function constMutationDemo() {
    const obj = {mutable: true};
    obj.mutable = false;              // ✅ allowed
    console.assert(obj.mutable === false);

    try {
      // Attempt to reassign the binding
      obj = {}; // ❌ TypeError
    } catch (e) {
      console.log('Expected Reassignment Error for const:', e.message);
    }
  }
  constMutationDemo();

  /*
   * 1.5 LOOPING WITH let (CLOSURE SAFETY)
   * --------------------------------------------------------
   * Using `let` in for‑loops provides a new binding per iteration,
   * fixing the classic closure‑in‑loop problem seen with `var`.
   */
  function closureLoopDemo() {
    var arrVar = [], arrLet = [];
    for (var i = 0; i < 3; ++i) {
      arrVar.push(() => i);
    }
    for (let j = 0; j < 3; ++j) {
      arrLet.push(() => j);
    }
    console.assert(arrVar.map(f => f()).join() === '3,3,3',
      '`var` → all functions capture final value');
    console.assert(arrLet.map(f => f()).join() === '0,1,2',
      '`let` → each function captures its own iteration value');
  }
  closureLoopDemo();
})();

/* ─────────────────────────────────────
 * SECTION 2 ─ Arrow Functions & `this`
 * ───────────────────────────────────── */
(() => {
  console.log('\n=== 2. Arrow Functions & lexical `this` =======================');

  /*
   * 2.1 SYNTAX FLAVORS
   * --------------------------------------------------------
   */
  const square1 = (n) => { return n * n; };    // explicit return
  const square2 = n => n * n;                  // implicit return
  const getObj  = () => ({id: 1, name: '🚀'}); // return object literal

  console.assert(square1(5) === 25 && square2(6) === 36);
  console.assert(getObj().id === 1);

  /*
   * 2.2 LEXICAL `this`
   * --------------------------------------------------------
   * Arrow functions do NOT have their own `this`; they capture
   * `this` from the surrounding scope at creation time.
   */
  class Counter {
    constructor() {
      this.count = 0;
      // Traditional function requires manual binding
      setInterval(function () {
        this.count++;           // `this` is the Interval object → NaN
      }.bind(this), 1000);

      // Arrow function auto‑binds lexically to the instance
      setInterval(() => { this.count++; }, 1000);
    }
  }
  const c = new Counter();
  setTimeout(() => {
    console.log('Counter after ~2.5s (should be > 0):', c.count);
  }, 2500);

  /*
   * 2.3 NO `arguments`, `prototype`, `super`, `new.target`
   * --------------------------------------------------------
   */
  const argless = () => {
    // `arguments` is taken from outer scope (here: undefined)
    console.log('arguments in arrow →', typeof arguments);
  };
  function outer() { argless(1, 2); }
  outer();

  // Not constructible
  const ArrowCtor = () => {};
  try { new ArrowCtor(); } catch (e) {
    console.log('Expected TypeError (arrow not constructible):', e.message);
  }

  // No prototype property
  console.assert(ArrowCtor.prototype === undefined, 'Arrow has no prototype');

  /*
   * 2.4 CAN’T BE GENERATORS OR HAVE `yield`
   * --------------------------------------------------------
   */
  try {
    eval('const gen = () => { yield 1; };'); // SyntaxError
  } catch (e) {
    console.log('Expected SyntaxError (arrow can’t yield):', e.message);
  }

  /*
   * 2.5 ARROW IN CALLBACKS – PRACTICAL PATTERN
   * --------------------------------------------------------
   */
  const nums = [1, 2, 3];
  const doubled = nums.map(n => n * 2);
  console.assert(doubled.join() === '2,4,6');

  /*
   * 2.6 RETURNING `this` (fluent APIs) – arrow vs method
   * --------------------------------------------------------
   */
  const fluentObj = {
    val: 0,
    inc(n = 1) { this.val += n; return this; },  // method
    dec: (n = 1) => {                            // arrow property
      // `this` here is NOT fluentObj (lexically bound to module/global)
      // So we break the chain intentionally to demonstrate the pitfall.
      try { this.val -= n; } catch(_) {}
      return this;
    }
  };
  fluentObj.inc(5);
  console.assert(fluentObj.val === 5, '.inc works');
  fluentObj.dec(2); // does NOT modify val
  console.assert(fluentObj.val === 5, '.dec failed due to arrow `this`');

  console.log('fluentObj.val =', fluentObj.val, '(expected 5)');
})();

/* EOF – All topics thoroughly covered in a single file */