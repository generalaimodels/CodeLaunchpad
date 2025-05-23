/**
 * File: es6-enhancements.js
 * Description: Comprehensive coverage of ES6+ language enhancements:
 *   1. let / const vs var
 *   2. Arrow functions & lexical `this`
 * No unnecessary content before/after. Single-file demonstration.
 */

/* ============================================================================
 * 1. let / const vs var
 * ============================================================================
 * Scope:
 *   - var: function-scoped (or global if declared at top level)
 *   - let/const: block-scoped (enclosed by `{ ... }`)
 *
 * Hoisting:
 *   - var declarations are hoisted and initialized with `undefined`
 *   - let/const declarations are hoisted but not initialized (Temporal Dead Zone)
 *
 * Redeclaration & Reassignment:
 *   - var: can be redeclared & reassigned
 *   - let: cannot be redeclared in same scope; can be reassigned
 *   - const: cannot be redeclared or reassigned (immutable binding)
 *
 * Best Practices:
 *   - Prefer `const` for all values that don’t change
 *   - Use `let` when you know the variable will be reassigned
 *   - Avoid `var` in modern code
 */

// Example: var vs let/const scoping & hoisting
function demoVarLetConst() {
    console.log('--- demoVarLetConst ---');
  
    // var is hoisted and initialized as undefined
    console.log('x before var declaration:', x); // undefined
    var x = 1;
    console.log('x after var declaration:', x);  // 1
  
    // let is hoisted but in Temporal Dead Zone until initialization
    try {
      console.log('y before let declaration:', y);
    } catch (e) {
      console.log('Accessing y before declaration throws:', e.name);
    }
    let y = 2;
    console.log('y after let declaration:', y); // 2
  
    // const behaves similarly to let w.r.t TDZ
    try {
      console.log('z before const declaration:', z);
    } catch (e) {
      console.log('Accessing z before declaration throws:', e.name);
    }
    const z = 3;
    console.log('z after const declaration:', z); // 3
  
    // Redeclaration
    var x = 100; // allowed
    console.log('x redeclared with var:', x);
    // let y = 200; // SyntaxError: Identifier 'y' has already been declared
    // const z = 300; // SyntaxError: Identifier 'z' has already been declared
  
    // Reassignment
    x = 10;       // allowed
    y = 20;       // allowed
    // z = 30;    // TypeError: Assignment to constant variable
  
    console.log({ x, y, z });
  }
  demoVarLetConst();
  
  /* Exceptions & Edge Cases:
   *  - var in for-loop headers leaks into enclosing scope
   *  - let/const create a fresh binding each iteration
   */
  function demoLoopScope() {
    console.log('--- demoLoopScope ---');
  
    for (var i = 0; i < 3; i++) {
      setTimeout(() => console.log('var i:', i), 0); // prints 3,3,3
    }
  
    for (let j = 0; j < 3; j++) {
      setTimeout(() => console.log('let j:', j), 0); // prints 0,1,2
    }
  }
  demoLoopScope();
  
  
  /* ============================================================================
   * 2. Arrow Functions & Lexical `this`
   * ============================================================================
   *
   * Syntax:
   *   param => expression
   *   (p1, p2) => { statements }
   *
   * Characteristics:
   *   - No own `this` — inherits from surrounding scope (lexical `this`)
   *   - No `arguments` object
   *   - Cannot be used as constructors (no `new`)
   *   - No `prototype` property
   *
   * Use Cases:
   *   - Inline callbacks
   *   - Methods that need to access outer `this`
   *
   * Pitfalls:
   *   - Not suitable for object methods where you expect a dynamic `this`
   */
  
  // Simple arrow function examples
  const add = (a, b) => a + b;
  const square = x => x * x;
  const logMessage = () => console.log('Hello from arrow!');
  
  console.log('add(2,3)=', add(2, 3));
  console.log('square(4)=', square(4));
  logMessage();
  
  // Lexical `this` demonstration
  const obj = {
    name: 'Alice',
    regularFunction: function() {
      console.log('regularFunction this.name =', this.name);
    },
    arrowFunction: () => {
      // `this` here is inherited from the module/global scope
      console.log('arrowFunction this.name =', this.name);
    },
    delayedLogRegular: function() {
      setTimeout(function() {
        console.log('delayedLogRegular this.name =', this.name);
      }, 10);
    },
    delayedLogArrow: function() {
      setTimeout(() => {
        console.log('delayedLogArrow this.name =', this.name);
      }, 10);
    }
  };
  
  console.log('--- this-binding demo ---');
  obj.regularFunction();       // Alice
  obj.arrowFunction();         // undefined (or global.name)
  obj.delayedLogRegular();     // undefined (callback's this is global/undefined)
  obj.delayedLogArrow();       // Alice (lexical this captured)
  
  /* Exceptions & Notes:
   *  - Arrow functions cannot be methods if you rely on dynamic `this`
   *  - You cannot call `new` on an arrow function: `new (() => {})` → TypeError
   *  - No `arguments` binding: use rest parameters instead
   */
  
   // Example: no `arguments` in arrow
  const arrowWithArgs = () => {
    try {
      console.log(arguments);
    } catch (e) {
      console.log('arguments not defined in arrow:', e.name);
    }
  };
  arrowWithArgs(1,2,3);
  
  // Use rest parameters instead
  const sumAll = (...nums) => nums.reduce((acc, n) => acc + n, 0);
  console.log('sumAll(1,2,3,4)=', sumAll(1,2,3,4));