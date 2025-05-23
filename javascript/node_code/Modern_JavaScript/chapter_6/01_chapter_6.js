/**************************************************************************************************
* Chapter 6 | Functional Programming in JS                                                        *
* ------------------------------------------------------------------------------------------------*
* One self-contained .js file with five discrete, progressively sophisticated code cases per item *
* to give developers a deep, example-driven understanding of FP concepts, idioms, and edge cases. *
**************************************************************************************************/

/* ────────────────────────────────────────────────────────────────────────────────────────────── */
/* 1. PURE FUNCTIONS & IMMUTABILITY (5 EXAMPLES)                                                 */
/* ────────────────────────────────────────────────────────────────────────────────────────────── */

/* 1-A  Pure mathematical function – no hidden inputs/outputs, referentially transparent. */
const add = (a, b) => a + b;
console.log('1-A:', add(2, 3)); // ➜ 5

/* 1-B  Immutable update of nested object using spread/rest. */
const user1 = { id: 1, info: { name: 'Ada', age: 42 } };
const updatedUser1 = {
  ...user1,
  info: { ...user1.info, age: 43 },   // deep-ish copy
};
console.log('1-B:', user1, updatedUser1); // original unchanged

/* 1-C  Object.freeze + shallow clones to guarantee run-time immutability. */
const CONFIG = Object.freeze({ URL: 'https://api.service.dev', TIMEOUT: 8000 });
function setTimeoutCfg(cfg, timeout) {
  return { ...cfg, TIMEOUT: timeout };         // returns new, doesn’t mutate original
}
console.log('1-C:', setTimeoutCfg(CONFIG, 10000));

/* 1-D  Pure recursion – functional factorial with immutable accumulator. */
const factorial = n => (n === 0 ? 1 : n * factorial(n - 1));
console.log('1-D:', factorial(5)); // 120

/* 1-E  Non-pure counter vs. pure alternative (exception pattern). */
let impureCounter = 0;                    // side-effectful
const incrementImpure = () => ++impureCounter;

const incrementPure = cnt => cnt + 1;     // pure
console.log('1-E:', incrementImpure(), incrementImpure(), incrementPure(10));



/* ────────────────────────────────────────────────────────────────────────────────────────────── */
/* 2. HIGHER-ORDER FUNCTIONS (HOFs) (5 EXAMPLES)                                                 */
/* ────────────────────────────────────────────────────────────────────────────────────────────── */

/* 2-A  map – classic HOF transforms array items. */
const double = x => x * 2;
console.log('2-A:', [1, 2, 3].map(double));

/* 2-B  filter – predicate decoupled from iteration strategy. */
const isPrime = n => {
  for (let i = 2; i <= Math.sqrt(n); i++) if (n % i === 0) return false;
  return n > 1;
};
console.log('2-B:', [...Array(20).keys()].filter(isPrime));

/* 2-C  reduce – aggregate with custom reducer (sum of squares). */
const sumSquares = (acc, val) => acc + val * val;
console.log('2-C:', [1, 2, 3, 4].reduce(sumSquares, 0));

/* 2-D  once – HOF producing a function that can run only once (imperative code wrapped). */
const once = fn => {
  let done = false, res;
  return (...args) => {
    if (!done) { done = true; res = fn(...args); }
    return res;
  };
};
const init = once(() => console.log('2-D: init side effect'));
init(); init(); // logged only once

/* 2-E  debounce – timing-based HOF (cancels rapid repeats). */
const debounce = (fn, ms) => {
  let id;
  return (...args) => {
    clearTimeout(id);
    id = setTimeout(() => fn(...args), ms);
  };
};
const logResize = debounce(() => console.log('2-E: resize!'), 250);
// window.addEventListener('resize', logResize); // browser-only illustrative



/* ────────────────────────────────────────────────────────────────────────────────────────────── */
/* 3. COMPOSITION, CURRYING & POINT-FREE STYLE (5 EXAMPLES)                                      */
/* ────────────────────────────────────────────────────────────────────────────────────────────── */

/* Utility compose (right-to-left) / pipe (left-to-right). */
const compose = (...fns) => x => fns.reduceRight((v, f) => f(v), x);
const pipe    = (...fns) => x => fns.reduce((v, f) => f(v), x);

/* 3-A  Currying by hand. */
const multiply = a => b => a * b;
const doublePF = multiply(2);
console.log('3-A:', doublePF(7));

/* 3-B  Partial application via bind (built-in). */
function power(base, exp) { return base ** exp; }
const squarePF = power.bind(null, undefined, 2);  // placeholder style
console.log('3-B:', squarePF(9));

/* 3-C  Composition to create data pipeline. */
const inc = n => n + 1;
const triple = n => n * 3;
const incThenTriple = pipe(inc, triple);
console.log('3-C:', incThenTriple(4));

/* 3-D  Point-free transform (“what” not “how”). */
const toUpper   = s => s.toUpperCase();
const exclaim   = s => s + '!';
const shout     = compose(exclaim, toUpper);
console.log('3-D:', shout('functional rocks'));

/* 3-E  Automatic currying util (generic). */
const curry = (fn, arity = fn.length) =>
  function nextCurried(prevArgs) {
    return function (...nextArgs) {
      const args = [...prevArgs, ...nextArgs];
      return args.length >= arity ? fn(...args) : nextCurried(args);
    };
  }([]);
const sum3 = (a, b, c) => a + b + c;
const curriedSum3 = curry(sum3);
console.log('3-E:', curriedSum3(1)(2)(3));



/* ────────────────────────────────────────────────────────────────────────────────────────────── */
/* 4. MONADS, FUNCTORS & TRANSDUCERS (5 EXAMPLES)                                                */
/* ────────────────────────────────────────────────────────────────────────────────────────────── */

/* Minimal Functor spec (map). */
class Box {
  constructor(x) { this.$value = x; }
  map(f)        { return new Box(f(this.$value)); }  // Functor
  chain(f)      { return f(this.$value); }           // Monad (flatMap)
  toString()    { return `Box(${this.$value})`; }
}

/* 4-A  Functor mapping. */
console.log('4-A:', new Box(4).map(x => x + 2).toString());

/* 4-B  Maybe Monad for null-safety. */
class Maybe {
  static of(x)         { return new Maybe(x); }
  constructor(x)       { this.__value = x; }
  isNothing()          { return this.__value == null; }
  map(f)               { return this.isNothing() ? this : Maybe.of(f(this.__value)); }
  chain(f)             { return this.map(f).join(); }
  join()               { return this.isNothing() ? this : this.__value; }
  toString()           { return this.isNothing() ? 'Nothing' : `Just(${this.__value})`; }
}
const safeProp = key => obj => Maybe.of(obj[key]);
console.log(
  '4-B:',
  safeProp('address')({})( ).toString()       // Nothing
);

/* 4-C  Promise as Monad – chaining ensures sequencing. */
Promise.resolve(2)
  .then(x => x + 3)
  .then(x => console.log('4-C:', x)); // 5

/* 4-D  Either Monad (Left for errors). */
class Left   { constructor(x) { this.__value = x; } map() { return this; }  toString(){return `Left(${this.__value})`;} }
class Right  { constructor(x) { this.__value = x; } map(f){ return new Right(f(this.__value)); } toString(){return `Right(${this.__value})`;} }
const either = (f, g, e) => e instanceof Left ? f(e.__value) : g(e.__value);
const parseJSON = str => { try { return new Right(JSON.parse(str)); } catch (e) { return new Left(e); } };
console.log('4-D:', parseJSON('{"ok":true}').toString(), parseJSON('💥').toString());

/* 4-E  Transducer – decouple iteration from transformation. */
const mapT = f => reducer => (acc, val) => reducer(acc, f(val));
const filterT = pred => reducer => (acc, val) => pred(val) ? reducer(acc, val) : acc;
const transduce = (xf, reducer, init, coll) =>
  coll.reduce(xf(reducer), init);

const xf = compose(
  mapT(x => x * 2),
  filterT(x => x % 3 === 0)
);

const sum = (a, b) => a + b;
console.log('4-E:', transduce(xf, sum, 0, [1,2,3,4,5,6])); // double → [2,4,6,8,10,12] filter → [6,12] => sum 18



/* ────────────────────────────────────────────────────────────────────────────────────────────── */
/* 5. FP LIBRARIES (RAMDA & LODASH/FP) (5 EXAMPLES)                                             */
/* ────────────────────────────────────────────────────────────────────────────────────────────── */

/* NOTE: run `npm i ramda lodash` to test these in Node. */
const R   = require?.('ramda')   || {};
const _fp = require?.('lodash/fp') || {};

/* 5-A  Ramda compose & curry are auto-curried. */
const addR = R.curry((a, b) => a + b);
const incR = addR(1);
console.log('5-A:', incR(10));

/* 5-B  Ramda lens for immutable deep updates. */
const lensAge = R.lensPath(['info', 'age']);
const user2 = { info: { age: 30 } };
const older = R.over(lensAge, R.inc, user2);
console.log('5-B:', user2, older);

/* 5-C  lodash/fp flow (left-to-right composition). */
const square = x => x * x;
const add1   = x => x + 1;
const pipeline = _fp.flow(add1, square);
console.log('5-C:', pipeline(4));

/* 5-D  lodash/fp get + set for functional property access. */
const nameLens = _fp.get('info.name');
const user3 = { info: { name: 'Grace' } };
console.log('5-D:', nameLens(user3));

/* 5-E  Ramda transduce utility vs manual transducer in 4-E. */
const doubleR = x => x * 2;
const isEven  = x => x % 2 === 0;
const result = R.transduce(R.compose(R.map(doubleR), R.filter(isEven)), R.add, 0, [1,2,3,4,5]);
console.log('5-E:', result); // double then even filter then sum