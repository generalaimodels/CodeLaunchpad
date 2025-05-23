/**************************************************************************************************
 *  Chapter 6 | Functional Programming in JavaScript
 *  -----------------------------------------------------------------------------------------------
 *  All concepts packed in ONE file, grouped into sections, ≥5 runnable examples each.
 *  Works in modern browsers & Node (≥14).  Copy/paste or import as a module for experimentation.
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION PF ── Pure Functions & Immutability                                                 */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

/* PF‑Example‑1:  Deterministic output with no side‑effects */
const add = (a, b) => a + b;
console.log('PF‑1:', add(2, 3));

/* PF‑Example‑2:  Avoiding mutation with spread (arrays) */
const pushPure = (arr, val) => [...arr, val];
const original = [1, 2];
console.log('PF‑2:', pushPure(original, 3), original);   // original untouched

/* PF‑Example‑3:  Object.freeze to enforce shallow immutability */
const cfg = Object.freeze({ host: 'localhost', port: 8080 });
// cfg.port = 9000; // throws in strict mode
console.log('PF‑3:', cfg);

/* PF‑Example‑4:  Deep freeze utility for nested immutability */
const deepFreeze = o => {
  Object.values(o).forEach(v => typeof v === 'object' && v && deepFreeze(v));
  return Object.freeze(o);
};
const nested = deepFreeze({ a: { b: 2 } });
// nested.a.b = 3; // TypeError in strict mode
console.log('PF‑4:', nested);

/* PF‑Example‑5:  Referential transparency test helper */
const referentiallyTransparent = (fn, ...args) =>
  JSON.stringify(fn(...args)) === JSON.stringify(fn(...args));
console.log('PF‑5 add is RT:', referentiallyTransparent(add, 1, 2));

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION HOF ── Higher‑Order Functions                                                       */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

/* HOF‑Example‑1:  Function returning function (power) */
const power = p => n => n ** p;
console.log('HOF‑1:', power(3)(2));

/* HOF‑Example‑2:  Function accepting function (generic iterate) */
const iterate = (f, seed, times) => {
  let acc = seed;
  for (let i = 0; i < times; ++i) acc = f(acc);
  return acc;
};
console.log('HOF‑2:', iterate(n => n * 2, 1, 5));

/* HOF‑Example‑3:  Array.prototype.map is HOF */
console.log('HOF‑3:', [1, 2, 3].map(n => n * n));

/* HOF‑Example‑4:  Custom comparator HOF for sort */
const by = key => (a, b) => a[key] - b[key];
console.log('HOF‑4:', [{ v: 3 }, { v: 1 }].sort(by('v')));

/* HOF‑Example‑5:  Once wrapper (decorator) */
const once = fn => {
  let done = false, res;
  return (...a) => (done ? res : ((done = true), (res = fn(...a))));
};
const init = once(() => 'initialized');
console.log('HOF‑5:', init(), init());

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION CCP ── Composition, Currying & Point‑Free Style                                     */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

/* CCP‑Example‑1:  compose & pipe */
const compose = (...fns) => x => fns.reduceRight((v, f) => f(v), x);
const pipe    = (...fns) => x => fns.reduce((v, f) => f(v), x);
const inc = x => x + 1, dbl = x => x * 2;
console.log('CCP‑1 compose:', compose(dbl, inc)(2), 'pipe:', pipe(dbl, inc)(2));

/* CCP‑Example‑2:  Curry helper */
const curry = fn => (...a) =>
  a.length >= fn.length ? fn(...a) : (...b) => curry(fn)(...a, ...b);
const sum3 = curry((a, b, c) => a + b + c);
console.log('CCP‑2:', sum3(1)(2)(3));

/* CCP‑Example‑3:  Partial application via bind */
const modulo = (d, n) => n % d;
const isOdd = modulo.bind(null, 2);
console.log('CCP‑3:', [1, 2, 3, 4].filter(isOdd));

/* CCP‑Example‑4:  Point‑free filter + map */
const words = str => str.split(/\s+/);
const lengths = pipe(words, arr => arr.map(w => w.length));
console.log('CCP‑4:', lengths('Point free programming'));

/* CCP‑Example‑5:  Infinite list generator composed */
const iteratee = f => function* (seed) { let x = seed; while (true) { x = f(x); yield x; } };
const take = n => iter => {
  const out = [];
  for (const v of iter) { if (out.length === n) break; out.push(v); }
  return out;
};
console.log('CCP‑5:', take(5)(iteratee(x => x + 1)(0)));

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION MFT ── Monads, Functors & Transducers                                               */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

/* MFT‑Example‑1:  Functor (Maybe) */
class Maybe {
  static of(v) { return new Just(v); }
  map() { return this; }          // default no‑op
}
class Just extends Maybe {
  constructor(v) { super(); this.val = v; }
  map(f) { return Maybe.of(f(this.val)); }
}
class Nothing extends Maybe {}
const safeHead = arr => (arr.length ? Maybe.of(arr[0]) : new Nothing());
console.log('MFT‑1:', safeHead([]).map(x => x * 2) instanceof Nothing);

/* MFT‑Example‑2:  Monad (Promise chaining) */
Promise.resolve(2)
  .then(x => x + 1)
  .then(x => console.log('MFT‑2:', x));

/* MFT‑Example‑3:  Custom monad (IO) */
const IO = f => ({
  map: g => IO(() => g(f())),
  run: () => f()
});
const readEnv = IO(() => process.env.USER || 'anon');
console.log('MFT‑3:', readEnv.map(u => `hi ${u}`).run());

/* MFT‑Example‑4:  Simple transducer pipeline */
const mapT = f => step => (acc, v) => step(acc, f(v));
const filterT = pred => step => (acc, v) => (pred(v) ? step(acc, v) : acc);
const transduce = (xform, reducer, init, arr) =>
  arr.reduce(xform(reducer), init);
const xf = compose(
  mapT(x => x * 2),
  filterT(x => x > 5)
);
console.log('MFT‑4:', transduce(xf, (a, v) => (a.push(v), a), [], [1, 2, 3, 4]));

/* MFT‑Example‑5:  Functor law test (identity) */
const Identity = x => ({
  val: x,
  map: f => Identity(f(x))
});
const id = x => x;
const testIdentityLaw = F =>
  JSON.stringify(F.map(id)) === JSON.stringify(F);
console.log('MFT‑5:', testIdentityLaw(Identity(7)));

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION FPL ── FP Libraries (Ramda, lodash/fp)                                              */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

/* NOTE: These examples assume Ramda & lodash/fp are available.  In Node: npm i ramda lodash */
/* FPL‑Example‑1:  Ramda compose */
(async () => {
  try {
    const R = await import('https://esm.run/ramda@latest');
    const f = R.compose(R.toUpper, R.trim);
    console.log('FPL‑1:', f('  hello '));
  } catch {}
})();

/* FPL‑Example‑2:  Ramda lens */
(async () => {
  try {
    const R = await import('https://esm.run/ramda@latest');
    const lensX = R.lensProp('x');
    console.log('FPL‑2:', R.set(lensX, 9, { x: 1 }));
  } catch {}
})();

/* FPL‑Example‑3:  lodash/fp flow */
(async () => {
  try {
    const _ = await import('https://esm.run/lodash@fp');
    const f = _.flow(_.map(_.add(1)), _.filter(_.gt(_, 3)));
    console.log('FPL‑3:', f([1, 2, 3]));
  } catch {}
})();

/* FPL‑Example‑4:  lodash/fp get or default */
(async () => {
  try {
    const _ = await import('https://esm.run/lodash@fp');
    console.log('FPL‑4:', _.getOr('N/A', 'user.name')({}));
  } catch {}
})();

/* FPL‑Example‑5:  Tree shaking friendly imports */
import('https://esm.run/lodash-es/add').then(({ default: addES }) =>
  console.log('FPL‑5:', [1, 2, 3].map(addES(10))));