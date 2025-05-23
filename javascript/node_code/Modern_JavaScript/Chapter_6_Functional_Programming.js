// Chapter 6: Functional Programming in JS

// 1. Pure Functions & Immutability
// --------------------------------
// Example 1: Pure add – no side effects
function add(a, b) { return a + b; }
console.log(add(2, 3), add(2, 3)); // always 5

// Example 2: Immutable array update
const arr1 = [1, 2, 3];
const arr2 = arr1.concat(4);
console.log(arr1, arr2); // [1,2,3], [1,2,3,4]

// Example 3: Immutable object merge via spread
const obj1 = { x: 1, y: 2 };
const obj2 = { ...obj1, y: 42 };
console.log(obj1, obj2); // y unchanged in obj1

// Example 4: Deep freeze and mutation attempt
const deepObj = Object.freeze({ a: { b: 2 } });
try {
  deepObj.a.b = 99; // mutation fails silently or throws in strict mode
  console.log(deepObj.a.b);
} catch (e) {
  console.log('Cannot mutate:', e.message);
}

// Example 5: Clone & transform using JSON (limitations: no functions)
const original = { foo: 'bar', nested: { n: 1 } };
const clone = JSON.parse(JSON.stringify(original));
clone.nested.n = 2;
console.log(original.nested.n, clone.nested.n); // 1, 2



// 2. Higher Order Functions
// --------------------------
// Example 1: map – transforms each element
const nums = [1, 2, 3];
const sq = nums.map(n => n * n);
console.log(sq); // [1,4,9]

// Example 2: filter – selects elements
const evens = nums.filter(n => n % 2 === 0);
console.log(evens); // [2]

// Example 3: reduce – accumulates
const sum = nums.reduce((acc, n) => acc + n, 0);
console.log(sum); // 6

// Example 4: custom HOF – takes fn, returns new fn
function logger(fn) {
  return (...args) => {
    console.log('Calling with', args);
    const result = fn(...args);
    console.log('Result:', result);
    return result;
  };
}
const loggedAdd = logger(add);
loggedAdd(5, 7);

// Example 5: sort by HOF comparator
const users = [{name:'A', age:30}, {name:'B', age:25}];
users.sort((u1,u2) => u1.age - u2.age);
console.log(users); // sorted by age



// 3. Composition, Currying & Point Free Style
// -------------------------------------------
// Example 1: compose implementation (right-to-left)
const compose = (...fns) => x => fns.reduceRight((v, f) => f(v), x);
const double = x => x * 2;
const inc = x => x + 1;
console.log(compose(double, inc)(3)); // double(inc(3)) = 8

// Example 2: pipe (left-to-right)
const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
console.log(pipe(inc, double)(3)); // 8

// Example 3: simple curry
function curry2(fn) {
  return a => b => fn(a, b);
}
const curriedAdd = curry2(add);
console.log(curriedAdd(4)(5)); // 9

// Example 4: partial application
function partial(fn, ...fixed) {
  return (...rest) => fn(...fixed, ...rest);
}
const add5 = partial(add, 5);
console.log(add5(10)); // 15

// Example 5: point‑free – compose property access & transform
const getProp = p => obj => obj[p];
const toUpper = s => s.toUpperCase();
const shoutName = compose(toUpper, getProp('name'));
console.log(shoutName({ name: 'alice' })); // ALICE



// 4. Monads, Functors & Transducers
// ---------------------------------
// Example 1: Array as Functor (map)
const doubled = [1,2,3].map(n => n*2);
console.log(doubled); // [2,4,6]

// Example 2: Maybe Monad
class Maybe {
  constructor(x){ this.value = x; }
  static of(x){ return new Maybe(x); }
  map(f){ return this.value == null ? Maybe.of(null) : Maybe.of(f(this.value)); }
  getOrElse(d){ return this.value==null ? d : this.value; }
}
console.log(Maybe.of(5).map(x=>x+1).getOrElse(0)); // 6
console.log(Maybe.of(null).map(x=>x+1).getOrElse('default')); // 'default'

// Example 3: Promise as Monad
Promise.resolve(10)
  .then(x => x * 2)
  .then(console.log); // 20

// Example 4: Identity Functor
class Identity {
  constructor(x){ this.value = x; }
  map(f){ return new Identity(f(this.value)); }
}
console.log(new Identity(3).map(x=>x+4)); // Identity { value: 7 }

// Example 5: Transducer – combine map+filter for reduce
const mapT = f => r => (acc, x) => r(acc, f(x));
const filterT = pred => r => (acc, x) => pred(x)? r(acc, x) : acc;
const arrayReducer = (acc, x) => { acc.push(x); return acc; };
const xf = filterT(n => n%2)(mapT(n => n*2)(arrayReducer));
const data = [1,2,3,4];
const result = data.reduce(xf, []);
console.log(result); // [2,6]



// 5. FP Libraries (Ramda, lodash/fp)
// ----------------------------------
// Example 1: Ramda map & filter
// import R from 'ramda';
const R = require('ramda');
console.log(R.map(x=>x*3, [1,2,3]));           // [3,6,9]
console.log(R.filter(x=>x%2===1, [1,2,3,4,5])); // [1,3,5]

// Example 2: Ramda compose & curry
const addC = R.curry((a,b) => a + b);
const incC = addC(1);
console.log(incC(5));                          // 6
const shoutR = R.compose(R.toUpper, R.prop('name'));
console.log(shoutR({name:'bob'}));             // BOB

// Example 3: lodash/fp pipe & get
// const lp = require('lodash/fp');
const lp = require('lodash/fp');
const fullName = lp.pipe(
  lp.get('name.first'),
  lp.concat(' '),
  s => s + lp.get('name.last')({name:{last:'Doe'}})
);
console.log(fullName({name:{first:'John', last:'Smith'}})); // John Smith

// Example 4: Ramda lens for immutable update
const user = { name: { first:'A', last:'B' } };
const lensLast = R.lensPath(['name','last']);
const updated = R.set(lensLast, 'Z', user);
console.log(user.name.last, updated.name.last); // B, Z

// Example 5: lodash/fp memoize & debounce
const fib = n => n < 2 ? n : fib(n-1) + fib(n-2);
const fastFib = lp.memoize(fib);
console.log(fastFib(35)); // memoized result
const onResize = () => console.log('resized');
const debounced = lp.debounce(200, onResize);
window.addEventListener('resize', debounced);