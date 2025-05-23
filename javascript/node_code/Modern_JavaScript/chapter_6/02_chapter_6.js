// Chapter 6 | Functional Programming in JS

// 1. Pure Functions & Immutability

// Pure Function: Output depends only on input, no side effects
function add(a, b) {
    return a + b;
}

// Example 1: Pure function
function square(x) {
    return x * x;
}

// Example 2: Impure function (for contrast)
let counter = 0;
function impureIncrement() {
    counter++;
    return counter;
}

// Example 3: Immutable array update
function addToArray(arr, value) {
    return [...arr, value];
}

// Example 4: Immutable object update
function updateUser(user, newName) {
    return { ...user, name: newName };
}

// Example 5: Pure function with recursion
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// 2. Higher Order Functions

// Example 1: Function that takes another function as argument
function map(arr, fn) {
    let result = [];
    for (let i = 0; i < arr.length; i++) {
        result.push(fn(arr[i]));
    }
    return result;
}

// Example 2: Function that returns another function
function multiplier(factor) {
    return function(x) {
        return x * factor;
    };
}

// Example 3: Using Array.prototype.filter (built-in HOF)
const evens = [1, 2, 3, 4, 5].filter(x => x % 2 === 0);

// Example 4: Custom reduce implementation
function reduce(arr, fn, initial) {
    let acc = initial;
    for (let i = 0; i < arr.length; i++) {
        acc = fn(acc, arr[i]);
    }
    return acc;
}

// Example 5: Function composition using HOF
function compose(f, g) {
    return function(x) {
        return f(g(x));
    };
}

// 3. Composition, Currying & Point Free Style

// Example 1: Function composition
const toUpper = str => str.toUpperCase();
const exclaim = str => str + '!';
const shout = compose(exclaim, toUpper);

// Example 2: Currying
function curriedAdd(a) {
    return function(b) {
        return a + b;
    };
}

// Example 3: Point-free style
const join = arr => arr.join('-');
const toLower = str => str.toLowerCase();
const slugify = compose(join, arr => arr.map(toLower));

// Example 4: Currying with multiple arguments
const multiply = a => b => c => a * b * c;

// Example 5: Compose multiple functions
function composeMany(...fns) {
    return function(x) {
        return fns.reduceRight((acc, fn) => fn(acc), x);
    };
}

// 4. Monads, Functors & Transducers

// Functor: Something that implements map
class Box {
    constructor(value) {
        this.value = value;
    }
    map(fn) {
        return new Box(fn(this.value));
    }
}

// Example 1: Functor usage
const box = new Box(2).map(x => x + 3).map(x => x * 2); // Box(10)

// Monad: Functor with flatMap (chain)
class Maybe {
    constructor(value) {
        this.value = value;
    }
    static of(value) {
        return new Maybe(value);
    }
    isNothing() {
        return this.value === null || this.value === undefined;
    }
    map(fn) {
        return this.isNothing() ? this : Maybe.of(fn(this.value));
    }
    flatMap(fn) {
        return this.isNothing() ? this : fn(this.value);
    }
}

// Example 2: Monad usage
const safeDivide = n => d => d === 0 ? Maybe.of(null) : Maybe.of(n / d);
const result = Maybe.of(10).flatMap(safeDivide(10));

// Example 3: Functor with Array
const arr = [1, 2, 3].map(x => x * 2); // [2, 4, 6]

// Example 4: Monad with Promise
const promise = Promise.resolve(5)
    .then(x => x * 2)
    .then(x => x + 1);

// Example 5: Transducer example
function mapT(fn) {
    return reducer => (acc, val) => reducer(acc, fn(val));
}
function filterT(pred) {
    return reducer => (acc, val) => pred(val) ? reducer(acc, val) : acc;
}
function transduce(transducer, reducer, initial, arr) {
    const xf = transducer(reducer);
    let acc = initial;
    for (const val of arr) {
        acc = xf(acc, val);
    }
    return acc;
}
const double = x => x * 2;
const isEven = x => x % 2 === 0;
const transducer = composeMany(mapT(double), filterT(isEven));
const transduced = transduce(transducer, (acc, val) => [...acc, val], [], [1,2,3,4]); // [4,8]

// 5. FP Libraries (Ramda, lodash/fp)

// Example 1: Ramda map
// const R = require('ramda');
// R.map(x => x * 2, [1,2,3]); // [2,4,6]

// Example 2: Ramda compose
// const shoutRamda = R.compose(R.toUpper, R.concat('!'));
// shoutRamda('hello'); // 'HELLO!'

// Example 3: lodash/fp map
// const _ = require('lodash/fp');
// _.map(x => x + 1, [1,2,3]); // [2,3,4]

// Example 4: lodash/fp flow (compose)
// const addExclamation = x => x + '!';
// const upper = x => x.toUpperCase();
// const shoutLodash = _.flow(upper, addExclamation);
// shoutLodash('hi'); // 'HI!'

// Example 5: Ramda curry
// const addRamda = R.curry((a, b) => a + b);
// addRamda(2)(3); // 5

// Note: Uncomment the require lines and install the libraries to run Ramda/lodash/fp examples in Node.js.