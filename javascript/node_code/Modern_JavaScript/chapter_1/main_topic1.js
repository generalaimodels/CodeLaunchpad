/********************************************************************************************
*  ADVANCED JAVASCRIPT PLAYGROUND – SINGLE FILE REFERENCE
*  Each section covers one big topic with 5 minimal-yet-illustrative examples.
*  Run with Node ≥14 (or browser console).  Use `node --experimental-modules` if needed.
*********************************************************************************************/

/*==============================================================
= 1. Promises vs. Async/Await =================================
==============================================================*/
(() => {
    console.log('\n--- 1. PROMISES vs ASYNC/AWAIT --------------------------------');
  
    // Ex-1  Promise chaining vs async/await (sequential)
    const delay = (ms, val) =>
      new Promise((res, rej) => (ms < 0 ? rej('neg delay') : setTimeout(() => res(val), ms)));
  
    delay(100, 'A')
      .then((v) => delay(100, v + 'B'))
      .then((v) => console.log('Promise chain:', v))
      .catch(console.error);
  
    (async () => {
      try {
        const v1 = await delay(100, 'A');
        const v2 = await delay(100, v1 + 'B');
        console.log(' async/await :', v2);
      } catch (e) {
        console.error(e);
      }
    })();
  
    // Ex-2  Parallel execution
    Promise.all([delay(120, 1), delay(80, 2)]).then((r) => console.log('Parallel(Promise)', r));
    (async () => {
      const r = await Promise.all([delay(120, 1), delay(80, 2)]);
      console.log('Parallel(await)', r);
    })();
  
    // Ex-3  Error bubbling differences
    delay(-1, 'bad')
      .then(console.log)
      .catch((e) => console.log('Promise error handled:', e));
  
    (async () => {
      try {
        await delay(-1);
      } catch (e) {
        console.log(' async/await error :', e);
      }
    })();
  
    // Ex-4  thenable vs awaitable
    const thenableObj = { then: (resolve) => resolve('I am thenable') };
    Promise.resolve(thenableObj).then(console.log);
    (async () => console.log(await thenableObj))();
  
    // Ex-5  Promise.finally symmetry with try/finally
    delay(50, 'clean')
      .finally(() => console.log('Promise.finally done'))
      .catch(() => {});
    (async () => {
      try {
        await delay(50, 'clean');
      } finally {
        console.log('try/finally done');
      }
    })();
  })();
  
  /*==============================================================
  = 2. Higher-Order Functions ===================================
  ==============================================================*/
  (() => {
    console.log('\n--- 2. HIGHER-ORDER FUNCTIONS ---------------------------------');
  
    // Ex-1  Function that takes fn
    const repeat = (fn, n) => [...Array(n)].forEach((_, i) => fn(i));
    repeat((i) => console.log('repeat', i), 3);
  
    // Ex-2  Function returning a function (logger factory)
    const logger = (prefix) => (msg) => console.log(prefix, msg);
    const warn = logger('[WARN]');
    warn('disk almost full');
  
    // Ex-3  Decorator pattern
    const measure = (fn) => (...args) => {
      const t0 = performance.now();
      const out = fn(...args);
      console.log(`${fn.name} ran in ${performance.now() - t0}ms`);
      return out;
    };
    const slowFib = measure((n) => (n < 2 ? n : slowFib(n - 1) + slowFib(n - 2)));
    slowFib(10);
  
    // Ex-4  Array prototype methods (map is HOF)
    console.log([1, 2, 3].map((x) => x ** 2));
  
    // Ex-5  Custom array combinator
    const zipWith = (fn, a, b) => a.map((v, i) => fn(v, b[i]));
    console.log(zipWith(Math.max, [1, 7, 3], [6, 2, 9]));
  })();
  
  /*==============================================================
  = 3. Closures and Applications ================================
  ==============================================================*/
  (() => {
    console.log('\n--- 3. CLOSURES ---------------------------------------------');
  
    // Ex-1  Counter
    const makeCounter = () => {
      let c = 0;
      return () => ++c;
    };
    const c1 = makeCounter();
    console.log(c1(), c1());
  
    // Ex-2  Data encapsulation
    const BankAccount = (initial = 0) => {
      let balance = initial;
      return {
        deposit: (amt) => (balance += amt),
        withdraw: (amt) => (balance -= amt),
        get balance() {
          return balance;
        },
      };
    };
    const acct = BankAccount(100);
    acct.deposit(40);
    console.log('Balance:', acct.balance);
  
    // Ex-3  Loop variable capture pitfalls & fix
    const fns = [];
    for (var i = 0; i < 3; i++) {
      ((j) => fns.push(() => console.log(j)))(i); // capture per-iteration
    }
    fns.forEach((fn) => fn());
  
    // Ex-4  Once initializer
    const once = (fn) => {
      let done = false,
        res;
      return (...args) => (done ? res : ((done = true), (res = fn(...args))));
    };
    const init = once(() => console.log('executed'));
    init();
    init();
  
    // Ex-5  Memoization via closure (see also section 4)
    const memoFib = (() => {
      const memo = [0, 1];
      return function fib(n) {
        return memo[n] ?? (memo[n] = fib(n - 1) + fib(n - 2));
      };
    })();
    console.log('Fib 10:', memoFib(10));
  })();
  
  /*==============================================================
  = 4. Memoization Techniques ===================================
  ==============================================================*/
  (() => {
    console.log('\n--- 4. MEMOIZATION ------------------------------------------');
  
    // Helper generic memoize
    const memoize = (fn) => {
      const cache = new Map();
      return (...args) => {
        const key = JSON.stringify(args);
        if (!cache.has(key)) cache.set(key, fn(...args));
        return cache.get(key);
      };
    };
  
    // Ex-1  Recursive memoized Fibonacci
    const fib = memoize((n) => (n < 2 ? n : fib(n - 1) + fib(n - 2)));
    console.log(fib(40));
  
    // Ex-2  Factorial with Map cache keyed by n
    const factorial = memoize((n) => (n <= 1 ? 1 : n * factorial(n - 1)));
    console.log(factorial(10));
  
    // Ex-3  Memoizing async functions
    const asyncMemo = (fn) => {
      const cache = new Map();
      return async (...args) => {
        const k = JSON.stringify(args);
        if (!cache.has(k)) cache.set(k, fn(...args));
        return cache.get(k);
      };
    };
    const fetchFake = asyncMemo((url) => delay(100, `data@${url}`));
    fetchFake('/api').then(console.log).then(() => fetchFake('/api').then(console.log));
  
    // Ex-4  WeakMap for object keys (avoids leaks)
    const heavy = (obj) => ({ ...obj, computed: true });
    const memoHeavy = (() => {
      const w = new WeakMap();
      return (o) => (w.has(o) ? w.get(o) : w.set(o, heavy(o)).get(o));
    })();
    const o = {};
    console.log(memoHeavy(o) === memoHeavy(o));
  
    // Ex-5  LRU Cache Memoizer (size 3)
    const memoizeLRU = (fn, size = 3) => {
      const cache = new Map();
      return (...args) => {
        const k = JSON.stringify(args);
        if (cache.has(k)) return cache.get(k);
        const val = fn(...args);
        cache.set(k, val);
        if (cache.size > size) cache.delete(cache.keys().next().value);
        return val;
      };
    };
    const sq = memoizeLRU((x) => x * x);
    [1, 2, 3, 4, 1].forEach((n) => console.log('LRU', n, sq(n)));
  })();
  
  /*==============================================================
  = 5. Currying =================================================
  ==============================================================*/
  (() => {
    console.log('\n--- 5. CURRYING ---------------------------------------------');
  
    // Ex-1  Simple curry
    const curry = (fn) => (...args) =>
      args.length >= fn.length ? fn(...args) : curry(fn.bind(null, ...args));
    const add3 = (a, b, c) => a + b + c;
    console.log(curry(add3)(1)(2)(3));
  
    // Ex-2  Partial application
    const multiply = (a) => (b) => a * b;
    const double = multiply(2);
    console.log('double 7 =', double(7));
  
    // Ex-3  Placeholder curry
    const _ = Symbol('placeholder');
    const curryP = (fn, arr = []) => (...args) => {
      const merged = arr.map((x) => (x === _ && args.length ? args.shift() : x)).concat(args);
      return merged.includes(_) ? curryP(fn, merged) : fn(...merged);
    };
    const greet = (a, b, c) => `${a} ${b} ${c}`;
    const hi = curryP(greet)(_, 'there')('Hi');
    console.log(hi('!'));
  
    // Ex-4  Lodash-style _.curryRight
    const curryRight = (fn, collected = []) => (...args) =>
      collected.length + args.length >= fn.length
        ? fn(...args.reverse(), ...collected)
        : curryRight(fn, [...args.reverse(), ...collected]);
    const divide = (a, b) => a / b;
    const half = curryRight(divide)(2);
    console.log('10 half =', half(10));
  
    // Ex-5  Infinite curry sum(..)
    const sum = (...a) => a.reduce((s, v) => s + v, 0);
    const curryInfinite = (total = 0) => (n) =>
      n === undefined ? total : curryInfinite(total + n);
    console.log('sum:', curryInfinite()(1)(2)(3)());
  })();
  
  /*==============================================================
  = 6. Event Loop Mechanics =====================================
  ==============================================================*/
  (() => {
    console.log('\n--- 6. EVENT LOOP -------------------------------------------');
  
    console.log('script start');
  
    setTimeout(() => console.log('macrotask - timeout 0'), 0);
  
    Promise.resolve().then(() => console.log('microtask - promise 1'));
  
    queueMicrotask(() => console.log('microtask - queueMicrotask'));
  
    setTimeout(() => console.log('macrotask - timeout 1'), 1);
  
    (async () => {
      console.log('async start');
      await null; // schedules microtask
      console.log('async after await');
    })();
  
    console.log('script end');
  })();
  
  /*==============================================================
  = 7. Prototype Chain & Inheritance ============================
  ==============================================================*/
  (() => {
    console.log('\n--- 7. PROTOTYPE CHAIN --------------------------------------');
  
    // Ex-1  Function constructor
    function Animal(name) {
      this.name = name;
    }
    Animal.prototype.speak = function () {
      console.log(this.name, 'makes noise');
    };
    const dog = new Animal('Rex');
    dog.speak();
  
    // Ex-2  Classical class extends
    class Person {
      constructor(n) {
        this.n = n;
      }
      greet() {
        console.log('Hi', this.n);
      }
    }
    class Employee extends Person {
      salute() {
        console.log(this.n, 'salutes');
      }
    }
    new Employee('Eve').greet();
  
    // Ex-3  Object.create
    const proto = { kind: 'proto' };
    const obj = Object.create(proto);
    console.log(obj.kind);
  
    // Ex-4  Overriding properties
    obj.kind = 'instance';
    console.log(obj.kind, proto.kind);
  
    // Ex-5  instanceof vs isPrototypeOf
    console.log(dog instanceof Animal, Animal.prototype.isPrototypeOf(dog));
  })();
  
  /*==============================================================
  = 8. Function Composition (pipe/compose) ======================
  ==============================================================*/
  (() => {
    console.log('\n--- 8. FUNCTION COMPOSITION ---------------------------------');
  
    const pipe =
      (...fns) =>
      (x) =>
        fns.reduce((v, fn) => fn(v), x);
    const compose =
      (...fns) =>
      (x) =>
        fns.reduceRight((v, fn) => fn(v), x);
  
    const inc = (x) => x + 1;
    const dbl = (x) => x * 2;
  
    // Ex-1  pipe
    console.log(pipe(inc, dbl)(3));
  
    // Ex-2  compose
    console.log(compose(dbl, inc)(3));
  
    // Ex-3  Complex pipeline
    const toStr = (x) => `#${x}`;
    console.log(pipe(inc, dbl, toStr)(5));
  
    // Ex-4  Async compose (handles Promise)
    const composeAsync =
      (...fns) =>
      (x) =>
        fns.reduceRight((p, fn) => Promise.resolve(p).then(fn), x);
    composeAsync(delay.bind(null, 50), inc)(2).then(console.log);
  
    // Ex-5  Transducer-like composition
    const map = (fn) => (arr) => arr.map(fn);
    const filter = (fn) => (arr) => arr.filter(fn);
    const result = pipe(map(inc), filter((x) => x % 2))( [1,2,3,4] );
    console.log(result);
  })();
  
  /*==============================================================
  = 9. Lazy Evaluation & Generators =============================
  ==============================================================*/
  (() => {
    console.log('\n--- 9. GENERATORS & LAZY EVAL --------------------------------');
  
    // Ex-1  Infinite numbers
    function* naturals() {
      let n = 0;
      while (true) yield n++;
    }
    const nat = naturals();
    console.log(nat.next().value, nat.next().value);
  
    // Ex-2  Range generator
    function* range(a, b, step = 1) {
      for (let i = a; i < b; i += step) yield i;
    }
    console.log([...range(0, 5)]);
  
    // Ex-3  Delegating yield*
    function* fibonacci(n) {
      let [a, b] = [0, 1];
      while (n--) {
        yield a;
        [a, b] = [b, a + b];
      }
    }
    function* fibSquares(n) {
      yield* fibonacci(n);
      return 'done';
    }
    console.log([...fibSquares(7)]);
  
    // Ex-4  Async generator (fetch mock)
    async function* paginate(pages) {
      for (let p = 1; p <= pages; p++) {
        yield await delay(50, `page-${p}`);
      }
    }
    (async () => {
      for await (const d of paginate(3)) console.log(d);
    })();
  
    // Ex-5  Lazy pipeline
    const take = (n, iter) => {
      const out = [];
      for (const v of iter) {
        out.push(v);
        if (out.length === n) break;
      }
      return out;
    };
    const evens = function* (iter) {
      for (const v of iter) if (v % 2 === 0) yield v;
    };
    console.log(take(3, evens(naturals())));
  })();
  
  /*==============================================================
  = 10. Web Workers & Multithreading ============================
  ==============================================================*/
  (() => {
    console.log('\n--- 10. WEB WORKERS -----------------------------------------');
  
    if (typeof Worker === 'undefined') {
      console.log('Workers not supported in this environment (Node without experimental Worker).');
      return;
    }
  
    // Helper to create inline worker from function body
    const mkWorker = (fn) =>
      new Worker(URL.createObjectURL(new Blob([`onmessage=${fn.toString()}`])));
  
    // Ex-1  Heavy compute offloaded
    const primeWorker = mkWorker(function (e) {
      const isPrime = (n) => {
        for (let i = 2; i * i <= n; i++) if (n % i === i) return false;
        return n > 1;
      };
      postMessage(isPrime(e.data));
    });
    primeWorker.onmessage = (e) => console.log('Is 104729 prime?', e.data);
    primeWorker.postMessage(104729);
  
    // Ex-2  Worker with transferable objects
    const bufWorker = mkWorker(function (e) {
      const arr = new Uint8Array(e.data);
      arr[0] = 255;
      postMessage(arr.buffer, [arr.buffer]);
    });
    const buf = new ArrayBuffer(4);
    bufWorker.onmessage = (e) => console.log('Modified buffer[0]=', new Uint8Array(e.data)[0]);
    bufWorker.postMessage(buf, [buf]);
  
    // Ex-3  Worker error handling
    const errWorker = mkWorker(() => {
      throw new Error('Worker failed');
    });
    errWorker.onerror = (e) => console.log('Worker error caught', e.message);
  
    // Ex-4  Pool of workers
    const makePool = (size, workerFn) => {
      const pool = Array.from({ length: size }, () => mkWorker(workerFn));
      let idx = 0;
      return (data) =>
        new Promise((res) => {
          const w = pool[idx = (idx + 1) % size];
          w.onmessage = (e) => res(e.data);
          w.postMessage(data);
        });
    };
    const squarePool = makePool(2, (e) => postMessage(e.data * e.data));
    squarePool(3).then((r) => console.log('pool result:', r));
  
    // Ex-5  Terminating worker
    const tmp = mkWorker((e) => postMessage(e.data));
    tmp.postMessage('bye');
    tmp.onmessage = (e) => {
      console.log(e.data);
      tmp.terminate();
    };
  })();