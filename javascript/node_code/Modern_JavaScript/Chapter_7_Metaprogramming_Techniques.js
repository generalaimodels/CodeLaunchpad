/**************************************************************************************************
 * Chapter 7 | Metaprogramming Techniques
 * --------------------------------------------------------------------
 * ONE self‑contained .js file — 5 sections × 5 runnable examples each.
 * Works under modern Node (≥16) & browsers (where features exist).
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION RFL — Reflect API & Metadata                            */
/*───────────────────────────────────────────────────────────────────*/

/* RFL‑Example‑1: Reflect.defineProperty returns boolean (no throw) */
(function () {
    const obj = {};
    const ok  = Reflect.defineProperty(obj, 'x', { value: 3 });
    console.log('RFL‑1:', ok, obj.x);
  })();
  
  /* RFL‑Example‑2: Reflect.set with receiver for correct this‑binding */
  (function () {
    const target = { set x(v) { this._x = v * 2; } };
    const receiver = {};
    Reflect.set(target, 'x', 4, receiver);
    console.log('RFL‑2:', receiver._x); // 8
  })();
  
  /* RFL‑Example‑3: Reflect.getOwnPropertyDescriptor shortcut */
  (function () {
    const o = { hidden: 1 };
    Reflect.defineProperty(o, 'hidden', { enumerable: false });
    console.log('RFL‑3:', Reflect.getOwnPropertyDescriptor(o, 'hidden').enumerable);
  })();
  
  /* RFL‑Example‑4: Reflect.ownKeys merges string & symbol keys */
  (function () {
    const s = Symbol('id');
    const o = { a: 1, [s]: 2 };
    console.log('RFL‑4:', Reflect.ownKeys(o)); // ['a', Symbol(id)]
  })();
  
  /* RFL‑Example‑5: Custom metadata via WeakMap keyed by object */
  (function () {
    const meta = new WeakMap();
    const addMeta = (obj, k, v) => {
      const store = meta.get(obj) || meta.set(obj, {}).get(obj);
      store[k] = v;
    };
    const getMeta = (obj, k) => (meta.get(obj) || {})[k];
    const o = {};
    addMeta(o, 'role', 'service');
    console.log('RFL‑5:', getMeta(o, 'role'));
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION PRX — Proxy Objects & Traps                              */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* PRX‑Example‑1: Logging property access */
  (function () {
    const logger = new Proxy({}, {
      get(t, p, r) { console.log('get', p); return Reflect.get(t, p, r); },
      set(t, p, v, r) { console.log('set', p, v); return Reflect.set(t, p, v, r); }
    });
    logger.a = 1; console.log(logger.a);
  })();
  
  /* PRX‑Example‑2: Negative array indices */
  (function () {
    const arr = new Proxy([], {
      get(t, p) { return p < 0 ? t[t.length + +p] : t[p]; }
    });
    arr.push(10, 11, 12);
    console.log('PRX‑2:', arr[-1]); // 12
  })();
  
  /* PRX‑Example‑3: Validation trap */
  (function () {
    function validator(obj, schema) {
      return new Proxy(obj, {
        set(t, p, v) {
          if (!schema[p](v)) throw TypeError('invalid ' + p);
          return Reflect.set(t, p, v);
        }
      });
    }
    const user = validator({}, { age: n => n >= 0 });
    user.age = 30;
    try { user.age = -5; } catch (e) { console.log('PRX‑3:', e.message); }
  })();
  
  /* PRX‑Example‑4: Revocable proxy for capability revoke */
  (function () {
    const { proxy, revoke } = Proxy.revocable({ secret: 42 }, {});
    console.log('PRX‑4a:', proxy.secret);
    revoke();
    try { console.log(proxy.secret); } catch (e) { console.log('PRX‑4b:', e.name); }
  })();
  
  /* PRX‑Example‑5: Auto‑memoized function proxy */
  (function () {
    const memo = fn => new Proxy(fn, {
      cache: new Map(),
      apply(t, _, args) {
        const key = JSON.stringify(args);
        const c = this.cache;
        if (!c.has(key)) c.set(key, Reflect.apply(t, _, args));
        return c.get(key);
      }
    });
    const slow = n => (console.log('calc'), n ** 2);
    const fast = memo(slow);
    fast(5); fast(5); // 'calc' logs once
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION DEC — Decorators (TC39 stage‑3 proposal, transpile req.) */
  /*───────────────────────────────────────────────────────────────────*/
   
  /* Using explicit functions to simulate decorator behaviour without syntax sugar. */
  (function () {
    const readonly = (target, key, desc) => ({ ...desc, writable: false });
    const log      = (target, key, desc) => {
      const fn = desc.value;
      return { ...desc, value: function (...args) {
        console.log(`DEC‑log ${key}`, args);
        return fn.apply(this, args);
      }};
    };
  
    function decorate(cls, prop, decorators) {
      let desc = Object.getOwnPropertyDescriptor(cls.prototype, prop);
      decorators.reverse().forEach(d => desc = d(cls.prototype, prop, desc));
      Object.defineProperty(cls.prototype, prop, desc);
    }
  
    class Service {
      run(a) { return a * 2; }
      version = '1.0';
    }
    decorate(Service, 'run', [log]);
    decorate(Service, 'version', [readonly]);
  
    const svc = new Service();
    svc.run(3);
    try { svc.version = '2.0'; } catch (e) { console.log('DEC‑1 readonly'); }
  })();
  
  /* DEC‑Example‑2: Class decorator adding metadata */
  (function () {
    const addTag = tag => cls => (cls.tag = tag, cls);
    function applyClassDec(dec, cls) { return dec(cls); }
  
    class Model {}
    applyClassDec(addTag('db'), Model);
    console.log('DEC‑2:', Model.tag);
  })();
  
  /* DEC‑Example‑3: Method deprecation decorator */
  (function () {
    const deprecate = msg => (_, key, desc) => ({
      ...desc,
      value(...a) {
        console.warn(`DEC‑3 WARNING: ${key} is deprecated. ${msg}`);
        return desc.value.apply(this, a);
      }
    });
    function decor(cls, prop, dec) {
      let d = Object.getOwnPropertyDescriptor(cls.prototype, prop);
      d = dec(cls.prototype, prop, d);
      Object.defineProperty(cls.prototype, prop, d);
    }
    class API { old() { return 1; } }
    decor(API, 'old', deprecate('Use new() instead.'));
    new API().old();
  })();
  
  /* DEC‑Example‑4: Memoize decorator */
  (function () {
    const memoize = (_, __, desc) => {
      const fn = desc.value, cache = new Map();
      return { ...desc, value(...a) {
        const k = JSON.stringify(a);
        return cache.has(k) ? cache.get(k) : cache.set(k, fn.apply(this, a)).get(k);
      }};
    };
    function apply(cls, prop, dec) {
      const d = dec(cls.prototype, prop, Object.getOwnPropertyDescriptor(cls.prototype, prop));
      Object.defineProperty(cls.prototype, prop, d);
    }
    class Fib {
      fib(n) { return n < 2 ? n : this.fib(n - 1) + this.fib(n - 2); }
    }
    apply(Fib, 'fib', memoize);
    console.log('DEC‑4:', new Fib().fib(35));
  })();
  
  /* DEC‑Example‑5: Property transform decorator */
  (function () {
    const upper = (_, key, desc) => ({
      get() { return desc.get.call(this).toUpperCase(); },
      set(v) { desc.set.call(this, v); }
    });
    function decorateProp(cls, key, d) {
      const desc = d(cls.prototype, key, Object.getOwnPropertyDescriptor(cls.prototype, key));
      Object.defineProperty(cls.prototype, key, desc);
    }
    class User {
      #name = '';
      get name() { return this.#name; }
      set name(v) { this.#name = v; }
    }
    decorateProp(User, 'name', upper);
    const u = new User(); u.name = 'alice'; console.log('DEC‑5:', u.name);
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION DYN — Dynamic Code (eval, new Function)                  */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* DYN‑Example‑1: Simple expression via eval */
  (function () {
    const src = '2 + 3 * 4';
    console.log('DYN‑1:', eval(src));
  })();
  
  /* DYN‑Example‑2: Sandboxed Function constructor */
  (function () {
    const safeEval = expr => Function('"use strict";return (' + expr + ')')();
    console.log('DYN‑2:', safeEval('Math.max(7,3)'));
  })();
  
  /* DYN‑Example‑3: Runtime created class */
  (function () {
    const ClassFactory = name => new Function(`return class ${name} { constructor(v){this.v=v;} }`)();
    const Foo = ClassFactory('Foo');
    console.log('DYN‑3:', new Foo(9).v);
  })();
  
  /* DYN‑Example‑4: Memoized Function generator */
  (function () {
    const cache = new Map();
    const compile = expr => cache.get(expr) || cache.set(expr, new Function('x', `return ${expr};`)).get(expr);
    const square = compile('x*x');
    console.log('DYN‑4:', square(8));
  })();
  
  /* DYN‑Example‑5: Self‑modifying code (monkey patch) */
  (function () {
    const obj = { get v() { return 1; } };
    eval('obj.get=function(){return 2;}');
    console.log('DYN‑5:', obj.get());
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION AST — AST Manipulation & Babel Plugin Authoring          */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* AST‑Example‑1: Minimal Babel plugin turning === into Object.is */
  const eqToObjectIsPlugin = () => ({
    visitor: {
      BinaryExpression(path) {
        if (path.node.operator === '===') {
          path.node.operator = 'ObjectIs';
        }
      }
    }
  });
  console.log('AST‑1 plugin skeleton created');
  
  /* AST‑Example‑2: Auto‑inject console.log after function declarations */
  const logInjection = () => ({
    visitor: {
      FunctionDeclaration(path) {
        const name = path.node.id.name;
        path.get('body').unshiftContainer('body',
          t.expressionStatement(
            t.callExpression(t.memberExpression(t.identifier('console'), t.identifier('log')), [
              t.stringLiteral(`enter ${name}`)
            ])
          )
        );
      }
    }
  });
  console.log('AST‑2 plugin skeleton created');
  
  /* AST‑Example‑3: Collecting strings for i18n */
  const i18nCollector = messages => () => ({
    visitor: {
      StringLiteral({ node }) { messages.add(node.value); }
    }
  });
  console.log('AST‑3 plugin skeleton created');
  
  /* AST‑Example‑4: Stripping debug code (process.env.DEBUG guards) */
  const stripDebug = () => ({
    visitor: {
      IfStatement(path) {
        const test = path.get('test');
        if (test.isMemberExpression() &&
            test.get('object').isIdentifier({ name: 'process' }) &&
            test.get('property').isIdentifier({ name: 'env' })) {
          path.remove();
        }
      }
    }
  });
  console.log('AST‑4 plugin skeleton created');
  
  /* AST‑Example‑5: Babel transform runner (local) */
  (async () => {
    if (typeof require === 'function') {
      const babel = await import('@babel/core');
      const t     = await import('@babel/types');
      const src   = 'const x = 1 === 1;';
      const out   = await babel.transformAsync(src, { plugins: [eqToObjectIsPlugin] });
      console.log('AST‑5 transformed code:', out.code);
    }
  })();