/**************************************************************************************************
 *  Chapter 5 | Advanced Object‑Oriented Patterns
 *  -----------------------------------------------------------------------------------------------
 *  Single‑file reference & playground.  Copy/paste into browser console or Node (≥16 for #fields).
 *
 *  Sections
 *  ========
 *   1. Prototypes & Prototype Chain ................................................... SECTION PTC
 *   2. Classical vs. Prototypal Inheritance ........................................... SECTION INH
 *   3. Mixins, Composition & Trait Patterns ........................................... SECTION MIX
 *   4. Symbols, Private Fields & Static Members ....................................... SECTION SPS
 *   5. Factory & Builder Patterns ..................................................... SECTION FAB
 *
 *  Each subsection contains ≥5 runnable examples.
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION PTC ── Prototypes & Prototype Chain                                                 */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

/* PTC‑Example‑1:  Basic prototype lookup */
(function () {
    const proto = { greet() { return `hi ${this.name}`; } };
    const obj   = Object.create(proto);
    obj.name = 'Ada';
    console.log('PTC‑1:', obj.greet());                    // => hi Ada
  })();
  
  /* PTC‑Example‑2:  Object.create with property descriptors */
  (function () {
    const proto  = { role: 'engineer' };
    const bob = Object.create(proto, {
      name: { value: 'Bob', enumerable: true },
      age:  { value: 42,   enumerable: true }
    });
    console.log('PTC‑2:', bob.role, bob.name, bob.age);    // delegated role
  })();
  
  /* PTC‑Example‑3:  Shadowing vs. delete to reveal prototype property */
  (function () {
    const proto = { x: 1 };
    const obj   = Object.create(proto);
    obj.x = 2;
    console.log('PTC‑3a:', obj.x);                         // 2 (own)
    delete obj.x;
    console.log('PTC‑3b:', obj.x);                         // 1 (prototype)
  })();
  
  /* PTC‑Example‑4:  in‑operator vs. hasOwnProperty */
  (function () {
    const proto = { shared: true };
    const o = Object.create(proto);
    o.own = true;
    console.log('PTC‑4:', 'shared' in o, o.hasOwnProperty('shared')); // true, false
  })();
  
  /* PTC‑Example‑5:  Dynamic prototype mutation via Object.setPrototypeOf */
  (function () {
    const a = { aProp: 1 };
    const b = { bProp: 2 };
    const target = {};
    Object.setPrototypeOf(target, a);
    console.log('PTC‑5a:', target.aProp);                  // 1
    Object.setPrototypeOf(target, b);
    console.log('PTC‑5b:', target.bProp);                  // 2
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION INH ── Classical vs. Prototypal Inheritance                                          */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* INH‑Example‑1:  Classical (constructor + prototype) */
  (function () {
    function Person(name) { this.name = name; }
    Person.prototype.say = function () { return `I am ${this.name}`; };
    const e = new Person('Eve');
    console.log('INH‑1:', e.say());
  })();
  
  /* INH‑Example‑2:  ES6 class extends (syntactic sugar) */
  (function () {
    class Animal { speak() { return '...'; } }
    class Dog extends Animal {
      speak() { return super.speak() + ' woof'; }
    }
    console.log('INH‑2:', new Dog().speak());
  })();
  
  /* INH‑Example‑3:  Pure prototypal inheritance with Object.create */
  (function () {
    const vehicle = { wheels: 4 };
    const car = Object.create(vehicle);
    car.brand = 'Tesla';
    console.log('INH‑3:', car.wheels, car.brand);
  })();
  
  /* INH‑Example‑4:  Parasitic‑combination inheritance (pre‑ES6 optimum) */
  (function () {
    function Parent(v) { this.val = v; }
    Parent.prototype.get = function () { return this.val; };
    function Child(v, extra) {
      Parent.call(this, v);
      this.extra = extra;
    }
    Child.prototype = Object.create(Parent.prototype);
    Child.prototype.constructor = Child;
    console.log('INH‑4:', new Child(5, 'x').get());
  })();
  
  /* INH‑Example‑5:  Extending built‑ins (subclassing Array) */
  (function () {
    class Stack extends Array {
      top() { return this[this.length - 1]; }
    }
    const s = new Stack(1, 2, 3);
    console.log('INH‑5:', s.top(), s instanceof Array);    // 3 true
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION MIX ── Mixins, Composition & Trait Patterns                                          */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* MIX‑Example‑1:  Simple Object.assign mixin */
  (function () {
    const canFly  = { fly() { return 'flying'; } };
    const canSwim = { swim() { return 'swimming'; } };
    const penguin = Object.assign({}, canSwim); // penguins don't fly
    console.log('MIX‑1:', penguin.swim());
  })();
  
  /* MIX‑Example‑2:  Functional mixin returning enhanced object */
  (function () {
    const Timestamped = o => ({ ...o, ts: Date.now() });
    const obj = Timestamped({ id: 7 });
    console.log('MIX‑2:', obj.id, obj.ts);
  })();
  
  /* MIX‑Example‑3:  Trait conflict resolution */
  (function () {
    const t1 = { greet() { return 'hi'; } };
    const t2 = { greet() { return 'hello'; } };
    const combined = { ...t1, ...t2, greet() { return t1.greet() + '/' + t2.greet(); } };
    console.log('MIX‑3:', combined.greet());
  })();
  
  /* MIX‑Example‑4:  EventEmitter mixin */
  (function () {
    const Eventable = Base => class extends Base {
      #events = new Map();
      on(e, fn) { (this.#events.get(e) || this.#events.set(e, []).get(e)).push(fn); }
      emit(e, ...a) { (this.#events.get(e) || []).forEach(fn => fn(...a)); }
    };
    class Model extends Eventable(Object) { set(v) { this.val = v; this.emit('change', v); } }
    new Model().on('change', v => console.log('MIX‑4 changed', v)).set(12);
  })();
  
  /* MIX‑Example‑5:  Composing multiple behaviors with higher‑order classes */
  (function () {
    const Serializable = Base => class extends Base {
      toJSON() { return JSON.stringify(this); }
    };
    const Identifiable = Base => class extends Base {
      static lastId = 0;
      constructor(...args) { super(...args); this.id = ++Identifiable.lastId; }
    };
    class Point extends Identifiable(Serializable(Object)) { constructor(x, y) { super(); this.x = x; this.y = y; } }
    console.log('MIX‑5:', new Point(2, 3).toJSON());
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION SPS ── Symbols, Private Fields & Static Members                                      */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* SPS‑Example‑1:  Using Symbol as semi‑private property */
  (function () {
    const _secret = Symbol('secret');
    class Vault { constructor(v) { this[_secret] = v; } get secret() { return this[_secret]; } }
    console.log('SPS‑1:', new Vault(777).secret);
  })();
  
  /* SPS‑Example‑2:  #private fields (hard private, ES2022) */
  (function () {
    class Counter {
      #count = 0;
      inc() { return ++this.#count; }
    }
    const c = new Counter();
    console.log('SPS‑2:', c.inc(), /* c.#count -> SyntaxError */);
  })();
  
  /* SPS‑Example‑3:  Static members */
  (function () {
    class MathX { static PI2 = Math.PI * 2; static circleLen(r) { return r * this.PI2; } }
    console.log('SPS‑3:', MathX.circleLen(3));
  })();
  
  /* SPS‑Example‑4:  Custom iterable via Symbol.iterator */
  (function () {
    class Range {
      constructor(a, b) { this.a = a; this.b = b; }
      [Symbol.iterator]() {
        let i = this.a;
        return { next: () => ({ value: i, done: i++ > this.b }) };
      }
    }
    console.log('SPS‑4:', [...new Range(1, 3)]);
  })();
  
  /* SPS‑Example‑5:  Static #private (stage‑3, supported in Node 20+) */
  (function () {
    class Registry {
      static #map = new Map();
      static set(k, v) { this.#map.set(k, v); }
      static get(k) { return this.#map.get(k); }
    }
    Registry.set('x', 9);
    console.log('SPS‑5:', Registry.get('x'));
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION FAB ── Factory & Builder Patterns                                                    */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* FAB‑Example‑1:  Simple factory function */
  (function () {
    function createUser(role) { return { role, created: Date.now() }; }
    console.log('FAB‑1:', createUser('admin'));
  })();
  
  /* FAB‑Example‑2:  Abstract factory producing themed widgets */
  (function () {
    const LightFactory = { button: () => ({ bg: '#fff', color: '#000' }) };
    const DarkFactory  = { button: () => ({ bg: '#000', color: '#fff' }) };
    const Theme = theme => (theme === 'dark' ? DarkFactory : LightFactory);
    console.log('FAB‑2:', Theme('dark').button());
  })();
  
  /* FAB‑Example‑3:  Factory method inside class */
  (function () {
    class Logger {
      constructor(level) { this.level = level; }
      static get(level) { return new Logger(level); }
    }
    console.log('FAB‑3:', Logger.get('debug') instanceof Logger);
  })();
  
  /* FAB‑Example‑4:  Builder pattern with chaining */
  (function () {
    class QueryBuilder {
      #parts = { sel: '*', from: '', where: '' };
      select(cols) { this.#parts.sel = cols; return this; }
      from(tbl)   { this.#parts.from = tbl; return this; }
      where(cl)   { this.#parts.where = cl; return this; }
      toString()  { return `SELECT ${this.#parts.sel} FROM ${this.#parts.from}` +
                           (this.#parts.where && ` WHERE ${this.#parts.where}`); }
    }
    console.log('FAB‑4:', new QueryBuilder().select('id,name').from('users').where('id=1').toString());
  })();
  
  /* FAB‑Example‑5:  Immutable builder returning new instance each step */
  (function () {
    class URLBuilder {
      constructor({ protocol = 'https', host = '', path = '' } = {}) { Object.assign(this, { protocol, host, path }); }
      withHost(h)  { return new URLBuilder({ ...this, host: h }); }
      withPath(p)  { return new URLBuilder({ ...this, path: p }); }
      toString()   { return `${this.protocol}://${this.host}/${this.path}`; }
    }
    console.log('FAB‑5:', new URLBuilder().withHost('example.com').withPath('docs').toString());
  })();