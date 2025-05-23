/****************************************************************************************
* Chapter-5 | Advanced Object-Oriented Patterns (25 Stand-Alone Examples, 5 / section)  *
****************************************************************************************/

/*────────────────────────────────────────────────────────────────────────────────────────
1. PROTOTYPES & PROTOTYPE CHAIN – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 1-A BASIC PROTOTYPE INHERITANCE */
function Animal(name) { this.name = name; }
Animal.prototype.speak = function () { return `${this.name} makes a noise`; };
const cat = new Animal('Kitty');
console.log(cat.speak()); // Kitty makes a noise

/* 1-B PROPERTY OVERRIDE & SHADOWING */
Animal.prototype.legs = 4;
const dog = new Animal('Doggo');
dog.legs = 3;                      // own prop shadows prototype
console.log(dog.legs, cat.legs);   // 3 4
delete dog.legs;                   // fallback to prototype
console.log(dog.legs);             // 4

/* 1-C Object.create FOR PURE PROTOTYPAL STYLE */
const vehicleProto = { wheels: 4 };
const car = Object.create(vehicleProto);
car.brand = 'Tesla';
console.log(car.wheels);           // 4 (delegated)
console.log(Object.getPrototypeOf(car) === vehicleProto); // true

/* 1-D CHAIN QUERY: hasOwnProperty VS. "in" */
console.log('brand' in car);           // true (checks chain)
console.log(car.hasOwnProperty('brand'));  // true
console.log('wheels' in car);          // true (prototype)
console.log(car.hasOwnProperty('wheels')); // false

/* 1-E MUTATING PROTOTYPE AT RUNTIME (ANTI-PATTERN) */
Animal.prototype.speak = function () { return `${this.name} says hi`; }; // affects all
console.log(cat.speak());            // Kitty says hi
// NOTE: Modifying after instances exist can break expectations.

/*────────────────────────────────────────────────────────────────────────────────────────
2. CLASSICAL VS. PROTOTYPAL INHERITANCE – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 2-A ES6 CLASS (CLASSICAL) */
class Person {
  constructor(n) { this.name = n; }
  greet() { return `Hi ${this.name}`; }
}
class Employee extends Person {
  constructor(n, id) { super(n); this.id = id; }
  greet() { return `${super.greet()} (#${this.id})`; }
}
console.log(new Employee('Eva', 7).greet());

/* 2-B PRE-ES6 CONSTRUCTOR "CLASSES" */
function Parent(v) { this.val = v; }
Parent.prototype.get = function () { return this.val; };
function Child(v) { Parent.call(this, v * 2); }
Child.prototype = Object.create(Parent.prototype);
Child.prototype.constructor = Child;
console.log(new Child(3).get()); // 6

/* 2-C PURE PROTOTYPAL (NO "this", NO "new") */
const point = (x, y) => ({ x, y, toString() { return `${x},${y}`; } });
const p = point(4, 5);
console.log(p.toString());

/* 2-D PARASITIC INHERITANCE (A MIX OF BOTH) */
function createAugmented(original) {
  const clone = Object.create(original);
  clone.extra = () => 'extra';
  return clone;
}
const aug = createAugmented(point(0,0));
console.log(aug.toString(), aug.extra());

/* 2-E PERFORMANCE NOTE: LOOKUP COST */
console.time('direct');
const obj = {x:1};
for(let i=0;i<1e6;i++) obj.x;
console.timeEnd('direct');

console.time('prototype');
function Proto(){ }
Proto.prototype.x = 1;
const protoObj = new Proto();
for(let i=0;i<1e6;i++) protoObj.x;
console.timeEnd('prototype'); // usually slightly slower due to lookup

/*────────────────────────────────────────────────────────────────────────────────────────
3. MIXINS, COMPOSITION & TRAIT PATTERNS – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 3-A SIMPLE MIXIN WITH Object.assign */
const canFly   = { fly()   { return 'flying'; } };
const canSwim  = { swim()  { return 'swimming'; } };
const Duck = function(){};
Object.assign(Duck.prototype, canFly, canSwim);
console.log(new Duck().fly(), new Duck().swim());

/* 3-B FUNCTIONAL MIXIN RETURNING NEW OBJECT */
const Timestamped = (o={}) => ({
  ...o,
  created: Date.now(),
  age() { return Date.now() - this.created; }
});
const obj1 = Timestamped({x:10});
console.log(obj1.age());

/* 3-C TRAIT WITH CONFLICT DETECTION */
function composeTraits(...traits) {
  const result = {};
  for (const t of traits) {
    for (const k of Object.keys(t)) {
      if (k in result) throw new Error(`Conflict on ${k}`);
      result[k] = t[k];
    }
  }
  return result;
}
const foo = {hello(){return 'foo';}};
const bar = {bye(){return 'bar';}};
const composite = composeTraits(foo, bar);
console.log(composite.hello(), composite.bye());

/* 3-D CLASS MIXIN HELPER */
const mix = (base, ...mixins) => mixins.reduce((c, m) => m(c), base);

const Serializable = Base => class extends Base {
  toJSON() { return JSON.stringify(this); }
};
class Point { constructor(x,y){this.x=x;this.y=y;} }
class JsonPoint extends mix(Point, Serializable) {}
console.log(new JsonPoint(1,2).toJSON());

/* 3-E STAMP (FACTORY + MIXINS) */
const stamp = (methods, state={}) => (props={}) =>
  Object.assign(Object.create(methods), state, props);
const Talker = stamp({say(){return this.msg;}}, {msg:'hi'});
console.log(Talker({msg:'hello'}).say());

/*────────────────────────────────────────────────────────────────────────────────────────
4. SYMBOLS, PRIVATE FIELDS & STATIC MEMBERS – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 4-A UNIQUE PROPERTY KEYS VIA Symbol */
const ID = Symbol('id');
const user = { [ID]: 123, name: 'Alice' };
console.log(user[ID]);           // 123
console.log(Object.keys(user));  // ['name'] – symbol not enumerated

/* 4-B CLASS WITH #PRIVATE FIELDS (TC39 Stage-4) */
class Counter {
  #val = 0;
  inc() { return ++this.#val; }
}
const c = new Counter();
console.log(c.inc());
// console.log(c.#val); // SyntaxError: private field

/* 4-C STATIC METHODS & PROPS */
class MathUtil {
  static PI = 3.14159;
  static area(r) { return MathUtil.PI * r * r; }
}
console.log(MathUtil.area(2));

/* 4-D STATIC #PRIVATE FIELD */
class Registry {
  static #store = new Map();
  static set(k,v){ Registry.#store.set(k,v); }
  static get(k){ return Registry.#store.get(k); }
}
Registry.set('token','abc');
console.log(Registry.get('token'));

/* 4-E Symbol.iterator FOR CUSTOM ITERABLE */
class Range {
  constructor(a,b){ this.a=a; this.b=b; }
  *[Symbol.iterator](){ for(let i=this.a;i<=this.b;i++) yield i; }
}
console.log([...new Range(1,5)]); // [1,2,3,4,5]

/*────────────────────────────────────────────────────────────────────────────────────────
5. FACTORY & BUILDER PATTERNS – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 5-A SIMPLE FACTORY */
function createLogger(type='info') {
  return { log: msg => console.log(`[${type}]`, msg) };
}
createLogger('warn').log('disk almost full');

/* 5-B ABSTRACT FACTORY */
const UIFactory = (() => {
  const html = { button: () => '<button>OK</button>' };
  const svg  = { button: () => '<svg><rect/></svg>' };
  return theme => theme === 'svg' ? svg : html;
})();
console.log(UIFactory('svg').button());

/* 5-C BUILDER WITH METHOD CHAINING */
class PizzaBuilder {
  constructor(){ this.pizza={}; }
  size(s){ this.pizza.size=s; return this; }
  topping(t){ (this.pizza.toppings??=[]).push(t); return this; }
  build(){ return this.pizza; }
}
const pizza = new PizzaBuilder().size('L').topping('cheese').topping('bacon').build();
console.log(pizza);

/* 5-D FLUENT BUILDER WITH VALIDATION */
class QueryBuilder {
  #where=[];
  filter(f){ this.#where.push(f); return this; }
  exec(arr){ return arr.filter(x=>this.#where.every(fn=>fn(x))); }
}
const result = new QueryBuilder()
  .filter(x=>x>3)
  .filter(x=>x%2===0)
  .exec([1,2,3,4,5,6]);
console.log(result); // [4,6]

/* 5-E FACTORY DECIDES CLASS AT RUNTIME */
class JsonStorage { save(d){ console.log('JSON',JSON.stringify(d)); } }
class XmlStorage  { save(d){ console.log('XML',   `<data>${d}</data>`); } }
function StorageFactory(fmt){
  return fmt==='xml' ? new XmlStorage() : new JsonStorage();
}
StorageFactory('xml').save('hello');