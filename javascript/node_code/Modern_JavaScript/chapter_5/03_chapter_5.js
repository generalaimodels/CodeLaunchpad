/****************************************************************************************
* Chapter-5 | Advanced Object-Oriented Patterns                                          *
*                                                                                       *
* 25 SELF-CONTAINED EXAMPLES (5 PER SECTION).                                           *
* Run in Node ≥16 or modern browsers (private fields & static blocks supported).        *
****************************************************************************************/

/*────────────────────────────────────────────────────────────────────────────────────────
1. PROTOTYPES & PROTOTYPE CHAIN – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 1-A BASIC PROTOTYPE LINKING ----------------------------------------------------------*/
function Animal(name) { this.name = name; }
Animal.prototype.speak = function () { return `${this.name} makes a noise`; };
const a1 = new Animal('Dog');
console.log(a1.speak()); // Dog makes a noise
console.log(Object.getPrototypeOf(a1) === Animal.prototype); // true

/* 1-B OBJECT.CREATE FOR PURE PROTOTYPAL INHERITANCE -----------------------------------*/
const protoCar = { wheels: 4, drive() { return `Driving with ${this.wheels} wheels`; } };
const myCar = Object.create(protoCar, { brand: { value: 'Tesla', writable: false } });
console.log(myCar.drive()); // Driving with 4 wheels

/* 1-C DYNAMICALLY EXTENDING THE PROTOTYPE CHAIN ---------------------------------------*/
Animal.prototype.eat = function () { return `${this.name} eats`; };
console.log(a1.eat()); // Dog eats

/* 1-D PROTOTYPE SHADOWING --------------------------------------------------------------*/
const a2 = new Animal('Cat');
a2.speak = function () { return `${this.name} meows`; }; // shadows prototype method
console.log(a2.speak()); // Cat meows
console.log(a1.speak()); // Dog makes a noise (unaffected)

/* 1-E READING THE FULL CHAIN -----------------------------------------------------------*/
let p = a1;
const chain = [];
while (p) { chain.push(p.constructor?.name ?? 'Object'); p = Object.getPrototypeOf(p); }
console.log(chain); // [ 'Animal', 'Object' ]

/*────────────────────────────────────────────────────────────────────────────────────────
2. CLASSICAL vs. PROTOTYPAL INHERITANCE – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 2-A CLASS SYNTAX (ES2015) – "CLASSICAL" FLAVOR --------------------------------------*/
class Person {
  constructor(name) { this.name = name; }
  greet() { return `Hi, I'm ${this.name}`; }
}
class Employee extends Person {
  constructor(name, role) { super(name); this.role = role; }
  work() { return `${this.name} works as ${this.role}`; }
}
const e1 = new Employee('Alice', 'Engineer');
console.log(e1.greet()); // Hi, I'm Alice

/* 2-B FUNCTION-CONSTRUCTOR + .CALL INHERITANCE ----------------------------------------*/
function Shape(color) { this.color = color; }
function Circle(color, radius) {
  Shape.call(this, color);      // classical super()
  this.radius = radius;
}
Circle.prototype = Object.create(Shape.prototype);
Circle.prototype.area = function () { return Math.PI * this.radius ** 2; };
const c1 = new Circle('red', 10);
console.log(c1.color, c1.area());

/* 2-C PURE PROTOTYPAL (DELEGATION) -----------------------------------------------------*/
const protoQueue = {
  items: [],
  enqueue(x) { this.items.push(x); },
  dequeue() { return this.items.shift(); }
};
const q1 = Object.create(protoQueue);
q1.enqueue(1); q1.enqueue(2);
console.log(q1.dequeue()); // 1

/* 2-D PARASITIC COMBINATION INHERITANCE (OPTIMIZED) -----------------------------------*/
function inherit(child, parent) {
  child.prototype = Object.create(parent.prototype, { constructor: { value: child } });
}
function Vehicle(type) { this.type = type; }
Vehicle.prototype.info = function () { return `Type=${this.type}`; };
function Boat(name) { Vehicle.call(this, 'boat'); this.name = name; }
inherit(Boat, Vehicle);
Boat.prototype.sail = function () { return `${this.name} sails`; };
console.log(new Boat('Titanic').sail());

/* 2-E WHEN TO CHOOSE WHICH -------------------------------------------------------------*/
console.log(`
Class syntax ➜ familiar, static checks, but fixed hierarchy.
Prototype delegation ➜ lightweight, dynamic extension, ideal for mix-ins.
`);

/*────────────────────────────────────────────────────────────────────────────────────────
3. MIXINS, COMPOSITION & TRAIT PATTERNS – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 3-A SIMPLE OBJECT MIXIN --------------------------------------------------------------*/
const canFly = { fly() { return `${this.name} flies`; } };
const canSwim = { swim() { return `${this.name} swims`; } };
const duck = { name: 'Duck' };
Object.assign(duck, canFly, canSwim);
console.log(duck.fly(), duck.swim());

/* 3-B FUNCTIONAL MIXIN (PRIVATES VIA CLOSURE) -----------------------------------------*/
const withTimestamp = (o = {}) => {
  const created = Date.now();
  return Object.assign(o, {
    get created() { return created; }
  });
};
const doc = withTimestamp({ title: 'README' });
console.log(doc.created);

/* 3-C TRAIT WITH CONFLICT DETECTION ----------------------------------------------------*/
function compose(base, ...traits) {
  const dupe = {};
  for (const t of traits)
    for (const k of Object.keys(t))
      if (base[k] || dupe[k]) throw Error(`Conflict on ${k}`);
      else dupe[k] = t[k];
  return Object.assign(Object.create(base), ...traits);
}
const traitA = { foo() { return 'A'; } };
const traitB = { bar() { return 'B'; } };
// const bad = compose({}, traitA, { foo(){} }); // throws
const good = compose({}, traitA, traitB);
console.log(good.foo(), good.bar());

/* 3-D COMPOSITION OVER INHERITANCE (HAS-A) -------------------------------------------*/
class Engine { start() { return 'engine started'; } }
class Car {
  constructor() { this.engine = new Engine(); }
  drive() { return this.engine.start() + ' & car moving'; }
}
console.log(new Car().drive());

/* 3-E DECORATOR MIXIN AT RUNTIME -------------------------------------------------------*/
function Logger(base) {
  return class extends base {
    log(msg) { console.log(`[${this.constructor.name}]`, msg); }
  };
}
class Service {}
class LoggedService extends Logger(Service) {}
new LoggedService().log('running');

/*────────────────────────────────────────────────────────────────────────────────────────
4. SYMBOLS, PRIVATE FIELDS & STATIC MEMBERS – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 4-A UNIQUE PROPERTY KEYS WITH SYMBOL -------------------------------------------------*/
const SECRET = Symbol('secret');
class Vault {
  constructor(code) { this[SECRET] = code; }
  check(x) { return x === this[SECRET]; }
}
console.log(new Vault(123).check(123));

/* 4-B PRIVATE FIELDS (#) --------------------------------------------------------------*/
class Counter {
  #count = 0;                              // truly private
  inc() { return ++this.#count; }
}
const ctr = new Counter();
console.log(ctr.inc(), ctr.inc());
// console.log(ctr.#count); // SyntaxError

/* 4-C STATIC PROPERTIES & BLOCK --------------------------------------------------------*/
class Config {
  static #instances = 0;
  constructor() { Config.#instances++; }
  static get instances() { return Config.#instances; }
  static { console.log('Config class loaded'); }
}
new Config(); new Config();
console.log(Config.instances);

/* 4-D SYMBOL-BASED ITERATOR ------------------------------------------------------------*/
class Range {
  constructor(a, b) { this.from = a; this.to = b; }
  [Symbol.iterator]() {
    let v = this.from;
    return { next: () => ({ value: v, done: v++ > this.to }) };
  }
}
console.log([...new Range(1, 3)]); // [1,2,3]

/* 4-E WELL-KNOWN SYMBOL: toStringTag ----------------------------------------------------*/
class Matrix { get [Symbol.toStringTag]() { return 'Matrix'; } }
console.log(Object.prototype.toString.call(new Matrix())); // [object Matrix]

/*────────────────────────────────────────────────────────────────────────────────────────
5. FACTORY & BUILDER PATTERNS – 5 EXAMPLES
────────────────────────────────────────────────────────────────────────────────────────*/

/* 5-A SIMPLE FACTORY -------------------------------------------------------------------*/
function createUser(role) {
  return role === 'admin'
    ? { canDelete: true, role }
    : { canDelete: false, role };
}
console.log(createUser('admin').canDelete);

/* 5-B FACTORY METHOD INSIDE CLASS ------------------------------------------------------*/
class Dialog {
  static create(type) {
    switch (type) {
      case 'alert':  return new AlertDialog();
      case 'confirm': return new ConfirmDialog();
      default: throw Error('unknown');
    }
  }
}
class AlertDialog extends Dialog { }
class ConfirmDialog extends Dialog { }
console.log(Dialog.create('confirm') instanceof ConfirmDialog);

/* 5-C ABSTRACT FACTORY (UI WIDGETS) ----------------------------------------------------*/
class WinButton { render() { return 'WinButton'; } }
class OSXButton { render() { return 'OSXButton'; } }
function widgetFactory(os) {
  return {
    createButton() { return os === 'win' ? new WinButton() : new OSXButton(); }
  };
}
console.log(widgetFactory('osx').createButton().render());

/* 5-D BUILDER – FLUENT API -------------------------------------------------------------*/
class QueryBuilder {
  #q = { select: '*', where: [], limit: 0 };
  select(cols) { this.#q.select = cols.join(','); return this; }
  where(clause) { this.#q.where.push(clause); return this; }
  limit(n) { this.#q.limit = n; return this; }
  build() {
    const {select, where, limit} = this.#q;
    return `SELECT ${select} WHERE ${where.join(' AND ')} LIMIT ${limit}`;
  }
}
const query = new QueryBuilder().select(['id','name']).where('age>18').limit(10).build();
console.log(query);

/* 5-E BUILDER + DIRECTOR (SEPARATION) --------------------------------------------------*/
class Pizza {
  constructor() { this.toppings = []; }
}
class PizzaBuilder {
  constructor() { this.pizza = new Pizza(); }
  addCheese() { this.pizza.toppings.push('cheese'); return this; }
  addPepperoni() { this.pizza.toppings.push('pepperoni'); return this; }
  getResult() { return this.pizza; }
}
function cookPepperoniPizza() {
  return new PizzaBuilder().addCheese().addPepperoni().getResult();
}
console.log(cookPepperoniPizza().toppings);