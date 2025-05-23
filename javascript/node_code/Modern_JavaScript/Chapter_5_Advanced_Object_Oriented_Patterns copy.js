// Chapter 5: Advanced Object Oriented Patterns

// 1. Prototypes & Prototype Chain
// --------------------------------
// Example 1: Literal object and __proto__
const animal = { eats: true };
const rabbit = { jumps: true };
rabbit.__proto__ = animal;
console.log(rabbit.jumps, rabbit.eats); // true, true

// Example 2: Constructor + prototype link
function Person(name) { this.name = name; }
Person.prototype.greet = function() { return `Hello, ${this.name}`; };
const alice = new Person('Alice');
console.log(alice.greet(), Person.prototype.isPrototypeOf(alice)); // Hello, Alice, true

// Example 3: Object.create for explicit prototype
const proto = { describe() { return `Type: ${this.type}`; } };
const obj = Object.create(proto);
obj.type = 'Widget';
console.log(obj.describe(), Object.getPrototypeOf(obj) === proto);

// Example 4: Dynamic prototype extension
function Point(x, y) { this.x = x; this.y = y; }
const p = new Point(1,2);
Point.prototype.toString = function() { return `(${this.x},${this.y})`; };
console.log(p.toString()); // works on existing instance

// Example 5: Property lookup and shadowing
const base = { val: 1 };
const child = Object.create(base);
console.log(child.val); // 1 (inherited)
child.val = 5;         // shadows prototype
console.log(child.val, base.val); // 5, 1


// 2. Classical vs. Prototypal Inheritance
// ---------------------------------------
// Example 1: Classical (constructor + prototype chain)
function Animal(name) { this.name = name; }
Animal.prototype.speak = function() { return `${this.name} makes noise`; };
function Dog(name) { Animal.call(this,name); }
Dog.prototype = Object.create(Animal.prototype);
Dog.prototype.constructor = Dog;
Dog.prototype.speak = function() { return `${this.name} barks`; };
const d = new Dog('Rex');
console.log(d.speak(), d instanceof Animal, d instanceof Dog);

// Example 2: ES6 class inheritance
class Vehicle {
  constructor(make) { this.make = make; }
  info() { return `Make: ${this.make}`; }
}
class Car extends Vehicle {
  constructor(make, model) { super(make); this.model = model; }
  info() { return `${super.info()}, Model: ${this.model}`; }
}
console.log(new Car('Toyota','Camry').info());

// Example 3: Pure prototypal (delegation)
const flyer = { fly() { return `${this.name} flies`; } };
function makeBird(name) {
  return Object.assign(Object.create(flyer), { name });
}
console.log(makeBird('Sparrow').fly());

// Example 4: Mix prototypes at runtime
function swim() { return `${this.name} swims`; }
const fish = Object.create({}); fish.name = 'Goldfish';
Object.setPrototypeOf(fish, { __proto__: fish.__proto__, swim });
console.log(fish.swim());

// Example 5: Copy vs. delegate differences
const source = { a:1, b:2 };
const delegate = Object.create(source);
const copy = Object.assign({}, source);
console.log(delegate.a, copy.a);
source.a = 5;
console.log(delegate.a, copy.a); // delegate sees new value, copy does not


// 3. Mixins, Composition & Trait Patterns
// ----------------------------------------
// Example 1: Simple mixin via Object.assign
const canEat = { eat() { return `${this.name} eats`; } };
const canWalk = { walk() { return `${this.name} walks`; } };
const person = Object.assign({ name:'Bob' }, canEat, canWalk);
console.log(person.eat(), person.walk());

// Example 2: Functional composition
function eater(state) { return { eat: () => `${state.name} eats` }; }
function flyer(state) { return { fly: () => `${state.name} flies` }; }
function createBird(name) {
  const state = { name };
  return Object.assign(state, eater(state), flyer(state));
}
console.log(createBird('Eagle').fly());

// Example 3: Traits with conflict resolution
const traitA = { action() { return 'A'; } };
const traitB = { action() { return 'B'; } };
function useTraits(target, ...traits) {
  traits.forEach(trait => {
    Object.keys(trait).forEach(k => {
      if (!target[k]) target[k] = trait[k];
    });
  });
}
const objTraits = { };
useTraits(objTraits, traitA, traitB);
console.log(objTraits.action()); // 'A' (traitB skipped)

// Example 4: Functional mixin with private state
function counterMixin(state) {
  let count = 0;
  return {
    inc() { count++; return count; },
    dec() { count--; return count; },
    get() { return count; }
  };
}
const c = Object.assign({ name:'ctr' }, counterMixin());
console.log(c.inc(), c.inc(), c.get());

// Example 5: Multiple mixins and namespacing
const draggable = { drag() { return `${this.name} dragging`; } };
const droppable = { drop() { return `${this.name} dropping`; } };
const widget = { name:'W' };
Object.assign(widget, { draggable, droppable });
console.log(widget.draggable.drag.call(widget), widget.droppable.drop.call(widget));


// 4. Symbols, Private Fields & Static Members
// --------------------------------------------
// Example 1: Symbol-based “private” property
const _id = Symbol('id');
class Entity {
  constructor(id) { this[_id] = id; }
  getId() { return this[_id]; }
}
const e = new Entity(123);
console.log(e.getId(), Object.getOwnPropertySymbols(e));

// Example 2: ES2020 private fields
class Secret {
  #token;
  constructor(token) { this.#token = token; }
  reveal() { return this.#token; }
}
const s = new Secret('abc');
console.log(s.reveal()); // cannot access s.#token externally

// Example 3: Static methods and properties
class MathUtil {
  static PI = 3.14159;
  static circleArea(r) { return MathUtil.PI * r * r; }
}
console.log(MathUtil.PI, MathUtil.circleArea(2));

// Example 4: Combining static & private
class Counter {
  static #count = 0;
  constructor() { Counter.#count++; }
  static getCount() { return Counter.#count; }
}
new Counter(); new Counter();
console.log(Counter.getCount()); // 2

// Example 5: Errors on private field access
// try {
//   console.log(s.#token);
// } catch (err) {
//   console.log(err.message); // Private field '#token' must be declared in an enclosing class
// }


// 5. Factory & Builder Patterns
// ------------------------------
// Example 1: Simple factory function
function shapeFactory(type) {
  if (type === 'circle') return { draw: () => 'Drawing circle' };
  if (type === 'square') return { draw: () => 'Drawing square' };
  throw new Error('Unknown shape');
}
console.log(shapeFactory('circle').draw());

// Example 2: Configurable factory with defaults
function pizzaFactory(options) {
  const defaults = { size:'M', cheese:true, pepperoni:false };
  const config = { ...defaults, ...options };
  return {
    bake: () => `Baking ${config.size} pizza${config.cheese?' +cheese':''}${config.pepperoni?' +pepperoni':''}`
  };
}
console.log(pizzaFactory({ pepperoni:true }).bake());

// Example 3: Abstract factory for UI components
function buttonFactory(style) {
  return style === 'win' ? { render:()=>'WinBtn' } : { render:()=>'MacBtn' };
}
function checkboxFactory(style) {
  return style === 'win' ? { render:()=>'WinChk' } : { render:()=>'MacChk' };
}
function uiFactory(style) {
  return { createBtn:() => buttonFactory(style), createChk:() => checkboxFactory(style) };
}
const ui = uiFactory('mac');
console.log(ui.createBtn().render(), ui.createChk().render());

// Example 4: Builder pattern with chainable setters
class CarBuilder {
  constructor() { this._car = {}; }
  setMake(m) { this._car.make = m; return this; }
  setModel(m) { this._car.model = m; return this; }
  setYear(y) { this._car.year = y; return this; }
  build() { return this._car; }
}
const myCar = new CarBuilder().setMake('Tesla').setModel('S').setYear(2021).build();
console.log(myCar);

// Example 5: Nested builder for complex object
class HouseBuilder {
  constructor() { this.house = { rooms: [] }; }
  addRoom(name) { this.house.rooms.push({ name, features: [] }); return this; }
  addFeatureToLast(feature) {
    this.house.rooms[this.house.rooms.length - 1].features.push(feature);
    return this;
  }
  build() { return this.house; }
}
const home = new HouseBuilder()
  .addRoom('Living').addFeatureToLast('Fireplace')
  .addRoom('Kitchen').addFeatureToLast('Island')
  .build();
console.log(home);