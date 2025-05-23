// Chapter 5 | Advanced Object Oriented Patterns

// 1. Prototypes & Prototype Chain

// Example 1: Basic Prototype Inheritance
function Animal(name) {
    this.name = name;
}
Animal.prototype.speak = function() {
    return `${this.name} makes a noise.`;
};
const dog = new Animal('Rex');

// Example 2: Prototype Chain Lookup
function Mammal() {}
Mammal.prototype.breathe = function() { return 'Breathing'; };
const cat = new Mammal();
cat.__proto__.eat = function() { return 'Eating'; }; // Not recommended, for demonstration

// Example 3: Object.create for Prototype Assignment
const proto = { greet() { return 'Hello'; } };
const obj = Object.create(proto);
obj.name = 'Alice';

// Example 4: hasOwnProperty vs. Prototype Property
function Vehicle() {}
Vehicle.prototype.wheels = 4;
const car = new Vehicle();
car.color = 'red';
const own = car.hasOwnProperty('color'); // true
const protoProp = car.hasOwnProperty('wheels'); // false

// Example 5: Modifying Prototypes at Runtime
function Bird() {}
Bird.prototype.fly = function() { return 'Flying'; };
const sparrow = new Bird();
Bird.prototype.sing = function() { return 'Singing'; }; // All instances gain 'sing'

// 2. Classical vs. Prototypal Inheritance

// Example 1: Classical Inheritance with ES6 Classes
class Person {
    constructor(name) { this.name = name; }
    greet() { return `Hi, I'm ${this.name}`; }
}
class Employee extends Person {
    constructor(name, role) { super(name); this.role = role; }
    work() { return `${this.name} works as ${this.role}`; }
}
const emp = new Employee('Bob', 'Engineer');

// Example 2: Prototypal Inheritance with Object.create
const human = { species: 'Homo sapiens' };
const worker = Object.create(human);
worker.job = 'Developer';

// Example 3: Constructor Stealing (Classical)
function Parent(name) { this.name = name; }
function Child(name, age) {
    Parent.call(this, name);
    this.age = age;
}
const child = new Child('Eve', 10);

// Example 4: Prototype Chain Manipulation (Prototypal)
function A() {}
A.prototype.sayA = function() { return 'A'; };
function B() {}
B.prototype = Object.create(A.prototype);
B.prototype.sayB = function() { return 'B'; };
const b = new B();

// Example 5: ES6 Class Inheritance with Method Overriding
class Animal2 {
    speak() { return 'Generic sound'; }
}
class Dog extends Animal2 {
    speak() { return 'Woof'; }
}
const d = new Dog();

// 3. Mixins, Composition & Trait Patterns

// Example 1: Simple Mixin via Object.assign
const canSwim = { swim() { return 'Swimming'; } };
const canFly = { fly() { return 'Flying'; } };
const duck = Object.assign({}, canSwim, canFly);

// Example 2: Functional Mixins
function withJump(obj) {
    return Object.assign(obj, {
        jump() { return 'Jumping'; }
    });
}
const frog = withJump({});

// Example 3: Class Mixins
const CanRun = Base => class extends Base {
    run() { return 'Running'; }
};
class Animal3 {}
class Cheetah extends CanRun(Animal3) {}
const cheetah = new Cheetah();

// Example 4: Trait Pattern with Conflict Resolution
const traitA = { greet() { return 'Hello from A'; } };
const traitB = { greet() { return 'Hello from B'; } };
const composed = Object.assign({}, traitA, traitB); // traitB overwrites traitA

// Example 5: Composition over Inheritance
function compose(...behaviors) {
    return Object.assign({}, ...behaviors);
}
const canEat = { eat() { return 'Eating'; } };
const canSleep = { sleep() { return 'Sleeping'; } };
const person = compose(canEat, canSleep);

// 4. Symbols, Private Fields & Static Members

// Example 1: Using Symbols for Unique Property Keys
const sym = Symbol('id');
const user = { [sym]: 123 };
const hasSym = user[sym];

// Example 2: Private Fields in ES2022 Classes
class Secret {
    #privateData = 42;
    getSecret() { return this.#privateData; }
}
const s = new Secret();

// Example 3: Static Members in Classes
class Counter {
    static count = 0;
    constructor() { Counter.count++; }
    static getCount() { return Counter.count; }
}
const c1 = new Counter();
const c2 = new Counter();

// Example 4: Symbol.iterator for Custom Iteration
const iterableObj = {
    data: [1, 2, 3],
    [Symbol.iterator]() {
        let i = 0, arr = this.data;
        return {
            next() {
                return i < arr.length ? { value: arr[i++], done: false } : { done: true };
            }
        };
    }
};
const arrFromIterable = [...iterableObj];

// Example 5: Private Methods in Classes
class Example {
    #privateMethod() { return 'Private'; }
    callPrivate() { return this.#privateMethod(); }
}
const ex = new Example();

// 5. Factory & Builder Patterns

// Example 1: Simple Factory Function
function createCar(type) {
    if (type === 'sedan') return { type, doors: 4 };
    if (type === 'coupe') return { type, doors: 2 };
    throw new Error('Unknown type');
}
const sedan = createCar('sedan');

// Example 2: Factory with Prototypes
function AnimalFactory(kind) {
    const proto = { speak() { return `I am a ${this.kind}`; } };
    return Object.create(proto, { kind: { value: kind } });
}
const cat2 = AnimalFactory('cat');

// Example 3: Builder Pattern (Fluent API)
class HouseBuilder {
    constructor() { this.house = {}; }
    setRooms(n) { this.house.rooms = n; return this; }
    setColor(c) { this.house.color = c; return this; }
    build() { return this.house; }
}
const house = new HouseBuilder().setRooms(3).setColor('blue').build();

// Example 4: Abstract Factory
function CarFactory() {
    return {
        createEngine() { return { type: 'V8' }; },
        createTire() { return { size: 18 }; }
    };
}
const factory = CarFactory();
const engine = factory.createEngine();

// Example 5: Parameterized Factory with Defaults
function createUser({ name = 'Anonymous', age = 0 } = {}) {
    return { name, age };
}
const user1 = createUser({ name: 'Alice', age: 30 });
const user2 = createUser();