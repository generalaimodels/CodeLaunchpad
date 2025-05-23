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
Mammal.prototype.breathe = function() {
    return 'Breathing...';
};
const cat = new Animal('Whiskers');
Object.setPrototypeOf(cat, Mammal.prototype);

// Example 3: Object.create for Prototypal Inheritance
const proto = { greet() { return 'Hello'; } };
const obj = Object.create(proto);

// Example 4: hasOwnProperty vs. Prototype Property
function Car() {}
Car.prototype.wheels = 4;
const myCar = new Car();
myCar.color = 'red';
const own = myCar.hasOwnProperty('color'); // true
const protoProp = myCar.hasOwnProperty('wheels'); // false

// Example 5: Modifying Prototypes at Runtime
function Bird() {}
Bird.prototype.fly = function() { return 'Flying'; };
const sparrow = new Bird();
Bird.prototype.sing = function() { return 'Singing'; };

// 2. Classical vs. Prototypal Inheritance

// Example 1: Classical Inheritance with ES6 Classes
class Person {
    constructor(name) { this.name = name; }
    greet() { return `Hi, I'm ${this.name}`; }
}
class Employee extends Person {
    constructor(name, role) {
        super(name);
        this.role = role;
    }
    work() { return `${this.name} works as ${this.role}`; }
}
const emp = new Employee('Alice', 'Engineer');

// Example 2: Prototypal Inheritance with Object.create
const vehicle = { move() { return 'Moving'; } };
const bike = Object.create(vehicle);
bike.ringBell = function() { return 'Ring ring!'; };

// Example 3: Constructor Function Inheritance
function Shape(color) { this.color = color; }
Shape.prototype.describe = function() { return `Color: ${this.color}`; };
function Circle(color, radius) {
    Shape.call(this, color);
    this.radius = radius;
}
Circle.prototype = Object.create(Shape.prototype);
Circle.prototype.constructor = Circle;
Circle.prototype.area = function() { return Math.PI * this.radius ** 2; };
const circ = new Circle('blue', 2);

// Example 4: ES6 Class Inheritance with Method Overriding
class Animal2 {
    speak() { return 'Generic sound'; }
}
class Dog2 extends Animal2 {
    speak() { return 'Woof!'; }
}
const d2 = new Dog2();

// Example 5: Delegation with Prototypal Inheritance
const canEat = { eat() { return 'Eating'; } };
const canWalk = { walk() { return 'Walking'; } };
const person = Object.assign(Object.create(canEat), canWalk);

// 3. Mixins, Composition & Trait Patterns

// Example 1: Mixin via Object.assign
const canSwim = { swim() { return 'Swimming'; } };
const canFly = { fly() { return 'Flying'; } };
const duck = Object.assign({}, canSwim, canFly);

// Example 2: Functional Mixins
function Jumpable(obj) {
    obj.jump = function() { return 'Jumping'; };
    return obj;
}
const rabbit = Jumpable({ name: 'Bunny' });

// Example 3: Class Mixins with ES6
const Flyer = Base => class extends Base {
    fly() { return 'Flying high'; }
};
class Creature {}
class Bat extends Flyer(Creature) {}
const bat = new Bat();

// Example 4: Trait Pattern with Conflict Resolution
const traitA = { greet() { return 'Hello from A'; } };
const traitB = { greet() { return 'Hello from B'; } };
const composed = Object.assign({}, traitA, traitB); // traitB overwrites traitA

// Example 5: Composition over Inheritance
function compose(...behaviors) {
    return Object.assign({}, ...behaviors);
}
const swimmer = { swim() { return 'Swimming'; } };
const runner = { run() { return 'Running'; } };
const triathlete = compose(swimmer, runner);

// 4. Symbols, Private Fields & Static Members

// Example 1: Using Symbols for Unique Properties
const sym = Symbol('id');
const user = { [sym]: 123, name: 'Bob' };

// Example 2: Private Fields in Classes (ES2022+)
class Secret {
    #hidden = 42;
    getHidden() { return this.#hidden; }
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
const range = {
    from: 1,
    to: 3,
    [Symbol.iterator]() {
        let current = this.from, last = this.to;
        return {
            next() {
                if (current <= last) return { value: current++, done: false };
                return { done: true };
            }
        };
    }
};
[...range]; // [1,2,3]

// Example 5: Private Methods in Classes
class Example {
    #privateMethod() { return 'secret'; }
    publicMethod() { return this.#privateMethod(); }
}
const ex = new Example();

// 5. Factory & Builder Patterns

// Example 1: Simple Factory Function
function createUser(name, role) {
    return { name, role, describe() { return `${name} is a ${role}`; } };
}
const u = createUser('Eve', 'Admin');

// Example 2: Factory with Prototypes
const animalProto = { speak() { return 'Animal sound'; } };
function animalFactory(name) {
    const obj = Object.create(animalProto);
    obj.name = name;
    return obj;
}
const a = animalFactory('Leo');

// Example 3: Builder Pattern (Fluent API)
class CarBuilder {
    constructor() { this.car = {}; }
    setWheels(w) { this.car.wheels = w; return this; }
    setColor(c) { this.car.color = c; return this; }
    build() { return this.car; }
}
const car = new CarBuilder().setWheels(4).setColor('red').build();

// Example 4: Abstract Factory Pattern
function createButtonFactory(theme) {
    if (theme === 'dark') {
        return () => ({ color: 'black', text: 'Dark Button' });
    } else {
        return () => ({ color: 'white', text: 'Light Button' });
    }
}
const darkButton = createButtonFactory('dark')();

// Example 5: Factory with Encapsulation
function PersonFactory() {
    let id = 0;
    return function(name) {
        id++;
        return { id, name };
    };
}
const makePerson = PersonFactory();
const p1 = makePerson('John');
const p2 = makePerson('Jane');