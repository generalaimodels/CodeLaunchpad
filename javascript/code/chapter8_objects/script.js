/**
 * 🚀 Chapter 8: Objects - Key-Value Pairs in JavaScript 🗝️🔑
 *
 * Objects are the fundamental building blocks in JavaScript, acting like containers 📦
 * for collections of related data and functionality. Think of them as real-world objects 🌎,
 * like a person, a car, or a book, each with its own properties (attributes) and actions (methods).
 *
 * 🌟 Key Concepts:
 * 1.  🔑 Key-Value Pairs: Objects store data in pairs of keys (names or identifiers) and values (the actual data).
 *     Imagine a dictionary 📖 where each word (key) has a definition (value).
 *
 * 2.  🧱 Object Literals:  A straightforward way to create objects using curly braces `{}`.
 *     It's like declaring a variable but for a structured data type.
 *
 * 3.  📍 Property Access:  Retrieve values from objects using keys, similar to looking up information in a database 🗂️.
 *     We have two main ways: Dot Notation (`.`) and Bracket Notation (`[]`).
 *
 * 4.  🛠️ Object Manipulation:  Dynamically add, change, or remove properties from objects, making them flexible and adaptable.
 *
 * 5.  ⚙️ Object Methods: Functions associated with objects, defining the actions or behaviors an object can perform.
 *     Think of methods as verbs that describe what an object *does*.
 *
 * Let's dive deep into each concept with code examples and visual aids! 🚀
 */

/**
 * ------------------------------------------------------------------------------------------
 * 1. Creating Objects using Object Literals 🧱{}
 * ------------------------------------------------------------------------------------------
 *
 * Object literals are defined using curly braces `{}`. Inside the braces, we define key-value pairs.
 *
 * Structure:
 * {
 *   key1: value1,  <-- Property 1 (Key-Value Pair)
 *   key2: value2,  <-- Property 2
 *   ...
 *   keyN: valueN   <-- Property N
 * }
 *
 * - Keys are always strings (or Symbols - advanced topic), but in object literals,
 *   you can often omit quotes if they are valid JavaScript identifiers (no spaces, special chars, etc.).
 * - Values can be any JavaScript data type: primitives (string, number, boolean, null, undefined, symbol, bigint),
 *   objects, arrays, functions... even other objects! 🤯
 *
 *  🌳 Tree Structure of an Object Literal:
 *
 *      Object {}
 *      ├── key1: value1 🍎
 *      ├── key2: value2 🍌
 *      ├── key3: value3 🍇
 *      └── ... 🌿
 */
// 🍎 Example 1: Simple Person Object
let person = {
    name: "Raju",       // Key: 'name', Value: "Raju" (String) 👤
    age: 30,          // Key: 'age', Value: 30 (Number) 🔢
    city: "Mumbai"      // Key: 'city', Value: "Mumbai" (String) 🏙️
};
console.log(person); // Output: { name: 'Raju', age: 30, city: 'Mumbai' }

// 🍌 Example 2: Car Object
let car = {
    brand: "Toyota",    // Key: 'brand', Value: "Toyota" (String) 🚗
    model: "Camry",    // Key: 'model', Value: "Camry" (String)
    year: 2023        // Key: 'year', Value: 2023 (Number) 📅
};
console.log(car); // Output: { brand: 'Toyota', model: 'Camry', year: 2023 }

// 🍇 Example 3: Empty Object
let emptyObject = {};  // An object with no properties. Like an empty box 📦.
console.log(emptyObject); // Output: {}

// 🌿 Example 4: Complex Object with nested structures
let complexObj = {
    name: "John Doe",     // String property 🧑
    address: {             // Value is another Object! Nested Object 🏠
        street: "123 Main St", // String property inside 'address'
        city: "Anytown"      // String property inside 'address'
    },
    hobbies: ["reading", "coding"] // Value is an Array! Array of strings 📚💻
};
console.log(complexObj);
/* Output:
{
  name: 'John Doe',
  address: { street: '123 Main St', city: 'Anytown' },
  hobbies: [ 'reading', 'coding' ]
}
*/

/**
 * ------------------------------------------------------------------------------------------
 * 2. Accessing Object Properties (Dot Notation and Bracket Notation) 📍
 * ------------------------------------------------------------------------------------------
 *
 * To get the value associated with a key in an object, we use property accessors.
 * Think of it as asking the object: "Hey Object, what's the value for the key 'name'?"
 *
 * We have two primary notations:
 *
 * a) Dot Notation (`.`):  Simple and clean, works when the key is a valid identifier.
 *    objectName.propertyName
 *
 * b) Bracket Notation (`[]`): More flexible, essential when keys are not valid identifiers (e.g., contain spaces or special chars)
 *    or when you need to use variables as keys.
 *    objectName["propertyName"]  or objectName[variableKey]
 *
 * 🌳 Tree Structure analogy (Accessing 'name' from 'person' object):
 *
 *      person Object {}
 *      ├── name: "Raju" 🍎  <-- We want to access this!
 *      ├── age: 30 🍌
 *      └── city: "Mumbai" 🍇
 *
 *      Access path using Dot Notation: person.name  ↘️🍎
 *      Access path using Bracket Notation: person["name"] ↘️🍎
 */

// 🍎 Example 1: Dot Notation - Direct and Readable
let student = {
    name: "Priya",      // 🧑‍🎓
    rollNumber: 123    // #️⃣
};

console.log(student.name);        // Output: "Priya"  ✅ - Accessing 'name' using dot notation. Easy!
console.log(student.rollNumber);   // Output: 123    ✅ - Accessing 'rollNumber' using dot notation.

// 🍌 Example 2: Bracket Notation - For special keys or variables
let product = {
    "product name": "Laptop", // Key with space!  💻 - Dot notation won't work directly here.
    "price": 50000           // Key with no space, but demonstrating bracket notation  💰
};

console.log(product["product name"]);  // Output: "Laptop" ✅ - Bracket notation handles keys with spaces.
console.log(product["price"]);         // Output: 50000    ✅ - Bracket notation works even without spaces.

// 🍇 Example 3: Using Variables as Keys (Bracket Notation is essential here)
let key = "price"; // We have a key stored in a variable

console.log(product[key]);           // Output: 50000    ✅ -  Bracket notation uses the *value* of 'key' variable.
// console.log(product.key);         // Output: undefined ❌ - Dot notation would look for a literal key named "key", not the value of the variable.

/**
 * ⚠️ Important Note:
 *  - Use Dot Notation whenever possible for clarity and brevity.
 *  - Bracket Notation is crucial for:
 *      - Keys that are not valid identifiers (contain spaces, special characters, start with a digit).
 *      - Accessing properties using variables.
 *  - If you try to access a property that doesn't exist, JavaScript returns `undefined`. 👻
 */
console.log(student.class);         // Output: undefined 👻 - 'class' property doesn't exist in 'student' object.

/**
 * ------------------------------------------------------------------------------------------
 * 3. Adding, Modifying, and Deleting Object Properties 🛠️
 * ------------------------------------------------------------------------------------------
 *
 * Objects in JavaScript are dynamic. You can change them after creation! 💪
 *
 * a) Adding Properties: Simply assign a value to a new key. If the key doesn't exist, it's created. ✨
 *
 * b) Modifying Properties: Assign a new value to an existing key. Overwrites the old value. 🔄
 *
 * c) Deleting Properties: Use the `delete` keyword followed by the object and the property to remove. 💥
 */

// 🍎 Example 1: Adding Properties
let book = {
    title: "The Alchemist",  // 📚
    author: "Paulo Coelho"  // ✍️
};
console.log("Initial book object:", book); // Output: { title: 'The Alchemist', author: 'Paulo Coelho' }

book.genre = "Fiction";       // Adding 'genre' using dot notation ➕
book["publish year"] = 1988; // Adding 'publish year' using bracket notation (key with space) ➕

console.log("Book object after adding properties:", book);
// Output: { title: 'The Alchemist', author: 'Paulo Coelho', genre: 'Fiction', 'publish year': 1988 }


// 🍌 Example 2: Modifying Properties
let employee = {
    name: "Rohan",    // 🧑‍💼
    salary: 50000   // 💰
};
console.log("Initial employee object:", employee); // Output: { name: 'Rohan', salary: 50000 }

employee.salary = 60000; // Modifying 'salary' property 🔄
console.log("Employee object after modifying salary:", employee);
// Output: { name: 'Rohan', salary: 60000 }


// 🍇 Example 3: Deleting Properties
let studentToDelete = {
    name: "Rahul",     // 🧑‍🎓
    rollNumber: 456  // #️⃣ - Let's remove this property!
};
console.log("Initial studentToDelete object:", studentToDelete); // Output: { name: 'Rahul', rollNumber: 456 }

delete studentToDelete.rollNumber; // Deleting 'rollNumber' property 💥
console.log("studentToDelete object after deleting rollNumber:", studentToDelete);
// Output: { name: 'Rahul' } - 'rollNumber' is gone!

/**
 * ------------------------------------------------------------------------------------------
 * 4. Object Methods ⚙️
 * ------------------------------------------------------------------------------------------
 *
 * Object methods are functions that are properties of objects. They define what an object *can do*.
 * Think of methods as actions or behaviors associated with the object.
 *
 * Inside a method, the `this` keyword refers to the object itself. 🎯
 * This is crucial for accessing other properties of the same object within its methods.
 *
 * a) Traditional Function as Method: Using `function()` syntax.
 *
 * b) ES6 Method Shorthand: Cleaner and more concise syntax for defining methods. 👍
 */

// 🍎 Example 1: Calculator Object with methods (Traditional function syntax)
let calculator = {
    add: function(a, b) { // 'add' method - performs addition
        return a + b;
    },
    subtract: function(a, b) { // 'subtract' method - performs subtraction
        return a - b;
    }
};

console.log("Calculator add method:", calculator.add(5, 3));    // Output: 8  ➕
console.log("Calculator subtract method:", calculator.subtract(10, 4)); // Output: 6  ➖


// 🍌 Example 2: Another Calculator (ES6 Method Shorthand - Cleaner syntax!)
let anotherCalculator = {
    add(a, b) {          // Shorthand 'add' method
        return a + b;
    },
    subtract(a, b) {     // Shorthand 'subtract' method
        return a - b;
    }
};

console.log("Another calculator add method:", anotherCalculator.add(5, 3));    // Output: 8  ➕
console.log("Another calculator subtract method:", anotherCalculator.subtract(10, 4)); // Output: 6  ➖

// 🍇 Example 3: Using `this` keyword inside a method
let personWithFullName = {
    firstName: "John",   // First name property 🧑
    lastName: "Doe",    // Last name property 👨‍🦰
    fullName: function() { // 'fullName' method - combines first and last name
        return this.firstName + " " + this.lastName; // 'this' refers to 'personWithFullName' object! 🎯
    }
};

console.log("Full name using method:", personWithFullName.fullName()); // Output: John Doe  🧑👨‍🦰

/**
 * ✅ Summary of Object Methods:
 * - Methods are functions stored as object properties.
 * - Use `this` inside methods to access the object's other properties.
 * - ES6 method shorthand makes the syntax cleaner.
 */

/**
 * 🎉 Congratulations! 🎉
 * You've now mastered the basics of JavaScript Objects - Key-Value Pairs! 🏆
 *
 * You can now:
 * - Create objects using object literals. 🧱{}
 * - Access object properties using dot and bracket notation. 📍. []
 * - Add, modify, and delete object properties. 🛠️➕ 🔄 💥
 * - Define object methods and use the `this` keyword. ⚙️🎯
 * - Understand objects as collections of key-value pairs. 🗝️🔑
 *
 * Keep practicing and experimenting with objects. They are incredibly powerful and versatile! 💪🚀
 *
 *  Next up:  ➡️ Scope and Closures! 🔒  Let's continue our JavaScript journey! 🛤️
 */


/**
 * 🚀 Advanced Chapter 8: Delving Deeper into JavaScript Objects 🗝️🔑 (advanced_objects.js)
 *
 *  Building upon the fundamentals of objects, let's explore more advanced and powerful concepts! 💪
 *  This file focuses on object creation patterns, inheritance, and modern object manipulation techniques.
 *
 *  🌟 Advanced Concepts We'll Cover:
 *  1. 🏭 Constructor Functions:  Creating objects using constructor functions as blueprints.
 *  2. 🧬 Prototypes and Prototypal Inheritance:  Understanding object relationships and inheritance in JavaScript.
 *  3. 🏢 ES6 Classes:  Syntactic sugar for constructor functions and prototypes, offering a class-based syntax.
 *  4. 🔪 Object Destructuring:  Extracting properties from objects with concise syntax.
 *  5. 🧽 Object Spread Syntax:  Creating shallow copies and merging objects easily.
 *
 * Let's supercharge your object mastery! ⚡
 */

/**
 * ------------------------------------------------------------------------------------------
 * 1. 🏭 Constructor Functions: Blueprints for Objects 🧱
 * ------------------------------------------------------------------------------------------
 *
 * Constructor functions are like templates or blueprints for creating objects. They define
 * the structure and initial properties of objects.  Think of them as cookie cutters 🍪 for objects.
 *
 * Key features of Constructor Functions:
 * - Defined using the `function` keyword, like regular functions but intended to be used with `new`.
 * - Naming convention: Typically capitalized (e.g., `Person`, `Car`) to indicate they are constructors.
 * - Use the `this` keyword inside to refer to the *newly created object* instance.
 * - No explicit `return` statement is needed. Implicitly returns the new object instance.
 * - Invoked using the `new` keyword to create object instances.
 *
 * 🌳 Analogy: Constructor Function as a Blueprint
 *
 *      Constructor Function (Blueprint) 📄
 *          |
 *          |----->  new  -----> Object Instance 1 👤
 *          |                   (Properties defined in constructor)
 *          |
 *          |----->  new  -----> Object Instance 2 🚗
 *          |                   (Same properties structure, different values)
 *          |
 *          |----->  ...
 *
 */
// 🍎 Example 1: Person Constructor Function
// /**
//  * @constructor 
//  * @param {string} name - The name of the person.
//  * @param {number} age - The age of the person.
//  */
function Person(name, age) {
    console.log("Person constructor called!", this); // 'this' refers to the new object being created
    this.name = name;   // Assign 'name' argument to the 'name' property of the new object 👤
    this.age = age;     // Assign 'age' argument to the 'age' property of the new object 🔢
    // No explicit return!
}

// Creating objects using the 'new' keyword and the Person constructor
const person1 = new Person("Alice", 25); // 👤 Creates a Person object: { name: "Alice", age: 25 }
const person2 = new Person("Bob", 30);   // 👨 Creates another Person object: { name: "Bob", age: 30 }

console.log("person1:", person1); // Output: Person { name: 'Alice', age: 25 }
console.log("person2:", person2); // Output: Person { name: 'Bob', age: 30 }

// 🍌 Example 2: Car Constructor Function with a method
/**
 * @constructor Car
 * @param {string} brand - Brand of the car.
 * @param {string} model - Model of the car.
 */
function Car(brand, model) {
    this.brand = brand; // 🚗
    this.model = model; // 🚘
    this.startEngine = function() { // Method to start the car engine ⚙️
        console.log(`Starting the ${this.brand} ${this.model} engine! 💨`); // 'this' inside method refers to car instance
    };
}

const car1 = new Car("Toyota", "Camry");  // 🚗 Creates a Car object
const car2 = new Car("Honda", "Civic");    // 🚘 Creates another Car object

console.log("car1:", car1); // Output: Car { brand: 'Toyota', model: 'Camry', startEngine: [Function (anonymous)] }
car1.startEngine(); // Output: Starting the Toyota Camry engine! 💨
car2.startEngine(); // Output: Starting the Honda Civic engine! 💨


/**
 * ------------------------------------------------------------------------------------------
 * 2. 🧬 Prototypes and Prototypal Inheritance: Object Lineage 🌳
 * ------------------------------------------------------------------------------------------
 *
 * Every object in JavaScript has a prototype.  A prototype is itself an object, and it's like a parent
 * object from which other objects inherit properties and methods. This is the basis of prototypal inheritance in JavaScript.
 *
 * Key Concepts:
 * - Prototype Object: An object associated with every function and object by default in JavaScript.
 * - `__proto__` (dunder proto):  Each object instance has a `__proto__` property that points to its constructor's `prototype` object, or to `null` if it's the end of the chain. (Avoid using `__proto__` in production code, for learning purposes here.)
 * - `.prototype` property:  Functions (including constructor functions) have a `.prototype` property that points to an object. This `prototype` object is used as the prototype for objects created with `new` and this function as a constructor.
 * - Prototypal Inheritance: Objects inherit properties and methods from their prototype chain. If a property or method is not found on the object itself, JavaScript looks up the prototype chain.
 *
 * 🌳 Prototype Chain Visualization (for objects created by `Person` constructor):
 *
 *      person1 Object --- __proto__ --> Person.prototype Object
 *                                          |
 *                                          |--- __proto__ --> Object.prototype Object (built-in prototype)
 *                                                               |
 *                                                               |--- __proto__ --> null (end of chain)
 *
 */
console.log("\n--- Prototypes and Prototypal Inheritance ---");

console.log("person1.__proto__ === Person.prototype:", person1.__proto__ === Person.prototype); // Output: true -  `person1`'s prototype is `Person.prototype`
console.log("Person.prototype.constructor === Person:", Person.prototype.constructor === Person); // Output: true - `Person.prototype`'s constructor points back to `Person`

// Adding a method to the Person.prototype - All Person objects will inherit this method! 🚀
Person.prototype.greet = function() {
    console.log(`Hello, my name is ${this.name}`); // 'this' inside prototype method refers to object instance
};

person1.greet(); // Output: Hello, my name is Alice  👋 - person1 inherits 'greet' from Person.prototype!
person2.greet(); // Output: Hello, my name is Bob    👋 - person2 also inherits 'greet'!

console.log("person1.hasOwnProperty('name'):", person1.hasOwnProperty('name')); // Output: true - 'name' is directly on person1
console.log("person1.hasOwnProperty('greet'):", person1.hasOwnProperty('greet')); // Output: false - 'greet' is NOT directly on person1, it's inherited from prototype

// 🌳 Prototype Chain Exploration
console.log("person1.__proto__:", person1.__proto__); // Output: { greet: [Function (anonymous)], constructor: ... } - Person.prototype object
console.log("person1.__proto__.__proto__:", person1.__proto__.__proto__); // Output: {} - Object.prototype object
console.log("person1.__proto__.__proto__.__proto__:", person1.__proto__.__proto__.__proto__); // Output: null - End of the prototype chain

/**
 * ------------------------------------------------------------------------------------------
 * 3. 🏢 ES6 Classes: Syntactic Sugar for Constructor Functions & Prototypes ✨
 * ------------------------------------------------------------------------------------------
 *
 * ES6 introduced `class` syntax in JavaScript. Classes are primarily syntactic sugar over
 * constructor functions and prototypes. They provide a more class-based syntax that is familiar
 * to developers from other languages, but under the hood, JavaScript still uses prototypes for inheritance.
 *
 * Key Class Syntax Elements:
 * - `class` keyword:  Declares a class.
 * - `constructor()` method:  Special method within a class that acts like a constructor function. It's called when you create a new object instance using `new ClassName()`.
 * - Methods inside class body: Define methods on the prototype of the class.
 * - `extends` keyword: Used for class inheritance.
 * - `super()`: Used within a child class constructor to call the parent class constructor.
 *
 * 🌳 Class Structure Analogy
 *
 *      class MyClass {
 *          constructor(...) { ... }  // Constructor (like constructor function)
 *          method1(...) { ... }      // Method (added to prototype)
 *          method2(...) { ... }      // Method (added to prototype)
 *          ...
 *      }
 */
console.log("\n--- ES6 Classes ---");

// 🍎 Example 1: Person Class (Equivalent to Person constructor function)
class PersonClass {
    /**
     * @constructor
     * @param {string} name - The name of the person.
     * @param {number} age - The age of the person.
     */
    constructor(name, age) {
        console.log("PersonClass constructor called!", this);
        this.name = name;
        this.age = age;
    }

    greet() { // Method defined in the class - added to PersonClass.prototype
        console.log(`Hello from class, I am ${this.name}`);
    }
}

const classPerson1 = new PersonClass("Charlie", 28); // Creates PersonClass object
const classPerson2 = new PersonClass("Diana", 32);   // Creates another PersonClass object

console.log("classPerson1:", classPerson1); // Output: PersonClass { name: 'Charlie', age: 28 }
classPerson1.greet(); // Output: Hello from class, I am Charlie
classPerson2.greet(); // Output: Hello from class, I am Diana

console.log("classPerson1.__proto__ === PersonClass.prototype:", classPerson1.__proto__ === PersonClass.prototype); // Output: true - Prototype relationship same as with constructor functions!

// 🍌 Example 2: Inheritance using 'extends' and 'super' (Brief introduction to inheritance)
class StudentClass extends PersonClass { // StudentClass inherits from PersonClass 🎓
    /**
     * @constructor
     * @param {string} name - Name of the student.
     * @param {number} age - Age of the student.
     * @param {string} studentId - Student ID.
     */
    constructor(name, age, studentId) {
        super(name, age); // Call the constructor of the parent class (PersonClass) using super()
        this.studentId = studentId; // Student-specific property
    }

    study() { // Student-specific method
        console.log(`${this.name} with ID ${this.studentId} is studying! 📚`);
    }
}

const student1 = new StudentClass("Eve", 20, "S1001"); // Creates a StudentClass object
console.log("student1:", student1); // Output: StudentClass { name: 'Eve', age: 20, studentId: 'S1001' }
student1.greet(); // Output: Hello from class, I am Eve (Inherited from PersonClass!)
student1.study(); // Output: Eve with ID S1001 is studying!

/**
 * 💡 Key Takeaways about ES6 Classes:
 * - Classes are syntactic sugar, making prototypal inheritance more palatable.
 * - `constructor` is like the constructor function.
 * - Methods are added to the prototype.
 * - `extends` and `super` facilitate inheritance. (More on inheritance in advanced topics!)
 */


/**
 * ------------------------------------------------------------------------------------------
 * 4. 🔪 Object Destructuring:  Unpacking Object Properties 🎁
 * ------------------------------------------------------------------------------------------
 *
 * Object destructuring is a concise and convenient way to extract values from objects and bind them to variables.
 * It simplifies accessing object properties, especially when you need to use multiple properties.
 *
 * Basic Syntax:
 * const { property1, property2, ... } = object;
 *
 * - You specify the property names you want to extract within curly braces `{}` on the left side of the assignment.
 * - JavaScript looks up these properties in the `object` on the right and assigns their values to variables with the same names.
 *
 * Renaming Properties during Destructuring:
 * const { property1: newVariableName1, property2: newVariableName2, ... } = object;
 * - You can rename the extracted properties while destructuring.
 *
 * Default Values:
 * const { property1 = defaultValue1, property2 = defaultValue2, ... } = object;
 * - Provide default values if a property might be missing in the object.
 */
console.log("\n--- Object Destructuring ---");

const address = {
    street: "456 Oak Ave",
    city: "Springfield",
    zipCode: "12345",
    country: "USA"
};

// 🍎 Example 1: Basic Destructuring
const { street, city } = address; // Extract 'street' and 'city' properties
console.log("Street:", street); // Output: Street: 456 Oak Ave
console.log("City:", city);   // Output: City: Springfield

// 🍌 Example 2: Destructuring with Renaming
const { street: addressStreet, zipCode: postalCode } = address; // Rename 'street' to 'addressStreet', 'zipCode' to 'postalCode'
console.log("Address Street:", addressStreet); // Output: Address Street: 456 Oak Ave
console.log("Postal Code:", postalCode);   // Output: Postal Code: 12345

// 🍇 Example 3: Destructuring with Default Values
const { street: road, state = "Unknown" } = address; // 'state' property is not in 'address', so default value is used
console.log("Road:", road);   // Output: Road: 456 Oak Ave
console.log("State:", state);  // Output: State: Unknown (default value)

// 🥝 Example 4: Destructuring in Function Parameters - very common pattern!
function printAddress({ city, zipCode }) { // Destructure 'city' and 'zipCode' directly from the object argument
    console.log(`Function Address: ${city}, ${zipCode}`);
}

printAddress(address); // Output: Function Address: Springfield, 12345

/**
 * ------------------------------------------------------------------------------------------
 * 5. 🧽 Object Spread Syntax: Cloning and Merging Objects ...
 * ------------------------------------------------------------------------------------------
 *
 * The spread syntax (`...`) has powerful applications for objects:
 * - Cloning (Shallow Copy): Create a new object with the same properties as an existing one. (Shallow copy means nested objects are still references).
 * - Merging Objects: Combine properties from multiple objects into a new object.
 *
 * Basic Syntax (for cloning):
 * const newObject = { ...originalObject };
 *
 * Basic Syntax (for merging):
 * const mergedObject = { ...object1, ...object2, ... };
 * - If there are duplicate keys, the properties from the later objects in the spread will overwrite the earlier ones.
 */
console.log("\n--- Object Spread Syntax ---");

const originalCar = {
    make: "Tesla",
    model: "Model S",
    year: 2023,
    features: ["Autopilot", "Ludicrous Mode"] // Nested array - shallow copy impact!
};

// 🍎 Example 1: Cloning (Shallow Copy)
const clonedCar = { ...originalCar }; // Creates a new object with all properties of originalCar

console.log("originalCar === clonedCar:", originalCar === clonedCar); // Output: false - Different objects in memory
console.log("originalCar.features === clonedCar.features:", originalCar.features === clonedCar.features); // Output: true - But nested objects/arrays are still references! (Shallow copy)

clonedCar.model = "Model 3"; // Modifying clonedCar.model does not affect originalCar.model
clonedCar.features.push("New Feature"); // Modifying nested array in clonedCar *does* affect originalCar.features (shallow copy effect!)

console.log("originalCar after clonedCar modification:", originalCar);
// Output: features: [ 'Autopilot', 'Ludicrous Mode', 'New Feature' ]  <- Changed!
console.log("clonedCar:", clonedCar);
// Output: features: [ 'Autopilot', 'Ludicrous Mode', 'New Feature' ], model: 'Model 3'

// 🍌 Example 2: Merging Objects
const carDetails = {
    color: "Red",
    price: 75000
};
const mergedCar = { ...originalCar, ...carDetails }; // Merges originalCar and carDetails into a new object

console.log("mergedCar:", mergedCar);
/* Output:
{
  make: 'Tesla',
  model: 'Model S',
  year: 2023,
  features: [ 'Autopilot', 'Ludicrous Mode', 'New Feature' ],  // From originalCar
  color: 'Red',      // From carDetails
  price: 75000     // From carDetails
}
*/

// 🍇 Example 3: Overwriting Properties during Merge
const updateCar = {
    year: 2024, // Will overwrite 'year' from originalCar
    color: "Blue" // Will overwrite 'color' if it was in originalCar or carDetails (in this case, adds if not present, or overwrites carDetails's color)
};
const updatedMergedCar = { ...originalCar, ...carDetails, ...updateCar }; // 'updateCar' properties take precedence

console.log("updatedMergedCar:", updatedMergedCar);
/* Output:
{
  make: 'Tesla',
  model: 'Model S',
  year: 2024,       // Overwritten by updateCar
  features: [ 'Autopilot', 'Ludicrous Mode', 'New Feature' ],
  color: 'Blue',      // Overwritten/added by updateCar, effectively overwrites 'carDetails' if 'carDetails' had 'color'
  price: 75000
}
*/

/**
 * 🎉 Advanced Object Mastery Achieved! 🏆
 *
 * You've now leveled up your JavaScript object skills with these advanced concepts:
 * - Constructor Functions 🏭
 * - Prototypes and Prototypal Inheritance 🧬
 * - ES6 Classes 🏢
 * - Object Destructuring 🔪
 * - Object Spread Syntax 🧽
 *
 *  Keep exploring and practicing to solidify your understanding.  JavaScript objects are incredibly versatile
 *  and these advanced techniques will empower you to write more efficient, organized, and modern code. 🚀
 *
 *  Where to go next? 🤔 Consider diving deeper into:
 *  - Object-Oriented Programming (OOP) principles in JavaScript.
 *  - More advanced inheritance patterns.
 *  - Object methods like `Object.create`, getters/setters, and more.
 *  - Working with APIs and complex data structures using objects.
 *
 *  Happy coding! 🚀✨
 */