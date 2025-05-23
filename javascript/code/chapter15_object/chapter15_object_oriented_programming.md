Okay boss, let's dive into Chapter 15: Object-Oriented Programming (OOP). This is a big chapter, and understanding OOP principles is crucial for writing well-structured, reusable, and maintainable code. We'll cover everything in detail, so you'll become an OOP expert in JavaScript!

**Chapter 15: Object-Oriented Programming (OOP)**

**1. Classes and Objects**

*   **Object-Oriented Programming (OOP):** OOP is a programming paradigm that organizes software design around objects, which are instances of classes. OOP is a way of structuring code so that properties and behaviours are bundled into individual objects.
*   **Class:** A class is a blueprint or a template for creating objects. It defines the properties (data/attributes) and methods (behavior/functions) that objects of that class will have. Think of it like a cookie cutter.
    * Classes are defined using the `class` keyword in ES6.
*   **Object:** An object is an instance of a class. It's a concrete realization of the blueprint defined by the class. Think of this as the cookie made using the cookie cutter.

    ```javascript
    // Defining a Class
    class Dog {
        constructor(name, breed) {
            this.name = name;
            this.breed = breed;
        }

        bark() {
            console.log("Woof!");
        }

        describe() {
          console.log("Dog Name: " + this.name + ", Breed: " + this.breed)
        }
    }

    // Creating objects (instances)
    const dog1 = new Dog("Charlie", "Labrador");
    const dog2 = new Dog("Bella", "Golden Retriever");

    console.log(dog1.name); // Output: Charlie
    dog1.bark();             // Output: Woof!
    dog1.describe();        // Output: Dog Name: Charlie, Breed: Labrador
    dog2.describe()          // Output: Dog Name: Bella, Breed: Golden Retriever
    ```

*   In the above example, `Dog` is the class, and `dog1` and `dog2` are objects of that class.

**2. Constructors and Prototypes**

*   **Constructor:** A special method within a class that is automatically called when you create a new object using the `new` keyword.
    *   Its primary job is to initialize the object's properties.
    *   It is defined with `constructor()`
    * If the class does not have constructor, then it will have a default constructor.
*  **`this` Keyword:** Within a class, `this` refers to the current object instance (it refers to the object being created or operated on).
*   **Prototypes:** JavaScript uses prototypes for inheritance and object properties.
    * Every object in JS has an associated prototype object.
    *  When you try to access a property of an object, JS first looks at the object's own properties, and then, if it can't find, looks at the object's prototype's properties, and so on. This is called "prototype chaining" or "prototype inheritance."

   ```javascript
    class Animal {
        constructor(name) {
            this.name = name;
        }

        eat() {
            console.log("Animal eating")
        }
    }
    let animal = new Animal("Tiger");
    console.log(animal.name); // Output: Tiger
    animal.eat()              // Output: Animal eating

    console.log(Animal.prototype); // Output: {constructor: f, eat: f}
    console.log(animal.__proto__); // Output: {constructor: f, eat: f} (Prototype)

    console.log(animal.__proto__ === Animal.prototype) // Output: true
   ```
   *   The `Animal` class is created with constructor and a method `eat()`.
   *   The prototype object is the parent of the object created from the class
   *   The prototype object contains the method defined in the class.
   *   Objects created from the same class shares a single prototype.

**3. Inheritance**

*   **Inheritance:** It's a mechanism where a class (child/subclass) can inherit properties and methods from another class (parent/superclass). This promotes code reuse and creates a hierarchy of classes.
*   **`extends` keyword:** Used to inherit from another class.
*   **`super()` keyword:** Used to call the parent class constructor and access parent methods.

    ```javascript
    class Animal {
        constructor(name) {
            this.name = name;
        }
        eat() {
          console.log("Animal eating")
        }
    }

    class Dog extends Animal {
        constructor(name, breed) {
            super(name); // Calling parent constructor
            this.breed = breed;
        }

        bark() {
            console.log("Woof!");
        }

        eat() {
           super.eat(); // Calling parent method
           console.log("Dog eating");
        }
    }

    const myDog = new Dog("Charlie", "Labrador");
    console.log(myDog.name); // Output: Charlie
    myDog.eat(); // Output: Animal eating, Dog eating
    myDog.bark(); // Output: Woof!
    ```

*   In the above example, `Dog` class inherits from the `Animal` class.
*   The `Dog` class has all properties and methods of `Animal` and it can add new properties and methods.
*  The `super()` method in constructor allows you to call parent's constructor. If parent has a constructor, then child class should also have a constructor, and it must call `super()` method.
*   The `super.methodName()` allows you to call the method of parent class.

**4. Polymorphism**

*   **Polymorphism (many forms):**  It allows objects of different classes to respond to the same method call in their own way.
*   Achieved in javascript by method overriding (child class having method with the same name as parent class).

    ```javascript
        class Animal {
            constructor(name) {
                this.name = name;
            }
            makeSound() {
             console.log("Generic animal sound");
            }
        }

        class Dog extends Animal {
          constructor(name, breed) {
            super(name);
            this.breed = breed
          }
            makeSound() { // Method Overriding
                console.log("Woof!");
            }
        }

        class Cat extends Animal {
            makeSound() { // Method Overriding
                console.log("Meow!");
            }
        }

        let animal = new Animal("Generic animal");
        let dog = new Dog("Charlie", "Labrador");
        let cat = new Cat("Bella");

        animal.makeSound(); // Output: Generic animal sound
        dog.makeSound(); // Output: Woof!
        cat.makeSound(); // Output: Meow!
    ```

*   In the above example, the `makeSound()` method behaves differently for each class.
*   This demonstrates polymorphism, where the same method can have different behaviors depending on the object.

**5. Encapsulation**

*   **Encapsulation:** It is the bundling of data (properties) and methods (operations on the data) that operate on that data within a single unit (class). It also involves hiding the internal details of an object and only exposing the necessary interface. This allows for control and protects data from unexpected access or modification.
*   JavaScript does not have direct support for private members (as in other OOP languages), but this can be achieved through closures.

    ```javascript
    function createPerson(name, age) {
      let _name = name // Private
      let _age = age  // Private
        return {
          getName: function() {
           return _name
          },
          getAge: function() {
            return _age
          },
           setAge: function(newAge){
             if(newAge < 0){
                console.log("Age cannot be negative")
              } else{
                _age = newAge;
              }

           }
        }
    }

    let person = createPerson("Raju", 30);
    console.log(person.getName()); // Output: Raju
    console.log(person.getAge()); // Output: 30

    person.setAge(-5) // Age cannot be negative
    person.setAge(35)
    console.log(person.getAge()) // Output: 35
    // console.log(person._age) // Gives error, the variables are private

    ```

*   Here `_name` and `_age` are not directly accessible from outside the `createPerson` function.
*  The variables are private, and only accessible by public getter and setter method.

**6. Example (from your instructions):**

```javascript
class Dog {
    constructor(name, breed) {
        this.name = name;
        this.breed = breed
    }
    bark() {
        console.log("Woof!");
    }
}
const myDog = new Dog("Charlie", "Labrador");
myDog.bark(); // Output: Woof!
```

*   This example demonstrates class definition, object creation, and calling a method using `.` operator.

**Expected Outcome:**

You should now be able to:

*   Understand the core principles of OOP (classes, objects, inheritance, polymorphism, encapsulation).
*   Create classes and objects in JavaScript.
*   Use constructors to initialize objects.
*   Implement inheritance using the `extends` keyword.
*   Understand and implement method overriding.
*   Achieve a basic form of encapsulation in JavaScript.
*  Write maintainable, reusable and robust code using OOP principles.

That's all for Chapter 15, boss! OOP is a very important concept. Take your time, practice with different examples, and don't hesitate to ask questions. We are moving towards modules next! Let's go! 🚀
