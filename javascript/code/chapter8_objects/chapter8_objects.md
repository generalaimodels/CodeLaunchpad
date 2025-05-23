Okay boss, let's get into Chapter 8: Objects - Key-Value Pairs. Objects are fundamental in JavaScript, as they allow you to represent complex entities with properties and behaviors. We'll cover everything in detail, ensuring you become an object master!

**Chapter 8: Objects - Key-Value Pairs**

**1. Creating Objects using Object Literals**

An object is a collection of key-value pairs. The keys are strings (property names), and the values can be any data type (numbers, strings, booleans, arrays, other objects, functions). You can create an object using curly braces `{}` (object literal notation).

```javascript
let person = {
    name: "Raju",
    age: 30,
    city: "Mumbai"
};

let car = {
  brand: "Toyota",
  model: "Camry",
  year: 2023
}

let emptyObject = {}; // Empty Object

let complexObj = {
  name: "John Doe",
  address: {
    street: "123 Main St",
    city: "Anytown"
  },
  hobbies: ["reading", "coding"]
}
```

*   `name`, `age`, `city`, `brand`, `model`, `year` etc. are the keys/properties of the object.
*   `"Raju"`, `30`, `"Mumbai"` etc. are the corresponding values for the key.
*  The value of object can be another object, array or even a function (methods).

**2. Accessing Object Properties (Dot Notation and Bracket Notation)**

You can access object properties using two ways:

*   **Dot Notation:** Use a dot (`.`) followed by the property name. It's easier and simpler when your property name is a valid identifier (no spaces, special chars, starting with a digit etc.).

    ```javascript
    let student = {
        name: "Priya",
        rollNumber: 123
    };

    console.log(student.name);       // Output: "Priya"
    console.log(student.rollNumber);  // Output: 123
    ```

*   **Bracket Notation:** Use square brackets `[]` with the property name as a string. This is useful when your property name has spaces or is a special character or a variable.

    ```javascript
    let product = {
        "product name": "Laptop",
        "price": 50000
    };

    console.log(product["product name"]);   // Output: "Laptop"
    console.log(product["price"]);          // Output: 50000

    let key = "price"
    console.log(product[key]);          //Output: 50000
    ```

**Important Points:**

*   Use dot notation whenever possible because it is cleaner.
*   If a property doesn't exist, it will return `undefined`.

**3. Adding, Modifying, and Deleting Object Properties**

*   **Adding Properties:** You can add new properties to an object simply by assigning a value to a new key.

    ```javascript
    let book = {
      title: "The Alchemist",
      author: "Paulo Coelho"
    };

    book.genre = "Fiction";      // Adding new property using dot notation
    book["publish year"] = 1988  // Adding new property using bracket notation
    console.log(book);
    // Output: { title: 'The Alchemist', author: 'Paulo Coelho', genre: 'Fiction', 'publish year': 1988 }
    ```

*   **Modifying Properties:** You can change the value of an existing property by simply assigning a new value to it.

    ```javascript
    let employee = {
        name: "Rohan",
        salary: 50000
    };

    employee.salary = 60000; // Modifying existing property
    console.log(employee); // Output: { name: 'Rohan', salary: 60000 }
    ```

*   **Deleting Properties:** You can remove a property from an object using the `delete` keyword.

    ```javascript
    let student = {
        name: "Rahul",
        rollNumber: 456
    };

    delete student.rollNumber;
    console.log(student); // Output: { name: 'Rahul' }
    ```

**4. Object Methods**

An object method is simply a function that is a property of an object. Methods can be used to perform operations specific to that object.

```javascript
let calculator = {
    add: function(a, b) {
        return a + b;
    },
    subtract: function(a, b) {
        return a - b;
    }
}

console.log(calculator.add(5, 3)); // Output: 8
console.log(calculator.subtract(10, 4)); //Output: 6

// ES6 shorthand for method
let anotherCalculator = {
    add(a, b) {
        return a + b;
    },
    subtract(a, b) {
        return a - b;
    }
}

console.log(anotherCalculator.add(5, 3)); // Output: 8
console.log(anotherCalculator.subtract(10, 4)); //Output: 6
```

*   Here `add` and `subtract` are the methods of `calculator` and `anotherCalculator` objects.
*   Inside object methods, we can use `this` keyword to refer to the current object.

```javascript
let person = {
    firstName: "John",
    lastName: "Doe",
    fullName: function() {
        return this.firstName + " " + this.lastName
    }
}

console.log(person.fullName()) // Output: John Doe
```

**Example (from your instructions):**

```javascript
let person = { name: "Rohan", age: 25 };
console.log(person.name); // Output: "Rohan"
```

**Expected Outcome:**

You should now be able to:

*   Create objects using object literals.
*   Access object properties using dot and bracket notation.
*   Add, modify, and delete object properties.
*   Define object methods and use `this` keyword.
*  Work with object as a collection of key-value pair.

That's all for Chapter 8, boss! You're now equipped to handle objects effectively. Practice creating and manipulating various objects to become proficient. Any questions? Let's move on to Scope and Closures next! 🚀
