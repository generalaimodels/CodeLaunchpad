/******************************************************************************************
 * Filename: oop_examplescode.cpp
 * Author: Kandimalla Hemanth
 * Description:
 *   Comprehensive examples demonstrating the concepts of Classes and Objects in C++.
 *   Each example builds upon the previous one, starting from basic to advanced,
 *   covering the entire concept of Object-Oriented Programming (OOP) with detailed explanations.
 *   Emojis are used to make the code engaging and flowcharts are included in comments where appropriate.
 ******************************************************************************************/

#include <iostream> // For input/output operations
#include <string>  // For string class
#include <vector>  // For vector class

using namespace std;

/*
 * ========================================
 * Example 1: Basic Class and Object Creation 🐶
 * ========================================
 *
 * Concept:
 * - Defining a simple class with public attributes and methods.
 * - Creating objects (instances) of the class.
 */

// Class definition
class Dog {
public:
    // Attributes (Properties)
    string name;
    string breed;
    int age;

    // Method (Behavior)
    void bark() {
        cout << name << " says: Woof! 🐶" << endl;
    }
};

void example1() {
    // Creating an object of the Dog class
    Dog dog1;
    dog1.name = "Buddy";
    dog1.breed = "Golden Retriever";
    dog1.age = 3;

    // Using the object's method
    dog1.bark();

    // Accessing the object's attributes
    cout << dog1.name << " is a " << dog1.age << " year old " << dog1.breed << "." << endl << endl;
}

/*
 * ========================================
 * Example 2: Private Members and Encapsulation 🔒
 * ========================================
 *
 * Concept:
 * - Using private access specifier to encapsulate data.
 * - Providing public getter and setter methods.
 */

// Class definition with encapsulation
class BankAccount {
private:
    string accountNumber;
    double balance;

public:
    // Constructor
    BankAccount(string accNum, double initialBalance) {
        accountNumber = accNum;
        balance = initialBalance;
    }

    // Public method to deposit money
    void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            cout << "Deposited $" << amount << ". New balance: $" << balance << endl;
        } else {
            cout << "Invalid deposit amount." << endl;
        }
    }

    // Public method to withdraw money
    void withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            cout << "Withdrew $" << amount << ". New balance: $" << balance << endl;
        } else {
            cout << "Insufficient funds or invalid amount." << endl;
        }
    }

    // Getter method for balance (read-only access)
    double getBalance() const {
        return balance;
    }
};

void example2() {
    // Creating a BankAccount object
    BankAccount account1("123456789", 1000.0);

    // Trying to access private members (will cause compilation error if uncommented)
    // account1.balance = 500.0; // ❌ Error: 'balance' is private

    // Using public methods to interact with the object
    account1.deposit(200.0);
    account1.withdraw(150.0);
    cout << "Current balance: $" << account1.getBalance() << endl << endl;
}

/*
 * ========================================
 * Example 3: Constructors and Destructors 🏗️🗑️
 * ========================================
 *
 * Concept:
 * - Using constructors for initializing objects.
 * - Using destructors for cleanup (demonstrated with a simple message).
 */

class Person {
public:
    string name;

    // Constructor
    Person(string personName) {
        name = personName;
        cout << name << " has been created. 👶" << endl;
    }

    // Destructor
    ~Person() {
        cout << name << " has been destroyed. ⚰️" << endl;
    }

    void introduce() {
        cout << "Hello, my name is " << name << "." << endl;
    }
};

void example3() {
    // Creating an object of Person class
    Person person1("Alice");
    person1.introduce();

    // Scope ends here, person1 will be destroyed
}

/*
 * ========================================
 * Example 4: The 'this' Pointer 🧭
 * ========================================
 *
 * Concept:
 * - Understanding and using the 'this' pointer.
 * - Differentiating between class attributes and method parameters with the same name.
 */

class Rectangle {
private:
    double width;
    double height;

public:
    // Constructor with parameters named same as attributes
    Rectangle(double width, double height) {
        // Using 'this' pointer to refer to class attributes
        this->width = width;
        this->height = height;
    }

    double area()  {
        return width * height;
    }
};

void example4() {
    // Creating a Rectangle object
    Rectangle rect(5.0, 3.0);
    cout << "Area of the rectangle: " << rect.area() << endl << endl;
}

/*
 * ========================================
 * Example 5: Static Members 🧲
 * ========================================
 *
 * Concept:
 * - Using static variables and methods.
 * - Demonstrating shared data among all objects of a class.
 */

class Student {
public:
    string name;
    int id;
    static int totalStudents; // Static variable declaration

    // Constructor
    Student(string studentName) {
        name = studentName;
        totalStudents++;
        id = totalStudents;
    }

    static void showTotalStudents() {
        cout << "Total students: " << totalStudents << endl;
    }
};

// Static variable definition
int Student::totalStudents = 0;

void example5() {
    Student s1("John");
    Student s2("Emma");
    Student s3("Liam");

    cout << s1.name << "'s ID: " << s1.id << endl;
    cout << s2.name << "'s ID: " << s2.id << endl;
    cout << s3.name << "'s ID: " << s3.id << endl;

    // Calling static method
    Student::showTotalStudents();
    cout << endl;
}

/*
 * ========================================
 * Example 6: Inheritance and Polymorphism 👪🔄
 * ========================================
 *
 * Concept:
 * - Creating a base class and derived classes.
 * - Using virtual functions and demonstrating polymorphism.
 */

class Animal {
public:
    virtual void speak() {
        cout << "The animal makes a sound. 🐾" << endl;
    }
};

class Cat : public Animal {
public:
    void speak() override {
        cout << "The cat says: Meow! 🐱" << endl;
    }
};

class Doggo : public Animal {
public:
    void speak() override {
        cout << "The dog says: Woof! 🐶" << endl;
    }
};

void example6() {
    Animal* animal1 = new Animal();
    Animal* animal2 = new Cat();
    Animal* animal3 = new Doggo();

    // Polymorphic behavior
    animal1->speak();
    animal2->speak();
    animal3->speak();

    // Cleanup
    delete animal1;
    delete animal2;
    delete animal3;
    cout << endl;
}

/*
 * ========================================
 * Example 7: Abstract Classes and Interfaces 🎭
 * ========================================
 *
 * Concept:
 * - Using pure virtual functions.
 * - Creating abstract classes that cannot be instantiated.
 */

class Shape {
public:
    virtual double area() = 0; // Pure virtual function
};

class Circle : public Shape {
private:
    double radius;

public:
    Circle(double r) {
        radius = r;
    }

    double area() override {
        return 3.14159 * radius * radius;
    }
};

void example7() {
    // Shape shape; // ❌ Error: Cannot instantiate abstract class

    Shape* shape = 
    cout << "Area of the circle: " << shape->area() << endl;

    delete shape;
    cout << endl;
}

/*
 * ========================================
 * Example 8: Exception Handling with Classes 🚫⚠️
 * ========================================
 *
 * Concept:
 * - Throwing and catching exceptions in OOP context.
 * - Creating custom exception classes.
 */

class InsufficientFundsException : public exception {
public:
    const char* what() const noexcept override {
        return "Insufficient funds in the account!";
    }
};

class Account {
private:
    double balance;

public:
    Account(double initialBalance) {
        balance = initialBalance;
    }

    void withdraw(double amount) {
        if (amount > balance) {
            throw InsufficientFundsException();
        } else {
            balance -= amount;
            cout << "Withdrawn $" << amount << ". New balance: $" << balance << endl;
        }
    }
};

void example8() {
    Account myAccount(500.0);

    try {
        myAccount.withdraw(600.0);
    } catch (const InsufficientFundsException& e) {
        cout << "Exception: " << e.what() << endl;
    }

    cout << endl;
}

/*
 * ========================================
 * Example 9: Copy Constructor and Assignment Operator 📋
 * ========================================
 *
 * Concept:
 * - Implementing custom copy constructor.
 * - Overloading the assignment operator.
 */

class MyString {
private:
    char* str;

public:
    // Constructor
    MyString(const char* s) {
        str = new char[strlen(s) + 1];
        strcpy(str, s);
    }

    // Copy Constructor
    MyString(const MyString& other) {
        str = new char[strlen(other.str) + 1];
        strcpy(str, other.str);
        cout << "Copy constructor called. 📋" << endl;
    }

    // Assignment Operator Overload
    MyString& operator=(const MyString& other) {
        if (this != &other) {
            delete[] str; // Clean up existing resource
            str = new char[strlen(other.str) + 1];
            strcpy(str, other.str);
            cout << "Assignment operator called. 🔄" << endl;
        }
        return *this;
    }

    // Method to display the string
    void display() const {
        cout << "String: " << str << endl;
    }

    // Destructor
    ~MyString() {
        delete[] str;
        cout << "String destroyed. ⚰️" << endl;
    }
};

void example9() {
    MyString s1("Hello");
    MyString s2 = s1; // Copy constructor called
    s2.display();

    MyString s3("World");
    s3 = s1; // Assignment operator called
    s3.display();

    cout << endl;
}

/*
 * ========================================
 * Example 10: Friend Functions and Classes 🤝
 * ========================================
 *
 * Concept:
 * - Using friend functions and classes to access private members.
 * - Understanding tight coupling and when to use 'friend'.
 */

class Box;

// Friend function declaration
void printBoxDimensions(const Box& b);

class Box {
private:
    double width;
    double height;
    double depth;

public:
    // Constructor
    Box(double w, double h, double d) : width(w), height(h), depth(d) {}

    // Declaring printBoxDimensions as a friend function
    friend void printBoxDimensions(const Box& b);
};

// Friend function definition
void printBoxDimensions(const Box& b) {
    cout << "Box dimensions (WxHxD): " << b.width << " x " << b.height << " x " << b.depth << endl;
}

void example10() {
    Box box1(3.0, 4.0, 5.0);
    printBoxDimensions(box1); // Friend function accessing private members

    cout << endl;
}

/*
 * ========================================
 * Main Function 🌟
 * ========================================
 *
 * - Calls each example in order.
 * - Demonstrates the progression from basic to advanced concepts.
 */

int main() {
    cout << "===== Example 1: Basic Class and Object Creation 🐶 =====" << endl;
    example1();

    cout << "===== Example 2: Private Members and Encapsulation 🔒 =====" << endl;
    example2();

    cout << "===== Example 3: Constructors and Destructors 🏗️🗑️ =====" << endl;
    example3();

    cout << "\n===== Example 4: The 'this' Pointer 🧭 =====" << endl;
    example4();

    cout << "===== Example 5: Static Members 🧲 =====" << endl;
    example5();

    cout << "===== Example 6: Inheritance and Polymorphism 👪🔄 =====" << endl;
    example6();

    cout << "===== Example 7: Abstract Classes and Interfaces 🎭 =====" << endl;
    example7();

    cout << "===== Example 8: Exception Handling with Classes 🚫⚠️ =====" << endl;
    example8();

    cout << "===== Example 9: Copy Constructor and Assignment Operator 📋 =====" << endl;
    example9();

    cout << "===== Example 10: Friend Functions and Classes 🤝 =====" << endl;
    example10();

    return 0;
}

/*
 * ========================================
 * Conclusion 🎉
 * ========================================
 *
 * We've covered 10 comprehensive examples, from basic class definitions to advanced concepts like
 * inheritance, polymorphism, and exception handling in C++. Each example builds upon the previous,
 * illustrating how to structure and use classes and objects effectively.
 *
 * Exception Cases Explained:
 * - Attempting to access private members from outside the class results in compilation errors.
 * - Trying to instantiate an abstract class is not allowed.
 * - Copying objects without proper copy constructors can lead to shallow copies and potential errors.
 *
 * By understanding these examples, you should have a solid grasp of OOP concepts in C++.
 * Happy Coding! 🚀
 */