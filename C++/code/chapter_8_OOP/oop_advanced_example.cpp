/******************************************************************************************
 * Filename: oop_advanced_examples.cpp
 * Author: Kandimalla Hemanth
 * Description:
 *   Level 2 OOP Examples: Advanced concepts of Object-Oriented Programming in C++.
 *   This file contains 10 examples ranging from basic to very advanced OOP concepts,
 *   covering classes, objects, inheritance, polymorphism, templates, and more.
 *   Detailed explanations are provided to ensure 100% understanding for developers.
 *   Emojis, diagrams (in comments), and flowcharts are included to make the code engaging.
 ******************************************************************************************/

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <exception>
#include <typeinfo>

using namespace std;

/*
 * =======================================================================
 * Example 1: Basic Class with Constructors and Methods 🚗
 * =======================================================================
 *
 * Concept:
 * - Defining a simple class `Car` with constructors, attributes, and methods.
 * - Understanding default and parameterized constructors.
 */

class Car {
private:
    string brand;
    string model;
    int year;

public:
    // Default Constructor
    Car() : brand("Unknown"), model("Unknown"), year(0) {
        cout << "Default Car created 🚗" << endl;
    }

    // Parameterized Constructor
    Car(string brand, string model, int year) : brand(brand), model(model), year(year) {
        cout << "Car " << brand << " " << model << " created 🚗" << endl;
    }

    // Method to display car details
    void displayInfo() const {
        cout << "Car Info: " << brand << " " << model << ", Year: " << year << endl;
    }
};

void example1() {
    Car car1; // Default constructor called
    car1.displayInfo();

    Car car2("Toyota", "Camry", 2020); // Parameterized constructor called
    car2.displayInfo();

    cout << endl;
}

/*
 * =======================================================================
 * Example 2: Encapsulation and Data Hiding 🔐
 * =======================================================================
 *
 * Concept:
 * - Using private members to hide data.
 * - Providing public methods for controlled access.
 * - Understanding exception cases of accessing private data.
 */

class Employee {
private:
    string name;
    double salary;

public:
    Employee(string name, double salary) : name(name) {
        setSalary(salary);
    }

    // Setter method with validation
    void setSalary(double salary) {
        if (salary >= 0) {
            this->salary = salary;
        } else {
            cout << "Invalid salary amount!" << endl;
            this->salary = 0;
        }
    }

    // Getter method
    double getSalary() const {
        return salary;
    }

    void display() const {
        cout << "Employee: " << name << ", Salary: $" << salary << endl;
    }
};

void example2() {
    Employee emp1("Alice", 50000);
    emp1.display();

    emp1.setSalary(-1000); // Exception case: Negative salary
    emp1.display();

    // emp1.salary = 60000; // ❌ Error: 'salary' is private

    cout << endl;
}

/*
 * =======================================================================
 * Example 3: Inheritance and Access Specifiers 📚
 * =======================================================================
 *
 * Concept:
 * - Demonstrating inheritance (public, protected, private).
 * - Understanding how access specifiers affect inheritance.
 */

class Base {
protected:
    int protectedVar;

private:
    int privateVar;

public:
    int publicVar;

    Base() : protectedVar(0), privateVar(0), publicVar(0) {}

    void setVars(int pVar, int priVar, int pubVar) {
        protectedVar = pVar;
        privateVar = priVar;
        publicVar = pubVar;
    }
};

class DerivedPublic : public Base {
public:
    void display() {
        cout << "DerivedPublic Access:" << endl;
        cout << "protectedVar: " << protectedVar << endl;
        // cout << "privateVar: " << privateVar << endl; // ❌ Error: 'privateVar' is private
        cout << "publicVar: " << publicVar << endl;
    }
};

class DerivedProtected : protected Base {
public:
    void display() {
        cout << "DerivedProtected Access:" << endl;
        cout << "protectedVar: " << protectedVar << endl;
        // cout << "privateVar: " << privateVar << endl; // ❌ Error: 'privateVar' is private
        cout << "publicVar: " << publicVar << endl;
    }
};

class DerivedPrivate : private Base {
public:
    void display() {
        cout << "DerivedPrivate Access:" << endl;
        cout << "protectedVar: " << protectedVar << endl;
        // cout << "privateVar: " << privateVar << endl; // ❌ Error: 'privateVar' is private
        cout << "publicVar: " << publicVar << endl;
    }
};

void example3() {
    DerivedPublic dp;
    dp.setVars(1, 2, 3);
    dp.display();
    cout << "Accessing publicVar from main: " << dp.publicVar << endl;
    // cout << dp.protectedVar; // ❌ Error: 'protectedVar' is protected

    cout << endl;
}

/*
 * =======================================================================
 * Example 4: Polymorphism with Virtual Functions 🦋
 * =======================================================================
 *
 * Concept:
 * - Implementing polymorphism using virtual functions.
 * - Understanding how virtual functions allow for dynamic binding.
 */

class Shape {
public:
    virtual void draw() {
        cout << "Drawing a generic shape 🖼️" << endl;
    }
};

class Circle : public Shape {
public:
    void draw() override {
        cout << "Drawing a circle ⭕" << endl;
    }
};

class Square : public Shape {
public:
    void draw() override {
        cout << "Drawing a square ◻️" << endl;
    }
};

void example4() {
    Shape* shape1 = new Shape();
    Shape* shape2 = new Circle();
    Shape* shape3 = new Square();

    shape1->draw(); // Calls Shape::draw()
    shape2->draw(); // Calls Circle::draw()
    shape3->draw(); // Calls Square::draw()

    // Clean up
    delete shape1;
    delete shape2;
    delete shape3;

    cout << endl;
}

/*
 * =======================================================================
 * Example 5: Abstract Classes and Pure Virtual Functions 🎨
 * =======================================================================
 *
 * Concept:
 * - Using pure virtual functions to create abstract classes.
 * - Forcing derived classes to implement certain methods.
 */

class Animal {
public:
    virtual void makeSound() = 0; // Pure virtual function
};

class Dog : public Animal {
public:
    void makeSound() override {
        cout << "Dog says: Woof! 🐶" << endl;
    }
};

class Cat : public Animal {
public:
    void makeSound() override {
        cout << "Cat says: Meow! 🐱" << endl;
    }
};

void example5() {
    // Animal animal; // ❌ Error: Cannot instantiate abstract class
    Animal* dog = new Dog();
    Animal* cat = new Cat();

    dog->makeSound();
    cat->makeSound();

    // Clean up
    delete dog;
    delete cat;

    cout << endl;
}

/*
 * =======================================================================
 * Example 6: Operator Overloading ➕
 * =======================================================================
 *
 * Concept:
 * - Overloading operators to work with user-defined classes.
 * - Understanding the syntax and use-cases for operator overloading.
 */

class Complex {
private:
    double real;
    double imag;

public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    // Overloading '+' operator
    Complex operator+(const Complex& other) {
        return Complex(real + other.real, imag + other.imag);
    }

    // Overloading '<<' operator for output
    friend ostream& operator<<(ostream& os, const Complex& c);
};

ostream& operator<<(ostream& os, const Complex& c) {
    os << "(" << c.real << " + " << c.imag << "i)";
    return os;
}

void example6() {
    Complex c1(2, 3);
    Complex c2(4, 5);
    Complex c3;

    c3 = c1 + c2; // Using overloaded '+' operator

    cout << c1 << " + " << c2 << " = " << c3 << endl;

    cout << endl;
}

/*
 * =======================================================================
 * Example 7: Templates and Generic Programming 🧬
 * =======================================================================
 *
 * Concept:
 * - Using templates to create generic classes and functions.
 * - Understanding how templates enable code reusability.
 */

template <typename T> 
class Calculator {
public:
    T add(T a, T b) { return a + b; }

    T subtract(T a, T b) { return a - b; }

    T multiply(T a, T b) { return a * b; }

    T divide(T a, T b) {
        if (b != 0)
            return a / b;
        else {
            throw runtime_error("Division by zero! 🚫");
        }
    }
};

void example7() {
    Calculator<int> intCalc; 
    Calculator<double> doubleCalc; 

    cout << "Int addition: " << intCalc.add(2, 3) << endl;
    cout << "Double multiplication: " << doubleCalc.multiply(2.5, 4.0) << endl;

    try {
        cout << "Division: " << intCalc.divide(10, 0) << endl; // Exception case
    } catch (const exception& e) {
        cout << "Exception: " << e.what() << endl;
    }

    cout << endl;
}

/*
 * =======================================================================
 * Example 8: Exception Handling in OOP 🚨
 * =======================================================================
 *
 * Concept:
 * - Handling exceptions within class methods.
 * - Propagating exceptions to calling functions.
 */

class DivisionCalculator {
public:
    double divide(double numerator, double denominator) {
        if (denominator == 0)
            throw invalid_argument("Cannot divide by zero! 🚫");
        return numerator / denominator;
    }
};

void example8() {
    DivisionCalculator dc;

    try {
        cout << "Result: " << dc.divide(10, 2) << endl;
        cout << "Result: " << dc.divide(10, 0) << endl; // Exception case
    } catch (const invalid_argument& e) {
        cout << "Caught exception: " << e.what() << endl;
    }

    cout << endl;
}

/*
 * =======================================================================
 * Example 9: Smart Pointers and Memory Management 🧠
 * =======================================================================
 *
 * Concept:
 * - Using smart pointers (`unique_ptr`, `shared_ptr`) to manage dynamic memory.
 * - Understanding ownership and automatic deallocation.
 */

class Sensor {
public:
    Sensor() { cout << "Sensor activated 🎛️" << endl; }
    ~Sensor() { cout << "Sensor deactivated 📴" << endl; }

    void readData() { cout << "Reading sensor data 📈" << endl; }
};

void example9() {
    {
        unique_ptr<Sensor> sensor1 = make_unique<Sensor>();
        sensor1->readData();
    } // sensor1 goes out of scope and is automatically deleted

    shared_ptr<Sensor> sensor2;
    {
        shared_ptr<Sensor> sensor3 = make_shared<Sensor>();
        sensor2 = sensor3; // sensor2 and sensor3 share ownership
        cout << "Sensor use count: " << sensor2.use_count() << endl;
    } // sensor3 goes out of scope, but sensor2 still owns the object

    cout << "Sensor use count after scope: " << sensor2.use_count() << endl;

    // sensor2 goes out of scope, object is deleted
    cout << endl;
}

/*
 * =======================================================================
 * Example 10: Multiple Inheritance and Virtual Base Classes 🧩
 * =======================================================================
 *
 * Concept:
 * - Demonstrating multiple inheritance.
 * - Resolving ambiguity with virtual base classes (Diamond Problem).
 */

class Device {
public:
    Device() { cout << "Device initialized 📱" << endl; }
    void identify() { cout << "This is a device." << endl; }
};

class Camera : virtual public Device {
public:
    Camera() { cout << "Camera initialized 📷" << endl; }
    void takePhoto() { cout << "Photo taken 📸" << endl; }
};

class Phone : virtual public Device {
public:
    Phone() { cout << "Phone initialized ☎️" << endl; }
    void makeCall() { cout << "Making a call 📞" << endl; }
};

class SmartPhone : public Camera, public Phone {
public:
    SmartPhone() { cout << "SmartPhone initialized 🤳" << endl; }
};

void example10() {
    SmartPhone sp;
    sp.identify(); // No ambiguity due to virtual inheritance
    sp.takePhoto();
    sp.makeCall();

    cout << endl;
}

/*
 * =======================================================================
 * Main Function 🚀
 * =======================================================================
 *
 * - Calls each example function in order.
 * - Demonstrates advanced OOP concepts with detailed explanations.
 */

int main() {
    cout << "===== Example 1: Basic Class with Constructors and Methods 🚗 =====" << endl;
    example1();

    cout << "===== Example 2: Encapsulation and Data Hiding 🔐 =====" << endl;
    example2();

    cout << "===== Example 3: Inheritance and Access Specifiers 📚 =====" << endl;
    example3();

    cout << "===== Example 4: Polymorphism with Virtual Functions 🦋 =====" << endl;
    example4();

    cout << "===== Example 5: Abstract Classes and Pure Virtual Functions 🎨 =====" << endl;
    example5();

    cout << "===== Example 6: Operator Overloading ➕ =====" << endl;
    example6();

    cout << "===== Example 7: Templates and Generic Programming 🧬 =====" << endl;
    example7();

    cout << "===== Example 8: Exception Handling in OOP 🚨 =====" << endl;
    example8();

    cout << "===== Example 9: Smart Pointers and Memory Management 🧠 =====" << endl;
    example9();

    cout << "===== Example 10: Multiple Inheritance and Virtual Base Classes 🧩 =====" << endl;
    example10();

    return 0;
}

/*
 * =======================================================================
 * Conclusion 🎯
 * =======================================================================
 *
 * This single C++ file encompasses 10 examples that progress from basic to
 * advanced concepts in Object-Oriented Programming (OOP). Each example is
 * designed to provide a deep understanding of classes and objects, along with
 * detailed explanations and exception cases for thorough comprehension.
 *
 * Exception Cases Explained:
 * - Example 2: Setting a negative salary demonstrates the need for validation.
 * - Example 3: Accessing private and protected members shows compilation errors.
 * - Example 5: Trying to instantiate an abstract class results in an error.
 * - Example 7: Division by zero throws a runtime exception.
 * - Example 8: Handling exceptions thrown from class methods.
 *
 * By studying these examples, developers can strengthen their grasp of OOP
 * principles and apply them effectively in real-world scenarios. The use of
 * emojis and commented diagrams aims to make the learning experience
 * engaging and enjoyable.
 *
 * Happy Coding! 🚀
 */