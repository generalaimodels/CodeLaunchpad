/*******************************************************
 * Chapter 8: Introduction to Object-Oriented Programming (OOP) - Thinking in Objects 🧬🤔
 * -----------------------------------------------------
 * This code file explores the basics of OOP in C++,
 * starting from simple class definitions to more advanced concepts.
 * Each example builds upon the previous one.
 *******************************************************/

#include <iostream>     // For std::cout and std::endl
#include <string>       // For std::string

// Example 1: Defining a Simple Class - The Blueprint 📐
class Car {
public:               // Public access specifier 🚪
    std::string brand; // Public attribute (data member)
    std::string model;
    int year;

    void honk() {      // Public method (member function)
        std::cout << "Beep! Beep!" << std::endl; // Action 🚗
    }
};


int main() {
    // Example 1: Creating Objects (Instances) of Class Car 🚗
    Car car1;              // Instantiate an object 'car1'
    car1.brand = "Toyota"; // Access and set public members via '.' operator
    car1.model = "Corolla";
    car1.year = 2020;

    car1.honk();           // Call method on the object

    std::cout << "Car 1: " << car1.brand << " " << car1.model << " (" << car1.year << ")" << std::endl;

    // Example 2: Private Members and Encapsulation 🔒
    class BankAccount {
    private:               // Private access specifier 🔒
        double balance;    // Private attribute

    public:
        BankAccount() {    // Constructor to initialize balance
            balance = 0.0;
        }

        void deposit(double amount) {
            if (amount > 0) {
                balance += amount;
            } else {
                std::cerr << "Invalid deposit amount." << std::endl; // Error message ⚠️
            }
        }

        void withdraw(double amount) {
            if (amount > 0 && amount <= balance) {
                balance -= amount;
            } else {
                std::cerr << "Invalid withdrawal amount." << std::endl; // Error message ⚠️
            }
        }

        double getBalance() const {
            return balance;
        }
    };

    BankAccount account;               // Create an instance of BankAccount 🏦
    account.deposit(1000.0);           // Deposit money
    account.withdraw(500.0);           // Withdraw money
    std::cout << "Current balance: $" << account.getBalance() << std::endl;

    // Possible Mistake: Attempting to access private member directly ❌
    // account.balance = 1000.0;        // Error: 'balance' is private within this context

    // Example 3: Constructors and this Pointer 🛠️
    class Person {
    private:
        std::string name;
        int age;
    public:
        Person(const std::string& name, int age) { // Parameterized constructor
            this->name = name;       // 'this' pointer refers to the current object
            this->age = age;
        }

        void displayInfo() const {
            std::cout << "Name: " << name << ", Age: " << age << std::endl;
        }
    };

    Person person("Alice", 30);       // Create a Person object
    person.displayInfo();

    // Example 4: Default and Overloaded Constructors ⚙️
    class Rectangle {
    private:
        double width;
        double height;
    public:
        Rectangle() {                 // Default constructor
            width = 1.0;
            height = 1.0;
        }

        Rectangle(double w, double h) { // Overloaded constructor
            width = w;
            height = h;
        }

        double area() const {
            return width * height;
        }
    };

    Rectangle rect1;                 // Invokes default constructor
    Rectangle rect2(5.0, 3.0);       // Invokes overloaded constructor

    std::cout << "Area of rect1: " << rect1.area() << std::endl;
    std::cout << "Area of rect2: " << rect2.area() << std::endl;

    // Example 5: Access Specifiers and Inheritance Basics 🧬
    // This is a prelude to inheritance, but we focus on access specifiers here.
    class Animal {
    protected:               // Protected access specifier 🛡️
        std::string species;

    public:
        void setSpecies(const std::string& sp) {
            species = sp;
        }

        void makeSound() const {
            std::cout << species << " makes a sound." << std::endl;
        }
    };

    Animal animal;
    // Possible Mistake: Accessing protected member directly ❌
    // animal.species = "Cat";         // Error: 'species' is protected within this context
    animal.setSpecies("Cat");
    animal.makeSound();

    return 0; // End of program
}