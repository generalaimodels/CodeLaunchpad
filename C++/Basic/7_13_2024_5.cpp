#include <iostream>
#include <string>





void greet(std::string name = "Guest") {
    std::cout << "Hello, " << name << "!" << std::endl;
}

int Greeting_Main() {
    greet();  // Output: Hello, Guest!
    greet("Alice");  // Output: Hello, Alice!
    return 0;
}



void describe(std::string name, int age) {
    std::cout << name << " is " << age << " years old." << std::endl;
}

int Describe_Main() {
    describe("Alice", 30);  // Output: Alice is 30 years old.
    describe("Bob", 25);  // Output: Bob is 25 years old.
    return 0;
}


struct UserInfo {
    std::string name = "Guest";
    int age = 0;
    std::string country = "Unknown";
};

void printUserInfo(const UserInfo& info = {}) {
    std::cout << "Name: " << info.name << std::endl;
    std::cout << "Age: " << info.age << std::endl;
    std::cout << "Country: " << info.country << std::endl;
}

int PrintUserInfo_Main() {
    printUserInfo();  // Uses all default values
    printUserInfo({"Alice", 30, "USA"});  // Specifies all values
    // printUserInfo({.name = "Bob", .country = "Canada"});  // Specifies some values, uses default for age
    return 0;
}

// Function with default argument values
int add(int a, int b = 0, int c = 0) {
    return a + b + c;
}

int Add_Main() {
    int result1 = add(5, 3);     // a = 5, b = 3, c = 0 (default)
    int result2 = add(5, 3, 2);  // a = 5, b = 3, c = 2
    // ...
    return 0;
}

// Function overloading
int add1(int a, int b) {
    return a + b;
}

double add2(double a, double b) {
    return a + b;
}

int Call_Main() {
    int result1 = add1(3, 4);     // calls add(int, int)
    double result2 = add2(3.5, 4.2); // calls add(double, double)
    // ...
    return 0;
}

int main() {
    Greeting_Main();
    Describe_Main();
    PrintUserInfo_Main();
    Add_Main();
    Call_Main();

    return 0;
}