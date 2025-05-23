/***************************************************************
 * Chapter 2: Data: The Building Blocks 🧱 of C++
 *
 * This file demonstrates:
 *   2.1 Variables: Containers for Information 📦
 *   2.2 Data Types: Kinds of Information 📊
 *   2.3 Operators: Performing Actions ➕➖✖️➗
 *
 * Each section is built with 5 examples to illustrate key concepts.
 * Comments include brief explanations, potential pitfalls, and some fun symbols.
 ***************************************************************/

#include <iostream>   // For std::cout, std::endl
#include <string>     // For std::string
#include <vector>     // For std::vector
#include <tuple>      // For std::tuple, std::make_tuple
#include <algorithm>  // For std::find

// ---------------------------
// 2.1 Variables: Containers for Information 📦
// ---------------------------
void variablesDemo() {
    std::cout << "=== Variables Demo ===" << std::endl;
    
    // Example 1: Declare an integer variable.
    int age = 30; // 📦 'age' contains 30
    std::cout << "Age: " << age << std::endl;
    
    // Example 2: Declare a string variable.
    std::string name = "Alice"; // 📦 'name' contains "Alice"
    std::cout << "Name: " << name << std::endl;
    
    // Example 3: Declare a boolean variable.
    bool isStudent = false; // 📦 'isStudent' contains false (✅/❌ state)
    std::cout << "Is Student: " << std::boolalpha << isStudent << std::endl;
    
    // Example 4: Incorrect variable assignment (Type mismatch).
    // Uncommenting the line below would result in a compile-time error ❌.
    // age = "Thirty"; // ERROR: Cannot assign a string to an int variable.
    
    // Example 5: Using 'auto' for type inference (modern C++ feature).
    auto dynamicVar = 3.14; // 'auto' deduces type as double
    std::cout << "Dynamic Variable (auto deduced as double): " << dynamicVar << std::endl;
    
    std::cout << std::endl;
}

// ---------------------------
// 2.2 Data Types: Kinds of Information 📊
// ---------------------------
void dataTypesDemo() {
    std::cout << "=== Data Types Demo ===" << std::endl;
    
    // Example 1: Integer (int)
    int count = 100;
    std::cout << "Count (int): " << count << std::endl;
    
    // Example 2: Floating point number (double)
    double pi = 3.14159;
    std::cout << "Pi (double): " << pi << std::endl;
    
    // Example 3: Character and string (char and std::string)
    char letter = 'A';    // single character
    std::string word = "Hello"; // sequence of characters
    std::cout << "Letter (char): " << letter << ", Word (string): " << word << std::endl;
    
    // Example 4: Boolean (bool)
    bool flag = true;
    std::cout << "Flag (bool): " << std::boolalpha << flag << std::endl;
    
    // Example 5: Composite Data Types:
    // 5a: Vector (similar to a mutable list in Python)
    std::vector<int> numbers = {1, 2, 3, 4, 5}; 
    std::cout << "Numbers (vector): ";
    for (int num : numbers) {
        std::cout << num << " ";  // iterating and printing each number
    }
    std::cout << std::endl;
    
    // 5b: Tuple (fixed collection of elements)
    std::tuple<int, std::string, double> person = std::make_tuple(1, "Alice", 3.5);
    // Access tuple elements using std::get<>
    std::cout << "Tuple Person: ID = " << std::get<0>(person)
         << ", Name = " << std::get<1>(person)
         << ", Score = " << std::get<2>(person) << std::endl;
    
    std::cout << std::endl;
}

// ---------------------------
// 2.3 Operators: Performing Actions ➕➖✖️➗
// ---------------------------
void operatorsDemo() {
    std::cout << "=== Operators Demo ===" << std::endl;
    
    // Example 1: Arithmetic Operators
    int a = 10, b = 3;
    std::cout << "a + b = " << a + b << "  // Addition" << std::endl;
    std::cout << "a - b = " << a - b << "  // Subtraction" << std::endl;
    std::cout << "a * b = " << a * b << "  // Multiplication" << std::endl;
    std::cout << "a / b = " << a / b << "  // Division (integer division)" << std::endl;
    std::cout << "a % b = " << a % b << "  // Modulo (remainder)" << std::endl;
    // For floating-point division, cast one operand.
    std::cout << "a / (double)b = " << a / (double)b << "  // Floating-point division" << std::endl;
    
    // Example 2: Comparison Operators
    std::cout << std::boolalpha; // Print booleans as true/false
    std::cout << "a == b: " << (a == b) << std::endl;
    std::cout << "a != b: " << (a != b) << std::endl;
    std::cout << "a > b: " << (a > b) << std::endl;
    std::cout << "a < b: " << (a < b) << std::endl;
    std::cout << "a >= b: " << (a >= b) << std::endl;
    std::cout << "a <= b: " << (a <= b) << std::endl;
    
    // Example 3: Assignment Operators (+=, *=, etc.)
    int c = 5;
    c += 3;  // c becomes 8 (c = c + 3)
    std::cout << "c after += 3: " << c << std::endl;
    c *= 2;  // c becomes 16 (c = c * 2)
    std::cout << "c after *= 2: " << c << std::endl;
    
    // Example 4: Logical Operators
    bool cond1 = true, cond2 = false;
    std::cout << "cond1 && cond2: " << (cond1 && cond2) << "  // Logical AND" << std::endl;
    std::cout << "cond1 || cond2: " << (cond1 || cond2) << "  // Logical OR" << std::endl;
    std::cout << "!cond1: " << (!cond1) << "  // Logical NOT" << std::endl;
    
    // Example 5: Membership-like Check using std::find in a vector
    std::vector<int> vec = {1, 2, 3, 4, 5};
    int searchValue = 3;
    bool found = (std::find(vec.begin(), vec.end(), searchValue) != vec.end());
    std::cout << "Value " << searchValue 
         << (found ? " is " : " is not ") 
         << "in the vector." << std::endl;
    
    // Bonus Example: Identity Check (comparing memory addresses)
    int x = 10;
    int y = 10;
    std::cout << "Address of x: " << &x << ", Address of y: " << &y << std::endl;
    std::cout << "x and y have " << ((&x == &y) ? "the same" : "different") 
         << " addresses (Identity check)" << std::endl;
    
    std::cout << std::endl;
}

int main() {
    // Each demo function gradually increases complexity while covering all key concepts.
    variablesDemo();
    dataTypesDemo();
    operatorsDemo();
    
    return 0;
}
