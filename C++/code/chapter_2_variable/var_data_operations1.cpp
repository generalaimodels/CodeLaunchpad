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

#include <iostream>  // for cout, cin
#include <string> // for string
#include <vector> // for vector
#include <tuple> // for tuple
#include <algorithm>  // for std::find

using namespace std;

// ---------------------------
// 2.1 Variables: Containers for Information 📦
// ---------------------------
void variablesDemo() {
    cout << "=== Variables Demo ===" << endl;
    
    // Example 1: Declare an integer variable.
    int age = 30; // 📦 'age' contains 30
    cout << "Age: " << age << endl;
    
    // Example 2: Declare a string variable.
    string name = "Alice"; // 📦 'name' contains "Alice"
    cout << "Name: " << name << endl;
    
    // Example 3: Declare a boolean variable.
    bool isStudent = false; // 📦 'isStudent' contains false (✅/❌ state)
    cout << "Is Student: " << boolalpha << isStudent << endl;
    
    // Example 4: Incorrect variable assignment (Type mismatch).
    // Uncommenting the line below would result in a compile-time error ❌.
    // age = "Thirty"; // ERROR: Cannot assign a string to an int variable.
    
    // Example 5: Using 'auto' for type inference (modern C++ feature).
    auto dynamicVar = 3.14; // 'auto' deduces type as double
    cout << "Dynamic Variable (auto deduced as double): " << dynamicVar << endl;
    
    cout << endl;
}

// ---------------------------
// 2.2 Data Types: Kinds of Information 📊
// ---------------------------
void dataTypesDemo() {
    cout << "=== Data Types Demo ===" << endl;
    
    // Example 1: Integer (int)
    int count = 100;
    cout << "Count (int): " << count << endl;
    
    // Example 2: Floating point number (double)
    double pi = 3.14159;
    cout << "Pi (double): " << pi << endl;
    
    // Example 3: Character and string (char and std::string)
    char letter = 'A';    // single character
    string word = "Hello"; // sequence of characters
    cout << "Letter (char): " << letter << ", Word (string): " << word << endl;
    
    // Example 4: Boolean (bool)
    bool flag = true;
    cout << "Flag (bool): " << boolalpha << flag << endl;
    
    // Example 5: Composite Data Types:
    // 5a: Vector (similar to a mutable list in Python)
    vector<int> numbers = {1, 2, 3, 4, 5}; 
    cout << "Numbers (vector): ";
    for (int num : numbers) {
        cout << num << " ";  // iterating and printing each number
    }
    cout << endl;
    
    // 5b: Tuple (fixed collection of elements)
    tuple<int, string, double> person = make_tuple(1, "Alice", 3.5);
    // Access tuple elements using std::get<>
    cout << "Tuple Person: ID = " << get<0>(person)
         << ", Name = " << get<1>(person)
         << ", Score = " << get<2>(person) << endl;
    
    cout << endl;
}

// ---------------------------
// 2.3 Operators: Performing Actions ➕➖✖️➗
// ---------------------------
void operatorsDemo() {
    cout << "=== Operators Demo ===" << endl;
    
    // Example 1: Arithmetic Operators
    int a = 10, b = 3;
    cout << "a + b = " << a + b << "  // Addition" << endl;
    cout << "a - b = " << a - b << "  // Subtraction" << endl;
    cout << "a * b = " << a * b << "  // Multiplication" << endl;
    cout << "a / b = " << a / b << "  // Division (integer division)" << endl;
    cout << "a % b = " << a % b << "  // Modulo (remainder)" << endl;
    // For floating-point division, cast one operand.
    cout << "a / (double)b = " << a / (double)b << "  // Floating-point division" << endl;
    
    // Example 2: Comparison Operators
    cout << boolalpha; // Print booleans as true/false
    cout << "a == b: " << (a == b) << endl;
    cout << "a != b: " << (a != b) << endl;
    cout << "a > b: " << (a > b) << endl;
    cout << "a < b: " << (a < b) << endl;
    cout << "a >= b: " << (a >= b) << endl;
    cout << "a <= b: " << (a <= b) << endl;
    
    // Example 3: Assignment Operators (+=, *=, etc.)
    int c = 5;
    c += 3;  // c becomes 8 (c = c + 3)
    cout << "c after += 3: " << c << endl;
    c *= 2;  // c becomes 16 (c = c * 2)
    cout << "c after *= 2: " << c << endl;
    
    // Example 4: Logical Operators
    bool cond1 = true, cond2 = false;
    cout << "cond1 && cond2: " << (cond1 && cond2) << "  // Logical AND" << endl;
    cout << "cond1 || cond2: " << (cond1 || cond2) << "  // Logical OR" << endl;
    cout << "!cond1: " << (!cond1) << "  // Logical NOT" << endl;
    
    // Example 5: Membership-like Check using std::find in a vector
    vector<int> vec = {1, 2, 3, 4, 5};
    int searchValue = 3;
    bool found = (find(vec.begin(), vec.end(), searchValue) != vec.end());
    cout << "Value " << searchValue 
         << (found ? " is " : " is not ") 
         << "in the vector." << endl;
    
    // Bonus Example: Identity Check (comparing memory addresses)
    int x = 10;
    int y = 10;
    cout << "Address of x: " << &x << ", Address of y: " << &y << endl;
    cout << "x and y have " << ((&x == &y) ? "the same" : "different") 
         << " addresses (Identity check)" << endl;
    
    cout << endl;
}

int main() {
    // Each demo function gradually increases complexity while covering all key concepts.
    variablesDemo();
    dataTypesDemo();
    operatorsDemo();
    
    return 0;
}
