/*****************************************************
 * Chapter 4: Functions - Creating Reusable Code Blocks 🧩⚙️
 * Advanced C++ Examples for Developers (Medium/High-Level)
 *
 * This file demonstrates:
 *   - Why Functions? Organizing and Reusing Code 🛠️➡️🧩
 *   - Defining and Calling Functions (Declaration, Definition, Call)
 *   - Parameter passing (by value, by reference, constant parameters)
 *   - Overloading, Recursion, Lambdas, Exception Handling, etc.
 *
 * Each section contains 5 examples (our signature).
 *****************************************************/

#include <iostream>
#include <string>
#include <stdexcept>  // For exception handling
#include <initializer_list>
using namespace std;

// ======================================================
// Section 1: Basic Function Definitions and Calls ⚙️
// ======================================================

// Example 1: Simple function with no parameters (prints a greeting).
void greet() { // Basic reusable code block.
    cout << "Hello, welcome to Chapter 4! 👋\n";
}

// Example 2: Function with parameters and a return value (addition).
int add(int a, int b) { // Returns the sum of two integers.
    return a + b;
}

// Example 3: Function with a default parameter (multiplication).
double multiply(double a, double b = 1.0) { 
    // Default parameter avoids errors if second argument is missing.
    return a * b;
}

// Example 4: Overloaded functions (same name, different parameter types).
void printValue(int value) {
    cout << "Integer value: " << value << "\n";
}
void printValue(const string &value) {
    cout << "String value: " << value << "\n";
}

// Example 5: Recursive function (calculates factorial).
int factorial(int n) {
    // Check for invalid input.
    if (n < 0) {
        throw invalid_argument("Factorial is not defined for negative numbers.");
    }
    if (n == 0 || n == 1) {
        return 1; // Base case.
    }
    return n * factorial(n - 1); // Recursive call.
}

// ======================================================
// Section 2: Parameter Passing, Scope & Lambdas 🧩
// ======================================================

// Example 1: Function demonstrating local variables.
void localVariableDemo() {
    int localVar = 42; // Local variable: only accessible in this function.
    cout << "Local variable value: " << localVar << "\n";
}

// Example 2: Pass by value (modification does not affect caller).
void incrementByValue(int num) {
    num++; // This change is local only.
    cout << "Inside incrementByValue (by value): " << num << "\n";
}

// Example 3: Pass by reference (modification affects caller).
void incrementByReference(int &num) {
    num++; // Changes affect the original variable.
    cout << "Inside incrementByReference (by reference): " << num << "\n";
}

// Example 4: Constant parameter (read-only input).
void displayMessage(const string &msg) {
    // msg cannot be modified here.
    cout << "Message: " << msg << "\n";
}

// Example 5: Lambda function (anonymous inline function).
void lambdaDemo() {
    // Lambda to add two numbers.
    auto addLambda = [](int x, int y) -> int { return x + y; };
    cout << "Lambda add result (3 + 4): " << addLambda(3, 4) << "\n";
}

// ======================================================
// Section 3: Advanced Function Concepts 🚀
// ======================================================

// Example 1: Function that throws an exception on invalid input.
double safeDivide(double numerator, double denominator) {
    if (denominator == 0) { // Check to avoid division by zero.
        throw runtime_error("Error: Division by zero!");
    }
    return numerator / denominator;
}

// Example 2: Using function pointers (store and call a function).
int subtract(int a, int b) {
    return a - b;
}
void functionPointerDemo() {
    // Declare a function pointer that takes two ints and returns an int.
    int (*funcPtr)(int, int) = subtract;
    cout << "Function pointer subtract (10 - 3): " << funcPtr(10, 3) << "\n";
}

// Example 3: Functor (function object) for customizable operations.
struct Adder {
    int operator()(int a, int b) const { 
        return a + b;
    }
};
void functorDemo() {
    Adder addObj;
    cout << "Functor add (7 + 8): " << addObj(7, 8) << "\n";
}

// Example 4: Inline function for performance (suggestion to compiler).
inline int square(int x) { // Hint: embed function code where called.
    return x * x;
}
void inlineDemo() {
    cout << "Inline square (5^2): " << square(5) << "\n";
}

// Example 5: Variadic template function (sums an arbitrary number of values).
template<typename T>
T variadicSum(T t) {
    return t;
}
template<typename T, typename... Args>
T variadicSum(T first, Args... args) {
    return first + variadicSum(args...);
}
void variadicTemplateDemo() {
    // Sum multiple integers.
    cout << "Variadic sum (1+2+3+4+5): " << variadicSum(1, 2, 3, 4, 5) << "\n";
}

// ======================================================
// Main function: Calls all the examples above.
// ======================================================
int main() {
    cout << "=== Chapter 4: Functions Demo ===\n\n";
    
    // --- Section 1: Basic Functions ---
    cout << "Section 1: Basic Function Definitions & Calls\n";
    greet(); // Example 1
    cout << "Add 5 + 3: " << add(5, 3) << "\n"; // Example 2
    cout << "Multiply 4 * 2 (with default): " << multiply(4, 2) << "\n"; // Example 3
    cout << "Multiply 7 (using default multiplier): " << multiply(7) << "\n";
    printValue(100);          // Example 4a
    printValue("Reusable");   // Example 4b
    try {
        cout << "Factorial of 5: " << factorial(5) << "\n"; // Example 5
    } catch (const exception &e) {
        cout << "Exception: " << e.what() << "\n";
    }
    cout << "\n";
    
    // --- Section 2: Parameter Passing & Lambdas ---
    cout << "Section 2: Parameter Passing, Scope & Lambdas\n";
    localVariableDemo(); // Example 1
    int val = 10;
    cout << "Before incrementByValue: " << val << "\n";
    incrementByValue(val); // Example 2 (by value, val remains unchanged)
    cout << "After incrementByValue: " << val << "\n";
    cout << "Before incrementByReference: " << val << "\n";
    incrementByReference(val); // Example 3 (by reference, val is updated)
    cout << "After incrementByReference: " << val << "\n";
    displayMessage("Constant parameters prevent modifications."); // Example 4
    lambdaDemo(); // Example 5
    cout << "\n";
    
    // --- Section 3: Advanced Function Concepts ---
    cout << "Section 3: Advanced Function Concepts\n";
    try {
        cout << "Safe division (10 / 2): " << safeDivide(10, 2) << "\n"; // Example 1
        // Uncomment the following line to see exception handling in action.
        // cout << "Safe division (10 / 0): " << safeDivide(10, 0) << "\n";
    } catch (const exception &e) {
        cout << "Exception caught: " << e.what() << "\n";
    }
    functionPointerDemo(); // Example 2
    functorDemo();         // Example 3
    inlineDemo();          // Example 4
    variadicTemplateDemo(); // Example 5
    
    cout << "\n=== End of Chapter 4 Demo ===\n";
    return 0;
}
