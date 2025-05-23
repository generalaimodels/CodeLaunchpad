/*
 * File: functions_examples.cpp
 * Description: This file demonstrates various concepts of functions in C++, 
 * ranging from basic to advanced examples. Each example is designed to cover
 * important aspects of functions including definition, declaration, parameters,
 * return types, scope, and more.
 * 
 * Author: Advanced C++ Developer
 * Date: 2023-10-01
 */

#include <iostream> // For standard input/output
#include <stdexcept> // For exception handling
#include <string> // For string operations
#include <vector> // For vector operations
#include <functional> // For std::function in advanced examples

using namespace std;

// ====================
// Example 1: Basic Function - No Parameters, No Return Value
// ====================

// Function Prototype (Declaration)
void sayHello(); // Function that prints a greeting message.

// ====================
// Example 2: Function with Parameters - Simple Addition
// ====================

// Function Prototype (Declaration)
int add(int a, int b); // Adds two integers and returns the result.

// ====================
// Example 3: Function with Return Value and Parameters - Multiplication
// ====================

// Function Prototype (Declaration)
double multiply(double a, double b); // Multiplies two doubles and returns the product.

// ====================
// Example 4: Function Overloading - Same Function Name, Different Parameters
// ====================

// Function Prototypes (Declarations)
int max(int a, int b);               // Returns the maximum of two integers.
double max(double a, double b);      // Returns the maximum of two doubles.
string max(const string& a, const string& b); // Returns the lexicographically larger string.

// ====================
// Example 5: Advanced Function - Templates, Recursion, Lambdas
// ====================

// Function Prototype (Declaration) - Template Function
template <typename T> // Template Function for factorial
T factorial(T n); // Calculates the factorial of a number using recursion.

// Function Prototype (Declaration) - Function Pointer and Lambda Function
void applyFunction(const vector<int>& data, function<int(int)> func); // Applies a function to each element in a vector.

// Main Function
int main() {
    // Example 1: Basic Function Call
    cout << "Example 1: Basic Function - sayHello()" << endl;
    sayHello();
    cout << endl;

    // Example 2: Function with Parameters
    cout << "Example 2: Function with Parameters - add(int a, int b)" << endl;
    int sum = add(5, 7);
    cout << "Sum of 5 and 7 is: " << sum << endl;
    cout << endl;

    // Example 3: Function with Return Value and Parameters
    cout << "Example 3: Function with Return Value - multiply(double a, double b)" << endl;
    double product = multiply(3.5, 2.0);
    cout << "Product of 3.5 and 2.0 is: " << product << endl;
    cout << endl;

    // Example 4: Function Overloading
    cout << "Example 4: Function Overloading - max()" << endl;
    int maxInt = max(10, 20);
    double maxDouble = max(15.5, 9.7);
    string maxString = max(string("apple"), string("banana"));
    cout << "Max of 10 and 20 is: " << maxInt << endl;
    cout << "Max of 15.5 and 9.7 is: " << maxDouble << endl;
    cout << "Lexicographically larger string between 'apple' and 'banana' is: " << maxString << endl;
    cout << endl;

    // Example 5: Advanced Function - Template, Recursion, Lambda
    cout << "Example 5: Advanced Functions - Templates, Recursion, Lambdas" << endl;

    // Factorial using template function
    int num = 5;
    cout << "Factorial of " << num << " is: " << factorial(num) << endl;

    // Using lambda function with applyFunction
    vector<int> data = {1, 2, 3, 4, 5};
    cout << "Applying lambda function to double each element in the vector:" << endl;
    applyFunction(data, [](int x) -> int { return x * 2; });

    cout << endl;
    return 0;
}

// ====================
// Function Definitions
// ====================

// Example 1: Basic Function Definition
void sayHello() {
    cout << "Hello! Welcome to the world of C++ functions." << endl;
}

// Example 2: Function with Parameters Definition
int add(int a, int b) {
    // Local variables
    int sum = a + b;
    return sum;
}

// Example 3: Function with Return Value and Parameters Definition
double multiply(double a, double b) {
    // Exception handling for special cases
    if (a == 0 || b == 0) {
        cerr << "Warning: One of the multiplicands is zero." << endl;
    }
    double product = a * b;
    return product;
}

// Example 4: Function Overloading Definitions
int max(int a, int b) {
    return (a > b) ? a : b;
}

double max(double a, double b) {
    return (a > b) ? a : b;
}

string max(const string& a, const string& b) {
    return (a > b) ? a : b;
}

// Example 5: Advanced Function Definitions

// Template Function Definition - Recursive Factorial
template <typename T>
T factorial(T n) {
    if (n < 0) {
        throw invalid_argument("Negative input not allowed in factorial.");
    }
    if (n == 0 || n == 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

// Function Definition - Apply Function to Vector Elements
void applyFunction(const vector<int>& data, function<int(int)> func) {
    for (int value : data) {
        int result = func(value); // Applying the passed function to each element
        cout << result << " ";
    }
    cout << endl;
}