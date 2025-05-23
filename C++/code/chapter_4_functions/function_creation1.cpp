/***********************************************************************
 * Chapter 4: Functions - Creating Reusable Code Blocks 🧩⚙️
 *
 * Concepts Covered:
 *   - Why Functions? Organizing and Reusing Code 🧩
 *   - Defining and Calling Functions ⚙️📞
 *   - Parameters, Return Types, and Scope
 *
 * Each section provides 5 examples to illustrate:
 *   1. Basic function definition & calling.
 *   2. Parameter usage and return values.
 *   3. Default parameters.
 *   4. Function overloading.
 *   5. Recursion & advanced function concepts.
 *
 * Comments include brief explanations and potential pitfalls.
 ***********************************************************************/

#include <iostream>   // For std::cout, std::endl
#include <string>     // For std::string

// ==========================================================
// 4.1 Basic Function Definition & Calling ⚙️📞
// ==========================================================

// Example 1: A simple function with no parameters and no return value.
void greet() {
    // 🛠️ A basic reusable tool that prints a greeting.
    std::cout << "Hello, welcome to C++ functions! 👋\n";
}

// Example 2: Function with parameters and a return value.
int add(int a, int b) {
    // Returns the sum of two numbers.
    return a + b;
}

// Example 3: Function with default parameters.
double multiply(double a, double b = 2.0) {
    // Default multiplier is 2.0 if not provided.
    return a * b;
}

// Example 4: Function overloading (same name, different parameter types).
// Overloaded function: prints integer value.
void printValue(int value) {
    std::cout << "Integer value: " << value << "\n";
}
// Overloaded function: prints string value.
void printValue(const std::string& value) {
    std::cout << "String value: " << value << "\n";
}

// Example 5: Recursive function to compute factorial.
// Note: Be cautious with recursion; ensure a proper base case.
int factorial(int n) {
    if (n < 0) { 
        std::cerr << "Error: Negative input for factorial! ❌\n";
        return -1; // Indicate error.
    }
    if (n == 0 || n == 1) { 
        return 1; // Base case.
    }
    return n * factorial(n - 1); // Recursive call.
}

// ==========================================================
// 4.2 Parameters, Return Types, and Variable Scope 🧩
// ==========================================================

// Example 1: Demonstrating local variables.
void localVariableDemo() {
    int localVar = 42; // Local to this function.
    std::cout << "Local variable value: " << localVar << "\n";
    // Note: localVar is not accessible outside this function.
}

// Example 2: Using a static local variable.
void staticVariableDemo() {
    static int count = 0; // Retains its value between calls.
    ++count;
    std::cout << "Static count: " << count << "\n";
}

// Example 3: Passing parameters by value vs. by reference.
// Pass by value: makes a copy.
void incrementByValue(int num) {
    num++; // This does not affect the original variable.
    std::cout << "Inside incrementByValue: " << num << "\n";
}
// Pass by reference: affects the original variable.
void incrementByReference(int &num) {
    num++;
    std::cout << "Inside incrementByReference: " << num << "\n";
}

// Example 4: Passing parameters by pointer.
void incrementByPointer(int *num) {
    if (num == nullptr) {
        std::cerr << "Error: nullptr passed to incrementByPointer! ❌\n";
        return;
    }
    (*num)++;
    std::cout << "Inside incrementByPointer: " << *num << "\n";
}

// Example 5: Function with constant parameters.
int sumConst(const int a, const int b) {
    // a and b cannot be modified inside the function.
    return a + b;
}

// ==========================================================
// 4.3 Advanced Function Concepts: Lambdas & Function Pointers 🧩
// ==========================================================

// Example 1: A simple lambda function.
void lambdaDemo() {
    auto square = [](int x) -> int { // Lambda to compute square.
        return x * x;
    };
    std::cout << "Square of 5 (lambda): " << square(5) << "\n";
}

// Example 2: Lambda with capture list.
void lambdaCaptureDemo() {
    int factor = 3;
    auto multiplyByFactor = [factor](int x) -> int {
        return x * factor;
    };
    std::cout << "6 multiplied by factor (lambda capture): " << multiplyByFactor(6) << "\n";
}

// Example 3: Function pointer as a parameter.
// A function that applies a given function pointer to two integers.
int applyOperation(int a, int b, int (*operation)(int, int)) {
    return operation(a, b);
}
int subtract(int a, int b) {
    return a - b;
}

// Example 4: Returning a function pointer.
typedef int (*Operation)(int, int);
Operation getOperation(char op) {
    if (op == '+') {
        return add; // Reusing previously defined add function.
    } else if (op == '-') {
        return subtract;
    } else {
        return nullptr;
    }
}

// Example 5: Using lambda with standard algorithm (as a tool).
// (For demonstration; not using <algorithm> here to keep file self-contained.)
void lambdaForLoopDemo() {
    // Using lambda to print numbers 1 to 5.
    auto printNum = [](int x) {
        std::cout << x << " ";
    };
    for (int i = 1; i <= 5; ++i) {
        printNum(i);
    }
    std::cout << "\n";
}

// ==========================================================
// main() - Entry point for the program
// ==========================================================
int main() {
    std::cout << "=== Chapter 4: Functions Demo ===\n\n";
    
    // -------- 4.1 Basic Function Usage --------
    std::cout << "4.1 Basic Function Usage:\n";
    // Ex1: Basic greeting.
    greet();
    
    // Ex2: Function with parameters and return.
    int sum = add(5, 7);
    std::cout << "Sum (5 + 7): " << sum << "\n";
    
    // Ex3: Function with default parameter.
    std::cout << "Multiply (3.5 * default 2.0): " << multiply(3.5) << "\n";
    std::cout << "Multiply (3.5 * 4.0): " << multiply(3.5, 4.0) << "\n";
    
    // Ex4: Function overloading.
    printValue(100);
    printValue("Overloaded function call");
    
    // Ex5: Recursive function.
    int fact = factorial(5);
    if (fact != -1) {  // Check for error.
        std::cout << "Factorial of 5: " << fact << "\n";
    }
    std::cout << "\n";
    
    // -------- 4.2 Parameters, Return Types, and Scope --------
    std::cout << "4.2 Variable Scope & Parameter Passing:\n";
    // Ex1: Local variable.
    localVariableDemo();
    
    // Ex2: Static local variable demonstration.
    staticVariableDemo();
    staticVariableDemo(); // Call twice to show persistence.
    
    // Ex3: Passing by value vs. reference.
    int num = 10;
    incrementByValue(num);
    std::cout << "After incrementByValue, num = " << num << "\n"; // Remains 10.
    incrementByReference(num);
    std::cout << "After incrementByReference, num = " << num << "\n"; // Now 11.
    
    // Ex4: Passing by pointer.
    incrementByPointer(&num);
    std::cout << "After incrementByPointer, num = " << num << "\n"; // Now 12.
    // Uncomment below to see error handling (passing nullptr).
    // incrementByPointer(nullptr);
    
    // Ex5: Function with constant parameters.
    std::cout << "Sum using constant params (7 + 8): " << sumConst(7, 8) << "\n\n";
    
    // -------- 4.3 Advanced Function Concepts --------
    std::cout << "4.3 Advanced Function Concepts:\n";
    // Ex1: Simple lambda.
    lambdaDemo();
    
    // Ex2: Lambda with capture.
    lambdaCaptureDemo();
    
    // Ex3: Function pointer as parameter.
    int result = applyOperation(10, 3, subtract);
    std::cout << "Applying subtract using function pointer: " << result << "\n";
    
    // Ex4: Returning a function pointer.
    Operation op = getOperation('+');
    if (op != nullptr) {
        std::cout << "Operation '+' (5 + 4): " << op(5, 4) << "\n";
    } else {
        std::cout << "Operation not recognized!\n";
    }
    
    // Ex5: Lambda used in a loop.
    lambdaForLoopDemo();
    
    std::cout << "\n=== End of Functions Demo ===\n";
    return 0;
}
