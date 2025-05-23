/***************************************************************
 * Chapter 5: Arrays and Strings - Working with Collections 📦📚
 * 
 * This file demonstrates:
 * 1. Arrays (one-dimensional, multi-dimensional, dynamic) 📦🔢
 * 2. Strings (C-style & std::string operations) 🏷️📜🔤
 *
 * Each section provides 5 example snippets to enhance your skills.
 * Review comments for tips, common pitfalls, and best practices.
 ***************************************************************/

#include <iostream>
#include <string>
#include <sstream> 
#include <stdexcept>
#include <exception>
using namespace std;

// ================================================================
// Section 1: Arrays Examples 📦🔢
// ================================================================

// Example 1.1: Basic Declaration & Initialization of a one-dimensional array.
// Analogy: Think of these as a row of numbered lockers.
void arrayExample1() {
    cout << "Array Example 1: Basic Declaration & Initialization 📦🔢\n";
    // Declare an array of 5 integers.
    int lockers[5] = {101, 102, 103, 104, 105}; // 🔢 Each locker holds a number.
    
    // Display each element using index.
    for (int i = 0; i < 5; i++) {
        cout << "Locker " << i << " holds: " << lockers[i] << "\n"; // 🔢 Access by index.
    }
    cout << "---------------------------------------------\n";
}

// Example 1.2: Safe Access to Array Elements with manual bounds checking.
// 💡 Always check bounds to avoid accessing non-existent lockers.
void arrayExample2() {
    cout << "Array Example 2: Safe Access with Bounds Checking 📦🚧\n";
    int lockers[5] = {201, 202, 203, 204, 205};
    int index = 3;  // Change this index to test safe access.
    
    // Check bounds before accessing.
    if (index >= 0 && index < 5) {
        cout << "Accessing locker " << index << ": " << lockers[index] << "\n";
    } else {
        cerr << "Error: Index " << index << " is out of bounds! 🚫\n";
    }
    cout << "---------------------------------------------\n";
}

// Example 1.3: Simulating an Out-of-Bounds Access with Exception Handling.
// Note: Raw arrays do not throw exceptions on out-of-bound access, so we simulate.
void arrayExample3() {
    cout << "Array Example 3: Simulated Out-of-Bounds Exception 📦❗\n";
    int lockers[5] = {301, 302, 303, 304, 305};
    int index = 7; // Deliberately out-of-bound.
    
    try {
        if (index < 0 || index >= 5)
            throw out_of_range("Index " + to_string(index) + " is out of bounds!");
        cout << "Locker " << index << ": " << lockers[index] << "\n";
    } catch (const exception& ex) {
        cerr << "Exception caught: " << ex.what() << "\n";
    }
    cout << "---------------------------------------------\n";
}

// Example 1.4: Multi-dimensional Arrays (Grids) 📦🔢🔢
// Think of it as a table with rows and columns.
void arrayExample4() {
    cout << "Array Example 4: Multi-dimensional Arrays (2D Grid) 📦🗂️\n";
    const int rows = 3, cols = 4;
    int grid[rows][cols] = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12}
    };
    
    // Print the 2D grid.
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << grid[i][j] << "\t"; // Tab-delimited for clarity.
        }
        cout << "\n";
    }
    cout << "---------------------------------------------\n";
}

// Example 1.5: Dynamic Array Allocation (Advanced) 📦💻
// Using dynamic memory allocation to handle arrays when size is not known at compile time.
void arrayExample5() {
    cout << "Array Example 5: Dynamic Array Allocation 📦💻\n";
    int n;
    cout << "Enter size of dynamic array: ";
    cin >> n;
    
    // Allocate dynamic array. Remember to free the memory later!
    int* dynamicLockers = new (nothrow) int[n]; // Using nothrow to avoid exceptions on failure.
    if (!dynamicLockers) {
        cerr << "Memory allocation failed! 🚫\n";
        return;
    }
    
    // Initialize and display.
    for (int i = 0; i < n; i++) {
        dynamicLockers[i] = 1000 + i; // Assigning sample values.
        cout << "Dynamic Locker " << i << ": " << dynamicLockers[i] << "\n";
    }
    
    // Free the dynamically allocated memory.
    delete[] dynamicLockers;
    cout << "---------------------------------------------\n";
}

// ================================================================
// Section 2: Strings Examples 🏷️📜🔤
// ================================================================

// Example 2.1: Working with C-style Strings (Character Arrays).
// Remember: C-style strings end with a null terminator '\0'.
void stringExample1() {
    cout << "String Example 1: C-style Strings 📜🔤\n";
    char greeting[20] = "Hello, World!"; // 🔤 Declaration and initialization.
    
    // Display the C-style string.
    cout << "Greeting: " << greeting << "\n";
    cout << "---------------------------------------------\n";
}

// Example 2.2: Basic Operations with std::string (Concatenation, Comparison).
void stringExample2() {
    cout << "String Example 2: std::string Basic Operations 🏷️✨\n";
    string firstName = "John";
    string lastName = "Doe";
    
    // Concatenation.
    string fullName = firstName + " " + lastName;
    cout << "Full Name: " << fullName << "\n";
    
    // Comparison.
    if (firstName == "John") {
        cout << "First name is John. ✅\n";
    }
    cout << "---------------------------------------------\n";
}

// Example 2.3: Using std::string Methods (Finding Substrings, etc.).
void stringExample3() {
    cout << "String Example 3: std::string Methods (Find, Substr) 🔤🔍\n";
    string sentence = "C++ programming is powerful!";
    
    // Finding a substring.
    size_t pos = sentence.find("Programming");
    if (pos != string::npos) {
        cout << "\"programming\" found at position: " << pos << "\n";
    } else {
        cout << "\"programming\" not found! 🚫\n";
    }
    
    // Extracting a substring.
    string extracted = sentence.substr(pos, 11); // "programming"
    cout << "Extracted substring: " << extracted << "\n";
    cout << "---------------------------------------------\n";
}

// Example 2.4: Input and Output of std::string.
// Demonstrates safe string input.
void stringExample4() {
    cout << "String Example 4: Input and Output with std::string 🏷️📝\n";
    cout << "Enter your favorite programming language: ";
    string language;
    // Using getline to capture spaces.
    getline(cin >> ws, language); // ws consumes any leading whitespace.
    cout << "You entered: " << language << "\n";
    cout << "---------------------------------------------\n";
}

// Example 2.5: Advanced String Manipulation with stringstream.
// Useful for formatting and parsing strings.
void stringExample5() {
    cout << "String Example 5: Advanced Manipulation with stringstream 📜🔧\n";
    string data = "2025 02 02"; // Format: YYYY MM DD
    stringstream ss(data);
    
    int year, month, day;
    ss >> year >> month >> day;
    
    // Display formatted date.
    cout << "Formatted Date: " << (month < 10 ? "0" : "") << month 
         << "/" << (day < 10 ? "0" : "") << day 
         << "/" << year << "\n";
    cout << "---------------------------------------------\n";
}

// ================================================================
// Main Function: Execute All Examples
// ================================================================
int main() {
    cout << "=== Chapter 5: Arrays and Strings ===\n\n";
    
    // ----- Arrays Section -----
    arrayExample1(); // Basic array initialization.
    arrayExample2(); // Safe access with bounds checking.
    arrayExample3(); // Simulated exception for out-of-bound access.
    arrayExample4(); // Multi-dimensional (2D) arrays.
    arrayExample5(); // Dynamic array allocation (advanced).
    
    // ----- Strings Section -----
    stringExample1(); // C-style string basics.
    stringExample2(); // std::string concatenation & comparison.
    stringExample3(); // String methods: find and substr.
    stringExample4(); // Input and output with std::string.
    stringExample5(); // Advanced string manipulation using stringstream.
    
    cout << "\n=== End of Examples ===\n";
    return 0;
}
