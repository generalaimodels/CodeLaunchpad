/* 
   advanced_examples.cpp
   -----------------------

   This file demonstrates advanced examples for two core C++ concepts:
   1. Arrays 📦🔢
   2. Strings 📜🔤

   Each section contains 10 examples, with detailed commentary,
   diagrams, and analogies to help you fully grasp the underlying concepts.

   To compile (using g++ as an example):
       g++ -std=c++11 -o advanced_examples advanced_examples.cpp
   Then run:
       ./advanced_examples
*/
#include <chrono>
#include <iostream>  // For basic input/output
#include <iomanip>   // For setw
#include <cstring>      // For C-style string functions
#include <array>        // For std::array
#include <vector>       // For std::vector
#include <algorithm>    // For sort, reverse, etc.
#include <sstream>      // For stringstream
#include <regex>        // For regular expressions
#include <cstdlib>      // For atoi, etc.

using namespace std;

// ==========================
//        ARRAYS EXAMPLES
// ==========================

// Example 1: Static One-Dimensional Array Initialization and Traversal
void arrayExample1() {
    cout << "Example 1: Static 1D Array Initialization and Traversal 📦🔢\n";
    // Imagine a row of lockers numbered 0 to 4:
    // [📦0] [📦1] [📦2] [📦3] [📦4]
    int numbers[5] = {10, 20, 30, 40, 50};

    // Traverse and print each element.
    for (int i = 0; i < 5; i++) {
        cout << "Locker " << i << " holds: " << numbers[i] << "\n";
    }
    cout << string(50, '==') << "\n\n";
}

// Example 2: Multi-Dimensional Arrays (2D Array) – Representing a Matrix
void arrayExample2() {
    cout << "Example 2: 2D Array (Matrix) Example 📦🧱\n";
    // Think of a spreadsheet where rows and columns form cells.
    int matrix[3][4] = {
        { 1,  2,  3,  4},   // Row 0
        { 5,  6,  7,  8},   // Row 1
        { 9, 10, 11, 12}    // Row 2
    };

    // Diagram (Row-Major Order):
    // [0,0] [0,1] [0,2] [0,3]
    // [1,0] [1,1] [1,2] [1,3]
    // [2,0] [2,1] [2,2] [2,3]

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            cout << "matrix[" << i << "][" << j << "] = " << setw(2) << matrix[i][j] << "  ";
        }
        cout << "\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 3: Array Initialization Using a Loop – Filling with a Pattern
void arrayExample3() {
    cout << "Example 3: Loop Initialization for a 1D Array 🔄🔢\n";
    const int SIZE = 10;
    int squares[SIZE];

    // Fill the array with square numbers: [0, 1, 4, 9, ...]
    for (int i = 0; i < SIZE; i++) {
        squares[i] = i * i;
    }

    // Visualize: [0] [1] [4] [9] [16] ... 
    for (int i = 0; i < SIZE; i++) {
        cout << "Index " << i << " -> " << squares[i] << "\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 4: Dynamic Array Allocation Using 'new' – Flexible Memory
void arrayExample4() {
    cout << "Example 4: Dynamic Array Allocation (new/delete) 📦🆕\n";
    int size;
    cout << "Enter array size: ";
    cin >> size;
    // Reserve a block of memory dynamically
    int* dynamicArray = new int[size];

    // Fill with a pattern: each element is its index multiplied by 3
    for (int i = 0; i < size; i++) {
        dynamicArray[i] = i * 3;
    }

    // Diagram: Imagine renting temporary lockers that can adjust in number.
    for (int i = 0; i < size; i++) {
        cout << "dynamicArray[" << i << "] = " << dynamicArray[i] << "\n";
    }
    delete[] dynamicArray;  // Always free dynamically allocated memory!
    cout << string(50, '=') << "\n\n";
}

// Example 5: Pointer Arithmetic on Arrays – Walking Through Memory
// Example 5: Demonstrates pointer arithmetic to traverse an array
void arrayExample5() {
    cout << "Example 5: Pointer Arithmetic with Arrays 🔗➡️\n";
    int arr[5] = {5, 10, 15, 20, 25};
    int* ptr = arr; // Points to the first element of the array

    // Using pointer arithmetic to access each element
    for (int i = 0; i < 5; i++) {
        cout << "*(ptr + " << i << ") = " << *(ptr + i) << "\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 6: Using std::array (C++11) – A Safer, Fixed-Size Array
void arrayExample6() {
    cout << "Example 6: std::array Usage 🛡️📦\n";
    std::array<int, 5> arr = {100, 200, 300, 400, 500};

    // Diagram: Think of std::array as a pre-packaged set of lockers with built-in safety.
    for (size_t i = 0; i < arr.size(); i++) {
        cout << "std::array[" << i << "] = " << arr[i] << "\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 7: Using std::vector – Dynamic and Resizable Arrays
void arrayExample7() {
    cout << "Example 7: std::vector Dynamic Array 🚀📦\n";
    vector<int> vec = {1, 2, 3};

    // Imagine a vector as an expandable series of lockers
    // Let's add more items dynamically
    vec.push_back(4);
    vec.push_back(5);

    // Display the vector contents
    for (size_t i = 0; i < vec.size(); i++) {
        cout << "vec[" << i << "] = " << vec[i] << "\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 8: Array Bounds – Safe Access Using std::vector::at()
void arrayExample8() {
    cout << "Example 8: Array Bounds Checking with std::vector::at() 🚧\n";
    vector<int> vec = {10, 20, 30, 40, 50};

    // Using .at() throws an exception if the index is out-of-bounds
    try {
        cout << "vec.at(2) = " << vec.at(2) << "\n";
        // Uncomment the next line to see exception handling in action:
        // cout << "vec.at(10) = " << vec.at(10) << "\n";
    } catch (const out_of_range& e) {
        cout << "Out of range error: " << e.what() << "\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 9: Array Algorithms – Sorting, Reversing, and Finding Elements
void arrayExample9() {
    cout << "Example 9: Using STL Algorithms with Arrays 🛠️📚\n";
    vector<int> vec = {42, 23, 17, 13, 57};

    // Sort the vector
    sort(vec.begin(), vec.end());
    cout << "Sorted vector: ";
    for (int n : vec) cout << n << " ";
    cout << "\n";

    // Reverse the vector
    reverse(vec.begin(), vec.end());
    cout << "Reversed vector: ";
    for (int n : vec) cout << n << " ";
    cout << "\n";

    // Find an element (analogy: find a specific locker)
    auto it = find(vec.begin(), vec.end(), 23);
    if (it != vec.end()) {
        cout << "Found 23 at index: " << distance(vec.begin(), it) << "\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 10: Multi-Dimensional Dynamic Array (2D) Allocation
void arrayExample10() {
    cout << "Example 10: Multi-Dimensional Dynamic Array Allocation 📦🧱 (2D)\n";
    int rows = 3, cols = 4;
    
    // Dynamically allocate an array of int pointers (each representing a row)
    int** matrix = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[cols];  // Each row gets its own block of columns
    }
    
    // Fill the matrix with values: value = (row * cols) + col
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = i * cols + j;
        }
    }
    
    // Diagram:
    // Row 0: [ 0] [ 1] [ 2] [ 3]
    // Row 1: [ 4] [ 5] [ 6] [ 7]
    // Row 2: [ 8] [ 9] [10] [11]
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << setw(3) << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    
    // Free memory
    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    cout << string(50, '=') << "\n\n";
}

// ==========================
//       STRINGS EXAMPLES
// ==========================

// Example 1: C-Style String Basic Manipulation
void stringExample1() {
    cout << "String Example 1: C-Style String Manipulation 📜🔤\n";
    // Think of C-style strings as old paper scrolls with a null terminator (🛑)
    char message[20] = "Hello";
    cout << "Original C-string: " << message << "\n";

    // Append a string using strcat (ensure buffer is large enough)
    strcat(message, " World");
    cout << "After strcat: " << message << "\n";
    cout << string(50, '=') << "\n\n";
}

// Example 2: Basic std::string Operations
void stringExample2() {
    cout << "String Example 2: Basic std::string Operations 📜🔤\n";
    std::string s = "Hello";
    s += " C++ World!";
    cout << "Combined string: " << s << "\n";
    cout << "Length: " << s.length() << "\n";
    cout << string(50, '=') << "\n\n";
}

// Example 3: Advanced Substring Extraction
void stringExample3() {
    cout << "String Example 3: Advanced Substring Extraction ✂️📜\n";
    std::string text = "The quick brown fox jumps over the lazy dog";
    // Extract "brown fox"
    std::string sub = text.substr(10, 9);
    cout << "Original: " << text << "\n";
    cout << "Extracted substring: " << sub << "\n";
    cout << string(50, '=') << "\n\n";
}

// Example 4: Searching for Substrings and Characters
void stringExample4() {
    cout << "String Example 4: Searching in Strings 🔍📜\n";
    std::string sentence = "C++ is powerful and versatile!";
    // Find position of the word "powerful"
    size_t pos = sentence.find("powerful");
    if (pos != std::string::npos) {
        cout << "\"powerful\" found at index: " << pos << "\n";
    } else {
        cout << "\"powerful\" not found.\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 5: Using std::stringstream for Conversion and Parsing
void stringExample5() {
    cout << "String Example 5: Using std::stringstream for Conversion 🔄📜\n";
    // Imagine a string as a conveyor belt where items are parsed.
    std::string data = "123 456 789";
    std::stringstream ss(data); // Load the string into a stringstream  
    // Imagine a conveyor belt where items are parsed.
    
    int a, b, c;
    ss >> a >> b >> c;
    cout << "Parsed numbers: " << a << ", " << b << ", " << c << "\n";
    cout << string(50, '=') << "\n\n";
}

// Example 6: Transforming std::string Using STL Algorithms and Lambdas
void stringExample6() {
    cout << "String Example 6: Transforming std::string with Lambdas 🔄📜\n";
    std::string text = "Hello C++";
    // Convert to uppercase using lambda and transform
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return toupper(c);
    });
    cout << "Uppercase: " << text << "\n";
    cout << string(50, '=') << "\n\n";
}

// Example 7: Regular Expressions with std::regex
void stringExample7() {
    cout << "String Example 7: Using std::regex for Pattern Matching 📜🔍\n";
    std::string input = "User: john_doe, Email: john@example.com";
    std::regex emailPattern(R"((\w+@\w+\.\w+))");
    std::smatch matches;
    if (std::regex_search(input, matches, emailPattern)) {
        cout << "Found email: " << matches[0] << "\n";
    } else {
        cout << "No email found.\n";
    }
    cout << string(50, '=') << "\n\n";
}

// Example 8: String to Number and Number to String Conversions
void stringExample8() {
    cout << "String Example 8: Conversions between Strings and Numbers 🔄🔢\n";
    std::string numStr = "2025";
    int num = std::stoi(numStr);  // Convert string to integer
    cout << "Converted integer: " << num << "\n";

    // Number to string conversion using to_string
    std::string converted = std::to_string(num * 2);
    cout << "After doubling, as string: " << converted << "\n";
    cout << string(50, '=') << "\n\n";
}

// Example 9: Custom Function to Reverse a std::string
void stringExample9() {
    cout << "String Example 9: Custom String Reversal 🔄📜\n";
    std::string original = "Advanced";
    std::string reversed(original.rbegin(), original.rend());
    // Diagram: Original: A d v a n c e d  -> Reversed: d e c n a v d A
    cout << "Original: " << original << "\n";
    cout << "Reversed: " << reversed << "\n";
    cout << string(50, '=') << "\n\n";
}

// Example 10: Using Lambda Functions to Transform Each Character in a String
void stringExample10() {
    cout << "String Example 10: Lambda Function to Transform Each Character 📜➡️🔤\n";
    std::string sentence = "lambda magic!";
    // Transform: Shift each character by 1 (e.g., 'a'->'b')
    std::transform(sentence.begin(), sentence.end(), sentence.begin(), [](char c) -> char {
        // Only shift alphabetic characters
        if (isalpha(c)) {
            // Wrap-around for 'z' or 'Z'
            if ((c == 'z') || (c == 'Z')) return c - 25;
            return c + 1;
        }
        return c;
    });
    cout << "Transformed sentence: " << sentence << "\n";
    cout << string(50, '=') << "\n\n";
}

// ==========================
//            MAIN
// ==========================
int main() {
    cout << "\n=== ADVANCED C++ EXAMPLES: ARRAYS AND STRINGS ===\n\n";

    // Arrays Examples
    arrayExample1();
    arrayExample2();
    arrayExample3();
    arrayExample4();
    arrayExample5();
    arrayExample6();
    arrayExample7();
    arrayExample8();
    arrayExample9();
    arrayExample10();

    // Strings Examples
    stringExample1();
    stringExample2();
    stringExample3();
    stringExample4();
    stringExample5();
    stringExample6();
    stringExample7();
    stringExample8();
    stringExample9();
    stringExample10();

    return 0;
}
