// main.cpp
// Chapter 5: Arrays and Strings - Working with Collections of Data 📦📚
// Each example builds in complexity. Follow the comments for detailed explanations.
// Note: Compile with a C++11 (or later) compliant compiler.

#include <iostream>
#include <string>    // For std::string operations
#include <stdexcept> // For std::out_of_range exception (if needed)

using namespace std;

// Example 1: Basic One-Dimensional Array Operations 📦🔢
// - Declaring, initializing, accessing, and handling bounds.
void example1_basicArray() {
    cout << "\n=== Example 1: One-Dimensional Arrays 📦🔢 ===\n";

    // Declare and initialize an array of 5 integers.
    int numbers[5] = {10, 20, 30, 40, 50}; // 📦 Array of ints

    // Accessing elements using indices (starting from 0)
    for (int i = 0; i < 5; ++i) {
        cout << "Element at index " << i << " : " << numbers[i] << "\n";
    }

    // ⚠️ Potential error: Accessing out-of-bound index.
    // Uncomment the following lines to see what happens (may lead to undefined behavior):
    /*
    cout << "Accessing element at index 5 (out-of-bound): " << numbers[5] << "\n"; // Undefined behavior!
    */
}

// Example 2: Multi-Dimensional Arrays (2D Arrays) - Grids or Tables 🔢📦
// - Declaring a 2D array and iterating through rows and columns.
void example2_multiDimensionalArray() {
    cout << "\n=== Example 2: Two-Dimensional Arrays (Grid) 🔢📦 ===\n";

    // Declare and initialize a 3x3 2D array.
    int grid[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Nested loops to iterate through the grid.
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            cout << grid[row][col] << " ";
        }
        cout << "\n";
    }
    // ⚠️ Note: Ensure indices remain within bounds (0-2 for each dimension).
}

// Example 3: C-Style Strings 📜🔤
// - Working with character arrays and the null terminator '\0'
void example3_cStyleStrings() {
    cout << "\n=== Example 3: C-Style Strings 📜🔤 ===\n";

    // Declare and initialize a C-style string (char array).
    char greeting[20] = "Hello, World!"; // Ensure size is sufficient for the string and null char.
    
    // Output the C-style string.
    cout << "Greeting: " << greeting << "\n";

    // ⚠️ Pitfall: Forgetting the null terminator may lead to undefined behavior.
    // Example: char wrongStr[5] = {'H', 'e', 'l', 'l', 'o'}; // No '\0' added!
}

// Example 4: std::string - Modern and Safer String Operations 🏷️📜🔤
// - Using std::string for concatenation, substring search, and comparisons.
void example4_stdString() {
    cout << "\n=== Example 4: std::string Operations 🏷️📜🔤 ===\n";

    // Declare std::string variables.
    string str1 = "Hello";
    string str2 = ", C++ World!";
    
    // Concatenation (joining strings)
    string combined = str1 + str2;
    cout << "Combined String: " << combined << "\n";

    // Finding a substring.
    size_t pos = combined.find("C++");
    if (pos != string::npos) {
        cout << "Substring 'C++' found at index: " << pos << "\n";
    } else {
        cout << "Substring not found!\n";
    }

    // Comparison of strings.
    if (str1 == "Hello") {
        cout << "str1 matches 'Hello' exactly.\n";
    }
    
    // ⚠️ Common mistakes: Using C-style functions (e.g., strcmp) on std::string instead of built-in operators.
}

// Example 5: Arrays of std::string - Combining Collections of Data Types 📦🏷️
// - Using an array to store multiple strings and iterating through them.
void example5_arrayOfStrings() {
    cout << "\n=== Example 5: Arrays of std::string 📦🏷️ ===\n";

    // Declare and initialize an array of std::string (fixed size).
    string fruits[5] = {"Apple", "Banana", "Cherry", "Date", "Elderberry"};

    // Iterating through the array using range-based for loop (C++11 feature)
    for (const auto& fruit : fruits) {
        cout << fruit << "\n";
    }

    // ⚠️ Exception handling: When working with indices, always ensure you remain in valid range.
    try {
        int index = 5; // Out-of-bound index for an array of size 5 (valid indices: 0-4)
        if (index < 0 || index >= 5) {
            throw out_of_range("Index out of range for 'fruits' array.");
        }
        cout << fruits[index] << "\n"; // This line would never be reached.
    } catch (const out_of_range& e) {
        cerr << "Exception caught: " << e.what() << "\n";
    }
}

// Main function - Executes all examples sequentially.
int main() {
    cout << "Chapter 5: Arrays and Strings - Demonstrations 📦📚\n";

    example1_basicArray();
    example2_multiDimensionalArray();
    example3_cStyleStrings();
    example4_stdString();
    example5_arrayOfStrings();

    cout << "\nAll examples executed successfully.\n";
    return 0;
}
