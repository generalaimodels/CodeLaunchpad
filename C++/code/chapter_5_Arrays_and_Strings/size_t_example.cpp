// size_t_example.cpp
// Demonstrates the use of size_t in C++ 📏🔢
// Note: size_t is an unsigned integer type, ideal for representing sizes & indices.

#include <iostream>   // For std::cout
#include <cstddef>    // For size_t (alternatively, it may be included via other headers)
#include <vector>     // For std::vector

using namespace std;

int main() {
    // Example 1: Using size_t with a static array 📦
    const size_t arraySize = 5;  // size_t ensures a non-negative size
    int numbers[arraySize] = { 10, 20, 30, 40, 50 };

    // Iterate over the array using size_t for the index
    for (size_t i = 0; i < arraySize; ++i) {
        cout << "Element at index " << i << ": " << numbers[i] << "\n";
    }

    // Example 2: Using size_t with std::vector 📜
    vector<string> names = {"Alice", "Bob", "Charlie", "Dana"};
    
    // Using size_t in loop for container's size (returned as size_t)
    for (size_t i = 0; i < names.size(); ++i) {
        cout << "Name[" << i << "]: " << names[i] << "\n";
    }

    // ⚠️ Common Pitfall: Mixing signed and unsigned types
    int signedIndex = -1;
    // Uncommenting the following line may trigger compiler warnings about comparison:
    // if (signedIndex < names.size()) { /* ... */ }

    // Best practice: Ensure that loop indices using size_t are compared with size_t values.
    // Use explicit casts if necessary, but always be cautious of negative values.

    // Summary:
    // - size_t is defined to be large enough to represent the size of any object.
    // - Commonly used for array indices, sizes, and loop counters with containers.
    // - Always be mindful of mixing with signed integers to avoid subtle bugs.

    return 0;
}
