// Chapter 2: Arrays & Strings - Linear Data Structures 📦📜

// Include necessary headers
#include <iostream>   // For input/output streams
#include <string>     // For 'std::string'
#include <cstring>    // For C-style string functions like 'strlen', 'strcmp'
#include <algorithm>  // For 'std::sort', 'std::find'

// Entry point of the program
int main() {
    // ARRAYS 📦➡️📚

    // Introduction: Contiguous memory locations holding elements of the same type.
    // Like numbered lockers 🔢 in a school.

    // 1D Arrays: Linear sequence ➡️📦📦📦📦
    int numbers[5]; // Declaration of an integer array of size 5
    // Possible mistake: Accessing uninitialized elements ⚠️
    // int sum = numbers[0] + numbers[1]; // ⚠️ May contain garbage values

    // Initializing the array
    numbers[0] = 10;
    numbers[1] = 20;
    numbers[2] = 30;
    numbers[3] = 40;
    numbers[4] = 50;

    // Accessing elements (O(1)) ⚡️📚[index]
    std::cout << "First element: " << numbers[0] << std::endl; // Outputs 10

    // Possible mistake: Array index out of bounds 🚫
    // std::cout << numbers[5]; // ⚠️ Undefined behavior, index 5 is out of bounds

    // Insertion (Worst Case O(n)) 📚➡️📚➡️📚
    // Inserting 25 at index 2 (shifting elements to the right)
    int insertIndex = 2;
    int insertValue = 25;
    // Shift elements to the right
    for (int i = 4; i > insertIndex; --i) {
        numbers[i] = numbers[i - 1];
    }
    numbers[insertIndex] = insertValue; // Insert the new value

    // Current array: 10, 20, 25, 30, 40

    // Deletion (Worst Case O(n)) 📚⬅️📚⬅️📚
    // Deleting element at index 3 (shifting elements to the left)
    int deleteIndex = 3;
    // Shift elements to the left
    for (int i = deleteIndex; i < 4; ++i) {
        numbers[i] = numbers[i + 1];
    }
    // Optionally, set the last element to zero
    numbers[4] = 0;

    // Current array: 10, 20, 25, 40, 0

    // Searching - Linear Search (O(n)) 🔍📚
    int searchValue = 25;
    bool found = false;
    for (int i = 0; i < 5; ++i) {
        if (numbers[i] == searchValue) {
            found = true;
            std::cout << "Value " << searchValue << " found at index " << i << std::endl;
            break;
        }
    }
    if (!found) {
        std::cout << "Value " << searchValue << " not found." << std::endl;
    }

    // For Binary Search, the array must be sorted
    // Let's sort the array
    std::sort(numbers, numbers + 5);

    // Binary Search (O(log n)) 🔍📚
    int left = 0;
    int right = 4;
    found = false;
    while (left <= right) {
        int mid = left + (right - left) / 2; // Avoids overflow
        if (numbers[mid] == searchValue) {
            found = true;
            std::cout << "Value " << searchValue << " found at index " << mid << " (Binary Search)" << std::endl;
            break;
        } else if (numbers[mid] < searchValue) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    if (!found) {
        std::cout << "Value " << searchValue << " not found (Binary Search)." << std::endl;
    }

    // 2D Arrays (Matrices): Rows and Columns 🧮
    int matrix[3][3]; // Declaration of a 3x3 matrix
    // Initializing the matrix
    int value = 1;
    for (int i = 0; i < 3; ++i) {        // Rows
        for (int j = 0; j < 3; ++j) {    // Columns
            matrix[i][j] = value++;
        }
    }
    // Accessing elements
    std::cout << "Element at (1,1): " << matrix[1][1] << std::endl; // Outputs 5

    // Multi-dimensional Arrays 🏢🅿️
    // 3D Array example
    int threeD[2][2][2]; // 2x2x2 cube
    // Initializing the 3D array
    value = 1;
    for (int i = 0; i < 2; ++i) {            // First dimension
        for (int j = 0; j < 2; ++j) {        // Second dimension
            for (int k = 0; k < 2; ++k) {    // Third dimension
                threeD[i][j][k] = value++;
            }
        }
    }
    // Accessing elements
    std::cout << "Element at (1,1,1): " << threeD[1][1][1] << std::endl; // Outputs 8

    // STRINGS 📖🔡

    // Introduction: Sequence of characters. Like words and sentences.

    // String as Array of Characters
    char cString[] = "Hello"; // C-style string (null-terminated)
    // Possible mistake: Not leaving room for the null terminator
    // char wrongCString[5] = "Hello"; // ⚠️ 'Hello' + '\0' requires 6 characters

    // Accessing characters
    std::cout << "First character: " << cString[0] << std::endl; // Outputs 'H'

    // Modifying characters
    cString[0] = 'Y'; // Now cString is "Yello"
    std::cout << "Modified string: " << cString << std::endl;

    // std::string usage
    std::string cppString = "World";
    // Concatenation 🔗➕🔗
    std::string greeting = std::string(cString) + " " + cppString + "!";
    std::cout << "Greeting: " << greeting << std::endl; // Outputs "Yello World!"

    // Substring ✂️🔗
    std::string sub = greeting.substr(0, 5); // Extracts "Yello"
    std::cout << "Substring: " << sub << std::endl;

    // Possible mistake: Out of range substring
    // std::string invalidSub = greeting.substr(50, 5); // ⚠️ May throw an exception

    // Comparison ⚖️🔗
    if (sub == "Yello") {
        std::cout << "Substrings are equal." << std::endl;
    } else {
        std::cout << "Substrings are not equal." << std::endl;
    }

    // Searching (substring search) 🔍🔗
    size_t foundPos = greeting.find("World");
    if (foundPos != std::string::npos) {
        std::cout << "'World' found at position " << foundPos << std::endl;
    } else {
        std::cout << "'World' not found." << std::endl;
    }

    // STRING MANIPULATION ALGORITHMS 🛠️

    // Palindrome Check 🚗↔️🚗
    std::string palindromeCandidate = "racecar";
    bool isPalindrome = true;
    int len = palindromeCandidate.length();
    for (int i = 0; i < len / 2; ++i) {
        if (palindromeCandidate[i] != palindromeCandidate[len - i - 1]) {
            isPalindrome = false;
            break;
        }
    }
    if (isPalindrome) {
        std::cout << palindromeCandidate << " is a palindrome." << std::endl;
    } else {
        std::cout << palindromeCandidate << " is not a palindrome." << std::endl;
    }

    // String Reversal ↩️🔗
    std::string originalString = "hello";
    std::string reversedString = originalString; // Copy original string
    std::reverse(reversedString.begin(), reversedString.end()); // Reverse in-place
    std::cout << "Original: " << originalString << ", Reversed: " << reversedString << std::endl;

    // Anagram Check 🔤🔄🔤
    std::string str1 = "listen";
    std::string str2 = "silent";
    // Remove spaces and convert to lowercase if necessary
    // Simple check: Sort both strings and compare
    std::string sortedStr1 = str1;
    std::string sortedStr2 = str2;
    std::sort(sortedStr1.begin(), sortedStr1.end());
    std::sort(sortedStr2.begin(), sortedStr2.end());
    if (sortedStr1 == sortedStr2) {
        std::cout << str1 << " and " << str2 << " are anagrams." << std::endl;
    } else {
        std::cout << str1 << " and " << str2 << " are not anagrams." << std::endl;
    }

    // Possible mistakes:
    // - Not handling case sensitivity (e.g., 'A' vs 'a')
    // - Not removing spaces and punctuation when necessary

    return 0; // End of program
}