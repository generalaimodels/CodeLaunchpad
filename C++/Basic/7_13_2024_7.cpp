// 4.8.4. Arbitrary Argument Lists
#include <iostream>
#include <cstdarg>
#include <vector>
#include <algorithm>
// #include <concepts>
void printNumbers(int count, ...) {
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; i++) {
        int num = va_arg(args, int);
        std::cout << num << " ";
    }

    va_end(args);
    std::cout << std::endl;
}



// int main() {
//     printNumbers(3, 10, 20, 30); // Output: 10 20 30
//     printNumbers(5, 1, 2, 3, 4, 5); // Output: 1 2 3 4 5
//     return 0;
// }

// 4.8.5. Unpacking Argument Lists


template <typename T>
void printArgs(T value) {
    std::cout << value << std::endl;
}

template <typename T, typename... Args>
void printArgs(T value, Args... args) {
    std::cout << value << " ";
    printArgs(args...);
}

// int main() {
//     printArgs(1, 2, 3.14, "hello"); // Output: 1 2 3.14 hello
//     return 0;
// }

// 4.8.6. Lambda Expressions

int Lambda_Main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // Lambda expression to multiply each element by 2
    std::transform(numbers.begin(), numbers.end(), numbers.begin(), [](int x) { return x * 2; });

    // Printing the transformed vector
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl; // Output: 2 4 6 8 10

    return 0;
}

// 4.8.7. Documentation Strings
/**
 * @brief Calculates the sum of two numbers.
 *
 * @param a The first number to add.
 * @param b The second number to add.
 * @return The sum of a and b.
 */
int add(int a, int b) {
    return a + b;
}

int Documentation_String_Main() {
    int result = add(3, 4);
    std::cout << "Result: " << result << std::endl; // Output: Result: 7
    return 0;
}

// 4.8.8. Function Annotations


template <typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
};

Addable auto add(Addable auto a, Addable auto b) {
    return a + b;
}

int Add_Test_Main() {
    int result1 = add(3, 4); // OK, int is Addable
    std::cout << "Result 1: " << result1 << std::endl; // Output: Result 1: 7

    std::string result2 = add("Hello", ", World!"); // OK, std::string is Addable
    std::cout << "Result 2: " << result2 << std::endl; // Output: Result 2: Hello, World!

    // Error: double and int are not Addable
    // double result3 = add(3.14, 2);

    return 0;
}


// 4.9. Intermezzo: Coding Style
// Example of good coding style in C++

// Constant naming convention: UPPER_CASE_WITH_UNDERSCORES
const int MAX_SCORE = 100;

// Function naming convention: camelCase
int calculateScore(const std::vector<int>& scores) {
    int totalScore = 0;
    for (int score : scores) {
        if (score > MAX_SCORE) {
            // Error handling: throw an exception for invalid scores
            throw std::runtime_error("Score cannot exceed " + std::to_string(MAX_SCORE));
        }
        totalScore += score;
    }
    return totalScore;
}

int Good_Intermezzo_Main() {
    std::vector<int> scores = {85, 92, 78, 105};  // Invalid score: 105

    try {
        int totalScore = calculateScore(scores);
        std::cout << "Total Score: " << totalScore << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}