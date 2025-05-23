// Chapter 1: Foundations - Laying the Groundwork 🧱

// Basic Building Blocks of Computation & Measurement 🧱📏
// Understanding materials (data types), tools (operators), and measurements (complexity) before building a skyscraper.

// Include necessary headers for input/output, strings, and vectors.
#include <iostream> // For standard input/output streams.
#include <string>   // For using the 'std::string' data type.
#include <vector>   // For using the 'std::vector' container.

// Function declarations (prototypes).
int add(int a, int b); // Function to add two integers.
void generateSubsets(std::vector<int>& set, std::vector<int>& subset, int index); // Function to generate all subsets of a set.

int main() {
    // DATA TYPES: The raw materials – like bricks, wood, steel. 🧱🪵🔩

    // Integer data type - 'int' for whole numbers.
    int age = 30; // Correct usage.
    // int age = 30.5; // ❌ Mistake: Assigning a float to an int (data loss).

    // Floating-point data types - 'float' and 'double' for decimal numbers.
    float temperature = 98.6f; // 'f' suffix indicates a float literal.
    double preciseValue = 3.1415926535; // 'double' has higher precision than 'float'.

    // Character data type - 'char' for single characters.
    char initial = 'A'; // Must use single quotes for characters.
    // char initial = "A"; // ❌ Mistake: Double quotes are for strings, not single characters.

    // Boolean data type - 'bool' for true/false values.
    bool isRaining = false; // Represents logical true or false.

    // String data type - 'std::string' for text strings.
    std::string greeting = "Hello, World!";
    // Remember to include <string> header when using 'std::string'.

    // VARIABLES & MEMORY: Containers holding data. Like labeled boxes 📦 in a warehouse.

    int apples = 5; // Variable 'apples' holds the integer value 5.

    // OPERATORS: Tools to manipulate data – like hammers 🔨, saws 🪚, drills 🪛.

    // Arithmetic Operators.
    int sum = apples + 2;        // Addition (+)
    int difference = apples - 2; // Subtraction (-)
    int product = apples * 2;    // Multiplication (*)
    float quotient = apples / 2.0f; // Division (/) - Note use of '2.0f' to get a float result.
    int remainder = apples % 2;  // Modulus (%) - Remainder of division.
    // int errorDivideByZero = apples / 0; // ❌ Mistake: Division by zero causes runtime error.

    // Logical Operators.
    bool isSunny = true;
    bool needUmbrella = isRaining && !isSunny; // Logical AND (&&), Logical NOT (!)
    // '&&' returns true if both operands are true.
    // '!' inverts the boolean value.
    // ❌ Mistake: Using bitwise '&' instead of logical '&&'.

    // Comparison Operators.
    bool isAdult = age >= 18; // Greater than or equal to (>=).
    // Other operators: == (equal to), != (not equal to), <, >, <=, >=.

    // CONTROL FLOW:

    // Sequential: Step-by-step execution. ➡️➡️➡️
    std::cout << "You have " << apples << " apples." << std::endl;
    std::cout << "Temperature is " << temperature << " degrees." << std::endl;

    // Conditional (if/else, switch): Decision making based on conditions. 🚦

    // if/else statement.
    if (isAdult) {
        std::cout << "You are an adult." << std::endl;
    } else {
        std::cout << "You are not an adult." << std::endl;
    }
    // ❌ Common mistake: Missing braces '{}' for multi-statement blocks.

    // switch-case statement.
    switch (initial) {
        case 'A':
            std::cout << "Excellent!" << std::endl;
            break; // 'break' prevents fall-through to the next case.
        case 'B':
            std::cout << "Good job!" << std::endl;
            break;
        default:
            std::cout << "Keep trying!" << std::endl;
            break;
    }
    // ❌ Mistake: Forgetting 'break;' leads to unintended fall-through.

    // Loops (for, while): Repetitive execution. 🔄🔄🔄

    // For loop - counts from 0 to 4.
    for (int i = 0; i < apples; ++i) {
        std::cout << "Apple number: " << i + 1 << std::endl;
    }
    // ❌ Common mistakes:
    // - Off-by-one errors (e.g., using '<=' instead of '<').
    // - Infinite loops (e.g., forgetting to increment 'i').

    // While loop - counts down from 'apples' to 1.
    int count = apples;
    while (count > 0) {
        std::cout << "Countdown: " << count << std::endl;
        --count; // Decrement 'count' by 1.
    }
    // ❌ Mistake: Infinite loop if 'count' is not decremented.

    // FUNCTIONS (Procedures/Methods): Reusable blocks of code. 🏗️

    int total = add(5, 10); // Call the 'add' function.
    std::cout << "Total: " << total << std::endl;
    // ❌ Mistake: Calling a function before it's declared or defined.

    // TIME COMPLEXITY - Big O Notation: Measuring Algorithm Efficiency in terms of time. ⏱️📈

    // O(1) - Constant Time: Instant access. ⚡️
    int numbers[] = {10, 20, 30, 40, 50};
    int first = numbers[0]; // Accessing an array element by index.
    std::cout << "First number: " << first << std::endl;

    // O(n) - Linear Time: Proportional to input size. 🚶‍♂️
    // Linear search: Checking each item one by one.
    int target = 30;
    bool found = false;
    for (int i = 0; i < 5; ++i) {
        if (numbers[i] == target) {
            found = true;
            break; // Exit loop when target is found.
        }
    }
    if (found) {
        std::cout << "Target found!" << std::endl;
    } else {
        std::cout << "Target not found." << std::endl;
    }

    // O(log n) - Logarithmic Time: Halving the search space. 🌲
    // Binary search (on a sorted array).
    int left = 0;
    int right = 4; // Last index in the array.
    found = false;
    while (left <= right) {
        int mid = (left + right) / 2; // Middle index.
        if (numbers[mid] == target) {
            found = true;
            break;
        } else if (numbers[mid] < target) {
            left = mid + 1; // Search in the right half.
        } else {
            right = mid - 1; // Search in the left half.
        }
        // ❌ Common mistake: Integer division errors; ensure correct calculation of 'mid'.
    }
    if (found) {
        std::cout << "Target found using binary search!" << std::endl;
    } else {
        std::cout << "Target not found using binary search." << std::endl;
    }

    // O(n log n) - Linearithmic Time: Efficient sorting. 🚀
    // Example: Sorting algorithms like Merge Sort, Quick Sort.
    // Using built-in sort function (requires <algorithm> header).
    // std::sort(numbers, numbers + 5); // ❌ Mistake: Needs #include <algorithm>.

    // O(n^2) - Quadratic Time: Nested loops. 🐌
    // Example: Printing all pairs of elements.
    for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
            std::cout << "Pair: (" << numbers[i] << ", " << numbers[j] << ")" << std::endl;
        }
    }
    // ❌ Mistake: High time complexity for large datasets due to nested loops.

    // O(2^n) - Exponential Time: Brute force, exploring all possibilities. 🤯
    // Example: Generating all subsets (the power set) of a set.
    std::vector<int> set = {1, 2, 3}; // Small set.
    std::vector<int> subset;          // Empty subset to start with.
    std::cout << "All subsets of the set {1, 2, 3}:" << std::endl;
    generateSubsets(set, subset, 0);  // Call the function.
    // ⚠️ Note: For larger sets, the number of subsets grows exponentially (2^n).
    // ❌ Mistake: Stack overflow due to deep recursion when 'n' is large.

    // SPACE COMPLEXITY: Measuring memory usage. Like the amount of land 🏞️ required for the building. 💾📈

    // O(1) Space - Using a constant amount of memory.
    int fixedVariable = 42;

    // O(n) Space - Using memory proportional to input size.
    int* dynamicArray = new int[apples]; // Dynamically allocated array.
    // Remember to release allocated memory to prevent memory leaks.
    delete[] dynamicArray;
    // ❌ Mistake: Forgetting to 'delete[]' leads to memory leaks.
    // ❌ Mistake: Accessing memory out of bounds (buffer overflow).

    return 0; // End of the program.
}

// Function definition for 'add' - adds two integers.
int add(int a, int b) {
    return a + b; // Returns the sum of 'a' and 'b'.
    // ❌ Possible mistake: Incorrect calculation (e.g., returning 'a - b' instead).
}

// Function to generate all subsets (the power set) of a set - demonstrates O(2^n) time complexity.
void generateSubsets(std::vector<int>& set, std::vector<int>& subset, int index) {
    if (index == set.size()) {
        // Base case: All elements have been considered.
        std::cout << "{ ";
        for (int num : subset) {
            std::cout << num << " ";
        }
        std::cout << "}" << std::endl;
    } else {
        // Recursive case: Include the current element and recurse.
        subset.push_back(set[index]);           // Include element at 'index'.
        generateSubsets(set, subset, index + 1); // Recurse with next index.

        // Exclude the current element and recurse.
        subset.pop_back();                      // Remove last element.
        generateSubsets(set, subset, index + 1); // Recurse with next index.
    }
    // ❌ Mistake: Not handling the base case correctly can lead to infinite recursion.
}