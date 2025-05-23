/*****************************************************
 * Chapter 3: Control Flow - Making Decisions and Repeating Actions 🚦🔄
 * Advanced C++ Examples for Developers (Medium/High-Level)
 * 
 * This file demonstrates:
 *   - Sequential Execution vs. Control Flow (➡️ vs. 🚦/🔄)
 *   - Conditional Statements (if, else, else if, switch)
 *   - Loops (for, while, do-while, with break/continue)
 * 
 * Each section contains 5 examples (our signature).
 *****************************************************/

#include <iostream> // For input/output operations
#include <string>   // For string operations
using namespace std; // Using standard namespace

int main() {
    // ======================================================
    // Section 1: Sequential Execution vs. Control Flow 🚦
    // ======================================================
    cout << "=== Section 1: Sequential Execution vs. Control Flow 🚦 ===\n\n";
    
    // Example 1: Sequential execution – code runs line by line.
    cout << "Step 1: This is executed first.\n";
    cout << "Step 2: This is executed second.\n";
    cout << "Step 3: This is executed third.\n";
    
    // Example 2: Sequential operations using variables.
    int a = 10;
    int b = 20;
    int sum = a + b; // Executes sequentially: declare a, b, then calculate sum.
    cout << "Sequential Sum (a + b): " << sum << "\n";

    // Example 3: Sequential function calls (simulate by separate code blocks).
    cout << "Function 1 executed.\n"; // Imagine a function here.
    cout << "Function 2 executed.\n"; // Then another function.
    
    // Example 4: Sequential assignment.
    int x = 5;
    x = x + 2; // Update x after initial assignment.
    cout << "Updated x: " << x << "\n";

    // Example 5: Sequential I/O operations.
    cout << "Enter a number: ";
    int userInput = 0;
    cin >> userInput; // Waits for user input before proceeding.
    cout << "You entered: " << userInput << "\n\n";

    // ======================================================
    // Section 2: Conditional Statements - Making Choices 🤔🚦
    // ======================================================
    cout << "=== Section 2: Conditional Statements - Making Choices 🤔🚦 ===\n\n";
    
    // Example 1: Basic if statement.
    if (userInput > 0) { // Executes only if condition is true.
        cout << "The number is positive.\n";
    }

    // Example 2: if-else statement.
    if (userInput % 2 == 0) {
        cout << "The number is even.\n";
    } else {
        cout << "The number is odd.\n";
    }

    // Example 3: if-else if-else chain.
    if (userInput > 100) {
        cout << "The number is large (>100).\n";
    } else if (userInput >= 50) {
        cout << "The number is medium (50-100).\n";
    } else {
        cout << "The number is small (<50).\n";
    }

    // Example 4: Nested if statements.
    if (userInput != 0) {
        if (userInput > 0) {
            cout << "Nested: The number is positive.\n";
        } else {
            cout << "Nested: The number is negative.\n";
        }
    } else {
        cout << "Nested: The number is zero.\n";
    }

    // Example 5: Ternary operator for simple condition.
    string result = (userInput > 0) ? "Positive" : "Non-positive";
    cout << "Ternary Check: " << result << "\n\n";

    // ======================================================
    // Section 3: Switch Statement - Efficient Multi-Way Branching 🚂
    // ======================================================
    cout << "=== Section 3: Switch Statement - Efficient Multi-Way Branching 🚂 ===\n\n";
    
    // Example 1: Basic switch statement.
    int option = userInput % 3; // Value: 0, 1, or 2.
    switch (option) {
        case 0:
            cout << "Switch Case 0: Option is zero.\n";
            break;
        case 1:
            cout << "Switch Case 1: Option is one.\n";
            break;
        case 2:
            cout << "Switch Case 2: Option is two.\n";
            break;
        default:
            cout << "Switch Default: Option is unknown.\n";
    }

    // Example 2: Switch with fall-through (intentional fall-through).
    int grade = userInput; // Assume userInput represents a grade.
    cout << "\nGrade Evaluation: ";
    switch (grade / 10) { // Check tens digit.
        case 10:
        case 9:
            cout << "Excellent\n";
            break;
        case 8:
            cout << "Very Good\n";
            break;
        case 7:
            cout << "Good\n";
            break;
        case 6:
            cout << "Satisfactory\n";
            break;
        default:
            cout << "Needs Improvement\n";
    }

    // Example 3: Switch statement with char.
    char choice = 'A';
    switch (choice) {
        case 'A':
            cout << "Choice A selected.\n";
            break;
        case 'B':
            cout << "Choice B selected.\n";
            break;
        case 'C':
            cout << "Choice C selected.\n";
            break;
        default:
            cout << "Invalid choice.\n";
    }

    // Example 4: Using switch with enum (advanced usage).
    enum Direction { North, East, South, West };
    Direction dir = East;
    switch (dir) {
        case North:
            cout << "Heading North.\n";
            break;
        case East:
            cout << "Heading East.\n";
            break;
        case South:
            cout << "Heading South.\n";
            break;
        case West:
            cout << "Heading West.\n";
            break;
        default:
            cout << "Unknown direction.\n";
    }

    // Example 5: Warning - switch cannot use ranges.
    // Developers must use if-else for range checks.
    int score = userInput;
    cout << "\nScore Evaluation (if-else required for ranges): ";
    if (score >= 90)
        cout << "Grade: A\n";
    else if (score >= 80)
        cout << "Grade: B\n";
    else if (score >= 70)
        cout << "Grade: C\n";
    else if (score >= 60)
        cout << "Grade: D\n";
    else
        cout << "Grade: F\n";

    // ======================================================
    // Section 4: Loops - Repeating Actions 🔄
    // ======================================================
    cout << "\n=== Section 4: Loops - Repeating Actions 🔄 ===\n\n";
    
    // ----- For Loop Examples -----
    cout << "--- For Loop Examples ---\n";
    // Example 1: Basic for loop.
    cout << "For Loop (0 to 4): ";
    for (int i = 0; i < 5; ++i) { // Initialization, condition, increment.
        cout << i << " ";
    }
    cout << "\n";

    // Example 2: For loop with summation.
    int total = 0;
    for (int i = 1; i <= 5; ++i) {
        total += i; // Sum 1+2+3+4+5.
    }
    cout << "Sum of 1 to 5: " << total << "\n";

    // Example 3: For loop with decrement.
    cout << "Countdown: ";
    for (int i = 5; i > 0; --i) {
        cout << i << " ";
    }
    cout << "\n";

    // Example 4: For loop with break (exit early).
    cout << "For Loop with break: ";
    for (int i = 0; i < 10; ++i) {
        if (i == 3) {
            break; // Exit loop when i equals 3.
        }
        cout << i << " ";
    }
    cout << "\n";

    // Example 5: For loop with continue (skip iteration).
    cout << "For Loop with continue (skip 2): ";
    for (int i = 0; i < 5; ++i) {
        if (i == 2) {
            continue; // Skip the rest of the loop when i equals 2.
        }
        cout << i << " ";
    }
    cout << "\n\n";

    // ----- While Loop Examples -----
    cout << "--- While Loop Examples ---\n";
    // Example 1: Basic while loop.
    int count = 0;
    cout << "While Loop (count to 3): ";
    while (count < 3) {
        cout << count << " ";
        count++; // Increment count to avoid infinite loop.
    }
    cout << "\n";

    // Example 2: While loop with condition check.
    int val = userInput; // Use userInput for demonstration.
    cout << "While Loop (reduce value until 0): ";
    while (val > 0) {
        cout << val << " ";
        val -= 10; // Reduce by 10; caution: ensure termination.
    }
    cout << "\n";

    // Example 3: While loop with potential infinite loop safeguard.
    int safeCount = 0;
    while (safeCount < 5) { // Use a simple counter to avoid infinite loops.
        cout << safeCount << " ";
        safeCount++;
    }
    cout << "\n";

    // Example 4: While loop with break.
    int number = 0;
    cout << "While Loop with break: ";
    while (true) {
        if (number >= 3) {
            break; // Exit loop when number reaches 3.
        }
        cout << number << " ";
        number++;
    }
    cout << "\n";

    // Example 5: While loop with continue.
    int cnt = 0;
    cout << "While Loop with continue (skip 1): ";
    while (cnt < 5) {
        cnt++;
        if (cnt == 1) {
            continue; // Skip printing when cnt equals 1.
        }
        cout << cnt << " ";
    }
    cout << "\n\n";

    // ----- Do-While Loop Examples -----
    cout << "--- Do-While Loop Examples ---\n";
    // Example 1: Basic do-while loop.
    int num = 0;
    cout << "Do-While Loop (execute at least once): ";
    do {
        cout << num << " ";
        num++;
    } while (num < 3);
    cout << "\n";

    // Example 2: Do-while loop with user prompt.
    int repeatCount = 0;
    cout << "Do-While Loop with condition: ";
    do {
        cout << "Iteration " << repeatCount << " ";
        repeatCount++;
    } while (repeatCount < 3);
    cout << "\n";

    // Example 3: Do-while loop ensuring execution.
    int execOnce = 10;
    do {
        cout << "This prints even if condition is false at first. ";
    } while (false); // Executes once.
    cout << "\n";

    // Example 4: Do-while loop with input simulation.
    int simulate = userInput;
    cout << "Do-While (simulate reducing input): ";
    do {
        cout << simulate << " ";
        simulate /= 2; // Caution: integer division.
    } while (simulate > 0);
    cout << "\n";

    // Example 5: Do-while with error-check (avoid infinite loop).
    int attempt = 0;
    cout << "Do-While with safeguard: ";
    do {
        cout << attempt << " ";
        attempt++;
        if (attempt > 100) { // Extremely cautious: break if loop runs too long.
            cout << "\nBreaking to avoid infinite loop.\n";
            break;
        }
    } while (attempt < 5);
    cout << "\n";

    // ======================================================
    // End of Control Flow Examples
    // ======================================================
    return 0;
}
