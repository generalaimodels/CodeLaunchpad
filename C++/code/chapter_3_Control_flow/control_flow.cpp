/***********************************************************************
 * Chapter 3: Control Flow - Making Decisions and Repeating Actions 🚦🔄
 *
 * This file demonstrates:
 *   - Sequential Execution vs. Control Flow
 *   - Conditional Statements (if, else, else if, switch)
 *   - Loops (for, while, do-while, plus break and continue)
 *
 * Each section contains 5 examples to illustrate key concepts, along with
 * comments indicating potential pitfalls or common mistakes.
 ***********************************************************************/

#include <iostream>   // For std::cout, std::endl
#include <string>     // For std::string

// -----------------------------------------------------------------
// 3.1 Sequential Execution vs. Control Flow 🚦
// -----------------------------------------------------------------
void sequentialFlowDemo() {
    std::cout << "=== Sequential Flow Demo ===\n";
    
    // Example 1: Print a welcome message.
    std::cout << "Step 1: Welcome! 🚀\n";
    
    // Example 2: Print the next step.
    std::cout << "Step 2: Processing data...\n";
    
    // Example 3: Continue in sequence.
    std::cout << "Step 3: Data processed ✅\n";
    
    // Example 4: Print a message before decision-making.
    std::cout << "Step 4: Preparing for decision-making 🤔\n";
    
    // Example 5: Print final message.
    std::cout << "Step 5: Finished execution 🎯\n\n";
}

// -----------------------------------------------------------------
// 3.2 Conditional Statements - Making Choices 🤔🚦
// -----------------------------------------------------------------
void conditionalStatementsDemo() {
    std::cout << "=== Conditional Statements Demo ===\n";
    
    // Example 1: Simple if statement.
    int weather = 1; // 1 indicates raining, 0 indicates sunny.
    if (weather == 1) {  // Check condition.
        std::cout << "It's raining 🌧️. Take an umbrella ☂️.\n";
    }
    
    // Example 2: if-else statement.
    weather = 0;
    if (weather == 1) {
        std::cout << "It's raining 🌧️. Take an umbrella ☂️.\n";
    } else {
        std::cout << "It's sunny ☀️. No umbrella needed.\n";
    }
    
    // Example 3: else-if ladder for multiple conditions.
    int temperature = 30;  // Temperature in Celsius.
    if (temperature > 35) {
        std::cout << "It's very hot 🔥. Have some ice cream 🍦.\n";
    } else if (temperature > 25) {
        std::cout << "It's warm 😊. Enjoy lemonade 🍹.\n";
    } else {
        std::cout << "It's cool ❄️. Drink water 💧.\n";
    }
    
    // Example 4: Nested if statement.
    int age = 20;
    if (age >= 18) {
        std::cout << "Adult. ";
        if (age < 21) {
            std::cout << "But certain privileges may be restricted 🚫.\n";
        } else {
            std::cout << "Full privileges granted 👍.\n";
        }
    } else {
        std::cout << "Minor 🚼. Limited privileges.\n";
    }
    
    // Example 5: switch statement for discrete choices.
    char grade = 'B';
    switch (grade) {
        case 'A':
            std::cout << "Excellent! 🌟\n";
            break;
        case 'B':
            std::cout << "Good job! 👍\n";
            break;
        case 'C':
            std::cout << "Satisfactory. Keep improving.\n";
            break;
        case 'D':
            std::cout << "Needs improvement.\n";
            break;
        default:
            std::cout << "Grade not recognized ❓\n";
            break;
    }
    
    std::cout << "\n";
}

// -----------------------------------------------------------------
// 3.3 Loops - Repeating Actions 🔄
// -----------------------------------------------------------------
void loopsDemo() {
    std::cout << "=== Loops Demo ===\n";
    
    // Example 1: for loop - counting 1 to 5.
    std::cout << "For loop (Counting 1 to 5): ";
    for (int i = 1; i <= 5; ++i) { // Initialization, condition, increment.
        std::cout << i << " ";     // Common mistake: forgetting ++ can lead to infinite loops.
    }
    std::cout << "\n";
    
    // Example 2: while loop - counting 1 to 5.
    std::cout << "While loop (Counting 1 to 5): ";
    int count = 1;
    while (count <= 5) {  // Condition is checked before each iteration.
        std::cout << count << " ";
        ++count;        // Ensure counter is incremented; missing this is a common pitfall.
    }
    std::cout << "\n";
    
    // Example 3: do-while loop - counting 1 to 5.
    std::cout << "Do-while loop (Counting 1 to 5): ";
    count = 1;
    do {  // Code block executes at least once.
        std::cout << count << " ";
        ++count;
    } while (count <= 5);  // Condition checked after the loop body.
    std::cout << "\n";
    
    // Example 4: for loop with break - exit early.
    std::cout << "For loop with break (Stop when i == 4): ";
    for (int i = 1; i <= 5; ++i) {
        if (i == 4) {
            std::cout << "\nBreaking loop at i = " << i << " 🚦\n";
            break;  // Exits the loop immediately.
        }
        std::cout << i << " ";
    }
    
    // Example 5: for loop with continue - skip current iteration.
    std::cout << "For loop with continue (Skip when i == 3): ";
    for (int i = 1; i <= 5; ++i) {
        if (i == 3) {
            std::cout << "\nSkipping number " << i << " ⏭️\n";
            continue;  // Skips the rest of the loop's body for this iteration.
        }
        std::cout << i << " ";
    }
    
    std::cout << "\n\n";
}

// -----------------------------------------------------------------
// main() - Entry point for the program
// -----------------------------------------------------------------
int main() {
    // Each demo function builds upon the previous concepts.
    sequentialFlowDemo();
    conditionalStatementsDemo();
    loopsDemo();
    
    return 0;
}
