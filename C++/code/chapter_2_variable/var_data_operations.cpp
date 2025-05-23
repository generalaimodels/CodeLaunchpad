/*****************************************************
 * Chapter 2: Variables, Data Types, and Operators 🧱🧩
 * Advanced C++ Examples for Medium/High-Level Coders
 * 
 * This file demonstrates:
 *   - Variables: Containers for Information 📦🏷️
 *   - Data Types: Types of Boxes (int, float, double, char, bool) 📦👟, 📦🧊, 📦📜
 *   - Operators: Performing Actions on Data ⚙️🛠️
 * 
 * Each section contains 5 examples (our signature) with detailed comments.
 *****************************************************/

#include <iostream> // For input/output operations
#include <string>  // For string data type
using namespace std; // Using the standard namespace

int main() {
    // ======================================================
    // Section 1: Variables - Containers for Information 📦🏷️
    // ======================================================
    cout << "=== Section 1: Variables - Containers for Information 📦🏷️ ===\n\n";
    
    // Example 1: Declare an integer variable without initialization.
    // 👉 Warning: Uninitialized variables may contain garbage values.
    int a ; 
    // cout << "Value of a (uninitialized): " << a << "\n"; // Avoid printing uninitialized variables!

    // Example 2: Declare and initialize an integer variable.
    int apples = 5; // 🍎 'apples' is initialized to 5.
    cout << "Apples: " << apples << "\n";

    // Example 3: Declare multiple variables in one statement.
    int oranges = 3, bananas = 7; // 🍊🍌 Declaring two variables at once.
    cout << "Oranges: " << oranges << ", Bananas: " << bananas << "\n";

    // Example 4: Use 'auto' for type inference (C++11 feature).
    auto grapes = 12; // 🍇 'auto' deduces that grapes is of type int.
    cout << "Grapes: " << grapes << "\n";

    // Example 5: Use descriptive variable names for clarity.
    int totalFruits = apples + oranges + bananas + grapes;
    cout << "Total Fruits: " << totalFruits << "\n\n";

    // ======================================================
    // Section 2: Data Types - Types of Boxes 📦👟, 📦🧊, 📦📜
    // ======================================================
    cout << "=== Section 2: Data Types - Types of Boxes 📦👟, 📦🧊, 📦📜 ===\n\n";
    
    // Example 1: int - Whole numbers (like counting apples 🍎)
    int count = 100;
    cout << "Count (int): " << count << "\n";

    // Example 2: float - Floating-point numbers with less precision.
    float temperature = 36.6f; // 'f' suffix denotes a float literal 🌡️
    cout << "Temperature (float): " << temperature << "\n";

    // Example 3: double - Floating-point numbers with higher precision.
    double preciseValue = 3.1415926535; // More precise value of π 📏
    cout << "Pi (double): " << preciseValue << "\n";

    // Example 4: char - Single characters.
    char letter = 'A'; // A single character from the alphabet 🔤
    cout << "Letter (char): " << letter << "\n";

    // Example 5: bool - Boolean values (true/false).
    bool isSunny = true; // Represents a light switch: ON (true) or OFF (false) 💡
    cout << "Is it sunny? (bool): " << (isSunny ? "true" : "false") << "\n\n";

    // ======================================================
    // Section 3: Operators - Performing Actions on Data ⚙️🛠️
    // ======================================================
    cout << "=== Section 3: Operators - Performing Actions on Data ⚙️🛠️ ===\n\n";

    // ----- Arithmetic Operators -----
    cout << "--- Arithmetic Operators ---\n";
    // Example 1: Addition (+)
    int sum = apples + oranges; // Adding two variables.
    cout << "Sum (apples + oranges): " << sum << "\n";

    // Example 2: Subtraction (-)
    int difference = bananas - grapes; // Subtracting grapes from bananas.
    cout << "Difference (bananas - grapes): " << difference << "\n";

    // Example 3: Multiplication (*)
    int product = apples * 2; // Doubling the number of apples.
    cout << "Product (apples * 2): " << product << "\n";

    // Example 4: Division (/)
    // 👉 Caution: Division by zero is undefined. Always check the denominator.
    int division = (oranges != 0) ? bananas / oranges : 0;
    cout << "Division (bananas / oranges): " << division << "\n";

    // Example 5: Modulus (%) - Remainder of division.
    int remainder = bananas % 3; // Remainder when bananas is divided by 3.
    cout << "Remainder (bananas % 3): " << remainder << "\n\n";

    // ----- Assignment Operators -----
    cout << "--- Assignment Operators ---\n";
    // Example 1: Basic assignment (=)
    int score = 10; // Assign 10 to score.
    cout << "Initial Score: " << score << "\n";

    // Example 2: Compound assignment (+=)
    score += 5; // score = score + 5.
    cout << "After adding 5 (score += 5): " << score << "\n";

    // Example 3: Compound assignment (-=)
    score -= 3; // score = score - 3.
    cout << "After subtracting 3 (score -= 3): " << score << "\n";

    // Example 4: Compound assignment (*=)
    score *= 2; // score = score * 2.
    cout << "After multiplying by 2 (score *= 2): " << score << "\n";

    // Example 5: Compound assignment (/=)
    if(score != 0) { // Always check to avoid division by zero.
        score /= 2; // score = score / 2.
        cout << "After dividing by 2 (score /= 2): " << score << "\n";
    } else {
        cout << "Cannot divide by zero!\n";
    }
    cout << "\n";

    // ----- Comparison Operators -----
    cout << "--- Comparison Operators ---\n";
    // Example 1: Equal to (==)
    bool isEqual = (apples == 5); // Check if apples equals 5.
    cout << "Is apples equal to 5? " << (isEqual ? "Yes" : "No") << "\n";

    // Example 2: Not equal to (!=)
    bool notEqual = (oranges != 5); // Check if oranges is not equal to 5.
    cout << "Is oranges not equal to 5? " << (notEqual ? "Yes" : "No") << "\n";

    // Example 3: Greater than (>)
    bool greater = (bananas > grapes);
    cout << "Are bananas greater than grapes? " << (greater ? "Yes" : "No") << "\n";

    // Example 4: Less than (<)
    bool less = (apples < bananas);
    cout << "Are apples less than bananas? " << (less ? "Yes" : "No") << "\n";

    // Example 5: Greater than or equal to (>=) and Less than or equal to (<=)
    bool ge = (oranges >= 3);
    bool le = (oranges <= 10);
    cout << "Are oranges between 3 and 10 (inclusive)? " 
         << ((ge && le) ? "Yes" : "No") << "\n\n";

    // ----- Logical Operators -----
    cout << "--- Logical Operators ---\n";
    // Example 1: Logical AND (&&)
    bool condition1 = (apples > 0 && oranges > 0); // Both conditions must be true.
    cout << "Both apples and oranges are positive? " 
         << (condition1 ? "Yes" : "No") << "\n";

    // Example 2: Logical OR (||)
    bool condition2 = (apples > 10 || bananas > 5); // At least one condition is true.
    cout << "Either apples > 10 or bananas > 5? " 
         << (condition2 ? "Yes" : "No") << "\n";

    // Example 3: Logical NOT (!)
    bool notSunny = !isSunny; // Inverts the boolean value.
    cout << "Is it not sunny? " << (notSunny ? "Yes" : "No") << "\n";

    // Example 4: Combining multiple logical operators.
    bool complexCondition = (apples > 0 && oranges > 0) || (bananas > 0 && grapes > 0);
    cout << "Complex condition (fruits > 0): " 
         << (complexCondition ? "True" : "False") << "\n";

    // Example 5: Using logical operators in an if statement.
    if ((apples > 2) && (bananas > 2)) {
        cout << "Both apples and bananas are more than 2 🍎🍌\n";
    } else {
        cout << "Not enough apples or bananas 🍎 or 🍌\n";
    }
    cout << "\n";

    // ----- Increment/Decrement Operators -----
    cout << "--- Increment/Decrement Operators ---\n";
    // Example 1: Post-increment (variable++)
    int counter = 0;
    cout << "Counter (initial): " << counter << "\n";
    cout << "Counter (post-increment, counter++): " << counter++ 
         << " (returns old value)\n"; // Returns 0, then counter becomes 1.
    cout << "Counter (after post-increment): " << counter << "\n";

    // Example 2: Pre-increment (++variable)
    cout << "Counter (pre-increment, ++counter): " << ++counter 
         << " (increments then returns new value)\n"; // Increments first.

    // Example 3: Post-decrement (variable--)
    cout << "Counter (post-decrement, counter--): " << counter-- 
         << " (returns old value)\n";
    cout << "Counter (after post-decrement): " << counter << "\n";

    // Example 4: Pre-decrement (--variable)
    cout << "Counter (pre-decrement, --counter): " << --counter 
         << " (decrements then returns new value)\n";

    // Example 5: Using increment in a for loop.
    cout << "\nLoop with Increment:\n";
    for (int i = 0; i < 5; ++i) { // Loop from 0 to 4.
        cout << "Iteration " << i << "\n";
    }

    // ======================================================
    // End of Examples
    // ======================================================
    return 0;
}
