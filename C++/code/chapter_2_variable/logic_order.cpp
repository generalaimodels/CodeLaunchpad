#include <iostream>

int main() {
    // Define some sample variables
    int a = 5;
    int b = 10;
    int c = 15;

    // Example 1:
    // Expression: a < b || b > c && c != a
    // Operator precedence in C++ makes '&&' evaluate before '||'.
    // Therefore, this expression is equivalent to: a < b || (b > c && c != a)
    bool result1 = a < b || b > c && c != a;
    std::cout << "Example 1 (a < b || b > c && c != a): " << std::boolalpha << result1 << std::endl;

    // Example 2:
    // Using parentheses to change the evaluation order.
    // Expression: (a < b || b > c) && c != a
    // This forces the OR to be evaluated before the AND.
    bool result2 = (a < b || b > c) && c != a;
    std::cout << "Example 2 ((a < b || b > c) && c != a): " << result2 << std::endl;

    // Example 3:
    // Combining logical NOT (!) with AND (&&) and OR (||).
    // Expression: !(a < b) || (b > c && c != a)
    // Here, the NOT operator inverts the result of (a < b), and then the result is OR-ed with (b > c && c != a).
    bool result3 = !(a < b) || (b > c && c != a);
    std::cout << "Example 3 (!(a < b) || (b > c && c != a)): " << result3 << std::endl;

    return 0;
}
