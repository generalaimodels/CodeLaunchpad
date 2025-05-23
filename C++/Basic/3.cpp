#include <iostream>
#include <iomanip> // For advanced formatting
#include <cmath> // For mathematical functions

class Calculator {
public:
    // Basic arithmetic operations
    static double add(double a, double b) { return a + b; }
    static double subtract(double a, double b) { return a - b; }
    static double multiply(double a, double b) { return a * b; }
    static double divide(double a, double b) {
        if (b != 0) return a / b;
        else {
            std::cout << "Error: Division by zero!" << std::endl;
            return 0;
        }
    }

    // Basic formatting
    static void basicFormat(double result) {
        std::cout << "Result: " << result << std::endl;
    }

    // Advanced formatting
    static void advancedFormat(double result) {
        std::cout << "Result: " << std::fixed << std::setprecision(2) << result << std::endl;
        std::cout << "Scientific notation: " << std::scientific << result << std::endl;
        std::cout << "Hexadecimal: " << std::hex << std::setprecision(0) << static_cast<long long>(result) << std::endl;
    }
};

// Function to demonstrate usage
void performCalculations() {
    double a = 10.5, b = 5.2;

    std::cout << "Addition:" << std::endl;
    Calculator::basicFormat(Calculator::add(a, b));
    Calculator::advancedFormat(Calculator::add(a, b));

    std::cout << "\nSubtraction:" << std::endl;
    Calculator::basicFormat(Calculator::subtract(a, b));
    Calculator::advancedFormat(Calculator::subtract(a, b));

    std::cout << "\nMultiplication:" << std::endl;
    Calculator::basicFormat(Calculator::multiply(a, b));
    Calculator::advancedFormat(Calculator::multiply(a, b));

    std::cout << "\nDivision:" << std::endl;
    Calculator::basicFormat(Calculator::divide(a, b));
    Calculator::advancedFormat(Calculator::divide(a, b));

    // Test division by zero
    std::cout << "\nDivision by zero:" << std::endl;
    Calculator::divide(a, 0);
}

int main() {
    performCalculations();
    return 0;
}