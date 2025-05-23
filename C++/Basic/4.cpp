#include <iostream>

class Calculator {
public:
    static int add(int a, int b) {
        return a + b;
    }

    static int sub(int a, int b) {
        return a - b;
    }

    static int mul(int a, int b) {
        return a * b;
    }

    static double div(int a, int b) {
        if (b == 0) {
            std::cerr << "Error: Division by zero" << std::endl;
            return 0;
        }
        return static_cast<double>(a) / b;
    }
};

int main() {
    int a = 10, b = 5;

    std::cout << "Addition:" << Calculator::add(a, b) << std::endl;
    std::cout << "Subtraction: " << Calculator::sub(a, b) << std::endl;
    std::cout << "Multiplication: " << Calculator::mul(a, b) << std::endl;
    std::cout << "Division: " << Calculator::div(a, b) << std::endl;

    return 0;
}