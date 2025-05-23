// #include <iostream>

// template<typename... Args>
// void print(Args... args) {
//     (std::cout << ... << args) << std::endl;
// }

// int Args_Main() {
//     print(1, 2, 3, "hello", 4.5);
//     return 0;
// }

#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

void function(int a, double b, std::string c) {
    std::cout << a << ", " << b << ", " << c << std::endl;
}

int Args_Main1() {
    auto args = std::make_tuple(1, 3.14, "hello");
    std::apply(function, args);
    return 0;
}



int Agrs_Main2() {
    std::vector<int> numbers = {1, 2, 3, 4, 5}; // List(Vector[int])
    
    auto square = [](int n) { return n * n; };

    
    std::transform(numbers.begin(), numbers.end(), numbers.begin(), square);
    
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    return 0;
}

/**
 * @brief Calculates the sum of two integers
 * @param a The first integer
 * @param b The second integer
 * @return The sum of a and b
 */
int add(int a, int b) {
    return a + b;
}


[[deprecated("Use newFunction() instead")]]
void oldFunction() {
    std::cout << "This function is deprecated" << std::endl;
}




// Use CamelCase for class names
class MyClass {
public:
    // Use camelCase for method names
    void doSomething() {
        // Use snake_case for variable names
        int some_variable = 42;
        std::cout << "Doing something: " << some_variable << std::endl;
    }
};

// Use ALL_CAPS for constants
const int MAX_SIZE = 100;

int Naming_Main() {
    // Use meaningful variable names
    std::vector<int> numbers;
    
    // Use consistent indentation (usually 2 or 4 spaces)
    for (int i = 0; i < 5; ++i) {
        numbers.push_back(i);
    }
    
    return 0;
}
int main() {
    // Args_Main();
    Args_Main1();
    Agrs_Main2();
    // oldFunction();
    Naming_Main();
    return 0;
}