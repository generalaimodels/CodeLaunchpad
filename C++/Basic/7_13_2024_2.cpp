#include <iostream>
#include <cmath>
#include <vector>
#include <string>

int main1() {
    // Integer arithmetic
    int a = 5, b = 3;
    std::cout << "Addition: " << a + b << std::endl;
    std::cout << "Subtraction: " << a - b << std::endl;
    std::cout << "Multiplication: " << a * b << std::endl;
    std::cout << "Division: " << a / b << std::endl;
    std::cout << "Modulus: " << a % b << std::endl;

    // Floating-point arithmetic
    double x = 5.5, y = 2.5;
    std::cout << "Float addition: " << x + y << std::endl;
    std::cout << "Float division: " << x / y << std::endl;

    // Using cmath functions
    std::cout << "Square root of 25: " << std::sqrt(25) << std::endl;
    std::cout << "2 raised to power 3: " << std::pow(2, 3) << std::endl;

    return 0;
}


int main2() {
    // String declaration and initialization
    std::string greeting = "Hello";
    std::string name = "Alice";

    // String concatenation
    std::string message = greeting + ", " + name + "!";
    std::cout << message << std::endl;

    // String length
    std::cout << "Message length: " << message.length() << std::endl;

    // Accessing individual characters
    std::cout << "First character: " << message[0] << std::endl;

    // Substring
    std::cout << "Substring: " << message.substr(0, 5) << std::endl;

    // Finding a substring
    size_t found = message.find("Alice");
    if (found != std::string::npos) {
        std::cout << "Found 'Alice' at position: " << found << std::endl;
    }

    return 0;
}

int main3() {
    // Create a vector of integers
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // Add elements to the vector
    numbers.push_back(6);
    numbers.push_back(7);

    // Print all elements
    std::cout << "Vector elements: ";
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Access elements
    std::cout << "First element: " << numbers[0] << std::endl;
    std::cout << "Last element: " << numbers.back() << std::endl;

    // Vector size
    std::cout << "Vector size: " << numbers.size() << std::endl;

    // Remove last element
    numbers.pop_back();

    // Check if vector is empty
    if (!numbers.empty()) {
        std::cout << "Vector is not empty" << std::endl;
    }

    return 0;
}



// Function to calculate the average of a vector of numbers
double calculateAverage(const std::vector<double>& numbers) {
    double sum = 0;
    for (double num : numbers) {
        sum += num;
    }
    return sum / numbers.size();
}

int main4() {
    std::string name;
    std::cout << "Enter your name: ";
    std::getline(std::cin, name);

    std::cout << "Hello, " << name << "! Let's calculate the average of some numbers." << std::endl;

    std::vector<double> numbers;
    double num;

    while (true) {
        std::cout << "Enter a number (or -1 to finish): ";
        std::cin >> num;

        if (num == -1) {
            break;
        }

        numbers.push_back(num);
    }

    if (!numbers.empty()) {
        double average = calculateAverage(numbers);
        std::cout << "The average of the numbers you entered is: " << average << std::endl;
    } else {
        std::cout << "No numbers were entered." << std::endl;
    }

    return 0;
}

int main(){
    main1();
    main2();
    main3();
    main4();


   
    return 0;
}