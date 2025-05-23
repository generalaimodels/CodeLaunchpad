// control_flow_examples.cpp

#include <iostream>
#include <vector>

class SequentialExecutionExamples {
public:
    static void example1() {
        // Example 1: Summing numbers in an array
        std::vector<int> numbers = {1, 2, 3, 4, 5};
        int sum = 0;
        sum += numbers[0];
        sum += numbers[1];
        sum += numbers[2];
        sum += numbers[3];
        sum += numbers[4];
        std::cout << "Sum: " << sum << std::endl;
    }

    static void example2() {
        // Example 2: Printing messages sequentially
        std::cout << "Starting program..." << std::endl;
        std::cout << "Performing task 1..." << std::endl;
        std::cout << "Performing task 2..." << std::endl;
        std::cout << "Program finished." << std::endl;
    }

    static void example3() {
        // Example 3: Calculating area and perimeter of a rectangle
        int width = 5;
        int height = 10;
        int area = width * height;
        int perimeter = 2 * (width + height);
        std::cout << "Area: " << area << ", Perimeter: " << perimeter << std::endl;
    }

    static void example4() {
        // Example 4: Simple interest calculation
        double principal = 1000.0;
        double rate = 5.0; // in percent
        int years = 3;
        double interest = principal * rate * years / 100.0;
        std::cout << "Simple interest: " << interest << std::endl;
    }

    static void example5() {
        // Example 5: Temperature conversion from Celsius to Fahrenheit
        double celsius = 25.0;
        double fahrenheit = celsius * 9.0 / 5.0 + 32;
        std::cout << celsius << "°C = " << fahrenheit << "°F" << std::endl;
    }
};

class ConditionalStatementExamples {
public:
    static void ifExample1() {
        // Example 1: Check if a number is positive
        int num = 10;
        if (num > 0) {
            std::cout << num << " is positive." << std::endl;
        }
    }

    static void ifExample2() {
        // Example 2: Check if a character is a vowel
        char ch = 'a';
        if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u') {
            std::cout << ch << " is a vowel." << std::endl;
        }
    }

    static void ifElseExample1() {
        // Example 3: Check if a number is even or odd
        int num = 5;
        if (num % 2 == 0) {
            std::cout << num << " is even." << std::endl;
        } else {
            std::cout << num << " is odd." << std::endl;
        }
    }

    static void ifElseExample2() {
        // Example 4: Check if age is eligible to vote
        int age = 16;
        if (age >= 18) {
            std::cout << "You are eligible to vote." << std::endl;
        } else {
            std::cout << "You are not eligible to vote." << std::endl;
        }
    }

    static void elseIfExample1() {
        // Example 5: Grade evaluation based on marks
        int marks = 85;
        if (marks >= 90) {
            std::cout << "Grade A" << std::endl;
        } else if (marks >= 80) {
            std::cout << "Grade B" << std::endl;
        } else if (marks >= 70) {
            std::cout << "Grade C" << std::endl;
        } else if (marks >= 60) {
            std::cout << "Grade D" << std::endl;
        } else {
            std::cout << "Grade F" << std::endl;
        }
    }

    static void elseIfExample2() {
        // Example 6: Determine time of day
        int hour = 14;
        if (hour < 12) {
            std::cout << "Good morning" << std::endl;
        } else if (hour < 18) {
            std::cout << "Good afternoon" << std::endl;
        } else {
            std::cout << "Good evening" << std::endl;
        }
    }

    static void switchExample1() {
        // Example 7: Simple calculator using switch
        char op = '+';
        int a = 5, b = 3, result = 0;
        switch (op) {
            case '+':
                result = a + b;
                break;
            case '-':
                result = a - b;
                break;
            case '*':
                result = a * b;
                break;
            case '/':
                if (b != 0) {
                    result = a / b;
                } else {
                    std::cout << "Division by zero error." << std::endl;
                    return;
                }
                break;
            default:
                std::cout << "Unsupported operation." << std::endl;
                return;
        }
        std::cout << "Result: " << result << std::endl;
    }

    static void switchExample2() {
        // Example 8: Print day of the week
        int day = 3;
        switch (day) {
            case 1:
                std::cout << "Monday" << std::endl;
                break;
            case 2:
                std::cout << "Tuesday" << std::endl;
                break;
            case 3:
                std::cout << "Wednesday" << std::endl;
                break;
            case 4:
                std::cout << "Thursday" << std::endl;
                break;
            case 5:
                std::cout << "Friday" << std::endl;
                break;
            case 6:
                std::cout << "Saturday" << std::endl;
                break;
            case 7:
                std::cout << "Sunday" << std::endl;
                break;
            default:
                std::cout << "Invalid day" << std::endl;
                break;
        }
    }
};

class LoopExamples {
public:
    static void forLoopExample1() {
        // Example 1: Print numbers from 1 to 5
        for (int i = 1; i <= 5; ++i) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

    static void forLoopExample2() {
        // Example 2: Calculate factorial of a number
        int n = 5;
        int factorial = 1;
        for (int i = 1; i <= n; ++i) {
            factorial *= i;
        }
        std::cout << "Factorial of " << n << " is " << factorial << std::endl;
    }

    static void whileLoopExample1() {
        // Example 3: Sum of digits of a number
        int num = 12345;
        int sum = 0;
        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }
        std::cout << "Sum of digits is " << sum << std::endl;
    }

    static void whileLoopExample2() {
        // Example 4: Print multiplication table
        int num = 12;
        int i = 1;
        while (i <= 20) {
            std::cout << num << " x " << i << " = " << num * i << std::endl;
            ++i;
        }
    }

    static void doWhileLoopExample1() {
        // Example 5: User input validation
        int number;
        do {
            std::cout << "Enter a positive number: ";
            std::cin >> number;
        } while (number <= 0);
        std::cout << "You entered: " << number << std::endl;
    }

    static void doWhileLoopExample2() {
        // Example 6: Menu selection
        char choice;
        do {
            std::cout << "Menu:\n";
            std::cout << "1. Option 1\n";
            std::cout << "2. Option 2\n";
            std::cout << "3. Exit\n";
            std::cout << "Enter your choice: ";
            std::cin >> choice;
            switch (choice) {
                case '1':
                    std::cout << "You selected Option 1\n";
                    break;
                case '2':
                    std::cout << "You selected Option 2\n";
                    break;
                case '3':
                    std::cout << "Exiting...\n";
                    break;
                default:
                    std::cout << "Invalid choice\n";
                    break;
            }
        } while (choice != '3');
    }

    static void breakContinueExample1() {
        // Example 7: Loop with break statement
        for (int i = 1; i <= 10; ++i) {
            if (i == 5) {
                break;
            }
            std::cout << i << " ";
        }
        std::cout << "\nLoop exited when i == 5" << std::endl;
    }

    static void breakContinueExample2() {
        // Example 8: Loop with continue statement
        for (int i = 1; i <= 10; ++i) {
            if (i % 2 == 0) {
                continue;
            }
            std::cout << i << " ";
        }
        std::cout << "\nOnly odd numbers are printed." << std::endl;
    }
};

int main() {
    // Sequential Execution Examples
    SequentialExecutionExamples::example1();
    SequentialExecutionExamples::example2();
    SequentialExecutionExamples::example3();
    SequentialExecutionExamples::example4();
    SequentialExecutionExamples::example5();

    // Conditional Statement Examples
    ConditionalStatementExamples::ifExample1();
    ConditionalStatementExamples::ifExample2();
    ConditionalStatementExamples::ifElseExample1();
    ConditionalStatementExamples::ifElseExample2();
    ConditionalStatementExamples::elseIfExample1();
    ConditionalStatementExamples::elseIfExample2();
    ConditionalStatementExamples::switchExample1();
    ConditionalStatementExamples::switchExample2();

    // Loop Examples
    LoopExamples::forLoopExample1();
    LoopExamples::forLoopExample2();
    LoopExamples::whileLoopExample1();
    LoopExamples::whileLoopExample2();
    // Commented out to avoid blocking program with input
    // LoopExamples::doWhileLoopExample1();
    // LoopExamples::doWhileLoopExample2();
    LoopExamples::breakContinueExample1();
    LoopExamples::breakContinueExample2();

    return 0;
}