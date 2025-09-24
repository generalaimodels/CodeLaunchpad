/*
 * OperatorPrecedenceExamples.cpp
 *
 * This file demonstrates the order of operations in C++ using 25 examples.
 * Each example is explained step by step with inline comments.
 *
 * The examples start with basic arithmetic and progress to more advanced topics.
 *
 * To compile (using C++11 or later):
 *     g++ -std=c++11 OperatorPrecedenceExamples.cpp -o OperatorPrecedenceExamples
 *
 * Run:
 *     ./OperatorPrecedenceExamples
 */

#include <iostream> // For input/output operations

class OperatorPrecedenceExamples {
public:
    // Example 1: Multiplication has higher precedence than addition.
    static void example1() {
        int result = 2 + 3 * 4; // Evaluated as 2 + (3 * 4) = 14.
        std::cout << "Example 1: 2 + 3 * 4 = " << result 
                  << "  (Multiplication before addition)\n";
    }

    // Example 2: Parentheses override the default precedence.
    static void example2() {
        int result = (2 + 3) * 4; // Evaluated as (2 + 3) * 4 = 20.
        std::cout << "Example 2: (2 + 3) * 4 = " << result 
                  << "  (Parentheses change evaluation order)\n";
    }

    // Example 3: Unary minus has higher precedence than multiplication.
    static void example3() {
        int result = -2 * 3; // Evaluated as (-2) * 3 = -6.
        std::cout << "Example 3: -2 * 3 = " << result 
                  << "  (Unary minus applies before multiplication)\n";
    }

    // Example 4: Pre-increment vs. post-increment.
    static void example4() {
        int a = 5;
        int pre  = ++a; // Pre-increment: a becomes 6, then pre = 6.
        int post = a++; // Post-increment: post = 6, then a becomes 7.
        std::cout << "Example 4: Pre-increment vs. Post-increment:\n"
                  << "   After pre-increment, a = " << pre << "\n"
                  << "   After post-increment, a = " << a << " and post = " << post << "\n";
    }

    // Example 5: Logical operators: && is evaluated before ||.
    static void example5() {
        // Evaluated as: false || (true && false) = false.
        bool result = false || true && false;
        std::cout << "Example 5: false || true && false = " << result 
                  << "  (&& is evaluated before ||)\n";
    }

    // Example 6: Bitwise AND (&) used with boolean comparisons.
    static void example6() {
        // Here the bitwise AND acts like a logical AND for bool values.
        bool result = (5 == 5) & (3 == 3); 
        std::cout << "Example 6: (5 == 5) & (3 == 3) = " << result << "\n";
    }

    // Example 7: Assignment operator is right-associative.
    static void example7() {
        int x, y;
        x = y = 10; // Evaluated as x = (y = 10).
        std::cout << "Example 7: After x = y = 10, x = " << x << ", y = " << y << "\n";
    }

    // Example 8: Comma operator evaluates the left operand, discards it, and then returns the right.
    static void example8() {
        int a = 0;
        int result = (a = 5, a + 10); // a is set to 5, then a+10 evaluates to 15.
        std::cout << "Example 8: (a = 5, a + 10) = " << result << "\n";
    }

    // Example 9: Ternary (conditional) operator.
    static void example9() {
        int a = 10, b = 20;
        int result = (a > b ? a : b); // Evaluates to 20 since a > b is false.
        std::cout << "Example 9: (10 > 20 ? 10 : 20) = " << result << "\n";
    }

    // Example 10: The sizeof operator returns the size (in bytes) of a type.
    static void example10() {
        std::cout << "Example 10: sizeof(int) = " << sizeof(int) << "\n";
    }

    // Example 11: Address-of (&) and dereference (*) operators.
    static void example11() {
        int a = 42;
        int* ptr = &a;   // '&' gets the address of a.
        int value = *ptr; // '*' dereferences ptr.
        std::cout << "Example 11: a = " << a << ", *(&a) = " << value << "\n";
    }

    // Example 12: Pointer arithmetic and dereference.
    static void example12() {
        int arr[5] = {10, 20, 30, 40, 50};
        int* ptr = arr;
        int value = *(ptr + 2); // Points to the third element (30).
        std::cout << "Example 12: Third element of array is " << value << "\n";
    }

    // Example 13: Left shift (<<) operator.
    static void example13() {
        int a = 8;         // Binary: 1000.
        int result = a << 2; // 8 << 2 equals 32.00100000.
        std::cout << "Example 13: 8 << 2 = " << result << "\n";
    }

    // Example 14: Bitwise OR (|) and XOR (^) operators.
    static void example14() {
        int a = 5; // 0101 in binary.
        int b = 3; // 0011 in binary.
        int orResult = a | b;  // 0101 | 0011 = 0111 (7).
        int xorResult = a ^ b; // 0101 ^ 0011 = 0110 (6).
        std::cout << "Example 14: 5 | 3 = " << orResult 
                  << ", 5 ^ 3 = " << xorResult << "\n";
    }

    // Example 15: Logical NOT (!) vs. Bitwise NOT (~).
    static void example15() {
        bool flag = false;
        std::cout << "Example 15: !false = " << !flag << "  (Logical NOT)\n";
        int num = 0;
        std::cout << "           ~0 = " << ~num << "  (Bitwise NOT)\n";
    }

    // Example 16: Multiplication and division are evaluated left-to-right.
    static void example16() {
        int result = 100 / 5 * 2; // Evaluated as (100 / 5) * 2 = 20 * 2 = 40.
        std::cout << "Example 16: 100 / 5 * 2 = " << result << "\n";
    }

    // Example 17: Combining addition and shift operators.
    static void example17() {
        int a = 3, b = 4, c = 2;
        // Since '+' has higher precedence than '<<', the expression is evaluated as (a + b) << c.
        int result = (a + b) << c; // (3 + 4) << 2 = 7 << 2 = 28.
        std::cout << "Example 17: (3 + 4) << 2 = " << result 
                  << "  (Addition before left shift)\n";
    }

    // Example 18: Assignment within an expression using the comma operator.
    static void example18() {
        int a = 5;
        int result = (a = 10, a * 2); // a is set to 10, then result becomes 10 * 2 = 20.
        std::cout << "Example 18: (a = 10, a * 2) = " << result 
                  << ", a = " << a << "\n";
    }

    // Example 19: Ternary operator in a nested expression to find the maximum.
    static void example19() {
        int a = 15, b = 20, c = 25;
        int result = a > b ? (a > c ? a : c) : (b > c ? b : c);
        std::cout << "Example 19: Maximum of (15, 20, 25) = " << result << "\n";
    }

    // Example 20: Type casting and multiplication.
    static void example20() {
        double x = 3.14159;
        int result = int(x) * 2; // The cast int(x) truncates 3.14159 to 3, then multiplied by 2.
        std::cout << "Example 20: int(3.14159) * 2 = " << result << "\n";
    }

    // Example 21: The 'new' operator creates a dynamically allocated object.
    static void example21() {
        int* p = new int(100);
        std::cout << "Example 21: Value pointed by new int(100) = " << *p << "\n";
        delete p;
    }

    // Example 22: Pointer-to-member operators (.* and ->*).
    static void example22() {
        struct S {
            int value;
        } 
        
        
        s {42};
        int S::* ptr = &S::value;
        std::cout << "Example 22: s.*ptr = " << s.*ptr << "\n";
    }

    // Example 23: Lambda expression capturing a variable.
    static void example23() {
        int a = 5;
        auto lambda = [a]() -> int { return a * 2; };
        std::cout << "Example 23: Lambda capturing a=5 returns a*2 = " << lambda() << "\n";
    }

    // Example 24: Function call precedence in an arithmetic expression.
    static void example24() {
        auto add = [](int a, int b) -> int { return a + b; };
        int result = add(2, 3) * 4; // add(2,3) returns 5, then multiplied by 4.
        std::cout << "Example 24: add(2, 3) * 4 = " << result << "\n";
    }

    // Example 25: Mixed expression with multiple operators.
    static void example25() {
        int a = 2, b = 3, c = 4, d = 5;
        // Multiplication and division are performed before addition and subtraction:
        // 2 + (3 * 4) - (5 / 2) = 2 + 12 - 2 = 12  (integer division: 5/2 = 2)
        int result = a + b * c - d / a;
        std::cout << "Example 25: 2 + 3 * 4 - 5 / 2 = " << result << "\n";
    }

    // Run all the examples in order.
    static void runAllExamples() {
        example1();
        example2();
        example3();
        example4();
        example5();
        example6();
        example7();
        example8();
        example9();
        example10();
        example11();
        example12();
        example13();
        example14();
        example15();
        example16();
        example17();
        example18();
        example19();
        example20();
        example21();
        example22();
        example23();
        example24();
        example25();
    }
};

int main() {
    OperatorPrecedenceExamples::runAllExamples();
    return 0;
}
