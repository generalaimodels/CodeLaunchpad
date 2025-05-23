#include <iostream>
#include <vector>
int Simple_If_Else_If_Else() {
    int x = 10;

    if (x > 0) {
        std::cout << "x is positive" << std::endl;
    } else if (x < 0) {
        std::cout << "x is negative" << std::endl;
    } else {
        std::cout << "x is zero" << std::endl;
    }

    return 0;
}



int Traditional_For_Loop() {
    // Traditional for loop
    for (int i = 0; i < 5; ++i) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    // Range-based for loop (C++11 and later)
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}



std::vector<int> range(int start, int end, int step = 1) {
    std::vector<int> result;
    for (int i = start; i < end; i += step) {
        result.push_back(i);
    }
    return result;
}

int Simple_For_Loop() {
    for (int i : range(0, 5)) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}


int Simple_fun_Pass() {
    // Empty block
    if (true) {
        // Do nothing
    }

    // Null statement
    ;

    return 0;
}



int Simple_Switch_Case() {
    int x = 2;

    switch (x) {
        case 1:
            std::cout << "One" << std::endl;
            break;
        case 2:
            std::cout << "Two" << std::endl;
            break;
        case 3:
            std::cout << "Three" << std::endl;
            break;
        default:
            std::cout << "Other" << std::endl;
    }

    return 0;
}



// Function declaration
int add(int a, int b);

// Function definition
int multiply(int a, int b) {
    return a * b;
}

int Function_calling_Use() {
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;
    std::cout << "5 * 3 = " << multiply(5, 3) << std::endl;
    return 0;
}

// Function definition
int add(int a, int b) {
    return a + b;
}

int main() {
    // Call main_test function
    Simple_If_Else_If_Else();
    Traditional_For_Loop();
    Simple_For_Loop();
    Simple_fun_Pass();
    Simple_Switch_Case();
    Function_calling_Use();
    return 0;
}