#include <iostream>
#include <iostream>
#include <vector>

int if_else_elseif_else() {
    int x = 10;

    if (x > 5) {
        std::cout << "x is greater than 5" << std::endl;
    } 
    else if (x == 5) {
        std::cout << "x is equal to 5" << std::endl;
    } 
    else {
        std::cout << "x is less than 5" << std::endl;
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



int simple_For_Loop() {
    for (int i = 0; i < 5; ++i) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}


int Simple_For_Break_Continue() {
    for (int i = 0; i < 10; ++i) {
        if (i == 3) {
            continue; // Skip the rest of this iteration
        }
        if (i == 7) {
            break; // Exit the loop
        }
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}



int Simple_Syntax_Condition() {
    if (true) {
        // Empty block (equivalent to 'pass')
    }

    // Null statement (equivalent to 'pass')
    ;

    return 0;
}



int Simple_Swith_Case() {
    int x = 2;

    switch (x) {
        case 1:
            std::cout << "x is 1" << std::endl;
            break;
        case 2:
            std::cout << "x is 2" << std::endl;
            break;
        default:
            std::cout << "x is neither 1 nor 2" << std::endl;
    }

    return 0;
}


// Function declaration
int add(int a, int b);

// Function definition
int multiply(int a, int b) {
    return a * b;
}

int main_test() {
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;
    std::cout << "4 * 6 = " << multiply(4, 6) << std::endl;
    return 0;
}

// Function definition
int add(int a, int b) {
    return a + b;
}

int main() {
    if_else_elseif_else();
    Traditional_For_Loop();
    simple_For_Loop();
    Simple_For_Break_Continue();
    Simple_Syntax_Condition();
    Simple_Swith_Case();
    main_test();

 return 0;   
}