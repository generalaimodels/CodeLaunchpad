#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>

// 5. Data Structures
// 5.1. More on Lists (using std::vector in C++)

int Data_Structures_Main() {
    // 5.1.1. Using Lists as Stacks
    std::cout << "5.1.1. Using Lists as Stacks:\n";
    std::stack<int> stack;
    
    // Push elements onto the stack
    stack.push(1);
    stack.push(2);
    stack.push(3);
    
    // Pop elements from the stack
    while (!stack.empty()) {
        std::cout << stack.top() << " ";
        stack.pop();
    }
    std::cout << "\n\n";

    // 5.1.2. Using Lists as Queues
    std::cout << "5.1.2. Using Lists as Queues:\n";
    std::queue<int> queue;
    
    // Enqueue elements
    queue.push(1);
    queue.push(2);
    queue.push(3);
    
    // Dequeue elements
    while (!queue.empty()) {
        std::cout << queue.front() << " ";
        queue.pop();
    }
    std::cout << "\n\n";

    // 5.1.3. List Comprehensions
    std::cout << "5.1.3. List Comprehensions:\n";
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // Equivalent to list comprehension: [x*x for x in numbers]
    std::vector<int> squares;
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(squares),
                   [](int x) { return x * x; });
    
    for (int square : squares) {
        std::cout << square << " ";
    }
    std::cout << "\n\n";

    // 5.1.4. Nested List Comprehensions
    std::cout << "5.1.4. Nested List Comprehensions:\n";
    std::vector<std::vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    
    // Equivalent to nested list comprehension: [[row[i] for row in matrix] for i in range(3)]
    std::vector<std::vector<int>> transposed(3, std::vector<int>(3));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            transposed[i][j] = matrix[j][i];
        }
    }
    
    for (const auto& row : transposed) {
        for (int element : row) {
            std::cout << element << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

// Case2_Code_Snippet.cpp
int Case2_Code_Snippet_Main() {
    // 5. Data Structures
    // 5.1. More on Lists

    // In C++, we can use std::vector or std::list for list-like functionality

    // 5.1.1. Using Lists as Stacks
    std::cout << "Using Lists as Stacks:" << std::endl;
    std::stack<int> stack;
    
    // Push elements onto the stack
    stack.push(1);
    stack.push(2);
    stack.push(3);
    
    // Pop elements from the stack
    while (!stack.empty()) {
        std::cout << stack.top() << " ";
        stack.pop();
    }
    std::cout << std::endl;

    // 5.1.2. Using Lists as Queues
    std::cout << "\nUsing Lists as Queues:" << std::endl;
    std::queue<int> queue;
    
    // Enqueue elements
    queue.push(1);
    queue.push(2);
    queue.push(3);
    
    // Dequeue elements
    while (!queue.empty()) {
        std::cout << queue.front() << " ";
        queue.pop();
    }
    std::cout << std::endl;

    // 5.1.3. List Comprehensions
    // C++ doesn't have built-in list comprehensions, but we can use algorithms and lambda functions
    std::cout << "\nList Comprehension-like operations:" << std::endl;
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<int> squared_numbers;
    
    // Square each number (similar to [x**2 for x in numbers] in Python)
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(squared_numbers),
                   [](int x) { return x * x; });
    
    for (int num : squared_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Filtering (similar to [x for x in numbers if x % 2 == 0] in Python)
    std::vector<int> even_numbers;
    std::copy_if(numbers.begin(), numbers.end(), std::back_inserter(even_numbers),
                 [](int x) { return x % 2 == 0; });
    
    for (int num : even_numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 5.1.4. Nested List Comprehensions
    // Again, C++ doesn't have built-in nested list comprehensions, but we can achieve similar results
    std::cout << "\nNested List Comprehension-like operations:" << std::endl;
    std::vector<std::vector<int>> matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<int> flattened;
    
    // Flatten the matrix (similar to [num for row in matrix for num in row] in Python)
    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    
    for (int num : flattened) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
int main () {
    std::cout << "Hello, World!\n";
    Data_Structures_Main();
    Case2_Code_Snippet_Main();
    return 0;
}