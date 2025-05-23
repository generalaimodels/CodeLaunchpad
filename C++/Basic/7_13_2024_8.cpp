#include <iostream>
#include <vector>
#include <algorithm>
#include <list>
#include <deque>

int Reverse_Main() {
    std::vector<int> stack;

    // Push elements onto the stack
    stack.push_back(1);
    stack.push_back(2);
    stack.push_back(3);

    // Pop elements from the stack
    while (!stack.empty()) {
        std::cout << stack.back() << " ";
        stack.pop_back();
    }
    // Output: 3 2 1

    return 0;
}




int Reversed_Square_Main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::vector<int> squared;

    // Equivalent to [x**2 for x in numbers]
    std::transform(numbers.begin(), numbers.end(), std::back_inserter(squared),
                   [](int x) { return x * x; });

    for (int num : squared) {
        std::cout << num << " ";
    }
    // Output: 1 4 9 16 25

    return 0;
}


int Marix_Flatten_Main() {
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<int> flattened;

    // Equivalent to [num for row in matrix for num in row]
    for (const auto& row : matrix) {
        for (int num : row) {
            flattened.push_back(num);
        }
    }

    for (int num : flattened) {
        std::cout << num << " ";
    }
    // Output: 1 2 3 4 5 6 7 8 9

    return 0;
}


// 5.1. More on Lists
// 5.1.1. Using Lists as Stacks
void stackDemo() {
    std::cout << "Stack Demo:\n";
    std::vector<int> stack;

    // Push elements onto the stack
    stack.push_back(1);
    stack.push_back(2);
    stack.push_back(3);

    std::cout << "Stack: ";
    for (int num : stack)
        std::cout << num << " ";
    std::cout << "\n";

    // Pop elements from the stack
    while (!stack.empty()) {
        std::cout << "Popped: " << stack.back() << "\n";
        stack.pop_back();
    }
}

// 5.1.2. Using Lists as Queues
void queueDemo() {
    std::cout << "\nQueue Demo:\n";
    std::deque<int> queue;

    // Enqueue elements to the queue
    queue.push_back(1);
    queue.push_back(2);
    queue.push_back(3);

    std::cout << "Queue: ";
    for (int num : queue)
        std::cout << num << " ";
    std::cout << "\n";

    // Dequeue elements from the queue
    while (!queue.empty()) {
        std::cout << "Dequeued: " << queue.front() << "\n";
        queue.pop_front();
    }
}

// 5.1.3. List Comprehensions
std::vector<int> listComprehension(int n) {
    std::vector<int> result;
    for (int i = 1; i <= n; ++i)
        if (i % 2 == 0)
            result.push_back(i * i);
    return result;
}

// 5.1.4. Nested List Comprehensions
std::vector<std::vector<int>> nestedListComprehension(int n) {
    std::vector<std::vector<int>> result;
    for (int i = 1; i <= n; ++i) {
        std::vector<int> row;
        for (int j = 1; j <= i; ++j)
            row.push_back(j);
        result.push_back(row);
    }
    return result;
}

int main() {
    stackDemo();
    queueDemo();
    Reverse_Main();
    Reversed_Square_Main();
    Marix_Flatten_Main();
    


    std::cout << "\nList Comprehension:\n";
    std::vector<int> list = listComprehension(5);
    for (int num : list)
        std::cout << num << " ";
    std::cout << "\n";

    std::cout << "\nNested List Comprehension:\n";
    std::vector<std::vector<int>> nestedList = nestedListComprehension(4);
    for (const auto& row : nestedList) {
        for (int num : row)
            std::cout << num << " ";
        std::cout << "\n";
    }

    return 0;
}