// Chapter 5: Recursion - The Self-Calling Function: Mirrors & Fractals 🪞🔄

#include <iostream>
#include <string>
#include <vector>
using namespace std;

/*
========================================
Concept: Function Calling Itself 🪞🔄🧩
- Solving problems by breaking them into smaller, self-similar subproblems.
- Like mirrors reflecting mirrors, creating smaller versions of themselves.
========================================
*/

// Example 1: Calculating Factorial using Recursion 🧮➡️🔄
int factorial(int n) {
    // Base Case: If n is 0 or 1, factorial is 1 🛑🪞
    if (n <= 1) {
        return 1;
    }
    // Recursive Step: n * factorial of (n - 1) ➡️🪞
    return n * factorial(n - 1);
}

// Example 2: Fibonacci Sequence using Recursion 🧮➡️🔄
int fibonacci(int n) {
    // Base Cases: If n is 0 or 1 🛑🪞
    if (n == 0) return 0;
    if (n == 1) return 1;
    // Recursive Step: Sum of two preceding numbers ➡️🪞
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Example 3: Printing Numbers from N to 1 using Recursion 🪞✨
void printDescending(int n) {
    // Base Case: If n is less than 1, stop 🛑🪞
    if (n < 1) {
        return;
    }
    cout << n << " ";
    // Recursive Step: Call function with (n - 1) ➡️🪞
    printDescending(n - 1);
}

// Handle possible mistake: Missing base case can cause infinite recursion ❌
// Correct version with base case included above.

// Example 4: Sum of Elements in an Array using Recursion 🧮➡️🔄
int sumArray(int arr[], int size) {
    // Base Case: If size is 0, sum is 0 🛑🪞
    if (size == 0) {
        return 0;
    }
    // Recursive Step: Last element + sum of the rest ➡️🪞
    return arr[size - 1] + sumArray(arr, size - 1);
}

// Example 5: Reversing a String using Recursion 🪞➡️
void reverseString(string& str, int start, int end) {
    // Base Case: When start >= end 🛑🪞
    if (start >= end) {
        return;
    }
    // Swap characters at positions start and end 🔄
    swap(str[start], str[end]);
    // Recursive Step: Move towards the center ➡️🪞
    reverseString(str, start + 1, end - 1);
}

// Example 6: Binary Search using Recursion ⚔️➡️🔄
int binarySearch(int arr[], int target, int left, int right) {
    // Base Case: When left > right, target not found 🛑🪞
    if (left > right) {
        return -1; // Not found
    }
    int mid = left + (right - left) / 2;
    // Check if mid is the target
    if (arr[mid] == target) {
        return mid;
    }
    // Recursive Step: Search in left or right half ➡️🪞
    if (arr[mid] > target) {
        return binarySearch(arr, target, left, mid - 1);
    } else {
        return binarySearch(arr, target, mid + 1, right);
    }
}

// Example 7: Tower of Hanoi Problem 🧩➡️🔄
void towerOfHanoi(int n, char from_rod, char to_rod, char aux_rod) {
    // Base Case: Only one disk to move 🛑🪞
    if (n == 1) {
        cout << "Move disk 1 from rod " << from_rod << " to rod " << to_rod << endl;
        return;
    }
    // Recursive Step: Move n-1 disks to auxiliary rod ➡️🪞
    towerOfHanoi(n - 1, from_rod, aux_rod, to_rod);
    // Move remaining disk to target rod
    cout << "Move disk " << n << " from rod " << from_rod << " to rod " << to_rod << endl;
    // Move n-1 disks from auxiliary rod to target rod ➡️🪞
    towerOfHanoi(n - 1, aux_rod, to_rod, from_rod);
}

// Example 8: Generating Permutations of a String 🌀➡️🔄
void permute(string str, int l, int r) {
    // Base Case: All positions fixed 🛑🪞
    if (l == r) {
        cout << str << endl;
    } else {
        // Recursive Step: Swap and permute ➡️🪞
        for (int i = l; i <= r; i++) {
            swap(str[l], str[i]); // Swap characters
            permute(str, l + 1, r); // Recurse
            swap(str[l], str[i]); // Backtrack 🔄
        }
    }
}

// Example 9: Recursive Tree Traversal (Inorder Traversal) 🌳➡️🔄
struct Node {
    int data;
    Node* left;
    Node* right;
    Node(int val) : data(val), left(nullptr), right(nullptr) {} // Constructor
};

void inorderTraversal(Node* root) {
    // Base Case: If node is null 🛑🪞
    if (root == nullptr) {
        return;
    }
    // Recursive Step: Visit left subtree ➡️🪞
    inorderTraversal(root->left);
    // Process current node 👀
    cout << root->data << " ";
    // Visit right subtree ➡️🪞
    inorderTraversal(root->right);
}

// Example 10: Merge Sort using Recursion ⚔️➡️🔄
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1; // Size of left subarray
    int n2 = right - mid;    // Size of right subarray

    // Create temporary arrays
    int* L = new int[n1];
    int* R = new int[n2];

    // Copy data to temporary arrays
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    // Merge the temporary arrays back into arr[l..r]
    int i = 0, j = 0, k = left; // Initial indices
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) { arr[k++] = L[i++]; }
        else { arr[k++] = R[j++]; }
    }

    // Copy any remaining elements
    while (i < n1) { arr[k++] = L[i++]; }
    while (j < n2) { arr[k++] = R[j++]; }

    // Free memory
    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int left, int right) {
    // Base Case: If left >= right 🛑🪞
    if (left >= right) {
        return;
    }
    int mid = left + (right - left) / 2; // Find the middle point
    // Recursive Steps: Sort first and second halves ➡️🪞
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    // Merge the sorted halves
    merge(arr, left, mid, right);
}

// Example 11: Fractal Generation (Simulated) 🌀➡️🔄
void drawFractal(int depth) {
    // Base Case: When depth is zero 🛑🪞
    if (depth == 0) {
        cout << "Line ";
        return;
    }
    // Recursive Step: Break line into parts ➡️🪞
    drawFractal(depth - 1);
    cout << "Left ";
    drawFractal(depth - 1);
    cout << "Right ";
    drawFractal(depth - 1);
    cout << "Left ";
    drawFractal(depth - 1);
}

// Example 12: Tail Recursion ➡️🪞(last step)
void tailRecursion(int n) {
    // Base Case: When n <= 0 🛑🪞
    if (n <= 0) {
        return;
    }
    cout << n << " ";
    // Recursive Call is the last operation ➡️🪞
    tailRecursion(n - 1);
}

// Note: Tail recursion can be optimized by compilers to avoid stack overflow.

// Example 13: Infinite Recursion (Leads to stack overflow) ❌🛑🪞
// Uncommenting and running this function will cause a stack overflow error.
// int infiniteRecursion(int n) {
//     // Missing Base Case!
//     return infiniteRecursion(n + 1); // No stopping condition ➡️🪞
// }

// Example 14: Mutual Recursion 🔄➡️🪞🪞
void functionB(int n); // Forward declaration
void functionA(int n) {
    if (n > 0) {
        cout << "A: " << n << " ";
        functionB(n - 1); // Calls functionB ➡️🪞
    } else {
        // Base Case for functionA 🛑🪞
        return;
    }
}

void functionB(int n) {
    if (n > 0) {
        cout << "B: " << n << " ";
        functionA(n / 2); // Calls functionA ➡️🪞
    } else {
        // Base Case for functionB 🛑🪞
        return;
    }
}

// Example 15: Calculating Combinations using Recursion 🧮➡️🔄
int combination(int n, int k) {
    // Base Cases: If k == 0 or k == n 🛑🪞
    if (k == 0 || k == n) {
        return 1;
    }
    // Recursive Step: C(n, k) = C(n-1, k-1) + C(n-1, k) ➡️🪞
    return combination(n - 1, k - 1) + combination(n - 1, k);
}

// Main Function to Demonstrate Examples
int main() {
    // Example 1: Factorial
    cout << "Example 1: Factorial of 5 is " << factorial(5) << endl;

    // Example 2: Fibonacci Sequence
    cout << "Example 2: Fibonacci sequence up to 10: ";
    for (int i = 0; i <= 10; i++) {
        cout << fibonacci(i) << " ";
    }
    cout << endl;

    // Example 3: Printing Numbers from N to 1
    cout << "Example 3: Numbers from 5 to 1: ";
    printDescending(5);
    cout << endl;

    // Example 4: Sum of Array Elements
    int arr[] = {1, 2, 3, 4, 5};
    cout << "Example 4: Sum of array elements: " << sumArray(arr, 5) << endl;

    // Example 5: Reversing a String
    string s = "Recursion";
    reverseString(s, 0, s.length() - 1);
    cout << "Example 5: Reversed string: " << s << endl;

    // Example 6: Binary Search
    int sortedArr[] = {1, 3, 5, 7, 9};
    int target = 7;
    int index = binarySearch(sortedArr, target, 0, 4);
    if (index != -1) {
        cout << "Example 6: Target " << target << " found at index " << index << endl;
    } else {
        cout << "Example 6: Target " << target << " not found" << endl;
    }

    // Example 7: Tower of Hanoi
    cout << "Example 7: Tower of Hanoi with 3 disks:\n";
    towerOfHanoi(3, 'A', 'C', 'B'); // From rod A to C using B

    // Example 8: Generating Permutations
    cout << "Example 8: Permutations of 'ABC':\n";
    permute("ABC", 0, 2);

    // Example 9: Inorder Traversal of Binary Tree
    cout << "Example 9: Inorder traversal of binary tree: ";
    Node* root = new Node(1); // Create root node
    root->left = new Node(2); // Left child
    root->right = new Node(3); // Right child
    root->left->left = new Node(4); // Left grandchild
    root->left->right = new Node(5); // Right grandchild
    inorderTraversal(root);
    cout << endl;

    // Example 10: Merge Sort
    int unsortedArr[] = {38, 27, 43, 3, 9, 82, 10};
    int arrSize = sizeof(unsortedArr) / sizeof(unsortedArr[0]);
    cout << "Example 10: Unsorted array: ";
    for (int i = 0; i < arrSize; i++) cout << unsortedArr[i] << " ";
    cout << endl;
    mergeSort(unsortedArr, 0, arrSize - 1);
    cout << "Sorted array: ";
    for (int i = 0; i < arrSize; i++) cout << unsortedArr[i] << " ";
    cout << endl;

    // Example 11: Fractal Generation (Simulated)
    cout << "Example 11: Drawing fractal of depth 2: ";
    drawFractal(2);
    cout << endl;

    // Example 12: Tail Recursion Example
    cout << "Example 12: Tail recursion counting down from 5: ";
    tailRecursion(5);
    cout << endl;

    // Example 13: Infinite Recursion (Commented out to prevent stack overflow)
    // infiniteRecursion(1); // ❌ This will cause a stack overflow error

    // Example 14: Mutual Recursion
    cout << "Example 14: Mutual recursion starting with functionA(10): ";
    functionA(10);
    cout << endl;

    // Example 15: Calculating Combinations
    int n = 5, k = 2;
    cout << "Example 15: Combination C(" << n << ", " << k << ") is " << combination(n, k) << endl;

    return 0;
}
/*
========================================
Key Points:
- Always define a base case to prevent infinite recursion 🛑🪞
- Ensure that recursive calls progress towards the base case ➡️🪞
- Be wary of stack overflow when using deep recursion 🚨
- Tail recursion can be optimized to iterations by compilers 🔄⚡️
- Recursion is elegant and readable for self-similar problems 🪞✨
- Iteration may be more memory efficient for simple repetitive tasks
========================================
*/