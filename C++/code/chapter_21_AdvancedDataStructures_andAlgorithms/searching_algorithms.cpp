//=========================================================================
// Searching Algorithms in C++
// Explaining various searching algorithms with examples.
// Author: OpenAI Assistant
//=========================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <list>
#include <unordered_map>
#include <ctime>

using namespace std;

// Example 1: Linear Search in an Array 🚶‍♂️📚
int linearSearch(const vector<int>& arr, int target) {
    // Traverse the array sequentially
    for (size_t i = 0; i < arr.size(); ++i) {
        if (arr[i] == target) {
            return i; // Target found at index i 🎯
        }
    }
    return -1; // Target not found ❌
}

// Example 2: Binary Search in a Sorted Array 📖✂️✂️
int binarySearch(const vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2; // Avoid overflow 🚫💥
        if (arr[mid] == target) {
            return mid; // Target found at index mid 🎯
        }
        if (arr[mid] < target) {
            left = mid + 1; // Search right half 🔜
        } else {
            right = mid - 1; // Search left half 🔙
        }
    }
    return -1; // Target not found ❌
}

// Example 3: Jump Search in a Sorted Array 📚➡️➡️🚶‍♂️
int jumpSearch(const vector<int>& arr, int target) {
    int n = arr.size();
    int step = sqrt(n); // Block size to jump 🔢
    int prev = 0;

    // Finding the block where the target may be present
    while (arr[min(step, n) - 1] < target) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) {
            return -1; // Target not found ❌
        }
    }

    // Linear search within the block
    for (int i = prev; i < min(step, n); ++i) {
        if (arr[i] == target) {
            return i; // Target found at index i 🎯
        }
    }
    return -1; // Target not found ❌
}

// Example 4: Interpolation Search in Uniformly Distributed Sorted Array 📖🧠
int interpolationSearch(const vector<int>& arr, int target) {
    int low = 0;
    int high = arr.size() - 1;

    while (low <= high && target >= arr[low] && target <= arr[high]) {
        // Estimate the position ⚖️
        int pos = low + ((double)(high - low) / (arr[high] - arr[low])) * (target - arr[low]);

        if (arr[pos] == target) {
            return pos; // Target found at position pos 🎯
        }
        if (arr[pos] < target) {
            low = pos + 1; // Move right 🔜
        } else {
            high = pos - 1; // Move left 🔙
        }
    }
    return -1; // Target not found ❌
}

// Example 5: Linear Search in a Linked List 🔗🔍
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* linearSearchLinkedList(ListNode* head, int target) {
    ListNode* current = head;
    while (current != nullptr) {
        if (current->val == target) {
            return current; // Target found 🎯
        }
        current = current->next; // Move to next node 🔜
    }
    return nullptr; // Target not found ❌
}

// Example 6: Binary Search Tree (BST) Node 🌳
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Example 7: Search in a Binary Search Tree 🌳🔍
TreeNode* searchBST(TreeNode* root, int target) {
    while (root != nullptr && root->val != target) {
        if (target < root->val) {
            root = root->left; // Go left 🔙
        } else {
            root = root->right; // Go right 🔜
        }
    }
    return root; // Returns the node if found 🎯 or nullptr if not ❌
}

// Example 8: Search in a Hash Table using Unordered Map 🔑📖⚡️
int searchHashTable(const unordered_map<int, int>& hashmap, int key) {
    auto it = hashmap.find(key);
    if (it != hashmap.end()) {
        return it->second; // Value associated with the key 🎯
    }
    return -1; // Key not found ❌
}

// Example 9: Ternary Search in a Sorted Array (Divide into 3 parts) 🔍🔍🔍
int ternarySearch(const vector<int>& arr, int left, int right, int target) {
    if (right >= left) {
        int mid1 = left + (right - left) / 3; // First partition 📍
        int mid2 = right - (right - left) / 3; // Second partition 📍

        if (arr[mid1] == target) {
            return mid1; // Target found at mid1 🎯
        }
        if (arr[mid2] == target) {
            return mid2; // Target found at mid2 🎯
        }
        if (target < arr[mid1]) {
            return ternarySearch(arr, left, mid1 - 1, target); // Search in first third 🔙
        } else if (target > arr[mid2]) {
            return ternarySearch(arr, mid2 + 1, right, target); // Search in third third 🔜
        } else {
            return ternarySearch(arr, mid1 + 1, mid2 - 1, target); // Search in middle third ➡️
        }
    }
    return -1; // Target not found ❌
}

// Example 10: Fibonacci Search in a Sorted Array 🐑🔍
int fibonacciSearch(const vector<int>& arr, int target) {
    int n = arr.size();
    // Initialize fibonacci numbers
    int fibMMm2 = 0; // (m-2)'th Fibonacci No.
    int fibMMm1 = 1; // (m-1)'th Fibonacci No.
    int fibM = fibMMm2 + fibMMm1; // m'th Fibonacci

    // fibM is the smallest Fibonacci number greater or equal to n
    while (fibM < n) {
        fibMMm2 = fibMMm1;
        fibMMm1 = fibM;
        fibM  = fibMMm2 + fibMMm1;
    }

    // Marks the eliminated range from front
    int offset = -1;

    // While there are elements to be inspected
    while (fibM > 1) {
        // Check if fibMMm2 is a valid location
        int i = min(offset + fibMMm2, n - 1);

        if (arr[i] < target) {
            fibM  = fibMMm1;
            fibMMm1 = fibMMm2;
            fibMMm2 = fibM - fibMMm1;
            offset = i; // Move offset to i
        } else if (arr[i] > target) {
            fibM  = fibMMm2;
            fibMMm1 = fibMMm1 - fibMMm2;
            fibMMm2 = fibM - fibMMm1;
        } else {
            return i; // Target found 🎯
        }
    }

    // Comparing the last element with target
    if (fibMMm1 && arr[offset + 1] == target) {
        return offset + 1; // Target found 🎯
    }

    return -1; // Target not found ❌
}

// Example 11: Exponential Search in a Sorted Array 🚀🔍
int exponentialSearch(const vector<int>& arr, int target) {
    if (arr.empty()) return -1;
    if (arr[0] == target) return 0; // Target found at index 0 🎯

    int i = 1;
    // Find range for binary search by repeated doubling
    while (i < arr.size() && arr[i] <= target) {
        i *= 2;
    }
    // Call binary search for the found range
    return binarySearch(vector<int>(arr.begin() + i / 2, arr.begin() + min(i, (int)arr.size())), target);
}

// Example 12: Sublist Search (Search a linked list in another) 🔗🔗🔍
bool isSublist(ListNode* list, ListNode* sublist) {
    if (!sublist) return true; // Empty sublist is always a sublist ✔️
    if (!list) return false; // Non-empty sublist can't be found in empty list ❌

    ListNode* ptr1 = list;
    ListNode* ptr2 = sublist;

    while (ptr1) {
        if (ptr1->val == ptr2->val) {
            ListNode* temp1 = ptr1;
            ListNode* temp2 = ptr2;
            while (temp1 && temp2 && temp1->val == temp2->val) {
                temp1 = temp1->next;
                temp2 = temp2->next;
            }
            if (!temp2) return true; // Sublist found 🎯
        }
        ptr1 = ptr1->next; // Move to next node 🔜
    }
    return false; // Sublist not found ❌
}

// Example 13: Searching in a 2D Matrix 🔍📐
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    int rows = matrix.size();
    int cols = matrix[0].size();
    int left = 0;
    int right = rows * cols - 1;

    // Treat 2D matrix as 1D array for binary search
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int mid_value = matrix[mid / cols][mid % cols];
        if (mid_value == target) {
            return true; // Target found 🎯
        } else if (mid_value < target) {
            left = mid + 1; // Search right half 🔜
        } else {
            right = mid - 1; // Search left half 🔙
        }
    }
    return false; // Target not found ❌
}

// Example 14: Search in Rotated Sorted Array 🔄🔍
int searchRotatedArray(const vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() -1;

    while (left <= right) {
        int mid = left + (right - left)/2;
        if (nums[mid] == target) return mid; // Target found 🎯

        // Determine which half is sorted
        if (nums[left] <= nums[mid]) {
            // Left half is sorted
            if (nums[left] <= target && target < nums[mid]) {
                right = mid -1; // Target in left half 🔙
            } else {
                left = mid +1; // Target in right half 🔜
            }
        } else {
            // Right half is sorted
            if (nums[mid] < target && target <= nums[right]) {
                left = mid +1; // Target in right half 🔜
            } else {
                right = mid -1; // Target in left half 🔙
            }
        }
    }
    return -1; // Target not found ❌
}

// Example 15: Search in a Trie Data Structure 🌳🔍
struct TrieNode {
    TrieNode* children[26];
    bool isEndOfWord;
    TrieNode() : isEndOfWord(false) {
        fill_n(children, 26, nullptr);
    }
};

void insertTrie(TrieNode* root, const string& key) {
    TrieNode* node = root;
    for (char c : key) {
        int index = c - 'a';
        if (!node->children[index]) {
            node->children[index] = new TrieNode(); // Create node if doesn't exist 🌱
        }
        node = node->children[index]; // Move to child node 🔜
    }
    node->isEndOfWord = true; // Mark end of word ✔️
}

bool searchTrie(TrieNode* root, const string& key) {
    TrieNode* node = root;
    for (char c : key) {
        int index = c - 'a';
        if (!node->children[index]) {
            return false; // Key not found ❌
        }
        node = node->children[index]; // Move to child node 🔜
    }
    return node->isEndOfWord; // Return true if end of word 🎯
}

// Main function to demonstrate examples
int main() {
    // Example usage of Linear Search
    vector<int> arr = {4, 2, 5, 1, 3};
    int target = 5;
    int index = linearSearch(arr, target);
    cout << "Linear Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Binary Search
    vector<int> sortedArr = {1, 2, 3, 4, 5};
    target = 3;
    index = binarySearch(sortedArr, target);
    cout << "Binary Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Jump Search
    target = 4;
    index = jumpSearch(sortedArr, target);
    cout << "Jump Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Interpolation Search
    vector<int> uniformArr = {10, 20, 30, 40, 50};
    target = 30;
    index = interpolationSearch(uniformArr, target);
    cout << "Interpolation Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Linear Search in Linked List
    ListNode* head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    ListNode* resultNode = linearSearchLinkedList(head, 2);
    cout << "Linked List Search: Element " << 2 << (resultNode ? " found." : " not found.") << endl;

    // Example usage of Search in BST
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(5);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);
    TreeNode* foundNode = searchBST(root, 3);
    cout << "BST Search: Element " << 3 << (foundNode ? " found." : " not found.") << endl;

    // Example usage of Search in Hash Table
    unordered_map<int, int> hashmap = {{1, 100}, {2, 200}, {3, 300}};
    int value = searchHashTable(hashmap, 2);
    cout << "Hash Table Search: Key " << 2 << (value != -1 ? " found with value " : " not found ") << value << endl;

    // Example usage of Ternary Search
    target = 4;
    index = ternarySearch(sortedArr, 0, sortedArr.size() -1, target);
    cout << "Ternary Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Fibonacci Search
    target = 5;
    index = fibonacciSearch(sortedArr, target);
    cout << "Fibonacci Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Exponential Search
    target = 3;
    index = exponentialSearch(sortedArr, target);
    cout << "Exponential Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Sublist Search
    ListNode* sublist = new ListNode(2);
    sublist->next = new ListNode(3);
    bool isSub = isSublist(head, sublist);
    cout << "Sublist Search: Sublist " << (isSub ? "found." : "not found.") << endl;

    // Example usage of Search in 2D Matrix
    vector<vector<int>> matrix = {{1, 3, 5}, {7, 9, 11}, {13, 15, 17}};
    bool found = searchMatrix(matrix, 9);
    cout << "2D Matrix Search: Element " << 9 << (found ? " found." : " not found.") << endl;

    // Example usage of Search in Rotated Sorted Array
    vector<int> rotatedArr = {4,5,6,7,0,1,2};
    target = 0;
    index = searchRotatedArray(rotatedArr, target);
    cout << "Rotated Array Search: Element " << target << (index != -1 ? " found at index " : " not found ") << index << endl;

    // Example usage of Search in Trie
    TrieNode* trieRoot = new TrieNode();
    insertTrie(trieRoot, "hello");
    insertTrie(trieRoot, "world");
    bool isFound = searchTrie(trieRoot, "world");
    cout << "Trie Search: Word 'world' " << (isFound ? "found." : "not found.") << endl;

    // Clean up dynamically allocated memory
    delete head->next->next;
    delete head->next;
    delete head;
    delete sublist->next;
    delete sublist;
    // Note: In a production environment, ensure to delete all dynamically allocated nodes.

    return 0;
}