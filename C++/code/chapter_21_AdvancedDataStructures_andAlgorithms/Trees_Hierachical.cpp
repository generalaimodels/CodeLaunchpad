// Chapter 6: Trees - Hierarchical Structures: Family Trees & Organizations 🌳👪🏢

#include <iostream>
#include <queue>   // For level order traversal
#include <stack>   // For iterative traversals
#include <algorithm> // For max function
using namespace std;

/*
========================================
Concept: Non-linear Data Structures Representing Hierarchical Relationships 🌳🔗
- Trees represent hierarchical data with nodes connected by edges.
- Analogous to family trees 👪 or organizational charts 🏢.
========================================
*/

// Example 1: Defining a Basic Tree Node 🌳
struct TreeNode {
    int data;           // Node value 👤
    TreeNode* left;     // Left child ⬅️
    TreeNode* right;    // Right child ➡️

    TreeNode(int val) : data(val), left(nullptr), right(nullptr) {} // Constructor
};

// Example 2: Inserting Nodes into a Binary Tree 🌳➕
TreeNode* insertBinaryTree(TreeNode* root, int data) {
    if (root == nullptr) {
        root = new TreeNode(data); // Create root node if tree is empty 🌳⬆️
    } else {
        queue<TreeNode*> q;
        q.push(root);

        while (!q.empty()) {
            TreeNode* temp = q.front();
            q.pop();

            // Insert as left child if empty
            if (temp->left == nullptr) {
                temp->left = new TreeNode(data);
                return root;
            } else {
                q.push(temp->left);
            }

            // Insert as right child if empty
            if (temp->right == nullptr) {
                temp->right = new TreeNode(data);
                return root;
            } else {
                q.push(temp->right);
            }
        }
    }
    return root;
}

// Example 3: Tree Traversal - Inorder (Left-Root-Right) ⬅️🌳➡️
void inorderTraversal(TreeNode* root) {
    if (root == nullptr) return; // Base case 🛑

    inorderTraversal(root->left);    // Visit left subtree ⬅️
    cout << root->data << " ";       // Visit root 🌳
    inorderTraversal(root->right);   // Visit right subtree ➡️
}

// Example 4: Tree Traversal - Preorder (Root-Left-Right) 🌳⬅️➡️
void preorderTraversal(TreeNode* root) {
    if (root == nullptr) return; // Base case 🛑

    cout << root->data << " ";       // Visit root 🌳
    preorderTraversal(root->left);    // Visit left subtree ⬅️
    preorderTraversal(root->right);   // Visit right subtree ➡️
}

// Example 5: Tree Traversal - Postorder (Left-Right-Root) ⬅️➡️🌳
void postorderTraversal(TreeNode* root) {
    if (root == nullptr) return; // Base case 🛑

    postorderTraversal(root->left);    // Visit left subtree ⬅️
    postorderTraversal(root->right);   // Visit right subtree ➡️
    cout << root->data << " ";         // Visit root 🌳
}

// Example 6: Level Order Traversal (Breadth-First Search) 층별 🌳
void levelOrderTraversal(TreeNode* root) {
    if (root == nullptr) return; // Base case 🛑

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        TreeNode* temp = q.front();
        q.pop();
        cout << temp->data << " "; // Visit current node 👤

        if (temp->left != nullptr) {
            q.push(temp->left);    // Enqueue left child ⬅️
        }
        if (temp->right != nullptr) {
            q.push(temp->right);   // Enqueue right child ➡️
        }
    }
}

// Example 7: Computing the Height of a Tree 🌳🔝
int treeHeight(TreeNode* root) {
    if (root == nullptr) {
        return 0; // Empty tree has height 0 🛑
    } else {
        // Height is max of left/right subtree heights plus 1
        int leftHeight = treeHeight(root->left);   // Height of left subtree
        int rightHeight = treeHeight(root->right); // Height of right subtree
        return max(leftHeight, rightHeight) + 1;   // Max height plus root
    }
}

// Example 8: Counting the Number of Nodes in a Tree 🌳🔢
int countNodes(TreeNode* root) {
    if (root == nullptr) {
        return 0; // No nodes in an empty tree 🛑
    } else {
        // Total nodes = left subtree nodes + right subtree nodes + 1 (root)
        return countNodes(root->left) + countNodes(root->right) + 1;
    }
}

// Example 9: Searching for a Value in a Binary Tree 🔍🌳
bool searchBinaryTree(TreeNode* root, int key) {
    if (root == nullptr) {
        return false; // Key not found 🛑
    }
    if (root->data == key) {
        return true; // Key found 🌟
    }
    // Search in left and right subtrees
    return searchBinaryTree(root->left, key) || searchBinaryTree(root->right, key);
}

// Example 10: Deleting a Tree (Freeing Memory) ✨🗑️
void deleteTree(TreeNode* &root) {
    if (root == nullptr) {
        return; // Nothing to delete 🛑
    }
    // Delete left and right subtrees
    deleteTree(root->left);
    deleteTree(root->right);
    // Delete current node
    delete root;
    root = nullptr; // Avoid dangling pointer
}

// Example 11: Defining a Binary Search Tree (BST) 🌳🔍
struct BSTNode {
    int data;
    BSTNode* left;
    BSTNode* right;

    BSTNode(int val) : data(val), left(nullptr), right(nullptr) {} // Constructor
};

// Example 12: Inserting a Node into a BST ➕🌳
BSTNode* insertBST(BSTNode* root, int data) {
    if (root == nullptr) {
        root = new BSTNode(data); // Insert at root if tree is empty 🌳⬆️
    } else if (data <= root->data) {
        // Insert into left subtree if data is less or equal ⚖️⬅️
        root->left = insertBST(root->left, data);
    } else {
        // Insert into right subtree if data is greater ⚖️➡️
        root->right = insertBST(root->right, data);
    }
    return root;
}

// Example 13: Searching in a BST 🔍🌳
bool searchBST(BSTNode* root, int key) {
    if (root == nullptr) {
        return false; // Key not found 🛑
    }
    if (root->data == key) {
        return true; // Key found 🌟
    } else if (key < root->data) {
        // Search in left subtree ⬅️
        return searchBST(root->left, key);
    } else {
        // Search in right subtree ➡️
        return searchBST(root->right, key);
    }
}

// Example 14: Finding Minimum Value in a BST ⚖️🌳
int findMin(BSTNode* root) {
    if (root == nullptr) {
        cout << "Tree is empty" << endl;
        return -1;
    } else if (root->left == nullptr) {
        // Leftmost leaf is minimum 🍃
        return root->data;
    } else {
        // Go left to find minimum ⬅️
        return findMin(root->left);
    }
}

// Example 15: Deleting a Node from a BST ➖🌳
BSTNode* deleteNode(BSTNode* root, int key) {
    if (root == nullptr) {
        return root; // Key not found 🛑
    }
    if (key < root->data) {
        // Key is in left subtree ⬅️
        root->left = deleteNode(root->left, key);
    } else if (key > root->data) {
        // Key is in right subtree ➡️
        root->right = deleteNode(root->right, key);
    } else {
        // Node to be deleted found 🌟
        if (root->left == nullptr && root->right == nullptr) {
            // Case 1: No child
            delete root;
            root = nullptr;
        } else if (root->left == nullptr) {
            // Case 2: One child (right)
            BSTNode* temp = root;
            root = root->right;
            delete temp;
        } else if (root->right == nullptr) {
            // Case 2: One child (left)
            BSTNode* temp = root;
            root = root->left;
            delete temp;
        } else {
            // Case 3: Two children
            // Find minimum in right subtree
            int minValue = findMin(root->right);
            root->data = minValue;
            root->right = deleteNode(root->right, minValue);
        }
    }
    return root;
}

// Example 16: Checking if a Tree is a BST 🌳✅
bool isBSTUtil(BSTNode* root, int minValue, int maxValue) {
    if (root == nullptr) {
        return true; // Empty tree is BST 🛑
    }
    if (root->data < minValue || root->data > maxValue) {
        return false; // Violates BST property ❌
    }
    // Check subtrees recursively ⬅️➡️
    return isBSTUtil(root->left, minValue, root->data - 1) &&
           isBSTUtil(root->right, root->data + 1, maxValue);
}

bool isBST(BSTNode* root) {
    return isBSTUtil(root, INT_MIN, INT_MAX);
}

// Example 17: Tree Traversal - Inorder Iterative using Stack ⬅️🌳➡️🔄
void inorderIterative(TreeNode* root) {
    stack<TreeNode*> s;
    TreeNode* curr = root;

    while (curr != nullptr || !s.empty()) {
        while (curr != nullptr) {
            s.push(curr);           // Push current node to stack 👍
            curr = curr->left;      // Move to left child ⬅️
        }
        curr = s.top();
        s.pop();
        cout << curr->data << " ";  // Visit node 🌳
        curr = curr->right;         // Move to right child ➡️
    }
}

// Example 18: Constructing a Binary Tree from Inorder and Preorder Traversals 🧩🌳
int preIndex = 0;
int search(int arr[], int start, int end, int value) {
    for (int i = start; i <= end; i++) {
        if (arr[i] == value) return i;
    }
    return -1;
}

TreeNode* buildTree(int inorder[], int preorder[], int start, int end) {
    if (start > end) return nullptr; // Base case 🛑

    TreeNode* node = new TreeNode(preorder[preIndex++]); // Pick current node 🌳

    if (start == end) return node; // Leaf node 🍃

    int inIndex = search(inorder, start, end, node->data);

    // Build left and right subtrees recursively ⬅️➡️
    node->left = buildTree(inorder, preorder, start, inIndex - 1);
    node->right = buildTree(inorder, preorder, inIndex + 1, end);

    return node;
}

// Example 19: Printing All Paths from Root to Leaves 🛣️🌳
void printPaths(TreeNode* root, int path[], int pathLen) {
    if (root == nullptr) return; // Base case 🛑

    // Append current node to path array
    path[pathLen] = root->data;
    pathLen++;

    // If leaf node, print the path
    if (root->left == nullptr && root->right == nullptr) {
        for (int i = 0; i < pathLen; i++) {
            cout << path[i] << " ";
        }
        cout << endl;
    } else {
        // Recurse on left and right subtrees ⬅️➡️
        printPaths(root->left, path, pathLen);
        printPaths(root->right, path, pathLen);
    }
}

// Example 20: Calculating the Diameter of a Tree 🌳↔️
int diameter(TreeNode* root, int& height) {
    if (root == nullptr) {
        height = 0; // Base case 🛑
        return 0;   // Diameter is 0
    }
    int leftHeight = 0, rightHeight = 0;
    // Get diameters of left and right subtrees
    int leftDiameter = diameter(root->left, leftHeight);
    int rightDiameter = diameter(root->right, rightHeight);

    // Height of current node
    height = max(leftHeight, rightHeight) + 1;

    // Diameter is max of left, right, or path through root
    return max({leftHeight + rightHeight + 1, leftDiameter, rightDiameter});
}

// Main Function to Demonstrate Examples
int main() {
    // Example 1 & 2: Creating a Binary Tree 🌳
    TreeNode* root = nullptr;
    root = insertBinaryTree(root, 1);
    insertBinaryTree(root, 2);
    insertBinaryTree(root, 3);
    insertBinaryTree(root, 4);
    insertBinaryTree(root, 5);

    // Example 3: Inorder Traversal ⬅️🌳➡️
    cout << "Inorder Traversal: ";
    inorderTraversal(root);
    cout << endl;

    // Example 4: Preorder Traversal 🌳⬅️➡️
    cout << "Preorder Traversal: ";
    preorderTraversal(root);
    cout << endl;

    // Example 5: Postorder Traversal ⬅️➡️🌳
    cout << "Postorder Traversal: ";
    postorderTraversal(root);
    cout << endl;

    // Example 6: Level Order Traversal 층별 🌳
    cout << "Level Order Traversal: ";
    levelOrderTraversal(root);
    cout << endl;

    // Example 7: Computing Height of the Tree 🌳🔝
    int height = treeHeight(root);
    cout << "Height of the tree: " << height << endl;

    // Example 8: Counting Nodes in the Tree 🌳🔢
    int totalNodes = countNodes(root);
    cout << "Total number of nodes: " << totalNodes << endl;

    // Example 9: Searching for a Value in the Tree 🔍🌳
    int key = 4;
    bool found = searchBinaryTree(root, key);
    cout << "Value " << key << (found ? " found" : " not found") << " in the tree." << endl;

    // Example 10: Deleting the Tree ✨🗑️
    deleteTree(root);
    cout << "Tree deleted." << endl;

    // Example 11 & 12: Creating a BST and Inserting Nodes 🌳🔍
    BSTNode* bstRoot = nullptr;
    bstRoot = insertBST(bstRoot, 10);
    insertBST(bstRoot, 5);
    insertBST(bstRoot, 15);
    insertBST(bstRoot, 3);
    insertBST(bstRoot, 7);

    // Example 13: Searching in a BST 🔍🌳
    key = 7;
    found = searchBST(bstRoot, key);
    cout << "Value " << key << (found ? " found" : " not found") << " in the BST." << endl;

    // Example 14: Finding Minimum Value in BST ⚖️🌳
    int minValue = findMin(bstRoot);
    cout << "Minimum value in the BST: " << minValue << endl;

    // Example 15: Deleting a Node from BST ➖🌳
    bstRoot = deleteNode(bstRoot, 5);
    cout << "Node with value 5 deleted from BST." << endl;

    // Example 16: Checking if Tree is BST 🌳✅
    bool isBSTTree = isBST(bstRoot);
    cout << "The tree is " << (isBSTTree ? "a BST." : "not a BST.") << endl;

    // Example 17: Inorder Traversal Iterative ⬅️🌳➡️🔄
    // cout << "Inorder Iterative Traversal: ";
    // inorderIterative(bstRoot);
    // cout << endl;

    // Example 18: Constructing Tree from Inorder and Preorder 🧩🌳
    int inorderArr[] = {3, 5, 7, 10, 15};
    int preorderArr[] = {10, 5, 3, 7, 15};
    int length = sizeof(inorderArr) / sizeof(inorderArr[0]);
    preIndex = 0;
    TreeNode* constructedRoot = buildTree(inorderArr, preorderArr, 0, length - 1);
    cout << "Tree constructed from inorder and preorder traversals." << endl;

    // Example 19: Printing All Paths from Root to Leaves 🛣️🌳
    cout << "All root-to-leaf paths:" << endl;
    int path[1000];
    printPaths(constructedRoot, path, 0);

    // Example 20: Calculating Diameter of Tree 🌳↔️
    int treeDiaHeight = 0;
    int dia = diameter(constructedRoot, treeDiaHeight);
    cout << "Diameter of the tree: " << dia << endl;

    return 0;
}

/*
========================================
Key Points:
- Trees represent hierarchical data with nodes and edges 🌳🔗
- Binary Trees have nodes with at most two children (left and right) 🌳⬅️➡️
- BSTs have an ordering property for efficient operations 🌳🔍
- Tree traversals visit nodes in different orders (inorder, preorder, etc.) 🚶‍♂️🌳
- Always check for null pointers to avoid segmentation faults ❌
- Deleting nodes requires careful handling to maintain tree structure 🗑️🌳
- Edge cases include empty trees, single-node trees, and balancing issues
- Iterative traversals use stacks or queues to simulate recursion 🔄
- Applications include file systems, expression parsing, and more 📁🧮
========================================
*/