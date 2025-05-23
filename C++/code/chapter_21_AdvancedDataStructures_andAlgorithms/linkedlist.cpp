// Chapter 3: Linked Lists - Dynamic Chains: Flexible Connections 🔗🚂

// Include necessary headers
#include <iostream>  // For input/output operations

// Introduction to Linked Lists 🔗🚂
// Non-contiguous data elements linked together - Chains of Nodes 🔗🔗🔗
// Analogy: Train cars connected by couplings 🚂🔗

// Definition of a Node in a Singly Linked List
struct Node {
    int data;     // Data held by the node 🚃
    Node* next;   // Pointer to the next node 🔗
};

// Example 1: Creating a simple Singly Linked List and traversing it
void example1() {
    // Creating individual nodes
    Node* first = new Node();   // First train car 🚃
    Node* second = new Node();  // Second train car 🚃
    Node* third = new Node();   // Third train car 🚃

    // Assigning data to the nodes
    first->data = 10;
    second->data = 20;
    third->data = 30;

    // Linking the nodes together 🔗🔗🔗
    first->next = second;
    second->next = third;
    third->next = nullptr;  // Last node points to nullptr (end of the train)

    // Traversal (O(n)): Visiting each node sequentially 🚶‍♂️🚂
    Node* current = first;  // Start at the front of the train
    while (current != nullptr) {
        std::cout << current->data << " ";  // Output cargo data
        current = current->next;            // Move to the next car
    }
    std::cout << std::endl;

    // Clean up memory to prevent memory leaks 🧹
    delete first;
    delete second;
    delete third;
}

// Example 2: Inserting a node at the beginning (O(1)) ➕🚂➡️
void example2() {
    // Existing linked list: null (empty list)
    Node* head = nullptr;  // Head pointer to the first node

    // Inserting nodes at the beginning
    for (int i = 1; i <= 5; ++i) {
        Node* newNode = new Node();
        newNode->data = i * 10;
        newNode->next = head;  // New node points to the current head
        head = newNode;        // Head now points to the new node
    }

    // The list is now: 50 -> 40 -> 30 -> 20 -> 10

    // Traversal to display the list
    Node* current = head;
    std::cout << "Linked List after insertion at the beginning: ";
    while (current != nullptr) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << std::endl;

    // Clean up memory
    current = head;
    while (current != nullptr) {
        Node* temp = current;
        current = current->next;
        delete temp;
    }
}

// Example 3: Inserting a node at the end in a Singly Linked List (O(n)) 🚂➕➡️
void example3() {
    // Creating a linked list with one node
    Node* head = new Node();
    head->data = 10;
    head->next = nullptr;

    // Function to insert at the end
    auto insertAtEnd = [](Node* head, int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->next = nullptr;

        Node* current = head;
        while (current->next != nullptr) {  // Traverse to the end
            current = current->next;
        }
        current->next = newNode;  // Link the new node at the end
    };

    // Inserting additional nodes
    insertAtEnd(head, 20);
    insertAtEnd(head, 30);
    insertAtEnd(head, 40);
    insertAtEnd(head, 50);

    // Traversal to display the list
    Node* current = head;
    std::cout << "Linked List after insertion at the end: ";
    while (current != nullptr) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << std::endl;

    // Clean up memory
    current = head;
    while (current != nullptr) {
        Node* temp = current;
        current = current->next;
        delete temp;
    }
}

// Example 4: Deleting a node from the beginning (O(1)) ➖🚂➡️
void example4() {
    // Creating a linked list with three nodes
    Node* head = new Node();
    head->data = 10;
    head->next = new Node();
    head->next->data = 20;
    head->next->next = new Node();
    head->next->next->data = 30;
    head->next->next->next = nullptr;

    // Deleting the first node
    Node* temp = head;
    head = head->next;  // Head now points to the second node
    delete temp;        // Delete the old first node

    // Traversal to display the list
    Node* current = head;
    std::cout << "Linked List after deletion from the beginning: ";
    while (current != nullptr) {
        std::cout << current->data << " ";
        current = current->next;
    }
    std::cout << std::endl;

    // Clean up memory
    current = head;
    while (current != nullptr) {
        temp = current;
        current = current->next;
        delete temp;
    }
}

// Example 5: Searching for a node with a specific value (O(n)) 🔍🚂
void example5() {
    // Creating a linked list
    Node* head = nullptr;
    for (int i = 1; i <= 5; ++i) {
        Node* newNode = new Node();
        newNode->data = i * 10;
        newNode->next = head;
        head = newNode;
    }

    // Searching for a value
    int searchValue = 30;
    Node* current = head;
    bool found = false;
    while (current != nullptr) {
        if (current->data == searchValue) {
            found = true;
            break;
        }
        current = current->next;
    }
    if (found) {
        std::cout << "Value " << searchValue << " found in the list." << std::endl;
    } else {
        std::cout << "Value " << searchValue << " not found in the list." << std::endl;
    }

    // Clean up memory
    current = head;
    while (current != nullptr) {
        Node* temp = current;
        current = current->next;
        delete temp;
    }
}

// Types of Linked Lists

// 1. Singly Linked List ➡️🔗➡️🔗➡️
// Nodes point to the next node only. One-way train track.

// Definition of Singly Linked List with basic operations
class SinglyLinkedList {
public:
    struct Node {
        int data;
        Node* next;
    };

    Node* head;

    SinglyLinkedList() : head(nullptr) {}

    // Insertion at the end
    void insertAtEnd(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->next = nullptr;
        if (head == nullptr) {
            head = newNode;  // The list was empty
        } else {
            Node* current = head;
            while (current->next != nullptr) {
                current = current->next;
            }
            current->next = newNode;
        }
    }

    // Display the list
    void display() {
        Node* current = head;
        std::cout << "Singly Linked List: ";
        while (current != nullptr) {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    // Destructor to clean up memory
    ~SinglyLinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* temp = current;
            current = current->next;
            delete temp;
        }
    }
};

// 2. Doubly Linked List ⬅️🔗➡️🔗⬅️🔗➡️
// Nodes point to both the next and previous nodes. Two-way train track.

// Definition of Doubly Linked List
class DoublyLinkedList {
public:
    struct Node {
        int data;
        Node* prev;
        Node* next;
    };

    Node* head;

    DoublyLinkedList() : head(nullptr) {}

    // Insertion at the beginning (O(1))
    void insertAtBeginning(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->prev = nullptr;
        newNode->next = head;
        if (head != nullptr) {
            head->prev = newNode;  // Update previous head's prev pointer
        }
        head = newNode;
    }

    // Deletion from the end (O(1) if tail pointer is used)
    void deleteFromEnd() {
        if (head == nullptr) {
            return;  // List is empty
        }
        Node* current = head;
        while (current->next != nullptr) {
            current = current->next;
        }
        if (current->prev != nullptr) {
            current->prev->next = nullptr;
        } else {
            head = nullptr;  // List had only one node
        }
        delete current;
    }

    // Display the list
    void displayForward() {
        Node* current = head;
        std::cout << "Doubly Linked List (forward): ";
        while (current != nullptr) {
            std::cout << current->data << " ";
            if (current->next == nullptr) {
                break;  // To use current for backward display
            }
            current = current->next;
        }
        std::cout << std::endl;

        // Display backward
        std::cout << "Doubly Linked List (backward): ";
        while (current != nullptr) {
            std::cout << current->data << " ";
            current = current->prev;
        }
        std::cout << std::endl;
    }

    // Destructor to clean up memory
    ~DoublyLinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* temp = current;
            current = current->next;
            delete temp;
        }
    }
};

// 3. Circular Linked List 🔄🔗🔄🔗🔄
// Last node points back to the first node. Circular train track.

// Definition of Circular Singly Linked List
class CircularLinkedList {
public:
    struct Node {
        int data;
        Node* next;
    };

    Node* tail;  // Tail pointer for easy insertion at end

    CircularLinkedList() : tail(nullptr) {}

    // Insertion at the end
    void insert(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        if (tail == nullptr) {
            tail = newNode;
            tail->next = tail;  // Points to itself
        } else {
            newNode->next = tail->next;
            tail->next = newNode;
            tail = newNode;  // Update tail to new node
        }
    }

    // Display the list
    void display() {
        if (tail == nullptr) {
            std::cout << "Circular Linked List is empty." << std::endl;
            return;
        }
        Node* current = tail->next;  // Start from head
        std::cout << "Circular Linked List: ";
        do {
            std::cout << current->data << " ";
            current = current->next;
        } while (current != tail->next);
        std::cout << std::endl;
    }

    // Destructor to clean up memory
    ~CircularLinkedList() {
        if (tail == nullptr) {
            return;  // List is empty
        }
        Node* current = tail->next;
        tail->next = nullptr;  // Break the circle to avoid infinite loop
        while (current != nullptr) {
            Node* temp = current;
            current = current->next;
            delete temp;
        }
    }
};

// Linked List Operations

// Example 6: Insertion at a specific position (O(n)) 🚂➡️➕🚂➡️
void example6() {
    // Creating a linked list
    SinglyLinkedList list;
    list.insertAtEnd(10);
    list.insertAtEnd(20);
    list.insertAtEnd(40);

    // Function to insert at a specific position
    auto insertAtPosition = [](SinglyLinkedList::Node*& head, int data, int position) {
        SinglyLinkedList::Node* newNode = new SinglyLinkedList::Node();
        newNode->data = data;
        if (position == 0) {
            newNode->next = head;
            head = newNode;
            return;
        }
        SinglyLinkedList::Node* current = head;
        for (int i = 0; i < position - 1 && current != nullptr; ++i) {
            current = current->next;
        }
        if (current == nullptr) {
            std::cout << "Position out of bounds." << std::endl;
            delete newNode;
            return;
        }
        newNode->next = current->next;
        current->next = newNode;
    };

    // Inserting 30 at position 2
    insertAtPosition(list.head, 30, 2);

    // Display the list
    list.display();
}

// Example 7: Deletion from a specific position (O(n)) 🚂➡️➖🚂➡️
void example7() {
    // Creating a linked list
    SinglyLinkedList list;
    for (int i = 1; i <= 5; ++i) {
        list.insertAtEnd(i * 10);
    }

    // Function to delete from a specific position
    auto deleteFromPosition = [](SinglyLinkedList::Node*& head, int position) {
        if (head == nullptr) {
            std::cout << "List is empty." << std::endl;
            return;
        }
        SinglyLinkedList::Node* temp = nullptr;
        if (position == 0) {
            temp = head;
            head = head->next;
            delete temp;
            return;
        }
        SinglyLinkedList::Node* current = head;
        for (int i = 0; i < position - 1 && current->next != nullptr; ++i) {
            current = current->next;
        }
        if (current->next == nullptr) {
            std::cout << "Position out of bounds." << std::endl;
            return;
        }
        temp = current->next;
        current->next = current->next->next;
        delete temp;
    };

    // Deleting node at position 2
    deleteFromPosition(list.head, 2);

    // Display the list
    list.display();
}

// Applications of Linked Lists

// Example 8: Implementing a Stack using Singly Linked List (LIFO) 📦
// Stack implementation
class StackLinkedList {
public:
    struct Node {
        int data;
        Node* next;
    };

    Node* top;

    StackLinkedList() : top(nullptr) {}

    // Push operation (O(1))
    void push(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->next = top;
        top = newNode;
    }

    // Pop operation (O(1))
    void pop() {
        if (top == nullptr) {
            std::cout << "Stack Underflow." << std::endl;
            return;
        }
        Node* temp = top;
        top = top->next;
        std::cout << "Popped: " << temp->data << std::endl;
        delete temp;
    }

    // Display stack
    void display() {
        Node* current = top;
        std::cout << "Stack: ";
        while (current != nullptr) {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    // Destructor
    ~StackLinkedList() {
        while (top != nullptr) {
            pop();
        }
    }
};

// Example 9: Implementing a Queue using Singly Linked List (FIFO) 🚶‍♂️
// Queue implementation
class QueueLinkedList {
public:
    struct Node {
        int data;
        Node* next;
    };

    Node* front;
    Node* rear;

    QueueLinkedList() : front(nullptr), rear(nullptr) {}

    // Enqueue operation (O(1))
    void enqueue(int data) {
        Node* newNode = new Node();
        newNode->data = data;
        newNode->next = nullptr;
        if (rear == nullptr) {
            front = rear = newNode;  // First element
        } else {
            rear->next = newNode;
            rear = newNode;
        }
    }

    // Dequeue operation (O(1))
    void dequeue() {
        if (front == nullptr) {
            std::cout << "Queue Underflow." << std::endl;
            return;
        }
        Node* temp = front;
        front = front->next;
        std::cout << "Dequeued: " << temp->data << std::endl;
        delete temp;
        if (front == nullptr) {
            rear = nullptr;  // Queue is empty now
        }
    }

    // Display queue
    void display() {
        Node* current = front;
        std::cout << "Queue: ";
        while (current != nullptr) {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    // Destructor
    ~QueueLinkedList() {
        while (front != nullptr) {
            dequeue();
        }
    }
};

// Example 10: Representing a Polynomial using a Linked List 📈
struct PolyNode {
    int coefficient;
    int exponent;
    PolyNode* next;
};

// Function to add a new term to the polynomial
void insertTerm(PolyNode*& poly, int coeff, int exp) {
    PolyNode* newNode = new PolyNode();
    newNode->coefficient = coeff;
    newNode->exponent = exp;
    newNode->next = nullptr;
    if (poly == nullptr || exp > poly->exponent) {
        newNode->next = poly;
        poly = newNode;
    } else {
        PolyNode* current = poly;
        while (current->next != nullptr && current->next->exponent >= exp) {
            current = current->next;
        }
        newNode->next = current->next;
        current->next = newNode;
    }
}

// Function to display the polynomial
void displayPolynomial(PolyNode* poly) {
    PolyNode* current = poly;
    std::cout << "Polynomial: ";
    while (current != nullptr) {
        std::cout << current->coefficient << "x^" << current->exponent;
        if (current->next != nullptr) {
            std::cout << " + ";
        }
        current = current->next;
    }
    std::cout << std::endl;
}

// Clean up polynomial linked list
void deletePolynomial(PolyNode*& poly) {
    PolyNode* current = poly;
    while (current != nullptr) {
        PolyNode* temp = current;
        current = current->next;
        delete temp;
    }
    poly = nullptr;
}

// Main function to run examples
int main() {
    // Run Example 1
    std::cout << "Example 1: Creating and Traversing a Singly Linked List" << std::endl;
    example1();

    // Run Example 2
    std::cout << "\nExample 2: Insertion at the Beginning" << std::endl;
    example2();

    // Run Example 3
    std::cout << "\nExample 3: Insertion at the End" << std::endl;
    example3();

    // Run Example 4
    std::cout << "\nExample 4: Deletion from the Beginning" << std::endl;
    example4();

    // Run Example 5
    std::cout << "\nExample 5: Searching for a Value" << std::endl;
    example5();

    // Demonstrate Singly Linked List
    std::cout << "\nDemonstrating Singly Linked List" << std::endl;
    SinglyLinkedList sList;
    sList.insertAtEnd(5);
    sList.insertAtEnd(15);
    sList.insertAtEnd(25);
    sList.display();

    // Demonstrate Doubly Linked List
    std::cout << "\nDemonstrating Doubly Linked List" << std::endl;
    DoublyLinkedList dList;
    dList.insertAtBeginning(100);
    dList.insertAtBeginning(200);
    dList.insertAtBeginning(300);
    dList.displayForward();
    dList.deleteFromEnd();
    dList.displayForward();

    // Demonstrate Circular Linked List
    std::cout << "\nDemonstrating Circular Linked List" << std::endl;
    CircularLinkedList cList;
    cList.insert(1);
    cList.insert(2);
    cList.insert(3);
    cList.display();

    // Run Example 6
    std::cout << "\nExample 6: Insertion at a Specific Position" << std::endl;
    example6();

    // Run Example 7
    std::cout << "\nExample 7: Deletion from a Specific Position" << std::endl;
    example7();

    // Run Example 8: Stack using Linked List
    std::cout << "\nExample 8: Implementing a Stack using Linked List" << std::endl;
    StackLinkedList stack;
    stack.push(50);
    stack.push(60);
    stack.push(70);
    stack.display();
    stack.pop();
    stack.display();

    // Run Example 9: Queue using Linked List
    std::cout << "\nExample 9: Implementing a Queue using Linked List" << std::endl;
    QueueLinkedList queue;
    queue.enqueue(100);
    queue.enqueue(200);
    queue.enqueue(300);
    queue.display();
    queue.dequeue();
    queue.display();

    // Run Example 10: Representing a Polynomial
    std::cout << "\nExample 10: Representing a Polynomial using Linked List" << std::endl;
    PolyNode* polynomial = nullptr;
    insertTerm(polynomial, 5, 2);  // 5x^2
    insertTerm(polynomial, 3, 4);  // 3x^4
    insertTerm(polynomial, 2, 3);  // 2x^3
    displayPolynomial(polynomial);
    deletePolynomial(polynomial);

    return 0;  // End of the program
}