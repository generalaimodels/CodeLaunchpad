// Chapter 4: Stacks & Queues - Ordered Operations: Last-In-First-Out & First-In-First-Out ⬆️⬇️ 🚶‍♂️🍽️

#include <iostream>
#include <stack>      // For STL stack
#include <queue>      // For STL queue
#include <string>     // For string operations
#include <cctype>     // For isdigit function
#include <vector>     // For vector in BFS example

using namespace std;

// Example 1: Using STL stack to demonstrate basic stack operations ⬆️🍽️⬇️
void example1_stl_stack() {
    cout << "Example 1: Using STL stack\n";
    stack<int> plates; // Stack represents plates 🍽️

    // Push elements onto the stack ⬆️🍽️
    plates.push(1); // Adding plate 1
    plates.push(2); // Adding plate 2
    plates.push(3); // Adding plate 3

    cout << "Stack size after pushing plates: " << plates.size() << endl;

    // Peek at the top element 👀🍽️
    cout << "Top plate is: " << plates.top() << endl;

    // Pop elements from the stack ⬇️🍽️
    while (!plates.empty()) { // Check if stack is not empty 🍽️❓
        cout << "Removing plate: " << plates.top() << endl;
        plates.pop(); // Remove top plate
    }

    cout << "Stack is empty: " << plates.empty() << endl;
}

// Example 2: Implementing a stack using an array ⬆️🍽️⬇️
#define MAX_SIZE 100 // Maximum size of the stack

class ArrayStack {
private:
    int arr[MAX_SIZE];
    int top; // Index of the top element
public:
    ArrayStack() : top(-1) {} // Constructor initializes top to -1

    void push(int x) {
        if (top >= MAX_SIZE - 1) {
            cout << "Stack Overflow! Cannot add more plates.\n";
            return;
        }
        arr[++top] = x;
        cout << "Pushed plate " << x << " onto the stack.\n";
    }

    void pop() {
        if (top < 0) {
            cout << "Stack Underflow! No plates to remove.\n";
            return;
        }
        cout << "Popped plate " << arr[top--] << " from the stack.\n";
    }

    int peek() {
        if (top < 0) {
            cout << "Stack is empty.\n";
            return -1;
        }
        return arr[top];
    }

    bool isEmpty() {
        return top < 0;
    }
};

void example2_array_stack() {
    cout << "\nExample 2: Implementing a stack using an array\n";
    ArrayStack plates;

    plates.push(10);
    plates.push(20);
    plates.push(30);

    cout << "Top plate is: " << plates.peek() << endl;

    plates.pop();
    plates.pop();
    plates.pop();
    plates.pop(); // Should show underflow
}

// Example 3: Implementing a stack using a linked list ⬆️🍽️⬇️
class LinkedListStack {
private:
    struct Node {
        int data;
        Node* next;
    }*top;

public:
    LinkedListStack() : top(nullptr) {} // Constructor initializes top to nullptr

    void push(int x) {
        Node* temp = new Node;
        temp->data = x;
        temp->next = top;
        top = temp;
        cout << "Pushed plate " << x << " onto the stack.\n";
    }

    void pop() {
        if (top == nullptr) {
            cout << "Stack Underflow! No plates to remove.\n";
            return;
        }
        Node* temp = top;
        top = top->next;
        cout << "Popped plate " << temp->data << " from the stack.\n";
        delete temp;
    }

    int peek() {
        if (top == nullptr) {
            cout << "Stack is empty.\n";
            return -1;
        }
        return top->data;
    }

    bool isEmpty() {
        return top == nullptr;
    }
};

void example3_linkedlist_stack() {
    cout << "\nExample 3: Implementing a stack using a linked list\n";
    LinkedListStack plates;

    plates.push(100);
    plates.push(200);
    plates.push(300);

    cout << "Top plate is: " << plates.peek() << endl;

    plates.pop();
    plates.pop();
    plates.pop();
    plates.pop(); // Should show underflow
}

// Example 4: Checking for balanced parentheses using a stack 🧮➡️⬆️⬇️
bool areParenthesesBalanced(string expr) {
    stack<char> s;
    for (char& ch : expr) {
        if (ch == '(' || ch == '{' || ch == '[') {
            s.push(ch); // Push opening brackets onto stack
        } else if (ch == ')' || ch == '}' || ch == ']') {
            if (s.empty()) return false;
            char top = s.top();
            if ((ch == ')' && top != '(') ||
                (ch == '}' && top != '{') ||
                (ch == ']' && top != '[')) {
                return false;
            }
            s.pop(); // Pop the matching opening bracket
        }
    }
    return s.empty();
}

void example4_balanced_parentheses() {
    cout << "\nExample 4: Checking for balanced parentheses\n";
    string expr = "{[()]}";
    if (areParenthesesBalanced(expr))
        cout << "The expression " << expr << " is balanced.\n";
    else
        cout << "The expression " << expr << " is not balanced.\n";

    expr = "{[(])}";
    if (areParenthesesBalanced(expr))
        cout << "The expression " << expr << " is balanced.\n";
    else
        cout << "The expression " << expr << " is not balanced.\n";
}

// Example 5: Converting infix expression to postfix using a stack 🧮➡️⬆️⬇️
int getPrecedence(char op) {
    if (op == '+' || op == '-') return 1; // Lowest precedence
    if (op == '*' || op == '/') return 2; // Higher precedence
    return 0; // Non-operator
}

string infixToPostfix(string infix) {
    stack<char> s;
    string postfix = "";
    for (char& ch : infix) {
        if (isdigit(ch)) {
            postfix += ch; // Append operands to postfix expression
        } else if (ch == '(') {
            s.push(ch); // Push '(' onto stack
        } else if (ch == ')') {
            while (!s.empty() && s.top() != '(') {
                postfix += s.top(); // Append operators to postfix expression
                s.pop();
            }
            if (!s.empty() && s.top() == '(') s.pop(); // Pop '(' from stack
        } else { // Operator encountered
            while (!s.empty() && getPrecedence(ch) <= getPrecedence(s.top())) {
                postfix += s.top();
                s.pop();
            }
            s.push(ch); // Push current operator
        }
    }
    // Pop any remaining operators from the stack
    while (!s.empty()) {
        postfix += s.top();
        s.pop();
    }
    return postfix;
}

void example5_infix_to_postfix() {
    cout << "\nExample 5: Converting infix expression to postfix\n";
    string infix = "(1+2)*3";
    string postfix = infixToPostfix(infix);
    cout << "Infix expression: " << infix << endl;
    cout << "Postfix expression: " << postfix << endl;
}

// Example 6: Using STL queue to demonstrate basic queue operations ➕🚶‍♂️➡️
void example6_stl_queue() {
    cout << "\nExample 6: Using STL queue\n";
    queue<int> line; // Queue represents a line of people 🚶‍♂️

    // Enqueue elements into the queue ➕🚶‍♂️➡️
    line.push(1); // Person 1 joins the line
    line.push(2); // Person 2 joins the line
    line.push(3); // Person 3 joins the line

    cout << "Queue size after enqueuing: " << line.size() << endl;

    // Peek at the front element 👀🚶‍♂️
    cout << "Front of the line is person: " << line.front() << endl;

    // Dequeue elements from the queue ➖🚶‍♂️➡️
    while (!line.empty()) { // Check if queue is not empty 🚶‍♂️❓
        cout << "Serving person: " << line.front() << endl;
        line.pop(); // Serve and remove person from the front
    }

    cout << "Queue is empty: " << line.empty() << endl;
}

// Example 7: Implementing a queue using a circular array ➕🚶‍♂️➡️
class CircularQueue {
private:
    int front, rear, size;
    int* array;
    int capacity;

public:
    CircularQueue(int cap) : capacity(cap) {
        front = size = 0;
        rear = capacity - 1;
        array = new int[capacity];
    }

    ~CircularQueue() {
        delete[] array;
    }

    bool isFull() {
        return size == capacity;
    }

    bool isEmpty() {
        return size == 0;
    }

    void enqueue(int x) {
        if (isFull()) {
            cout << "Queue Overflow! Cannot add more people to the line.\n";
            return;
        }
        rear = (rear + 1) % capacity;
        array[rear] = x;
        size++;
        cout << "Person " << x << " joined the line.\n";
    }

    void dequeue() {
        if (isEmpty()) {
            cout << "Queue Underflow! No people to serve.\n";
            return;
        }
        cout << "Serving person " << array[front] << ".\n";
        front = (front + 1) % capacity;
        size--;
    }

    int Front() {
        if (isEmpty()) {
            cout << "Queue is empty.\n";
            return -1;
        }
        return array[front];
    }
};

void example7_circular_queue() {
    cout << "\nExample 7: Implementing a queue using a circular array\n";
    CircularQueue line(3);

    line.enqueue(10);
    line.enqueue(20);
    line.enqueue(30);
    line.enqueue(40); // Should show overflow

    cout << "Front of the line is person: " << line.Front() << endl;

    line.dequeue();
    line.dequeue();
    line.dequeue();
    line.dequeue(); // Should show underflow
}

// Example 8: Implementing a queue using a linked list ➕🚶‍♂️➡️
class LinkedListQueue {
private:
    struct Node {
        int data;
        Node* next;
    }*front, *rear;

public:
    LinkedListQueue() : front(nullptr), rear(nullptr) {} // Constructor

    void enqueue(int x) {
        Node* temp = new Node;
        temp->data = x;
        temp->next = nullptr;
        if (rear == nullptr) {
            front = rear = temp;
            cout << "Person " << x << " joined the line.\n";
            return;
        }
        rear->next = temp;
        rear = temp;
        cout << "Person " << x << " joined the line.\n";
    }

    void dequeue() {
        if (front == nullptr) {
            cout << "Queue Underflow! No people to serve.\n";
            return;
        }
        Node* temp = front;
        cout << "Serving person " << temp->data << ".\n";
        front = front->next;
        if (front == nullptr) rear = nullptr;
        delete temp;
    }

    int Front() {
        if (front == nullptr) {
            cout << "Queue is empty.\n";
            return -1;
        }
        return front->data;
    }

    bool isEmpty() {
        return front == nullptr;
    }
};

void example8_linkedlist_queue() {
    cout << "\nExample 8: Implementing a queue using a linked list\n";
    LinkedListQueue line;

    line.enqueue(100);
    line.enqueue(200);
    line.enqueue(300);

    cout << "Front of the line is person: " << line.Front() << endl;

    line.dequeue();
    line.dequeue();
    line.dequeue();
    line.dequeue(); // Should show underflow
}

// Example 9: BFS Traversal of a graph using a queue 🌐➡️🚶‍♂️🚶‍♀️
void example9_bfs_graph() {
    cout << "\nExample 9: BFS Traversal of a graph\n";

    int numVertices = 5;
    vector<int> adj[5];

    // Constructing the graph
    adj[0].push_back(1); // Edge from vertex 0 to 1
    adj[0].push_back(2); // Edge from vertex 0 to 2
    adj[1].push_back(3); // Edge from vertex 1 to 3
    adj[1].push_back(4); // Edge from vertex 1 to 4

    bool visited[5] = { false };

    queue<int> q;
    int startVertex = 0;
    visited[startVertex] = true;
    q.push(startVertex);

    cout << "BFS Traversal starting from vertex " << startVertex << ": ";
    while (!q.empty()) {
        int v = q.front();
        cout << v << " ";
        q.pop();

        // Enqueue all adjacent unvisited vertices
        for (int u : adj[v]) {
            if (!visited[u]) {
                visited[u] = true;
                q.push(u);
            }
        }
    }
    cout << endl;
}

// Example 10: Simulating a task scheduler using a queue ⏰➡️🚶‍♂️🚶‍♀️
struct Task {
    string name;
    int duration; // in seconds
};

void example10_task_scheduler() {
    cout << "\nExample 10: Simulating a simple task scheduler\n";
    queue<Task> taskQueue;

    // Adding tasks to the scheduler
    taskQueue.push({ "Task1", 3 });
    taskQueue.push({ "Task2", 2 });
    taskQueue.push({ "Task3", 1 });

    // Processing tasks in order
    while (!taskQueue.empty()) {
        Task currentTask = taskQueue.front();
        cout << "Processing " << currentTask.name << " which will take " << currentTask.duration << " seconds.\n";
        // Simulate task processing (in real scenario, we'd have actual work here)
        // Here, we'll just output
        taskQueue.pop();
    }

    cout << "All tasks have been processed.\n";
}

int main() {
    example1_stl_stack();
    example2_array_stack();
    example3_linkedlist_stack();
    example4_balanced_parentheses();
    example5_infix_to_postfix();

    example6_stl_queue();
    example7_circular_queue();
    example8_linkedlist_queue();
    example9_bfs_graph();
    example10_task_scheduler();

    return 0;
}