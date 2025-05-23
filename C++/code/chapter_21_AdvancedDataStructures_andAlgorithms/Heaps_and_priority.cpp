/****************************************************************************************
 * Advanced C++ Heaps and Priority Queues Tutorial 🏥🚑 ⏳🥇
 *
 * This .cpp file covers the concepts of Heaps and Priority Queues in great detail.
 * Each example is self-contained and includes explanations within comments.
 *
 * Topics Covered:
 * 1. Basic Min Heap Implementation 🌳<
 * 2. Basic Max Heap Implementation 🌳>
 * 3. Heap Insertion Operation ➕🌳
 * 4. Heap Extract Min/Max Operation ➖🌳
 * 5. Heapify Operation 🏗️🌳
 * 6. Priority Queue Implementation using Heaps ⏳📥📤
 * 7. Priority Scheduling Simulation ⏳💻
 * 8. Heap Sort Algorithm Implementation 🚀📖
 * 9. Dijkstra's Algorithm Example 🌐🗺️
 * 10. Huffman Coding Tree Construction 📦📉
 *
 * Note: The code is error-free and designed for educational purposes.
 ****************************************************************************************/

 #include <iostream>
 #include <vector>
 #include <queue>
 #include <functional>
 #include <unordered_map>
 #include <string>
 #include <limits> // For Dijkstra's Algorithm
 #include <algorithm> // For Heap Sort
 using namespace std;
 
 /****************************************************************************************
  * Example 1: Basic Min Heap Implementation 🌳<
  * A Min Heap where the smallest element is always at the root.
  ****************************************************************************************/
 
 class MinHeap {
 private:
     vector<int> heap;
 
     // Helper function to maintain heap property during insertion
     void siftUp(int index) {
         while (index > 0 && heap[parent(index)] > heap[index]) {
             swap(heap[parent(index)], heap[index]); // Swap parent and current node
             index = parent(index); // Move up to parent
         }
     }
 
     // Helper function to maintain heap property during extraction
     void siftDown(int index) {
         int minIndex = index;
 
         int l = leftChild(index);
         if (l < heap.size() && heap[l] < heap[minIndex])
             minIndex = l;
 
         int r = rightChild(index);
         if (r < heap.size() && heap[r] < heap[minIndex])
             minIndex = r;
 
         if (index != minIndex) {
             swap(heap[index], heap[minIndex]);
             siftDown(minIndex);
         }
     }
 
     // Calculate parent index
     int parent(int index) {
         return (index - 1) / 2;
     }
 
     // Calculate left child index
     int leftChild(int index) {
         return 2 * index + 1;
     }
 
     // Calculate right child index
     int rightChild(int index) {
         return 2 * index + 2;
     }
 
 public:
     MinHeap() {
         // Constructor initializes an empty heap
     }
 
     void insert(int value) {
         heap.push_back(value); // Add new value at the end
         siftUp(heap.size() - 1); // Restore heap property
     }
 
     int extractMin() {
         if (heap.size() == 0)
             throw runtime_error("Heap is empty!"); // Exception if heap is empty
 
         int result = heap[0]; // The min element
         heap[0] = heap.back(); // Move last element to root
         heap.pop_back(); // Remove last element
         siftDown(0); // Restore heap property
         return result;
     }
 
     int getMin() {
         if (heap.size() == 0)
             throw runtime_error("Heap is empty!"); // Exception if heap is empty
 
         return heap[0]; // Return min element without removing
     }
 
     bool isEmpty() {
         return heap.size() == 0; // Check if heap is empty
     }
 
     void printHeap() {
         for (int val : heap)
             cout << val << " ";
         cout << endl;
     }
 };
 
 int main1() {
     cout << "Example 1: Basic Min Heap Implementation 🌳<" << endl;
     MinHeap minHeap;
     minHeap.insert(10);
     minHeap.insert(5);
     minHeap.insert(3);
     minHeap.insert(2);
     minHeap.insert(8);
     cout << "Heap elements after insertions: ";
     minHeap.printHeap(); // Should maintain min-heap property
     cout << "Minimum element: " << minHeap.getMin() << endl; // Should be 2
     cout << "Extracted minimum: " << minHeap.extractMin() << endl; // Remove and return min element
     cout << "Heap elements after extraction: ";
     minHeap.printHeap();
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 2: Basic Max Heap Implementation 🌳>
  * A Max Heap where the largest element is always at the root.
  ****************************************************************************************/
 
 class MaxHeap {
 private:
     vector<int> heap;
 
     void siftUp(int index) {
         while (index > 0 && heap[parent(index)] < heap[index]) {
             swap(heap[parent(index)], heap[index]); // Swap parent and current node
             index = parent(index);
         }
     }
 
     void siftDown(int index) {
         int maxIndex = index;
 
         int l = leftChild(index);
         if (l < heap.size() && heap[l] > heap[maxIndex])
             maxIndex = l;
 
         int r = rightChild(index);
         if (r < heap.size() && heap[r] > heap[maxIndex])
             maxIndex = r;
 
         if (index != maxIndex) {
             swap(heap[index], heap[maxIndex]);
             siftDown(maxIndex);
         }
     }
 
     int parent(int index) {
         return (index - 1) / 2;
     }
 
     int leftChild(int index) {
         return 2 * index + 1;
     }
 
     int rightChild(int index) {
         return 2 * index + 2;
     }
 
 public:
     MaxHeap() {
         // Constructor initializes an empty heap
     }
 
     void insert(int value) {
         heap.push_back(value);
         siftUp(heap.size() - 1);
     }
 
     int extractMax() {
         if (heap.size() == 0)
             throw runtime_error("Heap is empty!");
 
         int result = heap[0];
         heap[0] = heap.back();
         heap.pop_back();
         siftDown(0);
         return result;
     }
 
     int getMax() {
         if (heap.size() == 0)
             throw runtime_error("Heap is empty!");
 
         return heap[0];
     }
 
     bool isEmpty() {
         return heap.size() == 0;
     }
 
     void printHeap() {
         for (int val : heap)
             cout << val << " ";
         cout << endl;
     }
 };
 
 int main2() {
     cout << "Example 2: Basic Max Heap Implementation 🌳>" << endl;
     MaxHeap maxHeap;
     maxHeap.insert(10);
     maxHeap.insert(5);
     maxHeap.insert(3);
     maxHeap.insert(2);
     maxHeap.insert(8);
     cout << "Heap elements after insertions: ";
     maxHeap.printHeap(); // Should maintain max-heap property
     cout << "Maximum element: " << maxHeap.getMax() << endl; // Should be 10
     cout << "Extracted maximum: " << maxHeap.extractMax() << endl; // Remove and return max element
     cout << "Heap elements after extraction: ";
     maxHeap.printHeap();
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 3: Heap Insertion Operation ➕🌳
  * Demonstrate the insertion operation in a heap.
  ****************************************************************************************/
 
 int main3() {
     cout << "Example 3: Heap Insertion Operation ➕🌳" << endl;
     MinHeap heap;
     vector<int> values = {20, 15, 8, 10, 5, 7, 6, 2, 9, 1};
     cout << "Inserting values into heap: ";
     for (int val : values) {
         cout << val << " ";
         heap.insert(val);
     }
     cout << endl;
     cout << "Heap elements after insertions: ";
     heap.printHeap(); // Should maintain min-heap property
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 4: Heap Extract Min/Max Operation ➖🌳
  * Demonstrate the extraction operation in a heap.
  ****************************************************************************************/
 
 int main4() {
     cout << "Example 4: Heap Extract Min/Max Operation ➖🌳" << endl;
     MaxHeap heap;
     vector<int> values = {20, 15, 8, 10, 5, 7, 6, 2, 9, 1};
     for (int val : values) {
         heap.insert(val);
     }
     cout << "Heap elements before extraction: ";
     heap.printHeap();
     cout << "Extracting elements: ";
     while (!heap.isEmpty()) {
         cout << heap.extractMax() << " ";
     }
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 5: Heapify Operation 🏗️🌳
  * Building a heap from an existing array (heapify).
  ****************************************************************************************/
 
 void buildMinHeap(vector<int>& arr) {
     int size = arr.size();
     // Start from the last parent node and move upwards
     for (int i = (size / 2) - 1; i >= 0; i--) {
         // Sift down operation
         int index = i;
         while (true) {
             int minIndex = index;
             int l = 2 * index + 1;
             int r = 2 * index + 2;
 
             if (l < size && arr[l] < arr[minIndex])
                 minIndex = l;
             if (r < size && arr[r] < arr[minIndex])
                 minIndex = r;
 
             if (minIndex != index) {
                 swap(arr[index], arr[minIndex]);
                 index = minIndex;
             } else {
                 break;
             }
         }
     }
 }
 
 int main5() {
     cout << "Example 5: Heapify Operation 🏗️🌳" << endl;
     vector<int> arr = {3, 9, 2, 1, 4, 5};
     cout << "Original array: ";
     for (int val : arr)
         cout << val << " ";
     cout << endl;
 
     buildMinHeap(arr);
 
     cout << "Array after heapify (Min Heap): ";
     for (int val : arr)
         cout << val << " ";
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 6: Priority Queue Implementation using Heaps ⏳📥📤
  * Using STL priority_queue to manage prioritized elements.
  ****************************************************************************************/
 
 int main6() {
     cout << "Example 6: Priority Queue Implementation using Heaps ⏳📥📤" << endl;
     // Min Priority Queue using greater comparator
     priority_queue<int, vector<int>, greater<int>> minPQ;
 
     minPQ.push(10);
     minPQ.push(5);
     minPQ.push(15);
     minPQ.push(3);
 
     cout << "Priority Queue elements (Min Heap): ";
     while (!minPQ.empty()) {
         cout << minPQ.top() << " "; // Access smallest element
         minPQ.pop(); // Remove the top element
     }
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 7: Priority Scheduling Simulation ⏳💻
  * Simulating task scheduling based on priority using priority queues.
  ****************************************************************************************/
 
 struct Task {
     int priority; // Lower number means higher priority
     string name;
 
     // Overload operator for priority comparison
     bool operator<(const Task& other) const {
         return priority > other.priority; // For min-heap behavior
     }
 };
 
 int main7() {
     cout << "Example 7: Priority Scheduling Simulation ⏳💻" << endl;
     priority_queue<Task> taskQueue;
 
     // Adding tasks to the queue
     taskQueue.push({1, "Emergency Fix"});
     taskQueue.push({3, "Code Review"});
     taskQueue.push({2, "Feature Development"});
     taskQueue.push({4, "Team Meeting"});
 
     cout << "Processing tasks based on priority:" << endl;
     while (!taskQueue.empty()) {
         Task currentTask = taskQueue.top();
         taskQueue.pop();
         cout << "Processing task: " << currentTask.name << " with priority " << currentTask.priority << endl;
     }
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 8: Heap Sort Algorithm Implementation 🚀📖
  * Using heap data structure to sort an array.
  ****************************************************************************************/
 
 void heapSort(vector<int>& arr) {
     // Build max heap
     int size = arr.size();
     for (int i = (size / 2) - 1; i >= 0; i--) {
         // Sift down operation
         int index = i;
         while (true) {
             int maxIndex = index;
             int l = 2 * index + 1;
             int r = 2 * index + 2;
 
             if (l < size && arr[l] > arr[maxIndex])
                 maxIndex = l;
             if (r < size && arr[r] > arr[maxIndex])
                 maxIndex = r;
 
             if (maxIndex != index) {
                 swap(arr[index], arr[maxIndex]);
                 index = maxIndex;
             } else {
                 break;
             }
         }
     }
 
     // Perform heap sort
     for (int i = size - 1; i >= 1; i--) {
         swap(arr[0], arr[i]); // Move current max to the end
         int index = 0;
         int heapSize = i;
         // Sift down
         while (true) {
             int maxIndex = index;
             int l = 2 * index + 1;
             int r = 2 * index + 2;
 
             if (l < heapSize && arr[l] > arr[maxIndex])
                 maxIndex = l;
             if (r < heapSize && arr[r] > arr[maxIndex])
                 maxIndex = r;
 
             if (maxIndex != index) {
                 swap(arr[index], arr[maxIndex]);
                 index = maxIndex;
             } else {
                 break;
             }
         }
     }
 }
 
 int main8() {
     cout << "Example 8: Heap Sort Algorithm Implementation 🚀📖" << endl;
     vector<int> arr = {12, 11, 13, 5, 6, 7};
     cout << "Original array: ";
     for (int val : arr)
         cout << val << " ";
     cout << endl;
 
     heapSort(arr);
 
     cout << "Sorted array: ";
     for (int val : arr)
         cout << val << " ";
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 9: Dijkstra's Algorithm Example 🌐🗺️
  * Finding shortest paths from source to all vertices in a graph using a priority queue.
  ****************************************************************************************/
 
 struct Edge {
     int to;
     int weight;
 };
 
 void dijkstra(int src, vector<vector<Edge>>& graph) {
     int V = graph.size();
     vector<int> dist(V, numeric_limits<int>::max()); // Initialize distances
     dist[src] = 0;
 
     // Min priority queue to select edge with minimum weight
     priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
     pq.push({0, src}); // Distance to source is 0
 
     while (!pq.empty()) {
         int u = pq.top().second;
         pq.pop();
 
         for (Edge& edge : graph[u]) {
             int v = edge.to;
             int weight = edge.weight;
 
             // If shorter path found
             if (dist[u] + weight < dist[v]) {
                 dist[v] = dist[u] + weight;
                 pq.push({dist[v], v});
             }
         }
     }
 
     cout << "Vertex   Distance from Source" << endl;
     for (int i = 0; i < V; ++i)
         cout << i << "\t\t" << dist[i] << endl;
 }
 
 int main9() {
     cout << "Example 9: Dijkstra's Algorithm Example 🌐🗺️" << endl;
     int V = 5;
     vector<vector<Edge>> graph(V);
 
     // Adding edges to the graph
     graph[0].push_back({1, 9});
     graph[0].push_back({2, 6});
     graph[0].push_back({3, 5});
     graph[0].push_back({4, 3});
 
     graph[2].push_back({1, 2});
     graph[2].push_back({3, 4});
 
     int src = 0; // Source vertex
     dijkstra(src, graph);
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Example 10: Huffman Coding Tree Construction 📦📉
  * Constructing a Huffman Tree for optimal prefix codes.
  ****************************************************************************************/
 
 struct HuffmanNode {
     char data;
     int freq;
     HuffmanNode *left, *right;
 
     HuffmanNode(char data, int freq) : data(data), freq(freq), left(nullptr), right(nullptr) {}
 };
 
 // Comparator for priority queue
 struct Compare {
     bool operator()(HuffmanNode* l, HuffmanNode* r) {
         return l->freq > r->freq; // Min-heap based on frequency
     }
 };
 
 void printCodes(HuffmanNode* root, string str) {
     if (!root)
         return;
 
     if (root->data != '$') // '$' is a special value for non-leaf nodes
         cout << root->data << ": " << str << endl;
 
     printCodes(root->left, str + "0");
     printCodes(root->right, str + "1");
 }
 
 void HuffmanCodes(vector<char>& data, vector<int>& freq) {
     HuffmanNode *left, *right, *top;
     priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare> minHeap;
 
     // Create leaf nodes and add to min heap
     for (size_t i = 0; i < data.size(); ++i)
         minHeap.push(new HuffmanNode(data[i], freq[i]));
 
     // Iterate until the heap size is 1
     while (minHeap.size() != 1) {
         // Extract two minimum frequency nodes
         left = minHeap.top();
         minHeap.pop();
 
         right = minHeap.top();
         minHeap.pop();
 
         // Create new internal node with sum of frequencies
         top = new HuffmanNode('$', left->freq + right->freq);
         top->left = left;
         top->right = right;
 
         // Add new node to min heap
         minHeap.push(top);
     }
 
     // Print Huffman codes
     cout << "Huffman Codes:" << endl;
     printCodes(minHeap.top(), "");
 }
 
 int main10() {
     cout << "Example 10: Huffman Coding Tree Construction 📦📉" << endl;
     vector<char> data = {'a', 'b', 'c', 'd', 'e', 'f'};
     vector<int> freq = {5, 9, 12, 13, 16, 45};
 
     HuffmanCodes(data, freq);
     cout << endl;
     return 0;
 }
 
 /****************************************************************************************
  * Main Function to Run All Examples
  ****************************************************************************************/
 
 int main() {
     main1(); // Basic Min Heap
     main2(); // Basic Max Heap
     main3(); // Heap Insertion
     main4(); // Heap Extraction
     main5(); // Heapify Operation
     main6(); // Priority Queue Implementation
     main7(); // Priority Scheduling Simulation
     main8(); // Heap Sort Algorithm
     main9(); // Dijkstra's Algorithm
     main10(); // Huffman Coding Tree
     return 0;
 }
 
 /****************************************************************************************
  * End of Advanced C++ Heaps and Priority Queues Tutorial 🏥🚑 ⏳🥇
  ****************************************************************************************/