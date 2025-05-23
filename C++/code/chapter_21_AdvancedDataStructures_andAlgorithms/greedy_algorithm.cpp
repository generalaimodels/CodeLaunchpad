// Greedy Algorithms Demonstration in C++
// 🎯💰 Greedy Approach: Making locally optimal choices to reach global optimum
// Author: Advanced C++ Coder
// Date: 2025-2-9

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <string>
#include <map>
#include <set>
#include <climits>

using namespace std;

// === Example 1: Coin Change Problem (Standard Denominations) 💰 ===
// Greedy approach works optimally with standard coin denominations.

void coinChange(vector<int>& coins, int amount) {
    sort(coins.rbegin(), coins.rend()); // 🔄🔽 Sort coins in descending order
    vector<int> result;
    for (int coin : coins) {
        while (amount >= coin) {
            amount -= coin;      // 💰 Subtract coin value from amount
            result.push_back(coin); // 🪙 Collect coin
        }
    }
    if (amount != 0) {
        cout << "Change cannot be made with the given denominations." << endl; // ⚠️ Can't make exact change
    } else {
        cout << "Coins used to make change: ";
        for (int coin : result) {
            cout << coin << " ";
        }
        cout << endl; // ✅ Change made successfully
    }
}

// === Example 2: Coin Change Problem (Non-standard Denominations) ⚠️💰 ===
// Greedy approach may not yield optimal solution with non-standard denominations.

void coinChangeNonStandard(vector<int>& coins, int amount) {
    sort(coins.rbegin(), coins.rend()); // 🔄🔽 Sort coins in descending order
    vector<int> result;
    for (int coin : coins) {
        while (amount >= coin) {
            amount -= coin;      // 💰 Subtract coin value from amount
            result.push_back(coin); // 🪙 Collect coin
        }
    }
    if (amount != 0) {
        cout << "Change cannot be made with the given denominations." << endl; // ⚠️ Can't make exact change
    } else {
        cout << "Coins used to make change (Greedy): ";
        for (int coin : result) {
            cout << coin << " ";
        }
        cout << endl; // Result of greedy approach
    }
}

// Optimal solution using Dynamic Programming 🧮

void optimalCoinChange(vector<int>& coins, int amount) {
    vector<int> minCoins(amount + 1, amount + 1); // Initialize DP array
    minCoins[0] = 0;
    for (int a = 1; a <= amount; ++a) {
        for (int coin : coins) {
            if (a - coin >= 0) {
                minCoins[a] = min(minCoins[a], minCoins[a - coin] + 1); // 🧩 Update minimum coins
            }
        }
    }
    if (minCoins[amount] > amount) {
        cout << "Change cannot be made with the given denominations." << endl; // ⚠️ Can't make exact change
    } else {
        cout << "Minimum coins needed (Optimal): " << minCoins[amount] << endl; // ✅ Optimal solution
    }
}

// === Example 3: Activity Selection Problem 🗓️ ===
// Selecting maximum number of non-overlapping activities.

void activitySelection(vector<int>& startTimes, vector<int>& finishTimes) {
    int n = startTimes.size();
    vector<pair<int,int>> activities(n);
    for (int i = 0; i < n; ++i) {
        activities[i] = {finishTimes[i], startTimes[i]}; // Pair finish time with start time
    }
    sort(activities.begin(), activities.end()); // 🔄 Sort activities by finish time

    cout << "Selected activities: ";
    int lastFinishTime = -1;
    for (auto activity : activities) {
        int start = activity.second;
        int finish = activity.first;
        if (start >= lastFinishTime) {
            cout << "(" << start << ", " << finish << ") "; // 🗓️ Select activity
            lastFinishTime = finish;
        }
    }
    cout << endl;
}

// === Example 4: Fractional Knapsack Problem 🎒 ===
// Maximizing value within weight limit by taking fractions of items.

struct Item {
    int value;
    int weight;
};

bool cmp(Item a, Item b) {
    double r1 = (double)a.value / a.weight; // Ratio of value to weight
    double r2 = (double)b.value / b.weight;
    return r1 > r2; // 🔄 Sort in decreasing order
}

void fractionalKnapsack(int W, vector<Item>& items) {
    sort(items.begin(), items.end(), cmp); // 🔄 Sorting items
    int n = items.size();
    double totalValue = 0.0; // 💰 Total value collected

    for (int i = 0; i < n && W > 0; ++i) {
        if (items[i].weight <= W) {
            W -= items[i].weight;
            totalValue += items[i].value; // 👜 Take full item
        } else {
            totalValue += items[i].value * ((double)W / items[i].weight); // 👜 Take fraction of item
            W = 0;
        }
    }
    cout << "Maximum value in Knapsack = " << totalValue << endl; // ✅
}

// === Example 5: Huffman Coding 📦 ===
// Data compression by assigning variable-length codes based on frequencies.

struct MinHeapNode {
    char data;                 // Character
    unsigned freq;             // Frequency
    MinHeapNode *left, *right; // Left and right children

    MinHeapNode(char data, unsigned freq)
        : data(data), freq(freq), left(NULL), right(NULL) {}
};

struct compare {
    bool operator()(MinHeapNode* l, MinHeapNode* r) {
        return l->freq > r->freq; // 🔄 Min-heap based on frequency
    }
};

void printCodes(struct MinHeapNode* root, string str) {
    if (!root)
        return;
    if (root->data != '$') {
        cout << root->data << ": " << str << endl; // 📦 Character and its code
    }
    printCodes(root->left, str + "0"); // Left edge as '0'
    printCodes(root->right, str + "1"); // Right edge as '1'
}

void HuffmanCodes(vector<char>& data, vector<int>& freq) {
    int n = data.size();
    struct MinHeapNode *left, *right, *top;
    priority_queue<MinHeapNode*, vector<MinHeapNode*>, compare> minHeap;

    for (int i = 0; i < n; ++i)
        minHeap.push(new MinHeapNode(data[i], freq[i])); // Insert all characters

    while (minHeap.size() != 1) {
        left = minHeap.top(); minHeap.pop(); // Extract min frequency item
        right = minHeap.top(); minHeap.pop(); // Extract next min frequency item

        top = new MinHeapNode('$', left->freq + right->freq); // Combine frequencies
        top->left = left;
        top->right = right;
        minHeap.push(top); // Insert new node
    }

    printCodes(minHeap.top(), ""); // Output the codes
}

// === Example 6: Dijkstra's Algorithm 🌐 ===
// Finding the shortest path in a graph with non-negative weights.

#define V 9

int minDistance(int dist[], bool sptSet[]) {
    int min = INT_MAX, min_index;

    for (int v = 0; v < V; ++v)
        if (!sptSet[v] && dist[v] <= min)
            min = dist[v], min_index = v; // Find vertex with minimum distance

    return min_index;
}

void printSolution(int dist[]) {
    cout << "Vertex \t Distance from Source" << endl;
    for (int i = 0; i < V; ++i)
        cout << i << " \t\t " << dist[i] << endl; // 🚩 Distance to each vertex
}

void dijkstra(int graph[V][V], int src) {
    int dist[V];     // Output array. dist[i] holds the shortest distance from src to i
    bool sptSet[V];  // sptSet[i] is true if vertex i is included in shortest path tree

    for (int i = 0; i < V; ++i)
        dist[i] = INT_MAX, sptSet[i] = false; // Initialize distances as INFINITE

    dist[src] = 0; // Distance of source vertex from itself is always 0

    for (int count = 0; count < V - 1; ++count) {
        int u = minDistance(dist, sptSet); // Pick minimum distance vertex
        sptSet[u] = true; // Mark the picked vertex as processed

        for (int v = 0; v < V; ++v)
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX 
                && dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v]; // Update dist[v]
    }
    printSolution(dist);
}

// === Example 7: Prim's Algorithm 🌳 ===
// Finding Minimum Spanning Tree (MST) for a connected weighted undirected graph.

#define V_PRIM 5

int minKey(int key[], bool mstSet[]) {
    int min = INT_MAX, min_index;
    for (int v = 0; v < V_PRIM; ++v)
        if (!mstSet[v] && key[v] < min)
            min = key[v], min_index = v; // Find vertex with minimum key value
    return min_index;
}

void printMST(int parent[], int graph[V_PRIM][V_PRIM]) {
    cout << "Edge \tWeight" << endl;
    for (int i = 1; i < V_PRIM; ++i)
        cout << parent[i] << " - " << i << "\t" << graph[i][parent[i]] << endl; // 🌳 MST edges
}

void primMST(int graph[V_PRIM][V_PRIM]) {
    int parent[V_PRIM]; // Array to store constructed MST
    int key[V_PRIM];    // Key values used to pick minimum weight edge
    bool mstSet[V_PRIM]; // To represent set of vertices not yet included in MST

    for (int i = 0; i < V_PRIM; ++i)
        key[i] = INT_MAX, mstSet[i] = false;

    key[0] = 0;     // Include first vertex in MST
    parent[0] = -1; // First node is root of MST

    for (int count = 0; count < V_PRIM - 1; ++count) {
        int u = minKey(key, mstSet); // Pick minimum key vertex
        mstSet[u] = true; // Add vertex to MST Set

        for (int v = 0; v < V_PRIM; ++v)
            if (graph[u][v] && !mstSet[v] && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v]; // Update key[v]
    }
    printMST(parent, graph);
}

// === Example 8: Kruskal's Algorithm 🔗 ===
// Finding MST using Kruskal's Algorithm.

struct Edge {
    int src, dest, weight;
};

struct subset {
    int parent;
    int rank;
};

int find(struct subset subsets[], int i) {
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent); // Path compression
    return subsets[i].parent;
}

void Union(struct subset subsets[], int x, int y) {
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot; // Attach smaller rank under root of higher rank
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
    else {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}

bool cmpEdge(Edge a, Edge b) {
    return a.weight < b.weight; // Sort edges in increasing order
}

void KruskalMST(vector<Edge>& edges, int V) {
    vector<Edge> result(V); // Resultant MST
    int e = 0; // Index for result[]
    int i = 0; // Index for sorted edges

    sort(edges.begin(), edges.end(), cmpEdge); // Sort edges

    struct subset *subsets = new subset[V];

    for (int v = 0; v < V; ++v) {
        subsets[v].parent = v; // Initialize subsets
        subsets[v].rank = 0;
    }

    while (e < V - 1 && i < edges.size()) {
        Edge next_edge = edges[i++]; // Pick smallest edge
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);

        if (x != y) {
            result[e++] = next_edge; // Include in MST
            Union(subsets, x, y);    // Union sets
        }
    }

    cout << "Edges in the constructed MST:" << endl;
    for (i = 0; i < e; ++i)
        cout << result[i].src << " -- " << result[i].dest << " == " << result[i].weight << endl;

    delete[] subsets;
}

// === Main Function to Execute Examples ===

int main() {
    // --- Coin Change Example 1 ---
    vector<int> coinsStandard = {25, 10, 5, 1}; // 💵 US coin denominations
    int amountStandard = 68; // 💲 Amount
    coinChange(coinsStandard, amountStandard);

    // --- Coin Change Example 2 ---
    vector<int> coinsNonStandard = {4, 3, 1}; // Non-standard denominations
    int amountNonStandard = 6;
    coinChangeNonStandard(coinsNonStandard, amountNonStandard); // Greedy approach
    optimalCoinChange(coinsNonStandard, amountNonStandard);     // Optimal solution

    // --- Activity Selection Example ---
    vector<int> startTimes = {1, 3, 0, 5, 8, 5};
    vector<int> finishTimes = {2, 4, 6, 7, 9, 9};
    activitySelection(startTimes, finishTimes);

    // --- Fractional Knapsack Example ---
    int W = 50; // Knapsack capacity
    vector<Item> items = {{60, 10}, {100, 20}, {120, 30}}; // {value, weight}
    fractionalKnapsack(W, items);

    // --- Huffman Coding Example ---
    vector<char> data = {'a', 'b', 'c', 'd', 'e', 'f'};
    vector<int> freq = {5, 9, 12, 13, 16, 45};
    HuffmanCodes(data, freq);

    // --- Dijkstra's Algorithm Example ---
    int graphDijkstra[V][V] = { {0, 4, 0, 0, 0, 0, 0, 8, 0},
                                {4, 0, 8, 0, 0, 0, 0, 11, 0},
                                {0, 8, 0, 7, 0, 4, 0, 0, 2},
                                {0, 0, 7, 0, 9, 14, 0, 0, 0},
                                {0, 0, 0, 9, 0, 10, 0, 0, 0},
                                {0, 0, 4, 14, 10, 0, 2, 0, 0},
                                {0, 0, 0, 0, 0, 2, 0, 1, 6},
                                {8, 11, 0, 0, 0, 0, 1, 0, 7},
                                {0, 0, 2, 0, 0, 0, 6, 7, 0} };
    dijkstra(graphDijkstra, 0);

    // --- Prim's Algorithm Example ---
    int graphPrim[V_PRIM][V_PRIM] = { {0, 2, 0, 6, 0},
                                      {2, 0, 3, 8, 5},
                                      {0, 3, 0, 0, 7},
                                      {6, 8, 0, 0, 9},
                                      {0, 5, 7, 9, 0} };
    primMST(graphPrim);

    // --- Kruskal's Algorithm Example ---
    int V_Kruskal = 4; // Number of vertices
    vector<Edge> edgesKruskal = {
        {0, 1, 10},
        {0, 2, 6},
        {0, 3, 5},
        {1, 3, 15},
        {2, 3, 4}
    };
    KruskalMST(edgesKruskal, V_Kruskal);

    return 0;
}