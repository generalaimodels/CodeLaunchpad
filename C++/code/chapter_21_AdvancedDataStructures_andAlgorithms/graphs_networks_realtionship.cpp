// graph_examples.cpp
// 🗺️ Graphs - Networks & Relationships: Social Networks & Road Maps 🌐
// This file covers various graph concepts from basic to advanced with examples.

// 🚀 Includes
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <limits> // For infinity
#include <algorithm> // For sort
#include <cstring> // For memset

// 👥 Namespaces
using namespace std;

// 🌟 Example 1: Basic Graph Representation using Adjacency Matrix
void example1() {
    // 🔢 Number of vertices
    int V = 5;
    // 🧮 Adjacency Matrix initialization
    int adjMatrix[5][5] = {0};

    // 🔗 Adding edges (Undirected Graph)
    adjMatrix[0][1] = 1; adjMatrix[1][0] = 1; // Edge between 0 and 1
    adjMatrix[0][2] = 1; adjMatrix[2][0] = 1; // Edge between 0 and 2
    adjMatrix[1][2] = 1; adjMatrix[2][1] = 1; // Edge between 1 and 2
    adjMatrix[1][3] = 1; adjMatrix[3][1] = 1; // Edge between 1 and 3
    adjMatrix[2][4] = 1; adjMatrix[4][2] = 1; // Edge between 2 and 4

    // 👀 Display Adjacency Matrix
    cout << "Example 1: Adjacency Matrix Representation\n";
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < V; j++) {
            cout << adjMatrix[i][j] << " ";
        }
        cout << endl;
    }
}

// 🌟 Example 2: Basic Graph Representation using Adjacency List
void example2() {
    // 🔢 Number of vertices
    int V = 5;
    // 📝 Adjacency List initialization
    vector<int> adjList[5];

    // 🔗 Adding edges (Undirected Graph)
    adjList[0].push_back(1); adjList[1].push_back(0); // Edge between 0 and 1
    adjList[0].push_back(2); adjList[2].push_back(0); // Edge between 0 and 2
    adjList[1].push_back(2); adjList[2].push_back(1); // Edge between 1 and 2
    adjList[1].push_back(3); adjList[3].push_back(1); // Edge between 1 and 3
    adjList[2].push_back(4); adjList[4].push_back(2); // Edge between 2 and 4

    // 👀 Display Adjacency List
    cout << "\nExample 2: Adjacency List Representation\n";
    for(int i = 0; i < V; i++) {
        cout << i << " -> ";
        for(auto v : adjList[i]) {
            cout << v << " ";
        }
        cout << endl;
    }
}

// 🌟 Example 3: Directed Graph Representation
void example3() {
    int V = 4;
    vector<int> adjList[4];

    // ➡️ Adding edges (Directed Graph)
    adjList[0].push_back(1); // Edge from 0 to 1
    adjList[1].push_back(2); // Edge from 1 to 2
    adjList[2].push_back(3); // Edge from 2 to 3
    adjList[3].push_back(1); // Edge from 3 to 1 (creates a cycle)

    // 👀 Display Adjacency List
    cout << "\nExample 3: Directed Graph Representation\n";
    for(int i = 0; i < V; i++) {
        cout << i << " -> ";
        for(auto v : adjList[i]) {
            cout << v << " ";
        }
        cout << endl;
    }
}

// 🌟 Example 4: Weighted Graph Representation
struct Edge {
    int to;
    int weight;
};

void example4() {
    int V = 4;
    vector<Edge> adjList[4];

    // 🔢🔗 Adding weighted edges (Undirected Graph)
    adjList[0].push_back({1, 5}); adjList[1].push_back({0, 5}); // Edge between 0 and 1 with weight 5
    adjList[0].push_back({2, 3}); adjList[2].push_back({0, 3}); // Edge between 0 and 2 with weight 3
    adjList[1].push_back({2, 2}); adjList[2].push_back({1, 2}); // Edge between 1 and 2 with weight 2
    adjList[1].push_back({3, 6}); adjList[3].push_back({1, 6}); // Edge between 1 and 3 with weight 6

    // 👀 Display Weighted Adjacency List
    cout << "\nExample 4: Weighted Graph Representation\n";
    for(int i = 0; i < V; i++) {
        cout << i << " -> ";
        for(auto edge : adjList[i]) {
            cout << "(" << edge.to << ", w=" << edge.weight << ") ";
        }
        cout << endl;
    }
}

// 🌟 Example 5: Depth-First Search (DFS) using Recursion
void dfs_recursive(int v, vector<int> adjList[], vector<bool> &visited) {
    visited[v] = true; // Mark current node as visited
    cout << v << " "; // Process current node

    // Recurse for all adjacent vertices
    for(auto u : adjList[v]) {
        if(!visited[u]) {
            dfs_recursive(u, adjList, visited);
        }
    }
}

void example5() {
    int V = 5;
    vector<int> adjList[5];

    // 🔗 Constructing the graph
    adjList[0].push_back(1); adjList[0].push_back(2);
    adjList[1].push_back(0); adjList[1].push_back(3);
    adjList[2].push_back(0); adjList[2].push_back(4);
    adjList[3].push_back(1);
    adjList[4].push_back(2);

    vector<bool> visited(V, false);

    // 🌐 Starting DFS from vertex 0
    cout << "\nExample 5: Depth-First Search (DFS)\n";
    dfs_recursive(0, adjList, visited);
    cout << endl;
}

// 🌟 Example 6: Breadth-First Search (BFS) using Queue
void example6() {
    int V = 5;
    vector<int> adjList[5];

    // 🔗 Constructing the graph
    adjList[0].push_back(1); adjList[0].push_back(2);
    adjList[1].push_back(0); adjList[1].push_back(3);
    adjList[2].push_back(0); adjList[2].push_back(4);
    adjList[3].push_back(1);
    adjList[4].push_back(2);

    vector<bool> visited(V, false);
    queue<int> q;

    // 🌐 Starting BFS from vertex 0
    cout << "\nExample 6: Breadth-First Search (BFS)\n";
    visited[0] = true;
    q.push(0);

    while(!q.empty()) {
        int v = q.front();
        q.pop();
        cout << v << " ";

        for(auto u : adjList[v]) {
            if(!visited[u]) {
                visited[u] = true;
                q.push(u);
            }
        }
    }
    cout << endl;
}

// 🌟 Example 7: Dijkstra's Algorithm for Shortest Path
void example7() {
    int V = 5;
    vector<Edge> adjList[5];

    // 🔗 Constructing the graph with non-negative weights
    adjList[0].push_back({1, 2}); adjList[0].push_back({2, 4});
    adjList[1].push_back({2, 1}); adjList[1].push_back({3, 7});
    adjList[2].push_back({4, 3});
    adjList[3].push_back({4, 1});

    vector<int> dist(V, numeric_limits<int>::max());
    vector<bool> visited(V, false);
    dist[0] = 0;

    // 🌐 Dijkstra's Algorithm
    for(int i = 0; i < V; i++) {
        int u = -1;
        // Find the unvisited vertex with the smallest distance
        for(int j = 0; j < V; j++) {
            if(!visited[j] && (u == -1 || dist[j] < dist[u]))
                u = j;
        }

        if(dist[u] == numeric_limits<int>::max())
            break;

        visited[u] = true;

        for(auto edge : adjList[u]) {
            int to = edge.to;
            int weight = edge.weight;

            if(dist[u] + weight < dist[to]) {
                dist[to] = dist[u] + weight;
            }
        }
    }

    // 👀 Display shortest distances from vertex 0
    cout << "\nExample 7: Dijkstra's Algorithm\n";
    for(int i = 0; i < V; i++) {
        cout << "Distance from 0 to " << i << " is " << dist[i] << endl;
    }
}

// 🌟 Example 8: Bellman-Ford Algorithm for Shortest Path (with negative weights)
void example8() {
    int V = 5;
    vector<Edge> edges;

    // 🔗 Constructing the graph
    edges.push_back({0, 1, 6});
    edges.push_back({0, 2, 7});
    edges.push_back({1, 2, 8});
    edges.push_back({1, 3, 5});
    edges.push_back({1, 4, -4});
    edges.push_back({2, 3, -3});
    edges.push_back({2, 4, 9});
    edges.push_back({3, 1, -2});
    edges.push_back({4, 0, 2});
    edges.push_back({4, 3, 7});

    vector<int> dist(V, numeric_limits<int>::max());
    dist[0] = 0;

    // 🌐 Bellman-Ford Algorithm
    for(int i = 0; i < V - 1; i++) {
        for(auto edge : edges) {
            int u = edge.to;
            int v = edge.weight;
            int weight = edge.weight;
            if(dist[u] != numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
            }
        }
    }

    // 👀 Check for negative-weight cycles
    bool hasNegativeCycle = false;
    for(auto edge : edges) {
        int u = edge.to;
        int v = edge.weight;
        int weight = edge.weight;
        if(dist[u] != numeric_limits<int>::max() && dist[u] + weight < dist[v]) {
            hasNegativeCycle = true;
            break;
        }
    }

    cout << "\nExample 8: Bellman-Ford Algorithm\n";
    if(hasNegativeCycle) {
        cout << "Graph contains a negative-weight cycle\n";
    } else {
        for(int i = 0; i < V; i++) {
            cout << "Distance from 0 to " << i << " is " << dist[i] << endl;
        }
    }
}

// 🌟 Example 9: Floyd-Warshall Algorithm for All-Pairs Shortest Paths
void example9() {
    int V = 4;
    int dist[4][4];

    // 🌐 Initialize distances
    for(int i = 0; i < V; i++)
        for(int j = 0; j < V; j++)
            if(i == j) dist[i][j] = 0;
            else dist[i][j] = numeric_limits<int>::max();

    // 🔗 Adding edges
    dist[0][1] = 5;
    dist[0][3] = 10;
    dist[1][2] = 3;
    dist[2][3] = 1;

    // 🌐 Floyd-Warshall Algorithm
    for(int k = 0; k < V; k++)
        for(int i = 0; i < V; i++)
            for(int j = 0; j < V; j++)
                if(dist[i][k] != numeric_limits<int>::max() && dist[k][j] != numeric_limits<int>::max())
                    if(dist[i][k] + dist[k][j] < dist[i][j])
                        dist[i][j] = dist[i][k] + dist[k][j];

    // 👀 Display shortest distances between all pairs
    cout << "\nExample 9: Floyd-Warshall Algorithm\n";
    for(int i = 0; i < V; i++) {
        for(int j = 0; j < V; j++) {
            if(dist[i][j] == numeric_limits<int>::max())
                cout << "INF ";
            else
                cout << dist[i][j] << " ";
        }
        cout << endl;
    }
}

// 🌟 Example 10: Prim's Algorithm for Minimum Spanning Tree (MST)
void example10() {
    int V = 5;
    vector<Edge> adjList[5];

    // 🔗 Constructing the graph
    adjList[0].push_back({1, 2}); adjList[1].push_back({0, 2});
    adjList[0].push_back({3, 6}); adjList[3].push_back({0, 6});
    adjList[1].push_back({2, 3}); adjList[2].push_back({1, 3});
    adjList[1].push_back({3, 8}); adjList[3].push_back({1, 8});
    adjList[1].push_back({4, 5}); adjList[4].push_back({1, 5});
    adjList[2].push_back({4, 7}); adjList[4].push_back({2, 7});

    vector<int> key(V, numeric_limits<int>::max()); // Key values to pick minimum weight edge
    vector<bool> inMST(V, false); // To represent set of vertices included in MST
    vector<int> parent(V, -1); // Array to store constructed MST
    key[0] = 0; // Start from vertex 0

    // 🌐 Prim's Algorithm
    for(int count = 0; count < V - 1; count++) {
        int u = -1;
        // Pick the minimum key vertex from the set of vertices not yet included in MST
        for(int i = 0; i < V; i++)
            if(!inMST[i] && (u == -1 || key[i] < key[u]))
                u = i;

        inMST[u] = true;

        // Update key value and parent index of the adjacent vertices
        for(auto edge : adjList[u]) {
            int v = edge.to;
            int weight = edge.weight;
            if(!inMST[v] && weight < key[v]) {
                key[v] = weight;
                parent[v] = u;
            }
        }
    }

    // 👀 Display MST
    cout << "\nExample 10: Prim's Algorithm (MST)\n";
    for(int i = 1; i < V; i++)
        cout << parent[i] << " - " << i << "\tWeight: " << key[i] << endl;
}

// 🌟 Example 11: Kruskal's Algorithm for Minimum Spanning Tree (MST)
struct EdgeK {
    int from, to, weight;
};

bool compareEdges(EdgeK a, EdgeK b) {
    return a.weight < b.weight;
}

int findParent(int v, vector<int>& parent) {
    if(parent[v] != v)
        parent[v] = findParent(parent[v], parent);
    return parent[v];
}

void unionSets(int a, int b, vector<int>& parent) {
    a = findParent(a, parent);
    b = findParent(b, parent);
    if(a != b)
        parent[b] = a;
}

void example11() {
    int V = 4;
    vector<EdgeK> edges;

    // 🔗 Constructing the graph
    edges.push_back({0, 1, 10});
    edges.push_back({0, 2, 6});
    edges.push_back({0, 3, 5});
    edges.push_back({1, 3, 15});
    edges.push_back({2, 3, 4});

    // 🌐 Kruskal's Algorithm
    sort(edges.begin(), edges.end(), compareEdges);

    vector<int> parent(V);
    for(int i = 0; i < V; i++)
        parent[i] = i;

    vector<EdgeK> mst;

    for(auto edge : edges) {
        int u = findParent(edge.from, parent);
        int v = findParent(edge.to, parent);
        if(u != v) {
            mst.push_back(edge);
            unionSets(u, v, parent);
        }
    }

    // 👀 Display MST
    cout << "\nExample 11: Kruskal's Algorithm (MST)\n";
    for(auto edge : mst)
        cout << edge.from << " - " << edge.to << "\tWeight: " << edge.weight << endl;
}

// 🌟 Example 12: Topological Sort using DFS
void topologicalSortUtil(int v, vector<bool>& visited, stack<int>& Stack, vector<int> adjList[]) {
    visited[v] = true;
    for(auto u : adjList[v])
        if(!visited[u])
            topologicalSortUtil(u, visited, Stack, adjList);
    Stack.push(v);
}

void example12() {
    int V = 6;
    vector<int> adjList[6];

    // 🔗 Constructing the graph
    adjList[5].push_back(2);
    adjList[5].push_back(0);
    adjList[4].push_back(0);
    adjList[4].push_back(1);
    adjList[2].push_back(3);
    adjList[3].push_back(1);

    stack<int> Stack;
    vector<bool> visited(V, false);

    // 🌐 Performing Topological Sort
    for(int i = 0; i < V; i++)
        if(!visited[i])
            topologicalSortUtil(i, visited, Stack, adjList);

    // 👀 Display Topological Order
    cout << "\nExample 12: Topological Sort\n";
    while(!Stack.empty()) {
        cout << Stack.top() << " ";
        Stack.pop();
    }
    cout << endl;
}

// 🌟 Example 13: Cycle Detection in Undirected Graph using DFS
bool isCyclicUtil(int v, vector<bool>& visited, int parent, vector<int> adjList[]) {
    visited[v] = true;
    for(auto u : adjList[v]) {
        if(!visited[u]) {
            if(isCyclicUtil(u, visited, v, adjList))
                return true;
        }
        else if(u != parent)
            return true;
    }
    return false;
}

void example13() {
    int V = 5;
    vector<int> adjList[5];

    // 🔗 Constructing the graph
    adjList[0].push_back(1); adjList[1].push_back(0);
    adjList[1].push_back(2); adjList[2].push_back(1);
    adjList[2].push_back(0); adjList[0].push_back(2);
    adjList[1].push_back(3); adjList[3].push_back(1);
    adjList[3].push_back(4); adjList[4].push_back(3);

    vector<bool> visited(V, false);
    bool hasCycle = false;

    // 🌐 Cycle Detection
    for(int i = 0; i < V; i++)
        if(!visited[i])
            if(isCyclicUtil(i, visited, -1, adjList)) {
                hasCycle = true;
                break;
            }

    // 👀 Display Result
    cout << "\nExample 13: Cycle Detection in Undirected Graph\n";
    if(hasCycle)
        cout << "Graph contains cycle\n";
    else
        cout << "Graph doesn't contain cycle\n";
}

// 🌟 Example 14: Cycle Detection in Directed Graph using DFS
bool isCyclicDirectedUtil(int v, vector<bool>& visited, vector<bool>& recStack, vector<int> adjList[]) {
    visited[v] = true;
    recStack[v] = true;

    for(auto u : adjList[v]) {
        if(!visited[u] && isCyclicDirectedUtil(u, visited, recStack, adjList))
            return true;
        else if(recStack[u])
            return true;
    }
    recStack[v] = false;
    return false;
}

void example14() {
    int V = 4;
    vector<int> adjList[4];

    // 🔗 Constructing the graph
    adjList[0].push_back(1);
    adjList[1].push_back(2);
    adjList[2].push_back(0);
    adjList[2].push_back(3);

    vector<bool> visited(V, false);
    vector<bool> recStack(V, false);
    bool hasCycle = false;

    // 🌐 Cycle Detection
    for(int i = 0; i < V; i++)
        if(!visited[i])
            if(isCyclicDirectedUtil(i, visited, recStack, adjList)) {
                hasCycle = true;
                break;
            }

    // 👀 Display Result
    cout << "\nExample 14: Cycle Detection in Directed Graph\n";
    if(hasCycle)
        cout << "Graph contains cycle\n";
    else
        cout << "Graph doesn't contain cycle\n";
}

// 🌟 Example 15: Using Graphs in Social Networks (Friend Recommendation)
void example15() {
    int V = 5; // Users
    vector<int> adjList[5];

    // 👥 Constructing friendship graph
    adjList[0].push_back(1); adjList[1].push_back(0); // User 0 and 1 are friends
    adjList[0].push_back(2); adjList[2].push_back(0); // User 0 and 2 are friends
    adjList[1].push_back(3); adjList[3].push_back(1); // User 1 and 3 are friends
    adjList[2].push_back(4); adjList[4].push_back(2); // User 2 and 4 are friends

    int user = 0; // Current user
    vector<bool> friends(V, false);
    vector<int> recommendations;

    // 🌐 Mark direct friends
    for(auto f : adjList[user]) {
        friends[f] = true;
    }

    // 🔍 Find friend-of-friend recommendations
    for(auto f : adjList[user]) {
        for(auto fof : adjList[f]) {
            if(fof != user && !friends[fof]) {
                recommendations.push_back(fof);
            }
        }
    }

    // 👀 Display Friend Recommendations for User 0
    cout << "\nExample 15: Friend Recommendation in Social Network\n";
    cout << "Friend recommendations for User " << user << ": ";
    for(auto rec : recommendations)
        cout << rec << " ";
    cout << endl;
}

// 🏁 Main function to run all examples
int main() {
    example1();
    example2();
    example3();
    example4();
    example5();
    example6();
    example7();
    example8();
    example9();
    example10();
    example11();
    example12();
    example13();
    example14();
    example15();
    return 0;
}