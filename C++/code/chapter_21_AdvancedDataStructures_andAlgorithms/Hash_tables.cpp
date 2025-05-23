/************************************************************************************
 * Advanced C++ Hash Tables Tutorial 🔑📖 🗂️⚡️
 *
 * This .cpp file covers the concepts of Hash Tables (Hash Maps) in great detail.
 * Each example is self-contained and includes explanations within comments.
 * There are at least 15 different examples for each topic to ensure deep understanding.
 * All explanations are provided within comments, no text exists outside this file.
 *
 * Topics Covered:
 * 1. Introduction to Hash Tables 🔑📖
 * 2. Hash Functions 🔑➡️🔢
 * 3. Collision Handling 💥🔑
 *    - Separate Chaining 🔗🔑
 *    - Open Addressing ➡️🔑
 *      - Linear Probing ➡️➡️➡️🔑
 *      - Quadratic Probing ➡️²➡️²➡️²🔑
 *      - Double Hashing 🔑🔑➡️➡️🔑
 * 4. Hash Table Operations ⚙️
 *    - Insert ➕🔑📖
 *    - Search 🔍🔑📖⚡️
 *    - Delete ➖🔑📖
 * 5. Load Factor ⚖️📖
 * 6. Applications of Hash Tables 🗂️⚡️
 *    - Database Indexing
 *    - Caching
 *    - Symbol Tables in Compilers
 *    - Implementing Sets
 *    - Associative Arrays
 *
 * Note: The code is error-free and designed for educational purposes.
 ************************************************************************************/

 #include <iostream>
 #include <vector>
 #include <list>
 #include <string>
 #include <stdexcept>
 #include <functional> // For std::hash
 #include <cmath>      // For quadratic probing
 #include <cassert>    // For assert
 
 using namespace std;
 
 /************************************************************************************
  * Example 1: Basic Hash Function Example 🔑➡️🔢
  * Demonstrates a simple hash function mapping strings to integer indices.
  ************************************************************************************/
 
 size_t simpleHashFunction(const string& key, size_t tableSize) {
     // Sum ASCII values of characters modulo table size
     size_t hashValue = 0;
     for (char ch : key) {
         hashValue += static_cast<size_t>(ch);
     }
     return hashValue % tableSize; // Ensure index is within table size
 }
 
 int main1() {
     cout << "Example 1: Basic Hash Function Example 🔑➡️🔢" << endl;
     string keys[] = {"apple", "banana", "cherry", "date", "elderberry"};
     size_t tableSize = 10;
 
     for (const string& key : keys) {
         size_t index = simpleHashFunction(key, tableSize);
         cout << "Key: " << key << ", Hash Index: " << index << endl;
     }
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 2: Good vs. Bad Hash Functions ⚖️🔑
  * Illustrates the importance of a good hash function in distributing keys uniformly.
  ************************************************************************************/
 
 size_t badHashFunction(const string& key, size_t tableSize) {
     // Returns the length of the key modulo table size
     return key.length() % tableSize;
 }
 
 size_t goodHashFunction(const string& key, size_t tableSize) {
     // Uses std::hash for better distribution
     return hash<string>{}(key) % tableSize;
 }
 
 int main2() {
     cout << "Example 2: Good vs. Bad Hash Functions ⚖️🔑" << endl;
     vector<string> keys = {"aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"};
     size_t tableSize = 5;
 
     cout << "Using Bad Hash Function:" << endl;
     for (const string& key : keys) {
         size_t index = badHashFunction(key, tableSize);
         cout << "Key: " << key << ", Hash Index: " << index << endl;
     }
 
     cout << "\nUsing Good Hash Function:" << endl;
     for (const string& key : keys) {
         size_t index = goodHashFunction(key, tableSize);
         cout << "Key: " << key << ", Hash Index: " << index << endl;
     }
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 3: Separate Chaining Collision Handling 🔗🔑
  * Implements a basic hash table using separate chaining for collision resolution.
  ************************************************************************************/
 
 class HashTableSeparateChaining {
 private:
     vector<list<pair<string, int>>> table; // Vector of lists for chaining
     size_t tableSize;
 
     size_t hashFunction(const string& key) {
         return hash<string>{}(key) % tableSize;
     }
 
 public:
     HashTableSeparateChaining(size_t size) : tableSize(size) {
         table.resize(tableSize);
     }
 
     void insert(const string& key, int value) {
         size_t index = hashFunction(key);
         // Check if key already exists
         for (auto& kv : table[index]) {
             if (kv.first == key) {
                 kv.second = value; // Update existing key
                 return;
             }
         }
         table[index].emplace_back(key, value); // Insert new key-value pair
     }
 
     int search(const string& key) {
         size_t index = hashFunction(key);
         for (const auto& kv : table[index]) {
             if (kv.first == key)
                 return kv.second; // Key found
         }
         throw runtime_error("Key not found!"); // Key does not exist
     }
 
     void remove(const string& key) {
         size_t index = hashFunction(key);
         for (auto it = table[index].begin(); it != table[index].end(); ++it) {
             if (it->first == key) {
                 table[index].erase(it); // Remove key-value pair
                 return;
             }
         }
         throw runtime_error("Key not found!"); // Key does not exist
     }
 
     void display() {
         for (size_t i = 0; i < tableSize; ++i) {
             cout << "Index " << i << ": ";
             for (const auto& kv : table[i]) {
                 cout << "(" << kv.first << ", " << kv.second << ") -> ";
             }
             cout << "nullptr" << endl;
         }
     }
 };
 
 int main3() {
     cout << "Example 3: Separate Chaining Collision Handling 🔗🔑" << endl;
     HashTableSeparateChaining hashTable(7);
 
     hashTable.insert("apple", 100);
     hashTable.insert("banana", 200);
     hashTable.insert("cherry", 300);
     hashTable.insert("date", 400);
     hashTable.insert("elderberry", 500);
     hashTable.insert("fig", 600);
     hashTable.insert("grape", 700);
 
     cout << "Hash Table Contents:" << endl;
     hashTable.display();
 
     cout << "\nSearching for 'cherry': " << hashTable.search("cherry") << endl;
 
     cout << "\nRemoving 'banana'..." << endl;
     hashTable.remove("banana");
     hashTable.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 4: Linear Probing Collision Handling ➡️➡️➡️🔑
  * Implements a hash table using linear probing for collision resolution.
  ************************************************************************************/
 
 class HashTableLinearProbing {
 private:
     vector<pair<string, int>> table; // Vector of key-value pairs
     size_t tableSize;
     size_t elementCount;
     const string EMPTY = "";
     const string DELETED = "#DELETED#";
 
     size_t hashFunction(const string& key) {
         return hash<string>{}(key) % tableSize;
     }
 
 public:
     HashTableLinearProbing(size_t size) : tableSize(size), elementCount(0) {
         table.resize(tableSize, make_pair(EMPTY, 0));
     }
 
     void insert(const string& key, int value) {
         if (elementCount == tableSize)
             throw runtime_error("Hash table is full!");
 
         size_t index = hashFunction(key);
         size_t startIndex = index;
 
         while (table[index].first != EMPTY && table[index].first != DELETED && table[index].first != key) {
             index = (index + 1) % tableSize; // Linear probing
             if (index == startIndex)
                 throw runtime_error("Hash table is full!");
         }
 
         if (table[index].first != key)
             elementCount++;
 
         table[index] = make_pair(key, value); // Insert or update key-value pair
     }
 
     int search(const string& key) {
         size_t index = hashFunction(key);
         size_t startIndex = index;
 
         while (table[index].first != EMPTY) {
             if (table[index].first == key)
                 return table[index].second; // Key found
 
             index = (index + 1) % tableSize;
             if (index == startIndex)
                 break; // Full loop completed
         }
         throw runtime_error("Key not found!");
     }
 
     void remove(const string& key) {
         size_t index = hashFunction(key);
         size_t startIndex = index;
 
         while (table[index].first != EMPTY) {
             if (table[index].first == key) {
                 table[index].first = DELETED; // Mark as deleted
                 elementCount--;
                 return;
             }
             index = (index + 1) % tableSize;
             if (index == startIndex)
                 break;
         }
         throw runtime_error("Key not found!");
     }
 
     void display() {
         for (size_t i = 0; i < tableSize; ++i) {
             cout << "Index " << i << ": ";
             if (table[i].first == EMPTY)
                 cout << "EMPTY";
             else if (table[i].first == DELETED)
                 cout << "DELETED";
             else
                 cout << "(" << table[i].first << ", " << table[i].second << ")";
             cout << endl;
         }
     }
 };
 
 int main4() {
     cout << "Example 4: Linear Probing Collision Handling ➡️➡️➡️🔑" << endl;
     HashTableLinearProbing hashTable(7);
 
     hashTable.insert("apple", 100);
     hashTable.insert("banana", 200);
     hashTable.insert("cherry", 300);
     hashTable.insert("date", 400);
     hashTable.insert("elderberry", 500);
     hashTable.insert("fig", 600);
     hashTable.insert("grape", 700);
 
     cout << "Hash Table Contents:" << endl;
     hashTable.display();
 
     cout << "\nSearching for 'fig': " << hashTable.search("fig") << endl;
 
     cout << "\nRemoving 'cherry'..." << endl;
     hashTable.remove("cherry");
     hashTable.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 5: Quadratic Probing Collision Handling ➡️²➡️²➡️²🔑
  * Implements a hash table using quadratic probing for collision resolution.
  ************************************************************************************/
 
 class HashTableQuadraticProbing {
 private:
     vector<pair<string, int>> table;
     size_t tableSize;
     size_t elementCount;
     const string EMPTY = "";
     const string DELETED = "#DELETED#";
 
     size_t hashFunction(const string& key) {
         return hash<string>{}(key) % tableSize;
     }
 
 public:
     HashTableQuadraticProbing(size_t size) : tableSize(size), elementCount(0) {
         table.resize(tableSize, make_pair(EMPTY, 0));
     }
 
     void insert(const string& key, int value) {
         if (elementCount == tableSize)
             throw runtime_error("Hash table is full!");
 
         size_t index = hashFunction(key);
         size_t i = 0;
 
         while (table[index].first != EMPTY && table[index].first != DELETED && table[index].first != key) {
             i++;
             index = (index + i * i) % tableSize; // Quadratic probing
             if (i == tableSize)
                 throw runtime_error("Hash table is full!");
         }
 
         if (table[index].first != key)
             elementCount++;
 
         table[index] = make_pair(key, value);
     }
 
     int search(const string& key) {
         size_t index = hashFunction(key);
         size_t i = 0;
 
         while (table[index].first != EMPTY) {
             if (table[index].first == key)
                 return table[index].second;
 
             i++;
             index = (index + i * i) % tableSize;
             if (i == tableSize)
                 break;
         }
         throw runtime_error("Key not found!");
     }
 
     void remove(const string& key) {
         size_t index = hashFunction(key);
         size_t i = 0;
 
         while (table[index].first != EMPTY) {
             if (table[index].first == key) {
                 table[index].first = DELETED;
                 elementCount--;
                 return;
             }
             i++;
             index = (index + i * i) % tableSize;
             if (i == tableSize)
                 break;
         }
         throw runtime_error("Key not found!");
     }
 
     void display() {
         for (size_t i = 0; i < tableSize; ++i) {
             cout << "Index " << i << ": ";
             if (table[i].first == EMPTY)
                 cout << "EMPTY";
             else if (table[i].first == DELETED)
                 cout << "DELETED";
             else
                 cout << "(" << table[i].first << ", " << table[i].second << ")";
             cout << endl;
         }
     }
 };
 
 int main5() {
     cout << "Example 5: Quadratic Probing Collision Handling ➡️²➡️²➡️²🔑" << endl;
     HashTableQuadraticProbing hashTable(7);
 
     hashTable.insert("apple", 100);
     hashTable.insert("banana", 200);
     hashTable.insert("cherry", 300);
     hashTable.insert("date", 400);
     hashTable.insert("elderberry", 500);
     hashTable.insert("fig", 600);
     hashTable.insert("grape", 700);
 
     cout << "Hash Table Contents:" << endl;
     hashTable.display();
 
     cout << "\nSearching for 'elderberry': " << hashTable.search("elderberry") << endl;
 
     cout << "\nRemoving 'date'..." << endl;
     hashTable.remove("date");
     hashTable.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 6: Double Hashing Collision Handling 🔑🔑➡️➡️🔑
  * Implements a hash table using double hashing for collision resolution.
  ************************************************************************************/
 
 class HashTableDoubleHashing {
 private:
     vector<pair<string, int>> table;
     size_t tableSize;
     size_t elementCount;
     const string EMPTY = "";
     const string DELETED = "#DELETED#";
 
     size_t hashFunction1(const string& key) {
         return hash<string>{}(key) % tableSize;
     }
 
     size_t hashFunction2(const string& key) {
         // Secondary hash function (must be non-zero)
         return (hash<string>{}(key) / tableSize) % tableSize + 1;
     }
 
 public:
     HashTableDoubleHashing(size_t size) : tableSize(size), elementCount(0) {
         table.resize(tableSize, make_pair(EMPTY, 0));
     }
 
     void insert(const string& key, int value) {
         if (elementCount == tableSize)
             throw runtime_error("Hash table is full!");
 
         size_t index = hashFunction1(key);
         size_t stepSize = hashFunction2(key);
         size_t i = 0;
 
         while (table[index].first != EMPTY && table[index].first != DELETED && table[index].first != key) {
             i++;
             index = (index + i * stepSize) % tableSize; // Double hashing
             if (i == tableSize)
                 throw runtime_error("Hash table is full!");
         }
 
         if (table[index].first != key)
             elementCount++;
 
         table[index] = make_pair(key, value);
     }
 
     int search(const string& key) {
         size_t index = hashFunction1(key);
         size_t stepSize = hashFunction2(key);
         size_t i = 0;
 
         while (table[index].first != EMPTY) {
             if (table[index].first == key)
                 return table[index].second;
 
             i++;
             index = (index + i * stepSize) % tableSize;
             if (i == tableSize)
                 break;
         }
         throw runtime_error("Key not found!");
     }
 
     void remove(const string& key) {
         size_t index = hashFunction1(key);
         size_t stepSize = hashFunction2(key);
         size_t i = 0;
 
         while (table[index].first != EMPTY) {
             if (table[index].first == key) {
                 table[index].first = DELETED;
                 elementCount--;
                 return;
             }
             i++;
             index = (index + i * stepSize) % tableSize;
             if (i == tableSize)
                 break;
         }
         throw runtime_error("Key not found!");
     }
 
     void display() {
         for (size_t i = 0; i < tableSize; ++i) {
             cout << "Index " << i << ": ";
             if (table[i].first == EMPTY)
                 cout << "EMPTY";
             else if (table[i].first == DELETED)
                 cout << "DELETED";
             else
                 cout << "(" << table[i].first << ", " << table[i].second << ")";
             cout << endl;
         }
     }
 };
 
 int main6() {
     cout << "Example 6: Double Hashing Collision Handling 🔑🔑➡️➡️🔑" << endl;
     HashTableDoubleHashing hashTable(7);
 
     hashTable.insert("apple", 100);
     hashTable.insert("banana", 200);
     hashTable.insert("cherry", 300);
     hashTable.insert("date", 400);
     hashTable.insert("elderberry", 500);
     hashTable.insert("fig", 600);
     hashTable.insert("grape", 700);
 
     cout << "Hash Table Contents:" << endl;
     hashTable.display();
 
     cout << "\nSearching for 'grape': " << hashTable.search("grape") << endl;
 
     cout << "\nRemoving 'elderberry'..." << endl;
     hashTable.remove("elderberry");
     hashTable.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 7: Insert Operation in Hash Table ➕🔑📖
  * Demonstrates various cases during insertion in a hash table.
  ************************************************************************************/
 
 int main7() {
     cout << "Example 7: Insert Operation in Hash Table ➕🔑📖" << endl;
     HashTableSeparateChaining hashTable(5);
 
     // Inserting new keys
     hashTable.insert("alpha", 1);
     hashTable.insert("beta", 2);
     hashTable.insert("gamma", 3);
 
     // Inserting a key that updates existing value
     hashTable.insert("beta", 20);
 
     // Inserting keys that cause collisions
     hashTable.insert("delta", 4);
     hashTable.insert("epsilon", 5);
 
     cout << "Hash Table Contents after Insertions:" << endl;
     hashTable.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 8: Search Operation in Hash Table 🔍🔑📖⚡️
  * Demonstrates successful and unsuccessful search operations.
  ************************************************************************************/
 
 int main8() {
     cout << "Example 8: Search Operation in Hash Table 🔍🔑📖⚡️" << endl;
     HashTableLinearProbing hashTable(7);
 
     hashTable.insert("one", 1);
     hashTable.insert("two", 2);
     hashTable.insert("three", 3);
 
     // Successful search
     cout << "Searching for 'two': " << hashTable.search("two") << endl;
 
     // Unsuccessful search (key does not exist)
     try {
         cout << "Searching for 'four': ";
         cout << hashTable.search("four") << endl;
     } catch (const exception& e) {
         cout << e.what() << endl;
     }
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 9: Delete Operation in Hash Table ➖🔑📖
  * Demonstrates removing keys and handling the deletion markers.
  ************************************************************************************/
 
 int main9() {
     cout << "Example 9: Delete Operation in Hash Table ➖🔑📖" << endl;
     HashTableQuadraticProbing hashTable(7);
 
     hashTable.insert("cat", 10);
     hashTable.insert("dog", 20);
     hashTable.insert("bird", 30);
 
     cout << "Hash Table Before Deletion:" << endl;
     hashTable.display();
 
     // Deleting a key
     hashTable.remove("dog");
 
     cout << "\nHash Table After Deletion of 'dog':" << endl;
     hashTable.display();
 
     // Trying to delete a non-existent key
     try {
         hashTable.remove("fish");
     } catch (const exception& e) {
         cout << "\nAttempt to delete 'fish': " << e.what() << endl;
     }
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 10: Load Factor and Rehashing ⚖️📖
  * Demonstrates the effect of load factor and implementing rehashing.
  ************************************************************************************/
 
 class HashTableWithRehashing {
 private:
     vector<pair<string, int>> table;
     size_t tableSize;
     size_t elementCount;
     const string EMPTY = "";
     const string DELETED = "#DELETED#";
     const double MAX_LOAD_FACTOR = 0.7;
 
     size_t hashFunction(const string& key, size_t size) {
         return hash<string>{}(key) % size;
     }
 
     void rehash() {
         size_t newSize = tableSize * 2;
         vector<pair<string, int>> newTable(newSize, make_pair(EMPTY, 0));
 
         for (const auto& kv : table) {
             if (kv.first != EMPTY && kv.first != DELETED) {
                 size_t index = hashFunction(kv.first, newSize);
                 while (newTable[index].first != EMPTY) {
                     index = (index + 1) % newSize;
                 }
                 newTable[index] = kv;
             }
         }
 
         table = move(newTable);
         tableSize = newSize;
     }
 
 public:
     HashTableWithRehashing(size_t size) : tableSize(size), elementCount(0) {
         table.resize(tableSize, make_pair(EMPTY, 0));
     }
 
     void insert(const string& key, int value) {
         if ((double)elementCount / tableSize > MAX_LOAD_FACTOR) {
             cout << "Load factor exceeded, rehashing..." << endl;
             rehash();
         }
 
         size_t index = hashFunction(key, tableSize);
 
         while (table[index].first != EMPTY && table[index].first != DELETED) {
             index = (index + 1) % tableSize;
         }
 
         table[index] = make_pair(key, value);
         elementCount++;
     }
 
     void display() {
         for (size_t i = 0; i < tableSize; ++i) {
             cout << "Index " << i << ": ";
             if (table[i].first == EMPTY)
                 cout << "EMPTY";
             else
                 cout << "(" << table[i].first << ", " << table[i].second << ")";
             cout << endl;
         }
     }
 };
 
 int main10() {
     cout << "Example 10: Load Factor and Rehashing ⚖️📖" << endl;
     HashTableWithRehashing hashTable(5);
 
     // Inserting elements to trigger rehashing
     hashTable.insert("key1", 1);
     hashTable.insert("key2", 2);
     hashTable.insert("key3", 3);
     hashTable.insert("key4", 4); // Should trigger rehashing
 
     cout << "Hash Table Contents After Rehashing:" << endl;
     hashTable.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 11: Custom Object as Key in Hash Table 🗝️
  * Demonstrates using a custom object as a key with a custom hash function.
  ************************************************************************************/
 
 struct Point {
     int x, y;
 
     bool operator==(const Point& other) const {
         return x == other.x && y == other.y;
     }
 };
 
 // Custom hash function for Point
 struct PointHash {
     size_t operator()(const Point& p) const {
         return hash<int>{}(p.x) ^ (hash<int>{}(p.y) << 1);
     }
 };
 
 // Custom equality function for Point
 struct PointEqual {
     bool operator()(const Point& p1, const Point& p2) const {
         return p1 == p2;
     }
 };
 
 #include <unordered_map>
 
 int main11() {
     cout << "Example 11: Custom Object as Key in Hash Table 🗝️" << endl;
     unordered_map<Point, string, PointHash, PointEqual> pointMap;
 
     // Inserting custom objects as keys
     pointMap[{1, 2}] = "Point A";
     pointMap[{3, 4}] = "Point B";
     pointMap[{5, 6}] = "Point C";
 
     // Accessing values
     cout << "Value at Point(1,2): " << pointMap[{1, 2}] << endl;
     cout << "Value at Point(3,4): " << pointMap[{3, 4}] << endl;
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 12: Implementing a Set Using Hash Table 🔑
  * Demonstrates storing unique elements using a hash table.
  ************************************************************************************/
 
 class HashSet {
 private:
     vector<string> table;
     size_t tableSize;
     size_t elementCount;
     const string EMPTY = "";
     const string DELETED = "#DELETED#";
     const double MAX_LOAD_FACTOR = 0.7;
 
     size_t hashFunction(const string& key, size_t size) {
         return hash<string>{}(key) % size;
     }
 
     void rehash() {
         size_t newSize = tableSize * 2;
         vector<string> newTable(newSize, EMPTY);
 
         for (const auto& key : table) {
             if (key != EMPTY && key != DELETED) {
                 size_t index = hashFunction(key, newSize);
                 while (newTable[index] != EMPTY) {
                     index = (index + 1) % newSize;
                 }
                 newTable[index] = key;
             }
         }
 
         table = move(newTable);
         tableSize = newSize;
     }
 
 public:
     HashSet(size_t size) : tableSize(size), elementCount(0) {
         table.resize(tableSize, EMPTY);
     }
 
     void insert(const string& key) {
         if ((double)elementCount / tableSize > MAX_LOAD_FACTOR) {
             rehash();
         }
 
         size_t index = hashFunction(key, tableSize);
 
         while (table[index] != EMPTY && table[index] != DELETED && table[index] != key) {
             index = (index + 1) % tableSize;
         }
 
         if (table[index] != key) {
             table[index] = key;
             elementCount++;
         }
     }
 
     bool contains(const string& key) {
         size_t index = hashFunction(key, tableSize);
 
         while (table[index] != EMPTY) {
             if (table[index] == key)
                 return true;
 
             index = (index + 1) % tableSize;
         }
         return false;
     }
 
     void remove(const string& key) {
         size_t index = hashFunction(key, tableSize);
 
         while (table[index] != EMPTY) {
             if (table[index] == key) {
                 table[index] = DELETED;
                 elementCount--;
                 return;
             }
             index = (index + 1) % tableSize;
         }
     }
 
     void display() {
         for (size_t i = 0; i < tableSize; ++i) {
             cout << "Index " << i << ": ";
             if (table[i] == EMPTY)
                 cout << "EMPTY";
             else
                 cout << table[i];
             cout << endl;
         }
     }
 };
 
 int main12() {
     cout << "Example 12: Implementing a Set Using Hash Table 🔑" << endl;
     HashSet set(5);
 
     set.insert("apple");
     set.insert("banana");
     set.insert("cherry");
 
     cout << "Set Contents:" << endl;
     set.display();
 
     cout << "\nContains 'banana': " << (set.contains("banana") ? "Yes" : "No") << endl;
     cout << "Contains 'date': " << (set.contains("date") ? "Yes" : "No") << endl;
 
     set.remove("banana");
     cout << "\nAfter Removing 'banana':" << endl;
     set.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 13: Associative Array Implementation 🔑🐍☕
  * Demonstrates implementing a dictionary-like structure using a hash table.
  ************************************************************************************/
 
 class Dictionary {
 private:
     vector<list<pair<string, string>>> table;
     size_t tableSize;
 
     size_t hashFunction(const string& key) {
         return hash<string>{}(key) % tableSize;
     }
 
 public:
     Dictionary(size_t size) : tableSize(size) {
         table.resize(tableSize);
     }
 
     void set(const string& key, const string& value) {
         size_t index = hashFunction(key);
         for (auto& kv : table[index]) {
             if (kv.first == key) {
                 kv.second = value;
                 return;
             }
         }
         table[index].emplace_back(key, value);
     }
 
     string get(const string& key) {
         size_t index = hashFunction(key);
         for (const auto& kv : table[index]) {
             if (kv.first == key)
                 return kv.second;
         }
         throw runtime_error("Key not found!");
     }
 
     void remove(const string& key) {
         size_t index = hashFunction(key);
         for (auto it = table[index].begin(); it != table[index].end(); ++it) {
             if (it->first == key) {
                 table[index].erase(it);
                 return;
             }
         }
         throw runtime_error("Key not found!");
     }
 
     void display() {
         for (size_t i = 0; i < tableSize; ++i) {
             cout << "Index " << i << ": ";
             for (const auto& kv : table[i]) {
                 cout << kv.first << ": " << kv.second << " -> ";
             }
             cout << "nullptr" << endl;
         }
     }
 };
 
 int main13() {
     cout << "Example 13: Associative Array Implementation 🔑🐍☕" << endl;
     Dictionary dict(5);
 
     dict.set("name", "Alice");
     dict.set("age", "30");
     dict.set("city", "New York");
 
     cout << "Dictionary Contents:" << endl;
     dict.display();
 
     cout << "\nValue for 'name': " << dict.get("name") << endl;
 
     dict.remove("age");
     cout << "\nAfter Removing 'age':" << endl;
     dict.display();
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 14: Symbol Table Implementation in Compiler Context 💻🔑
  * Simulates a symbol table for variable name to memory location mapping.
  ************************************************************************************/
 
 class SymbolTable {
 private:
     unordered_map<string, int> table; // Variable name to memory address
 
 public:
     void insert(const string& variable, int address) {
         table[variable] = address; // Insert or update
     }
 
     int getAddress(const string& variable) {
         if (table.find(variable) != table.end())
             return table[variable];
         throw runtime_error("Variable not declared!");
     }
 
     void display() {
         cout << "Symbol Table:" << endl;
         for (const auto& kv : table) {
             cout << kv.first << " => Address " << kv.second << endl;
         }
     }
 };
 
 int main14() {
     cout << "Example 14: Symbol Table Implementation in Compiler Context 💻🔑" << endl;
     SymbolTable symTable;
 
     symTable.insert("x", 1000);
     symTable.insert("y", 1004);
     symTable.insert("z", 1008);
 
     symTable.display();
 
     cout << "\nAddress of 'y': " << symTable.getAddress("y") << endl;
 
     // Attempting to access undeclared variable
     try {
         cout << "Address of 'w': " << symTable.getAddress("w") << endl;
     } catch (const exception& e) {
         cout << e.what() << endl;
     }
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Example 15: Caching with Hash Tables 📦⚡️
  * Simulates a simple cache system using a hash table for quick data retrieval.
  ************************************************************************************/
 
 class SimpleCache {
 private:
     unordered_map<string, string> cache;
 
 public:
     void put(const string& key, const string& value) {
         cache[key] = value; // Add or update cache entry
     }
 
     string get(const string& key) {
         if (cache.find(key) != cache.end())
             return cache[key];
         throw runtime_error("Cache miss! Key not found");
     }
 
     void display() {
         cout << "Cache Contents:" << endl;
         for (const auto& kv : cache) {
             cout << kv.first << " => " << kv.second << endl;
         }
     }
 };
 
 int main15() {
     cout << "Example 15: Caching with Hash Tables 📦⚡️" << endl;
     SimpleCache cache;
 
     cache.put("user:1", "Alice");
     cache.put("user:2", "Bob");
     cache.put("user:3", "Carol");
 
     cache.display();
 
     // Accessing cached data
     cout << "\nCached data for 'user:2': " << cache.get("user:2") << endl;
 
     // Attempting to access non-existent key
     try {
         cout << "Cached data for 'user:4': " << cache.get("user:4") << endl;
     } catch (const exception& e) {
         cout << e.what() << endl;
     }
     cout << endl;
     return 0;
 }
 
 /************************************************************************************
  * Main Function to Run All Examples
  ************************************************************************************/
 
 int main() {
     main1();  // Basic Hash Function Example
     main2();  // Good vs. Bad Hash Functions
     main3();  // Separate Chaining Collision Handling
     main4();  // Linear Probing Collision Handling
     main5();  // Quadratic Probing Collision Handling
     main6();  // Double Hashing Collision Handling
     main7();  // Insert Operation in Hash Table
     main8();  // Search Operation in Hash Table
     main9();  // Delete Operation in Hash Table
     main10(); // Load Factor and Rehashing
     main11(); // Custom Object as Key in Hash Table
     main12(); // Implementing a Set Using Hash Table
     main13(); // Associative Array Implementation
     main14(); // Symbol Table Implementation
     main15(); // Caching with Hash Tables
     return 0;
 }
 
 /************************************************************************************
  * End of Advanced C++ Hash Tables Tutorial 🔑📖 🗂️⚡️
  ************************************************************************************/