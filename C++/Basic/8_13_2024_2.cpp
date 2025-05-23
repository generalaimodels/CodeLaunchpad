#include <iostream>
#include <vector>
#include <tuple>
#include <set>
#include <map>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>  // Include this header for set_union and set_intersection

int Data_Structures_Cotinue1() {
    // 5.2. The del statement
    std::cout << "5.2. The del statement equivalent:" << std::endl;
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // Remove the last element (similar to del numbers[-1] in Python)
    numbers.pop_back();
    
    // Remove an element at a specific index (similar to del numbers[1] in Python)
    numbers.erase(numbers.begin() + 1);
    
    // Print the modified vector
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 5.3. Tuples and Sequences
    std::cout << "\n5.3. Tuples and Sequences:" << std::endl;
    
    // Tuple
    std::tuple<int, std::string, double> person(30, "John", 5.8);
    std::cout << "Age: " << std::get<0>(person) << ", Name: " << std::get<1>(person) 
              << ", Height: " << std::get<2>(person) << std::endl;
    
    // Sequence (using vector as an example)
    std::vector<int> sequence = {1, 2, 3, 4, 5};
    std::cout << "Sequence: ";
    for (int num : sequence) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 5.4. Sets
    std::cout << "\n5.4. Sets:" << std::endl;
    
    // Ordered set
    std::set<int> orderedSet = {3, 1, 4, 1, 5, 9, 2, 6, 5};
    std::cout << "Ordered Set: ";
    for (int num : orderedSet) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // Unordered set (hash set)
    std::unordered_set<int> hashSet = {3, 1, 4, 1, 5, 9, 2, 6, 5};
    std::cout << "Unordered Set: ";
    for (int num : hashSet) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    // Set operations
    std::set<int> set1 = {1, 2, 3, 4, 5};
    std::set<int> set2 = {4, 5, 6, 7, 8};
    std::set<int> unionSet, intersectionSet;
    
    std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), 
                   std::inserter(unionSet, unionSet.begin()));
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), 
                          std::inserter(intersectionSet, intersectionSet.begin()));
    
    std::cout << "Union: ";
    for (int num : unionSet) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Intersection: ";
    for (int num : intersectionSet) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // 5.5. Dictionaries
    std::cout << "\n5.5. Dictionaries:" << std::endl;
    
    // Ordered dictionary (map)
    std::map<std::string, int> orderedDict = {{"apple", 1}, {"banana", 2}, {"orange", 3}};
    std::cout << "Ordered Dictionary:" << std::endl;
    for (const auto& pair : orderedDict) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    // Unordered dictionary (hash map)
    std::unordered_map<std::string, int> hashDict = {{"apple", 1}, {"banana", 2}, {"orange", 3}};
    std::cout << "Unordered Dictionary:" << std::endl;
    for (const auto& pair : hashDict) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    // Adding and removing elements
    hashDict["grape"] = 4;  // Add a new key-value pair
    hashDict.erase("banana");  // Remove a key-value pair
    
    std::cout << "Modified Unordered Dictionary:" << std::endl;
    for (const auto& pair : hashDict) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}

int Data_Structures_Cotinue2() {
    // 5.2. The del statement
    std::cout << "5.2. The del statement:\n";
    // C++ doesn't have a direct equivalent to Python's 'del' statement.
    // Instead, we can use erase() for containers or delete for dynamically allocated objects.
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Before: ";
    for (int num : numbers) std::cout << num << " ";
    std::cout << "\n";

    numbers.erase(numbers.begin() + 2); // Remove the third element (index 2)
    std::cout << "After: ";
    for (int num : numbers) std::cout << num << " ";
    std::cout << "\n\n";

    // 5.3. Tuples and Sequences
    std::cout << "5.3. Tuples and Sequences:\n";
    // C++ has std::tuple for fixed-size collections of heterogeneous types
    std::tuple<int, std::string, double> person(30, "John", 5.8);
    std::cout << "Age: " << std::get<0>(person) << "\n";
    std::cout << "Name: " << std::get<1>(person) << "\n";
    std::cout << "Height: " << std::get<2>(person) << "\n\n";

    // 5.4. Sets
    std::cout << "5.4. Sets:\n";
    std::set<int> uniqueNumbers;
    uniqueNumbers.insert(5);
    uniqueNumbers.insert(2);
    uniqueNumbers.insert(5); // Duplicate, won't be added
    uniqueNumbers.insert(1);

    std::cout << "Set contents: ";
    for (int num : uniqueNumbers) std::cout << num << " ";
    std::cout << "\n";

    if (uniqueNumbers.find(2) != uniqueNumbers.end()) {
        std::cout << "2 is in the set\n";
    }

    uniqueNumbers.erase(2);
    std::cout << "After removing 2: ";
    for (int num : uniqueNumbers) std::cout << num << " ";
    std::cout << "\n\n";

    // 5.5. Dictionaries
    std::cout << "5.5. Dictionaries:\n";
    // C++ uses std::map or std::unordered_map for dictionary-like functionality
    std::map<std::string, int> ages;
    ages["Alice"] = 30;
    ages["Bob"] = 25;
    ages["Charlie"] = 35;

    std::cout << "Ages:\n";
    for (const auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << "\n";
    }

    std::cout << "Bob's age: " << ages["Bob"] << "\n";

    ages.erase("Alice");
    std::cout << "After removing Alice:\n";
    for (const auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << "\n";
    }

    if (ages.find("David") == ages.end()) {
        std::cout << "David is not in the dictionary\n";
    }

    return 0;
}

int main() {
    Data_Structures_Cotinue1();
    Data_Structures_Cotinue2();
    return 0;
}
