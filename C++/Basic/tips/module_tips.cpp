// This code demonstrates the detailed API of the C++ Standard Library's <vector>.
// It is structured in a class with static member functions to showcase various APIs
// such as constructors, element access, capacity management, modifiers, iterators, etc.

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

class VectorAPIDemo {
public:
    // Demonstrates various constructors of std::vector.
    static void demonstrateConstruction() {
        // 1. Default constructor: creates an empty vector.
        std::vector<int> vecDefault;
        
        // 2. Fill constructor: creates a vector with 5 elements, each initialized to 10.
        std::vector<int> vecFill(5, 10);
        
        // 3. Range constructor: creates a vector by copying elements from another container.
        std::vector<int> source = {1, 2, 3, 4, 5};
        std::vector<int> vecRange(source.begin(), source.end());
        
        // 4. Copy constructor: creates a new vector as a copy of an existing vector.
        std::vector<int> vecCopy(vecRange);
        
        // 5. Move constructor: moves the contents of one vector to another.
        std::vector<int> vecMove(std::move(vecCopy));
    }
    
    // Demonstrates element access member functions.
    static void demonstrateElementAccess() {
        std::vector<int> vec = {10, 20, 30, 40, 50};

        // operator[]: returns reference to element without bounds checking.
        int elementViaSubscript = vec[2];  // 30

        // at(): returns reference to element with bounds checking (throws std::out_of_range if invalid).
        int elementViaAt = vec.at(2);

        // front(): returns reference to the first element.
        int firstElement = vec.front();

        // back(): returns reference to the last element.
        int lastElement = vec.back();

        // data(): returns pointer to the underlying array.
        int* rawData = vec.data();

        std::cout << "Element Access:\n"
                  << "  operator[]: " << elementViaSubscript << "\n"
                  << "  at(): " << elementViaAt << "\n"
                  << "  front(): " << firstElement << "\n"
                  << "  back(): " << lastElement << "\n\n";
    }
    
    // Demonstrates capacity-related functions.
    static void demonstrateCapacity() {
        std::vector<int> vec;

        // reserve(): pre-allocates memory for at least the specified number of elements.
        vec.reserve(100);

        // capacity(): returns the total number of elements that can be held in currently allocated storage.
        size_t currentCapacity = vec.capacity();

        // size(): returns the number of elements in the vector.
        size_t currentSize = vec.size();

        // empty(): checks whether the vector is empty.
        bool isEmpty = vec.empty();

        // shrink_to_fit(): requests the reduction of capacity to fit size.
        vec.shrink_to_fit();

        std::cout << "Capacity Management:\n"
                  << "  Capacity: " << currentCapacity << "\n"
                  << "  Size: " << currentSize << "\n"
                  << "  Empty: " << std::boolalpha << isEmpty << "\n\n";
    }
    
    // Demonstrates modifier functions.
    static void demonstrateModifiers() {
        std::vector<int> vec = {1, 2, 3};

        // push_back(): appends an element at the end.
        vec.push_back(4);

        // pop_back(): removes the last element.
        vec.pop_back();

        // insert(): inserts an element at the specified position.
        vec.insert(vec.begin() + 1, 10);  // Inserts 10 at index 1.

        // erase(): removes element(s) from the vector.
        vec.erase(vec.begin() + 2);       // Erases element at index 2.

        // emplace(): constructs an element in-place at the specified position.
        vec.emplace(vec.begin(), 0);      // Inserts 0 at the beginning.

        // emplace_back(): constructs an element in-place at the end.
        vec.emplace_back(5);

        // clear(): removes all elements from the vector.
        vec.clear();

        // swap(): swaps the contents with another vector.
        std::vector<int> vecOther = {7, 8, 9};
        vec.swap(vecOther);

        // Assignment operator: copies the contents from one vector to another.
        std::vector<int> vecAssigned = vecOther;

        std::cout << "Modifiers Demo:\n"
                  << "  Swapped vector size: " << vecOther.size() << "\n\n";
    }
    
    // Demonstrates iterator functionality.
    static void demonstrateIterators() {
        std::vector<int> vec = {1, 2, 3, 4, 5};

        std::cout << "Iterators (forward): ";
        // Using iterator to traverse the vector.
        for (auto it = vec.begin(); it != vec.end(); ++it)
            std::cout << *it << " ";
        std::cout << "\n";

        std::cout << "Iterators (reverse): ";
        // Using reverse iterator to traverse the vector in reverse.
        for (auto rit = vec.rbegin(); rit != vec.rend(); ++rit)
            std::cout << *rit << " ";
        std::cout << "\n";

        std::cout << "Iterators (const): ";
        // Using const_iterator to traverse the vector without allowing modification.
        for (auto cit = vec.cbegin(); cit != vec.cend(); ++cit)
            std::cout << *cit << " ";
        std::cout << "\n\n";
    }
    
    // Demonstrates non-member functions associated with std::vector.
    static void demonstrateNonMemberFunctions() {
        std::vector<int> vec1 = {1, 2, 3};
        std::vector<int> vec2 = {1, 2, 3};

        // Equality operator: compares two vectors.
        bool areEqual = (vec1 == vec2);

        // std::swap: swaps the contents of two vectors.
        std::swap(vec1, vec2);

        std::cout << "Non-Member Functions:\n"
                  << "  Vectors are equal: " << std::boolalpha << areEqual << "\n\n";
    }
};

int main() {
    VectorAPIDemo::demonstrateConstruction();
    VectorAPIDemo::demonstrateElementAccess();
    VectorAPIDemo::demonstrateCapacity();
    VectorAPIDemo::demonstrateModifiers();
    VectorAPIDemo::demonstrateIterators();
    VectorAPIDemo::demonstrateNonMemberFunctions();
    
    return 0;
}
