#include <iostream>
#include <list>
#include <tuple>
#include <string>
#include <unordered_map>
#include <set>
#include <vector>
#include <map>
int List_Operation_Insert() {
    std::list<int> myList = {1, 2, 3, 4, 5};
    
    // Insert element at the beginning
    myList.push_front(0);
    
    // Insert element at the end
    myList.push_back(6);
    
    // Insert element at a specific position
    auto it = std::next(myList.begin(), 3);
    myList.insert(it, 10);
    
    // Remove element
    myList.remove(4);
    
    // Iterate and print
    for (const auto& elem : myList) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    
    return 0;
}



int Tuple_Operation_Insert() {
    std::tuple<int, std::string, double> myTuple(42, "Hello", 3.14);
    
    // Access elements
    std::cout << std::get<0>(myTuple) << std::endl;
    std::cout << std::get<1>(myTuple) << std::endl;
    std::cout << std::get<2>(myTuple) << std::endl;
    
    // Modify elements
    std::get<1>(myTuple) = "World";
    
    // Unpack tuple
    int a;
    std::string b;
    double c;
    std::tie(a, b, c) = myTuple;
    
    std::cout << a << ", " << b << ", " << c << std::endl;
    
    return 0;
}



int Dictionary_Operation_Insert() {
    std::unordered_map<std::string, int> myDict = {
        {"apple", 5},
        {"banana", 3},
        {"orange", 7}
    };
    
    // Insert or update a key-value pair
    myDict["grape"] = 4;
    
    // Access value by key
    std::cout << "Value of 'banana': " << myDict["banana"] << std::endl;
    
    // Check if key exists
    if (myDict.find("kiwi") == myDict.end()) {
        std::cout << "Key 'kiwi' not found" << std::endl;
    }
    
    // Iterate and print
    for (const auto& pair : myDict) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    return 0;
}



int Set_Operation_Insert() {
    std::set<int> mySet = {5, 2, 8, 1, 9};
    
    // Insert elements
    mySet.insert(3);
    mySet.insert(7);
    
    // Check if element exists
    if (mySet.find(4) == mySet.end()) {
        std::cout << "Element 4 not found" << std::endl;
    }
    
    // Remove element
    mySet.erase(2);
    
    // Iterate and print
    for (const auto& elem : mySet) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    
    return 0;
}



int More_On_Control_Flow() {
    int x = 10;
    int y = 20;
    
    // If-else statement
    if (x > y) {
        std::cout << "x is greater than y" << std::endl;
    } else if (x < y) {
        std::cout << "x is less than y" << std::endl;
    } else {
        std::cout << "x is equal to y" << std::endl;
    }
    
    // Ternary operator
    int max = (x > y) ? x : y;
    std::cout << "Max value: " << max << std::endl;
    
    // Switch statement
    int choice = 2;
    switch (choice) {
        case 1:
            std::cout << "Option 1 selected" << std::endl;
            break;
        case 2:
            std::cout << "Option 2 selected" << std::endl;
            break;
        default:
            std::cout << "Invalid option" << std::endl;
    }
    
    return 0;
}



int Compare_Two_Objects() {
    std::vector<int> vec1 = {1, 2, 3};
    std::vector<int> vec2 = {1, 2, 3, 4};
    
    // Compare vectors
    if (vec1 == vec2) {
        std::cout << "Vectors are equal" << std::endl;
    } else if (vec1 < vec2) {
        std::cout << "vec1 is less than vec2" << std::endl;
    } else {
        std::cout << "vec1 is greater than vec2" << std::endl;
    }
    
    std::string str1 = "hello";
    std::string str2 = "world";
    
    // Compare strings
    if (str1 == str2) {
        std::cout << "Strings are equal" << std::endl;
    } else if (str1 < str2) {
        std::cout << "str1 is less than str2" << std::endl;
    } else {
        std::cout << "str1 is greater than str2" << std::endl;
    }
    
    return 0;
}



int If_Else_Ternary_Switch() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // If-else statement
    if (numbers.size() > 3) {
        std::cout << "Vector has more than 3 elements" << std::endl;
    } else {
        std::cout << "Vector has 3 or fewer elements" << std::endl;
    }

    // Ternary operator
    int x = 10;
    int y = (x > 5) ? 1 : 0;

    // Switch statement
    int choice = 2;
    switch (choice) {
        case 1:
            std::cout << "Option 1 selected" << std::endl;
            break;
        case 2:
            std::cout << "Option 2 selected" << std::endl;
            break;
        default:
            std::cout << "Invalid option" << std::endl;
    }

    // Short-circuit evaluation
    if (numbers.size() > 0 && numbers[0] == 1) {
        std::cout << "First element is 1" << std::endl;
    }

    return 0;
}


int List_Operation_Insert1() {
    // Using std::vector (dynamic array)
    std::vector<int> vec = {1, 2, 3, 4, 5};
    vec.push_back(6);
    vec.pop_back();

    // Using std::list (doubly-linked list)
    std::list<int> lst = {1, 2, 3, 4, 5};
    lst.push_back(6);
    lst.pop_front();

    // Iterating through a vector
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;

    return 0;
}


int Dictionary_Operation_Insert1() {
    // Using std::map (ordered)
    std::map<std::string, int> ordered_map = {
        {"apple", 1},
        {"banana", 2},
        {"orange", 3}
    };

    // Using std::unordered_map (unordered)
    std::unordered_map<std::string, int> unordered_map = {
        {"apple", 1},
        {"banana", 2},
        {"orange", 3}
    };

    // Accessing and modifying elements
    ordered_map["grape"] = 4;
    std::cout << "Value of 'banana': " << ordered_map["banana"] << std::endl;

    // Iterating through a map
    for (const auto& pair : ordered_map) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}

int Range_Based_For_Loop() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::map<std::string, int> fruits = {
        {"apple", 1},
        {"banana", 2},
        {"orange", 3}
    };

    // Range-based for loop
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Traditional for loop
    for (size_t i = 0; i < numbers.size(); ++i) {
        std::cout << numbers[i] << " ";
    }
    std::cout << std::endl;

    // While loop
    size_t j = 0;
    while (j < numbers.size()) {
        std::cout << numbers[j] << " ";
        ++j;
    }
    std::cout << std::endl;

    // Do-while loop
    size_t k = 0;
    do {
        std::cout << numbers[k] << " ";
        ++k;
    } while (k < numbers.size());
    std::cout << std::endl;

    // Iterating through a map
    for (const auto& pair : fruits) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}



int Tuple_Operation_Insert1() {
    // Creating a tuple
    std::tuple<int, std::string, double> my_tuple(42, "Hello", 3.14);

    // Accessing tuple elements
    std::cout << std::get<0>(my_tuple) << std::endl;
    std::cout << std::get<1>(my_tuple) << std::endl;
    std::cout << std::get<2>(my_tuple) << std::endl;

    // Unpacking a tuple
    int a;
    std::string b;
    double c;
    std::tie(a, b, c) = my_tuple;

    // Creating a tuple with auto
    auto another_tuple = std::make_tuple(1, "World", 2.71);

    return 0;
}
int main() {
    List_Operation_Insert();
    List_Operation_Insert1();
    Tuple_Operation_Insert();
    Tuple_Operation_Insert1();
    Dictionary_Operation_Insert();
    Dictionary_Operation_Insert1();
    Set_Operation_Insert();
    More_On_Control_Flow();
    Compare_Two_Objects();
    If_Else_Ternary_Switch();
    Range_Based_For_Loop();
    return 0;
}