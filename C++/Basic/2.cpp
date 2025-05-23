#include <iostream>
#include <vector>
#include <map>
#include <string>

using namespace std;

int main() {
    // Variables
    int age = 25; //    int age = 25; // integer general 4 bytes
    double height = 1.75; // double height = 1.75; // double general 8 bytes
    string name = "John Doe"; // string name = "John Doe"; 
    

    // List (Vector)
    vector<int> numbers = {1, 2, 3, 4, 5};

    // Dictionary (Map)
    map<string, string> person = {
        {"name", "Alice"},
        {"city", "New York"},
        {"job", "Engineer"}
    };

    // Printing variables
    cout << "Name: " << name << endl;
    cout << "Age: " << age << endl;
    cout << "Height: " << height << " meters" << endl;

    // Printing list elements
    cout << "Numbers: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;

    // Printing dictionary elements
    cout << "Person details:" << endl;
    for (const auto& pair : person) {
        cout << pair.first << ": " << pair.second << endl;
    }

    return 0;
}