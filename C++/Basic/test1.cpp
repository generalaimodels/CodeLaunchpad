#include <iostream> //
#include <vector> //
#include <map>  //
#include <string> // string libraray

using namespace std;

int main() {
    // Variables
    
    int age = 25;
    double height = 1.75;
    char grade = 'A';
    bool isStudent = true;

    // Printing variables
    cout << "Age: " << age << endl;
    cout << "Height: " << height << " meters" << endl;
    cout << "Grade: " << grade << endl;
    cout << "Is student: " << (isStudent ? "Yes" : "No") << endl;

    // String
    string name = "John Doe";
    cout << "Name: " << name << endl;

    // List (Vector)
    vector<int> numbers = {1, 2, 3, 4, 5};
    cout << "Numbers: ";
    for (int num : numbers) {
        cout << num << " ";
    }
    cout << endl;

    // Dictionary (Map)
    map<string, int> scores;
    scores["Alice"] = 95;
    scores["Bob"] = 87;
    scores["Charlie"] = 92;

    cout << "Scores:" << endl;
    for (const auto& pair : scores) {
        cout << pair.first << ": " << pair.second << endl;
    }

    return 0;
}