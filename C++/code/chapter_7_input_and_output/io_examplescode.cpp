// chapter7_io_examplescode.cpp

/*
 * Chapter 7: Input and Output (I/O) in C++
 * Author: Kandimalla Hemanth
 * 
 *
 * This program contains 20 examples demonstrating standard I/O and file I/O in C++.
 * Each example covers different aspects, from basic to advanced, including exception handling.
 */

// Include necessary headers
#include <iostream>
#include <iomanip>   // For manipulators
#include <fstream>   // For file streams
#include <string>
#include <sstream>   // For string streams
#include <limits>    // For numeric limits
#include <exception> // For exception handling

// Use the std namespace for convenience
using namespace std;

// Main function
int main() {
    // Example separators for clarity
    cout << "===== Standard Input and Output Examples =====" << endl << endl;

    // ==========================
    // **Standard Input and Output Examples**
    // ==========================

    // Example 1: Basic Input and Output
    {
        cout << "[Example 1] Basic Input and Output" << endl;
        string name;
        int age;

        cout << "Enter your name: ";
        cin >> name; // Reads up to whitespace
        cout << "Enter your age: ";
        cin >> age;

        cout << "Hello, " << name << "! You are " << age << " years old." << endl << endl;
    }

    // Example 2: Reading Multiple Values
    {
        cout << "[Example 2] Reading Multiple Values" << endl;
        int num1, num2, sum;

        cout << "Enter two integers separated by space: ";
        cin >> num1 >> num2;

        sum = num1 + num2;
        cout << "The sum is: " << sum << endl << endl;
    }

    // Example 3: Handling Whitespace in Input
    {
        cout << "[Example 3] Handling Whitespace in Input" << endl;
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear input buffer

        string fullName;
        cout << "Enter your full name: ";
        getline(cin, fullName);

        cout << "Hello, " << fullName << "!" << endl << endl;
    }

    // Example 4: Input Validation
    {
        cout << "[Example 4] Input Validation" << endl;
        int number;
        cout << "Enter an integer: ";

        while (!(cin >> number)) {
            // Input failed, handle the error
            cin.clear(); // Clear error flags
            cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Discard invalid input
            cout << "Invalid input. Please enter an integer: ";
        }

        cout << "You entered: " << number << endl << endl;
    }

    // Example 5: Formatting Output with Manipulators
    {
        cout << "[Example 5] Formatting Output with Manipulators" << endl;

        // Table header
        cout << left << setw(10) << "Number" << right << setw(10) << "Square" << endl;
        cout << setw(20) << setfill('-') << "-" << setfill(' ') << endl; // Separator

        // Display numbers and their squares
        for (int i = 1; i <= 5; ++i) {
            cout << left << setw(10) << i << right << setw(10) << i * i << endl;
        }
        cout << endl;
    }

    // Example 6: Using std::cerr and std::clog
    {
        cout << "[Example 6] Using std::cerr and std::clog" << endl;

        cout << "Program started successfully." << endl;
        clog << "Log: This is a log message." << endl;
        cerr << "Error: An error occurred!" << endl << endl;
    }

    // Example 7: Reading Numerical Data with Delimiters
    {
        cout << "[Example 7] Reading Numerical Data with Delimiters" << endl;
        int a, b, c;

        cout << "Enter three integers separated by commas (e.g., 1,2,3): ";
        cin >> a;
        cin.ignore(); // Ignore comma
        cin >> b;
        cin.ignore(); // Ignore comma
        cin >> c;

        cout << "You entered: " << a << ", " << b << ", " << c << endl << endl;
    }

    // Example 8: Advanced Input with String Streams
    {
        cout << "[Example 8] Advanced Input with String Streams" << endl;
        cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Clear input buffer

        string input;
        cout << "Enter integers separated by spaces: ";
        getline(cin, input);

        istringstream stream(input);
        int sum = 0, number;

        while (stream >> number) {
            sum += number;
        }

        cout << "The sum is: " << sum << endl << endl;
    }

    // Example 9: Setting Precision for Floating-Point Output
    {
        cout << "[Example 9] Setting Precision for Floating-Point Output" << endl;
        double pi = 3.14159265359;

        cout << "Default precision: " << pi << endl;
        cout << "Precision 2: " << fixed << setprecision(2) << pi << endl;
        cout << "Precision 5: " << fixed << setprecision(5) << pi << endl << endl;
    }

    // Example 10: Using std::noskipws Manipulator
    {
        cout << "[Example 10] Using std::noskipws Manipulator" << endl;
        char ch;

        cout << "Enter characters (type 'q' to quit): ";

        cin >> noskipws; // Do not skip whitespaces
        while (cin >> ch && ch != 'q') {
            cout << "You entered: [" << ch << "]" << endl;
        }
        cout << endl;

        cin >> skipws; // Reset to skip whitespaces
    }

    cout << "===== File Input and Output Examples =====" << endl << endl;

    // ==========================
    // **File Input and Output Examples**
    // ==========================

    // Example 1: Writing Text to a File
    {
        cout << "[File Example 1] Writing Text to a File" << endl;
        ofstream outFile("example1.txt");

        if (outFile.is_open()) {
            outFile << "Hello, this is a test file." << endl;
            outFile << "Writing text to a file in C++ is straightforward!" << endl;
            outFile.close();
            cout << "Data written to file successfully." << endl << endl;
        } else {
            cerr << "Error: Unable to open file for writing." << endl << endl;
        }
    }

    // Example 2: Reading Text from a File
    {
        cout << "[File Example 2] Reading Text from a File" << endl;
        ifstream inFile("example1.txt");
        string line;

        if (inFile.is_open()) {
            while (getline(inFile, line)) {
                cout << line << endl;
            }
            inFile.close();
            cout << endl;
        } else {
            cerr << "Error: Unable to open file for reading." << endl << endl;
        }
    }

    // Example 3: Appending to a File
    {
        cout << "[File Example 3] Appending to a File" << endl;
        ofstream outFile("example1.txt", ios::app);

        if (outFile.is_open()) {
            outFile << "This line is appended to the file." << endl;
            outFile.close();
            cout << "Data appended to file successfully." << endl << endl;
        } else {
            cerr << "Error: Unable to open file for appending." << endl << endl;
        }
    }

    // Example 4: Reading and Writing Binary Files
    {
        cout << "[File Example 4] Reading and Writing Binary Files" << endl;
        struct Data {
            int id;
            double value;
        };

        // Create data to write
        Data dataOut = {1, 99.99};

        // Write binary data to file
        ofstream outFile("data.bin", ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<char*>(&dataOut), sizeof(Data));
            outFile.close();
            cout << "Binary data written to file." << endl;
        } else {
            cerr << "Error: Unable to open file for binary writing." << endl;
        }

        // Read binary data from file
        Data dataIn;
        ifstream inFile("data.bin", ios::binary);
        if (inFile.is_open()) {
            inFile.read(reinterpret_cast<char*>(&dataIn), sizeof(Data));
            inFile.close();
            cout << "Binary data read from file." << endl;
            cout << "ID: " << dataIn.id << ", Value: " << dataIn.value << endl << endl;
        } else {
            cerr << "Error: Unable to open file for binary reading." << endl << endl;
        }
    }

    // Example 5: File Positioning with Seek
    {
        cout << "[File Example 5] File Positioning with Seek" << endl;
        ofstream outFile("numbers.txt");

        // Write numbers to file
        if (outFile.is_open()) {
            for (int i = 1; i <= 10; ++i) {
                outFile << i << endl;
            }
            outFile.close();
        }

        ifstream inFile("numbers.txt");
        if (inFile.is_open()) {
            inFile.seekg(0, ios::end); // Move to end of file
            streampos fileSize = inFile.tellg();

            cout << "File size: " << fileSize << " bytes." << endl;

            // Read the last number (approximate method)
            inFile.seekg(-3, ios::end); // Move back a few bytes
            int lastNumber;
            inFile >> lastNumber;
            cout << "Last number in file: " << lastNumber << endl << endl;

            inFile.close();
        } else {
            cerr << "Error: Unable to open file for seeking." << endl << endl;
        }
    }

    // Example 6: Handling File Open Exceptions
    {
        cout << "[File Example 6] Handling File Open Exceptions" << endl;
        try {
            ifstream inFile("nonexistent.txt");
            if (!inFile) {
                throw ios_base::failure("Failed to open file.");
            }
            // File operations would go here
            inFile.close();
        } catch (const ios_base::failure& e) {
            cerr << "File I/O exception caught: " << e.what() << endl << endl;
        }
    }

    // Example 7: Reading a File into a String
    {
        cout << "[File Example 7] Reading a File into a String" << endl;
        ifstream inFile("example1.txt");
        stringstream buffer;

        if (inFile.is_open()) {
            buffer << inFile.rdbuf(); // Read the file into the stringstream
            inFile.close();

            // Get the string from the buffer
            string contents = buffer.str();
            cout << "File contents:\n" << contents << endl << endl;
        } else {
            cerr << "Error: Unable to open file for reading into string." << endl << endl;
        }
    }

    // Example 8: Checking for End-of-File
    {
        cout << "[File Example 8] Checking for End-of-File" << endl;
        ifstream inFile("numbers.txt");
        int number;

        if (inFile.is_open()) {
            while (inFile >> number) {
                cout << "Read number: " << number << endl;
            }

            if (inFile.eof()) {
                cout << "Reached end of file." << endl << endl;
            } else {
                cerr << "An error occurred before reaching EOF." << endl << endl;
            }

            inFile.close();
        } else {
            cerr << "Error: Unable to open file for EOF testing." << endl << endl;
        }
    }

    // Example 9: File Modes and Truncation
    {
        cout << "[File Example 9] File Modes and Truncation" << endl;
        // Open file with truncation mode
        ofstream outFile("example1.txt", ios::out | ios::trunc);

        if (outFile.is_open()) {
            outFile << "This is new content; previous content is deleted." << endl;
            outFile.close();
            cout << "File truncated and new data written." << endl << endl;
        } else {
            cerr << "Error: Unable to open file for truncation." << endl << endl;
        }
    }

    // Example 10: Synchronized I/O with std::sync_with_stdio
    {
        cout << "[File Example 10] Synchronized I/O with std::sync_with_stdio" << endl;
        ios::sync_with_stdio(false); // Disable synchronization for performance

        int n;
        cout << "Enter a number: ";
        cin >> n;

        // Use C-style I/O functions
        printf("You entered: %d\n\n", n);

        // Note: Be cautious when mixing C and C++ I/O when synchronization is disabled.
    }

    cout << "===== End of Examples =====" << endl;

    return 0;
}