/******************************************************************************************
 * Filename: io_examples.cpp
 * Description: Comprehensive examples covering C++ Input and Output (I/O) concepts.
 *              This file includes 10 examples for each case, from basic to advanced,
 *              demonstrating standard input/output streams, formatting, and file I/O,
 *              including handling of exceptional cases.
 * Author: Kandimalla Hemanth
 * Date: 2-5-2025
 ***************************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <limits>
#include <exception>

int main() {
    // SECTION 1: Standard Input and Output Streams

    // Example 1.1: Basic std::cout usage
    std::cout << "Example 1.1: Hello, World!" << std::endl;

    // Example 1.2: Basic std::cin usage with error handling
    int number;
    std::cout << "Example 1.2: Please enter an integer: ";
    while (!(std::cin >> number)) {
        std::cout << "Invalid input. Please enter a valid integer: ";
        std::cin.clear(); // Clear error flags
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
    }
    std::cout << "You entered: " << number << std::endl;

    // Example 1.3: Reading a string with whitespace using std::getline()
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Clear input buffer
    std::string fullName;
    std::cout << "Example 1.3: Please enter your full name: ";
    std::getline(std::cin, fullName);
    std::cout << "Hello, " << fullName << "!" << std::endl;

    // Example 1.4: Formatted output using manipulators
    double pi = 3.14159265;
    std::cout << "Example 1.4: Formatted Pi value:" << std::endl;
    std::cout << std::fixed << std::setprecision(4) << pi << std::endl;

    // Example 1.5: Using std::setw and std::left/right for alignment
    std::cout << "Example 1.5: Table with alignment:" << std::endl;
    std::cout << std::left << std::setw(10) << "Name" << std::right << std::setw(5) << "Age" << std::endl;
    std::cout << std::left << std::setw(10) << "Alice" << std::right << std::setw(5) << 30 << std::endl;
    std::cout << std::left << std::setw(10) << "Bob" << std::right << std::setw(5) << 25 << std::endl;

    // Example 1.6: Handling input failure and exceptions
    try {
        std::cout << "Example 1.6: Enter a double value: ";
        double value;
        if (!(std::cin >> value)) {
            throw std::runtime_error("Input is not a valid double.");
        }
        std::cout << "You entered: " << value << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // Example 1.7: Reading multiple values in a loop
    std::cout << "Example 1.7: Enter integers (-1 to stop): ";
    int sum = 0, val = 0;
    while (std::cin >> val && val != -1) {
        sum += val;
    }
    std::cout << "Sum: " << sum << std::endl;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Example 1.8: Using stringstream for input parsing
    std::cout << "Example 1.8: Parsing comma-separated integers." << std::endl;
    std::string input;
    std::cout << "Enter numbers separated by commas: ";
    std::getline(std::cin, input);
    std::stringstream ss(input);
    int num;
    char comma;
    sum = 0;
    while (ss >> num) {
        sum += num;
        ss >> comma;
    }
    std::cout << "Total sum: " << sum << std::endl;

    // Example 1.9: Reading and validating multiple data types
    std::cout << "Example 1.9: Enter a string and an integer: ";
    std::string text;
    int integer;
    if ((std::cin >> text) && (std::cin >> integer)) {
        std::cout << "You entered: Text = " << text << ", Integer = " << integer << std::endl;
    } else {
        std::cerr << "Invalid input for text or integer." << std::endl;
    }
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // Example 1.10: Advanced formatting with manipulators
    std::cout << "Example 1.10: Scientific notation and hex output:" << std::endl;
    double largeNumber = 1234567.89;
    int hexNumber = 255;
    std::cout << "Scientific: " << std::scientific << largeNumber << std::endl;
    std::cout << "Hexadecimal: " << std::hex << std::showbase << hexNumber << std::endl;
    std::cout << std::dec; // Reset to decimal

    // SECTION 2: File Input and Output

    // Example 2.1: Writing to a file using std::ofstream
    {
        std::ofstream outFile("output1.txt");
        if (outFile.is_open()) {
            outFile << "Example 2.1: Writing first line to file." << std::endl;
            outFile << "Second line in file." << std::endl;
            outFile.close();
            std::cout << "Data successfully written to 'output1.txt'." << std::endl;
        } else {
            std::cerr << "Unable to open file 'output1.txt' for writing." << std::endl;
        }
    }

    // Example 2.2: Reading from a file using std::ifstream
    {
        std::ifstream inFile("output1.txt");
        if (inFile.is_open()) {
            std::string line;
            std::cout << "Example 2.2: Reading from 'output1.txt':" << std::endl;
            while (std::getline(inFile, line)) {
                std::cout << line << std::endl;
            }
            inFile.close();
        } else {
            std::cerr << "Unable to open file 'output1.txt' for reading." << std::endl;
        }
    }

    // Example 2.3: Appending to a file
    {
        std::ofstream outFile("output1.txt", std::ios::app);
        if (outFile.is_open()) {
            outFile << "Example 2.3: Appending a new line." << std::endl;
            outFile.close();
            std::cout << "Data appended to 'output1.txt'." << std::endl;
        } else {
            std::cerr << "Unable to open file 'output1.txt' for appending." << std::endl;
        }
    }

    // Example 2.4: Binary file writing and reading
    {
        std::ofstream binOut("data.bin", std::ios::binary);
        if (binOut.is_open()) {
            int data[5] = {10, 20, 30, 40, 50};
            binOut.write(reinterpret_cast<char*>(data), sizeof(data));
            binOut.close();
            std::cout << "Example 2.4: Binary data written to 'data.bin'." << std::endl;
        } else {
            std::cerr << "Unable to open 'data.bin' for writing." << std::endl;
        }

        std::ifstream binIn("data.bin", std::ios::binary);
        if (binIn.is_open()) {
            int readData[5] = {0};
            binIn.read(reinterpret_cast<char*>(readData), sizeof(readData));
            binIn.close();
            std::cout << "Binary data read from 'data.bin': ";
            for (int i = 0; i < 5; ++i) {
                std::cout << readData[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Unable to open 'data.bin' for reading." << std::endl;
        }
    }

    // Example 2.5: Using std::fstream for both reading and writing
    {
        std::fstream ioFile("iofile.txt", std::ios::in | std::ios::out | std::ios::trunc);
        if (ioFile.is_open()) {
            ioFile << "Example 2.5: Writing and reading in the same file." << std::endl;
            ioFile.seekg(0); // Move file pointer to the beginning for reading
            std::string content;
            std::getline(ioFile, content);
            std::cout << "Read from 'iofile.txt': " << content << std::endl;
            ioFile.close();
        } else {
            std::cerr << "Unable to open 'iofile.txt'." << std::endl;
        }
    }

    // Example 2.6: Exception handling with file operations
    {
        std::ifstream nonExistentFile("nonexistent.txt");
        if (!nonExistentFile.is_open()) {
            std::cerr << "Example 2.6: File 'nonexistent.txt' does not exist." << std::endl;
        } else {
            nonExistentFile.close();
        }
    }

    // Example 2.7: Reading structured data from a file
    {
        std::ofstream outFile("data.csv");
        if (outFile.is_open()) {
            outFile << "Name,Age,Salary" << std::endl;
            outFile << "Alice,30,50000" << std::endl;
            outFile << "Bob,28,45000" << std::endl;
            outFile.close();
        }

        std::ifstream inFile("data.csv");
        if (inFile.is_open()) {
            std::string line;
            std::getline(inFile, line); // Skip header

            std::cout << "Example 2.7: Reading structured data from 'data.csv':" << std::endl;
            while (std::getline(inFile, line)) {
                std::stringstream ss(line);
                std::string name, age, salary;
                std::getline(ss, name, ',');
                std::getline(ss, age, ',');
                std::getline(ss, salary, ',');
                std::cout << "Name: " << name << ", Age: " << age << ", Salary: " << salary << std::endl;
            }
            inFile.close();
        } else {
            std::cerr << "Unable to open 'data.csv'." << std::endl;
        }
    }

    // Example 2.8: Handling file read/write errors
    {
        std::ifstream inFile("readonly.txt");
        if (!inFile.is_open()) {
            std::cerr << "Example 2.8: Error opening 'readonly.txt'." << std::endl;
        } else {
            // Attempt to write to a read-only file
            std::ofstream outFile("readonly.txt", std::ios::app);
            if (!outFile.is_open()) {
                std::cerr << "Cannot write to 'readonly.txt'." << std::endl;
            }
            inFile.close();
        }
    }

    // Example 2.9: File positioning and seeking
    {
        std::fstream file("position.txt", std::ios::out | std::ios::in | std::ios::trunc);
        if (file.is_open()) {
            file << "Example 2.9: File seeking operations." << std::endl;
            file.seekg(0);
            std::string content;
            std::getline(file, content);
            std::cout << "Read content: " << content << std::endl;

            // Move write pointer to beginning and overwrite
            file.seekp(0);
            file << "Overwritten line." << std::endl;

            // Read again
            file.seekg(0);
            std::getline(file, content);
            std::cout << "After overwrite: " << content << std::endl;

            file.close();
        } else {
            std::cerr << "Unable to open 'position.txt'." << std::endl;
        }
    }

    // Example 2.10: Advanced file operations with error handling
    {
        std::ifstream inFile("largefile.txt");
        if (inFile.is_open()) {
            inFile.seekg(0, std::ios::end);
            std::streampos fileSize = inFile.tellg();
            if (fileSize > 1024 * 1024) { // Arbitrary large file size check (e.g., >1MB)
                std::cerr << "File is too large to process." << std::endl;
            } else {
                inFile.seekg(0);
                std::string content((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
                std::cout << "File content: " << content << std::endl;
            }
            inFile.close();
        } else {
            std::cerr << "Unable to open 'largefile.txt'." << std::endl;
        }
    }

    return 0;
}