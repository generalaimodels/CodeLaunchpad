/*******************************************************
 * Chapter 7: Input and Output (I/O) - Communicating with the User and Files ⌨️🖥️ 🗂️
 * -----------------------------------------------------
 * This code file explores the basics of input and output in C++,
 * starting from standard console I/O to file I/O.
 * Each example builds upon the previous one.
 *******************************************************/

#include <iostream>     // For std::cin and std::cout
#include <iomanip>      // For manipulators like std::setw, std::setprecision
#include <fstream>      // For file input/output streams
#include <sstream>      // For string streams
#include <string>       // For std::string

int main() {
    // Example 1: Basic Console Input and Output ⌨️🖥️
    std::cout << "Example 1: Basic Console Input and Output ⌨️🖥️" << std::endl;
    
    int age;
    std::cout << "Enter your age: ";  // Prompt the user for input ⌨️
    std::cin >> age;                  // Read input from the user
    
    if (std::cin.fail()) {
        std::cerr << "Invalid input. Please enter a numeric value." << std::endl; // Error message ⚠️
        std::cin.clear();                // Clear the error flag 🧹
        std::cin.ignore(1000, '\n');     // Discard invalid input
    } else {
        std::cout << "You are " << age << " years old." << std::endl; // Output the result 🖥️
    }
    
    // Example 2: Using std::getline for String Input 📝
    std::cout << "\nExample 2: Using std::getline for String Input 📝" << std::endl;

    std::cin.ignore();                  // Clear the newline character left in the buffer 🧹
    std::string name;
    std::cout << "Enter your full name: ";
    std::getline(std::cin, name);       // Read a line of text (including spaces)
    std::cout << "Hello, " << name << "!" << std::endl;

    // Possible Mistake: Not clearing input buffer before std::getline, resulting in empty input ❌

    // Example 3: Formatting Output with Manipulators 🎨
    std::cout << "\nExample 3: Formatting Output with Manipulators 🎨" << std::endl;

    double pi = 3.1415926535;
    std::cout << "Default precision: " << pi << std::endl;                            // Default precision
    std::cout << "Fixed precision (3): " << std::fixed << std::setprecision(3) << pi << std::endl; // Fixed to 3 decimal places
    std::cout << "Scientific notation: " << std::scientific << pi << std::endl;       // Scientific notation
    
    // Reset format flags
    std::cout.unsetf(std::ios::fixed | std::ios::scientific);
    std::cout.precision(6);  // Reset precision

    // Example 4: Aligning Text with std::setw 📐
    std::cout << "\nExample 4: Aligning Text with std::setw 📐" << std::endl;
    
    std::cout << std::setw(15) << "Item" << std::setw(10) << "Price" << std::endl;
    std::cout << std::setw(15) << "Apple" << std::setw(10) << "$1.99" << std::endl;
    std::cout << std::setw(15) << "Banana" << std::setw(10) << "$0.99" << std::endl;
    std::cout << std::setw(15) << "Cherry" << std::setw(10) << "$2.49" << std::endl;

    // Example 5: Reading and Writing to a File 🗂️
    std::cout << "\nExample 5: Reading and Writing to a File 🗂️" << std::endl;

    std::ofstream outFile("example.txt");  // Open a file for writing ✍️

    if (!outFile) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return 1;  // Exit with error code
    }

    outFile << "This is a line of text." << std::endl;
    outFile << "This is another line of text." << std::endl;

    outFile.close(); // Close the file when done 🗝️

    // Now read from the same file
    std::ifstream inFile("example.txt");   // Open a file for reading 📖

    if (!inFile) {
        std::cerr << "Error: Could not open file for reading." << std::endl;
        return 1;
    }

    std::string line;
    std::cout << "Contents of example.txt:" << std::endl;
    while (std::getline(inFile, line)) {
        std::cout << line << std::endl;    // Output each line 🖥️
    }

    inFile.close(); // Close the file

    // Possible Mistake: Forgetting to check if the file was opened successfully ❌

    // Example 6: Appending to a File 📎
    std::cout << "\nExample 6: Appending to a File 📎" << std::endl;

    std::ofstream appendFile("example.txt", std::ios::app); // Open in append mode

    if (!appendFile) {
        std::cerr << "Error: Could not open file for appending." << std::endl;
        return 1;
    }

    appendFile << "This line is appended." << std::endl;

    appendFile.close(); // Close the file

    // Read the file again to see the changes
    inFile.open("example.txt"); // Reopen the file for reading

    if (!inFile) {
        std::cerr << "Error: Could not open file for reading." << std::endl;
        return 1;
    }

    std::cout << "Updated contents of example.txt:" << std::endl;
    while (std::getline(inFile, line)) {
        std::cout << line << std::endl;
    }

    inFile.close();

    // Example 7: Error Handling in File Operations 🚫
    std::cout << "\nExample 7: Error Handling in File Operations 🚫" << std::endl;

    std::ifstream nonExistentFile("nonexistent.txt");

    if (!nonExistentFile) {
        std::cerr << "Error: File 'nonexistent.txt' does not exist." << std::endl;
    } else {
        // This block won't be executed
        nonExistentFile.close();
    }

    // Example 8: Using std::fstream for Both Input and Output 🔄
    std::cout << "\nExample 8: Using std::fstream for Both Input and Output 🔄" << std::endl;

    std::fstream ioFile("data.txt", std::ios::in | std::ios::out | std::ios::trunc);

    if (!ioFile) {
        std::cerr << "Error: Could not open 'data.txt' for I/O." << std::endl;
        return 1;
    }

    ioFile << "Line 1" << std::endl;
    ioFile << "Line 2" << std::endl;

    ioFile.seekg(0); // Move read position to the beginning of the file 🏁

    std::cout << "Contents of data.txt:" << std::endl;
    while (std::getline(ioFile, line)) {
        std::cout << line << std::endl;
    }

    ioFile.close();

    // Example 9: Reading and Writing Binary Files 💾
    std::cout << "\nExample 9: Reading and Writing Binary Files 💾" << std::endl;

    std::ofstream binaryOut("binary.dat", std::ios::binary);

    if (!binaryOut) {
        std::cerr << "Error: Could not open 'binary.dat' for writing in binary mode." << std::endl;
        return 1;
    }

    int numbers[] = {1, 2, 3, 4, 5};
    binaryOut.write(reinterpret_cast<char*>(numbers), sizeof(numbers));

    binaryOut.close();

    // Read the binary file
    std::ifstream binaryIn("binary.dat", std::ios::binary);

    if (!binaryIn) {
        std::cerr << "Error: Could not open 'binary.dat' for reading." << std::endl;
        return 1;
    }

    int readNumbers[5];
    binaryIn.read(reinterpret_cast<char*>(readNumbers), sizeof(readNumbers));

    binaryIn.close();

    std::cout << "Numbers read from 'binary.dat': ";
    for (int num : readNumbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Possible Mistake: Not using binary mode when required, leading to data corruption ❌

    // Example 10: String Streams for Parsing and Formatting 🧵
    std::cout << "\nExample 10: String Streams for Parsing and Formatting 🧵" << std::endl;

    std::stringstream ss;
    ss << "42 3.14 Hello";

    int intValue;
    double doubleValue;
    std::string strValue;

    ss >> intValue >> doubleValue >> strValue;

    std::cout << "Parsed values:" << std::endl;
    std::cout << "Integer: " << intValue << std::endl;
    std::cout << "Double: " << doubleValue << std::endl;
    std::cout << "String: " << strValue << std::endl;

    // Example 11: Redirecting Output to a File 📤
    std::cout << "\nExample 11: Redirecting Output to a File 📤" << std::endl;

    std::ofstream logFile("log.txt");

    if (!logFile) {
        std::cerr << "Error: Could not open 'log.txt' for logging." << std::endl;
    } else {
        std::streambuf* coutBuf = std::cout.rdbuf(); // Save original buffer
        std::cout.rdbuf(logFile.rdbuf()); // Redirect std::cout to logFile

        std::cout << "This message will be logged to 'log.txt' instead of console." << std::endl;

        std::cout.rdbuf(coutBuf); // Restore original buffer

        logFile.close();
    }

    std::cout << "Logging complete. Check 'log.txt' for the message." << std::endl;

    // Example 12: Using std::getline with Delimiters 🛑
    std::cout << "\nExample 12: Using std::getline with Delimiters 🛑" << std::endl;

    std::stringstream dataStream("apple,banana,cherry");
    std::string fruit;

    std::cout << "Fruits:" << std::endl;
    while (std::getline(dataStream, fruit, ',')) { // Use comma as delimiter
        std::cout << fruit << std::endl;
    }

    // Example 13: Reading a File Character by Character 🎹
    std::cout << "\nExample 13: Reading a File Character by Character 🎹" << std::endl;

    inFile.open("example.txt");

    if (!inFile) {
        std::cerr << "Error: Could not open 'example.txt' for character-wise reading." << std::endl;
        return 1;
    }

    char ch;
    std::cout << "Contents of 'example.txt' (character by character):" << std::endl;

    while (inFile.get(ch)) { // Read one character at a time
        std::cout << ch;
    }

    inFile.close();

    // Example 14: Flushing the Output Buffer 🔄
    std::cout << "\nExample 14: Flushing the Output Buffer 🔄" << std::endl;

    std::cout << "This message will be flushed immediately." << std::flush; // Flush the buffer

    // Example 15: Setting and Getting the Position in a File 🧭
    std::cout << "\nExample 15: Setting and Getting the Position in a File 🧭" << std::endl;

    std::ifstream positionFile("data.txt", std::ios::binary);

    if (!positionFile) {
        std::cerr << "Error: Could not open 'data.txt' to get positions." << std::endl;
        return 1;
    }

    positionFile.seekg(0, std::ios::end);  // Move to the end
    std::streampos fileSize = positionFile.tellg(); // Get position (file size)

    std::cout << "'data.txt' size: " << fileSize << " bytes." << std::endl;

    positionFile.close();

    return 0; // End of program
}