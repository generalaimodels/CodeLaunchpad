#include <iostream>

// Chapter 6: Pointers - Unlocking Memory Addresses (Basics) 📍🔑

// Concept: What are Pointers? Memory Addresses 📍🔑
// Analogy: Imagine every house 🏠 on a street has an address 📍. A pointer is like knowing the address of a house instead of just the house itself. It "points" to a location in memory.
// Emoji: 🏠📍🔑 (House, Address, Key to access the house)
// Details:
// Memory addresses: Every location in computer memory has a unique address (like a number).
// Pointer variables: Variables that store memory addresses.
// Pointer declaration: Using * to indicate a pointer type (e.g., int* ptr; - pointer to an integer).
// Address-of operator &: Getting the memory address of a variable. ptr = &variable; (Get the address of 'variable' and store it in 'ptr').
// Dereference operator *: Accessing the value stored at the memory address pointed to by a pointer. value = *ptr; (Go to the address in 'ptr' and get the value there).

int main() {
    std::cout << "Chapter 6: Pointers - Unlocking Memory Addresses (Basics) 📍🔑\n";

    // ------------------------------------------------------------------------------------------------
    std::cout << "\nConcept: What are Pointers? Memory Addresses 📍🔑\n";

    // Example 1: Declaring a regular integer variable and a pointer to an integer
    int number = 10; // Declare an integer variable 'number' and initialize it with 10.
    int* ptr;       // Declare a pointer variable 'ptr' that can store the address of an integer.
                    // Note the '*' symbol, which signifies that 'ptr' is a pointer.

    std::cout << "\nExample 1: Declaration and Basic Variables\n";
    std::cout << "Integer variable 'number': " << number << "\n";
    std::cout << "Pointer variable 'ptr' (uninitialized): " << ptr << " (likely garbage value or 0)\n"; // Uninitialized pointers contain garbage values.

    // Common mistake: Using an uninitialized pointer is undefined behavior and can lead to crashes.
    //                 It's crucial to initialize pointers before using them.

    // Example 2: Using the address-of operator '&' to get the memory address
    ptr = &number; // Assign the memory address of 'number' to the pointer 'ptr'.
                   // The '&' operator gives the memory address of 'number'.

    std::cout << "\nExample 2: Address-of Operator '&'\n";
    std::cout << "Address of 'number' (&number): " << &number << "\n"; // Display the memory address of 'number'.
    std::cout << "Value of 'ptr' (address of 'number'): " << ptr << "\n";   // Display the value of 'ptr', which is now the address of 'number'.
    std::cout << "Is address of 'number' same as value of 'ptr'? : " << (&number == ptr ? "Yes" : "No") << "\n"; // Verify that they are the same.

    // Example 3: Using the dereference operator '*' to access the value at the address
    int value = *ptr; // Access the value stored at the memory address pointed to by 'ptr' and assign it to 'value'.
                     // The '*' operator, when used before a pointer, dereferences it. It goes to the memory location
                     // stored in 'ptr' and retrieves the value at that location.

    std::cout << "\nExample 3: Dereference Operator '*'\n";
    std::cout << "Value of 'number': " << number << "\n";
    std::cout << "Value of 'ptr' (address of 'number'): " << ptr << "\n";
    std::cout << "Value at the address pointed to by 'ptr' (*ptr): " << *ptr << "\n"; // This should be the same as 'number'.
    std::cout << "Value of 'value' (assigned from *ptr): " << value << "\n";
    std::cout << "Is value of 'number' same as value of '*ptr'? : " << (number == *ptr ? "Yes" : "No") << "\n"; // Verify they are the same.

    // Example 4: Modifying the value through the pointer
    *ptr = 25; // Modify the value at the memory address pointed to by 'ptr'.
              // Because 'ptr' points to the memory location of 'number', this will change the value of 'number'.

    std::cout << "\nExample 4: Modifying Value via Pointer\n";
    std::cout << "Original value of 'number' (before modification via pointer): 10\n"; // Remember original value
    std::cout << "Modified value of 'number' (after *ptr = 25): " << number << "\n"; // 'number' is now changed to 25.
    std::cout << "Value at the address pointed to by 'ptr' (*ptr): " << *ptr << "\n"; // '*ptr' also reflects the change.

    // Example 5: Pointers to different data types
    double pi = 3.14159;
    double* piPtr = &pi; // Pointer to a double.
    char initial = 'J';
    char* charPtr = &initial; // Pointer to a char.

    std::cout << "\nExample 5: Pointers to Different Types\n";
    std::cout << "Double variable 'pi': " << pi << "\n";
    std::cout << "Pointer to double 'piPtr': " << piPtr << "\n";
    std::cout << "Value at 'piPtr' (*piPtr): " << *piPtr << "\n";

    std::cout << "Char variable 'initial': " << initial << "\n";
    std::cout << "Pointer to char 'charPtr': " << charPtr << "\n"; // Note: char pointers behave differently when printed, often printing strings.
                                                                  // We'll cast it to void* to see the address directly.
    std::cout << "Pointer to char 'charPtr' (as address): " << static_cast<void*>(charPtr) << "\n"; // Correct way to see char pointer address.
    std::cout << "Value at 'charPtr' (*charPtr): " << *charPtr << "\n";


    // ------------------------------------------------------------------------------------------------
    std::cout << "\n\nConcept: Basic Pointer Operations 🔑\n";
    // Analogy: Using the address (pointer) to find and interact with the house (data in memory).
    // Emoji: 📍🔑➡️🏠 (Address and key to access the house/data)
    // Details:
    // Pointer initialization (pointing to a valid memory location).
    // Null pointers (pointers that don't point to anything valid - like an invalid address).
    // Basic pointer arithmetic (moving pointers in memory - be careful!). (Covered in more detail in Advanced level).
    // Pointers and arrays (array names are often treated as pointers to the first element).

    // Example 1: Pointer Initialization
    int num1 = 100;
    int* ptr1 = &num1; // Initialization at the time of declaration, pointing to 'num1'.
    int* ptr2;         // Declaration without initialization.
    ptr2 = &num1;     // Initialization after declaration, pointing to 'num1'.

    std::cout << "\nExample 1: Pointer Initialization\n";
    std::cout << "Value of 'num1': " << num1 << "\n";
    std::cout << "Pointer 'ptr1' (initialized at declaration): " << ptr1 << ", points to value: " << *ptr1 << "\n";
    std::cout << "Pointer 'ptr2' (initialized after declaration): " << ptr2 << ", points to value: " << *ptr2 << "\n";

    // Example 2: Null Pointers
    int* nullPtr = nullptr; // Modern C++ way to represent a null pointer. (Alternatively, you can use '0' or 'NULL' from older C++ but 'nullptr' is preferred).

    std::cout << "\nExample 2: Null Pointers\n";
    std::cout << "Null pointer 'nullPtr': " << nullPtr << "\n";

    // Common mistake: Dereferencing a null pointer leads to a crash (segmentation fault).
    //                 Always check if a pointer is null before dereferencing it, especially if it could potentially be uninitialized
    //                 or if its previous pointed-to memory has been deallocated.

    // Example of checking for null pointer (Defensive programming practice)
    if (nullPtr != nullptr) {
        // std::cout << "*nullPtr: " << *nullPtr << "\n"; // This would crash if uncommented!
        std::cout << "nullPtr is NOT null, but we are not dereferencing it here.\n"; // Safe path.
    } else {
        std::cout << "nullPtr is indeed a null pointer.\n";
    }


    // Example 3: Pointers and Arrays
    int arr[] = {1, 2, 3, 4, 5};
    int* arrPtr = arr; // Array name 'arr' decays to a pointer to its first element.
                       // 'arrPtr' now points to the memory location of arr[0].

    std::cout << "\nExample 3: Pointers and Arrays\n";
    std::cout << "Array 'arr': {";
    for (int i = 0; i < 5; ++i) {
        std::cout << arr[i] << (i < 4 ? ", " : "");
    }
    std::cout << "}\n";
    std::cout << "Pointer 'arrPtr' (points to arr[0]): " << arrPtr << ", value at arrPtr: " << *arrPtr << "\n"; // Should print the address and value of arr[0].
    std::cout << "Address of arr[0]: " << &arr[0] << "\n"; // Verify addresses are the same.
    std::cout << "Is 'arrPtr' same as address of arr[0]? : " << (arrPtr == &arr[0] ? "Yes" : "No") << "\n";

    // Example 4: Accessing array elements using pointer arithmetic (Basic - more in advanced section)
    std::cout << "\nExample 4: Accessing Array Elements via Pointer (Basic)\n";
    std::cout << "arrPtr points to arr[0] with value: " << *arrPtr << "\n";
    std::cout << "*(arrPtr + 1) points to arr[1] with value: " << *(arrPtr + 1) << "\n"; // Moving the pointer by 1 integer size to the next element.
    std::cout << "*(arrPtr + 2) points to arr[2] with value: " << *(arrPtr + 2) << "\n"; // And so on...

    // Important Note: Pointer arithmetic depends on the data type of the pointer.
    //                 For an 'int*' pointer, incrementing it by 1 moves it forward by 'sizeof(int)' bytes in memory,
    //                 which is typically enough to reach the next integer in an array.
    //                 Be very cautious with pointer arithmetic, going beyond array bounds can lead to serious issues.

    // Example 5: Iterating through an array using a pointer
    std::cout << "\nExample 5: Iterating Array with Pointer\n";
    std::cout << "Array 'arr' elements using pointer iteration: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << *(arrPtr + i) << (i < 4 ? ", " : ""); // Accessing elements using pointer arithmetic.
    }
    std::cout << "\n";


    std::cout << "\n--- End of Chapter 6 (Basics) ---\n";
    return 0;
}
/*

Developer Signature: ✍️ Example Code Snippets and Explanations by an Advanced C++ Coder 🚀

*/