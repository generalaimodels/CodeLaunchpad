/*******************************************************
 * Chapter 6: Pointers - Unlocking Memory Addresses 📍🔑
 * -----------------------------------------------------
 * This code file explores the basics of pointers in C++,
 * starting from simple declarations to more advanced uses.
 * Each example builds upon the previous one.
 *******************************************************/

#include <iostream> // Include the input-output stream library

int main() {
    // Example 1: Pointer Declaration and Initialization 📍
    int variable = 42;         // An integer variable 🧮
    int* ptr = &variable;      // Pointer 'ptr' stores the address of 'variable' 📍

    // Output the value and address
    std::cout << "Variable value: " << variable << std::endl;             // Prints 42
    std::cout << "Variable address: " << &variable << std::endl;          // Prints address (e.g., 0x7ffee4b5d6ac)
    std::cout << "Pointer value (address): " << ptr << std::endl;         // Same as above
    std::cout << "Value pointed by pointer: " << *ptr << std::endl;       // Prints 42

    // Example 2: Modifying Value through Pointer 🔧
    *ptr = 100;   // Change the value at the memory address pointed by 'ptr' 🔄
    std::cout << "\nAfter modifying through pointer:" << std::endl;
    std::cout << "Variable value: " << variable << std::endl;             // Prints 100
    std::cout << "Value pointed by pointer: " << *ptr << std::endl;       // Prints 100

    // Example 3: Null Pointer 🚫
    int* nullPtr = nullptr;    // 'nullptr' represents a null pointer 🚫
    if (nullPtr == nullptr) {
        std::cout << "\nnullPtr is null (doesn't point to a valid memory location)." << std::endl;
    }

    // Example 4: Pointer Arithmetic ➕➖
    int arr[5] = {10, 20, 30, 40, 50};   // An array of integers 📚
    int* arrPtr = arr;                   // Pointer to the first element of the array 📍

    std::cout << "\nArray elements accessed via pointer arithmetic:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << *(arrPtr + i) << " ";   // Access array elements using pointer arithmetic ➡️
    }
    std::cout << std::endl;

    // Example 5: Common Mistake - Dereferencing a Null Pointer ❌
    // int value = *nullPtr;  // Uncommenting this line will cause a runtime error (segmentation fault) ⚠️

    // Tip: Always check if a pointer is not null before dereferencing it ✅
    if (nullPtr != nullptr) {
        int value = *nullPtr;
    } else {
        std::cout << "\nCannot dereference nullPtr as it is null." << std::endl;
    }

    // Example 6: Double Pointers (Pointer to a Pointer) 🪆
    int** doublePtr = &ptr;    // 'doublePtr' points to the address of 'ptr' 🪆
    std::cout << "\nValue of variable using double pointer: " << **doublePtr << std::endl; // Prints 100

    // Example 7: Constant Pointers and Pointers to Constants 🔒
    int constVar = 5;
    const int* ptrToConst = &constVar;   // Pointer to a constant integer 🔒
    int* const constPtr = &variable;     // Constant pointer to an integer 🔒

    // *ptrToConst = 10;    // Error: Cannot modify value through pointer to constant ❌
    ptrToConst = &variable;  // Allowed: Can change what it points to ✅

    *constPtr = 200;         // Allowed: Can modify value through constant pointer ✅
    // constPtr = &constVar; // Error: Cannot change the address stored in constant pointer ❌

    std::cout << "\nAfter modifying via constPtr:" << std::endl;
    std::cout << "Variable value: " << variable << std::endl;             // Prints 200

    // Example 8: Void Pointers (Generic Pointers) 🎩
    void* voidPtr = &variable;  // 'voidPtr' can point to any data type 🎩
    // std::cout << *voidPtr;   // Error: Cannot dereference void pointer directly ❌

    // Need to cast void pointer to appropriate type before dereferencing ✅
    std::cout << "\nValue using void pointer: " << *(static_cast<int*>(voidPtr)) << std::endl; // Prints 200

    // Example 9: Pointer Arrays vs. Array Pointers 📚🔀
    int* ptrArray[5];    // An array of integer pointers 📚
    int (*arrayPtr)[5];  // A pointer to an array of 5 integers 🔀

    // Assign addresses to ptrArray elements
    for (int i = 0; i < 5; ++i) {
        ptrArray[i] = &arr[i];   // Each pointer in the array points to an element in 'arr' 📍
    }

    // Assign the address of array 'arr' to 'arrayPtr'
    arrayPtr = &arr;

    std::cout << "\nAccessing elements using array of pointers:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << *ptrArray[i] << " ";    // Accessing values via pointers in array 📬
    }
    std::cout << std::endl;

    std::cout << "\nAccessing elements using pointer to array:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << (*arrayPtr)[i] << " ";  // Accessing elements via pointer to array 📦
    }
    std::cout << std::endl;

    // Example 10: Dynamic Memory Allocation with Pointers 💾
    int* dynPtr = new int(25);   // Dynamically allocate memory for an integer 💾
    std::cout << "\nDynamically allocated value: " << *dynPtr << std::endl; // Prints 25

    // Don't forget to free allocated memory to prevent memory leaks ❗
    delete dynPtr;   // Free the allocated memory 🗑️
    dynPtr = nullptr; // Good practice to set pointer to nullptr after deleting 🧹

    return 0; // End of program
}