// pointers_basics.cpp
// Author: Hemanth Kandimalla 🧠💻
// Date: 2025-02-05
// Description: Demonstrating Chapter 6: Pointers - Unlocking Memory Addresses (Basics)
// Each concept is illustrated with at least 10 examples using diagrams, analogies, and detailed comments.

#include <iostream>
using namespace std;

// ========================================================
// Section 1: Pointer Declaration Examples 📍🔑
// ========================================================
// Analogy: Think of each pointer as a label that holds the address (GPS coordinate) of a "house" (data).
void pointerDeclarationExamples() {
    cout << "\n=== Pointer Declaration Examples 📍🔑 ===\n";
    
    // Diagram:
    //    [Data Center]
    //         |
    //    [House: int a = 10]
    //         |
    //    [Label: int* p1 = &a]
    
    // Example 1: int pointer
    int a = 10;
    int* p1 = &a;
    cout << "Example 1: int a = 10; int* p1 = &a; -> *p1 = " << *p1 << "  Address of a by p1:"<< p1 << "\n";
    
    // Example 2: double pointer
    double d = 3.1415;
    double* p2 = &d;
    cout << "Example 2: double d = 3.1415; double* p2 = &d; -> *p2 = " << *p2 << "\n";
    
    // Example 3: char pointer
    char ch = 'Z';
    char* p3 = &ch;
    cout << "Example 3: char ch = 'Z'; char* p3 = &ch; -> *p3 = " << *p3 << "\n";
    
    // Example 4: float pointer
    float f = 2.718f;
    float* p4 = &f;
    cout << "Example 4: float f = 2.718f; float* p4 = &f; -> *p4 = " << *p4 << "\n";
    
    // Example 5: long pointer
    long l = 123456789L;
    long* p5 = &l;
    cout << "Example 5: long l = 123456789L; long* p5 = &l; -> *p5 = " << *p5 << "\n";
    
    // Example 6: short pointer
    short s = 32000;
    short* p6 = &s;
    cout << "Example 6: short s = 32000; short* p6 = &s; -> *p6 = " << *p6 << "\n";
    
    // Example 7: bool pointer
    bool flag = true;
    bool* p7 = &flag;
    cout << "Example 7: bool flag = true; bool* p7 = &flag; -> *p7 = " << *p7 << "\n";
    
    // Example 8: unsigned int pointer
    unsigned int ui = 4000000000u;
    unsigned int* p8 = &ui;
    cout << "Example 8: unsigned int ui = 4000000000u; unsigned int* p8 = &ui; -> *p8 = " << *p8 << "\n";
    
    // Example 9: pointer to pointer (int**)
    int** pp = &p1;
    cout << "Example 9: int** pp = &p1; -> **pp = " << **pp << "\n";
    
    // Example 10: const int pointer (pointer to constant data)
    const int ci = 99;
    const int* p10 = &ci;
    cout << "Example 10: const int ci = 99; const int* p10 = &ci; -> *p10 = " << *p10 << "\n";
    
    // Diagram Summary:
    //   [House = Data] <-- [Label = Pointer] 
    //   Each pointer "holds" an address pointing to its corresponding data.
}

// ========================================================
// Section 2: Address-Of Operator Examples (&) 📍
// ========================================================
// Analogy: The address-of operator '&' is like asking "Where is that house located?" 
// It returns the GPS coordinate (memory address) of the variable.
void addressOfOperatorExamples() {
    cout << "\n=== Address-Of Operator Examples (&) 📍 ===\n";
    
    // Example 1: Get address of an int variable
    int num1 = 42;
    int* addr1 = &num1;
    cout << "Example 1: int num1 = 42; int* addr1 = &num1; -> addr1 = " << addr1 << "\n";
    
    // Example 2: Get address of a double variable
    double num2 = 6.28;
    double* addr2 = &num2;
    cout << "Example 2: double num2 = 6.28; double* addr2 = &num2; -> addr2 = " << addr2 << "\n";
    
    // Example 3: Get address of a char variable
    char letter = 'A';
    char* addr3 = &letter;
    cout << "Example 3: char letter = 'A'; char* addr3 = &letter; -> addr3 = " << static_cast<void*>(addr3) << "\n";
    
    // Example 4: Get address of a float variable
    float num4 = 9.81f;
    float* addr4 = &num4;
    cout << "Example 4: float num4 = 9.81f; float* addr4 = &num4; -> addr4 = " << addr4 << "\n";
    
    // Example 5: Get address of a long variable
    long num5 = 100000L;
    long* addr5 = &num5;
    cout << "Example 5: long num5 = 100000L; long* addr5 = &num5; -> addr5 = " << addr5 << "\n";
    
    // Example 6: Get address of a short variable
    short num6 = 32000; // Note: short num6 = 32000; is also valid, but it's less explicit.
    short* addr6 = &num6; // Note: short* addr6 = &num6; is also valid, but it's less explicit.
    cout << "Example 6: short num6 = 32000; short* addr6 = &num6; -> addr6 = " << addr6 << "\n";
    
    // Example 7: Get address of a bool variable
    bool isTrue = false; // Note: bool isTrue = false; is also valid, but it's less explicit.
    bool* addr7 = &isTrue; // Note: bool* addr7 = &isTrue; is also valid, but it's less explicit.
    cout << "Example 7: bool isTrue = false; bool* addr7 = &isTrue; -> addr7 = " << addr7 << "\n";
    
    // Example 8: Get address of an unsigned int variable
    unsigned int num8 = 123456u; // Note: unsigned int num8 = 123456u; is also valid, but it's less explicit.
    unsigned int* addr8 = &num8; // Note: unsigned int* addr8 = &num8; is also valid, but it's less explicit.
    cout << "Example 8: unsigned int num8 = 123456u; unsigned int* addr8 = &num8; -> addr8 = " << addr8 << "\n";
    
    // Example 9: Get address of an array (array name decays to pointer)
    int arr[3] = {1, 2, 3};
    int* addr9 = arr;  // same as &arr[0]
    cout << "Example 9: int arr[3] = {1,2,3}; int* addr9 = arr; -> addr9 = " << addr9 << "\n";
    
    // Example 10: Get address of a pointer variable itself
    int* addr10 = &num1; // a pointer to num1
    int** pAddr = &addr10; // now get the address of the pointer variable addr10
    cout << "Example 10: int* addr10 = &num1; int** pAddr = &addr10; -> pAddr = " << pAddr << "\n";
    
    // Diagram:
    //  [Variable (House)] --&--> [Memory Address (GPS coordinate)]
}

// ========================================================
// Section 3: Dereference Operator Examples (*) 🔑
// ========================================================
// Analogy: The dereference operator '*' is like using your key 🔑 to open a house 🏠 at a given address (GPS coordinate)
// and see what’s inside (the data).
void dereferenceOperatorExamples() {
    cout << "\n=== Dereference Operator Examples (*) 🔑 ===\n";
    
    // Example 1: Dereference int pointer
    int x = 100;
    int* pX = &x;
    cout << "Example 1: int x = 100; int* pX = &x; -> *pX = " << *pX << "\n";
    
    // Example 2: Dereference double pointer
    double y = 2.71828;
    double* pY = &y;
    cout << "Example 2: double y = 2.71828; double* pY = &y; -> *pY = " << *pY << "\n";
    
    // Example 3: Dereference char pointer
    char c = 'K';
    char* pC = &c;
    cout << "Example 3: char c = 'K'; char* pC = &c; -> *pC = " << *pC << "\n";
    
    // Example 4: Dereference float pointer
    float f = 1.414f;
    float* pF = &f;
    cout << "Example 4: float f = 1.414f; float* pF = &f; -> *pF = " << *pF << "\n";
    
    // Example 5: Dereference long pointer
    long l = 987654321L;
    long* pL = &l;
    cout << "Example 5: long l = 987654321L; long* pL = &l; -> *pL = " << *pL << "\n";
    
    // Example 6: Dereference short pointer
    short s = 12345;
    short* pS = &s;
    cout << "Example 6: short s = 12345; short* pS = &s; -> *pS = " << *pS << "\n";
    
    // Example 7: Dereference bool pointer
    bool flag = true;
    bool* pFlag = &flag;
    cout << "Example 7: bool flag = true; bool* pFlag = &flag; -> *pFlag = " << *pFlag << "\n";
    
    // Example 8: Dereference unsigned int pointer
    unsigned int u = 55555u;
    unsigned int* pU = &u;
    cout << "Example 8: unsigned int u = 55555u; unsigned int* pU = &u; -> *pU = " << *pU << "\n";
    
    // Example 9: Dereference pointer to pointer
    int** pp = &pX;
    cout << "Example 9: int** pp = &pX; -> **pp = " << **pp << "\n";
    
    // Example 10: Dereference const int pointer
    const int ci = 77;
    const int* pCI = &ci;
    cout << "Example 10: const int ci = 77; const int* pCI = &ci; -> *pCI = " << *pCI << "\n";
    
    // Diagram:
    //    pX ---> [Memory Address] ---contains---> [x = 100]
    //    *pX "opens" the door to reveal the value 100.
}

// ========================================================
// Section 4: Pointer Initialization Examples 🔧
// ========================================================
// Analogy: Initializing a pointer is like programming your GPS with a valid destination before you start driving.
// You must ensure that your pointer “knows” where to go.
void pointerInitializationExamples() {
    cout << "\n=== Pointer Initialization Examples 🔧 ===\n";
    
    // Example 1: Initialize pointer using address-of operator
    int a = 10; // a is an int variable
    int* p1 = &a; // p1 points to a
    cout << "Example 1: int a = 10; int* p1 = &a; -> *p1 = " << *p1 << "\n";
    
    // Example 2: Initialize pointer with a literal array (decay to pointer)
    int arr1[3] = {1, 2, 3}; // arr1 is an array of 3 ints
    int* pArr1 = arr1;  // pArr1 points to arr1[0]
    cout << "Example 2: int arr1[3] = {1,2,3}; int* pArr1 = arr1; -> *pArr1 = " << *pArr1 << "\n";
    
    // Example 3: Initialize pointer using new (dynamic memory allocation)
    int* pDynamic = new int(500); // dynamically allocate memory for an int and initialize it to 500
    cout << "Example 3: int* pDynamic = new int(500); -> *pDynamic = " << *pDynamic << "\n";
    delete pDynamic;  // free memory
    
    // Example 4: Initialize pointer to a string literal (char pointer)
    const char* str = "Hello, Pointer!"; // str points to a string literal
    cout << "Example 4: const char* str = \"Hello, Pointer!\"; -> str = " << str << "\n";
    
    // Example 5: Initialize pointer to an element of an array explicitly
    int arr2[4] = {10, 20, 30, 40}; // arr2 is an array of 4 ints
    int* pArr2 = &arr2[2];  // points to the third element (30)
    cout << "Example 5: int arr2[4] = {10,20,30,40}; int* pArr2 = &arr2[2]; -> *pArr2 = " << *pArr2 << "\n";
    
    // Example 6: Initialize pointer using nullptr (explicitly not pointing anywhere yet)
    int* pNull = nullptr;
    cout << "Example 6: int* pNull = nullptr; -> pNull = " << pNull << "\n";
    
    // Example 7: Initialize pointer with an already declared variable's address
    double dVal = 3.33;
    double* pDouble = &dVal;
    cout << "Example 7: double dVal = 3.33; double* pDouble = &dVal; -> *pDouble = " << *pDouble << "\n";
    
    // Example 8: Initialize a pointer to point to a constant variable
    const int constVal = 88;
    const int* pConst = &constVal;
    cout << "Example 8: const int constVal = 88; const int* pConst = &constVal; -> *pConst = " << *pConst << "\n";
    
    // Example 9: Initialize pointer using address-of operator on a function parameter
    auto initFunction = [](int param) {
        int* ptr = &param;
        cout << "Example 9 (Inside lambda): int param = " << param << "; int* ptr = &param; -> *ptr = " << *ptr << "\n";
    };
    initFunction(77);
    
    // Example 10: Initialize pointer using dynamic array allocation
    int* pDynArray = new int[3]{7, 8, 9};
    cout << "Example 10: int* pDynArray = new int[3]{7,8,9}; -> *pDynArray = " << *pDynArray << "\n";
    delete[] pDynArray;  // free memory
    
    // Diagram:
    //   [Initialization] ---> [Pointer holds a valid memory address] ---> [Dereference reveals data]
}

// ========================================================
// Section 5: Null Pointer Examples 🚫📍
// ========================================================
// Analogy: A null pointer is like setting your GPS to "no destination" or an invalid address.
// It indicates that the pointer currently points to nothing valid.
void nullPointerExamples() {
    cout << "\n=== Null Pointer Examples 🚫📍 ===\n";
    
    // Example 1: Initialize int pointer to nullptr
    int* p1 = nullptr;
    cout << "Example 1: int* p1 = nullptr; -> p1 = " << p1 << "\n";
    
    // Example 2: Check for null pointer before dereferencing
    int* p2 = nullptr;
    if (p2 == nullptr)
        cout << "Example 2: p2 is null; cannot dereference safely.\n";
    
    // Example 3: Using nullptr with a double pointer
    double* pDouble = nullptr;
    cout << "Example 3: double* pDouble = nullptr; -> pDouble = " << pDouble << "\n";
    
    // Example 4: Assign pointer to nullptr after deletion (safety measure)
    int* pTemp = new int(50);
    delete pTemp;
    pTemp = nullptr;
    cout << "Example 4: pTemp deleted and set to nullptr; -> pTemp = " << pTemp << "\n";
    
    // Example 5: Null pointer in conditional expression
    char* pChar = nullptr;
    cout << "Example 5: char* pChar = nullptr; -> (pChar == nullptr) = " << (pChar == nullptr ? "true" : "false") << "\n";
    
    // Example 6: Using nullptr in function parameters (simulate no valid pointer passed)
    auto checkPointer = [](int* ptr) {
        if (ptr == nullptr)
            cout << "Example 6: Received a nullptr.\n";
        else
            cout << "Example 6: Received valid pointer with value " << *ptr << "\n";
    };
    checkPointer(nullptr);
    
    // Example 7: Null pointer with pointer arithmetic (should not be done, but showing initialization)
    int* pArr = nullptr;
    cout << "Example 7: int* pArr = nullptr; -> pArr = " << pArr << "\n";
    
    // Example 8: Demonstrate pointer re-assignment from nullptr to valid address
    int var = 2025;
    int* pVar = nullptr;
    pVar = &var;
    cout << "Example 8: int var = 2025; int* pVar = nullptr; pVar = &var; -> *pVar = " << *pVar << "\n";
    
    // Example 9: Null pointer constant assignment
    int* p9 = 0; // older C++ style (0 is interpreted as NULL)
    cout << "Example 9: int* p9 = 0; -> p9 = " << p9 << "\n";
    
    // Example 10: Using nullptr with pointer to pointer
    int* pA = nullptr;
    int** ppA = &pA;
    cout << "Example 10: int* pA = nullptr; int** ppA = &pA; -> *ppA = " << *ppA << " (should be nullptr)\n";
    
    // Diagram:
    //   [GPS with no destination] = nullptr ---> [Pointer safely indicates nothing is assigned]
}

// ========================================================
// Section 6: Pointer Arithmetic Examples ➕➖
// ========================================================
// Analogy: Pointer arithmetic is like moving your GPS coordinates along a street.
// Adding 1 moves you to the next "house" (memory location for the data type).
void pointerArithmeticExamples() {
    cout << "\n=== Pointer Arithmetic Examples ➕➖ ===\n";
    
    int arr[5] = {10, 20, 30, 40, 50};
    int* p = arr; // p initially points to arr[0]
    
    // Example 1: Dereference initial pointer (arr[0])
    cout << "Example 1: int arr[5] = {10,20,30,40,50}; int* p = arr; -> *p = " << *p << "\n";
    
    // Example 2: Increment pointer to point to arr[1]
    p++; // now p points to arr[1]
    cout << "Example 2: p++ -> *p = " << *p << "\n";
    
    // Example 3: Add 2 to pointer (move to arr[3])
    p += 2; // now p points to arr[3]
    cout << "Example 3: p += 2 -> *p = " << *p << "\n";
    
    // Example 4: Subtract 1 from pointer (move back to arr[2])
    p--; // now p points to arr[2]
    cout << "Example 4: p-- -> *p = " << *p << "\n";
    
    // Example 5: Calculate difference between pointers (number of elements apart)
    int* start = arr;
    int* current = p; // currently pointing to arr[2]
    cout << "Example 5: Difference (current - start) = " << (current - start) << "\n";
    
    // Example 6: Reset pointer to start and access using array indexing via pointer arithmetic
    p = arr;
    cout << "Example 6: p = arr; *(p + 3) = " << *(p + 3) << " (access arr[3])\n";
    
    // Example 7: Use pointer arithmetic in a loop to traverse an array
    cout << "Example 7: Traversing array using pointer arithmetic: ";
    for (int i = 0; i < 5; i++) {
        cout << *(p + i) << " ";
    }
    cout << "\n";
    
    // Example 8: Pointer arithmetic with a different data type (double array)
    double darr[4] = {1.1, 2.2, 3.3, 4.4};
    double* pd = darr;
    pd++; // now points to darr[1]
    cout << "Example 8: double darr[4] = {1.1,2.2,3.3,4.4}; double* pd = darr; pd++ -> *pd = " << *pd << "\n";
    
    // Example 9: Arithmetic on pointer in a function call (simulate navigation)
    auto addOffset = [](int* ptr, int offset) {
        return *(ptr + offset);
    };
    cout << "Example 9: addOffset(arr, 4) -> " << addOffset(arr, 4) << "\n";
    
    // Example 10: Demonstrate pointer arithmetic with pointer to struct (for extra insight)
    struct Point { int x, y; };
    Point points[2] = { {1,2}, {3,4} };
    Point* pPoint = points;
    cout << "Example 10: Point* pPoint = points; -> pPoint->x = " << pPoint->x << ", pPoint->y = " << pPoint->y << "\n";
    pPoint++; // move to next Point in array
    cout << "           After pPoint++ -> pPoint->x = " << pPoint->x << ", pPoint->y = " << pPoint->y << "\n";
    
    // Diagram:
    //   Array Memory Layout:
    //     +-----+-----+-----+-----+-----+
    //     | 10  | 20  | 30  | 40  | 50  |
    //     +-----+-----+-----+-----+-----+
    //        ^      ^      ^      ^      ^
    //       p0     p0+1   p0+2   p0+3   p0+4
}

// ========================================================
// Section 7: Pointers and Arrays Examples 🛣️📍
// ========================================================
// Analogy: An array is like a route with multiple houses. The array name is the starting point,
// and pointer arithmetic helps you visit each house along the route.
void pointersAndArraysExamples() {
    cout << "\n=== Pointers and Arrays Examples 🛣️📍 ===\n";
    
    // Example 1: Array name decays to pointer to first element
    int numbers[5] = {5, 10, 15, 20, 25};
    int* pNumbers = numbers;
    cout << "Example 1: int numbers[5] = {5,10,15,20,25}; int* pNumbers = numbers; -> *pNumbers = " << *pNumbers << "\n";
    
    // Example 2: Access array elements using pointer arithmetic
    cout << "Example 2: *(numbers + 2) = " << *(numbers + 2) << " (should be 15)\n";
    
    // Example 3: Traverse array using pointer increment
    cout << "Example 3: Traversing array using pointer increment: ";
    int* ptr = numbers;
    for (int i = 0; i < 5; i++) {
        cout << *(ptr++) << " ";
    }
    cout << "\n";
    
    // Example 4: Modify array elements using pointers
    int arrMod[3] = {1, 2, 3};
    int* pMod = arrMod;
    *(pMod + 1) = 99;
    cout << "Example 4: After modifying, arrMod[1] = " << arrMod[1] << "\n";
    
    // Example 5: Pointer arithmetic on multi-dimensional array (decay to pointer to array)
    int matrix[2][3] = { {1,2,3}, {4,5,6} };
    int (*pMatrix)[3] = matrix;
    cout << "Example 5: 2D array first row: " << pMatrix[0][0] << ", " << pMatrix[0][1] << ", " << pMatrix[0][2] << "\n";
    
    // Example 6: Using pointer indexing with array notation
    cout << "Example 6: numbers[3] = " << pNumbers[3] << "\n";
    
    // Example 7: Passing array pointer to a function
    auto printArray = [](int* arr, int size) {
        cout << "Array elements: ";
        for (int i = 0; i < size; i++)
            cout << arr[i] << " ";
        cout << "\n";
    };
    printArray(numbers, 5);
    
    // Example 8: Copying array elements using pointers
    int source[3] = {7, 8, 9};
    int destination[3] = {0};
    int* pSrc = source;
    int* pDest = destination;
    for (int i = 0; i < 3; i++) {
        *(pDest + i) = *(pSrc + i);
    }
    cout << "Example 8: Copied array: " << destination[0] << ", " << destination[1] << ", " << destination[2] << "\n";
    
    // Example 9: Pointer comparison with array bounds
    int* pStart = numbers;
    int* pEnd = numbers + 5;
    cout << "Example 9: pStart = " << pStart << ", pEnd = " << pEnd << "\n";
    
    // Example 10: Using pointers with a const array (read-only access)
    const int constArr[3] = {100, 200, 300};
    const int* pConstArr = constArr;
    cout << "Example 10: constArr via pointer: " << pConstArr[0] << ", " << pConstArr[1] << ", " << pConstArr[2] << "\n";
    
    // Diagram:
    //   [Array: numbers]
    //    +-----+-----+-----+-----+-----+
    //    |  5  | 10  | 15  | 20  | 25  |
    //    +-----+-----+-----+-----+-----+
    //     ^ (numbers / pNumbers) pointer to first element.
}

// ========================================================
// Main Function
// ========================================================
int main() {
    cout << "Chapter 6: Pointers - Unlocking Memory Addresses (Basics) 📍🔑\n";
    cout << "-------------------------------------------------------------\n";
    
    // Call all demonstration functions:
    pointerDeclarationExamples();
    addressOfOperatorExamples();
    dereferenceOperatorExamples();
    pointerInitializationExamples();
    nullPointerExamples();
    pointerArithmeticExamples();
    pointersAndArraysExamples();
    
    cout << "\nAll examples completed successfully! 🚀\n";
    return 0;
}
