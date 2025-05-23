/* 
======================================================
🏠🛠️ Constructors and Destructors Examples in C++ 🧹🗑️
======================================================

This file contains multiple examples demonstrating the usage of constructors and destructors
in C++. Each concept is illustrated with at least 10 different examples.

Think of constructors as the birth of an object 👶, where it gets initialized and ready to use.
Destructors are like the object's cleanup crew 🧹, tidying up before the object is destroyed.

Let's dive into the code! 🚀
*/

/*
----------------------------------------
🛠️ Constructors Examples 🛠️
----------------------------------------
*/

#include <iostream>
#include <cstring>

// Example 1: Default Constructor
class DefaultConstructorExample {
public:
    int x;
    DefaultConstructorExample() {
        x = 0;
        std::cout << "DefaultConstructorExample: Default constructor called, x = " << x << std::endl;
    }
};

// Example 2: Parameterized Constructor
class ParameterizedConstructorExample {
public:
    int x;
    ParameterizedConstructorExample(int value) {
        x = value;
        std::cout << "ParameterizedConstructorExample: Parameterized constructor called, x = " << x << std::endl;
    }
};

// Example 3: Constructor Overloading
class OverloadedConstructorExample {
public:
    int x, y;
    // Default constructor
    OverloadedConstructorExample() {
        x = 0;
        y = 0;
        std::cout << "OverloadedConstructorExample: Default constructor called, x = " << x << ", y = " << y << std::endl;
    }
    // Single parameter constructor
    OverloadedConstructorExample(int value) {
        x = value;
        y = value;
        std::cout << "OverloadedConstructorExample: Single parameter constructor called, x = y = " << x << std::endl;
    }
    // Two parameter constructor
    OverloadedConstructorExample(int xValue, int yValue) {
        x = xValue;
        y = yValue;
        std::cout << "OverloadedConstructorExample: Two parameter constructor called, x = " << x << ", y = " << y << std::endl;
    }
};

// Example 4: Copy Constructor
class CopyConstructorExample {
public:
    int x;
    CopyConstructorExample(int value) {
        x = value;
        std::cout << "CopyConstructorExample: Parameterized constructor called, x = " << x << std::endl;
    }
    CopyConstructorExample(const CopyConstructorExample &obj) {
        x = obj.x;
        std::cout << "CopyConstructorExample: Copy constructor called, x = " << x << std::endl;
    }
};

// Example 5: Constructor with Initializer List
class InitializerListConstructorExample {
public:
    const int x;
    InitializerListConstructorExample(int value) : x(value) {
        std::cout << "InitializerListConstructorExample: Constructor called, x = " << x << std::endl;
    }
};

// Example 6: Dynamic Memory Allocation in Constructor
class DynamicAllocationConstructorExample {
public:
    int *ptr;
    DynamicAllocationConstructorExample(int value) {
        ptr = new int(value);
        std::cout << "DynamicAllocationConstructorExample: Constructor called, ptr points to " << *ptr << std::endl;
    }
    ~DynamicAllocationConstructorExample() {
        delete ptr;
        std::cout << "DynamicAllocationConstructorExample: Destructor called, memory released" << std::endl;
    }
};

// Example 7: Default Arguments in Constructor
class DefaultArgumentConstructorExample {
public:
    int x, y;
    DefaultArgumentConstructorExample(int xValue = 0, int yValue = 0) {
        x = xValue;
        y = yValue;
        std::cout << "DefaultArgumentConstructorExample: Constructor called, x = " << x << ", y = " << y << std::endl;
    }
};

// Example 8: Explicit Constructor
class ExplicitConstructorExample {
public:
    int x;
    explicit ExplicitConstructorExample(int value) {
        x = value;
        std::cout << "ExplicitConstructorExample: Explicit constructor called, x = " << x << std::endl;
    }
};

// Example 9: Constructor Delegation (C++11 and above)
class DelegatingConstructorExample {
public:
    int x, y;
    DelegatingConstructorExample() : DelegatingConstructorExample(0, 0) {
        std::cout << "DelegatingConstructorExample: Default constructor called" << std::endl;
    }
    DelegatingConstructorExample(int value) : DelegatingConstructorExample(value, value) {
        std::cout << "DelegatingConstructorExample: Single parameter constructor called" << std::endl;
    }
    DelegatingConstructorExample(int xValue, int yValue) {
        x = xValue;
        y = yValue;
        std::cout << "DelegatingConstructorExample: Two parameter constructor called, x = " << x << ", y = " << y << std::endl;
    }
};

// Example 10: Class with Static Data Member and Constructor
class StaticMemberConstructorExample {
public:
    static int count;
    StaticMemberConstructorExample() {
        count++;
        std::cout << "StaticMemberConstructorExample: Constructor called, count = " << count << std::endl;
    }
};
int StaticMemberConstructorExample::count = 0;

/*
----------------------------------------
🧹 Destructors Examples 🧹
----------------------------------------
*/

// Example 1: Destructor Basic Example
class BasicDestructorExample {
public:
    BasicDestructorExample() {
        std::cout << "BasicDestructorExample: Constructor called" << std::endl;
    }
    ~BasicDestructorExample() {
        std::cout << "BasicDestructorExample: Destructor called" << std::endl;
    }
};

// Example 2: Destructor Releasing Memory
class MemoryReleasingDestructorExample {
public:
    int *ptr;
    MemoryReleasingDestructorExample() {
        ptr = new int(42);
        std::cout << "MemoryReleasingDestructorExample: Constructor called, allocated memory" << std::endl;
    }
    ~MemoryReleasingDestructorExample() {
        delete ptr;
        std::cout << "MemoryReleasingDestructorExample: Destructor called, memory released" << std::endl;
    }
};

// Example 3: Destructor for Closing File
#include <fstream>
class FileClosingDestructorExample {
public:
    std::ofstream file;
    FileClosingDestructorExample(const char* filename) {
        file.open(filename);
        std::cout << "FileClosingDestructorExample: File opened" << std::endl;
    }
    ~FileClosingDestructorExample() {
        if (file.is_open()) {
            file.close();
            std::cout << "FileClosingDestructorExample: File closed" << std::endl;
        }
    }
};

// Example 4: Destructor in Base and Derived Classes
class BaseClassDestructorExample {
public:
    BaseClassDestructorExample() {
        std::cout << "BaseClassDestructorExample: Constructor called" << std::endl;
    }
    virtual ~BaseClassDestructorExample() {
        std::cout << "BaseClassDestructorExample: Destructor called" << std::endl;
    }
};
class DerivedClassDestructorExample : public BaseClassDestructorExample {
public:
    DerivedClassDestructorExample() {
        std::cout << "DerivedClassDestructorExample: Constructor called" << std::endl;
    }
    ~DerivedClassDestructorExample() {
        std::cout << "DerivedClassDestructorExample: Destructor called" << std::endl;
    }
};

// Example 5: Destructor Order of Execution
class First {
public:
    First() {
        std::cout << "First: Constructor called" << std::endl;
    }
    ~First() {
        std::cout << "First: Destructor called" << std::endl;
    }
};
class Second {
public:
    Second() {
        std::cout << "Second: Constructor called" << std::endl;
    }
    ~Second() {
        std::cout << "Second: Destructor called" << std::endl;
    }
};
class Third {
public:
    First f;
    Second s;
    Third() {
        std::cout << "Third: Constructor called" << std::endl;
    }
    ~Third() {
        std::cout << "Third: Destructor called" << std::endl;
    }
};

// Example 6: Destructor with Dynamic Array
class DynamicArrayDestructorExample {
public:
    int *arr;
    int size;
    DynamicArrayDestructorExample(int s) {
        size = s;
        arr = new int[size];
        std::cout << "DynamicArrayDestructorExample: Constructor called, array of size " << size << " allocated" << std::endl;
    }
    ~DynamicArrayDestructorExample() {
        delete[] arr;
        std::cout << "DynamicArrayDestructorExample: Destructor called, array memory released" << std::endl;
    }
};

// Example 7: Destructor in Smart Pointers (unique_ptr)
#include <memory>
class UniquePtrDestructorExample {
public:
    std::unique_ptr<int> ptr;
    UniquePtrDestructorExample(int value) {
        ptr = std::make_unique<int>(value);
        std::cout << "UniquePtrDestructorExample: Constructor called, unique_ptr allocated" << std::endl;
    }
    ~UniquePtrDestructorExample() {
        // unique_ptr automatically cleans up
        std::cout << "UniquePtrDestructorExample: Destructor called, unique_ptr will release memory" << std::endl;
    }
};

// Example 8: Destructor in Virtual Functions
class VirtualDestructorExampleBase {
public:
    VirtualDestructorExampleBase() {
        std::cout << "VirtualDestructorExampleBase: Constructor called" << std::endl;
    }
    virtual ~VirtualDestructorExampleBase() {
        std::cout << "VirtualDestructorExampleBase: Destructor called" << std::endl;
    }
};
class VirtualDestructorExampleDerived : public VirtualDestructorExampleBase {
public:
    VirtualDestructorExampleDerived() {
        std::cout << "VirtualDestructorExampleDerived: Constructor called" << std::endl;
    }
    ~VirtualDestructorExampleDerived() {
        std::cout << "VirtualDestructorExampleDerived: Destructor called" << std::endl;
    }
};

// Example 9: Destructor in Exception Handling
class ExceptionHandlingDestructorExample {
public:
    ExceptionHandlingDestructorExample() {
        std::cout << "ExceptionHandlingDestructorExample: Constructor called" << std::endl;
        throw std::runtime_error("Exception in constructor");
    }
    ~ExceptionHandlingDestructorExample() {
        std::cout << "ExceptionHandlingDestructorExample: Destructor called" << std::endl;
    }
};

// Example 10: Destructor in Static Object
class StaticObjectDestructorExample {
public:
    StaticObjectDestructorExample() {
        std::cout << "StaticObjectDestructorExample: Constructor called" << std::endl;
    }
    ~StaticObjectDestructorExample() {
        std::cout << "StaticObjectDestructorExample: Destructor called" << std::endl;
    }
};

// Global static object (will be destroyed after main exits)
StaticObjectDestructorExample staticObj;

/*
----------------------------------------
🚀 Main Function 🚀
----------------------------------------
*/

int main() {
    std::cout << "=== Constructors Examples ===" << std::endl;

    // Constructors Examples
    std::cout << "\nExample 1: Default Constructor" << std::endl;
    DefaultConstructorExample obj1;

    std::cout << "\nExample 2: Parameterized Constructor" << std::endl;
    ParameterizedConstructorExample obj2(10);

    std::cout << "\nExample 3: Constructor Overloading" << std::endl;
    OverloadedConstructorExample obj3a;
    OverloadedConstructorExample obj3b(5);
    OverloadedConstructorExample obj3c(3, 7);

    std::cout << "\nExample 4: Copy Constructor" << std::endl;
    CopyConstructorExample obj4a(20);
    CopyConstructorExample obj4b = obj4a;

    std::cout << "\nExample 5: Initializer List Constructor" << std::endl;
    InitializerListConstructorExample obj5(30);

    std::cout << "\nExample 6: Dynamic Memory Allocation in Constructor" << std::endl;
    DynamicAllocationConstructorExample obj6(40);

    std::cout << "\nExample 7: Default Arguments in Constructor" << std::endl;
    DefaultArgumentConstructorExample obj7a;
    DefaultArgumentConstructorExample obj7b(50);
    DefaultArgumentConstructorExample obj7c(60, 70);

    std::cout << "\nExample 8: Explicit Constructor" << std::endl;
    ExplicitConstructorExample obj8(80);
    //ExplicitConstructorExample obj8b = 90; // Error due to 'explicit', uncommenting will cause compile error

    std::cout << "\nExample 9: Constructor Delegation" << std::endl;
    DelegatingConstructorExample obj9a;
    DelegatingConstructorExample obj9b(100);
    DelegatingConstructorExample obj9c(110, 120);

    std::cout << "\nExample 10: Static Member in Constructor" << std::endl;
    StaticMemberConstructorExample obj10a;
    StaticMemberConstructorExample obj10b;
    StaticMemberConstructorExample obj10c;

    std::cout << "\n=== Destructors Examples ===" << std::endl;

    // Destructors Examples
    std::cout << "\nExample 1: Basic Destructor" << std::endl;
    {
        BasicDestructorExample objD1;
    } // objD1 goes out of scope here

    std::cout << "\nExample 2: Destructor Releasing Memory" << std::endl;
    {
        MemoryReleasingDestructorExample objD2;
    } // objD2 goes out of scope here

    std::cout << "\nExample 3: Destructor Closing File" << std::endl;
    {
        FileClosingDestructorExample objD3("example.txt");
    } // objD3 goes out of scope here

    std::cout << "\nExample 4: Destructor in Inheritance" << std::endl;
    {
        BaseClassDestructorExample* objD4 = new DerivedClassDestructorExample();
        delete objD4; // Correctly calls both destructors because of virtual destructor
    }

    std::cout << "\nExample 5: Destructor Order of Execution" << std::endl;
    {
        Third objD5;
    } // objD5 goes out of scope here; destructors called in reverse order of construction

    std::cout << "\nExample 6: Destructor with Dynamic Array" << std::endl;
    {
        DynamicArrayDestructorExample objD6(5);
    }

    std::cout << "\nExample 7: Destructor with unique_ptr" << std::endl;
    {
        UniquePtrDestructorExample objD7(130);
    }

    std::cout << "\nExample 8: Destructor in Virtual Functions" << std::endl;
    {
        VirtualDestructorExampleBase* objD8 = new VirtualDestructorExampleDerived();
        delete objD8; // Correctly calls both destructors
    }

    std::cout << "\nExample 9: Destructor in Exception Handling" << std::endl;
    try {
        ExceptionHandlingDestructorExample objD9;
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    } // Destructor called for partially constructed objD9

    std::cout << "\nExample 10: Destructor in Static Object" << std::endl;
    std::cout << "Static object 'staticObj' was created before main and will be destroyed after main exits." << std::endl;

    std::cout << "\nExiting main function." << std::endl;
    return 0;
}