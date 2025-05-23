#include <iostream>
#include <cstring>
#include <memory>
#include <vector>
#include <exception>
#include <stdexcept>
#include <climits> // For SIZE_MAX

// Example 1: Default Constructor and Destructor
class Example1 {
public:
    Example1() {
        std::cout << "Example1: Default Constructor called" << std::endl;
    }
    ~Example1() {
        std::cout << "Example1: Destructor called" << std::endl;
    }
};

// Example 2: Parameterized Constructor
class Example2 {
private:
    int value;
public:
    Example2(int v) {
        this-> value = v; 
        std::cout << "Example2: Parameterized Constructor called with value = " << value << std::endl;
    }
    ~Example2() {
        std::cout << "Example2: Destructor called for value = " << value << std::endl;
    }
};

// Example 3: Constructor Overloading
class Example3 {
private:
    int x, y;
public:
    Example3() : x(0), y(0) {
        std::cout << "Example3: Default Constructor called" << std::endl;
    }
    Example3(int val) : x(val), y(val) {
        std::cout << "Example3: Constructor with one int called" << std::endl;
    }
    Example3(int xVal, int yVal) : x(xVal), y(yVal) {
        std::cout << "Example3: Constructor with two ints called" << std::endl;
    }
    ~Example3() {
        std::cout << "Example3: Destructor called" << std::endl;
    }
};

// Example 4: Copy Constructor
class Example4 {
private:
    int* data;
public:
    Example4(int value) {
        data = new int(value);
        std::cout << "Example4: Constructor called, data = " << *data << std::endl;
    }
    Example4(const Example4& other) {
        data = new int(*other.data);
        std::cout << "Example4: Copy Constructor called, data = " << *data << std::endl;
    }
    ~Example4() {
        delete data;
        std::cout << "Example4: Destructor called" << std::endl;
    }
};

// Example 5: Move Constructor (Advanced)
class Example5 {
private:
    int* data;
public:
    Example5(int value) {
        data = new int(value);
        std::cout << "Example5: Constructor called, data = " << *data << std::endl;
    }
    Example5(Example5&& other) noexcept : data(nullptr) {
        data = other.data;
        other.data = nullptr;
        std::cout << "Example5: Move Constructor called" << std::endl;
    }
    ~Example5() {
        delete data;
        std::cout << "Example5: Destructor called" << std::endl;
    }
};

// Example 6: Constructors with Default Arguments
class Example6 {
private:
    int x, y;
public:
    Example6(int xVal = 10, int yVal = 20) : x(xVal), y(yVal) {
        std::cout << "Example6: Constructor called with x = " << x << ", y = " << y << std::endl;
    }
    ~Example6() {
        std::cout << "Example6: Destructor called" << std::endl;
    }
};

// Example 7: Initializer List
class Example7 {
private:
    const int x; // const member variable must be initialized in the constructor
    int& y;
public:
    Example7(int val, int& ref) : x(val), y(ref) {
        std::cout << "Example7: Constructor called with x = " << x << ", y = " << y << std::endl;
    }
    ~Example7() {
        std::cout << "Example7: Destructor called" << std::endl;
    }
};

// Example 8: Constructor with Exception Handling
class Example8 {
private:
    int* data;
public:
    Example8(int size) : data(nullptr) {
        std::cout << "Example8: Constructor called" << std::endl;
        if (size <= 0) {
            throw std::invalid_argument("Size must be positive");
        }
        data = new int[size];
    }
    ~Example8() {
        delete[] data;
        std::cout << "Example8: Destructor called" << std::endl;
    }
};

// Example 9: Destructor Exception Handling (Avoid Throwing Exceptions)
class Example9 {
public:
    ~Example9() noexcept {
        try {
            // Cleanup code that might throw
            std::cout << "Example9: Destructor called" << std::endl;
        } catch (...) {
            std::cerr << "Exception caught in Destructor" << std::endl;
        }
    }
};

// Example 10: Virtual Destructor for Base Class
class BaseExample10 {
public:
    BaseExample10() {
        std::cout << "BaseExample10: Constructor called" << std::endl;
    }
    virtual ~BaseExample10() {
        std::cout << "BaseExample10: Virtual Destructor called" << std::endl;
    }
};

class DerivedExample10 : public BaseExample10 {
private:
    int* data;
public:
    DerivedExample10() {
        data = new int(10);
        std::cout << "DerivedExample10: Constructor called" << std::endl;
    }
    ~DerivedExample10() {
        delete data;
        std::cout << "DerivedExample10: Destructor called" << std::endl;
    }
};

// Example 11: Copy Assignment Operator (deep copy)
class Example11 {
private:
    int* data;
public:
    Example11(int val) {
        data = new int(val);
        std::cout << "Example11: Constructor called, data = " << *data << std::endl;
    }
    Example11(const Example11& other) {
        data = new int(*other.data);
        std::cout << "Example11: Copy Constructor called, data = " << *data << std::endl;
    }
    Example11& operator=(const Example11& other) {
        if (this != &other) {
            delete data;
            data = new int(*other.data);
            std::cout << "Example11: Copy Assignment Operator called, data = " << *data << std::endl;
        }
        return *this;
    }
    ~Example11() {
        delete data;
        std::cout << "Example11: Destructor called" << std::endl;
    }
};

// Example 12: Move Assignment Operator
class Example12 {
private:
    int* data;
public:
    Example12(int val) : data(new int(val)) {
        std::cout << "Example12: Constructor called, data = " << *data << std::endl;
    }
    Example12(Example12&& other) noexcept : data(nullptr) {
        data = other.data;
        other.data = nullptr;
        std::cout << "Example12: Move Constructor called" << std::endl;
    }
    Example12& operator=(Example12&& other) noexcept {
        if (this != &other) {
            delete data;
            data = other.data;
            other.data = nullptr;
            std::cout << "Example12: Move Assignment Operator called" << std::endl;
        }
        return *this;
    }
    ~Example12() {
        delete data;
        std::cout << "Example12: Destructor called" << std::endl;
    }
};

// Example 13: Destructor in Inheritance without Virtual Destructor
class BaseExample13 {
public:
    BaseExample13() {
        std::cout << "BaseExample13: Constructor called" << std::endl;
    }
    ~BaseExample13() {
        std::cout << "BaseExample13: Destructor called" << std::endl;
    }
};

class DerivedExample13 : public BaseExample13 {
private:
    int* data;
public:
    DerivedExample13() {
        data = new int(800);
        std::cout << "DerivedExample13: Constructor called" << std::endl;
    }
    ~DerivedExample13() {
        delete data;
        std::cout << "DerivedExample13: Destructor called" << std::endl;
    }
};

// Example 14: Explicit Constructor
class Example14 {
private:
    int value;
public:
    explicit Example14(int v) : value(v) {
        std::cout << "Example14: Explicit Constructor called with value = " << value << std::endl;
    }
    ~Example14() {
        std::cout << "Example14: Destructor called" << std::endl;
    }
};

// Example 15: Static Member Initialization in Constructor
class Example15 {
private:
    static int count;
public:
    Example15() {
        ++count;
        std::cout << "Example15: Constructor called, count = " << count << std::endl;
    }
    ~Example15() {
        --count;
        std::cout << "Example15: Destructor called, count = " << count << std::endl;
    }
    static int getCount() {
        return count;
    }
};
int Example15::count = 0;

// Example 16: Object Slicing in Copy Constructor
class BaseExample16 {
public:
    BaseExample16() {
        std::cout << "BaseExample16: Constructor called" << std::endl;
    }
    ~BaseExample16() {
        std::cout << "BaseExample16: Destructor called" << std::endl;
    }
};

class DerivedExample16 : public BaseExample16 {
public:
    DerivedExample16() {
        std::cout << "DerivedExample16: Constructor called" << std::endl;
    }
    ~DerivedExample16() {
        std::cout << "DerivedExample16: Destructor called" << std::endl;
    }
};

// Example 17: Allocation Failure in Constructor
class Example17 {
private:
    int* data;
public:
    Example17(size_t size) : data(nullptr) {
        data = new(std::nothrow) int[size];
        if (!data) {
            throw std::bad_alloc();
        }
        std::cout << "Example17: Constructor called, memory allocated" << std::endl;
    }
    ~Example17() {
        delete[] data;
        std::cout << "Example17: Destructor called, memory released" << std::endl;
    }
};

// Example 18: Destructor and Virtual Base Class
class BaseExample18 {
public:
    BaseExample18() {
        std::cout << "BaseExample18: Constructor called" << std::endl;
    }
    virtual ~BaseExample18() {
        std::cout << "BaseExample18: Virtual Destructor called" << std::endl;
    }
};

class DerivedExample18 : public virtual BaseExample18 {
public:
    DerivedExample18() {
        std::cout << "DerivedExample18: Constructor called" << std::endl;
    }
    ~DerivedExample18() {
        std::cout << "DerivedExample18: Destructor called" << std::endl;
    }
};

// Example 19: Destructor and Smart Pointers
class Example19 {
public:
    Example19() {
        std::cout << "Example19: Constructor called" << std::endl;
    }
    ~Example19() {
        std::cout << "Example19: Destructor called" << std::endl;
    }
};

// Example 20: Preventing Object Copy (Deleted Copy Constructor)
class Example20 {
public:
    Example20() {
        std::cout << "Example20: Constructor called" << std::endl;
    }
    Example20(const Example20&) = delete; // Disable copy constructor
    Example20& operator=(const Example20&) = delete; // Disable copy assignment
    ~Example20() {
        std::cout << "Example20: Destructor called" << std::endl;
    }
};

// MAIN FUNCTION TO DEMONSTRATE EXAMPLES
int main() {
    // Example 1 Demonstration
    {
        Example1 e1;
    }

    // Example 2 Demonstration
    {
        Example2 e2(100);
    }

    // Example 3 Demonstration
    {
        Example3 e31;
        Example3 e32(50);
        Example3 e33(30, 40);
    }

    // Example 4 Demonstration
    {
        Example4 e4(200);
        Example4 e4Copy(e4);
    }

    // Example 5 Demonstration
    {
        Example5 e5(300);
        Example5 e5Moved(std::move(e5));
    }

    // Example 6 Demonstration
    {
        Example6 e6a;
        Example6 e6b(15);
        Example6 e6c(25, 35);
    }

    // Example 7 Demonstration
    {
        int refVal = 50;
        Example7 e7(40, refVal);
    }

    // Example 8 Demonstration with Exception Handling
    try {
        Example8 e8(-5); // This should throw an exception
    } catch (const std::exception& ex) {
        std::cerr << "Exception caught in main: " << ex.what() << std::endl;
    }

    // Example 9 Demonstration
    {
        Example9 e9;
    }

    // Example 10 Demonstration
    {
        BaseExample10* e10 = new DerivedExample10();
        delete e10; // Should call both Base and Derived destructors
    }

    // Additional Examples to cover 10 cases

    // Example 11: Copy Assignment Operator (deep copy)
    {
        Example11 e11a(400);
        Example11 e11b = e11a;
        Example11 e11c(500);
        e11c = e11a;
    }

    // Example 12: Move Assignment Operator
    {
        Example12 e12a(600);
        Example12 e12b = std::move(e12a);
        Example12 e12c(700);
        e12c = std::move(e12b);
    }

    // Example 13: Destructor in Inheritance without Virtual Destructor
    {
        BaseExample13* e13 = new DerivedExample13();
        delete e13; // Only Base Destructor called, memory leak occurs
    }

    // Example 14: Explicit Constructor
    {
        Example14 e14(900);
        // Example14 e14b = 1000; // Error: cannot convert from int to Example14
    }

    // Example 15: Static Member Initialization in Constructor
    {
        Example15 e15a;
        Example15 e15b;
        std::cout << "Current count: " << Example15::getCount() << std::endl;
    }
    std::cout << "Count after objects destroyed: " << Example15::getCount() << std::endl;

    // Example 16: Object Slicing in Copy Constructor
    {
        DerivedExample16 derivedObj;
        BaseExample16 baseCopy = derivedObj; // Object slicing occurs
    }

    // Example 17: Allocation Failure in Constructor
    try {
        Example17 e17(SIZE_MAX); // Likely to fail allocation
    } catch (const std::exception& ex) {
        std::cerr << "Exception caught in main: " << ex.what() << std::endl;
    }

    // Example 18: Destructor and Smart Pointers
    {
        BaseExample18* e18 = new DerivedExample18();
        delete e18;
    }

    // Example 19: Destructor and Smart Pointers
    {
        std::shared_ptr<Example19> sptr = std::make_shared<Example19>();
    }

    // Example 20: Preventing Object Copy (Deleted Copy Constructor)
    {
        Example20 e20;
        // Example20 e20Copy(e20); // Error: Call to deleted copy constructor
    }

    return 0;
}