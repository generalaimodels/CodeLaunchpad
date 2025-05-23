// AdvancedTemplatesExamples.cpp
#include <iostream>
#include <vector>
#include <string>
#include <type_traits>
#include <utility>
#include <tuple>
#include <array>

using namespace std;

/*
👨‍💻 Example 1: Basic Function Template
- Concept: Generic functions using templates.
*/
template <typename T>
T add(T a, T b) {
    return a + b;
}

/*
👩‍💻 Example 2: Function Template with Multiple Parameters
- Concept: Templates with multiple type parameters.
*/
template <typename T, typename U>
auto multiply(T a, U b) -> decltype(a * b) {
    return a * b;
}

/*
👨‍💻 Example 3: Class Template
- Concept: Defining classes with templates.
*/
template <typename T>
class Pair {
public:
    Pair(T first, T second) : first_(first), second_(second) {}
    T first() const { return first_; }
    T second() const { return second_; }
private:
    T first_, second_;
};

/*
👩‍💻 Example 4: Non-type Template Parameters
- Concept: Using constexpr values as template parameters.
*/
template <typename T, size_t Size>
class Array {
public:
    T& operator[](size_t index) { return data_[index]; }
    size_t size() const { return Size; }
private:
    T data_[Size];
};

/*
👨‍💻 Example 5: Template Specialization
- Concept: Specializing templates for specific types.
*/
template <>
class Pair<string> {
public:
    Pair(string first, string second) : first_(first), second_(second) {}
    string first() const { return first_; }
    string second() const { return second_; }
private:
    string first_, second_;
};

/*
👩‍💻 Example 6: Partial Specialization
- Concept: Partially specializing class templates.
*/
template <typename T>
class Pair<T*> {
public:
    Pair(T* first, T* second) : first_(first), second_(second) {}
    T* first() const { return first_; }
    T* second() const { return second_; }
private:
    T *first_, *second_;
};

/*
👨‍💻 Example 7: Variadic Templates
- Concept: Templates that take a variable number of arguments.
*/
template <typename... Args>
void print(Args... args) {
    (cout << ... << args) << endl; // Fold expression (C++17)
}

/*
👩‍💻 Example 8: Fold Expressions (C++17)
- Concept: Simplify operations on parameter packs.
*/
template <typename... Args>
auto sum(Args... args) {
    return (args + ...); // Right fold
}

/*
👨‍💻 Example 9: Template Template Parameters
- Concept: Templates that accept other templates as parameters.
*/
template <template <typename, typename> class ContainerType, typename ValueType>
class Container {
public:
    void add(const ValueType& value) { container_.push_back(value); }
    void show() const {
        for (const auto& val : container_) cout << val << " ";
        cout << endl;
    }
private:
    ContainerType<ValueType, allocator<ValueType>> container_;
};

/*
👩‍💻 Example 10: SFINAE (Substitution Failure Is Not An Error)
- Concept: Enable/disable templates based on type traits.
*/
template <typename T>
auto is_iterable(int) -> decltype(begin(declval<T&>()), true_type{}) {
    return true_type{};
}

template <typename T>
false_type is_iterable(...) {
    return false_type{};
}

/*
👨‍💻 Example 11: Enable_if for Function Overloading
- Concept: Conditionally enable functions.
*/
template <typename T>
typename enable_if<is_integral<T>::value, bool>::type
is_even(T x) {
    return x % 2 == 0;
}

/*
👩‍💻 Example 12: constexpr and Templates
- Concept: Compile-time computations with templates.
*/
template <int N>
constexpr int factorial() {
    if constexpr (N > 1)
        return N * factorial<N - 1>();
    else
        return 1;
}

/*
👨‍💻 Example 13: Variable Templates (C++14)
- Concept: Templates for variables.
*/
template <typename T>
constexpr T pi = T(3.1415926535897932385);

/*
👩‍💻 Example 14: Alias Templates
- Concept: Simplifying complex template syntax.
*/
template <typename T>
using Vec = vector<T>;

/*
👨‍💻 Example 15: Template Recursion
- Concept: Recursive templates for compile-time computations.
*/
template <int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template <>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

template <>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

/*
👩‍💻 Example 16: Type Traits
- Concept: Inspecting type properties at compile time.
*/
template <typename T>
void type_properties() {
    cout << boolalpha;
    cout << "Is pointer: " << is_pointer<T>::value << endl;
    cout << "Is array: " << is_array<T>::value << endl;
    cout << "Is integral: " << is_integral<T>::value << endl;
    cout << "Is floating point: " << is_floating_point<T>::value << endl;
}

/*
👨‍💻 Example 17: CRTP (Curiously Recurring Template Pattern)
- Concept: Static polymorphism with templates.
*/
template <typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
    // ...
};

class Derived : public Base<Derived> {
public:
    void implementation() {
        cout << "Derived implementation\n";
    }
};

/*
👩‍💻 Example 18: Template Meta-programming
- Concept: Code that generates or manipulates code at compile time.
*/
template <bool Condition, typename TrueType, typename FalseType>
struct Conditional {
    typedef TrueType type;
};

template <typename TrueType, typename FalseType>
struct Conditional<false, TrueType, FalseType> {
    typedef FalseType type;
};

/*
👨‍💻 Example 19: Perfect Forwarding
- Concept: Forwarding arguments while preserving value category.
*/
template <typename T, typename U>
void wrapper(T&& t, U&& u) {
    func(forward<T>(t), forward<U>(u));
}

void func(int& a, int& b) {
    cout << "Lvalue overload\n";
}

void func(int&& a, int&& b) {
    cout << "Rvalue overload\n";
}

/*
👩‍💻 Example 20: Compile-time String Hashing
- Concept: Generating hashes at compile time.
*/
constexpr unsigned int hash(const char* str, int h = 0) {
    return !str[h] ? 5381 : (hash(str, h+1) * 33) ^ str[h];
}

/*
👨‍💻 Example 21: Concept Constraints (C++20)
- Concept: Enforcing template parameters satisfy certain conditions.
*/
#if __cpp_concepts
template <typename T>
concept Integral = is_integral_v<T>;

template <Integral T>
T gcd(T a, T b) {
    if (b == 0) return a;
    else return gcd(b, a % b);
}
#endif

/*
👩‍💻 Example 22: Templates and Lambdas
- Concept: Generic lambdas with auto parameters.
*/
auto lambda_add = [](auto a, auto b) {
    return a + b;
};

/*
👨‍💻 Example 23: Tag Dispatching
- Concept: Choosing implementations based on types.
*/
void process(int value) {
    cout << "Processing integer: " << value << endl;
}

void processImpl(int value, true_type) {
    cout << "Processing even integer: " << value << endl;
}

void processImpl(int value, false_type) {
    cout << "Processing odd integer: " << value << endl;
}

void process(int value) {
    processImpl(value, bool_constant<(value % 2 == 0)>());
}

/*
👩‍💻 Example 24: Expression SFINAE
- Concept: Using SFINAE on expressions rather than types.
*/
template <typename T>
auto has_begin(T&& t) -> decltype(t.begin(), true) {
    return true;
}

template <typename T>
bool has_begin(...) {
    return false;
}

/*
👨‍💻 Example 25: Detection Idiom
- Concept: Checking if types have certain members.
*/
template <typename, typename = void>
struct has_size_type : false_type {};

template <typename T>
struct has_size_type<T, void_t<typename T::size_type>> : true_type {};

/*
👩‍💻 Example 26: Template Introspection
- Concept: Extracting information from templates.
*/
template <typename T>
struct TypeInfo;

template <template <typename, typename...> class Container, typename T, typename... Args>
struct TypeInfo<Container<T, Args...>> {
    using ValueType = T;
    static constexpr size_t ArgCount = sizeof...(Args);
};

/*
👨‍💻 Example 27: Policy-Based Design with Templates
- Concept: Building classes with interchangeable policies.
*/
template <typename SortingPolicy>
class DataManager : public SortingPolicy {
public:
    void process() {
        this->sort(data_);
    }
private:
    vector<int> data_{3, 1, 2};
};

class AscendingSort {
public:
    void sort(vector<int>& data) {
        sort(data.begin(), data.end());
    }
};

/*
👩‍💻 Example 28: Mixins using Templates
- Concept: Extending classes with additional functionality.
*/
template <typename T>
class Logging : public T {
public:
    void log(const string& message) {
        cout << "Log: " << message << endl;
    }
};

/*
👨‍💻 Example 29: Member Function Templates
- Concept: Templates within class methods.
*/
class Calculator {
public:
    template <typename T>
    T multiply(T a, T b) {
        return a * b;
    }
};

/*
👩‍💻 Example 30: Template Argument Deduction Guides (C++17)
- Concept: Deduce template parameters without specifying them explicitly.
*/
template <typename T>
class Wrapper {
public:
    Wrapper(T value) : value_(value) {}
private:
    T value_;
};

// Deduction guide
template <typename T>
Wrapper(T) -> Wrapper<T>;