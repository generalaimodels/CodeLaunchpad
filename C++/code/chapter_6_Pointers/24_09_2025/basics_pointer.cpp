#include <iostream>
#include <string>
#include <cstddef>

// This file explains C++ pointer and reference fundamentals through many small, focused examples.
// All explanations live in comments right next to the code they describe.
// Compile and run to observe console output; read comments to understand behavior and rules.

struct Point {
  int x{0};
  int y{0};
};

struct Base {
  virtual ~Base() = default;
  virtual const char* name() const { return "Base"; }
};

struct Derived : Base {
  int extra{42};
  const char* name() const override { return "Derived"; }
};

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

void overload_pick(int) { std::cout << "picked int overload\n"; }
void overload_pick(const char*) { std::cout << "picked pointer-to-char overload\n"; }

void print_separator(const char* title) {
  std::cout << "\n---------------- " << title << " ----------------\n";
}

// Concept 1: What is a pointer & reference
void concept1_pointer_and_reference() {
  print_separator("1) What is a pointer & reference");

  // Example 1: A pointer holds an address; a reference is an alias to an existing object.
  {
    int value{10};
    int* p = &value;       // pointer to int (can be reseated, can be null)
    int& r = value;        // reference to int (must bind to a valid object, never reseatable)
    *p = 11;               // changes 'value' through the pointer
    r = 12;                // changes 'value' through the reference (r is just another name for value)
    std::cout << "Ex1: value=" << value << " *p=" << *p << " r=" << r << '\n';
  }

  // Example 2: Pointers/references work with any object type (here: double)
  {
    double d{3.14};
    double* pd = &d;
    double& rd = d;
    *pd *= 2;   // d becomes 6.28
    rd += 1;    // d becomes 7.28
    std::cout << "Ex2: d=" << d << '\n';
  }

  // Example 3: Pointers/references to user-defined types (struct/class)
  {
    Point pt{1, 2};
    Point* ppt = &pt;   // pointer to Point
    Point& rpt = pt;    // reference to Point
    ppt->x = 5;         // use -> with pointers to access members
    rpt.y = 9;          // use . with references (they behave like the object itself)
    std::cout << "Ex3: pt=(" << pt.x << ',' << pt.y << ")\n";
  }

  // Example 4: Pointers/references with std::string (objects with dynamic internals)
  {
    std::string s{"hello"};
    std::string* ps = &s;
    std::string& rs = s;
    ps->append(" world");
    rs += "!";
    std::cout << "Ex4: s=" << s << '\n';
  }

  // Example 5: Pointer to function and reference to function
  {
    using BinOp = int(int, int);  // function type
    BinOp* op_ptr = &add;         // pointer to function
    BinOp& op_ref = mul;          // reference to function (binds to mul)
    std::cout << "Ex5: add(2,3)=" << op_ptr(2,3)
              << " mul(2,3)=" << op_ref(2,3) << '\n';
  }
}

// Concept 2: Definition of pointer (int* p), reference (int& r)
void concept2_definitions() {
  print_separator("2) Definitions: int* p, int& r");

  // Example 1: Clear pointer/reference declarations
  {
    int x{42};
    int* p = &x;   // int* is the pointer type
    int& r = x;    // int& is the reference type
    std::cout << "Ex1: *p=" << *p << " r=" << r << '\n';
  }

  // Example 2: Pitfall with multiple declarations in one line
  {
    // Only a is a pointer; b is an int. Prefer declaring one per line or use type aliases.
    int* a = nullptr, b = 0;
    std::cout << "Ex2: a=" << a << " (pointer), b=" << b << " (int)\n";
  }

  // Example 3: Pointers to pointers and references to pointers
  {
    int x{10};
    int* p = &x;      // pointer to int
    int** pp = &p;    // pointer to pointer to int
    int*& rp = p;     // reference to pointer (alias to 'p' itself)
    **pp = 20;        // modifies x through pp
    rp = &x;          // reseats p via reference-to-pointer (still &x here)
    std::cout << "Ex3: x=" << x << " *p=" << *p << " **pp=" << **pp << '\n';
  }

  // Example 4: References to const and pointers to const
  {
    int x{5};
    const int& rc = x;     // cannot modify x through rc
    const int* pc = &x;    // pointer to const int (cannot modify through pc)
    (void)rc; (void)pc;
    std::cout << "Ex4: x=" << x << " (rc/pc cannot modify it)\n";
  }

  // Example 5: Prefer type aliases for clarity
  {
    using IntPtr = int*;
    int y{7};
    IntPtr p1 = &y;  // clearer that p1 is a pointer type
    std::cout << "Ex5: *p1=" << *p1 << '\n';
  }
}

// Concept 3: Null pointers (nullptr in C++11)
void concept3_nullptr() {
  print_separator("3) Null pointers (nullptr)");

  // Example 1: Initialize pointers to nullptr when you don't have a valid address yet
  {
    int* p = nullptr;         // safe "no address" value
    if (p == nullptr) {
      std::cout << "Ex1: p is null\n";
    }
  }

  // Example 2: nullptr participates in overload resolution (type-safe compared to 0/NULL)
  {
    overload_pick(nullptr);   // picks pointer overload unambiguously
  }

  // Example 3: After delete, set pointer to nullptr to avoid dangling
  {
    int* p = new int(123);
    delete p;                 // object freed; p is now dangling
    p = nullptr;              // reset to null to avoid accidental dereference
    std::cout << "Ex3: p reset -> " << p << '\n';
  }

  // Example 4: Null function pointer
  {
    using FnPtr = int(*)(int,int);
    FnPtr f = nullptr;
    if (!f) {
      std::cout << "Ex4: function pointer is null\n";
    }
  }

  // Example 5: Always guard before dereferencing if pointer may be null
  {
    int* maybe = nullptr;
    if (maybe) {
      std::cout << *maybe << '\n';
    } else {
      std::cout << "Ex5: guard prevented null dereference\n";
    }
  }
}

// Concept 4: Pointer declaration vs initialization
void concept4_decl_vs_init() {
  print_separator("4) Declaration vs Initialization");

  // Example 1: Declaration without initialization is dangerous; always initialize
  {
    int* p_uninit;            // uninitialized (indeterminate) – do NOT use
    int* p_safe = nullptr;    // preferred
    (void)p_uninit; (void)p_safe;
    std::cout << "Ex1: prefer initializing pointers (e.g., to nullptr)\n";
  }

  // Example 2: Delayed initialization when address becomes available
  {
    int* p = nullptr;
    int x{9};
    p = &x;                   // now initialized
    std::cout << "Ex2: *p=" << *p << '\n';
  }

  // Example 3: Immediate initialization with address-of
  {
    int x{100};
    int* p = &x;
    std::cout << "Ex3: *p=" << *p << '\n';
  }

  // Example 4: Initialization via dynamic allocation
  {
    int* p = new int(77);
    std::cout << "Ex4: *p=" << *p << '\n';
    delete p;
  }

  // Example 5: Initialization to the first element of an array
  {
    int arr[3]{1,2,3};
    int* p = arr;   // arr decays to pointer to its first element
    std::cout << "Ex5: first=" << *p << " second=" << *(p+1) << '\n';
  }
}

// Concept 5: Basic usage of pointers and references
void concept5_basic_usage() {
  print_separator("5) Basic usage");

  // Example 1: Modify a variable via pointer
  {
    int x{1};
    int* p = &x;
    *p = 2;
    std::cout << "Ex1: x=" << x << '\n';
  }

  // Example 2: Modify a variable via reference (no dereference operator needed)
  {
    int y{3};
    int& r = y;
    r = 4;
    std::cout << "Ex2: y=" << y << '\n';
  }

  // Example 3: Pass pointer to a function to allow optional modification
  {
    auto set_if_pointer = [](int* p, int v) {
      if (p) { *p = v; }
    };
    int z{0};
    set_if_pointer(&z, 10);
    set_if_pointer(nullptr, 10); // safely ignored
    std::cout << "Ex3: z=" << z << '\n';
  }

  // Example 4: Pass by reference to express "must have object"
  {
    auto increment = [](int& ref) { ++ref; };
    int n{5};
    increment(n);
    std::cout << "Ex4: n=" << n << '\n';
  }

  // Example 5: Pointer to pointer can modify which object a pointer refers to
  {
    int a{1}, b{2};
    int* p = &a;
    int** pp = &p;
    *pp = &b;  // reseat p to point at b
    *p = 99;   // modifies b
    std::cout << "Ex5: a=" << a << " b=" << b << '\n';
  }
}

// Concept 6: Address-of (&) and dereference (*) operators
void concept6_address_and_deref() {
  print_separator("6) Address-of & and dereference *");

  // Example 1: & obtains the address; * dereferences the pointer
  {
    int x{7};
    int* p = &x;
    std::cout << "Ex1: &x=" << &x << " *p=" << *p << '\n';
  }

  // Example 2: Arrays: 'arr' decays to pointer to first element; '&arr' is pointer to the whole array
  {
    int arr[3]{10,20,30};
    int* p1 = arr;        // int*
    int (*p2)[3] = &arr;  // pointer to array of 3 int
    std::cout << "Ex2: p1=" << p1 << " &arr=" << p2 << " *p1=" << *p1 << " (*p2)[1]=" << (*p2)[1] << '\n';
  }

  // Example 3: Address of a reference is the address of the referenced object
  {
    int v{5};
    int& r = v;
    std::cout << "Ex3: &v=" << &v << " &r=" << &r << '\n';
  }

  // Example 4: *& cancels out (with correct types)
  {
    int x{42};
    int y = *&x;  // y becomes a copy of x
    std::cout << "Ex4: y=" << y << '\n';
  }

  // Example 5: Pointer to pointer: * and & compose naturally
  {
    int x{1};
    int* p = &x;
    int** pp = &p;
    std::cout << "Ex5: **pp=" << **pp << '\n';
  }
}

// Concept 7: Assigning addresses to pointers
void concept7_assigning_addresses() {
  print_separator("7) Assigning addresses to pointers");

  // Example 1: Assign the address of a variable to a pointer of matching type
  {
    int x{10};
    int* p = &x;
    std::cout << "Ex1: *p=" << *p << '\n';
  }

  // Example 2: Pointer to const when pointing at a const object (or non-const)
  {
    const int cx{3};
    const int* pc = &cx;  // ok
    int y{4};
    const int* pc2 = &y; // also ok (can't modify y through pc2)
    (void)pc; (void)pc2;
    std::cout << "Ex2: pointers to const assigned from const/non-const\n";
  }

  // Example 3: Upcasting from Derived* to Base*
  {
    Derived d;
    Derived* pd = &d;
    Base* pb = pd;    // implicit upcast
    std::cout << "Ex3: pb->name()=" << pb->name() << '\n';
  }

  // Example 4: void* can hold the address of any object type (but needs cast to use)
  {
    double pi{3.14159};
    void* vp = &pi;     // generic pointer
    double* dp = static_cast<double*>(vp); // cast back to correct type before dereference
    std::cout << "Ex4: *dp=" << *dp << '\n';
  }

  // Example 5: Function pointers assigned to function addresses
  {
    int (*op)(int,int) = &add;  // pointer to function
    std::cout << "Ex5: op(3,4)=" << op(3,4) << '\n';
  }
}

// Concept 8: Simple pointer dereferencing
void concept8_simple_deref() {
  print_separator("8) Simple pointer dereferencing");

  // Example 1: Read and write through an int*
  {
    int x{5};
    int* p = &x;
    *p = 6;
    std::cout << "Ex1: x=" << x << '\n';
  }

  // Example 2: Access struct members via pointer
  {
    Point pt{1,2};
    Point* p = &pt;
    p->x = 10;       // same as (*p).x = 10;
    (*p).y = 20;
    std::cout << "Ex2: pt=(" << pt.x << ',' << pt.y << ")\n";
  }

  // Example 3: Double dereference to reach the ultimate value
  {
    int n{7};
    int* p = &n;
    int** pp = &p;
    **pp = 8;
    std::cout << "Ex3: n=" << n << '\n';
  }

  // Example 4: Dereference and array indexing via pointer arithmetic
  {
    int arr[4]{1,2,3,4};
    int* p = arr;
    std::cout << "Ex4: *(p+2)=" << *(p+2) << " p[3]=" << p[3] << '\n';
  }

  // Example 5: Dereferencing a void* requires a cast to the correct type
  {
    int x{42};
    void* vp = &x;
    int* ip = static_cast<int*>(vp);
    std::cout << "Ex5: *ip=" << *ip << '\n';
  }
}

// Concept 9: References vs Pointers
void concept9_refs_vs_ptrs() {
  print_separator("9) References vs Pointers");

  // Example 1: Pointers can be reseated; references cannot
  {
    int a{1}, b{2};
    int* p = &a;  // can later point to b
    int& r = a;   // bound to 'a' forever
    p = &b;       // reseated
    // r = b;     // assigns value of b to a (does NOT reseat the reference)
    std::cout << "Ex1: a=" << a << " b=" << b << " *p=" << *p << " r=" << r << '\n';
  }

  // Example 2: References are never null; pointers can be null
  {
    int x{5};
    int* p = nullptr; // possible
    int& r = x;       // must bind to an object
    (void)p; (void)r;
    std::cout << "Ex2: pointer may be null, reference cannot (by design)\n";
  }

  // Example 3: Syntax ergonomics: reference looks like normal variable usage
  {
    int v{10};
    int* p = &v;
    int& r = v;
    *p += 1; // pointer requires explicit dereference
    r += 1;  // reference uses normal syntax
    std::cout << "Ex3: v=" << v << '\n';
  }

  // Example 4: Function parameters: reference means must-have, pointer can signal optional
  {
    auto must_have = [](int& x) { x *= 2; };
    auto maybe_have = [](int* x) { if (x) *x += 3; };
    int a{4};
    must_have(a);
    maybe_have(&a);
    maybe_have(nullptr);
    std::cout << "Ex4: a=" << a << '\n';
  }

  // Example 5: sizeof(pointer) is defined; reference size is not a language-level object size
  {
    int x{0};
    int* p = &x;
    int& r = x;
    std::cout << "Ex5: sizeof(p)=" << sizeof(p)
              << " (sizeof reference is not meaningful; usually compiler implements as an alias)\n";
    (void)r;
  }
}

// Concept 10: Reference alias vs actual pointer
void concept10_alias_vs_pointer_object() {
  print_separator("10) Reference alias vs pointer object");

  // Example 1: Reference is an alias: same address as the original
  {
    int x{9};
    int& rx = x;
    std::cout << "Ex1: &x=" << &x << " &rx=" << &rx << '\n';
  }

  // Example 2: Aliasing a struct; both names refer to the same memory
  {
    Point p{1,2};
    Point& rp = p;
    rp.x = 99; // modifies p.x
    std::cout << "Ex2: p.x=" << p.x << '\n';
  }

  // Example 3: Reference to pointer lets you reseat the pointer itself
  {
    int a{1}, b{2};
    int* ptr = &a;
    int*& rptr = ptr;  // alias to the pointer object
    rptr = &b;         // reseats ptr
    *ptr = 7;          // modifies b
    std::cout << "Ex3: a=" << a << " b=" << b << '\n';
  }

  // Example 4: Reference cannot be reseated; pointer can
  {
    int x{5}, y{6};
    int& r = x;
    int* p = &x;
    p = &y; // ok
    // r = y; // assigns y to x; does not reseat r
    std::cout << "Ex4: x=" << x << " y=" << y << " *p=" << *p << " r=" << r << '\n';
  }

  // Example 5: Const reference can bind to a temporary (extends its lifetime)
  {
    const std::string& ref = std::string("hello"); // temporary bound, lifetime extended to end of scope
    std::cout << "Ex5: ref=" << ref << '\n';
  }
}

// Concept 11: When to use reference vs pointer (syntax & semantics)
void concept11_when_to_use() {
  print_separator("11) When to use reference vs pointer");

  // Example 1: Use reference when parameter must be valid and non-null
  {
    auto scale = [](Point& pt, int k) { pt.x *= k; pt.y *= k; };
    Point p{2,3};
    scale(p, 2);
    std::cout << "Ex1: p=(" << p.x << ',' << p.y << ")\n";
  }

  // Example 2: Use pointer to indicate optional parameter (may be null)
  {
    auto maybe_increment = [](int* p) { if (p) ++*p; };
    int x{1};
    maybe_increment(&x);
    maybe_increment(nullptr);
    std::cout << "Ex2: x=" << x << '\n';
  }

  // Example 3: Return by reference to avoid copying and allow chaining (never return ref to local)
  {
    auto at = [](int* arr, std::size_t i) -> int& { return arr[i]; }; // safe if caller ensures array lives
    int data[3]{10,20,30};
    at(data, 1) = 99; // assign via returned reference
    std::cout << "Ex3: data[1]=" << data[1] << " (avoid returning reference to a local variable)\n";
  }

  // Example 4: Pointer for APIs dealing with buffers/arrays and re-seating
  {
    int buf[4]{1,2,3,4};
    int* p = buf;
    p += 2; // re-seat to third element
    std::cout << "Ex4: *p=" << *p << '\n';
  }

  // Example 5: Reference for operator overloading or fluent interfaces
  {
    struct Acc {
      int v{0};
      Acc& add(int x) { v += x; return *this; } // returns reference for chaining
    };
    Acc a;
    a.add(1).add(2).add(3);
    std::cout << "Ex5: a.v=" << a.v << '\n';
  }
}

// Concept 12: Pointer arithmetic (basics)
void concept12_pointer_arithmetic() {
  print_separator("12) Pointer arithmetic (basics)");

  // Example 1: p + 1 advances by sizeof(T); different for int vs char
  {
    int arr[3]{10,20,30};
    char bytes[3]{'a','b','c'};
    int* ip = arr;
    char* cp = bytes;
    std::cout << "Ex1: ip=" << static_cast<const void*>(ip)
              << " ip+1=" << static_cast<const void*>(ip+1) << '\n';
    std::cout << "     cp=" << static_cast<const void*>(cp)
              << " cp+1=" << static_cast<const void*>(cp+1) << '\n';
  }

  // Example 2: p - 1 moves backward within the same array
  {
    int arr[4]{1,2,3,4};
    int* p = &arr[2]; // points at 3
    std::cout << "Ex2: *p=" << *p << " *(p-1)=" << *(p-1) << '\n';
  }

  // Example 3: Pointer difference yields element count (within same array)
  {
    int arr[5]{0,1,2,3,4};
    int* a = &arr[1];
    int* b = &arr[4];
    std::ptrdiff_t diff = b - a; // 3 elements apart
    std::cout << "Ex3: b-a=" << diff << '\n';
  }

  // Example 4: Loop over array with pointer
  {
    int arr[3]{7,8,9};
    for (int* p = arr; p != arr + 3; ++p) {
      std::cout << "Ex4: " << *p << ' ';
    }
    std::cout << '\n';
  }

  // Example 5: One-past-the-end pointer is allowed, but must not be dereferenced
  {
    int arr[2]{5,6};
    int* end = arr + 2; // one past last
    std::cout << "Ex5: end pointer computed (not dereferenced)\n";
    (void)end;
  }
}

// Concept 13: Iterating arrays with pointers
void concept13_iterating_with_pointers() {
  print_separator("13) Iterating arrays with pointers");

  // Example 1: Forward iteration
  {
    int arr[4]{1,2,3,4};
    for (int* p = arr; p != arr + 4; ++p) {
      std::cout << "Ex1: " << *p << ' ';
    }
    std::cout << '\n';
  }

  // Example 2: Reverse iteration
  {
    int arr[4]{1,2,3,4};
    for (int* p = arr + 4; p != arr; ) {
      --p;
      std::cout << "Ex2: " << *p << ' ';
    }
    std::cout << '\n';
  }

  // Example 3: Iterate over a 2D array (row-major) using a flat pointer
  {
    int m[2][3]{{1,2,3},{4,5,6}};
    int* p = &m[0][0];
    for (std::size_t i = 0; i < 2 * 3; ++i) {
      std::cout << "Ex3: " << *(p + i) << ' ';
    }
    std::cout << '\n';
  }

  // Example 4: Iterate an array of structs and modify elements
  {
    Point pts[3]{{1,1},{2,2},{3,3}};
    for (Point* p = pts; p != pts + 3; ++p) {
      p->x += 10;
      p->y += 20;
    }
    std::cout << "Ex4: (" << pts[0].x << ',' << pts[0].y << ") ("
              << pts[1].x << ',' << pts[1].y << ") ("
              << pts[2].x << ',' << pts[2].y << ")\n";
  }

  // Example 5: Iterate a C-string until null terminator
  {
    const char* s = "abc";
    const char* p = s;
    while (*p != '\0') {
      std::cout << "Ex5: " << *p << ' ';
      ++p;
    }
    std::cout << '\n';
  }
}

// Concept 14: Const correctness basics (const int* p, int* const p, const int* const p)
void concept14_const_correctness() {
  print_separator("14) Const correctness basics");

  // Example 1: const int* p => pointer to const data (cannot modify through p; can reseat p)
  {
    int x{10};
    const int* p = &x; // pointer to const int
    // *p = 11;        // error: cannot modify through pointer-to-const
    int y{20};
    p = &y;            // ok: can reseat pointer
    std::cout << "Ex1: *p=" << *p << " (reseated to y)\n";
  }

  // Example 2: int* const p => const pointer to non-const data (cannot reseat; can modify data)
  {
    int x{5};
    int* const p = &x;
    *p = 6;            // ok: modify x
    // p = &x;         // error: cannot reseat const pointer
    std::cout << "Ex2: x=" << x << '\n';
  }

  // Example 3: const int* const p => const pointer to const data (neither modify data nor reseat)
  {
    int x{7};
    const int* const p = &x;
    std::cout << "Ex3: *p=" << *p << " (no modify, no reseat)\n";
  }

  // Example 4: Conversion rules: non-const -> pointer-to-const (ok); reverse (not ok)
  {
    int x{1};
    const int* pc = &x; // ok
    // int* pn = pc;    // error: cannot convert const int* to int*
    (void)pc;
    std::cout << "Ex4: non-const can be seen as const through pointer-to-const\n";
  }

  // Example 5: Functions respecting const-correctness
  {
    auto print = [](const int* p) { if (p) std::cout << "val=" << *p << '\n'; };
    auto increment = [](int* p) { if (p) ++*p; };
    int v{9};
    print(&v);     // read allowed
    increment(&v); // write allowed
    print(&v);
  }
}

int main() {
  concept1_pointer_and_reference();
  concept2_definitions();
  concept3_nullptr();
  concept4_decl_vs_init();
  concept5_basic_usage();
  concept6_address_and_deref();
  concept7_assigning_addresses();
  concept8_simple_deref();
  concept9_refs_vs_ptrs();
  concept10_alias_vs_pointer_object();
  concept11_when_to_use();
  concept12_pointer_arithmetic();
  concept13_iterating_with_pointers();
  concept14_const_correctness();
  return 0;
}