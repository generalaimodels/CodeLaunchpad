/*
    File: pointers_and_references_fundamentals.cpp

    Purpose:
      A deeply commented, example-driven C++ file that teaches the absolute fundamentals
      of pointers and references. Each concept below is covered by five focused examples.

      Concepts covered (5 examples each):
        1) What is a pointer & reference
        2) Definition of pointer (int* p), reference (int& r)
        3) Null pointers (nullptr in C++11)
        4) Pointer declaration vs initialization
        5) Basic usage
        6) Address-of (&) and dereference (*) operators
        7) Assigning addresses to pointers
        8) Simple pointer dereferencing
        9) References vs Pointers
        10) When to use reference vs pointer (syntax & semantics)
        11) Pointer arithmetic (basics)
        12) Iterating arrays with pointers
        13) Const correctness basics (const int* p, int* const p, const int* const p)

    Notes:
      - All explanatory text is provided as comments within this file.
      - Illegal or dangerous code is commented out with an explanation.
      - Program prints markers so you can correlate runtime output with examples.
      - Standard C++11 (or later) is assumed because we use nullptr.
*/

#include <iostream>
#include <iomanip> // for std::hex, std::dec
#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>
#include <string>

namespace util {
    // Helper to print titles and keep output structured.
    inline void title(const char* tag) {
        std::cout << "\n==== " << tag << " ====\n";
    }
    inline void ex(const char* tag) {
        std::cout << "[Example] " << tag << '\n';
    }
    inline const void* addr(const void* p) { return p; } // For uniform address printing.
}

/* =========================================================================================
   1) What is a pointer & reference
      - Pointer: an object that holds a memory address of another object or function.
      - Reference: an alias (second name) for an existing object; it is not a separate object.
      - Pointers can be reseated and can be null; references must be bound at initialization
        and cannot be reseated (and are conceptually never null).
   ========================================================================================= */
namespace concept1 {

    void example1_basic_int_pointer_and_reference() {
        util::ex("1.1 Basic int pointer and reference");
        int x = 42;

        int* p = &x;   // p holds the address of x
        int& r = x;    // r is another name (alias) for x

        std::cout << "x value: " << x << "\n";
        std::cout << "p points to: " << util::addr(p) << ", *p: " << *p << "\n";
        std::cout << "r refers to x, r: " << r << " (address of r is address of x): " << util::addr(&r) << "\n";

        *p = 100; // changes x
        std::cout << "After *p = 100, x: " << x << ", r: " << r << '\n';

        r = 200; // changes x
        std::cout << "After r = 200, x: " << x << ", *p: " << *p << '\n';
    }

    void example2_sizes_pointer_vs_object() {
        util::ex("1.2 Pointer size vs object size");
        double d = 3.14;
        double* pd = &d; // pointer to double
        std::cout << "sizeof(double): " << sizeof(double) << ", sizeof(double*): " << sizeof(double*) << '\n';
        std::cout << "Address of d: " << util::addr(&d) << ", *pd: " << *pd << '\n';
        // Pointer size is independent of the pointee type; depends on platform (e.g., 8 on 64-bit).
    }

    struct Point { int x; int y; };

    void example3_struct_pointer_and_reference() {
        util::ex("1.3 Pointer/reference to a struct");
        Point pt{10, 20};
        Point* pp = &pt;  // pointer to struct
        Point& rp = pt;   // reference to struct

        // Access via pointer: dereference then dot, or use arrow (->)
        (*pp).x = 11;
        pp->y = 21;

        // Access via reference: direct member access
        rp.x += 1;
        rp.y += 1;

        std::cout << "pt: {" << pt.x << ", " << pt.y << "} (address: " << util::addr(&pt) << ")\n";
    }

    // Function pointer demo to solidify "pointer" can also point to code.
    int add(int a, int b) { return a + b; }

    void example4_function_pointer() {
        util::ex("1.4 Function pointer");
        int (*fp)(int, int) = &add; // pointer to function taking (int,int) returning int
        // The & is optional on functions: int (*fp)(int,int) = add;
        int result1 = fp(2, 3);
        int result2 = (*fp)(5, 7); // dereferencing function pointer is optional
        std::cout << "fp points to 'add', add(2,3)=" << result1 << ", add(5,7)=" << result2 << '\n';
    }

    void example5_array_name_and_reference_alias() {
        util::ex("1.5 Array name decays to pointer; reference aliases");
        int arr[3] = {1, 2, 3};
        int* p = arr;           // array decays to pointer to first element (int*)
        int& first = arr[0];    // reference to first element is an alias
        std::cout << "arr: [" << arr[0] << ", " << arr[1] << ", " << arr[2] << "]\n";
        *p = 10;                 // changes arr[0]
        first = 20;              // changes arr[0] again
        std::cout << "after *p=10 and first=20 -> arr[0]=" << arr[0] << '\n';
    }

    void run() {
        util::title("1) What is a pointer & reference");
        example1_basic_int_pointer_and_reference();
        example2_sizes_pointer_vs_object();
        example3_struct_pointer_and_reference();
        example4_function_pointer();
        example5_array_name_and_reference_alias();
    }
}

/* =========================================================================================
   2) Definition of pointer (int* p), reference (int& r)
      - Syntax pitfalls: 'int* p, q;' makes p a pointer, q an int. Prefer 'int* p; int* q;'.
      - Pointer to pointer: int** pp.
      - Reference to pointer: int*& rp (reference that aliases a pointer variable).
      - Reference must be initialized; pointer may be default-initialized (danger if uninitialized).
   ========================================================================================= */
namespace concept2 {

    void example1_pointer_definition_syntax_trap() {
        util::ex("2.1 Definition syntax: placement of * binds to the declarator");
        int* a, b;       // a is int*, b is int (not a pointer) -> common pitfall
        a = nullptr;     // OK
        // b = nullptr;  // ERROR: nullptr is not an int; illustrates the trap (kept commented)
        b = 5;
        std::cout << "a is int*, value: " << a << " (nullptr), b is int: " << b << '\n';
        // Prefer: int* a; int* b; for clarity.
    }

    void example2_reference_definition() {
        util::ex("2.2 Reference definition and mandatory initialization");
        int x = 10;
        int& r = x;        // must bind to an existing object
        r = 12;            // changes x
        std::cout << "x: " << x << ", r: " << r << '\n';
        // int& r2;       // ERROR: references must be initialized (cannot compile)
    }

    void example3_pointer_to_pointer() {
        util::ex("2.3 Pointer to pointer (int**)");
        int v = 7;
        int* p = &v;     // pointer to int
        int** pp = &p;   // pointer to pointer to int
        **pp = 99;       // dereference twice to reach int
        std::cout << "v: " << v << " via **pp\n";
    }

    void example4_reference_to_pointer() {
        util::ex("2.4 Reference to pointer (int*&)");
        int x = 1, y = 2;
        int* p = &x;
        int*& rp = p;  // rp is a reference that aliases p itself (reseating affects p)
        rp = &y;       // changes p to point to y
        *p = 10;       // changes y now
        std::cout << "x: " << x << ", y: " << y << " (p now points to y)\n";
    }

    void example5_pointer_and_reference_const_mix() {
        util::ex("2.5 Reference to const and pointer to const");
        int x = 5;
        const int& rc = x;   // reference to const: cannot modify through rc
        const int* pc = &x;  // pointer to const: cannot modify through pc
        // rc = 6;           // ERROR: cannot assign to const reference target
        // *pc = 6;          // ERROR: cannot modify through pointer to const
        x = 6;               // modifying the original is still allowed
        std::cout << "x changed directly to: " << x << ", rc: " << rc << ", *pc: " << *pc << '\n';
    }

    void run() {
        util::title("2) Definition of pointer (int* p), reference (int& r)");
        example1_pointer_definition_syntax_trap();
        example2_reference_definition();
        example3_pointer_to_pointer();
        example4_reference_to_pointer();
        example5_pointer_and_reference_const_mix();
    }
}

/* =========================================================================================
   3) Null pointers (nullptr in C++11)
      - nullptr is a distinct null pointer constant (type: std::nullptr_t).
      - Safer than 0 or NULL because of type-safe overload resolution.
      - Never dereference a null pointer.
   ========================================================================================= */
namespace concept3 {

    void take_int(int)                    { std::cout << "take_int(int)\n"; }
    void take_ptr(int*)                   { std::cout << "take_ptr(int*)\n"; }
    void take_overload(std::nullptr_t)    { std::cout << "take_overload(nullptr_t)\n"; }
    void take_overload(int*)              { std::cout << "take_overload(int*)\n"; }

    void example1_init_with_nullptr() {
        util::ex("3.1 Initialize and compare with nullptr");
        int* p = nullptr; // explicit null pointer
        if (p == nullptr) {
            std::cout << "p is null\n";
        }
        // int x = *p;    // ERROR at runtime (would crash): never dereference nullptr (kept commented)
    }

    void example2_overload_resolution_with_nullptr() {
        util::ex("3.2 Overload resolution: nullptr selects pointer overload");
        take_int(0);     // picks take_int
        // take_ptr(0);  // ambiguous if both int and pointer overloads exist; here we only have take_ptr(int*)
        take_ptr(nullptr); // unambiguously pointer overload
    }

    void example3_nullptr_t_type() {
        util::ex("3.3 std::nullptr_t type");
        std::nullptr_t n = nullptr; // the exact type of nullptr
        int* p = n;                 // converts to any pointer type as a null pointer
        std::cout << "n assigned to int* p; p is " << (p == nullptr ? "null" : "non-null") << '\n';
    }

    int f_overload(int)        { return 1; }
    int f_overload(int*)       { return 2; }
    int f_overload(std::nullptr_t) { return 3; }

    void example4_overload_with_zero_NULL_nullptr() {
        util::ex("3.4 0 vs NULL vs nullptr (prefer nullptr)");
        // Note: NULL may be defined as 0 or 0L depending on platform.
        // Using 0 can cause unintended int overload selection.
        std::cout << "f_overload(0) = " << f_overload(0) << " (int overload)\n";
        std::cout << "f_overload(nullptr) = " << f_overload(nullptr) << " (nullptr_t overload)\n";
        // f_overload(NULL) -> may be ambiguous in some contexts; platform-dependent; avoid.
    }

    void example5_safe_null_checks() {
        util::ex("3.5 Safe null checks before dereference");
        int value = 10;
        int* p = &value;
        if (p) { // idiomatic: pointer in boolean context compares against null
            *p += 5;
            std::cout << "*p after safe access: " << *p << '\n';
        }
        p = nullptr;
        if (!p) {
            std::cout << "p is null, skipping dereference\n";
        }
    }

    void run() {
        util::title("3) Null pointers (nullptr in C++11)");
        example1_init_with_nullptr();
        example2_overload_resolution_with_nullptr();
        example3_nullptr_t_type();
        example4_overload_with_zero_NULL_nullptr();
        example5_safe_null_checks();
    }
}

/* =========================================================================================
   4) Pointer declaration vs initialization
      - Declaration: introducing a name with type.
      - Initialization: giving the pointer an initial value (address or nullptr).
      - Never use an uninitialized pointer; initialize to nullptr if you don't have an address yet.
   ========================================================================================= */
namespace concept4 {

    void example1_uninitialized_danger() {
        util::ex("4.1 Uninitialized pointer is dangerous (UB) — always initialize");
        int* p; // uninitialized; indeterminate value
        // std::cout << *p; // Undefined Behavior if used; DO NOT DO THIS (commented for safety)
        std::cout << "Always initialize pointers to a known value (e.g., nullptr)\n";
    }

    void example2_initialize_to_nullptr() {
        util::ex("4.2 Initialize to nullptr when no target exists yet");
        int* p = nullptr;
        if (p == nullptr) {
            std::cout << "p is safely null\n";
        }
    }

    void example3_initialize_with_address_later() {
        util::ex("4.3 Declare, then later initialize with a valid address");
        int* p = nullptr;        // safe initial state
        int x = 42;
        p = &x;                  // now initialized with a valid address
        std::cout << "*p: " << *p << " (points to x)\n";
    }

    void example4_dynamic_allocation_then_init() {
        util::ex("4.4 Initialize via dynamic allocation, then delete");
        int* p = new int(123);   // p initialized to the address of a dynamically allocated int
        std::cout << "*p: " << *p << '\n';
        delete p;                // release memory
        p = nullptr;             // avoid dangling pointer
    }

    void example5_const_pointer_requires_init() {
        util::ex("4.5 const pointer (int* const) requires initialization at declaration");
        int x = 10, y = 20;
        int* const cp = &x;  // const pointer (cannot reseat), must initialize here
        *cp = 11;            // allowed: we can modify the pointee
        // cp = &y;          // ERROR: cannot reseat const pointer
        std::cout << "x: " << x << " (cp still points to x)\n";
    }

    void run() {
        util::title("4) Pointer declaration vs initialization");
        example1_uninitialized_danger();
        example2_initialize_to_nullptr();
        example3_initialize_with_address_later();
        example4_dynamic_allocation_then_init();
        example5_const_pointer_requires_init();
    }
}

/* =========================================================================================
   5) Basic usage
      - Obtain addresses with & (address-of).
      - Dereference with * to access the object pointed to.
      - References act as an alias: use them as if they were the original object.
   ========================================================================================= */
namespace concept5 {

    void example1_get_address_and_deref() {
        util::ex("5.1 & (address-of) and * (dereference)");
        int x = 5;
        int* p = &x;     // address-of
        *p = 6;          // dereference to assign
        std::cout << "x after *p=6: " << x << '\n';
    }

    void example2_reference_modification() {
        util::ex("5.2 Reference is an alias");
        int x = 10;
        int& r = x;  // alias to x
        r += 7;      // modifies x
        std::cout << "x: " << x << " (modified via reference)\n";
    }

    void example3_pointer_reseating() {
        util::ex("5.3 Pointers can reseat to another object");
        int a = 1, b = 2;
        int* p = &a;
        *p = 10;      // changes a
        p = &b;       // reseat
        *p = 20;      // changes b
        std::cout << "a: " << a << ", b: " << b << " (p reseated)\n";
    }

    void example4_pointer_to_pointer_basic() {
        util::ex("5.4 Pointer to pointer");
        int n = 3;
        int* p = &n;
        int** pp = &p;
        **pp = 30;
        std::cout << "n: " << n << " via **pp\n";
    }

    void example5_ref_binding_to_const_temporary() {
        util::ex("5.5 const reference can bind to temporary");
        int x = 2;
        const int& rc = x + 3; // binds to temporary (lifetime extended for rc)
        std::cout << "rc bound to temporary value: " << rc << '\n';
    }

    void run() {
        util::title("5) Basic usage");
        example1_get_address_and_deref();
        example2_reference_modification();
        example3_pointer_reseating();
        example4_pointer_to_pointer_basic();
        example5_ref_binding_to_const_temporary();
    }
}

/* =========================================================================================
   6) Address-of (&) and dereference (*) operators
      - & yields address of an object (with nuances for arrays/functions).
      - * yields the referenced object from a pointer.
      - Use -> for accessing members through a pointer to a struct/class.
   ========================================================================================= */
namespace concept6 {

    void example1_address_of_stack_var() {
        util::ex("6.1 Address-of a stack variable");
        int x = 42;
        int* p = &x;
        std::cout << "Address of x: " << util::addr(&x) << ", p: " << util::addr(p) << ", *p: " << *p << '\n';
    }

    void example2_address_of_array_vs_decay() {
        util::ex("6.2 Array decay vs address of whole array");
        int arr[4] = {1,2,3,4};
        int* p1 = arr;          // decay to pointer to first element (int*)
        int (*p2)[4] = &arr;    // pointer to the whole array (type: int (*)[4])
        std::cout << "p1 (int*): " << util::addr(p1) << ", *p1: " << *p1 << '\n';
        std::cout << "p2 (int (*)[4]): " << util::addr(p2) << ", (*p2)[0]: " << (*p2)[0] << '\n';
    }

    struct Node { int value; };

    void example3_deref_and_member_access() {
        util::ex("6.3 (*p).member vs p->member");
        Node n{7};
        Node* p = &n;
        (*p).value = 8;   // dereference then dot
        p->value += 1;    // arrow for convenience
        std::cout << "n.value: " << n.value << '\n';
    }

    void example4_deref_as_lvalue() {
        util::ex("6.4 Dereference yields an lvalue (assignable if not const)");
        int x = 1;
        int* p = &x;
        *p = 2;  // assign through pointer
        std::cout << "x after *p=2: " << x << '\n';
    }

    void example5_deref_pointer_to_const() {
        util::ex("6.5 Dereference pointer-to-const yields const lvalue");
        int x = 10;
        const int* p = &x;
        std::cout << "*p: " << *p << '\n';
        // *p = 11; // ERROR: cannot modify through pointer-to-const
        x = 11;     // changing underlying object is fine
        std::cout << "x after direct change: " << x << ", *p: " << *p << '\n';
    }

    void run() {
        util::title("6) Address-of (&) and dereference (*) operators");
        example1_address_of_stack_var();
        example2_address_of_array_vs_decay();
        example3_deref_and_member_access();
        example4_deref_as_lvalue();
        example5_deref_pointer_to_const();
    }
}

/* =========================================================================================
   7) Assigning addresses to pointers
      - Pointers hold addresses of compatible types.
      - You can reseat pointers; ensure types match or use appropriate casts (with caution).
      - void* can hold the address of any object type (but must cast back to use).
   ========================================================================================= */
namespace concept7 {

    void example1_assign_address() {
        util::ex("7.1 Assigning an address to a pointer");
        int x = 9;
        int* p = nullptr;
        p = &x; // assign address
        std::cout << "p: " << util::addr(p) << ", *p: " << *p << '\n';
    }

    void example2_reassign_to_different_variable() {
        util::ex("7.2 Reseat pointer to a different variable");
        int a = 1, b = 2;
        int* p = &a;
        p = &b; // reseat
        *p = 5; // modifies b
        std::cout << "a: " << a << ", b: " << b << '\n';
    }

    void example3_void_pointer_roundtrip() {
        util::ex("7.3 void* can store any object address (must cast back)");
        double d = 3.5;
        void* vp = &d; // store address in void*
        // To use the value, cast back to the correct type:
        double* pd = static_cast<double*>(vp);
        *pd = 4.5;
        std::cout << "d after roundtrip: " << d << '\n';
    }

    void example4_pointer_to_const_assignment() {
        util::ex("7.4 Assign non-const address to pointer-to-const");
        int x = 10;
        const int* pc = &x; // ok: converting int* to const int*
        std::cout << "*pc: " << *pc << '\n';
        // *pc = 11;     // ERROR: can't modify through pointer-to-const
    }

    void example5_incompatible_types_need_cast() {
        util::ex("7.5 Incompatible pointer types require explicit cast (avoid if possible)");
        int x = 5;
        // double* pd = &x;            // ERROR: incompatible types
        double* pd = reinterpret_cast<double*>(&x); // Dangerous; reinterpret cast breaks type rules
        // Using pd is undefined behavior if we try to read as double; avoid in real code.
        (void)pd; // suppress unused warning
        std::cout << "Avoid reinterpreting unrelated types via pointers (shown only as a caution)\n";
    }

    void run() {
        util::title("7) Assigning addresses to pointers");
        example1_assign_address();
        example2_reassign_to_different_variable();
        example3_void_pointer_roundtrip();
        example4_pointer_to_const_assignment();
        example5_incompatible_types_need_cast();
    }
}

/* =========================================================================================
   8) Simple pointer dereferencing
      - Dereferencing reads/writes the pointed-to object.
      - Ensure the pointer is non-null and points to a valid object.
   ========================================================================================= */
namespace concept8 {

    void example1_read_via_pointer() {
        util::ex("8.1 Read via pointer");
        int x = 100;
        int* p = &x;
        std::cout << "*p: " << *p << '\n';
    }

    void example2_write_via_pointer() {
        util::ex("8.2 Write via pointer");
        int x = 0;
        int* p = &x;
        *p = 77;
        std::cout << "x: " << x << '\n';
    }

    void example3_double_deref() {
        util::ex("8.3 Double dereference (int**)");
        int x = 5;
        int* p = &x;
        int** pp = &p;
        **pp = 55;
        std::cout << "x: " << x << '\n';
    }

    void example4_array_first_element_deref() {
        util::ex("8.4 *p on an array pointer reads first element");
        int a[3] = {9, 8, 7};
        int* p = a; // points to a[0]
        std::cout << "*p: " << *p << " (equals a[0])\n";
    }

    void example5_deref_after_pointer_arithmetic() {
        util::ex("8.5 Deref after pointer arithmetic (within the same array)");
        int a[4] = {1,2,3,4};
        int* p = a;   // a[0]
        std::cout << "*(p+2): " << *(p + 2) << " (equals a[2])\n";
    }

    void run() {
        util::title("8) Simple pointer dereferencing");
        example1_read_via_pointer();
        example2_write_via_pointer();
        example3_double_deref();
        example4_array_first_element_deref();
        example5_deref_after_pointer_arithmetic();
    }
}

/* =========================================================================================
   9) References vs Pointers
      - Reference is an alias: must bind at init, cannot be null (conceptually), cannot reseat.
      - Pointer is an object: can be reseated, can be null, requires explicit dereference to access.
   ========================================================================================= */
namespace concept9 {

    void example1_reseating_vs_alias() {
        util::ex("9.1 Pointer can reseat; reference cannot");
        int a = 1, b = 2;
        int* p = &a;
        int& r = a;
        p = &b;       // reseat pointer
        // &r = b;    // ERROR: cannot reseat a reference (address-of expression not assignable)
        r = 99;       // assigns to 'a', not reseat
        std::cout << "a: " << a << ", b: " << b << " (r still aliases a; p points to b)\n";
    }

    void example2_nullability() {
        util::ex("9.2 Pointer may be null; reference cannot be null");
        int* p = nullptr;   // ok
        if (!p) std::cout << "p is null\n";
        int x = 5;
        int& r = x;         // must bind to existing object
        std::cout << "r bound to x: " << r << '\n';
        // int& bad = *static_cast<int*>(nullptr); // Never do this: would be UB at runtime (commented)
    }

    void example3_pass_by_reference_vs_pointer() {
        util::ex("9.3 Pass-by-reference vs pass-by-pointer");
        auto increment_ref = [](int& v) { v += 1; };       // caller passes variable, non-null, clear intent
        auto increment_ptr = [](int* v) { if (v) *v += 1; }; // caller may pass nullptr; must check
        int x = 10, y = 20;
        increment_ref(x);            // cannot be null
        increment_ptr(&y);           // pass address
        increment_ptr(nullptr);      // allowed, does nothing
        std::cout << "x: " << x << ", y: " << y << '\n';
    }

    void example4_const_ref_binds_temporary() {
        util::ex("9.4 const reference binds temporary (pointer cannot point to temporary literal)");
        const int& r = 40 + 2; // ok: lifetime extended for r
        std::cout << "r: " << r << '\n';
        // int* p = &(40 + 2); // ERROR: cannot take address of temporary literal
        (void)r;
    }

    void example5_addresses_and_identity() {
        util::ex("9.5 Address of reference is the address of its referent");
        int x = 7;
        int* p = &x;
        int& r = x;
        std::cout << "Address of x: " << util::addr(&x) << ", p: " << util::addr(p) << ", &r: " << util::addr(&r) << '\n';
        // Note: &r == &x, while 'p' has its own address as a separate object variable.
    }

    void run() {
        util::title("9) References vs Pointers");
        example1_reseating_vs_alias();
        example2_nullability();
        example3_pass_by_reference_vs_pointer();
        example4_const_ref_binds_temporary();
        example5_addresses_and_identity();
    }
}

/* =========================================================================================
   10) When to use reference vs pointer (syntax & semantics)
       - Prefer references for "must-have" parameters (non-null), output params, operator overloads.
       - Use pointers when null is meaningful (optional), reseating needed, or interop with C APIs.
   ========================================================================================= */
namespace concept10 {

    // Example helper functions
    void scale_inplace(double& value, double factor) { value *= factor; } // must-have (non-null)
    void try_scale(double* value, double factor) { if (value) *value *= factor; } // optional

    int& choose_bigger_ref(int& a, int& b) { return (a > b) ? a : b; } // returns alias to caller
    int* find_even_ptr(int* begin, int* end) { // nullptr return means "not found"
        for (int* p = begin; p != end; ++p) if (*p % 2 == 0) return p;
        return nullptr;
    }

    void example1_mandatory_inout_use_reference() {
        util::ex("10.1 Mandatory in/out: use references");
        double v = 10.0;
        scale_inplace(v, 1.5);
        std::cout << "v after scale_inplace: " << v << '\n';
    }

    void example2_optional_use_pointer() {
        util::ex("10.2 Optional argument: use pointer to allow null");
        double v = 8.0;
        try_scale(&v, 2.0);    // scaled
        try_scale(nullptr, 2.0); // safely ignored
        std::cout << "v after try_scale: " << v << '\n';
    }

    void example3_return_reference_vs_pointer() {
        util::ex("10.3 Return reference to existing object vs pointer for optional");
        int a = 3, b = 9;
        int& big = choose_bigger_ref(a, b);
        big = -1; // modifies the bigger one directly via alias
        std::cout << "a: " << a << ", b: " << b << " (b was bigger, now -1)\n";

        int arr[] = {1,3,5,7};
        if (int* p = find_even_ptr(std::begin(arr), std::end(arr))) {
            std::cout << "found even: " << *p << '\n';
        } else {
            std::cout << "no even found (nullptr)\n";
        }
    }

    struct Box { int v; };
    std::ostream& operator<<(std::ostream& os, const Box& b) { return os << "Box(" << b.v << ")"; }

    void example4_operator_overloads_use_refs() {
        util::ex("10.4 Operator overloads typically use references");
        Box b{5};
        std::cout << b << '\n'; // operator<< takes const Box& usually
    }

    // Simulated C-style API
    void c_api_get_value(int* out) { if (out) *out = 123; }

    void example5_c_api_uses_pointers() {
        util::ex("10.5 Interop: C-style APIs use pointers for outputs");
        int value = 0;
        c_api_get_value(&value);
        std::cout << "value from C API: " << value << '\n';
    }

    void run() {
        util::title("10) When to use reference vs pointer (syntax & semantics)");
        example1_mandatory_inout_use_reference();
        example2_optional_use_pointer();
        example3_return_reference_vs_pointer();
        example4_operator_overloads_use_refs();
        example5_c_api_uses_pointers();
    }
}

/* =========================================================================================
   11) Pointer arithmetic (basics)
       - Adding/subtracting integers moves by multiples of sizeof(T) for T*.
       - Only valid within the same array object (or one past the end).
       - p2 - p1 yields the number of elements between pointers into the same array.
   ========================================================================================= */
namespace concept11 {

    void example1_int_pointer_arithmetic() {
        util::ex("11.1 int* pointer arithmetic moves by sizeof(int)");
        int a[4] = {10,20,30,40};
        int* p = a;          // a[0]
        int* q = p + 2;      // points to a[2]
        std::cout << "*q: " << *q << " (should be 30)\n";
    }

    void example2_char_pointer_arithmetic() {
        util::ex("11.2 char* moves by 1 byte");
        char s[] = "ABCD";
        char* p = s;         // s[0] = 'A'
        std::cout << "*p: " << *p << ", *(p+1): " << *(p+1) << '\n';
    }

    void example3_pointer_difference() {
        util::ex("11.3 Pointer difference (same array)");
        int a[] = {1,2,3,4,5};
        int* p = &a[1]; // a[1]
        int* q = &a[4]; // a[4]
        std::ptrdiff_t diff = q - p; // number of int elements between them
        std::cout << "q - p = " << diff << " (should be 3)\n";
    }

    void example4_end_pointer_one_past() {
        util::ex("11.4 End pointer is one-past-last; do not dereference end");
        int a[] = {1,2,3};
        int* end = a + 3;     // one past last
        int* it = a;
        while (it != end) {
            std::cout << *it << ' ';
            ++it;             // safe
        }
        std::cout << '\n';
        // *end; // ERROR: dereferencing end is undefined behavior (commented)
    }

    void example5_struct_pointer_arithmetic_alignment() {
        util::ex("11.5 Pointer arithmetic respects type size and alignment");
        struct Pair { int x; int y; };
        Pair arr[3] = {{1,2},{3,4},{5,6}};
        Pair* p = arr;
        Pair* p2 = p + 2; // jumps sizeof(Pair) * 2
        std::cout << "p2->x: " << p2->x << ", p2->y: " << p2->y << '\n';
    }

    void run() {
        util::title("11) Pointer arithmetic (basics)");
        example1_int_pointer_arithmetic();
        example2_char_pointer_arithmetic();
        example3_pointer_difference();
        example4_end_pointer_one_past();
        example5_struct_pointer_arithmetic_alignment();
    }
}

/* =========================================================================================
   12) Iterating arrays with pointers
       - Use begin and end pointers (p, e) and loop while p != e.
       - Works for raw arrays and dynamically allocated arrays.
       - Pointer to array needed for true 2D arrays (T (*)[N]).
   ========================================================================================= */
namespace concept12 {

    void example1_iterate_forward() {
        util::ex("12.1 Iterate int array forward");
        int a[] = {1,2,3,4};
        for (int* p = a; p != a + 4; ++p) {
            std::cout << *p << ' ';
        }
        std::cout << '\n';
    }

    void example2_iterate_reverse() {
        util::ex("12.2 Iterate int array in reverse");
        int a[] = {1,2,3,4};
        for (int* p = a + 4; p != a; ) {
            --p;
            std::cout << *p << ' ';
        }
        std::cout << '\n';
    }

    void example3_iterate_with_const_pointer_to_const_data() {
        util::ex("12.3 Iterate with const int* (read-only)");
        const int a[] = {5,6,7};
        for (const int* p = a; p != a + 3; ++p) {
            std::cout << *p << ' ';
            // *p = 0; // ERROR: cannot modify through pointer-to-const
        }
        std::cout << '\n';
    }

    void example4_iterate_2d_array_with_pointer_to_array() {
        util::ex("12.4 Iterate 2D array using pointer-to-array");
        int m[2][3] = {{1,2,3},{4,5,6}};
        int (*row)[3] = m;  // pointer to array of 3 ints
        for (int i = 0; i < 2; ++i) {
            for (int* p = row[i]; p != row[i] + 3; ++p) {
                std::cout << *p << ' ';
            }
            std::cout << '\n';
        }
    }

    void example5_iterate_dynamic_array() {
        util::ex("12.5 Iterate dynamic array with pointers");
        std::size_t n = 5;
        int* a = new int[n]{1,2,3,4,5};
        for (int* p = a; p != a + n; ++p) {
            std::cout << *p << ' ';
        }
        std::cout << '\n';
        delete[] a; // clean up
    }

    void run() {
        util::title("12) Iterating arrays with pointers");
        example1_iterate_forward();
        example2_iterate_reverse();
        example3_iterate_with_const_pointer_to_const_data();
        example4_iterate_2d_array_with_pointer_to_array();
        example5_iterate_dynamic_array();
    }
}

/* =========================================================================================
   13) Const correctness basics
       - const int*       : pointer to const int (data is const through the pointer; pointer can reseat).
       - int* const       : const pointer to int (pointer cannot reseat; data is mutable).
       - const int* const : const pointer to const int (neither reseat nor modify data).
       - Important for conveying intent and enabling compiler-enforced safety.
   ========================================================================================= */
namespace concept13 {

    void example1_pointer_to_const() {
        util::ex("13.1 const int* (pointer to const data)");
        int x = 10;
        const int* p = &x; // cannot modify x through p
        std::cout << "*p: " << *p << '\n';
        // *p = 11; // ERROR
        x = 11;     // allowed
        p = &x;     // reseat allowed
        std::cout << "x: " << x << ", *p: " << *p << '\n';
    }

    void example2_const_pointer() {
        util::ex("13.2 int* const (const pointer)");
        int x = 1, y = 2;
        int* const p = &x; // must initialize; cannot reseat
        *p = 10;           // allowed: data mutable
        // p = &y;         // ERROR: cannot change where p points
        std::cout << "x: " << x << '\n';
    }

    void example3_const_pointer_to_const() {
        util::ex("13.3 const int* const (const pointer to const data)");
        int x = 5;
        const int* const p = &x;
        std::cout << "*p: " << *p << '\n';
        // *p = 6;  // ERROR: cannot modify through p
        // p = &x;  // ERROR: cannot reseat
    }

    void example4_function_param_constness() {
        util::ex("13.4 Function parameters with pointer constness");
        auto sum = [](const int* begin, const int* end) {
            int s = 0;
            for (auto p = begin; p != end; ++p) {
                s += *p;      // read-only access
                // *p = 0;    // ERROR if attempted
            }
            return s;
        };
        int a[] = {1,2,3};
        std::cout << "sum: " << sum(a, a + 3) << '\n';
    }

    void example5_casting_away_const_is_unsafe() {
        util::ex("13.5 Casting away const is unsafe (UB if object is truly const)");
        const int x = 42;
        const int* pc = &x;
        // int* p = const_cast<int*>(pc);
        // *p = 7; // Undefined Behavior if x is const; do not do this. Safe only if original object was non-const.
        std::cout << "Do not cast away const to modify a const object\n";
    }

    void run() {
        util::title("13) Const correctness basics");
        example1_pointer_to_const();
        example2_const_pointer();
        example3_const_pointer_to_const();
        example4_function_param_constness();
        example5_casting_away_const_is_unsafe();
    }
}

int main() {
    // Run all concept sections in order. Each section prints a header and per-example markers.
    concept1::run();
    concept2::run();
    concept3::run();
    concept4::run();
    concept5::run();
    concept6::run();
    concept7::run();
    concept8::run();
    concept9::run();
    concept10::run();
    concept11::run();
    concept12::run();
    concept13::run();

    std::cout << "\nAll examples completed.\n";
    return 0;
}