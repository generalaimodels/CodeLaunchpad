#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <functional>
#include <string>
using namespace std;

/*
This single C++ source file contains 100 self‐contained examples that demonstrate
all major concepts and possible usages of the <vector> library.
Each example is implemented as a static method inside the VectorExamples class.
Run the program to see console output for each example.
*/

class VectorExamples {
public:
    // 01. Default constructor: create an empty vector.
    static void example01() {
        vector<int> v;
        cout << "example01: default constructor, size = " << v.size() << "\n";
    }
    // 02. Fill constructor: create a vector with 5 copies of value 10.
    static void example02() {
        vector<int> v(5, 10);
        cout << "example02: fill constructor, size = " << v.size() << "\n";
    }
    // 03. Range constructor: create vector from C-array.
    static void example03() {
        int arr[] = {1, 2, 3, 4, 5};
        vector<int> v(begin(arr), end(arr));
        cout << "example03: range constructor, first = " << v.front() << "\n";
    }
    // 04. Copy constructor.
    static void example04() {
        vector<int> v1(3, 7);
        vector<int> v2(v1);
        cout << "example04: copy constructor, v2[0] = " << v2[0] << "\n";
    }
    // 05. Move constructor.
    static void example05() {
        vector<int> v1(4, 8);
        vector<int> v2(move(v1));
        cout << "example05: move constructor, v2 size = " << v2.size() << ", v1 size = " << v1.size() << "\n";
    }
    // 06. operator[] for element access.
    static void example06() {
        vector<int> v = {10, 20, 30};
        cout << "example06: operator[], v[1] = " << v[1] << "\n";
    }
    // 07. at() with bounds checking.
    static void example07() {
        vector<int> v = {100, 200, 300};
        cout << "example07: at(), v.at(2) = " << v.at(2) << "\n";
    }
    // 08. front() and back() access.
    static void example08() {
        vector<int> v = {5, 10, 15};
        cout << "example08: front() = " << v.front() << ", back() = " << v.back() << "\n";
    }
    // 09. data() returns pointer to underlying array.
    static void example09() {
        vector<int> v = {1, 2, 3};
        cout << "example09: data(), *data() = " << *(v.data()) << "\n";
    }
    // 10. size() and capacity().
    static void example10() {
        vector<int> v(5, 1);
        cout << "example10: size = " << v.size() << ", capacity = " << v.capacity() << "\n";
    }
    // 11. reserve() to preallocate memory.
    static void example11() {
        vector<int> v;
        v.reserve(50);
        cout << "example11: after reserve(50), capacity = " << v.capacity() << "\n";
    }
    // 12. shrink_to_fit() to reduce excess capacity.
    static void example12() {
        vector<int> v(100, 2);
        v.resize(10);
        v.shrink_to_fit();
        cout << "example12: after shrink_to_fit, size = " << v.size() << ", capacity = " << v.capacity() << "\n";
    }
    // 13. empty() to check if vector is empty.
    static void example13() {
        vector<int> v;
        cout << "example13: empty() = " << boolalpha << v.empty() << "\n";
    }
    // 14. push_back() to add an element.
    static void example14() {
        vector<int> v;
        v.push_back(42);
        cout << "example14: push_back, size = " << v.size() << "\n";
    }
    // 15. pop_back() to remove the last element.
    static void example15() {
        vector<int> v = {1, 2, 3};
        v.pop_back();
        cout << "example15: pop_back, size = " << v.size() << "\n";
    }
    // 16. insert() a single element.
    static void example16() {
        vector<int> v = {1, 2, 3};
        v.insert(v.begin() + 1, 99);
        cout << "example16: insert at index 1, element = " << v[1] << "\n";
    }
    // 17. insert() multiple elements.
    static void example17() {
        vector<int> v = {1, 2, 3};
        v.insert(v.begin(), 3, 7); // insert three 7's at beginning.
        cout << "example17: insert multiple, first element = " << v.front() << "\n";
    }
    // 18. erase() a single element.
    static void example18() {
        vector<int> v = {10, 20, 30, 40};
        v.erase(v.begin() + 2);
        cout << "example18: erase index 2, new element at index 2 = " << v[2] << "\n";
    }
    // 19. erase() a range of elements.
    static void example19() {
        vector<int> v = {1, 2, 3, 4, 5};
        v.erase(v.begin() + 1, v.begin() + 3);
        cout << "example19: erase range, size = " << v.size() << "\n";
    }
    // 20. clear() to remove all elements.
    static void example20() {
        vector<int> v = {1, 2, 3};
        v.clear();
        cout << "example20: clear, empty() = " << boolalpha << v.empty() << "\n";
    }
    // 21. assign() with count and value.
    static void example21() {
        vector<int> v;
        v.assign(5, 3);
        cout << "example21: assign 5 elements of 3, size = " << v.size() << "\n";
    }
    // 22. assign() with a range.
    static void example22() {
        vector<int> source = {1, 2, 3, 4, 5};
        vector<int> v;
        v.assign(source.begin() + 1, source.end() - 1);
        cout << "example22: assign range, first element = " << v.front() << "\n";
    }
    // 23. emplace() to construct element in place.
    static void example23() {
        vector<pair<int, int>> v;
        v.emplace(v.begin(), 1, 2);
        cout << "example23: emplace, first pair first = " << v[0].first << "\n";
    }
    // 24. emplace_back() to construct element at the end.
    static void example24() {
        vector<string> v;
        v.emplace_back("hello");
        cout << "example24: emplace_back, element = " << v.back() << "\n";
    }
    // 25. swap() member function.
    static void example25() {
        vector<int> v1 = {1, 2, 3}, v2 = {4, 5};
        v1.swap(v2);
        cout << "example25: after swap, v1 size = " << v1.size() << ", v2 size = " << v2.size() << "\n";
    }
    // 26. Non-member swap.
    static void example26() {
        vector<int> v1 = {1, 2}, v2 = {3, 4, 5};
        swap(v1, v2);
        cout << "example26: non-member swap, v1 front = " << v1.front() << "\n";
    }
    // 27. Forward iteration using iterators.
    static void example27() {
        vector<int> v = {10, 20, 30};
        cout << "example27: iterators forward: ";
        for (auto it = v.begin(); it != v.end(); ++it)
            cout << *it << " ";
        cout << "\n";
    }
    // 28. Reverse iteration using reverse_iterator.
    static void example28() {
        vector<int> v = {10, 20, 30};
        cout << "example28: iterators reverse: ";
        for (auto rit = v.rbegin(); rit != v.rend(); ++rit)
            cout << *rit << " ";
        cout << "\n";
    }
    // 29. Using const_iterator.
    static void example29() {
        vector<int> v = {5, 6, 7};
        cout << "example29: const_iterator: ";
        for (auto cit = v.cbegin(); cit != v.cend(); ++cit)
            cout << *cit << " ";
        cout << "\n";
    }
    // 30. Iterator arithmetic.
    static void example30() {
        vector<int> v = {1, 2, 3, 4, 5};
        auto it = v.begin();
        it += 2;
        cout << "example30: iterator arithmetic, *it = " << *it << "\n";
    }
    // 31. Reverse iterator arithmetic.
    static void example31() {
        vector<int> v = {1, 2, 3, 4, 5};
        auto rit = v.rbegin();
        rit += 2;
        cout << "example31: reverse_iterator arithmetic, *rit = " << *rit << "\n";
    }
    // 32. Using std::for_each with vector iterators.
    static void example32() {
        vector<int> v = {1, 2, 3};
        cout << "example32: for_each: ";
        for_each(v.begin(), v.end(), [](int n) { cout << n << " "; });
        cout << "\n";
    }
    // 33. Using std::find algorithm.
    static void example33() {
        vector<int> v = {10, 20, 30};
        auto it = find(v.begin(), v.end(), 20);
        cout << "example33: find, found = " << (it != v.end() ? *it : -1) << "\n";
    }
    // 34. Using std::count to count occurrences.
    static void example34() {
        vector<int> v = {1, 2, 2, 3, 2};
        int cnt = count(v.begin(), v.end(), 2);
        cout << "example34: count, 2 appears " << cnt << " times\n";
    }
    // 35. Using std::accumulate to sum elements.
    static void example35() {
        vector<int> v = {1, 2, 3, 4};
        int sum = accumulate(v.begin(), v.end(), 0);
        cout << "example35: accumulate, sum = " << sum << "\n";
    }
    // 36. Vector of vectors (2D vector).
    static void example36() {
        vector<vector<int>> vv = {{1, 2}, {3, 4}};
        cout << "example36: 2D vector, first element = " << vv[0][0] << "\n";
    }
    // 37. Vector of strings.
    static void example37() {
        vector<string> v = {"a", "b", "c"};
        cout << "example37: vector of strings, first = " << v.front() << "\n";
    }
    // 38. Initializer list construction.
    static void example38() {
        vector<double> v = {3.14, 2.71, 1.41};
        cout << "example38: initializer list, size = " << v.size() << "\n";
    }
    // 39. Using vector with custom objects.
    static void example39() {
        struct Point { int x, y; };
        vector<Point> v = { {1, 2}, {3, 4} };
        cout << "example39: custom object, first point = (" << v[0].x << ", " << v[0].y << ")\n";
    }
    // 40. Demonstrating vector reallocation.
    static void example40() {
        vector<int> v;
        size_t oldCapacity = v.capacity();
        for (int i = 0; i < 100; ++i) v.push_back(i);
        cout << "example40: reallocation, old capacity = " << oldCapacity << ", new capacity = " << v.capacity() << "\n";
    }
    // 41. Exception safety with at() method.
    static void example41() {
        vector<int> v = {1, 2, 3};
        try {
            int val = v.at(10);
            cout << "example41: value = " << val << "\n";
        } catch (const out_of_range& e) {
            cout << "example41: caught exception: " << e.what() << "\n";
        }
    }
    // 42. Using swap() for efficiency.
    static void example42() {
        vector<int> v1 = {1, 2, 3}, v2 = {4, 5, 6};
        v1.swap(v2);
        cout << "example42: after swap, v1 front = " << v1.front() << "\n";
    }
    // 43. Iterator invalidation demonstration.
    static void example43() {
        vector<int> v = {1, 2, 3, 4};
        auto it = v.begin();
        v.push_back(5); // may invalidate iterators
        cout << "example43: potential iterator invalidation, *it = " << *it << "\n";
    }
    // 44. Reserve and push_back impact.
    static void example44() {
        vector<int> v;
        v.reserve(10);
        for (int i = 0; i < 10; i++) v.push_back(i);
        cout << "example44: after reserve and push_back, size = " << v.size() << "\n";
    }
    // 45. Observing capacity growth strategy.
    static void example45() {
        vector<int> v;
        for (int i = 0; i < 20; i++) {
            v.push_back(i);
            cout << "example45: size = " << v.size() << ", capacity = " << v.capacity() << "\n";
        }
    }
    // 46. assign() from another vector’s range.
    static void example46() {
        vector<int> source = {10, 20, 30, 40};
        vector<int> v;
        v.assign(source.begin() + 1, source.end());
        cout << "example46: assign range, first element = " << v.front() << "\n";
    }
    // 47. emplace() with move semantics.
    static void example47() {
        vector<string> v;
        string s = "test";
        v.emplace(v.begin(), move(s));
        cout << "example47: emplace with move, element = " << v.front() << "\n";
    }
    // 48. insert() using initializer list.
    static void example48() {
        vector<int> v = {1, 2, 3};
        v.insert(v.end(), {4, 5, 6});
        cout << "example48: insert initializer list, size = " << v.size() << "\n";
    }
    // 49. clear() then push_back().
    static void example49() {
        vector<int> v = {1, 2, 3};
        v.clear();
        v.push_back(99);
        cout << "example49: clear then push_back, front = " << v.front() << "\n";
    }
    // 50. Storing pointers in a vector.
    static void example50() {
        int a = 10, b = 20;
        vector<int*> v;
        v.push_back(&a);
        v.push_back(&b);
        cout << "example50: vector of pointers, *v[0] = " << *v[0] << "\n";
    }
    // 51. Dynamic memory simulation with push_back.
    static void example51() {
        vector<int> v;
        for (int i = 0; i < 5; i++) v.push_back(i * i);
        cout << "example51: dynamic push_back, element[3] = " << v[3] << "\n";
    }
    // 52. resize() to increase vector size.
    static void example52() {
        vector<int> v = {1, 2, 3, 4};
        v.resize(6, 0);
        cout << "example52: resize increase, new size = " << v.size() << "\n";
    }
    // 53. resize() to decrease vector size.
    static void example53() {
        vector<int> v = {1, 2, 3, 4, 5};
        v.resize(3);
        cout << "example53: resize decrease, new size = " << v.size() << "\n";
    }
    // 54. reserve() then resize().
    static void example54() {
        vector<int> v;
        v.reserve(100);
        v.resize(10, -1);
        cout << "example54: reserve then resize, size = " << v.size() << "\n";
    }
    // 55. Sorting a vector with std::sort.
    static void example55() {
        vector<int> v = {5, 3, 8, 1};
        sort(v.begin(), v.end());
        cout << "example55: sort, first element = " << v.front() << "\n";
    }
    // 56. Reverse sorting with custom comparator.
    static void example56() {
        vector<int> v = {5, 3, 8, 1};
        sort(v.begin(), v.end(), greater<int>());
        cout << "example56: reverse sort, first element = " << v.front() << "\n";
    }
    // 57. Using std::unique to remove duplicates.
    static void example57() {
        vector<int> v = {1, 2, 2, 3, 3, 3, 4};
        auto it = unique(v.begin(), v.end());
        v.erase(it, v.end());
        cout << "example57: unique, size = " << v.size() << "\n";
    }
    // 58. Reversing vector elements with std::reverse.
    static void example58() {
        vector<int> v = {1, 2, 3, 4, 5};
        reverse(v.begin(), v.end());
        cout << "example58: reverse, first element = " << v.front() << "\n";
    }
    // 59. Accumulating product using std::accumulate.
    static void example59() {
        vector<int> v = {1, 2, 3, 4};
        int prod = accumulate(v.begin(), v.end(), 1, multiplies<int>());
        cout << "example59: accumulate product = " << prod << "\n";
    }
    // 60. Copying a vector using std::copy.
    // static void example60() {
    //     vector<int> v = {10, 20, 30};
    //     vector<int> copy(v.size());
    //     copy(v.begin(), v.end(), copy.begin());
    //     cout << "example60: std::copy, copy[0] = " << copy[0] << "\n";
    // }
    // 61. Transforming elements with std::transform.
    static void example61() {
        vector<int> v = {1, 2, 3};
        vector<int> result(v.size());
        transform(v.begin(), v.end(), result.begin(), [](int n) { return n * 2; });
        cout << "example61: transform, first doubled = " << result[0] << "\n";
    }
    // 62. Remove-erase idiom with std::remove.
    static void example62() {
        vector<int> v = {1, 2, 3, 2, 4};
        auto it = remove(v.begin(), v.end(), 2);
        v.erase(it, v.end());
        cout << "example62: remove value 2, size = " << v.size() << "\n";
    }
    // 63. Using std::find_if with a lambda.
    static void example63() {
        vector<int> v = {10, 15, 20, 25};
        auto it = find_if(v.begin(), v.end(), [](int n) { return n > 18; });
        cout << "example63: find_if >18, value = " << (it != v.end() ? *it : -1) << "\n";
    }
    // 64. Range-based for loop.
    static void example64() {
        vector<int> v = {1, 2, 3, 4};
        cout << "example64: range-based for: ";
        for (int n : v) cout << n << " ";
        cout << "\n";
    }
    // 65. Pointer arithmetic using data().
    static void example65() {
        vector<int> v = {7, 8, 9};
        int* p = v.data();
        cout << "example65: pointer arithmetic, *p = " << *p << "\n";
    }
    // 66. Move assignment.
    static void example66() {
        vector<int> v1 = {1, 2, 3};
        vector<int> v2;
        v2 = move(v1);
        cout << "example66: move assignment, v2 size = " << v2.size() << ", v1 size = " << v1.size() << "\n";
    }
    // 67. Initializer list assignment.
    static void example67() {
        vector<int> v;
        v = {10, 20, 30};
        cout << "example67: initializer list assignment, size = " << v.size() << "\n";
    }
    // 68. Using vector as a stack (LIFO).
    static void example68() {
        vector<int> stack;
        stack.push_back(1);
        stack.push_back(2);
        stack.pop_back();
        cout << "example68: vector as stack, top = " << stack.back() << "\n";
    }
    // 69. Using vector as a queue (FIFO) via erase.
    static void example69() {
        vector<int> queue = {1, 2, 3};
        queue.erase(queue.begin());
        cout << "example69: vector as queue, new front = " << queue.front() << "\n";
    }
    // 70. Reserve and push_back in a loop.
    static void example70() {
        vector<int> v;
        v.reserve(50);
        for (int i = 0; i < 50; i++) v.push_back(i);
        cout << "example70: reserved vector, size = " << v.size() << "\n";
    }
    // 71. Specialization: vector<bool>.
    static void example71() {
        vector<bool> vb = {true, false, true};
        cout << "example71: vector<bool>, first element = " << vb[0] << "\n";
    }
    // 72. Using vector with default allocator.
    static void example72() {
        vector<int> v;
        v.push_back(1);
        cout << "example72: default allocator, size = " << v.size() << "\n";
    }
    // 73. Nested initializer lists.
    static void example73() {
        vector<vector<int>> v = {{1, 2}, {3, 4}, {5, 6}};
        cout << "example73: nested initializer, second row first element = " << v[1][0] << "\n";
    }
    // 74. Descending sort using lambda.
    static void example74() {
        vector<int> v = {3, 1, 4, 1, 5};
        sort(v.begin(), v.end(), [](int a, int b) { return a > b; });
        cout << "example74: descending sort, first element = " << v.front() << "\n";
    }
    // 75. Binary search on sorted vector.
    static void example75() {
        vector<int> v = {1, 3, 5, 7, 9};
        bool found = binary_search(v.begin(), v.end(), 5);
        cout << "example75: binary_search for 5, found = " << boolalpha << found << "\n";
    }
    // 76. lower_bound usage.
    static void example76() {
        vector<int> v = {10, 20, 30, 40};
        auto lb = lower_bound(v.begin(), v.end(), 25);
        cout << "example76: lower_bound for 25, value = " << (lb != v.end() ? *lb : -1) << "\n";
    }
    // 77. upper_bound usage.
    static void example77() {
        vector<int> v = {10, 20, 30, 40};
        auto ub = upper_bound(v.begin(), v.end(), 30);
        cout << "example77: upper_bound for 30, value = " << (ub != v.end() ? *ub : -1) << "\n";
    }
    // 78. equal_range to get a range of equal elements.
    static void example78() {
        vector<int> v = {10, 20, 20, 30, 40};
        auto range = equal_range(v.begin(), v.end(), 20);
        cout << "example78: equal_range for 20, count = " << distance(range.first, range.second) << "\n";
    }
    // 79. Emulating a circular buffer via rotate.
    static void example79() {
        vector<int> v = {1, 2, 3, 4, 5};
        rotate(v.begin(), v.begin() + 1, v.end());
        cout << "example79: circular buffer rotate, first element = " << v.front() << "\n";
    }
    // 80. Filling vector using std::fill.
    static void example80() {
        vector<int> v(5);
        fill(v.begin(), v.end(), 7);
        cout << "example80: std::fill, first element = " << v.front() << "\n";
    }
    // 81. Generating sequence with std::generate.
    static void example81() {
        vector<int> v(5);
        int n = 1;
        generate(v.begin(), v.end(), [&n]() { return n++; });
        cout << "example81: std::generate, first element = " << v.front() << "\n";
    }
    // 82. Using std::iota to fill sequential numbers.
    static void example82() {
        vector<int> v(5);
        iota(v.begin(), v.end(), 10);
        cout << "example82: std::iota, first element = " << v.front() << "\n";
    }
    // 83. Vector storing characters.
    static void example83() {
        vector<char> vc = {'a', 'b', 'c'};
        cout << "example83: vector<char>, first element = " << vc.front() << "\n";
    }
    // 84. Finding maximum element with std::max_element.
    static void example84() {
        vector<int> v = {3, 1, 4, 1, 5};
        auto maxIt = max_element(v.begin(), v.end());
        cout << "example84: max_element = " << (maxIt != v.end() ? *maxIt : -1) << "\n";
    }
    // 85. Finding minimum element with std::min_element.
    static void example85() {
        vector<int> v = {3, 1, 4, 1, 5};
        auto minIt = min_element(v.begin(), v.end());
        cout << "example85: min_element = " << (minIt != v.end() ? *minIt : -1) << "\n";
    }
    // 86. Using std::distance to calculate iterator distance.
    static void example86() {
        vector<int> v = {10, 20, 30};
        auto dist = distance(v.begin(), v.end());
        cout << "example86: std::distance = " << dist << "\n";
    }
    // 87. Advancing an iterator with std::advance.
    static void example87() {
        vector<int> v = {1, 2, 3, 4};
        auto it = v.begin();
        advance(it, 2);
        cout << "example87: std::advance, element = " << *it << "\n";
    }
    // 88. Using std::swap to exchange vectors.
    static void example88() {
        vector<int> v1 = {1, 2, 3}, v2 = {4, 5, 6};
        swap(v1, v2);
        cout << "example88: std::swap, v1 front = " << v1.front() << "\n";
    }
    // 89. Inserting using back_inserter.
    static void example89() {
        vector<int> v = {1, 2, 3};
        vector<int> result;
        copy(v.begin(), v.end(), back_inserter(result));
        cout << "example89: back_inserter, result size = " << result.size() << "\n";
    }
    // 90. Simulating front insertion.
    static void example90() {
        vector<int> v = {2, 3};
        v.insert(v.begin(), 1);
        cout << "example90: front insertion simulated, first element = " << v.front() << "\n";
    }
    // 91. Emplacing objects with complex constructors.
    static void example91() {
        struct Person { string name; int age; Person(const string &n, int a) : name(n), age(a) {} };
        vector<Person> people;
        people.emplace_back("Alice", 30);
        cout << "example91: emplace_back custom object, name = " << people[0].name << "\n";
    }
    // 92. Assignment from an initializer list.
    static void example92() {
        vector<int> v;
        v = {9, 8, 7};
        cout << "example92: initializer list assignment, first element = " << v.front() << "\n";
    }
    // 93. Using move iterators.
    static void example93() {
        vector<string> v = {"a", "b", "c"};
        vector<string> moved(v.size());
        move(v.begin(), v.end(), moved.begin());
        cout << "example93: move iterators, first moved element = " << moved[0] << "\n";
    }
    // 94. Const reverse iteration.
    static void example94() {
        vector<int> v = {1, 2, 3, 4};
        cout << "example94: const reverse iterator: ";
        for (auto it = v.crbegin(); it != v.crend(); ++it)
            cout << *it << " ";
        cout << "\n";
    }
    // 95. assign() from a C-array.
    static void example95() {
        int arr[] = {5, 6, 7};
        vector<int> v;
        v.assign(begin(arr), end(arr));
        cout << "example95: assign from array, first element = " << v.front() << "\n";
    }
    // 96. Sorting with std::stable_sort.
    static void example96() {
        vector<int> v = {4, 2, 5, 1, 3};
        stable_sort(v.begin(), v.end());
        cout << "example96: stable_sort, first element = " << v.front() << "\n";
    }
    // 97. Partitioning a vector.
    static void example97() {
        vector<int> v = {1, 2, 3, 4, 5, 6};
        auto it = partition(v.begin(), v.end(), [](int n) { return n % 2 == 0; });
        cout << "example97: partition, first element in partition = " << *it << "\n";
    }
    // 98. Remove_if with erase.
    static void example98() {
        vector<int> v = {1, 2, 3, 4, 5};
        auto it = remove_if(v.begin(), v.end(), [](int n) { return n % 2 == 0; });
        v.erase(it, v.end());
        cout << "example98: remove_if, size = " << v.size() << "\n";
    }
    // 99. Transforming vector elements to another type.
    static void example99() {
        vector<int> v = {1, 2, 3};
        vector<double> vd(v.size());
        transform(v.begin(), v.end(), vd.begin(), [](int n) { return n * 1.5; });
        cout << "example99: transform int to double, first element = " << vd.front() << "\n";
    }
    // 100. Combined vector operations.
    static void example100() {
        vector<int> v;
        v.assign({1, 2, 3, 4, 5});
        v.push_back(6);
        v.insert(v.begin() + 3, 99);
        v.erase(v.begin() + 1);
        sort(v.begin(), v.end());
        cout << "example100: combined operations, first element = " << v.front() << "\n";
    }
};

int main() {
    VectorExamples::example01();
    VectorExamples::example02();
    VectorExamples::example03();
    VectorExamples::example04();
    VectorExamples::example05();
    VectorExamples::example06();
    VectorExamples::example07();
    VectorExamples::example08();
    VectorExamples::example09();
    VectorExamples::example10();
    VectorExamples::example11();
    VectorExamples::example12();
    VectorExamples::example13();
    VectorExamples::example14();
    VectorExamples::example15();
    VectorExamples::example16();
    VectorExamples::example17();
    VectorExamples::example18();
    VectorExamples::example19();
    VectorExamples::example20();
    VectorExamples::example21();
    VectorExamples::example22();
    VectorExamples::example23();
    VectorExamples::example24();
    VectorExamples::example25();
    VectorExamples::example26();
    VectorExamples::example27();
    VectorExamples::example28();
    VectorExamples::example29();
    VectorExamples::example30();
    VectorExamples::example31();
    VectorExamples::example32();
    VectorExamples::example33();
    VectorExamples::example34();
    VectorExamples::example35();
    VectorExamples::example36();
    VectorExamples::example37();
    VectorExamples::example38();
    VectorExamples::example39();
    VectorExamples::example40();
    VectorExamples::example41();
    VectorExamples::example42();
    VectorExamples::example43();
    VectorExamples::example44();
    VectorExamples::example45();
    VectorExamples::example46();
    VectorExamples::example47();
    VectorExamples::example48();
    VectorExamples::example49();
    VectorExamples::example50();
    VectorExamples::example51();
    VectorExamples::example52();
    VectorExamples::example53();
    VectorExamples::example54();
    VectorExamples::example55();
    VectorExamples::example56();
    VectorExamples::example57();
    VectorExamples::example58();
    VectorExamples::example59();
    // VectorExamples::example60();
    VectorExamples::example61();
    VectorExamples::example62();
    VectorExamples::example63();
    VectorExamples::example64();
    VectorExamples::example65();
    VectorExamples::example66();
    VectorExamples::example67();
    VectorExamples::example68();
    VectorExamples::example69();
    VectorExamples::example70();
    VectorExamples::example71();
    VectorExamples::example72();
    VectorExamples::example73();
    VectorExamples::example74();
    VectorExamples::example75();
    VectorExamples::example76();
    VectorExamples::example77();
    VectorExamples::example78();
    VectorExamples::example79();
    VectorExamples::example80();
    VectorExamples::example81();
    VectorExamples::example82();
    VectorExamples::example83();
    VectorExamples::example84();
    VectorExamples::example85();
    VectorExamples::example86();
    VectorExamples::example87();
    VectorExamples::example88();
    VectorExamples::example89();
    VectorExamples::example90();
    VectorExamples::example91();
    VectorExamples::example92();
    VectorExamples::example93();
    VectorExamples::example94();
    VectorExamples::example95();
    VectorExamples::example96();
    VectorExamples::example97();
    VectorExamples::example98();
    VectorExamples::example99();
    VectorExamples::example100();
    return 0;
}
