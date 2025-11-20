/*
    Chapter 0 — Complexity, Models, and Analysis
    Executable commentary in modern C++ (C++17). The file is self-contained and well-commented.
    All explanations, definitions, and proofs are embedded as precise comments next to the code.

    CONTENT MAP
    ------------------------------------------------------------------------------
    1) Asymptotics
       - Formal definitions: Big-O/Ω/Θ, little-o/ω, tight bounds, logarithms/exponentials
       - Examples and micro-proofs via comments and small instrumented routines

    2) Cost Models
       - RAM vs word-RAM (word size, constant-time assumptions)
       - Cache-aware vs cache-oblivious (row/col scans; recursive transpose)
       - External memory (I/O model; blocked merge conceptualization)
       - PRAM (work/span; EREW/CREW/CRCW); prefix-sum with threads (work/span comments)

    3) Analyses: worst/average/amortized/probabilistic
       - Quicksort (worst n^2, expected n log n)
       - Amortized analysis via dynamic array doubling (aggregate/accounting/potential)
       - Probabilistic expectation examples (hashing intuition; randomized pivot)

    4) Recurrences
       - Techniques: substitution, recursion tree, Master Theorem, Akra–Bazzi
       - MergeSort; T(n)=2T(n/2)+Θ(n)
       - T(n)=T(n/3)+T(2n/3)+Θ(n) with Akra–Bazzi → Θ(n log n)

    5) Correctness
       - Loop invariants: Insertion sort
       - Structural/loop induction: Binary search
       - Exchange arguments: Greedy interval scheduling (earliest finishing time)
    ------------------------------------------------------------------------------

    STYLE
    - Strong type usage, const-correctness, noexcept where applicable.
    - Clean naming, consistent casing, side-effect transparency.
    - Each routine documents complexity and (where relevant) proof sketch and model.

    NOTE
    - This program prints small demo results. All theory resides in comments tied to the code.
*/

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace edu::complexity {

/*
    SECTION 1 — ASYMPTOTICS
    ------------------------------------------------------------------------------
    DEFINITIONS (precise):
    - f(n) ∈ O(g(n))  iff ∃ c>0, n0 s.t. ∀ n≥n0: 0 ≤ f(n) ≤ c·g(n).
    - f(n) ∈ Ω(g(n))  iff ∃ c>0, n0 s.t. ∀ n≥n0: f(n) ≥ c·g(n) ≥ 0.
    - f(n) ∈ Θ(g(n))  iff f(n) ∈ O(g(n)) and f(n) ∈ Ω(g(n)).
    - f(n) ∈ o(g(n))  iff ∀ c>0, ∃ n0 s.t. ∀ n≥n0: 0 ≤ f(n) < c·g(n). (strictly smaller growth; lim f/g = 0)
    - f(n) ∈ ω(g(n))  iff ∀ c>0, ∃ n0 s.t. ∀ n≥n0: f(n) > c·g(n) ≥ 0. (strictly larger growth; lim f/g = ∞)

    TIGHT BOUNDS:
    - "Tight" means Θ(·); asymptotically matches from both above and below.

    LOGS/EXPONENTIALS (assume all logs are base ≥ 2 unless noted; bases differ by constant factor):
    - log_a n = log_b n / log_b a  ⇒ All logarithms differ by a constant factor ⇒ Θ(log n) is base-invariant.
    - Polynomial vs logarithm: log^k(n) ∈ o(n^ε) for any fixed k, ε>0.
    - Polynomial vs exponential: n^k ∈ o(c^n) for any c>1.
    - n^α log^β n grows slower than n^γ if α<γ (polynomial dominates logs).
    - 2^(log n) = n, a frequent algebraic identity (when logs base 2).
*/

// Utility for generating increasing arrays for searches
[[nodiscard]] inline std::vector<int> makeSortedArray(std::size_t n) {
    std::vector<int> a(n);
    std::iota(a.begin(), a.end(), 0);
    return a;
}

/*
    Linear scan — O(n). Worst-case and average-case both Θ(n).
*/
[[nodiscard]] inline int linearSearch(const std::vector<int>& a, int target) noexcept {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == target) return static_cast<int>(i);
    }
    return -1;
}

/*
    Binary search — O(log n). Correctness proof by loop invariant and induction below in Section 5.
    Model: RAM/word-RAM treats comparisons and index arithmetic as O(1).
*/
[[nodiscard]] inline int binarySearch(const std::vector<int>& a, int target) noexcept {
    std::size_t lo = 0, hi = a.size(); // [lo, hi)
    while (lo < hi) {
        std::size_t mid = lo + (hi - lo) / 2;
        if (a[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < a.size() && a[lo] == target) return static_cast<int>(lo);
    return -1;
}

/*
    A simple Θ(n^2) loop to illustrate polynomial growth. Useful for sanity checks.
*/
inline std::uint64_t quadraticWork(std::size_t n) noexcept {
    std::uint64_t s = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            s += (i ^ j) & 1u;
    return s;
}

/*
    SECTION 2 — COST MODELS
    ------------------------------------------------------------------------------
    MODELS:
    - RAM model: each primitive operation (add, compare, deref) costs O(1). Ignores word size.
    - word-RAM: similar but fixes a word size w (e.g., 64). Operations on words cost O(1); on big integers cost Ω(#words).
    - Cache-aware: algorithm parameterized by cache size M and block size B; can explicitly tile blocks.
    - Cache-oblivious: algorithm uses divide-and-conquer to obtain optimal/asymptotically optimal cache behavior without explicit M, B.
    - External-memory (I/O) model: dominant cost is number of block transfers between disk and main memory; compute is free; measure in I/Os.
    - PRAM (Parallel RAM): ignores synchronization cost; variants:
        - EREW (Exclusive Read Exclusive Write) — strongest restrictions (no concurrent reads/writes).
        - CREW (Concurrent Read Exclusive Write) — concurrent reads allowed.
        - CRCW (Concurrent Read Concurrent Write) — also allows concurrent writes with tie-breaking (e.g., common, arbitrary, priority).

    DEMO 2.1 — Cache effects: Row-major vs column-major sum on a dense matrix stored row-major.
    - Real cache miss counts require hardware counters; here we expose the code structure that causes good/bad locality.
*/

struct Matrix {
    std::size_t n{};
    std::vector<double> data; // row-major

    explicit Matrix(std::size_t n_, double seed = 1.0) : n(n_), data(n_ * n_) {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& x : data) x = seed * dist(rng);
    }
    [[nodiscard]] inline double& at(std::size_t r, std::size_t c) noexcept { return data[r * n + c]; }
    [[nodiscard]] inline const double& at(std::size_t r, std::size_t c) const noexcept { return data[r * n + c]; }
};

[[nodiscard]] inline double sumRowMajor(const Matrix& A) noexcept {
    // Good locality: contiguous access along rows.
    double s = 0.0;
    for (std::size_t r = 0; r < A.n; ++r)
        for (std::size_t c = 0; c < A.n; ++c)
            s += A.at(r, c);
    return s;
}

[[nodiscard]] inline double sumColMajor(const Matrix& A) noexcept {
    // Poor locality under row-major storage: stride-n access along columns.
    double s = 0.0;
    for (std::size_t c = 0; c < A.n; ++c)
        for (std::size_t r = 0; r < A.n; ++r)
            s += A.at(r, c);
    return s;
}

/*
    DEMO 2.2 — Cache-oblivious transpose via divide-and-conquer.
    - Recursively split largest dimension; base case handles small tiles (fits in cache across recursion).
    - Asymptotically optimal cache complexity up to constants for “ideal cache” (tall cache assumption).
*/
inline void transposeCacheOblivious(const Matrix& A, Matrix& B,
                                    std::size_t ar0, std::size_t ac0,
                                    std::size_t br0, std::size_t bc0,
                                    std::size_t h, std::size_t w) noexcept {
    if (h == 0 || w == 0) return;
    if (h * w <= 64) { // base case tile; constant small tile threshold
        for (std::size_t i = 0; i < h; ++i)
            for (std::size_t j = 0; j < w; ++j)
                B.at(br0 + j, bc0 + i) = A.at(ar0 + i, ac0 + j);
        return;
    }
    if (h >= w) {
        std::size_t h2 = h / 2;
        transposeCacheOblivious(A, B, ar0, ac0, br0, bc0, h2, w);
        transposeCacheOblivious(A, B, ar0 + h2, ac0, br0, bc0 + h2, h - h2, w);
    } else {
        std::size_t w2 = w / 2;
        transposeCacheOblivious(A, B, ar0, ac0, br0, bc0, h, w2);
        transposeCacheOblivious(A, B, ar0, ac0 + w2, br0 + w2, bc0, h, w - w2);
    }
}

/*
    DEMO 2.3 — External memory (I/O) model sketch via blocked merge.
    - We abstract block size B; merging reads/writes in blocks to reduce I/O.
    - Computing exact I/Os needs block-buffer accounting; here we count logical block transfers.
*/
struct IOModelCounter {
    std::size_t blockSize;
    std::size_t reads{0}, writes{0};
    explicit IOModelCounter(std::size_t B) : blockSize(B) {}
    inline void readBlock() noexcept { ++reads; }
    inline void writeBlock() noexcept { ++writes; }
};

// Merge two sorted runs using blocks of size B; counts I/Os abstractly.
inline std::vector<int> mergeBlocked(const std::vector<int>& A, const std::vector<int>& B, IOModelCounter& io) {
    const std::size_t nA = A.size(), nB = B.size();
    std::vector<int> out;
    out.reserve(nA + nB);
    std::size_t i = 0, j = 0;

    while (i < nA || j < nB) {
        // Simulate bringing a block into memory if crossing a block boundary
        if (i < nA && (i % io.blockSize == 0)) io.readBlock();
        if (j < nB && (j % io.blockSize == 0)) io.readBlock();

        std::size_t iEnd = std::min(nA, i + io.blockSize);
        std::size_t jEnd = std::min(nB, j + io.blockSize);

        // Merge sub-blocks (RAM-ideal assumption for inner loop)
        while (i < iEnd && j < jEnd) {
            if (A[i] <= B[j]) out.push_back(A[i++]);
            else out.push_back(B[j++]);
        }
        while (i < iEnd) out.push_back(A[i++]);
        while (j < jEnd) out.push_back(B[j++]);

        // Simulate writing an output block when enough accumulated
        if (out.size() % io.blockSize == 0) io.writeBlock();
    }
    if (out.size() % io.blockSize != 0) io.writeBlock(); // final partial block flush
    return out;
}

/*
    DEMO 2.4 — PRAM-like parallel prefix sum (scan).
    - Work = Θ(n); Span (critical-path length) = Θ(log n) with tree up-sweep/down-sweep.
    - Implementation with threads (not PRAM) just demonstrates structure; real performance depends on scheduler and contention.
*/
template <class T>
inline void parallelPrefixSumExclusive(std::vector<T>& a) {
    // Up-sweep (reduce) phase
    std::size_t n = a.size();
    if (n == 0) return;
    std::size_t dMax = 0;
    while ((1ull << dMax) < n) ++dMax;

    for (std::size_t d = 0; d < dMax; ++d) {
        std::size_t stride = 1ull << (d + 1);
        std::size_t half = 1ull << d;
        std::vector<std::thread> workers;
        for (std::size_t i = 0; i + stride - 1 < n; i += stride) {
            workers.emplace_back([i, half, &a]() {
                a[i + stride - 1] += a[i + half - 1];
            });
        }
        for (auto& t : workers) t.join();
    }

    // Set last to identity (0) for exclusive scan
    if (n > 0) a[n - 1] = T{};

    // Down-sweep phase
    for (std::size_t d = dMax; d-- > 0;) {
        std::size_t stride = 1ull << (d + 1);
        std::size_t half = 1ull << d;
        std::vector<std::thread> workers;
        for (std::size_t i = 0; i + stride - 1 < n; i += stride) {
            workers.emplace_back([i, half, stride, &a]() {
                T t = a[i + half - 1];
                a[i + half - 1] = a[i + stride - 1];
                a[i + stride - 1] += t;
            });
        }
        for (auto& t : workers) t.join();
    }
}

/*
    SECTION 3 — ANALYSIS TYPES
    ------------------------------------------------------------------------------
    WORST-CASE: guarantees across all inputs of length n.
    AVERAGE-CASE: expectation under specified input distribution.
    AMORTIZED: average over sequences of operations on worst-case inputs.
    PROBABILISTIC: expectation over algorithm's internal randomness.

    DEMO 3.1 — Quicksort (randomized pivot)
    - Worst-case: Θ(n^2) (e.g., already sorted with bad pivot rule).
    - Expected-case: Θ(n log n) with random pivot; probability-balanced partitions.
    - Correctness: partition is a permutation and recursive solves subproblems.
*/
template <class It, class RNG>
inline It partitionRandom(It first, It last, RNG& rng) {
    using std::swap;
    std::uniform_int_distribution<std::ptrdiff_t> dist(0, (last - first) - 1);
    It pivotIt = first + dist(rng);
    swap(*(last - 1), *pivotIt);
    auto& pivot = *(last - 1);
    It i = first;
    for (It j = first; j != last - 1; ++j) {
        if (*j <= pivot) {
            swap(*i, *j);
            ++i;
        }
    }
    swap(*i, *(last - 1));
    return i;
}

template <class It, class RNG>
inline void quicksortRandom(It first, It last, RNG& rng) {
    if (last - first <= 1) return;
    It p = partitionRandom(first, last, rng);
    quicksortRandom(first, p, rng);
    quicksortRandom(p + 1, last, rng);
}

/*
    DEMO 3.2 — Amortized analysis via dynamic array doubling
    - Operation: push_back. Strategy: if size==capacity, allocate new array of capacity*2, copy, then push.
    - Aggregate proof: Over m pushes, total copies ≤ 2m → amortized O(1).
    - Accounting: Charge 3 tokens per push; 1 pays for write, 2 banked to pay future copies.
    - Potential: Φ = 2*size - capacity (one valid choice). Amortized cost of push ≤ constant with doubling.
*/
template <class T>
class DoublingVector {
public:
    DoublingVector() = default;

    void pushBack(const T& value) {
        if (_size == _capacity) grow();
        _data[_size++] = value;
    }
    [[nodiscard]] std::size_t size() const noexcept { return _size; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return _data[i]; }
    [[nodiscard]] T& operator[](std::size_t i) noexcept { return _data[i]; }

private:
    std::vector<T> _data{};
    std::size_t _size{0};
    std::size_t _capacity{0};

    void grow() {
        std::size_t newCap = std::max<std::size_t>(1, _capacity * 2);
        std::vector<T> newData(newCap);
        for (std::size_t i = 0; i < _size; ++i) newData[i] = std::move(_data[i]); // copies counted in analysis
        _data.swap(newData);
        _capacity = newCap;
    }
};

/*
    DEMO 3.3 — Probabilistic analysis intuition
    - Hashing with chaining: Expected O(1) for lookup if load factor α = n/m is O(1).
    - Randomized selection (Quickselect): Expected O(n). We demonstrate Quickselect below for completeness.
*/
template <class It, class RNG>
inline It partitionDeterministic(It first, It last, It pivotIt) {
    using std::swap;
    swap(*(last - 1), *pivotIt);
    auto& pivot = *(last - 1);
    It i = first;
    for (It j = first; j != last - 1; ++j) {
        if (*j <= pivot) {
            swap(*i, *j);
            ++i;
        }
    }
    swap(*i, *(last - 1));
    return i;
}

template <class It, class RNG>
inline It quickselectRandom(It first, It last, std::size_t k, RNG& rng) {
    while (true) {
        if (last - first == 1) return first;
        std::uniform_int_distribution<std::ptrdiff_t> dist(0, (last - first) - 1);
        It pivotIt = first + dist(rng);
        It p = partitionDeterministic(first, last, pivotIt);
        std::size_t idx = static_cast<std::size_t>(p - first);
        if (k == idx) return p;
        if (k < idx) last = p;
        else { k -= idx + 1; first = p + 1; }
    }
}

/*
    SECTION 4 — RECURRENCES
    ------------------------------------------------------------------------------
    TECHNIQUES:
    - Substitution: hypothesize a bound, prove by induction and adjust constants.
    - Recursion tree: visualize levels; sum work per level; bound leaf and root contributions.
    - Master Theorem (common case aT(n/b) + f(n)): Cases comparing f(n) to n^{log_b a}.
    - Akra–Bazzi (general a_i T(b_i n + h_i(n)) + g(n)): find p s.t. sum a_i b_i^p = 1; then T(n) = Θ(n^p(1 + ∫ g(u)/u^{p+1} du)) under regularity.

    DEMO 4.1 — MergeSort
    - Recurrence: T(n) = 2T(n/2) + Θ(n) ⇒ T(n) = Θ(n log n) (Master Theorem, a=2, b=2, f(n)=n).
*/
template <class It>
inline void merge(It first, It mid, It last) {
    using T = typename std::iterator_traits<It>::value_type;
    std::vector<T> left(first, mid), right(mid, last);
    auto i = left.begin(), j = right.begin();
    It k = first;
    while (i != left.end() && j != right.end()) {
        if (*i <= *j) *k++ = *i++;
        else *k++ = *j++;
    }
    while (i != left.end()) *k++ = *i++;
    while (j != right.end()) *k++ = *j++;
}

template <class It>
inline void mergeSort(It first, It last) {
    auto n = last - first;
    if (n <= 1) return;
    It mid = first + n / 2;
    mergeSort(first, mid);
    mergeSort(mid, last);
    merge(first, mid, last);
}

/*
    DEMO 4.2 — Akra–Bazzi example: T(n) = T(n/3) + T(2n/3) + Θ(n)
    - Find p from (1/3)^p + (2/3)^p = 1 → p = 1 (check: 1/3 + 2/3 = 1).
    - g(n)=n ⇒ ∫ g(u)/u^{p+1} du = ∫ u/u^{2} du = ∫ 1/u du = log u → T(n)=Θ(n log n).
    - We implement a routine that uses this split to reflect the structure.
*/
inline std::size_t workAB(std::size_t n) noexcept {
    if (n <= 1) return 1;
    std::size_t a = workAB(n / 3);
    std::size_t b = workAB((2 * n) / 3);
    return a + b + n; // Θ(n) combine
}

/*
    SECTION 5 — CORRECTNESS
    ------------------------------------------------------------------------------
    INVARIANTS (Insertion Sort):
    - Loop invariant at iteration i: A[0..i) is a sorted multiset equal to initial A[0..i) permuted.
    - Initialization: holds for i=1 (one element sorted).
    - Maintenance: inserting A[i] into sorted prefix keeps it sorted and preserves elements.
    - Termination: i=n ⇒ A[0..n) sorted and is a permutation of initial array.

    INDUCTION (Binary Search):
    - Loop invariant: target ∈ A[lo..hi) if present. Mid partitions interval and shrinks it while preserving membership.
    - Termination: lo==hi ⇒ interval empty; if target exists, it must be at lo (checked).

    EXCHANGE ARGUMENT (Interval Scheduling — maximize number of non-overlapping intervals):
    - Greedy choice: pick the interval with earliest finishing time next.
    - Exchange argument: Show that any optimal solution can be transformed into one that begins with the greedy choice without reducing optimality.
      Repeating gives greedy optimality.
*/
struct Interval {
    int start;
    int end; // end time is exclusive
};

inline std::vector<Interval> intervalSchedulingMaxNonOverlapping(std::vector<Interval> intervals) {
    std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
        if (a.end != b.end) return a.end < b.end; // earliest finish time first
        return a.start < b.start;
    });
    std::vector<Interval> result;
    int currentEnd = std::numeric_limits<int>::min();
    for (const auto& itv : intervals) {
        if (itv.start >= currentEnd) {
            result.push_back(itv);
            currentEnd = itv.end;
        }
    }
    return result;
}

/*
    BONUS: Union-Find (Disjoint Set Union) with union by rank + path compression.
    - Amortized time per operation: almost constant; precisely O(α(n)) where α is inverse Ackermann, extremely slow-growing.
    - Proof uses potential and accounting over structure of trees; standard result in literature.
*/
struct DisjointSet {
    std::vector<int> parent;
    std::vector<int> rank;

    explicit DisjointSet(std::size_t n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) noexcept {
        if (parent[x] != x) parent[x] = find(parent[x]); // path compression
        return parent[x];
    }

    void unite(int x, int y) noexcept {
        x = find(x); y = find(y);
        if (x == y) return;
        if (rank[x] < rank[y]) parent[x] = y;
        else if (rank[y] < rank[x]) parent[y] = x;
        else { parent[y] = x; rank[x]++; }
    }
};

/*
    SMALL DRIVER — Exercise the building blocks with tiny inputs.
    - Focus is on code clarity and annotated theory; output is illustrative only.
*/
} // namespace edu::complexity

int main() {
    using namespace edu::complexity;

    // Asymptotics demo: searches
    {
        auto a = makeSortedArray(32);
        int ix1 = linearSearch(a, 17);
        int ix2 = binarySearch(a, 17);
        std::cout << "[Search] linear=" << ix1 << " binary=" << ix2 << "\n";
    }

    // Cost model: cache-friendly vs unfriendly traversal (timings are illustrative)
    {
        Matrix A(128);
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile double s1 = sumRowMajor(A);
        auto t1 = std::chrono::high_resolution_clock::now();
        volatile double s2 = sumColMajor(A);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto d1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto d2 = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "[Matrix] row-major ms=" << d1 << " col-major ms=" << d2 << " sums=" << (s1 + s2 > 0) << "\n";
    }

    // Cache-oblivious transpose
    {
        Matrix A(64), B(64);
        transposeCacheOblivious(A, B, 0, 0, 0, 0, A.n, A.n);
        std::cout << "[Transpose] done\n";
    }

    // External memory (I/O model) blocked merge simulation
    {
        std::vector<int> x(128), y(256);
        std::iota(x.begin(), x.end(), 0);
        std::iota(y.begin(), y.end(), 1);
        IOModelCounter io(32);
        auto z = mergeBlocked(x, y, io);
        std::cout << "[I/O Merge] size=" << z.size() << " reads=" << io.reads << " writes=" << io.writes << "\n";
    }

    // PRAM-like parallel prefix sum (exclusive)
    {
        std::vector<int> v{1,2,3,4,5,6,7,8};
        parallelPrefixSumExclusive(v);
        std::cout << "[Scan] ";
        for (auto x : v) std::cout << x << " ";
        std::cout << "\n";
    }

    // Quicksort (randomized) and Quickselect (expected complexities)
    {
        std::mt19937 rng(123);
        std::vector<int> arr{5,1,9,3,7,2,6,8,4,0};
        quicksortRandom(arr.begin(), arr.end(), rng);
        std::cout << "[Quicksort] ";
        for (auto x : arr) std::cout << x << " ";
        std::cout << "\n";

        std::vector<int> arr2{5,1,9,3,7,2,6,8,4,0};
        auto it = quickselectRandom(arr2.begin(), arr2.end(), 3, rng);
        std::cout << "[Quickselect] k=3 value=" << *it << "\n";
    }

    // Amortized doubling vector pushBack
    {
        DoublingVector<int> dv;
        for (int i = 0; i < 16; ++i) dv.pushBack(i);
        std::cout << "[DoublingVector] size=" << dv.size() << " last=" << dv[dv.size()-1] << "\n";
    }

    // MergeSort and Akra–Bazzi structured work
    {
        std::vector<int> arr{9,8,7,6,5,4,3,2,1,0};
        mergeSort(arr.begin(), arr.end());
        std::cout << "[MergeSort] ";
        for (auto x : arr) std::cout << x << " ";
        std::cout << "\n";

        std::size_t w = workAB(1024);
        std::cout << "[Akra–Bazzi work proxy] W(1024)=" << w << " (∼ n log n scale)\n";
    }

    // Correctness: interval scheduling greedy
    {
        std::vector<Interval> I{{0,3},{1,2},{3,5},{4,7},{5,9},{6,10},{8,9}};
        auto S = intervalSchedulingMaxNonOverlapping(I);
        std::cout << "[IntervalScheduling] count=" << S.size() << " chosen:";
        for (auto [s,e] : S) std::cout << " (" << s << "," << e << ")";
        std::cout << "\n";
    }

    // Union-Find
    {
        DisjointSet dsu(10);
        dsu.unite(1,2); dsu.unite(2,3); dsu.unite(5,6); dsu.unite(6,7);
        std::cout << "[DSU] find(1)=" << dsu.find(1) << " find(3)=" << dsu.find(3) << " same=" << (dsu.find(1)==dsu.find(3)) << "\n";
    }

    return 0;
}