/*
  Chapter 0 — Complexity, Models, and Analysis (annotated, executable)

  This single C++ file is a compact, runnable "notebook" with deeply commented code
  that teaches and demonstrates the following foundational analysis topics:

  - Asymptotics: Big-O/Ω/Θ, little-o/ω, tight bounds, logs/exponentials
  - Cost models: RAM, word-RAM, cache-aware/oblivious, external memory, PRAM
  - Analyses: worst/average/amortized (aggregate, accounting, potential), probabilistic
  - Recurrences: substitution, recursion tree, Master/Akra–Bazzi
  - Correctness: invariants, loop/structural induction, exchange arguments

  Conventions and reading tips:
  - All explanations are in comments right beside the code they refer to.
  - We instrument several algorithms to "count" work using different cost models.
  - The printed output is intentionally small; the meat is in the inline comments.
  - Build/run:  g++ -std=c++17 -O2 chapter0.cpp && ./a.out
*/

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/*
  SECTION 0 — Utilities used throughout
  - Steps: counts work in various cost models (RAM, bit/word ops, block transfers)
  - RNG: reproducible pseudo-random number generator
  - Helpers: math utilities we use in a few demonstrations
*/

struct Steps {
    // RAM model: count each primitive op as cost 1 (idealized unit cost).
    long long ram = 0;

    // Bit-level: approximate number of bit operations (e.g., in big integer arithmetic).
    long long bits = 0;

    // Word-level: count operations that are constant-time on a fixed-width word machine.
    long long words = 0;

    // Block transfers: count cache/IO misses or block moves (for cache/external-memory models).
    long long blocks = 0;

    void reset() { ram = bits = words = blocks = 0; }
};

// Deterministic RNG so the demo is stable across runs.
static std::mt19937_64 rng(123456789);

// Change-of-base for logs: log_base(x) = ln(x)/ln(base).
inline double log_base(double x, double base) {
    return std::log(x) / std::log(base);
}

// Clamp helper (for safety in a few places).
template <typename T>
inline T clamp_val(T x, T lo, T hi) {
    return std::min(std::max(x, lo), hi);
}

/*
  SECTION 1 — Asymptotics
  ---------------------------------------------------------------------------
  Core vocabulary:
  - Big-O (O): Asymptotic upper bound. f(n) ∈ O(g(n)) means ∃ c,n0 s.t. f(n) ≤ c·g(n) ∀ n≥n0.
  - Big-Ω (Ω): Asymptotic lower bound. f(n) ∈ Ω(g(n)) means ∃ c,n0 s.t. f(n) ≥ c·g(n) ∀ n≥n0.
  - Big-Θ (Θ): Tight bound. f(n) ∈ Θ(g(n)) iff f(n) ∈ O(g(n)) and f(n) ∈ Ω(g(n)).
  - little-o (o): Strictly smaller. f(n) ∈ o(g(n)) means lim_{n→∞} f(n)/g(n) = 0.
  - little-ω (ω): Strictly larger. f(n) ∈ ω(g(n)) means lim_{n→∞} f(n)/g(n) = ∞.
  - Logs: log bases differ by a constant factor, so asymptotically equivalent.
  - Exponentials: a^n dominates n^k for any k; exp grows faster than any polynomial.

  We instrument a few functions and show (light) outputs that match the stated growth.
*/

// Constant-time example (O(1)).
int constant_time_example(int n, Steps& st) {
    // One unit of work regardless of n (ignoring cost to pass/store n).
    st.ram += 1;
    return n * n + 42; // A single arithmetic op counted as 1 in the RAM model (we're modeling coarsely).
}

// Logarithmic-time example: binary search on a sorted array (O(log n)).
bool binary_search_counted(const std::vector<int>& a, int target, Steps& st) {
    int lo = 0, hi = static_cast<int>(a.size()) - 1;
    while (lo <= hi) {
        st.ram += 1; // loop guard check
        int mid = lo + (hi - lo) / 2;
        st.ram += 1; // compute mid (coarsely count as 1)
        if (a[mid] == target) { st.ram += 1; return true; } // compare and return
        st.ram += 1;
        if (a[mid] < target) { st.ram += 1; lo = mid + 1; }
        else { st.ram += 1; hi = mid - 1; }
    }
    return false;
}

// Linear-time example: sum array elements (O(n)).
long long linear_sum_counted(const std::vector<int>& a, Steps& st) {
    long long sum = 0;
    for (int x : a) {
        st.ram += 1; // loop overhead (coarsely)
        sum += x;    // addition
        st.ram += 1;
    }
    return sum;
}

// n log n example: mergesort with counted comparisons (O(n log n)).
struct MergeSortCount {
    long long comparisons = 0;
    void merge(std::vector<int>& a, int l, int m, int r) {
        int n1 = m - l + 1;
        int n2 = r - m;
        std::vector<int> L(n1), R(n2);
        for (int i = 0; i < n1; ++i) L[i] = a[l + i];
        for (int j = 0; j < n2; ++j) R[j] = a[m + 1 + j];

        int i = 0, j = 0, k = l;
        while (i < n1 && j < n2) {
            ++comparisons;            // count the key comparison L[i] <= R[j]
            if (L[i] <= R[j]) a[k++] = L[i++];
            else               a[k++] = R[j++];
        }
        while (i < n1) a[k++] = L[i++];
        while (j < n2) a[k++] = R[j++];
    }
    void sort(std::vector<int>& a, int l, int r) {
        if (l >= r) return;
        int m = l + (r - l) / 2;
        sort(a, l, m);
        sort(a, m + 1, r);
        merge(a, l, m, r);
    }
};

// Quadratic example: pairwise scan (O(n^2)).
long long quadratic_pairs_counted(int n, Steps& st) {
    long long work = 0;
    for (int i = 0; i < n; ++i) {
        st.ram += 1; // loop overhead
        for (int j = i + 1; j < n; ++j) {
            st.ram += 1; // inner loop overhead
            // "Work" for a pair (i,j)
            work += i + j;
            st.ram += 1;
        }
    }
    return work;
}

// Exponential example: naive Fibonacci (≈ φ^n). Use small n to keep runtime reasonable.
long long fib_slow(int n, Steps& st) {
    if (n <= 1) { st.ram += 1; return n; }
    st.ram += 1;
    return fib_slow(n - 1, st) + fib_slow(n - 2, st);
}

/*
  Logs/exponentials quick sanity checks:
  - log(n)/n → 0 (so log n ∈ o(n)).
  - n/log(n) → ∞ (so n ∈ ω(log n)).
  - poly vs exp: n^k / c^n → 0 for any fixed k,c>1 (exponentials win).
*/


/*
  SECTION 2 — Cost Models
  ---------------------------------------------------------------------------
  Why models? "Operation = 1 unit" is a convenient fiction. Real machines:
  - RAM model: count each basic arithmetic/logic as 1. Good first order model.
  - word-RAM: operations on a fixed-width word (e.g., 64 bits) are constant, but big-integer ops scale with length.
  - cache-aware/oblivious: data moves in blocks; locality dominates performance.
  - external memory (I/O model): large data lives on disk/SSD; block transfers are the bottleneck.
  - PRAM: idealized parallel machine with many processors and shared memory.

  We give tiny, focused demos for each.
*/

// 2.1 Word-RAM vs. big-integer bit complexity: a simple BigUInt supporting addition.
//
// In word-RAM, adding two 64-bit integers is O(1).
// In bit complexity, adding two B-bit integers costs Θ(B) bit ops.
// Below we "count" carries/adds at the limb (word) level to approximate bit cost.
struct BigUInt {
    // Store little-endian 32-bit limbs: limbs[0] is least significant.
    std::vector<uint32_t> limbs;

    BigUInt() = default;
    explicit BigUInt(uint64_t x) {
        if (x == 0) { limbs = {0}; return; }
        while (x > 0) {
            limbs.push_back(static_cast<uint32_t>(x & 0xFFFFFFFFu));
            x >>= 32;
        }
    }
    static BigUInt from_decimal_string(const std::string& s) {
        // Not performance-optimized; just enough to make large numbers for the demo.
        BigUInt x(0);
        for (char c : s) {
            if (c < '0' || c > '9') continue;
            x.mul_small(10);
            x.add_small(static_cast<uint32_t>(c - '0'));
        }
        return x;
    }
    void normalize() {
        while (limbs.size() > 1 && limbs.back() == 0) limbs.pop_back();
    }
    void add_small(uint32_t y) {
        uint64_t carry = y;
        size_t i = 0;
        while (carry > 0) {
            if (i == limbs.size()) limbs.push_back(0);
            uint64_t sum = static_cast<uint64_t>(limbs[i]) + carry;
            limbs[i] = static_cast<uint32_t>(sum & 0xFFFFFFFFu);
            carry = sum >> 32;
            ++i;
        }
        normalize();
    }
    void mul_small(uint32_t m) {
        uint64_t carry = 0;
        for (size_t i = 0; i < limbs.size(); ++i) {
            uint64_t prod = static_cast<uint64_t>(limbs[i]) * m + carry;
            limbs[i] = static_cast<uint32_t>(prod & 0xFFFFFFFFu);
            carry = prod >> 32;
        }
        if (carry) limbs.push_back(static_cast<uint32_t>(carry));
        normalize();
    }
};

// Add two BigUInts and count limb ops; each limb addition approximates 32 bit ops.
BigUInt big_add(const BigUInt& a, const BigUInt& b, Steps& st) {
    BigUInt res;
    const size_t n = std::max(a.limbs.size(), b.limbs.size());
    res.limbs.resize(n + 1, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t x = i < a.limbs.size() ? a.limbs[i] : 0;
        const uint64_t y = i < b.limbs.size() ? b.limbs[i] : 0;
        uint64_t sum = x + y + carry;
        res.limbs[i] = static_cast<uint32_t>(sum & 0xFFFFFFFFu);
        carry = sum >> 32;

        // Count: one limb addition (we approximate this as 32 bit ops) + carry handling.
        st.bits += 32;   // model 32 bit-level ops
        st.words += 1;   // one word-level op
        st.ram += 1;     // one RAM op (coarse)
    }
    res.limbs[n] = static_cast<uint32_t>(carry);
    res.normalize();
    return res;
}

// 2.2 Cache/external-memory model: count distinct blocks touched by a strided walk.
//
// The point: data moves in cache-line-sized blocks (say B elements). Consecutive access
// touches ≈ N/B blocks; strided access touches many more. We simulate the number of
// distinct blocks visited; this is a proxy for cache-miss or block-transfer count.
long long count_block_transfers_strided(size_t n_elems, size_t block_size, size_t stride) {
    if (n_elems == 0) return 0;
    std::set<size_t> blocks;
    for (size_t i = 0; i < n_elems; i += stride) {
        size_t block_idx = i / block_size;
        blocks.insert(block_idx);
    }
    // If stride doesn't divide n, last few elements may be skipped; this is just a model.
    return static_cast<long long>(blocks.size());
}

// 2.3 External-memory (I/O) complexity estimates (not a simulation):
//
// - Scanning N items uses ≈ ceil(N/B) I/Os (each I/O moves one block of B items).
// - External mergesort uses Θ((N/B) log_{M/B} (N/B)) I/Os under the tall-cache model (M is memory size).
struct ExternalMemoryIO {
    static long long scan(long long N, long long B) {
        return (N + B - 1) / B;
    }
    static double mergesort(long long N, long long B, long long M) {
        // Guard against degenerate params; if memory is tiny, base case degenerates.
        if (B <= 0 || M <= B) return std::numeric_limits<double>::infinity();
        double nb = static_cast<double>(N) / static_cast<double>(B);
        double fanout = static_cast<double>(M) / static_cast<double>(B);
        fanout = std::max(2.0, fanout); // at least 2-way merge
        // (N/B) * log_{M/B}(N/B)
        return nb * (std::log(nb) / std::log(fanout));
    }
};

// 2.4 Cache-oblivious taste: recursive (divide-and-conquer) traversal improves locality.
// We demonstrate a cache-oblivious array sum by recursively summing halves, which tends
// to use subarrays that fit in cache at some recursion depth without knowing B, M.
long long sum_cache_oblivious(const std::vector<int>& a, int l, int r) {
    if (l >= r) return 0;
    if (r - l == 1) return a[l];
    int m = l + (r - l) / 2;
    return sum_cache_oblivious(a, l, m) + sum_cache_oblivious(a, m, r);
}

// 2.5 PRAM-style parallel prefix-sum (scan) simulated in rounds.
// We simulate EREW PRAM style in "rounds": in each round, all processors read/write disjoint cells.
// Complexity: O(log n) rounds with O(n) work total.
std::vector<long long> pram_prefix_sum_rounds(const std::vector<int>& in, int& rounds, Steps& st) {
    const int n = static_cast<int>(in.size());
    std::vector<long long> cur(in.begin(), in.end()), nxt(n);
    rounds = 0;
    for (int d = 0; (1 << d) < n; ++d) {
        // Parallel for: each iteration independent (we simulate sequentially but count per-element ops).
        for (int i = 0; i < n; ++i) {
            if (i - (1 << d) >= 0) {
                nxt[i] = cur[i] + cur[i - (1 << d)];
                st.ram += 1; // one addition per active cell
            } else {
                nxt[i] = cur[i];
            }
        }
        cur.swap(nxt);
        rounds++;
    }
    return cur; // inclusive prefix sums
}

/*
  SECTION 3 — Analyses: worst-case, average-case, amortized, probabilistic
  ---------------------------------------------------------------------------
  - Worst-case: maximum cost over all inputs of a fixed size.
  - Average-case: expected cost over a distribution of inputs.
  - Amortized: average cost per operation in a sequence (even if individual ops can be expensive).
    - Aggregate method: sum costs over the sequence / number of ops.
    - Accounting method: prepay/credit cheap ops to pay for expensive ones.
    - Potential method: define Φ(state) and show amortized cost a_i = t_i + Φ(S_i) - Φ(S_{i-1}).
  - Probabilistic: use randomness (in input or algorithm) to bound expected cost.

  We demonstrate:
  - Quicksort: worst-case O(n^2) vs expected O(n log n).
  - Dynamic array (doubling): amortized O(1) push_back via aggregate/accounting/potential.
  - Binary counter: amortized O(1) increment flips via potential.
  - Hashing: expected O(1) search under simple uniform hashing (quick simulation).
*/

// 3.1 Quicksort with counted comparisons: adversarial vs randomized pivots.
struct QuickSortCount {
    long long comparisons = 0;
    std::mt19937_64 gen;
    bool randomized = false;

    explicit QuickSortCount(bool randomized_pivot, uint64_t seed = 42)
        : gen(seed), randomized(randomized_pivot) {}

    int partition(std::vector<int>& a, int l, int r) {
        int pivot_idx = randomized ? std::uniform_int_distribution<int>(l, r)(gen) : r;
        std::swap(a[pivot_idx], a[r]);
        int x = a[r];
        int i = l - 1;
        for (int j = l; j < r; ++j) {
            ++comparisons; // compare a[j] <= x
            if (a[j] <= x) std::swap(a[++i], a[j]);
        }
        std::swap(a[i + 1], a[r]);
        return i + 1;
    }
    void sort(std::vector<int>& a, int l, int r) {
        if (l >= r) return;
        int p = partition(a, l, r);
        sort(a, l, p - 1);
        sort(a, p + 1, r);
    }
};

// 3.2 Dynamic array with doubling: amortized O(1) push_back.
// Aggregate analysis: pushing n elements moves each element O(1) times on average
// because total copies <= 1 + 2 + 4 + ... < 2n, so amortized cost < 3 per push (a small constant).
class IntDynArray {
public:
    explicit IntDynArray() : data_(nullptr), size_(0), cap_(0), move_count_(0) {}
    ~IntDynArray() { delete[] data_; }

    void push_back(int x) {
        if (size_ == cap_) {
            grow(); // expensive occasionally
        }
        data_[size_++] = x; // cheap usually
    }
    size_t size() const { return size_; }
    long long moves() const { return move_count_; } // total element moves across all resizes

private:
    int* data_;
    size_t size_;
    size_t cap_;
    long long move_count_;

    void grow() {
        size_t new_cap = cap_ == 0 ? 1 : cap_ * 2;
        int* nd = new int[new_cap];
        for (size_t i = 0; i < size_; ++i) {
            nd[i] = data_[i]; // moves existing elements
            move_count_ += 1;
        }
        delete[] data_;
        data_ = nd;
        cap_ = new_cap;
    }
};

// 3.3 Binary counter: amortized O(1) flips per increment.
// Potential function Φ = number of 1-bits; increment flips some suffix of 1s to 0 and one 0 to 1.
// Actual flips per increment can be large (e.g., 111..1 -> 000..0 with carry out),
// but amortized flips ≤ 2 using Φ.
struct BinaryCounter {
    std::vector<int> bit; // least significant bit at index 0
    long long total_flips = 0;

    explicit BinaryCounter(size_t bits) : bit(bits, 0) {}

    void increment() {
        size_t i = 0;
        while (i < bit.size() && bit[i] == 1) {
            bit[i] = 0;
            total_flips += 1;
            ++i;
        }
        if (i < bit.size()) {
            bit[i] = 1;
            total_flips += 1;
        }
        // If we overflow the highest bit, we ignore carry-out in this simple model.
    }
};

// 3.4 Hashing: expected constant-time search with simple uniform hashing.
// We simulate inserting random keys into m buckets using modular hashing.
struct HashSim {
    int m; // buckets
    std::vector<int> count; // chain lengths

    explicit HashSim(int buckets) : m(buckets), count(buckets, 0) {}

    void insert(uint64_t key) {
        int b = static_cast<int>(key % static_cast<uint64_t>(m));
        count[b] += 1;
    }
    double average_chain_length() const {
        long long total = 0;
        int nonempty = 0;
        for (int c : count) {
            total += c;
            if (c) nonempty++;
        }
        return nonempty ? static_cast<double>(total) / nonempty : 0.0;
    }
    int max_chain() const {
        return *std::max_element(count.begin(), count.end());
    }
};


/*
  SECTION 4 — Recurrences and their solutions
  ---------------------------------------------------------------------------
  Techniques:
  - Substitution: guess a bound and prove by induction.
  - Recursion tree: expand levels and sum the cost per level.
  - Master Theorem: T(n) = a T(n/b) + f(n), compares f(n) to n^{log_b a}.
    Cases:
      1) f(n) = O(n^{log_b a - ε}) => T(n) = Θ(n^{log_b a})
      2) f(n) = Θ(n^{log_b a} log^k n) => T(n) = Θ(n^{log_b a} log^{k+1} n)
      3) f(n) = Ω(n^{log_b a + ε}), regularity => T(n) = Θ(f(n))
    Example: mergesort has a=2, b=2, f(n)=Θ(n) => case 2 with log_b a = 1 => Θ(n log n).
  - Akra–Bazzi: T(x) = Σ a_i T(x/b_i) + g(x). Let p solve Σ a_i b_i^{-p} = 1.
    Then T(x) = Θ(x^p (1 + ∫_1^x g(u)/u^{p+1} du)) under mild regularity constraints.

  We implement a small solver for p and demonstrate a non-Master example:
    T(n) = T(n/2) + T(n/3) + n => p solves 2^{-p} + 3^{-p} = 1, and since g(n)=n dominates n^p,
    T(n) = Θ(n).
*/

// Binary search recurrence illustration: T(n) = T(n/2) + O(1) => Θ(log n) (by Master with a=1,b=2,f=1).
// Mergesort recurrence: T(n) = 2 T(n/2) + Θ(n) => Θ(n log n).
// We'll showcase an Akra–Bazzi root solver to get p for a general split.

// Find p ∈ [p_lo, p_hi] such that sum_i a_i * b_i^{-p} = 1.
double akra_bazzi_find_p(const std::vector<double>& a, const std::vector<double>& b,
                         double p_lo = -8.0, double p_hi = 8.0) {
    auto f = [&](double p) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * std::pow(b[i], -p);
        return s;
    };
    // Ensure the function crosses 1 in [p_lo, p_hi].
    // If not, expand until it does (within some reasonable limit).
    for (int iter = 0; iter < 30 && (f(p_lo) - 1.0) * (f(p_hi) - 1.0) > 0.0; ++iter) {
        p_lo -= 1.0;
        p_hi += 1.0;
    }
    for (int it = 0; it < 80; ++it) {
        double mid = 0.5 * (p_lo + p_hi);
        if (f(mid) > 1.0) p_lo = mid;
        else p_hi = mid;
    }
    return 0.5 * (p_lo + p_hi);
}


/*
  SECTION 5 — Correctness: invariants, induction, exchange arguments
  ---------------------------------------------------------------------------
  - Loop invariants: state a property that holds before/after each iteration; prove init, maintenance, termination.
  - Structural induction: prove correctness of recursive procedures by induction on structure/size.
  - Exchange arguments: prove greedy optimality by exchanging steps of an arbitrary optimal solution.

  We illustrate:
  - Insertion sort invariant.
  - Binary search correctness via loop invariant (and a comment on strong induction).
  - Activity selection (greedy by earliest finish) with an exchange argument.
*/

// 5.1 Insertion sort with a loop invariant.
// Invariant: at the start of iteration i, a[0..i-1] is the same multiset as the original
// elements in that range and is in nondecreasing order. After inserting a[i], the invariant holds for i+1.
void insertion_sort(std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i) {
        int key = a[i];
        size_t j = i;
        // Move larger elements one position to the right to make room for key.
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = key;
        // Proof sketch in comments:
        // - Initialization: before i=1, a[0..0] is trivially sorted.
        // - Maintenance: inner loop shifts all elements greater than key; the relative order of others unchanged.
        // - Termination: when j==0 or a[j-1] ≤ key, placing key at j preserves sorted order of a[0..i].
    }
}

// 5.2 Binary search correctness via a loop invariant.
// Invariant: the target, if present, lies within the current search interval [lo, hi].
bool binary_search_with_invariant(const std::vector<int>& a, int x) {
    int lo = 0, hi = static_cast<int>(a.size()) - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] == x) return true;
        if (a[mid] < x) {
            // All elements at or before mid are ≤ a[mid] < x; target must be right of mid.
            lo = mid + 1;
        } else {
            // All elements at or after mid are ≥ a[mid] > x; target must be left of mid.
            hi = mid - 1;
        }
        // The invariant is maintained because we never discard a region that could contain x.
    }
    // On termination, lo > hi and the feasible region is empty, so x ∉ a.
    return false;
}

// 5.3 Greedy activity selection with exchange argument.
// Greedy choice: always pick the compatible activity with the earliest finishing time next.
// Exchange argument (sketch): given any optimal set S, replacing its first activity with the
// earliest-finishing compatible one never reduces the number of activities. Iterating yields the greedy solution.
struct Activity { int start, finish; };
std::vector<Activity> select_max_activities(std::vector<Activity> acts) {
    std::sort(acts.begin(), acts.end(),
              [](const Activity& a, const Activity& b) {
                  if (a.finish != b.finish) return a.finish < b.finish;
                  return a.start < b.start;
              });
    std::vector<Activity> chosen;
    int last_finish = std::numeric_limits<int>::min();
    for (const auto& a : acts) {
        if (a.start >= last_finish) {
            chosen.push_back(a);
            last_finish = a.finish;
        }
    }
    // Correctness note (exchange):
    // Let G be greedy, O be any optimal solution. If O picks some first activity f that finishes after greedy's g,
    // we can swap f with g: g finishes no later than f, so the rest of O remains compatible.
    // Repeating swaps aligns O with G without reducing its size; hence greedy is optimal.
    return chosen;
}


/*
  MAIN — Small demo harness
  ---------------------------------------------------------------------------
  We print compact metrics from each section. The important details live in comments
  near each function above. Keep n small enough for fast CI runs while still illustrative.
*/
int main() {
    std::cout << std::fixed << std::setprecision(6);

    // SECTION 1: Asymptotics demos
    {
        Steps st;
        const int n = 1'000;

        // Prepare a sorted array for search/sort demos.
        std::vector<int> a(n);
        std::iota(a.begin(), a.end(), 0);

        // Constant time
        st.reset();
        int c = constant_time_example(n, st);
        (void)c;
        std::cout << "[O(1)] RAM ops ~ " << st.ram << "\n";

        // Logarithmic
        st.reset();
        bool found = binary_search_counted(a, n - 1, st);
        (void)found;
        std::cout << "[O(log n)] RAM ops ~ " << st.ram << " for n=" << n << "\n";

        // Linear
        st.reset();
        long long sum = linear_sum_counted(a, st);
        (void)sum;
        std::cout << "[O(n)] RAM ops ~ " << st.ram << " for n=" << n << "\n";

        // n log n mergesort comparisons
        std::vector<int> b = a;
        std::shuffle(b.begin(), b.end(), rng);
        MergeSortCount msc;
        msc.sort(b, 0, static_cast<int>(b.size()) - 1);
        std::cout << "[O(n log n)] mergesort comparisons ~ " << msc.comparisons << " for n=" << n << "\n";

        // Quadratic pairs
        st.reset();
        long long work = quadratic_pairs_counted(1000, st);
        (void)work;
        std::cout << "[O(n^2)] RAM ops ~ " << st.ram << " for n=1000\n";

        // Exponential fib (small n)
        st.reset();
        long long f = fib_slow(35, st);
        (void)f;
        std::cout << "[~φ^n] fib_slow(35) RAM ops ~ " << st.ram << "\n";

        // Little-o sanity: log2(n)/n -> 0
        for (int N : {1000, 1000000}) {
            double ratio = std::log2(static_cast<double>(N)) / static_cast<double>(N);
            std::cout << "[little-o] log2(n)/n at n=" << N << " = " << ratio << "\n";
        }
    }

    // SECTION 2: Cost models
    {
        // Word-RAM vs big-int: add two 1024-bit-ish numbers (32 limbs * 32 bits)
        Steps st;
        BigUInt A, B;
        // Create ~1024-bit numbers by repeatedly multiplying and adding.
        A = BigUInt::from_decimal_string("1234567890123456789012345678901234567890");
        B = BigUInt::from_decimal_string("9876543210987654321098765432109876543210");
        st.reset();
        BigUInt C = big_add(A, B, st);
        (void)C;
        std::cout << "[word-RAM vs bits] limb adds ~ " << st.words << ", approx bit ops ~ " << st.bits << "\n";

        // Cache/external memory: block transfers in strided access
        const size_t N = 1 << 20;   // 1M elements
        const size_t B = 64;        // block size in elements (pretend cache line)
        long long blocks1 = count_block_transfers_strided(N, B, 1);
        long long blocks64 = count_block_transfers_strided(N, B, 64);
        long long blocks1024 = count_block_transfers_strided(N, B, 1024);
        std::cout << "[cache blocks] stride=1 -> " << blocks1
                  << ", stride=64 -> " << blocks64
                  << ", stride=1024 -> " << blocks1024 << "\n";

        // External memory scans and mergesort I/O complexity estimates
        long long io_scan = ExternalMemoryIO::scan(1LL << 26, 1LL << 12); // 64M items, 4K block
        double io_sort = ExternalMemoryIO::mergesort(1LL << 26, 1LL << 12, 1LL << 20); // M=1M items
        std::cout << "[I/O scan] ~" << io_scan << " block transfers; [I/O mergesort] ~" << io_sort << "\n";

        // PRAM prefix sum rounds demo
        Steps stp;
        std::vector<int> small = {1,2,3,4,5,6,7,8};
        int rounds = 0;
        auto pref = pram_prefix_sum_rounds(small, rounds, stp);
        std::cout << "[PRAM scan] rounds=" << rounds << ", work adds ~" << stp.ram
                  << ", last value=" << pref.back() << "\n";
    }

    // SECTION 3: Analyses
    {
        // Quicksort: worst case vs randomized average
        const int n = 5000;
        std::vector<int> inc(n);
        std::iota(inc.begin(), inc.end(), 0);

        // Worst case: sorted input + last-element pivot
        QuickSortCount q_worst(false, 123);
        auto worst_vec = inc;
        q_worst.sort(worst_vec, 0, n - 1);
        std::cout << "[quicksort worst] comparisons ~ " << q_worst.comparisons << " (≈ n^2/2)\n";

        // Average case: random input + randomized pivot
        QuickSortCount q_avg(true, 123);
        auto rnd_vec = inc;
        std::shuffle(rnd_vec.begin(), rnd_vec.end(), rng);
        q_avg.sort(rnd_vec, 0, n - 1);
        std::cout << "[quicksort expected] comparisons ~ " << q_avg.comparisons << " (≈ c·n log n)\n";

        // Dynamic array amortized push_back demo
        IntDynArray arr;
        const int pushes = 100000;
        for (int i = 0; i < pushes; ++i) arr.push_back(i);
        std::cout << "[dyn array] total moves=" << arr.moves()
                  << ", amortized per push=" << (double)arr.moves()/pushes << "\n";
        // Accounting/potential sketches (in comments):
        // - Accounting: charge each push 3 "credits": 1 to write the new element, 2 saved to pay future copies.
        // - Potential: Φ = 2*size - capacity. Amortized push cost a_i = t_i + ΔΦ ≤ 3.

        // Binary counter amortized flips
        BinaryCounter bc(32);
        const int incs = 100000;
        for (int i = 0; i < incs; ++i) bc.increment();
        std::cout << "[binary counter] total flips=" << bc.total_flips
                  << ", amortized flips=" << (double)bc.total_flips / incs << " (≈ 2)\n";

        // Hashing: expected chain lengths
        const int m = 10007;  // a prime bucket count
        const int keys = 20000;
        HashSim hs(m);
        std::uniform_int_distribution<uint64_t> U;
        for (int i = 0; i < keys; ++i) hs.insert(U(rng));
        std::cout << "[hashing] load factor α=" << (double)keys/m
                  << ", avg nonempty chain=" << hs.average_chain_length()
                  << ", max chain=" << hs.max_chain() << "\n";
        // Expected analysis note:
        // With simple uniform hashing, expected chain length is α and expected successful search cost is 1+α/2.
    }

    // SECTION 4: Recurrences (Akra–Bazzi demo)
    {
        // Solve 2^{-p} + 3^{-p} = 1 for T(n) = T(n/2)+T(n/3)+n.
        std::vector<double> a = {1.0, 1.0};
        std::vector<double> b = {2.0, 3.0};
        double p = akra_bazzi_find_p(a, b);
        std::cout << "[Akra–Bazzi] p for T(n)=T(n/2)+T(n/3)+n is ~ " << p
                  << " => T(n) = Θ(n) (since g(n)=n dominates n^p)\n";
        // Recursion tree & substitution are described in comments above; see mergesort/binary search examples.
    }

    // SECTION 5: Correctness
    {
        // Insertion sort sanity
        std::vector<int> v = {5, 2, 4, 6, 1, 3};
        insertion_sort(v);
        bool sorted = std::is_sorted(v.begin(), v.end());
        std::cout << "[insertion sort] sorted=" << (sorted ? "true" : "false") << "\n";

        // Binary search invariant check
        std::vector<int> s(100);
        std::iota(s.begin(), s.end(), 0);
        bool ok = binary_search_with_invariant(s, 42);
        bool miss = binary_search_with_invariant(s, 4242);
        std::cout << "[binary search correctness] find42=" << (ok ? "true" : "false")
                  << ", find4242=" << (miss ? "true" : "false") << "\n";

        // Greedy activity selection sample
        std::vector<Activity> acts = {
            {1, 4}, {3, 5}, {0, 6}, {5, 7}, {3, 9}, {5, 9}, {6, 10}, {8, 11}, {8, 12}, {2, 14}, {12, 16}
        };
        auto chosen = select_max_activities(acts);
        std::cout << "[activity selection] chosen=" << chosen.size()
                  << " (greedy optimal by exchange argument)\n";
    }

    return 0;
}