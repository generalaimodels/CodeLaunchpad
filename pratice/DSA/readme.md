
Chapter 1 — Math Tools for DSA
- Discrete math: sets, sums/series, inequalities, combinatorics
- Probability: expectation/variance, linearity, indicator vars, Chernoff bounds (overview)
- Number bases and bit math: bitwise ops, popcount, masks, gray codes
- Modular arithmetic: congruences, modular inverse (extended GCD, Fermat), CRT
- Linear algebra (for algorithms): vectors, matrices, Gaussian elimination basics

Chapter 2 — Programming & Memory Foundations
- Call stack, heap, recursion vs iteration, tail recursion
- Data representation: integers (overflow), floats (precision), strings/encodings
- References/pointers, copy/move semantics, immutability
- I/O throughput, profiling, benchmarking, micro-optimizations (branching, cache)

Chapter 3 — Recursion & Divide-and-Conquer
- Design patterns, recursion depth control
- Merge sort, quicksort (randomized), closest pair of points
- Karatsuba/Toom-Cook (overview), CDQ divide-and-conquer (offline)
- Master theorem usage and pitfalls

Chapter 4 — Sorting & Selection
- Stable/unstable, in-place/external, comparisons vs counting
- Algorithms: merge/heap/quick/intro, counting/radix/bucket, external multiway merge
- Selection: quickselect (median of medians, randomized), order statistics
- Lower bounds (Ω(n log n)), decision trees; stability and ties

Chapter 5 — Arrays & Two-Pointers
- Dynamic arrays, slicing, sliding windows, two pointers
- Prefix sums/difference arrays, range add
- Sparse tables (idempotent queries), RMQ
- Block/sqrt decomposition, Mo’s algorithm (Hilbert order)

Chapter 6 — Strings I: Fundamentals
- Encodings (ASCII/UTF-8), substrings/subsequence, prefix-function (KMP), Z-function
- Pattern matching: KMP, Z, Rabin–Karp (double hash), Boyer–Moore (overview)
- Manacher’s (longest palindromic substring), Booth’s (min rotation)
- Rolling hashes, collision handling, polynomial bases/moduli

Chapter 7 — Linked Lists & Variants
- Singly/doubly/circular, sentinel nodes, O(1) ops
- Cycle detection (Floyd), reverse, K-group reverse
- Skip lists (probabilistic), unrolled lists, rope basics

Chapter 8 — Stacks, Queues, Deques
- Stack/queue/deque ops and applications
- Expression parsing: infix→postfix (shunting-yard), evaluation
- Monotonic stack/queue, histogram/largest rectangle, sliding window min/max
- Multi-queue scheduling, bounded/deques with circular buffers

Chapter 9 — Heaps & Priority Queues
- Binary, d-ary, leftist, pairing, binomial, Fibonacci heaps (theory vs practice)
- Heapsort, decrease-key support, meldable heaps
- Indexed PQ, min-max heap, interval heap

Chapter 10 — Hashing & Hash Tables
- Separate chaining vs open addressing (linear/quadratic/double hashing)
- Cuckoo hashing, Hopscotch, Robin Hood hashing
- Hash functions: universal hashing, tabulation, Murmur/xxHash, salted/double
- Load factor, resizing, tombstones, clustering analysis

Chapter 11 — Probabilistic Structures
- Bloom filters, counting Bloom, Cuckoo filters, quotient filters
- Count-Min sketch, Misra–Gries heavy hitters, HyperLogLog (cardinality)
- Tradeoffs: FP/FN, memory, update/query time

Chapter 12 — Trees I: Basics
- Representations, recursion, traversals (pre/in/post, level order)
- Properties: height, balance, full/perfect/complete
- Threaded trees, parent pointers, binary lifting idea

Chapter 13 — BSTs & Balanced Trees
- BST ops, invariants, rotations
- AVL, Red–Black, AA, Splay, Treap, Scapegoat (balances, tradeoffs)
- Order-statistics tree (k-th, rank), interval tree, segment/interval BST

Chapter 14 — B-Tree Family & External Memory
- B-Tree, B+Tree (disk/page-aware), bulk-load, range scans
- LSM-trees (overview), write amplification
- van Emde Boas tree, y-fast/x-fast tries (integer sets)

Chapter 15 — Range Query Structures
- Fenwick (BIT): prefix sums, k-th order statistic
- Segment trees: point/range updates, lazy propagation
- Segment tree beats (chmin/chmax, complex ops)
- 2D/3D BIT/Segment tree, merge-sort tree, wavelet tree (overview)
- Li Chao tree (min/max line queries)

Chapter 16 — Persistence & Implicit Structures
- Partial/full persistence, path copying, fat nodes
- Persistent segment tree/Fenwick, persistent treap
- Implicit treap (sequence ops), rope for strings

Chapter 17 — Union-Find (Disjoint Set Union)
- Find/union, path compression, union by rank/size
- Rollback DSU (offline), DSU on tree (small-to-large), offline queries

Chapter 18 — Graphs I: Representation & Traversal
- Adjacency list/matrix, edge lists, compressed sparse row
- BFS/DFS, connected components, bipartite check (coloring)
- Topological sort (Kahn/DFS), DAG properties
- Euler tour, eulerian path/cycle (Hierholzer)

Chapter 19 — Shortest Paths
- Unweighted: BFS, multi-source BFS, 0–1 BFS
- Weighted: Dijkstra (binary/paired/Fibonacci heaps), Dial’s buckets
- Negative edges: Bellman–Ford, SPFA caveats
- All-pairs: Floyd–Warshall, Johnson’s reweighting
- A* search, heuristics, consistency

Chapter 20 — Minimum Spanning/Steiner (intro)
- MST: Kruskal (with DSU), Prim (binary/Fibonacci heap), Borůvka
- Sensitivity analysis, second-best MST
- Steiner tree (NP-hard, approximation/heuristics overview)

Chapter 21 — Connectivity, Cuts, and Components
- Bridges, articulation points (Tarjan), biconnected components
- SCC (Kosaraju, Tarjan), condensation DAG
- Min-cut: s–t cut via max-flow, global min-cut (Karger, Stoer–Wagner)

Chapter 22 — Flows, Circulations, and Matchings
- Max-flow: Ford–Fulkerson/Edmonds–Karp, Dinic (scaling), Push–Relabel (gap/HL)
- Min-cost flow: successive shortest path, potentials, SPFA/Dijkstra
- Lower/upper bounds, circulations with demands
- Matchings: bipartite (Hopcroft–Karp), Hungarian (assignment), Blossom (general)

Chapter 23 — Trees II: Advanced Techniques
- LCA: binary lifting, Euler tour + RMQ, Tarjan offline
- Heavy–Light decomposition (path queries), centroid decomposition
- Rerooting DP, tree DP patterns, virtual trees
- Dynamic trees: Link–Cut trees, Euler tour trees

Chapter 24 — Strings II: Structures
- Trie/Patricia (radix) trie, ternary search tree
- Aho–Corasick automaton (multi-pattern)
- Suffix array (doubling, SA-IS), LCP (Kasai), RMQ on LCP for LCE
- Suffix tree (Ukkonen), suffix automaton (SAM), palindromic tree (Eertree)
- BWT/ FM-index (compressed indexing, overview)

Chapter 25 — Computational Geometry I
- Vectors, dot/cross, orientation, distance, projections
- Segment intersection, line sweep (events), polygon area (shoelace)
- Convex hull (Graham/Andrew), rotating calipers (diameter/width, antipodal pairs)
- Point in polygon, winding vs ray casting, convex polygon intersection

Chapter 26 — Computational Geometry II
- Closest pair (divide-and-conquer), Delaunay/Voronoi (overview)
- Half-plane intersection, convex polygon DP, Minkowski sum
- KD-tree, range trees; R-tree (spatial index)
- Robust predicates, epsilon handling, integer geometry

Chapter 27 — Number Theory I
- GCD/extended GCD, Diophantine equations
- Modular arithmetic: inverse, pow (fast exponentiation)
- CRT (classic/Garner), modular systems with non-coprime moduli

Chapter 28 — Number Theory II: Primes & Factorization
- Sieves: Eratosthenes (segmented), linear sieve (phi/mobius)
- Primality: Miller–Rabin (deterministic bases), Baillie–PSW (overview)
- Factorization: Pollard’s Rho (ρ, Brent), trial division, wheel factorization
- Multiplicative functions: phi, mu, sigma; prefix sums; Dirichlet conv (overview)

Chapter 29 — Polynomials & Transforms
- Convolution via FFT/NTT (moduli, primitive roots), CRT for multi-mod
- Applications: big integer mult, combinatorics, string matching with convolution
- Kitamasa, Berlekamp–Massey (linear recurrences)

Chapter 30 — Combinatorics for Coding
- nCr: factorials with modular inverses, precomputation, Pascal DP
- Lucas theorem, stars and bars, Catalan numbers
- Inclusion–exclusion, Burnside/Polya (overview)

Chapter 31 — Dynamic Programming I: Core Patterns
- State modeling, transitions, base cases, reconstruction
- Classic sets: knapsack (0/1, unbounded, bounded), LIS (n log n)
- Edit distance, LCS/LPS, coin change, partition, subset sum
- DP on grids, DAGs, trees (rooted/unrooted)

Chapter 32 — Dynamic Programming II: Optimizations
- Rolling arrays, divide-and-conquer DP (quadrangle inequality), Knuth optimization
- Monotone queue optimization, Convex Hull Trick, Li Chao tree, slope trick (intro)
- Bitset DP (subset sum, SOS DP, subset convolution)
- Digit DP, profile DP, intervals DP, meet-in-the-middle
- SMAWK (monge/total monotone matrices), quadrangle/concavity conditions

Chapter 33 — Greedy & Matroids
- Greedy correctness proofs: exchange/greedy stays ahead
- Scheduling, interval problems, Huffman coding
- Matroids, greedy solvability, graphic/partition matroids

Chapter 34 — Randomized Algorithms
- Randomized quicksort/select (expected bounds), hashing analysis
- Reservoir sampling (k-sample), random projections (sketch overview)
- Monte Carlo vs Las Vegas, tail bounds usage

Chapter 35 — Online & Streaming Algorithms
- Competitive analysis: paging/caching (LRU/LFU/optimal)
- Streaming: Count–Min sketch, AMS moments (F2), quantile sketch (Greenwald–Khanna)
- Heavy hitters (Misra–Gries), HyperLogLog (cardinality), reservoir sampling
- Sliding-window variants

Chapter 36 — Constraints Graphs & SAT
- 2-SAT (implication graph), difference constraints (SP on potentials)
- Toposort feasibility, longest path on DAG, transitive closure (bitset)
- ILP/LP (overview), reductions to flow/matching

Chapter 37 — Parallel & Cache-Aware Algorithms
- Work–span model, parallel prefix/sort, union-find (parallel)
- Cache-aware/oblivious: blocked algorithms, vEB layout
- External memory algorithms: sorting, search, joins

Chapter 38 — Advanced Indexes & Compression
- Suffix arrays/trees (engineering), FM-index, wavelet trees
- Rank/select bitvectors, succinct trees (LOUDS)
- Dictionary coding: LZ77/78, Huffman/arith coding (overview)

Chapter 39 — Advanced Data Structures
- Range trees, fractional cascading, persistent kd-trees (overview)
- Euler tour trees, Link–Cut trees (dynamic connectivity/paths)
- Distance oracles, spanners (overview), centroid path cover

Chapter 40 — Practice Patterns & Heuristics
- Invariants, case analysis, small-to-large merging, offline vs online
- Coordinate compression, hashing IDs, difference constraints trick
- Binary search on answer (parametric search), meet-in-the-middle
- Randomization for robustness, stress/fuzz testing

Chapter 41 — Implementation Patterns
- Templates/generics, iterators, RAII; immutable vs mutable APIs
- Error handling, sentinels, guards, asserts
- Modular design for DS/algos, reusable components, interfaces

Chapter 42 — Testing, Verification, and Benchmarking
- Unit tests, property-based/fuzz testing, randomized tests
- Worst-case generators, adversarial inputs (hash seeding)
- Profiling: hotspots, cache misses, perf counters
- Precision traps, integer overflow guards, UB avoidance

Chapter 43 — Contest/Interview Strategy
- Reading constraints, picking approaches by bounds
- Time–memory tradeoffs, pruning, memoization vs tabulation
- Template library, snippets, checklists
- Post-mortems and pattern cataloging

Chapter 44 — Appendices (quick refs)
- Complexity cheat sheets, identities, geometry predicates
- FFT/NTT mod lists and roots, Miller–Rabin bases
- Common bugs and edge-case checklists across topics