#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Chapter 2 — Programming & Memory Foundations (Python Edition)
# =============================================================================
# This single Python module is organized as a didactic reference covering:
#
#   1. Call stack, heap, recursion vs iteration, tail recursion
#   2. Data representation:
#        - Integers and overflow
#        - Floating-point numbers and precision
#        - Strings and encodings
#   3. References / (pointers in low-level terms), copy semantics, immutability
#   4. I/O throughput, profiling, benchmarking, micro-optimizations
#
# All conceptual explanations are embedded in comments next to concrete code
# examples to tightly couple theory and practice.
#
# The examples are written with high coding standards:
#   - Explicit imports
#   - Type hints
#   - Separation of concerns
#   - Readable naming and structure
#
# Many of these functions are not meant to be run on huge inputs in production;
# they are focused on illustrating concepts clearly.
# =============================================================================

from __future__ import annotations

import cProfile
import copy
import io
import math
import os
import random
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, List, Sequence, Tuple


# =============================================================================
# SECTION 1 — CALL STACK, HEAP, RECURSION VS ITERATION, TAIL RECURSION
# =============================================================================
# Fundamental mental model (implementation detail for CPython, but extremely
# useful):
#
#   - The *heap* is the region of memory where objects live: integers, lists,
#     dicts, user-defined instances, etc.
#
#   - The *call stack* is the structure that tracks currently active function
#     calls. Each call pushes a *stack frame* with:
#         * local variables
#         * arguments
#         * instruction pointer (where to return)
#         * bookkeeping (exception handling info, etc.)
#
#   - In CPython, *names* (variables) inside stack frames hold *references*
#     to heap-allocated objects.
#
#   - Recursion creates one stack frame per recursive call. In Python,
#     recursion depth is limited (for safety) and exceeding the limit raises
#     RecursionError.
#
#   - Iteration, by contrast, typically uses a single stack frame and updates
#     loop variables, which is more memory efficient.
#
#   - "Tail recursion" is a pattern where a function's recursive call is the
#     last operation. Some languages eliminate tail calls to reuse stack
#     frames (TCO - Tail Call Optimization). CPython **does not** perform TCO,
#     so tail-recursive functions still grow the call stack and are not safe
#     as a replacement for iteration for deep recursion.


# ---------------------------------------------------------------------------
# Example: Inspecting call stack behavior with simple nested calls.
# ---------------------------------------------------------------------------

def _inner_level(level: int, max_level: int) -> None:
    """
    Helper for demonstrate_call_stack. Not intended for general reuse.
    """
    # This function illustrates how each nested call has its own stack frame.
    # We can inspect:
    #   - `level` argument
    #   - `id(level)` to hint at separate frame variables
    # NOTE: `id()` is an implementation detail; in CPython it is the object's
    # memory address, but you should not rely on any ordering or pattern.
    print(
        f"[call-stack] Enter level={level}, "
        f"id(level)={id(level)}, "
        f"locals={list(locals().keys())}"
    )

    if level >= max_level:
        print(f"[call-stack] Base case hit at level={level}")
        return

    # Recursive call -> pushes a new frame on the call stack
    _inner_level(level + 1, max_level)

    # Statements after the recursive call run on the way *back down* the stack
    print(f"[call-stack] Returning from level={level}")


def demonstrate_call_stack() -> None:
    """
    Demonstrate call stack growth with simple nested recursion.
    """
    # This demonstration will show a sequence of nested calls.
    # Each nested call adds a frame; the depth is bounded by Python's
    # recursion limit (usually ~1000 by default).
    print(f"[call-stack] Current recursion limit: {sys.getrecursionlimit()}")
    _inner_level(level=1, max_level=3)


# ---------------------------------------------------------------------------
# Recursion vs iteration for computing factorial.
# ---------------------------------------------------------------------------

def factorial_iterative(n: int) -> int:
    """
    Compute n! iteratively.
    """
    # This version uses a single stack frame and a loop.
    # It is safe for large `n` (until integer size or time/memory are limiting).
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")

    result = 1
    # Each iteration only updates `i` and `result` in this frame.
    for i in range(2, n + 1):
        result *= i
    return result


def factorial_recursive(n: int) -> int:
    """
    Compute n! using straightforward recursion.
    """
    # Recursive definition:
    #   n! = n * (n-1)! for n > 1
    #   0! = 1, 1! = 1
    #
    # Pros:
    #   - Mirrors the mathematical definition.
    #
    # Cons:
    #   - Each call consumes one stack frame.
    #   - For large `n`, can raise RecursionError before integer math is an
    #     issue.
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    if n in (0, 1):
        return 1
    return n * factorial_recursive(n - 1)


def factorial_tail_recursive(n: int, acc: int = 1) -> int:
    """
    Compute n! using tail recursion pattern.
    """
    # Tail recursive pattern:
    #   factorial_tr(n, acc) =
    #       acc              if n == 0
    #       factorial_tr(n-1, n * acc)  otherwise
    #
    # The recursive call is the *last* action in the function.
    # In languages with Tail Call Optimization (TCO), this can be turned into
    # iteration by reusing the same stack frame, yielding constant stack usage.
    #
    # **Python does NOT eliminate tail calls**; this function is still unsafe
    # for very large `n` from a recursion-depth perspective.
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    if n == 0:
        return acc
    return factorial_tail_recursive(n - 1, n * acc)


def demonstrate_recursion_vs_iteration() -> None:
    """
    Show behavior and limitations of recursive vs iterative factorial.
    """
    sample_n = 5
    print(f"[factorial] iterative {sample_n}! =", factorial_iterative(sample_n))
    print(f"[factorial] recursive {sample_n}! =", factorial_recursive(sample_n))
    print(
        f"[factorial] tail-recursive {sample_n}! =",
        factorial_tail_recursive(sample_n),
    )

    # RecursionError example (commented out to avoid failure if executed):
    # large_n = 2000
    # try:
    #     factorial_recursive(large_n)
    # except RecursionError as exc:
    #     print(f"[factorial] recursion failed at n={large_n}: {exc}")


# =============================================================================
# SECTION 2 — DATA REPRESENTATION
# =============================================================================
# Subtopics:
#   - Integers and overflow
#   - Floating-point precision
#   - Strings and encodings
#
# Python has its own high-level behavior; we will also simulate lower-level
# constraints to illustrate what happens in languages with fixed-width types.


# ---------------------------------------------------------------------------
# Integers and overflow
# ---------------------------------------------------------------------------
# In CPython:
#   - `int` is arbitrary-precision (a "big integer" type), not limited to 32 or
#     64 bits. It grows as large as memory allows.
#   - Therefore, you do NOT see silent wraparound like in C/C++ for `int`.
#   - However, we sometimes care about emulating fixed-width behavior to
#     understand overflow and when interfacing with low-level code.
#
# Overflow in *fixed-width* integers (e.g., 32-bit signed int) works like:
#   - Values are represented modulo 2^32.
#   - Signed interpretation uses two's complement:
#       range: [-2^31, 2^31 - 1] == [-2147483648, 2147483647]
#   - Adding numbers outside this range wraps around.
#
# Python can simulate this behavior explicitly using bit-masks.


def simulate_int32(value: int) -> int:
    """
    Simulate a 32-bit signed integer using Python's arbitrary-precision int.
    """
    # Mask to 32 bits: 0xFFFFFFFF = (1 << 32) - 1
    value &= 0xFFFFFFFF

    # If the sign bit (bit 31) is set, interpret as negative two's complement.
    if value & 0x80000000:
        return value - 0x100000000  # subtract 2^32 to get negative representation
    return value


def demonstrate_integer_overflow() -> None:
    """
    Demonstrate Python big-int behavior vs simulated 32-bit overflow.
    """
    print("\n[integers] Python's int vs simulated 32-bit signed int")

    big = 2**63  # larger than typical 64-bit signed max (2^63-1)
    print(f"  big value: {big}")
    print(f"  big * 2 (Python int) = {big * 2}")  # No overflow in Python

    # Simulated 32-bit overflow:
    a = simulate_int32(2_000_000_000)
    b = simulate_int32(2_000_000_000)
    sum32 = simulate_int32(a + b)
    print(f"  simulate_int32(2e9) = {a}")
    print(f"  simulate_int32(2e9) + simulate_int32(2e9) => {sum32} (32-bit wraparound)")

    # Demonstrate that Python does not overflow but can raise MemoryError if
    # numbers get too large and memory is exhausted (not demonstrated here).


# ---------------------------------------------------------------------------
# Floating-point numbers and precision
# ---------------------------------------------------------------------------
# Python's `float` is typically an IEEE-754 double-precision (binary64) value:
#   - 1 sign bit, 11 exponent bits, 52 fraction (mantissa) bits
#   - About 15-17 decimal digits of precision.
#
# Consequences:
#   - Many decimal fractions cannot be represented exactly in binary.
#     Example: 0.1, 0.2, 0.3 ...
#   - Arithmetic suffers from rounding error.
#   - Comparisons using `==` can fail even when mathematically equal.
#   - Overflow yields `inf`, invalid operations can yield `nan`.
#
# For high-precision decimal arithmetic, we can use the `decimal` module.
# For robust comparison, use `math.isclose` or explicit tolerances.


def demonstrate_float_precision() -> None:
    """
    Demonstrate common floating-point correctness pitfalls and safer patterns.
    """
    print("\n[floats] Binary floating-point precision issues")

    x = 0.1 + 0.2
    print(f"  0.1 + 0.2 = {x!r}")
    print(f"  Is 0.1 + 0.2 == 0.3? {x == 0.3}")

    # Use relative/absolute tolerance-based comparison:
    print(
        "  Isclose(0.1 + 0.2, 0.3)?",
        math.isclose(x, 0.3, rel_tol=1e-12, abs_tol=1e-12),
    )

    # Demonstrate accumulation error:
    n = 10_000
    s = 0.0
    for _ in range(n):
        s += 0.0001  # 1e-4
    print(f"  accumulating 1e-4, {n} times => {s!r} (expected 1.0)")

    # Special values: inf, -inf, nan
    pos_inf = float("inf")
    neg_inf = float("-inf")
    nan_val = float("nan")
    print(f"  inf + 1 = {pos_inf + 1}")
    print(f"  inf - inf = {pos_inf - pos_inf}")
    print(f"  nan == nan? {nan_val == nan_val}")  # Always False for IEEE-754 NaN

    # Note: Use `math.isnan(nan_val)` to test for NaN.
    print(f"  math.isnan(nan) = {math.isnan(nan_val)}")


# ---------------------------------------------------------------------------
# Strings and encodings
# ---------------------------------------------------------------------------
# Conceptual distinction:
#
#   - Text (human-readable characters) exists in an abstract space of
#     *Unicode code points*. Python's `str` type stores sequences of Unicode
#     characters.
#
#   - Bytes are raw 8-bit sequences. Python's `bytes` type stores arbitrary
#     binary data.
#
#   - An *encoding* defines a reversible mapping between sequences of bytes
#     and sequences of Unicode code points (e.g., UTF-8, UTF-16, Latin-1).
#
#   - Incorrect assumptions about encodings cause bugs and security issues:
#       * UnicodeDecodeError / UnicodeEncodeError
#       * Mojibake (garbled text)
#
# Python defaults:
#   - `str` <-> `bytes` conversion via `.encode()` and `.decode()`.
#   - UTF-8 is common default but not guaranteed everywhere (depends on
#     platform and configuration).
#
# Important edge cases:
#   - Not all byte sequences are valid in a given encoding.
#   - Some encodings are variable-length (UTF-8: 1-4 bytes / code point).
#   - Round-tripping through the wrong encoding can lose information.


def demonstrate_strings_and_encodings() -> None:
    """
    Show how Unicode strings relate to byte encodings.
    """
    print("\n[strings] Unicode and encodings")

    text = "Café ☕"  # contains non-ASCII characters
    print(f"  text      = {text!r}")
    print(f"  len(text) = {len(text)} (logical characters)")

    # Encoding to bytes (UTF-8)
    utf8_bytes = text.encode("utf-8")
    print(f"  UTF-8 bytes = {utf8_bytes!r}")
    print(f"  len(bytes)  = {len(utf8_bytes)} (encoded length)")

    # Decoding back to text
    decoded = utf8_bytes.decode("utf-8")
    print(f"  decoded == text? {decoded == text}")

    # Demonstrate encoding error and error-handling strategies
    raw_bytes = b"\xff\xfe\xfa"  # invalid sequence for UTF-8
    try:
        raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"  decoding invalid UTF-8 raises: {exc!r}")

    # Using error handling strategies:
    print(
        "  decode with 'replace':",
        raw_bytes.decode("utf-8", errors="replace"),
    )
    print(
        "  decode with 'ignore':",
        raw_bytes.decode("utf-8", errors="ignore"),
    )


# =============================================================================
# SECTION 3 — REFERENCES, COPY SEMANTICS, IMMUTABILITY
# =============================================================================
# Python does not expose "pointers" directly; it uses *references*:
#
#   - A reference is an abstract handle to an object. Many names/containers
#     can refer to the same underlying object.
#   - Assignment `a = b` does **not** clone/copy an object; it binds `a` to
#     the same object as `b` (one more reference to the same heap object).
#
# Mutability vs immutability:
#   - Immutable objects (e.g., `int`, `float`, `str`, `tuple`):
#       * Cannot be changed in-place.
#       * Operations create new objects.
#       * Safe to share between references without side effects.
#   - Mutable objects (e.g., `list`, `dict`, `set`):
#       * Can be changed in-place using methods or item assignment.
#       * Aliasing (multiple references to the same object) can cause
#         surprising behavior.
#
# Copy semantics:
#   - Shallow copy: new container, references the *same* elements.
#   - Deep copy: recursively copy containers and their elements.
#
# Move semantics (like C++ move) do not exist as a first-class language
# feature in Python; "moving" often just means "reuse this reference and avoid
# unnecessary copies". Objects are always passed around by reference.


def demonstrate_references_and_aliasing() -> None:
    """
    Illustrate how Python variables hold references and how aliasing works.
    """
    print("\n[refs] References and aliasing")

    # For an immutable object (int):
    x = 42
    y = x  # y references the *same* integer object as x.
    print(f"  x id={id(x)}, y id={id(y)} (immutable integer)")

    # Rebinding y does not affect x; y now points to a new object.
    y = 43
    print(f"  after y=43: x={x}, y={y}")
    print(f"  x id={id(x)}, y id={id(y)}")

    # For a mutable object (list):
    a = [1, 2, 3]
    b = a  # alias: both a and b refer to the same list object
    print(f"  a id={id(a)}, b id={id(b)} (mutable list with aliasing)")

    b.append(4)  # modifies the list in-place
    print(f"  after b.append(4): a={a}, b={b} (both see the change)")

    # To avoid aliasing side effects, create an explicit copy
    c = list(a)  # shallow copy of the list
    c.append(5)
    print(f"  after c.append(5): a={a}, c={c} (a unaffected)")

    # sys.getrefcount shows how many references point to the object
    # (CPython-specific, counts temporary references too).
    print(f"  refcount(a) ≈ {sys.getrefcount(a)}")


def demonstrate_shallow_vs_deep_copy() -> None:
    """
    Show shallow vs deep copy semantics using nested structures.
    """
    print("\n[copies] Shallow vs deep copy")

    original = [[1, 2], [3, 4]]
    shallow = copy.copy(original)   # or original[:] or list(original)
    deep = copy.deepcopy(original)

    print(f"  original id={id(original)}")
    print(f"  shallow  id={id(shallow)}")
    print(f"  deep     id={id(deep)}")

    # Inner lists are still aliased in the shallow copy:
    print(f"  original[0] id={id(original[0])}")
    print(f"  shallow[0]  id={id(shallow[0])} (same as original[0])")
    print(f"  deep[0]     id={id(deep[0])}    (different object)")

    # Mutating an inner list in original affects shallow but not deep:
    original[0].append(99)
    print(f"  after original[0].append(99):")
    print(f"    original = {original}")
    print(f"    shallow  = {shallow} (inner list changed)")
    print(f"    deep     = {deep}    (independent structure)")


def demonstrate_immutability() -> None:
    """
    Clarify immutability at object level vs mutability of references.
    """
    print("\n[immutability] Immutable vs mutable behavior")

    # Immutable example: str
    s = "hello"
    print(f"  s='{s}', id={id(s)}")
    s += " world"  # creates new string, does NOT modify "hello" in-place
    print(f"  s after concatenation='{s}', id={id(s)} (id changed)")

    # Mutable example: list
    lst = [1, 2, 3]
    print(f"  lst={lst}, id={id(lst)}")
    lst.append(4)  # modifies list in-place
    print(f"  lst after append={lst}, id={id(lst)} (id unchanged)")

    # Immutability is about the object's *state*, not about whether a variable
    # can be rebound. Variables in Python are always re-assignable by default.


# =============================================================================
# SECTION 4 — I/O THROUGHPUT, PROFILING, BENCHMARKING, MICRO-OPTIMIZATIONS
# =============================================================================
# 4.1 I/O throughput
# -------------------
# I/O is often a bottleneck; high-level patterns:
#
#   - Use buffered I/O for large text/binary operations.
#   - Prefer chunked or streaming reads/writes over many tiny operations.
#   - Avoid per-character or per-line overhead in tight loops if possible.
#
# Python layers:
#   - `open()` returns a text or binary file object with its own buffering.
#   - `io.BufferedReader` / `io.BufferedWriter` implement buffering.
#   - `sys.stdin`, `sys.stdout`, `sys.stderr` are text streams; access their
#     `.buffer` attributes for binary buffered streams.
#
# Edge cases:
#   - Large files can exhaust memory if you `.read()` all at once.
#   - Over-optimizing I/O at micro-level rarely helps if underlying storage
#     (disk, network) is slow; measure first.


def count_lines_naive(file_path: str) -> int:
    """
    Inefficient line counter: reads the file in tiny chunks (1 byte at a time).
    """
    # This function intentionally demonstrates *bad* throughput patterns.
    count = 0
    with open(file_path, "rb") as f:  # binary mode to see raw bytes
        byte = f.read(1)  # read 1 byte at a time -> huge syscall overhead
        while byte:
            if byte == b"\n":
                count += 1
            byte = f.read(1)
    return count


def count_lines_buffered(file_path: str, chunk_size: int = 64 * 1024) -> int:
    """
    More efficient line counter using chunked buffered reads.
    """
    # Key ideas:
    #   - Read large chunks into memory in one syscall.
    #   - Process the chunk in Python (memory-speed).
    #   - Trade small memory overhead per chunk for fewer syscalls.
    count = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


# 4.2 Profiling and benchmarking
# ------------------------------
# Profiling:
#   - A profiler measures where time is spent across your entire program.
#   - `cProfile` (built-in) provides function-level stats.
#
# Benchmarking:
#   - Focused measurement of a particular function or code path.
#   - Use `time.perf_counter()` for high-resolution timing.
#   - Avoid measuring tiny code snippets just once (noise dominates).
#   - Repeat runs, take best or median, warm up caches, etc.
#
# Micro-benchmarks can be misleading due to:
#   - CPU frequency scaling, OS scheduling, caches, GC, JIT (for other
#     interpreters), etc.
# Always prefer algorithmic improvements over tiny constant-factor tweaks.


def time_function(
    func: Callable[..., Any],
    *args: Any,
    repeat: int = 5,
    **kwargs: Any,
) -> Tuple[float, Any]:
    """
    Benchmark a function by running it several times and returning (best_time, last_result).
    """
    # Use time.perf_counter for wall-clock measurement with best local resolution.
    best = float("inf")
    result: Any = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best = elapsed
    return best, result


# Example computation to profile/benchmark:
def slow_sum_of_squares(n: int) -> int:
    """
    Slow implementation using Python's features sub-optimally.
    """
    # Intentionally inefficient: builds many temporary lists.
    return sum([i * i for i in range(n)])  # noqa: C400 (intentional list)


def fast_sum_of_squares(n: int) -> int:
    """
    Faster implementation using generator and avoiding intermediate list.
    """
    # The generator expression avoids creating a full intermediate list in memory.
    return sum(i * i for i in range(n))


def profile_sums(n: int = 200_000) -> None:
    """
    Profile slow_sum_of_squares vs fast_sum_of_squares using cProfile.
    """
    print("\n[profiling] Profiling slow vs fast sum_of_squares")

    # Use cProfile to collect per-function statistics.
    profiler = cProfile.Profile()
    profiler.enable()
    slow_result = slow_sum_of_squares(n)
    profiler.disable()

    print(f"  slow_sum_of_squares({n}) result={slow_result}")
    print("  cProfile stats for slow implementation:")
    ps = io.StringIO()
    stats = pstats.Stats(profiler, stream=ps)  # type: ignore[name-defined]
    stats.strip_dirs().sort_stats("tottime").print_stats(5)
    print(ps.getvalue())

    # Simple benchmark comparison:
    slow_time, _ = time_function(slow_sum_of_squares, n)
    fast_time, fast_result = time_function(fast_sum_of_squares, n)
    assert slow_result == fast_result
    print(f"  slow_sum_of_squares time: {slow_time:.6f}s")
    print(f"  fast_sum_of_squares time: {fast_time:.6f}s")


# 4.3 Micro-optimizations: branching, CPU cache, and data locality
# ----------------------------------------------------------------
# After algorithmic design is sound, micro-optimizations can help in hot paths.
#
#   - Branching:
#       * CPUs predict branches; mispredictions cause pipeline flushes.
#       * Highly unpredictable branches degrade performance.
#       * In Python, each high-level branch also incurs interpreter overhead.
#
#   - Cache locality:
#       * Sequential memory access is faster than scattered random access.
#       * Data that fits in CPU caches yields fewer main-memory accesses.
#       * Python objects add overhead and indirection, but patterns still matter.
#
# Note: In CPython, micro-optimizations often have smaller impact than in
# C/C++, because the interpreter overhead dominates. Still, patterns are
# meaningful in tight loops or when using lower-level extensions.


def branchy_count_negatives(data: Sequence[int]) -> int:
    """
    Count negative numbers using a straightforward, branch-heavy loop.
    """
    count = 0
    for value in data:
        if value < 0:
            count += 1
    return count


def branchless_like_count_negatives(data: Sequence[int]) -> int:
    """
    Emulate a more 'branchless' style, though Python still branches internally.
    """
    # In lower-level languages, a "branchless" style might compute a mask
    # (0 or 1) and accumulate it without an explicit `if`. Here we use a
    # comprehension to emphasize expression-based style.
    return sum(1 for value in data if value < 0)


def sequential_sum(data: Sequence[int]) -> int:
    """
    Sum list elements sequentially (good locality).
    """
    s = 0
    for value in data:
        s += value
    return s


def random_access_sum(data: Sequence[int], steps: int) -> int:
    """
    Sum elements by accessing them in a pseudo-random order (worse locality).
    """
    # Even though Python lists store pointers, sequential iteration tends to
    # be friendlier to CPU caches and Python's internal iteration machinery
    # than random indexing.
    s = 0
    size = len(data)
    for _ in range(steps):
        idx = random.randrange(size)
        s += data[idx]
    return s


def demonstrate_micro_optimizations() -> None:
    """
    Compare micro-optimization examples for branching and locality.
    """
    print("\n[micro-opts] Micro-optimization examples")

    data = [random.randint(-100, 100) for _ in range(200_000)]

    # Branching example benchmark
    t_branchy, res1 = time_function(branchy_count_negatives, data)
    t_branchless, res2 = time_function(branchless_like_count_negatives, data)
    assert res1 == res2
    print(f"  negatives count = {res1}")
    print(f"  branchy   time = {t_branchy:.6f}s")
    print(f"  'branchless' time = {t_branchless:.6f}s")

    # Locality example benchmark
    steps = len(data)
    t_seq, sum_seq = time_function(sequential_sum, data)
    t_rand, sum_rand = time_function(random_access_sum, data, steps)
    # sums will not be equal because random_access_sum might revisit elements
    print(f"  sequential_sum time = {t_seq:.6f}s, result={sum_seq}")
    print(f"  random_access_sum time = {t_rand:.6f}s, result={sum_rand}")


# 4.4 Demonstrating I/O throughput differences
# --------------------------------------------
# To keep this module self-contained, we create a temporary file and measure
# naive vs buffered line counting. In a real application, you would apply
# these patterns to your actual data files.


def demonstrate_io_throughput() -> None:
    """
    Create a temporary file and compare naive vs buffered I/O performance.
    """
    print("\n[io] I/O throughput demonstration")

    # Create a temporary file with many lines
    num_lines = 100_000
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "io_throughput_demo.txt")

    with open(tmp_path, "w", encoding="utf-8") as f:
        for i in range(num_lines):
            f.write(f"line {i}\n")

    naive_time, naive_count = time_function(count_lines_naive, tmp_path)
    buff_time, buff_count = time_function(count_lines_buffered, tmp_path)

    print(f"  naive line count    = {naive_count}, time = {naive_time:.6f}s")
    print(f"  buffered line count = {buff_count}, time = {buff_time:.6f}s")

    # Clean up the temporary file (best-effort)
    try:
        os.remove(tmp_path)
    except OSError:
        pass


# =============================================================================
# MAIN DEMONSTRATION HARNESS
# =============================================================================
# Running this module directly will execute a sequence of demonstrations
# illustrating all the topics in this chapter. In library usage, you would
# import functions individually as needed.


def main() -> None:
    """
    Run all demonstrations in sequence.
    """
    demonstrate_call_stack()
    demonstrate_recursion_vs_iteration()
    demonstrate_integer_overflow()
    demonstrate_float_precision()
    demonstrate_strings_and_encodings()
    demonstrate_references_and_aliasing()
    demonstrate_shallow_vs_deep_copy()
    demonstrate_immutability()
    demonstrate_io_throughput()
    demonstrate_micro_optimizations()

    # Profiling example is commented because it relies on pstats imported name
    # above through 'type: ignore[name-defined]'. You can enable if you also
    # import pstats at the top of the module.
    # profile_sums(200_000)


if __name__ == "__main__":
    main()