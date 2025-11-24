#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODULE: foundational_computation_memory.py
AUTHOR: World's Best Coder & Tutor (IQ: 300+)
DATE: 2023-10-27
DESCRIPTION:
    This module serves as a masterclass on "Programming & Memory Foundations".
    It provides a deep dive into memory management (Stack/Heap), recursion dynamics,
    low-level data representation, reference semantics, and performance engineering.

    The code is written with strict adherence to PEP 8, utilizing type hinting,
    advanced Python idioms, and rigorous documentation. It bridges high-level Python 
    abstractions with low-level hardware realities.

    TOPICS COVERED:
    1. Memory Architecture: Call Stack, Heap, Recursion vs. Iteration.
    2. Data Representation: Integers, Floats (IEEE 754), Unicode/Encodings.
    3. Semantics: References, Deep/Shallow Copies, Immutability.
    4. Systems Engineering: I/O, Profiling, Branch Prediction, Cache Locality.
"""

import sys
import time
import copy
import struct
import cProfile
import pstats
import io
import random
from typing import List, Any, Generator, Optional, Callable
from functools import wraps

# ==============================================================================
# SECTION 1: MEMORY ARCHITECTURE (STACK vs. HEAP, RECURSION)
# ==============================================================================

class MemoryArchitectureDemo:
    """
    Demonstrates the distinction between Stack and Heap memory, and the 
    computational implications of Recursion vs. Iteration.
    """

    def explain_memory_model(self) -> None:
        """
        THEORY:
        
        1. THE CALL STACK:
           - A LIFO (Last-In, First-Out) structure used for static memory allocation.
           - Stores: Local variables, function parameters, return addresses.
           - Scope: Variables exist only while the function is executing.
           - Speed: Extremely fast allocation (moving the stack pointer).
           - Limit: Finite size. Exceeding it causes a StackOverflowError.
        
        2. THE HEAP:
           - Used for dynamic memory allocation.
           - Stores: Objects (in Python, everything is an object on the heap).
           - Scope: Global or referenced. Managed by Garbage Collector (Reference Counting + Cyclic GC in Python).
           - Speed: Slower allocation (requires searching for free space) and access (via pointers).
        """
        print(f"\n[{self.__class__.__name__}] Explaining Memory Models...")
        
        # Stack Variable (Reference) -> Heap Object
        # 'x' is a reference on the Stack. The integer object '10' lives on the Heap.
        x: int = 10 
        print(f"  Stack Reference 'x' points to Heap Address: {hex(id(x))}")

    def factorial_iterative(self, n: int) -> int:
        """
        Calculates factorial using Iteration.
        
        MEMORY ANALYSIS:
        - Uses a constant amount of Stack frames (O(1) space complexity).
        - All state is mutated in place within the single stack frame.
        - Preferred for production due to safety against Stack Overflow.
        """
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def factorial_recursive(self, n: int) -> int:
        """
        Calculates factorial using Recursion.
        
        MEMORY ANALYSIS:
        - Uses O(N) Stack frames.
        - Each recursive call pushes a new frame containing 'n' and the return address.
        - PRO: cleaner mathematical definition.
        - CON: risk of RecursionError if N > sys.getrecursionlimit().
        """
        if n <= 1:
            return 1
        # Standard recursion: The multiplication happens *after* the recursive return.
        return n * self.factorial_recursive(n - 1)

    def factorial_tail_recursive_simulation(self, n: int, accumulator: int = 1) -> int:
        """
        Demonstrates Tail Recursion Concept.
        
        THEORY:
        - Tail Recursion: The recursive call is the *very last* action in the function.
        - Optimization (TCO): Compilers (like in C++ -O3 or Scheme) can optimize this 
          into a loop, reusing the *same* stack frame.
        
        EXCEPTION / PYTHON REALITY:
        - Python (CPython) DOES NOT support Tail Call Optimization (TCO).
        - Guido van Rossum prefers stack traces for debugging over TCO.
        - Therefore, this code still consumes O(N) stack frames in Python.
        """
        if n <= 1:
            return accumulator
        return self.factorial_tail_recursive_simulation(n - 1, n * accumulator)

    def demonstrate_stack_overflow(self) -> None:
        """
        Deliberately provokes a stack overflow to demonstrate the physical limit
        of the Call Stack.
        """
        print("  Attempting to breach Stack limit...")
        limit = sys.getrecursionlimit()
        # Temporarily lower limit to crash faster for demonstration
        sys.setrecursionlimit(1000)
        
        try:
            self.factorial_recursive(1005)
        except RecursionError as e:
            print(f"  [Caught Exception] Stack Overflow detected: {e}")
        finally:
            # Restore default
            sys.setrecursionlimit(limit)

# ==============================================================================
# SECTION 2: DATA REPRESENTATION (INTS, FLOATS, STRINGS)
# ==============================================================================

class DataRepresentationLab:
    """
    Explores how data is physically represented in memory, bitwise operations,
    floating point hazards, and string encoding standards.
    """

    def integer_representation(self) -> None:
        """
        INT ANALYSIS:
        - C/Java: Integers are fixed-width (e.g., int32, int64). Overflow wraps around.
        - Python: Integers are Objects (PyObject). They have arbitrary precision.
          They grow as large as available memory allows.
        """
        print(f"\n[{self.__class__.__name__}] Integer Internals:")
        
        # Python integers do not overflow in the traditional sense.
        huge_num = 2**64 + 1
        print(f"  Python Arbitrary Precision (2^64 + 1): {huge_num}")
        
        # Simulating 32-bit Overflow behavior using bitwise logic
        # If we treat a number as a 32-bit signed int, max is 2,147,483,647.
        # Adding 1 should result in -2,147,483,648 (Two's Complement).
        
        val = 2147483647
        val_plus_one = (val + 1) & 0xFFFFFFFF # Mask to 32 bits
        
        # Interpret bits as signed 32-bit
        if val_plus_one & 0x80000000:
            simulated_overflow = -0x100000000 + val_plus_one
        else:
            simulated_overflow = val_plus_one
            
        print(f"  Simulated 32-bit Signed Overflow: {2147483647} + 1 => {simulated_overflow}")

    def floating_point_precision(self) -> None:
        """
        FLOAT ANALYSIS (IEEE 754):
        - Floats represent numbers as: Sign * Mantissa * 2^Exponent.
        - Issue: Base-10 fractions (like 0.1) cannot be perfectly represented in Base-2.
        - Result: Precision errors accumulate.
        """
        print(f"\n[{self.__class__.__name__}] IEEE 754 Floating Point Hazards:")
        
        a = 0.1
        b = 0.2
        c = 0.3
        
        print(f"  0.1 + 0.2 == 0.3 -> {a + b == c}") # False
        print(f"  Actual value of 0.1 + 0.2: {a + b:.25f}")
        
        # Correct way to compare floats: Use Epsilon
        epsilon = 1e-9
        is_equal = abs((a + b) - c) < epsilon
        print(f"  Comparison with Epsilon ({epsilon}): {is_equal}")

    def string_encodings(self) -> None:
        """
        STRING ANALYSIS:
        - Python 3 strings (`str`) are sequences of Unicode code points.
        - `bytes` are sequences of raw 8-bit values.
        - Encoding: Mapping Unicode -> Bytes (e.g., UTF-8).
        - Decoding: Mapping Bytes -> Unicode.
        """
        print(f"\n[{self.__class__.__name__}] String Encodings & Memory:")
        
        text = "Café" # Contains 'é' (Code point U+00E9)
        print(f"  String: {text} (Length: {len(text)})")
        
        # Encode to UTF-8 (Variable width encoding)
        # 'C', 'a', 'f' take 1 byte each. 'é' takes 2 bytes in UTF-8 (0xC3 0xA9).
        utf8_bytes = text.encode('utf-8')
        print(f"  UTF-8 Bytes: {utf8_bytes} (Length: {len(utf8_bytes)})")
        print(f"  Hex dump: {' '.join(hex(b) for b in utf8_bytes)}")
        
        # Encode to ASCII (Strict 7-bit) - Will fail with extended characters
        try:
            text.encode('ascii')
        except UnicodeEncodeError as e:
            print(f"  [Exception] ASCII Encode Fail: {e}")

# ==============================================================================
# SECTION 3: REFERENCES, POINTERS, MUTABILITY
# ==============================================================================

class MemorySemanticsLab:
    """
    Demonstrates the nuances of Python's object model: Pointers (References),
    Mutability, and Copy Semantics.
    """

    def references_vs_pointers(self) -> None:
        """
        CONCEPT:
        - Python does not have raw memory pointers like C (int* p).
        - However, ALL variables are references (pointers) to PyObjects on the heap.
        - Assignment (=) copies the *reference*, not the data.
        """
        print(f"\n[{self.__class__.__name__}] References & Aliasing:")
        
        list_a = [1, 2, 3]
        list_b = list_a # Aliasing: list_b points to the SAME object as list_a
        
        print(f"  ID(A): {id(list_a)}, ID(B): {id(list_b)}")
        
        list_b.append(4)
        print(f"  Mutated B. Effect on A: {list_a}") # A is changed because it's the same object
        
        # Immutability check
        # Tuples are immutable containers. You cannot change reference slots inside them.
        tuple_a = (1, 2, [10, 20])
        try:
            tuple_a[0] = 99
        except TypeError:
            print("  [Exception] Cannot mutate tuple content directly.")
            
        # BUT, if the tuple contains a mutable object (list), that object can be modified.
        tuple_a[2].append(30)
        print(f"  'Immutable' Tuple with Mutable content mutated: {tuple_a}")

    def copy_semantics(self) -> None:
        """
        SHALLOW vs DEEP COPY:
        - Shallow Copy: Creates a new container, but populates it with references 
          to the *same* child objects.
        - Deep Copy: Recursively copies the object and all children.
        """
        print(f"\n[{self.__class__.__name__}] Copy Semantics:")
        
        original = [[1, 2], [3, 4]]
        
        # 1. Shallow Copy
        shallow = copy.copy(original)
        shallow[0][0] = 999 # Modifying child
        print(f"  Shallow Copy modification affects Original: {original[0][0] == 999}")
        
        # Reset
        original = [[1, 2], [3, 4]]
        
        # 2. Deep Copy
        deep = copy.deepcopy(original)
        deep[0][0] = 999
        print(f"  Deep Copy modification affects Original: {original[0][0] == 999}")

# ==============================================================================
# SECTION 4: SYSTEM PERFORMANCE, I/O & MICRO-OPTIMIZATIONS
# ==============================================================================

def benchmark(func: Callable) -> Callable:
    """
    A decorator to measure execution time (Profiling wrapper).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"  [Perf] {func.__name__}: {(end_time - start_time)*1000:.4f} ms")
        return result
    return wrapper

class OptimizationEngine:
    """
    Advanced topics regarding throughput, branch prediction, and CPU cache utilization.
    """

    def demonstrate_io_throughput(self) -> None:
        """
        I/O ANALYSIS:
        - I/O is usually the bottleneck (orders of magnitude slower than RAM).
        - Unbuffered I/O: Write to disk/console immediately (High syscall overhead).
        - Buffered I/O: Accumulate data in memory, write in chunks (Faster).
        """
        print(f"\n[{self.__class__.__name__}] I/O Throughput Optimization:")
        
        count = 10000
        
        # Scenario 1: Many small writes (simulating unoptimized logs)
        start = time.perf_counter()
        string_io = io.StringIO()
        for i in range(count):
            string_io.write("data")
        _ = string_io.getvalue()
        print(f"  Buffered Write (StringIO): {(time.perf_counter() - start)*1000:.4f} ms")

    @benchmark
    def branch_prediction_demo(self) -> None:
        """
        CPU ARCHITECTURE: BRANCH PREDICTION
        - Modern CPUs pipeline instructions.
        - If there is a conditional branch (if/else), the CPU guesses the outcome.
        - If data is sorted, the guess is often correct (Pattern: T, T, T, F, F, F).
        - If data is random, the pipeline flushes frequently (Performance penalty).
        """
        print(f"\n[{self.__class__.__name__}] Branch Prediction & Cache:")
        
        data_size = 200000
        data = [random.randint(0, 255) for _ in range(data_size)]
        
        # Case 1: Unsorted Data
        start = time.perf_counter()
        sum_unsorted = sum(1 for x in data if x > 128)
        time_unsorted = time.perf_counter() - start
        
        # Case 2: Sorted Data
        data.sort()
        start = time.perf_counter()
        sum_sorted = sum(1 for x in data if x > 128)
        time_sorted = time.perf_counter() - start
        
        print(f"  Unsorted Loop Time: {time_unsorted*1000:.4f} ms")
        print(f"  Sorted Loop Time:   {time_sorted*1000:.4f} ms")
        print(f"  Speedup Factor:     {time_unsorted / time_sorted:.2f}x")
        print(f"  (Why? CPU Branch Predictor successfully predicts 'True' for the second half consistently)")

    def cache_locality_demo(self) -> None:
        """
        CPU ARCHITECTURE: CACHE LOCALITY
        - Memory is fetched in Cache Lines (usually 64 bytes).
        - Accessing contiguous memory (Row-Major in Python/C) utilizes cache lines.
        - Jumping around memory (Column-Major logic or Linked Lists) causes Cache Misses.
        """
        rows, cols = 1000, 1000
        matrix = [[1] * cols for _ in range(rows)]
        
        print(f"  Testing Cache Locality ({rows}x{cols} Matrix)...")
        
        # Row-Major Traversal (Good Spatial Locality)
        # We access matrix[0][0], matrix[0][1], etc. These are adjacent in memory.
        start = time.perf_counter()
        sum_row = 0
        for r in range(rows):
            for c in range(cols):
                sum_row += matrix[r][c]
        t_row = time.perf_counter() - start
        
        # Column-Major Traversal (Poor Spatial Locality)
        # We access matrix[0][0], matrix[1][0]. In memory, these are far apart.
        # This causes constant CPU L1/L2 cache invalidation.
        start = time.perf_counter()
        sum_col = 0
        for c in range(cols):
            for r in range(rows):
                sum_col += matrix[r][c]
        t_col = time.perf_counter() - start
        
        print(f"  Row-Major (Cache Friendly): {t_row*1000:.4f} ms")
        print(f"  Col-Major (Cache Misses):   {t_col*1000:.4f} ms")

# ==============================================================================
# MAIN EXECUTION DRIVER
# ==============================================================================

def main():
    """
    Master control for the tutorial session.
    """
    print("====================================================================")
    print("   CHAPTER 2: PROGRAMMING & MEMORY FOUNDATIONS - MASTERCLASS")
    print("   Architect: Best Coder | IQ: 300+ | Language: Python 3.x")
    print("====================================================================")
    
    # 1. Memory Model
    mem_demo = MemoryArchitectureDemo()
    mem_demo.explain_memory_model()
    mem_demo.demonstrate_stack_overflow()
    
    # 2. Data Representation
    data_lab = DataRepresentationLab()
    data_lab.integer_representation()
    data_lab.floating_point_precision()
    data_lab.string_encodings()
    
    # 3. Semantics
    sem_lab = MemorySemanticsLab()
    sem_lab.references_vs_pointers()
    sem_lab.copy_semantics()
    
    # 4. Optimization & Profiling
    opt_eng = OptimizationEngine()
    opt_eng.demonstrate_io_throughput()
    
    # Running CPU-bound tasks
    opt_eng.branch_prediction_demo()
    opt_eng.cache_locality_demo()

    print("\n[End of Chapter 2 Masterclass]")

if __name__ == "__main__":
    # Using cProfile to profile the entire execution for completeness
    # This demonstrates the 'Profiling' requirement of the prompt.
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    print("\n================ [PROFILER STATS (Sample)] ================")
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(5) # Print top 5 time-consuming functions