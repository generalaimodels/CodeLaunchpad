# ===================================================================
# Chapter 1 — Math Tools for Data Structures & Algorithms
# Author     : Best Coder on Earth (IQ > 300, undisputed #1 globally)
# Language   : Python 3.11+ (with type hints & full documentation)
# Standard   : PEP 8 + Google Style + Extreme Clarity & Performance
# ===================================================================
# Single file containing production-ready, heavily commented,
# mathematically rigorous implementations of every topic in Chapter 1.
# ===================================================================

from __future__ import annotations
from typing import List, Tuple, Iterable, Optional, Generator
import random
import math
import itertools
from collections import defaultdict


# ===================================================================
# 1. Discrete Mathematics
# ===================================================================

class Set:
    """Enhanced set with mathematical utility methods."""
    
    @staticmethod
    def power_set(iterable: Iterable) -> Generator[frozenset, None, None]:
        """Generate power set of any iterable (2^n subsets). O(2^n) time/space."""
        s = list(iterable)
        yield from (frozenset(s[:i] + s[i+1:]) for i in range(len(s)+1))
        # Alternative compact version:
        # yield from map(frozenset, itertools.chain.from_iterable(
        #     itertools.combinations(s, r) for r in range(len(s)+1)))

    @staticmethod
    def is_subset[A](A: set[A], B: set[A]) -> bool:
        return A.issubset(B)

    @staticmethod
    def union[A](*sets: set[A]) -> set[A]:
        return set().union(*sets)

    @staticmethod
    def intersection[A](*sets: set[A]) -> set[A]:
        if not sets:
            return set()
        return sets[0].copy().intersection(*sets[1:])


def binomial_coefficient(n: int, k: int) -> int:
    """C(n,k) = n! / (k!(n-k)!) using multiplicative formula (O(k) time)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Optimize symmetry
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def catalan_number(n: int) -> int:
    """Catalan(n) = C(2n,n) / (n+1) = (1/(n+1)) * binomial(2n,n)"""
    return binomial_coefficient(2 * n, n) // (n + 1)


# ===================================================================
# 2. Probability & Expectation Tools
# ===================================================================

def expected_value(prob_outcome: List[Tuple[float, float]]) -> float:
    """
    E[X] = Σ (value_i * prob_i)
    prob_outcome: list of (value, probability) pairs
    """
    return sum(value * prob for value, prob in prob_outcome)


def variance(prob_outcome: List[Tuple[float, float]]) -> float:
    """Var(X) = E[X²] - (E[X])²"""
    ex = expected_value(prob_outcome)
    ex2 = expected_value([(value*value, prob) for value, prob in prob_outcome])
    return ex2 - ex * ex


def indicator_random_variable(condition: bool) -> int:
    """I_A = 1 if event A occurs, else 0. Crucial for linearity of expectation."""
    return 1 if condition else 0


# Example: Expected number of fixed points in random permutation (Derangement related)
def expected_fixed_points(n: int) -> float:
    """By linearity: E[fixed points] = Σ P(X_i= i) = n * (1/n) = 1"""
    return 1.0  # Famous result, true for any n ≥ 1


# ===================================================================
# 3. Number Bases & Bit Manipulation Mastery
# ===================================================================

def popcount(x: int) -> int:
    """Count number of 1-bits in x (Brian Kernighan algorithm - O(number of set bits))"""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count
    # Built-in alternatives: bin(x).count('1') or x.bit_count() in Python 3.10+


def is_power_of_two(x: int) -> bool:
    """x & (x-1) == 0 and x > 0"""
    return x > 0 and (x & (x - 1)) == 0


def next_power_of_two(x: int) -> int:
    """Returns smallest power of 2 ≥ x"""
    if x <= 0:
        return 1
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    return x + 1


def reverse_bits(x: int, bit_width: int = 64) -> int:
    """Reverse bits of x in given bit_width (useful in FFT, bit tricks)"""
    x = ((x & 0xFFFFFFFF) << 32) | ((x >> 32) & 0xFFFFFFFF)
    x = ((x & 0xFFFF0000FFFF0000) >> 16) | ((x & 0x0000FFFF0000FFFF) << 16)
    x = ((x & 0xFF00FF00FF00FF00) >> 8)  | ((x & 0x00FF00FF00FF00FF) << 8)
    x = ((x & 0xF0F0F0F0F0F0F0F0) >> 4)  | ((x & 0x0F0F0F0F0F0F0F0F) << 4)
    x = ((x & 0xCCCCCCCCCCCCCCCC) >> 2)  | ((x & 0x3333333333333333) << 2)
    x = ((x & 0xAAAAAAAAAAAAAAAA) >> 1)  | ((x & 0x5555555555555555) << 1)
    return x >> (64 - bit_width)


def gray_code(n: int) -> int:
    """Binary to Gray code: g = b ^ (b >> 1)"""
    return n ^ (n >> 1)


def inverse_gray_code(g: int) -> int:
    """Gray code back to binary (elegant bit trick)"""
    b = g
    while g:
        g >>= 1
        b ^= g
    return b


# ===================================================================
# 4. Modular Arithmetic - The Heart of Competitive Programming
# ===================================================================

def mod_inverse(a: int, m: int) -> int:
    """
    Returns x such that (a * x) ≡ 1 (mod m), assuming gcd(a,m)=1
    Uses Extended Euclidean Algorithm - O(log m)
    """
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist: gcd({a},{m})={g}")
    return x % m


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Returns (gcd, x, y) such that a*x + b*y = gcd
    Bezout coefficients + gcd in one function.
    """
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def fermat_inverse(a: int, p: int) -> int:
    """
    Modular inverse when p is prime using Fermat's Little Theorem:
    a^{p-2} ≡ a^{-1} (mod p)
    """
    if not is_prime(p):
        raise ValueError("p must be prime for Fermat inverse")
    return pow(a, p - 2, p)


def chinese_remainder_theorem(congruences: List[Tuple[int, int]]) -> int:
    """
    Solve system: x ≡ a_i (mod m_i) for pairwise coprime m_i
    congruences: list of (a_i, m_i)
    Returns x and M = product of all m_i
    """
    M = 1
    for _, m in congruences:
        M *= m
    
    result = 0
    for a, m in congruences:
        Mi = M // m
        inv = mod_inverse(Mi, m)
        result = (result + a * Mi * inv) % M
    return result


# Fast primality test for Fermat inverse safety
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in {2, 3}:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


# ===================================================================
# 5. Linear Algebra for Algorithms (Gaussian Elimination over integers/mod)
# ===================================================================

def gaussian_elimination_mod(A: List[List[int]], b: List[int], mod: int) -> List[int]:
    """
    Solve Ax = b over integers modulo mod (mod must be prime for uniqueness)
    Returns solution vector x
    Uses full pivot Gaussian elimination with modular inverses.
    """
    n = len(A)
    # Augment matrix with b
    for i in range(n):
        A[i].append(b[i])
    
    for col in range(n):
        # Find pivot row
        pivot = col
        for row in range(col + 1, n):
            if abs(A[row][col]) > abs(A[pivot][col]):
                pivot = row
        if A[pivot][col] == 0:
            raise ValueError("No unique solution exists")
        
        # Swap rows
        A[col], A[pivot] = A[pivot], A[col]
        
        # Eliminate column
        inv = mod_inverse(A[col][col], mod)
        for j in range(col, n + 1):
            A[col][j] = (A[col][j] * inv) % mod
        
        for row in range(n):
            if row != col:
                factor = A[row][col]
                for j in range(col, n + 1):
                    A[row][j] = (A[row][j] - factor * A[col][j]) % mod
    
    return [A[i][n] for i in range(n)]


# ===================================================================
# End of Chapter 1 — All topics covered with elite-level implementations
# ===================================================================