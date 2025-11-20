#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 1 — Math Tools for DSA (Python Edition)

This module is a compact but rigorous "math toolkit" for competitive programming
and algorithm design. It is written as a single file to keep concepts local,
with detailed inline commentary. Topics:

1. Discrete mathematics
   - Sets and power sets
   - Sums and series (arithmetic, geometric, harmonic, prefix sums)
   - Inequalities (basic tools used in analysis)
   - Combinatorics (factorials, binomial coefficients, combinatorial generation)

2. Probability (for algorithms)
   - Expectation, variance, linearity of expectation
   - Indicator random variables
   - Chernoff bounds (as code implementing standard tail inequalities)

3. Number bases and bit math
   - Base conversion
   - Bitwise operations, popcount, masks
   - Submask enumeration and Gray codes

4. Modular arithmetic
   - Congruences and safe modular operations
   - Extended Euclidean algorithm and modular inverses
   - Fast modular exponentiation
   - Chinese Remainder Theorem (CRT)

5. Linear algebra (for algorithms)
   - Vector and matrix utilities
   - Gaussian elimination over reals
   - Gaussian elimination modulo a prime

The code emphasizes:
- Correctness and clarity over micro-optimizations
- Clean interfaces
- Explicit handling of corner cases and preconditions
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Sequence,
    Tuple,
    TypeVar,
)
import math
import random


# ============================================================================
# Type aliases and generic parameters
# ============================================================================

T = TypeVar("T")
Number = TypeVar("Number", int, float)  # used where both int/float are acceptable


# ============================================================================
# 1. DISCRETE MATHEMATICS UTILITIES
# ============================================================================

# ---------------------------------------------------------------------------
# 1.1 Sets and power sets
# ---------------------------------------------------------------------------


def power_set(elements: Sequence[T]) -> List[List[T]]:
    """
    Compute the power set (set of all subsets) of a small finite set/list.

    elements:
        A finite sequence of distinct elements (distinctness is assumed but not
        strictly required; duplicates would generate duplicate subsets).
    returns:
        A list of subsets, where each subset is represented as a list.
        The subsets are in lexicographic order induced by bit masks:
            - subset index 'mask' corresponds to all elements i where bit i is 1.

    Complexity:
        O(2^n * n) for n = len(elements), dominated by subset construction.
        In practice, power sets are only tractable for n <= 20 or so.
    """
    n = len(elements)
    subsets: List[List[T]] = []
    # 2^n subsets, indices from 0 to (1<<n) - 1
    for mask in range(1 << n):
        subset: List[T] = []
        # test each bit position 'i' of the mask
        for i in range(n):
            if (mask >> i) & 1:
                subset.append(elements[i])
        subsets.append(subset)
    return subsets


def set_union(*sets: Iterable[T]) -> set[T]:
    """
    Compute union of multiple iterables as a Python set.

    Notes:
        - Python's built-in set type already provides `.union()` and `|`.
        - Time complexity is linear in the total number of elements.
    """
    result: set[T] = set()
    for s in sets:
        result.update(s)
    return result


def set_intersection(*sets: Iterable[T]) -> set[T]:
    """
    Compute intersection of multiple iterables as a Python set.

    Implementation details:
        - We convert each iterable to a set.
        - Start from the smallest set to minimize intersection cost.
    """
    converted: List[set[T]] = [set(s) for s in sets]
    if not converted:
        return set()
    converted.sort(key=len)
    result = converted[0].copy()
    for s in converted[1:]:
        result.intersection_update(s)
        if not result:  # early exit if intersection is already empty
            break
    return result


# ---------------------------------------------------------------------------
# 1.2 Sums and series
# ---------------------------------------------------------------------------


def arithmetic_series_sum(n: int, a1: Number = 1, d: Number = 1) -> Number:
    """
    Sum of first n terms of an arithmetic progression:
        a_k = a1 + (k - 1) * d, for k = 1..n

    Closed form:
        S_n = n * (2*a1 + (n - 1)*d) / 2

    Pre-conditions:
        - n >= 0

    Notes:
        - Uses integer arithmetic when inputs are integers (no precision loss).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0  # convention: empty sum is 0
    return n * (2 * a1 + (n - 1) * d) / 2


def geometric_series_sum(n: int, a1: Number = 1, r: Number = 2) -> float:
    """
    Sum of first n terms of a geometric progression:
        a_k = a1 * r^(k - 1), for k = 1..n

    Closed form (for r != 1):
        S_n = a1 * (1 - r^n) / (1 - r)

    Special cases:
        - If r == 1, the sequence is constant => sum = n * a1

    Pre-conditions:
        - n >= 0

    Warning:
        - For large n and |r| > 1, r^n might overflow or lose precision.
        - We keep everything as float for uniformity.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0.0
    if r == 1:
        return float(n * a1)
    return float(a1) * (1.0 - float(r) ** n) / (1.0 - float(r))


def harmonic_number(n: int) -> float:
    """
    Compute the nth harmonic number:
        H_n = sum_{k=1}^n 1/k

    Asymptotics:
        H_n = ln(n) + gamma + O(1/n), where gamma is Euler-Mascheroni constant.

    Usage in algorithms:
        - Time complexity of some algorithms (e.g. randomized quickselect,
          coupon collector) involves harmonic numbers.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0.0
    # Direct summation is O(n); good up to ~10^7 if necessary, but we usually
    # handle smaller n or approximate with log n.
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / k
    return total


def prefix_sums(arr: Sequence[Number]) -> List[Number]:
    """
    Compute prefix sums for a sequence:
        prefix[i] = sum_{k=0}^{i-1} arr[k], with prefix[0] = 0

    Complexity:
        O(n) time, O(n) additional space.

    Use-case:
        - Range sum queries:
            sum(arr[l:r]) = prefix[r] - prefix[l]
        - Many DP transitions can be optimized with prefix sums.
    """
    prefix: List[Number] = [0]  # prefix[0] is zero by convention
    running: Number = 0  # type: ignore[assignment]
    for x in arr:
        running = running + x  # type: ignore[assignment]
        prefix.append(running)
    return prefix


# ---------------------------------------------------------------------------
# 1.3 Inequalities (basic tools)
# ---------------------------------------------------------------------------


def verify_am_gm(values: Sequence[float], eps: float = 1e-9) -> bool:
    """
    Verify the Arithmetic Mean - Geometric Mean (AM-GM) inequality numerically:

        For non-negative values x_i,
            (x1 + x2 + ... + xn) / n >= (x1 * x2 * ... * xn)^(1/n)

    This function checks the inequality within numerical tolerance 'eps'.

    Pre-conditions:
        - values must be non-empty
        - all values must be >= 0
    """
    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    for v in values:
        if v < 0:
            raise ValueError("AM-GM requires non-negative values")

    # Arithmetic mean
    am = sum(values) / n

    # Geometric mean; if any v == 0, geometric mean is 0 (but AM-GM still holds)
    if any(v == 0 for v in values):
        gm = 0.0
    else:
        # Use logs for numerical stability: log(gm) = (1/n) * sum log(v)
        log_sum = sum(math.log(v) for v in values)
        gm = math.exp(log_sum / n)

    return am + eps >= gm  # allow tiny numerical slack


def harmonic_series_upper_bound(n: int) -> float:
    """
    Simple analytical upper bound on the harmonic series:

        H_n <= ln(n) + 1

    This is a classical inequality used to bound sums in algorithm analysis.

    We return the RHS: ln(n) + 1, which upper-bounds H_n for all n >= 1.
    """
    if n <= 0:
        raise ValueError("n must be positive for ln(n)")
    return math.log(n) + 1.0


# ---------------------------------------------------------------------------
# 1.4 Combinatorics (factorials, binomial coefficients, generation)
# ---------------------------------------------------------------------------


def factorial(n: int) -> int:
    """
    Compute n! (factorial of n) iteratively.

    Definition:
        n! = 1 * 2 * ... * n, with 0! = 1 by convention.

    Pre-conditions:
        - n >= 0
    """
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def binomial_coefficient(n: int, k: int) -> int:
    """
    Compute binomial coefficient "n choose k" = C(n, k).

    Definition:
        C(n, k) = n! / (k! (n-k)!) for 0 <= k <= n, else 0.

    Implementation:
        - Uses multiplicative formula to avoid large intermediate factorials:
            C(n, k) = product_{i=1}^k (n + 1 - i) / i
        - Uses symmetry C(n, k) = C(n, n-k) to minimize k.

    Complexity:
        O(k), where k = min(k, n-k).
    """
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n + 1 - i) // i
    return result


def binomial_coefficient_mod(
    n: int,
    k: int,
    mod: int,
    precomputed_fact: Sequence[int] | None = None,
    precomputed_inv_fact: Sequence[int] | None = None,
) -> int:
    """
    Compute binomial coefficient C(n, k) modulo a prime 'mod'.

    Two modes:
        1) If precomputed_fact and precomputed_inv_fact are provided:
            - Use them directly for O(1) evaluation.
        2) Otherwise:
            - Use the multiplicative formula with modular inverses in O(k).

    Pre-conditions:
        - mod must be prime if using factorial-based method (for modular inverses).
    """
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    if precomputed_fact is not None and precomputed_inv_fact is not None:
        # Precomputed factorials: fact[i] = i! % mod
        # Precomputed inverse factorials: inv_fact[i] = (i!)^{-1} % mod
        if n >= len(precomputed_fact):
            raise ValueError("n exceeds precomputed factorial range")
        fact = precomputed_fact
        inv_fact = precomputed_inv_fact
        return fact[n] * inv_fact[k] % mod * inv_fact[n - k] % mod

    # Fallback: multiplicative formula with modular inverses
    result = 1
    for i in range(1, k + 1):
        numerator = n + 1 - i
        # We need numerator / i modulo mod => numerator * inverse(i) mod mod
        inv_i = mod_inverse(i, mod)
        result = (result * (numerator % mod)) % mod
        result = (result * inv_i) % mod
    return result


def precompute_factorials_mod(n: int, mod: int) -> Tuple[List[int], List[int]]:
    """
    Precompute factorials and inverse factorials up to n modulo a prime 'mod'.

    Returns:
        (fact, inv_fact) where:
            - fact[i] = i! % mod
            - inv_fact[i] = (i!)^{-1} % mod

    Complexity:
        O(n) time and space.

    Pre-conditions:
        - mod must be prime (we use Fermat's little theorem to invert n!).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if mod <= 1:
        raise ValueError("mod must be > 1 and typically prime")

    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % mod

    inv_fact = [1] * (n + 1)
    # Compute (n!)^{-1} via Fermat: x^{-1} ≡ x^{mod-2} (mod mod) for prime mod
    inv_fact[n] = pow(fact[n], mod - 2, mod)
    # Use inv_fact[i-1] = inv_fact[i] * i mod mod (reverse recurrence)
    for i in range(n, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod

    return fact, inv_fact


def generate_combinations(elements: Sequence[T], k: int) -> List[Tuple[T, ...]]:
    """
    Generate all k-combinations of 'elements' in lexicographic order.

    We implement a standard iterative combinatorial generation algorithm
    without relying on itertools, to make the logic explicit.

    Complexity:
        O(C(n, k) * k) time and O(k) working space.
    """
    n = len(elements)
    if k < 0 or k > n:
        return []

    # Index combination: we generate combinations of indices into 'elements'
    idx = list(range(k))  # first combination: [0, 1, ..., k-1]
    result: List[Tuple[T, ...]] = []

    while True:
        # Map index combination to actual elements
        result.append(tuple(elements[i] for i in idx))

        # Find rightmost index that can be incremented
        # idx[i] < n - k + i ensures enough room for the remaining positions
        i = k - 1
        while i >= 0 and idx[i] == n - k + i:
            i -= 1
        if i < 0:
            break  # all combinations generated

        # Increment position i and reset following positions
        idx[i] += 1
        for j in range(i + 1, k):
            idx[j] = idx[j - 1] + 1

    return result


# ============================================================================
# 2. PROBABILITY TOOLS
# ============================================================================

# ---------------------------------------------------------------------------
# 2.1 Expectation, variance, linearity
# ---------------------------------------------------------------------------


def expectation(values: Sequence[float], probabilities: Sequence[float]) -> float:
    """
    Compute E[X] for a discrete random variable X with:
        P[X = values[i]] = probabilities[i].

    Pre-conditions:
        - len(values) == len(probabilities)
        - probabilities[i] >= 0
        - sum(probabilities) ~= 1 (within small numerical epsilon)

    Implementation:
        E[X] = sum_i values[i] * probabilities[i]
    """
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have same length")
    if not values:
        raise ValueError("distribution must be non-empty")
    for p in probabilities:
        if p < 0:
            raise ValueError("probabilities must be non-negative")

    total_prob = sum(probabilities)
    if not math.isclose(total_prob, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"probabilities must sum to 1 (got {total_prob})")

    return float(sum(v * p for v, p in zip(values, probabilities)))


def variance(values: Sequence[float], probabilities: Sequence[float]) -> float:
    """
    Compute Var(X) for discrete X using:
        Var(X) = E[X^2] - (E[X])^2
    """
    ex = expectation(values, probabilities)
    ex2 = float(sum((v * v) * p for v, p in zip(values, probabilities)))
    return ex2 - ex * ex


def sample_mean(random_var: Callable[[], float], trials: int) -> float:
    """
    Monte-Carlo estimate of E[X] by averaging i.i.d. samples.

    random_var:
        Function that returns one sample of the random variable X.
    trials:
        Number of samples to draw. Larger trials => smaller variance.

    returns:
        Empirical mean = (1/trials) * sum_{i=1}^trials X_i
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    acc = 0.0
    for _ in range(trials):
        acc += float(random_var())
    return acc / trials


# ---------------------------------------------------------------------------
# 2.2 Indicator random variables
# ---------------------------------------------------------------------------


def indicator(condition: bool) -> int:
    """
    Indicator variable I(condition):
        1 if condition is True, else 0.

    This trivial function is conceptually important: many combinatorial
    quantities (like counts) can be written as sums of indicators, which
    makes linearity of expectation applicable even when variables are
    dependent.
    """
    return 1 if condition else 0


def count_even_numbers_using_indicators(values: Sequence[int]) -> int:
    """
    Example usage of indicator random variables:

        Let I_i = 1 if values[i] is even, else 0.
        Then the count of even numbers is sum_i I_i.

    This is both a conceptual demonstration and a practical counting method.
    """
    return sum(indicator(v % 2 == 0) for v in values)


# ---------------------------------------------------------------------------
# 2.3 Chernoff bounds (upper tail, overview implementation)
# ---------------------------------------------------------------------------


def chernoff_upper_tail(
    mu: float,
    delta: float,
    mode: str = "standard",
) -> float:
    """
    One-sided Chernoff upper tail bound for sums of independent Bernoulli RVs.

    Setup:
        - X = sum_{i=1}^n X_i where each X_i is Bernoulli(p_i)
        - mu = E[X] = sum p_i
        - For any delta > 0,
            P[X >= (1 + delta) * mu] <= bound(mu, delta)

    We implement a few standard forms:

        mode = "standard":
            For 0 < delta <= 1:
                P[X >= (1 + delta) * mu] <= exp(-delta^2 * mu / 3)
            For delta > 1:
                P[X >= (1 + delta) * mu] <= exp(-delta * mu / 3)

        mode = "multiplicative":
            General multiplicative Chernoff:
                P[X >= (1 + delta) * mu]
                <= exp(-mu * (delta^2 / (2 + delta))) for delta >= 0

    Note:
        - These bounds are not tight but very convenient.
        - mu must be > 0.
    """
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")

    if mode == "standard":
        if delta <= 1.0:
            exponent = -delta * delta * mu / 3.0
        else:
            exponent = -delta * mu / 3.0
        return math.exp(exponent)

    if mode == "multiplicative":
        # generic Chernoff form: exp(-mu * delta^2 / (2 + delta))
        exponent = -mu * (delta * delta) / (2.0 + delta)
        return math.exp(exponent)

    raise ValueError(f"unknown mode: {mode}")


# ============================================================================
# 3. NUMBER BASES AND BIT MATH
# ============================================================================

# ---------------------------------------------------------------------------
# 3.1 Base conversion
# ---------------------------------------------------------------------------

DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def int_to_base(n: int, base: int) -> str:
    """
    Convert integer n (>= 0) to a string representation in given base.

    Supported bases:
        - 2 <= base <= 36

    Implementation:
        - Uses repeated division by base (classic conversion algorithm).
        - For n == 0, returns "0".
    """
    if base < 2 or base > 36:
        raise ValueError("base must be between 2 and 36")
    if n < 0:
        raise ValueError("n must be non-negative for this simple converter")
    if n == 0:
        return "0"

    digits: List[str] = []
    while n > 0:
        n, rem = divmod(n, base)
        digits.append(DIGITS[rem])
    digits.reverse()
    return "".join(digits)


def base_to_int(s: str, base: int) -> int:
    """
    Convert string s representing an integer in base 'base' to Python int.

    Constraints:
        - 2 <= base <= 36
        - s must be non-empty and contain only valid digits for given base.
    """
    if base < 2 or base > 36:
        raise ValueError("base must be between 2 and 36")
    if not s:
        raise ValueError("input string must be non-empty")

    s = s.strip().upper()
    value = 0
    for ch in s:
        if ch not in DIGITS[:base]:
            raise ValueError(f"invalid digit '{ch}' for base {base}")
        value = value * base + DIGITS.index(ch)
    return value


# ---------------------------------------------------------------------------
# 3.2 Bitwise operations and masks
# ---------------------------------------------------------------------------


def is_bit_set(mask: int, bit: int) -> bool:
    """
    Test whether 'bit' (0-based) is set in 'mask'.

    Usage:
        - In subsets DP, we encode a subset S of {0..n-1} as an integer mask.
        - Bit i is 1 iff i in S.
    """
    if bit < 0:
        raise ValueError("bit index must be non-negative")
    return (mask >> bit) & 1 == 1


def set_bit(mask: int, bit: int) -> int:
    """
    Set (turn to 1) the specified bit in 'mask'.
    """
    if bit < 0:
        raise ValueError("bit index must be non-negative")
    return mask | (1 << bit)


def clear_bit(mask: int, bit: int) -> int:
    """
    Clear (turn to 0) the specified bit in 'mask'.
    """
    if bit < 0:
        raise ValueError("bit index must be non-negative")
    return mask & ~(1 << bit)


def toggle_bit(mask: int, bit: int) -> int:
    """
    Toggle (flip) the specified bit in 'mask'.
    """
    if bit < 0:
        raise ValueError("bit index must be non-negative")
    return mask ^ (1 << bit)


def popcount(mask: int) -> int:
    """
    Count the number of set bits (Hamming weight) of 'mask'.

    Implementation:
        - Uses Python's built-in int.bit_count() which is highly optimized.
    """
    if mask < 0:
        # Treat as infinite two's-complement if negative; but in DSA we
        # typically only use non-negative masks. Enforce that constraint.
        raise ValueError("mask must be non-negative")
    return mask.bit_count()


def popcount_kernighan(mask: int) -> int:
    """
    Alternative popcount via Brian Kernighan's algorithm:

        Repeatedly clear the lowest set bit:
            mask &= mask - 1

        Number of iterations equals number of set bits.

    Complexity:
        O(popcount(mask)).
    """
    if mask < 0:
        raise ValueError("mask must be non-negative")
    count = 0
    while mask:
        mask &= mask - 1
        count += 1
    return count


def iterate_submasks(mask: int) -> Iterable[int]:
    """
    Generate all submasks of a given mask, including 0 and mask itself.

    Standard pattern:
        sub = mask
        while True:
            ...
            if sub == 0: break
            sub = (sub - 1) & mask

    Complexity:
        O(3^k) in worst-case if used for all masks, but for a fixed mask with
        k bits set, there are exactly 2^k submasks.
    """
    if mask < 0:
        raise ValueError("mask must be non-negative")
    sub = mask
    # Note: yields mask, then all non-zero submasks, then finally 0
    while True:
        yield sub
        if sub == 0:
            break
        sub = (sub - 1) & mask


# ---------------------------------------------------------------------------
# 3.3 Gray codes
# ---------------------------------------------------------------------------


def gray_code(n: int) -> int:
    """
    Compute n-th Gray code using binary-reflected Gray code formula:

        g(n) = n ^ (n >> 1)

    Properties:
        - Consecutive Gray codes differ in exactly one bit.
        - In DSA, Gray codes can be used for certain DP or state enumeration
          tasks where single-bit changes are desirable.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return n ^ (n >> 1)


def gray_code_sequence(n_bits: int) -> List[int]:
    """
    Generate full Gray code sequence of length 2^n_bits.

    Returns:
        List of integers, each in [0, 2^n_bits - 1], where consecutive entries
        differ in exactly one bit.

    Complexity:
        O(2^n_bits).
    """
    if n_bits < 0:
        raise ValueError("n_bits must be non-negative")
    return [gray_code(i) for i in range(1 << n_bits)]


def gray_to_binary(g: int) -> int:
    """
    Convert Gray code 'g' back to binary representation.

    Algorithm:
        - Let b be binary, then:
            b_0 = g_0
            b_i = b_{i-1} XOR g_i
        - Implemented efficiently using repeated shifts and XORs.
    """
    if g < 0:
        raise ValueError("g must be non-negative")
    b = g
    shift = 1
    # Repeatedly XOR with right-shifted versions until mask is zero.
    while (g >> shift) > 0:
        b ^= g >> shift
        shift += 1
    return b


# ============================================================================
# 4. MODULAR ARITHMETIC AND RELATED TOOLS
# ============================================================================

# ---------------------------------------------------------------------------
# 4.1 Basic modular operations
# ---------------------------------------------------------------------------


def mod_add(a: int, b: int, mod: int) -> int:
    """
    Safe modular addition: (a + b) mod mod.

    Guarantees:
        - Handles negative inputs by normalizing the result to [0, mod-1].
    """
    if mod <= 0:
        raise ValueError("mod must be positive")
    return (a % mod + b % mod) % mod


def mod_sub(a: int, b: int, mod: int) -> int:
    """
    Safe modular subtraction: (a - b) mod mod.
    """
    if mod <= 0:
        raise ValueError("mod must be positive")
    return (a % mod - b % mod) % mod


def mod_mul(a: int, b: int, mod: int) -> int:
    """
    Safe modular multiplication: (a * b) mod mod.

    Notes:
        - Python's big integers prevent overflow, but for languages with
          fixed-width integers, you'd use 128-bit intermediate or long
          multiplication methods.
    """
    if mod <= 0:
        raise ValueError("mod must be positive")
    return (a % mod) * (b % mod) % mod


def mod_pow(base: int, exponent: int, mod: int) -> int:
    """
    Fast modular exponentiation via exponentiation by squaring:

        Compute base^exponent mod mod in O(log exponent) time.

    Pre-conditions:
        - exponent >= 0
        - mod > 0

    Note:
        - Python's built-in pow(base, exponent, mod) already implements this,
          but we write it explicitly for clarity.
    """
    if mod <= 0:
        raise ValueError("mod must be positive")
    if exponent < 0:
        raise ValueError("exponent must be non-negative")

    base %= mod
    result = 1
    e = exponent
    while e > 0:
        if e & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        e >>= 1
    return result


# ---------------------------------------------------------------------------
# 4.2 Extended Euclidean algorithm and modular inverse
# ---------------------------------------------------------------------------


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm.

    Returns:
        (g, x, y) such that:
            g = gcd(a, b) and ax + by = g

    Complexity:
        O(log min(a, b)) time.

    Notes:
        - Works for negative inputs as well, but gcd is always non-negative.
        - Critical for computing modular inverses and solving Diophantine equations.
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    # At this point, old_r = gcd(a, b), and old_s, old_t are the coefficients.
    return old_r, old_s, old_t


def mod_inverse(a: int, mod: int) -> int:
    """
    Compute modular inverse of 'a' modulo 'mod', if it exists.

    Definition:
        x is a modular inverse of a mod m if:
            a * x ≡ 1 (mod m)

    Pre-conditions:
        - mod > 1
    Existence:
        - Inverse exists iff gcd(a, mod) == 1.

    Implementation:
        - Uses extended Euclidean algorithm for general mod (not necessarily prime).
        - For prime mod, Fermat's little theorem also works:
            a^{mod-2} mod mod
    """
    if mod <= 1:
        raise ValueError("mod must be > 1")
    g, x, _ = extended_gcd(a, mod)
    if g != 1 and g != -1:
        raise ValueError(f"modular inverse does not exist since gcd({a}, {mod}) = {g}")
    # Normalize inverse to [0, mod-1]
    inv = x % mod
    return inv


def mod_inverse_fermat(a: int, mod: int) -> int:
    """
    Modular inverse via Fermat's little theorem, assuming mod is prime:

        a^{mod-2} ≡ a^{-1} (mod mod)

    Pre-conditions:
        - mod must be prime
        - gcd(a, mod) == 1
    """
    if mod <= 1:
        raise ValueError("mod must be > 1 and typically prime")
    # pow with third argument uses fast modular exponentiation
    return pow(a, mod - 2, mod)


# ---------------------------------------------------------------------------
# 4.3 Chinese Remainder Theorem (CRT)
# ---------------------------------------------------------------------------


def crt_pair(a1: int, m1: int, a2: int, m2: int) -> Tuple[int, int]:
    """
    Solve a pair of congruences:

        x ≡ a1 (mod m1)
        x ≡ a2 (mod m2)

    Returns:
        (x, lcm) where:
            - x is one solution in [0, lcm-1]
            - lcm = lcm(m1, m2) is the combined modulus

    Existence:
        - A solution exists iff a1 ≡ a2 (mod gcd(m1, m2)).

    Implementation:
        - Use extended_gcd(m1, m2) to handle non-coprime moduli.
        - Let g = gcd(m1, m2), and write:
            m1 * p + m2 * q = g
          We need k such that:
            a1 + k * m1 ≡ a2 (mod m2)
          => k * m1 ≡ a2 - a1 (mod m2)
          => (a2 - a1) must be divisible by g.
          => reduce to:
              (m1/g) * k ≡ (a2 - a1)/g (mod m2/g)
    """
    if m1 <= 0 or m2 <= 0:
        raise ValueError("moduli must be positive")

    g, p, _ = extended_gcd(m1, m2)
    diff = a2 - a1
    if diff % g != 0:
        raise ValueError("no solution exists for the given congruences")

    # Reduce problem modulo m2/g
    m2_g = m2 // g
    k = (diff // g) * (p % m2_g) % m2_g

    x = a1 + k * m1
    lcm = m1 // g * m2  # lcm(m1, m2)
    x %= lcm
    return x, lcm


def crt_list(remainders: Sequence[int], moduli: Sequence[int]) -> Tuple[int, int]:
    """
    Solve a system of congruences:

        x ≡ remainders[i] (mod moduli[i]) for all i

    Returns:
        (x, M) where M is the combined modulus (LCM of all moduli),
        and x is one solution in [0, M-1].

    Implementation:
        - Reduce to repeated combination via crt_pair.

    Complexity:
        O(k * log^2 M) where k = number of congruences, dominated by gcd and
        modulus operations.
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    if not remainders:
        raise ValueError("at least one congruence required")

    x = remainders[0] % moduli[0]
    m = moduli[0]
    for a_i, m_i in zip(remainders[1:], moduli[1:]):
        x, m = crt_pair(x, m, a_i, m_i)
    return x, m


# ============================================================================
# 5. LINEAR ALGEBRA TOOLS (FOR ALGORITHMS)
# ============================================================================

# ---------------------------------------------------------------------------
# 5.1 Vector and matrix utilities
# ---------------------------------------------------------------------------


def dot(u: Sequence[float], v: Sequence[float]) -> float:
    """
    Compute dot product of two vectors u, v of same length:

        dot(u, v) = sum_i u[i] * v[i]

    Complexity:
        O(n).
    """
    if len(u) != len(v):
        raise ValueError("vectors must have the same length")
    return float(sum(ux * vx for ux, vx in zip(u, v)))


def matmul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> List[List[float]]:
    """
    Matrix multiplication C = A * B.

    Dimensions:
        - A: m x n
        - B: n x p
        - C: m x p

    Complexity:
        O(m * n * p), which is cubic in the generic dense case.

    Pre-conditions:
        - len(A) > 0 and len(B) > 0
        - inner dimensions must match: len(A[0]) == len(B)
    """
    if not A or not B:
        raise ValueError("matrices must be non-empty")

    m = len(A)
    n = len(A[0])
    nB = len(B)
    p = len(B[0])

    if n != nB:
        raise ValueError("inner matrix dimensions must match")

    for row in A:
        if len(row) != n:
            raise ValueError("matrix A must be rectangular")
    for row in B:
        if len(row) != p:
            raise ValueError("matrix B must be rectangular")

    # Initialize result with zeros
    C: List[List[float]] = [[0.0] * p for _ in range(m)]

    # Naive triple-loop multiplication
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(p):
                C[i][j] += aik * B[k][j]

    return C


def mat_vec_mul(A: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    """
    Matrix-vector multiplication y = A * x.

    Dimensions:
        - A: m x n
        - x: n
        - y: m

    Complexity:
        O(m * n).
    """
    if not A:
        raise ValueError("matrix A must be non-empty")
    n = len(A[0])
    if len(x) != n:
        raise ValueError("dimension mismatch: A is m x n, x must be length n")
    for row in A:
        if len(row) != n:
            raise ValueError("matrix A must be rectangular")

    return [dot(row, x) for row in A]


# ---------------------------------------------------------------------------
# 5.2 Gaussian elimination over reals
# ---------------------------------------------------------------------------


def gaussian_elimination_solve(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    eps: float = 1e-12,
) -> List[float]:
    """
    Solve linear system A x = b using Gaussian elimination with partial pivoting.

    Dimensions:
        - A: n x n (square) matrix
        - b: length n vector

    Returns:
        - x: solution vector of length n

    Behavior:
        - Raises ValueError if the matrix is singular or nearly singular.
        - Uses partial pivoting to improve numerical stability.

    Complexity:
        O(n^3).
    """
    n = len(A)
    if n == 0:
        raise ValueError("matrix A must be non-empty")
    if len(b) != n:
        raise ValueError("dimension mismatch between A and b")

    # Make a working copy (augmented matrix) to avoid mutating inputs
    aug: List[List[float]] = [list(row) + [b_i] for row, b_i in zip(A, b)]

    # Validate rectangular
    for row in aug:
        if len(row) != n + 1:
            raise ValueError("matrix A must be rectangular and match b length")

    # Forward elimination
    for col in range(n):
        # Pivot selection: find row with largest |aug[row][col]| for row >= col
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < eps:
            raise ValueError("matrix is singular or nearly singular")

        # Swap current row with pivot_row, if needed
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Eliminate rows below
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            if abs(factor) < eps:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back substitution
    x = [0.0] * n
    for i in reversed(range(n)):
        # Compute RHS - sum_{j>i} A[i][j] * x[j]
        rhs = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
        diag = aug[i][i]
        if abs(diag) < eps:
            raise ValueError("matrix is singular in back-substitution phase")
        x[i] = rhs / diag

    return x


# ---------------------------------------------------------------------------
# 5.3 Gaussian elimination modulo a prime
# ---------------------------------------------------------------------------


def gaussian_elimination_mod_prime(
    A: Sequence[Sequence[int]],
    b: Sequence[int],
    mod: int,
) -> List[int]:
    """
    Solve linear system A x = b over Z_mod (integers modulo a prime 'mod').

    Dimensions:
        - A: n x n matrix of ints
        - b: length n vector of ints

    Returns:
        - x: list of ints representing the solution modulo 'mod'

    Pre-conditions:
        - mod must be prime (so that any non-zero element has an inverse).
        - System must have a unique solution; otherwise raises ValueError.

    Implementation:
        - Standard Gaussian elimination with modular arithmetic and modular
          inverses instead of division.
    """
    if mod <= 1:
        raise ValueError("mod must be > 1 and typically prime")

    n = len(A)
    if n == 0:
        raise ValueError("matrix A must be non-empty")
    if len(b) != n:
        raise ValueError("dimension mismatch between A and b")

    aug: List[List[int]] = [list(row) + [b_i] for row, b_i in zip(A, b)]
    for row in aug:
        if len(row) != n + 1:
            raise ValueError("matrix A must be square and match b length")

    # Forward elimination
    row = 0
    for col in range(n):
        # Find pivot row with non-zero entry in this column
        pivot_row = None
        for r in range(row, n):
            if aug[r][col] % mod != 0:
                pivot_row = r
                break
        if pivot_row is None:
            # No pivot in this column; move to next column
            continue

        # Swap rows
        if pivot_row != row:
            aug[row], aug[pivot_row] = aug[pivot_row], aug[row]

        # Normalize pivot row: make pivot 1 using modular inverse
        pivot_val = aug[row][col] % mod
        inv_pivot = mod_inverse(pivot_val, mod)
        for j in range(col, n + 1):
            aug[row][j] = (aug[row][j] * inv_pivot) % mod

        # Eliminate other rows
        for r in range(n):
            if r == row:
                continue
            factor = aug[r][col] % mod
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] = (aug[r][j] - factor * aug[row][j]) % mod

        row += 1
        if row == n:
            break

    # Check for inconsistency (0 = non-zero)
    for r in range(n):
        all_zero = all(aug[r][c] % mod == 0 for c in range(n))
        rhs = aug[r][n] % mod
        if all_zero and rhs != 0:
            raise ValueError("system has no solution")

    # Extract solution: system should now be in reduced row-echelon form
    x = [0] * n
    for i in range(n):
        # Find pivot in row i
        pivot_col = None
        for j in range(n):
            if aug[i][j] % mod != 0:
                pivot_col = j
                break
        if pivot_col is not None:
            x[pivot_col] = aug[i][n] % mod

    return x


# ============================================================================
# Example usage (simple self-check, not exhaustive)
# This block is intentionally minimal; in production you would write unit tests.
# ============================================================================

if __name__ == "__main__":
    # Discrete math: combinatorics sanity check
    assert factorial(5) == 120
    assert binomial_coefficient(5, 2) == 10

    # Probability: simple Bernoulli distribution
    vals = [0.0, 1.0]
    probs = [0.7, 0.3]
    ex = expectation(vals, probs)
    var = variance(vals, probs)
    print("E[X] for Bernoulli(0.3):", ex)
    print("Var[X] for Bernoulli(0.3):", var)

    # Bit math: submask enumeration
    mask = 0b1011  # subset {0,1,3}
    print("Submasks of 0b1011:")
    for sub in iterate_submasks(mask):
        print(f"{sub:04b}")

    # Modular arithmetic: CRT example
    x, m = crt_list([2, 3, 2], [3, 5, 7])  # classic example => x = 23 mod 105
    print("CRT solution x ≡ 2 (mod 3), 3 (mod 5), 2 (mod 7):", x, "mod", m)

    # Linear algebra: real Gaussian elimination
    A_real = [
        [2.0, 1.0],
        [5.0, 7.0],
    ]
    b_real = [11.0, 13.0]
    x_real = gaussian_elimination_solve(A_real, b_real)
    print("Solution of A x = b over reals:", x_real)

    # Linear algebra modulo prime
    A_mod = [
        [1, 1],
        [1, 2],
    ]
    b_mod = [3, 5]
    x_mod = gaussian_elimination_mod_prime(A_mod, b_mod, mod=7)
    print("Solution of A x = b over mod 7:", x_mod)