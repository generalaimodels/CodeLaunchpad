#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================================
# High-performance array search algorithms implemented in Python.
# Every algorithm is implemented in a clean, standards-compliant
# style with detailed inline comments describing:
#   - Preconditions / assumptions
#   - Behavior and return values
#   - Complexity
#   - Edge cases and failure modes
#
# At the end of the file a simple test harness generates dummy
# inputs and exercises each algorithm.
# ================================================================

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple


# ================================================================
# 1. LINEAR SEARCH
# ================================================================

def linear_search(arr: Sequence[Any], target: Any) -> int:
    """
    Linear search on a generic sequence.

    Returns:
        Index of the first occurrence of target, or -1 if not found.

    Complexity:
        Time  : O(n)
        Space : O(1)

    Notes:
        - Works for any sequence type and any comparable element type.
        - Does not assume sorting.
        - On duplicate elements, returns the first index.
        - On empty sequence, returns -1.
    """
    for index, value in enumerate(arr):
        if value == target:
            return index
    return -1


# ================================================================
# 2. BINARY SEARCH (STANDARD)
# ================================================================

def binary_search(arr: Sequence[Any], target: Any) -> int:
    """
    Classic binary search on a sorted sequence.

    Preconditions:
        - arr must be sorted in non-decreasing order with respect to target.

    Returns:
        Index of one occurrence of target, or -1 if not found.

    Complexity:
        Time  : O(log n)
        Space : O(1)

    Notes:
        - If duplicates exist, which occurrence is returned is unspecified
          (depends on comparisons during search).
        - For lower/upper bound semantics, dedicated variants are provided.
    """
    left: int = 0
    right: int = len(arr) - 1

    while left <= right:
        # Compute mid in overflow-safe way (overflow is not an issue in Python,
        # but this pattern is standard and language-agnostic).
        mid: int = left + (right - left) // 2
        mid_val = arr[mid]

        if mid_val == target:
            return mid
        if mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# ================================================================
# 3. JUMP SEARCH
# ================================================================

def jump_search(arr: Sequence[Any], target: Any) -> int:
    """
    Jump search on a sorted sequence.

    Idea:
        - Jump ahead in fixed-size steps (≈ sqrt(n)) until the block where
          target could reside is found, then perform linear search inside.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(√n)
        Space : O(1)

    Notes:
        - Performs best when random access is cheap and arr is large.
        - For very large arrays, binary search is usually better in practice,
          but jump search can be useful for certain memory models.
    """
    n: int = len(arr)
    if n == 0: return -1

    step: int = int(math.sqrt(n)) or 1  # Ensure step >= 1
    prev: int = 0

    # Jump in blocks until we overshoot or find a block where arr[block_end] >= target.
    while prev < n and arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n)) or 1
        if prev >= n:
            return -1

    # Linear search within the identified block.
    for index in range(prev, min(step, n)):
        if arr[index] == target:
            return index

    return -1


# ================================================================
# 4. INTERPOLATION SEARCH
# ================================================================

def interpolation_search(arr: Sequence[int], target: int) -> int:
    """
    Interpolation search on a sorted, approximately uniformly distributed array of integers.

    Idea:
        - Instead of probing the middle (as in binary search), estimate
          the likely position of target using linear interpolation.

    Preconditions:
        - arr must be sorted in non-decreasing order.
        - arr elements and target must be integers (or at least support
          subtraction / division / addition in the same arithmetic domain).
        - Best suited for approximately uniform distributions.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Average : O(log log n) for uniform distributions.
        Worst   : O(n) (e.g., when distribution is highly skewed).

    Notes:
        - Handles degenerate case where arr[low] == arr[high] by falling back
          to direct comparison.
    """
    n: int = len(arr)
    if n == 0: return -1

    low: int = 0
    high: int = n - 1

    while low <= high and arr[low] <= target <= arr[high]:
        # Avoid division by zero when all elements in the range are equal.
        if arr[high] == arr[low]:
            if arr[low] == target:
                return low
            return -1

        # Interpolation formula:
        # pos = low + (target - arr[low]) * (high - low) / (arr[high] - arr[low])
        pos_float = low + (target - arr[low]) * (high - low) / (arr[high] - arr[low])
        pos: int = int(pos_float)

        # Clamp pos to valid range to protect against numerical issues.
        if pos < low:
            pos = low
        elif pos > high:
            pos = high

        current = arr[pos]

        if current == target:
            return pos
        if current < target:
            low = pos + 1
        else:
            high = pos - 1

    return -1


# ================================================================
# 5. EXPONENTIAL SEARCH
# ================================================================

def exponential_search(arr: Sequence[Any], target: Any) -> int:
    """
    Exponential search on a sorted array.

    Idea:
        - Quickly find a range [bound/2, bound] that may contain the target by
          exponentially increasing the bound (1, 2, 4, 8, ...).
        - Then perform binary search inside that range.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log pos) where pos is position of target.
        Space : O(1)

    Notes:
        - Especially useful when the array is conceptually unbounded or when
          we only know that target is not too far from the beginning.
    """
    n: int = len(arr)
    if n == 0: return -1

    if arr[0] == target: return 0 

    # Find range for binary search by repeated doubling.
    bound: int = 1
    while bound < n and arr[bound] < target:
        bound *= 2

    left: int = bound // 2
    right: int = min(bound, n - 1)

    # Reuse standard binary search logic on the identified range.
    while left <= right:
        mid: int = left + (right - left) // 2
        mid_val = arr[mid]
        if mid_val == target:
            return mid
        if mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# ================================================================
# 6. FIBONACCI SEARCH
# ================================================================

def fibonacci_search(arr: Sequence[Any], target: Any) -> int:
    """
    Fibonacci search on a sorted array.

    Idea:
        - Use Fibonacci numbers to divide the array into sections similar to
          binary search, potentially optimizing cache behavior on some systems.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log n)
        Space : O(1)

    Notes:
        - Uses a decreasing Fibonacci sequence to locate the target.
    """
    n: int = len(arr)
    if n == 0: return -1 

    # Initialize Fibonacci numbers F(m-2), F(m-1), F(m) such that F(m) >= n.
    fib_mm2: int = 0  # F(m-2)
    fib_mm1: int = 1  # F(m-1)
    fib_m: int = fib_mm1 + fib_mm2  # F(m)

    while fib_m < n:
        fib_mm2, fib_mm1 = fib_mm1, fib_m 
        fib_m = fib_mm1 + fib_mm2 

    offset: int = -1  # Eliminated range from front

    while fib_m > 1:
        # Calculate index i. Ensure it stays within bounds.
        i: int = min(offset + fib_mm2, n - 1) 

        if arr[i] < target:
            # Move three Fibonacci variables down one.
            fib_m = fib_mm1
            fib_mm1 = fib_mm2
            fib_mm2 = fib_m - fib_mm1
            offset = i
        elif arr[i] > target:
            # Move two Fibonacci variables down two.
            fib_m = fib_mm2
            fib_mm1 = fib_mm1 - fib_mm2
            fib_mm2 = fib_m - fib_mm1
        else:
            return i

    # Check if the last possible element is target.
    if fib_mm1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1

    return -1


# ================================================================
# 7. TERNARY SEARCH (DISCRETE VARIANT ON SORTED ARRAY)
# ================================================================

def ternary_search(arr: Sequence[Any], target: Any) -> int:
    """
    Ternary search on a sorted array for equality.

    Idea:
        - Split the search interval into three parts using two midpoints.
        - This is suboptimal compared to binary search for equality search,
          but is shown here as a theoretical variant.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log n) comparisons (with larger constant factor than binary).
        Space : O(1)

    Notes:
        - Ternary search is more commonly used for optimizing unimodal
          functions; its use for equality search is mostly academic.
    """
    left: int = 0
    right: int = len(arr) - 1

    while left <= right:
        # Compute two midpoints.
        third: int = (right - left) // 3
        mid1: int = left + third
        mid2: int = right - third

        val1 = arr[mid1]
        val2 = arr[mid2]

        if val1 == target:
            return mid1
        if val2 == target:
            return mid2

        if target < val1:
            right = mid1 - 1
        elif target > val2:
            left = mid2 + 1
        else:
            left = mid1 + 1
            right = mid2 - 1

    return -1


# ================================================================
# 8. META BINARY SEARCH
# ================================================================

def meta_binary_search(arr: Sequence[Any], target: Any) -> int:
    """
    Meta binary search using bitwise decomposition of the index.

    Idea:
        - Instead of adjusting left/right bounds, build the answer index bit
          by bit, starting from the most significant bit of the highest index.
        - At each bit, we tentatively set it and check if the item at that
          index is <= target; if so, we keep that bit.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log n)
        Space : O(1)

    Notes:
        - Effectively equivalent to lower_bound + equality check.
    """
    n: int = len(arr)
    if n == 0:
        return -1

    # Highest index we may reach.
    hi_index: int = n - 1
    # Most significant bit needed to represent hi_index.
    bit: int = 1 << (hi_index.bit_length() - 1)

    idx: int = 0

    # Build idx bit-by-bit.
    while bit > 0:
        next_idx: int = idx | bit
        if next_idx <= hi_index and arr[next_idx] <= target:
            idx = next_idx
        bit >>= 1

    # idx is now the largest index such that arr[idx] <= target (if possible).
    if arr[idx] == target:
        return idx
    return -1


# ================================================================
# 9. GOLDEN SECTION SEARCH (DISCRETE UNIMODAL ARRAY)
# ================================================================

def golden_section_search_unimodal(arr: Sequence[float],
                                   find_maximum: bool = True) -> int:
    """
    Golden-section search for extremum in a unimodal discrete array.

    Idea:
        - Apply golden-section search on index domain [0, n-1] treating arr
          as samples of a unimodal function.
        - This is used to find a (near) optimal index rather than equality.

    Preconditions:
        - arr must be unimodal (strictly increases then strictly decreases) if
          find_maximum is True.
        - Similarly, unimodal towards a minimum if find_maximum is False.
        - If these conditions do not hold, result is not guaranteed.

    Returns:
        Index of approximate maximum (or minimum) element.

    Complexity:
        Time  : O(log n) evaluations.
        Space : O(1)

    Notes:
        - For discrete unimodal arrays, a simpler method is "peak finding"
          via binary search on neighbors; golden section is shown as the
          classic continuous optimization analog.
    """
    n: int = len(arr)
    if n == 0:
        raise ValueError("golden_section_search_unimodal requires non-empty array")

    if n <= 3:
        # For very small arrays, just scan.
        best_index: int = 0
        for i in range(1, n):
            if find_maximum:
                if arr[i] > arr[best_index]:
                    best_index = i
            else:
                if arr[i] < arr[best_index]:
                    best_index = i
        return best_index

    # Golden ratio constants.
    phi: float = (1 + math.sqrt(5.0)) / 2.0
    inv_phi: float = 1.0 / phi

    left: int = 0
    right: int = n - 1

    # Initial interior points.
    c: int = right - int((right - left) * inv_phi)
    d: int = left + int((right - left) * inv_phi)

    while right - left > 3:
        val_c = arr[c]
        val_d = arr[d]
        if not find_maximum:
            # Convert minimum search to maximum by negating comparison.
            val_c = -val_c
            val_d = -val_d

        if val_c > val_d:
            # Best is in [left, d]
            right = d
            d = c
            c = right - int((right - left) * inv_phi)
        else:
            # Best is in [c, right]
            left = c
            c = d
            d = left + int((right - left) * inv_phi)

    # Final scan of small range [left, right].
    best_index = left
    for i in range(left + 1, right + 1):
        if find_maximum:
            if arr[i] > arr[best_index]:
                best_index = i
        else:
            if arr[i] < arr[best_index]:
                best_index = i
    return best_index


# ================================================================
# 10. SUBLIST SEARCH (SUBARRAY / PATTERN SEARCH, NAIVE)
# ================================================================

def sublist_search(haystack: Sequence[Any], needle: Sequence[Any]) -> int:
    """
    Naive sublist (subarray) search.

    Idea:
        - Search for the first occurrence of needle as a contiguous sub-sequence
          of haystack using direct comparison.

    Returns:
        Starting index of the first occurrence of needle, or -1 if not found.

    Complexity:
        Time  : O((n - m + 1) * m) in worst case, where n = len(haystack),
                m = len(needle).
        Space : O(1)

    Notes:
        - For large patterns and sequences, more sophisticated algorithms
          (KMP, Boyer-Moore, etc.) are preferable.
        - An empty needle is considered to match at index 0 by convention.
    """
    n: int = len(haystack)
    m: int = len(needle)

    if m == 0:
        return 0
    if m > n:
        return -1

    for start in range(n - m + 1):
        match = True
        for j in range(m):
            if haystack[start + j] != needle[j]:
                match = False
                break
        if match:
            return start

    return -1


# ================================================================
# 11. UNIFORM BINARY SEARCH (SIMPLE WRAPPER)
# ================================================================

def uniform_binary_search(arr: Sequence[Any], target: Any) -> int:
    """
    Uniform binary search.

    Idea:
        - For repeated searches on arrays of fixed size, uniform binary search
          precomputes probing offsets to avoid dynamic mid-point calculations.
        - In high-level languages, the practical benefit is minimal.

    Implementation:
        - For clarity and correctness, this function delegates to standard
          binary_search, which has the same asymptotic behavior.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log n)
        Space : O(1)
    """
    return binary_search(arr, target)


# ================================================================
# 12. GALLOPING SEARCH (a.k.a. DOUBLING / UNBOUNDED SEARCH)
# ================================================================

def galloping_search(arr: Sequence[Any], target: Any) -> int:
    """
    Galloping (doubling) search on a sorted array.

    Idea:
        - Similar to exponential search: quickly grows the search interval by
          doubling until the target is bracketed, then uses binary search.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log pos) where pos is position of target.
        Space : O(1)

    Notes:
        - Frequently used in merging routines (e.g., in TimSort) where a
          galloping search finds run boundaries efficiently.
    """
    # This implementation is functionally identical to exponential_search
    # but kept separate for conceptual clarity.
    return exponential_search(arr, target)


# ================================================================
# 13. BINARY SEARCH VARIANTS: LOWER_BOUND & UPPER_BOUND
# ================================================================

def lower_bound(arr: Sequence[Any], target: Any) -> int:
    """
    Returns the index of the first element in arr that is >= target.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index in range [0, len(arr)]:
            - If all elements < target, returns len(arr).
            - Otherwise returns smallest index i with arr[i] >= target.

    Complexity:
        Time  : O(log n)
        Space : O(1)
    """
    return bisect.bisect_left(arr, target)


def upper_bound(arr: Sequence[Any], target: Any) -> int:
    """
    Returns the index of the first element in arr that is > target.

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index in range [0, len(arr)]:
            - If all elements <= target, returns len(arr).
            - Otherwise returns smallest index i with arr[i] > target.

    Complexity:
        Time  : O(log n)
        Space : O(1)
    """
    return bisect.bisect_right(arr, target)


# ================================================================
# 14. RANDOMIZED BINARY SEARCH
# ================================================================

def randomized_binary_search(arr: Sequence[Any], target: Any) -> int:
    """
    Randomized binary search on a sorted array.

    Idea:
        - Similar to classic binary search, but instead of always picking the
          midpoint, choose a random pivot within [left, right].
        - The expected number of iterations remains O(log n).

    Preconditions:
        - arr must be sorted in non-decreasing order.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log n) expected.
        Space : O(1)
    """
    left: int = 0
    right: int = len(arr) - 1

    while left <= right:
        mid: int = random.randint(left, right)
        mid_val = arr[mid]

        if mid_val == target:
            return mid
        if mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# ================================================================
# 15. K-ARY SEARCH
# ================================================================

def k_ary_search(arr: Sequence[Any], target: Any, k: int) -> int:
    """
    K-ary search (generalization of binary search) on a sorted array.

    Idea:
        - At each iteration, split the range into k segments using k-1 pivot
          points and select the subrange where the target can reside.

    Preconditions:
        - arr must be sorted in non-decreasing order.
        - k must be an integer >= 2.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log_k n) iterations, each doing O(k) comparisons,
                so total O((k / ln k) * ln n). For k=2 this becomes standard
                binary search.

    Notes:
        - Very large k is usually counterproductive due to overhead.
    """
    if k < 2:
        raise ValueError("k_ary_search requires k >= 2")

    left: int = 0
    right: int = len(arr) - 1

    while left <= right:
        # For small ranges, switch to linear scan.
        if right - left + 1 <= k:
            for i in range(left, right + 1):
                if arr[i] == target:
                    return i
            return -1

        # Compute k-1 sample indices that partition [left, right] into k parts.
        sample_indices: List[int] = []
        for i in range(1, k):
            idx: int = left + (right - left) * i // k
            sample_indices.append(idx)

        # Determine which segment to search next.
        prev_idx: int = left - 1
        chosen_left: int = left
        chosen_right: int = right
        chosen: bool = False

        for idx in sample_indices:
            val = arr[idx]
            if val == target:
                return idx
            if target < val:
                chosen_left = prev_idx + 1
                chosen_right = idx - 1
                chosen = True
                break
            prev_idx = idx

        if not chosen:
            chosen_left = prev_idx + 1
            chosen_right = right

        left, right = chosen_left, chosen_right

    return -1


# ================================================================
# 16. FRACTIONAL CASCADING SEARCH (SIMPLIFIED INDEXING VERSION)
# ================================================================

@dataclass
class FractionalCascadingIndex:
    """
    Simplified fractional cascading structure.

    Idea:
        - Given multiple sorted arrays, precompute a shared index so that
          searching a value across all arrays can be reduced to:
              one global lookup + O(k) direct index reads.
        - This is a simpler variant retaining the high-level spirit of
          fractional cascading (cascading search information), trading
          extra space for clarity.

    Limitations:
        - The precomputed indices are exact only for keys that already appear
          in at least one of the arrays.
        - For keys not present in the union, we fall back to a per-array
          binary search (still O(k log n)).
    """
    lists: List[List[int]]
    global_values: List[int]
    value_to_row: dict
    index_table: List[List[int]]


class FractionalCascadingSearch:
    """
    Fractional Cascading search helper.

    Usage:
        fc = FractionalCascadingSearch([sorted_list1, sorted_list2, ...])
        indices = fc.search(target)
        # indices[i] is the lower_bound index of target in lists[i]
    """

    def __init__(self, lists: List[List[int]]) -> None:
        # Store sorted copies to ensure ordering.
        self.lists: List[List[int]] = [sorted(lst) for lst in lists]

        # Build union of all values.
        values_set = set()
        for lst in self.lists:
            values_set.update(lst)
        global_values: List[int] = sorted(values_set)

        value_to_row = {v: i for i, v in enumerate(global_values)}

        # Precompute lower_bound indices for each global value in each list.
        # index_table[row][col] = lower_bound index of global_values[row] in self.lists[col].
        index_table: List[List[int]] = [
            [0] * len(self.lists) for _ in range(len(global_values))
        ]
        for row, v in enumerate(global_values):
            for col, lst in enumerate(self.lists):
                index_table[row][col] = bisect.bisect_left(lst, v)

        self.index = FractionalCascadingIndex(
            lists=self.lists,
            global_values=global_values,
            value_to_row=value_to_row,
            index_table=index_table,
        )

    def search(self, target: int) -> List[int]:
        """
        Search target across all lists.

        Returns:
            list of indices idx[i] such that:
                - If target is present in lists[i], lists[i][idx[i]] == target.
                - Otherwise, idx[i] is the position where target could be inserted
                  to keep lists[i] sorted (lower_bound semantics).

        Complexity:
            - For targets in the union, one hash lookup (average O(1)) and
              O(k) array lookups.
            - For other targets, we fall back to O(k log n) binary search.
        """
        if target in self.index.value_to_row:
            row = self.index.value_to_row[target]
            return list(self.index.index_table[row])

        # Fallback: compute lower_bound independently for each list.
        return [bisect.bisect_left(lst, target) for lst in self.index.lists]


# ================================================================
# 17. SADDLEBACK SEARCH (2D MONOTONE MATRIX)
# ================================================================

def saddleback_search(matrix: List[List[int]], target: int) -> Tuple[int, int]:
    """
    Saddleback search in a 2D matrix where:
        - Each row is sorted in non-decreasing order.
        - Each column is sorted in non-decreasing order.

    Idea:
        - Start at top-right corner.
        - If current element > target, move left.
        - If current element < target, move down.

    Returns:
        (row_index, col_index) if found, otherwise (-1, -1).

    Complexity:
        Time  : O(m + n) where m = number of rows, n = number of columns.
        Space : O(1)
    """
    if not matrix or not matrix[0]:
        return -1, -1

    rows: int = len(matrix)
    cols: int = len(matrix[0])

    row: int = 0
    col: int = cols - 1

    while row < rows and col >= 0:
        value = matrix[row][col]
        if value == target:
            return row, col
        if value > target:
            col -= 1
        else:
            row += 1

    return -1, -1


# ================================================================
# 18. PIVOT BINARY SEARCH (SEARCH IN ROTATED SORTED ARRAY)
# ================================================================

def _find_rotation_pivot(arr: Sequence[Any]) -> int:
    """
    Find index of smallest element (rotation pivot) in a rotated sorted array.

    Assumes:
        - arr is sorted in non-decreasing order, then rotated.
        - Allows duplicates; in the presence of many duplicates, worst-case
          degenerates to O(n).

    Returns:
        Index of minimum element (pivot).
    """
    n: int = len(arr)
    if n == 0:
        return 0

    left: int = 0
    right: int = n - 1

    while left < right:
        mid: int = left + (right - left) // 2
        if arr[mid] > arr[right]:
            # Minimum is in (mid, right].
            left = mid + 1
        elif arr[mid] < arr[right]:
            # Minimum is in [left, mid].
            right = mid
        else:
            # arr[mid] == arr[right], cannot decide; shrink right.
            right -= 1

    return left


def _binary_search_range(arr: Sequence[Any],
                         target: Any,
                         left: int,
                         right: int) -> int:
    """
    Standard binary search restricted to [left, right] inclusive.
    """
    while left <= right:
        mid: int = left + (right - left) // 2
        val = arr[mid]
        if val == target:
            return mid
        if val < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def pivot_binary_search(arr: Sequence[Any], target: Any) -> int:
    """
    Search target in a sorted array that has been rotated.

    Example:
        arr = [4,5,6,7,0,1,2]
        pivot_binary_search(arr, 0) -> 4

    Preconditions:
        - arr is a rotated version of a non-decreasing sorted array.
        - Duplicates are allowed; pivot detection may degrade to O(n) in
          worst-case (when many equal elements).

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Average:
            - O(log n) for pivot detection.
            - O(log n) for binary search.
        Worst (with many duplicates):
            - O(n).
    """
    n: int = len(arr)
    if n == 0:
        return -1

    pivot: int = _find_rotation_pivot(arr)

    # If array is not rotated (pivot == 0), just use standard binary search.
    if pivot == 0:
        return binary_search(arr, target)

    # Decide which half to search: [0, pivot-1] or [pivot, n-1].
    if arr[pivot] <= target <= arr[n - 1]:
        return _binary_search_range(arr, target, pivot, n - 1)
    return _binary_search_range(arr, target, 0, pivot - 1)


# ================================================================
# 19. WEIGHTED BINARY SEARCH
# ================================================================

def weighted_binary_search(arr: Sequence[Any],
                           target: Any,
                           weights: Sequence[float]) -> int:
    """
    Weighted binary search on a sorted array.

    Idea:
        - Instead of splitting the interval by position (midpoint), choose a
          pivot index whose cumulative weight is closest to the median weight
          of the interval.
        - If elements with certain weights are more likely to be queried,
          this can reduce expected search cost under appropriate models.

    Preconditions:
        - arr must be sorted in non-decreasing order.
        - weights must be non-negative and have the same length as arr.

    Returns:
        Index of target, or -1 if not found.

    Complexity:
        Time  : O(log n) for index steps, with an inner binary search on
                prefix sums for each step, thus O((log n)^2) in this
                straightforward implementation.
        Space : O(n) for prefix sums.

    Notes:
        - More sophisticated implementations can reduce the inner complexity
          by using segment trees or other structures.
    """
    n: int = len(arr)
    if n == 0:
        return -1
    if len(weights) != n:
        raise ValueError("weights must have same length as arr")

    # Build prefix sums of weights for quick segment weight computations.
    prefix: List[float] = [0.0] * (n + 1)
    for i, w in enumerate(weights):
        if w < 0:
            raise ValueError("weights must be non-negative")
        prefix[i + 1] = prefix[i] + w

    left: int = 0
    right: int = n - 1

    while left <= right:
        total_weight: float = prefix[right + 1] - prefix[left]

        if total_weight <= 0:
            # Degenerate: all zero weights; fall back to regular midpoint.
            mid: int = left + (right - left) // 2
        else:
            # Target the median weight inside [left, right].
            median_weight: float = prefix[left] + total_weight / 2.0

            # Binary search on prefix to find smallest index mid with
            # prefix[mid + 1] >= median_weight.
            lo: int = left
            hi: int = right
            mid: int = left
            while lo <= hi:
                mid_candidate: int = (lo + hi) // 2
                if prefix[mid_candidate + 1] < median_weight:
                    lo = mid_candidate + 1
                else:
                    mid = mid_candidate
                    hi = mid_candidate - 1

        mid_val = arr[mid]
        if mid_val == target:
            return mid
        if mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# ================================================================
# 20. SPARSE TABLE SEARCH (RANGE MIN / MAX QUERY)
# ================================================================

class SparseTable:
    """
    Sparse table for idempotent range queries (e.g., min/max/gcd).

    Purpose:
        - Preprocess an array so that queries like "find minimum (or maximum)
          in range [L, R]" can be answered in O(1) time.

    Construction:
        - Precomputation is O(n log n).

    Usage:
        st = SparseTable(values, func=min)
        min_value = st.query(L, R)
    """

    def __init__(self, values: Sequence[Any], func: Callable[[Any, Any], Any]):
        if not values:
            raise ValueError("SparseTable requires non-empty values")

        self.values: List[Any] = list(values)
        self.func: Callable[[Any, Any], Any] = func
        n: int = len(self.values)

        # Precompute floor(log2) for 1..n.
        self.log: List[int] = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1

        K: int = self.log[n] + 1  # Number of layers needed.

        # st[k][i] holds function result over range of length 2^k starting at i.
        self.st: List[List[Any]] = [[None] * n for _ in range(K)]
        self.st[0] = list(self.values)

        # Build table bottom-up.
        for k in range(1, K):
            length = 1 << k
            half = length >> 1
            for i in range(0, n - length + 1):
                self.st[k][i] = self.func(self.st[k - 1][i],
                                          self.st[k - 1][i + half])

    def query(self, left: int, right: int) -> Any:
        """
        Query function value on range [left, right] inclusive.

        Preconditions:
            - 0 <= left <= right < len(values).

        Returns:
            func(values[left..right]).

        Complexity:
            Time  : O(1)
            Space : O(1) extra (beyond precomputed table).
        """
        if left < 0 or right >= len(self.values) or left > right:
            raise IndexError("Invalid query range")

        length: int = right - left + 1
        k: int = self.log[length]
        # Use overlapping intervals [left, left+2^k-1] and
        # [right-2^k+1, right] to cover [left, right].
        return self.func(
            self.st[k][left],
            self.st[k][right - (1 << k) + 1],
        )


def sparse_table_range_min_search(values: Sequence[int],
                                  left: int,
                                  right: int) -> int:
    """
    Search for the minimum value in values[left..right] using a sparse table.

    Returns:
        The minimum value in the given range.

    Complexity:
        Preprocessing (to build sparse table): O(n log n)
        Query: O(1)
    """
    st = SparseTable(values, func=min)
    return st.query(left, right)


# ================================================================
# DUMMY TEST INPUT GENERATION AND DEMONSTRATION
# ================================================================

def _generate_sorted_array(size: int,
                           start: int = 0,
                           step_range: Tuple[int, int] = (1, 5)) -> List[int]:
    """
    Generate a sorted array of given size where differences between consecutive
    elements are random in the given step_range.
    """
    arr: List[int] = []
    current: int = start
    for _ in range(size):
        current += random.randint(*step_range)
        arr.append(current)
    return arr


def main() -> None:
    # Seed for reproducibility.
    random.seed(42)

    # ------------------------------------------------------------
    # Base arrays for 1D search algorithms.
    # ------------------------------------------------------------
    sorted_arr: List[int] = _generate_sorted_array(size=20, start=0)
    target_existing: int = sorted_arr[len(sorted_arr) // 2]
    target_missing: int = sorted_arr[-1] + 10

    print("=== 1D Sort-based Searches ===")
    print("Array:", sorted_arr)
    print("Existing target:", target_existing)
    print("Missing target:", target_missing)

    print("Linear Search (existing):", linear_search(sorted_arr, target_existing))
    print("Linear Search (missing):", linear_search(sorted_arr, target_missing))

    print("Binary Search (existing):", binary_search(sorted_arr, target_existing))
    print("Binary Search (missing):", binary_search(sorted_arr, target_missing))

    print("Jump Search (existing):", jump_search(sorted_arr, target_existing))
    print("Jump Search (missing):", jump_search(sorted_arr, target_missing))

    print("Interpolation Search (existing):",
          interpolation_search(sorted_arr, target_existing))
    print("Interpolation Search (missing):",
          interpolation_search(sorted_arr, target_missing))

    print("Exponential Search (existing):",
          exponential_search(sorted_arr, target_existing))
    print("Exponential Search (missing):",
          exponential_search(sorted_arr, target_missing))

    print("Fibonacci Search (existing):",
          fibonacci_search(sorted_arr, target_existing))
    print("Fibonacci Search (missing):",
          fibonacci_search(sorted_arr, target_missing))

    print("Ternary Search (existing):",
          ternary_search(sorted_arr, target_existing))
    print("Ternary Search (missing):",
          ternary_search(sorted_arr, target_missing))

    print("Meta Binary Search (existing):",
          meta_binary_search(sorted_arr, target_existing))
    print("Meta Binary Search (missing):",
          meta_binary_search(sorted_arr, target_missing))

    print("Uniform Binary Search (existing):",
          uniform_binary_search(sorted_arr, target_existing))
    print("Uniform Binary Search (missing):",
          uniform_binary_search(sorted_arr, target_missing))

    print("Galloping Search (existing):",
          galloping_search(sorted_arr, target_existing))
    print("Galloping Search (missing):",
          galloping_search(sorted_arr, target_missing))

    print("Randomized Binary Search (existing):",
          randomized_binary_search(sorted_arr, target_existing))
    print("Randomized Binary Search (missing):",
          randomized_binary_search(sorted_arr, target_missing))

    print("Lower Bound of existing:", lower_bound(sorted_arr, target_existing))
    print("Upper Bound of existing:", upper_bound(sorted_arr, target_existing))

    print("K-ary Search (k=3, existing):",
          k_ary_search(sorted_arr, target_existing, k=3))
    print("K-ary Search (k=3, missing):",
          k_ary_search(sorted_arr, target_missing, k=3))

    # ------------------------------------------------------------
    # Sublist (subarray) search.
    # ------------------------------------------------------------
    haystack: List[int] = [1, 2, 3, 4, 2, 3, 4, 5]
    needle: List[int] = [2, 3, 4]
    needle_missing: List[int] = [3, 5]
    print("\n=== Sublist Search ===")
    print("Haystack:", haystack)
    print("Needle:", needle)
    print("Sublist Search (present):", sublist_search(haystack, needle))
    print("Sublist Search (missing):", sublist_search(haystack, needle_missing))

    # ------------------------------------------------------------
    # Golden-section search on unimodal array (search for maximum).
    # ------------------------------------------------------------
    unimodal_arr: List[float] = [1.0, 3.0, 8.0, 12.0, 11.0, 7.0, 2.0]
    max_index: int = golden_section_search_unimodal(unimodal_arr, find_maximum=True)
    print("\n=== Golden Section Search (Unimodal) ===")
    print("Unimodal array:", unimodal_arr)
    print("Approximate maximum at index:", max_index,
          "value:", unimodal_arr[max_index])

    # ------------------------------------------------------------
    # Fractional Cascading Search over multiple sorted arrays.
    # ------------------------------------------------------------
    multi_lists: List[List[int]] = [
        _generate_sorted_array(10, start=0),
        _generate_sorted_array(12, start=3),
        _generate_sorted_array(8, start=5),
    ]
    fc_target: int = multi_lists[1][4]  # pick a target guaranteed in second list
    fc = FractionalCascadingSearch(multi_lists)
    fc_indices = fc.search(fc_target)
    print("\n=== Fractional Cascading Search (Simplified) ===")
    print("Lists:")
    for idx, lst in enumerate(multi_lists):
        print(f"List {idx}:", lst)
    print("Target:", fc_target)
    print("Lower-bound indices in each list:", fc_indices)

    # ------------------------------------------------------------
    # Saddleback search in 2D matrix.
    # ------------------------------------------------------------
    matrix: List[List[int]] = [
        [1, 4, 7, 11],
        [2, 5, 8, 12],
        [3, 6, 9, 16],
        [10, 13, 14, 17],
    ]
    saddle_target: int = 9
    saddle_missing: int = 15
    print("\n=== Saddleback Search (2D matrix) ===")
    print("Matrix:")
    for row in matrix:
        print(row)
    print("Saddleback Search (existing):",
          saddleback_search(matrix, saddle_target))
    print("Saddleback Search (missing):",
          saddleback_search(matrix, saddle_missing))

    # ------------------------------------------------------------
    # Pivot binary search in a rotated sorted array.
    # ------------------------------------------------------------
    base_sorted: List[int] = sorted_arr[:]
    pivot_index: int = 7
    rotated_arr: List[int] = base_sorted[pivot_index:] + base_sorted[:pivot_index]
    pivot_target_existing: int = base_sorted[3]
    pivot_target_missing: int = base_sorted[-1] + 20
    print("\n=== Pivot Binary Search (Rotated Array) ===")
    print("Original sorted:", base_sorted)
    print("Rotated array :", rotated_arr)
    print("Pivot Binary Search (existing):",
          pivot_binary_search(rotated_arr, pivot_target_existing))
    print("Pivot Binary Search (missing):",
          pivot_binary_search(rotated_arr, pivot_target_missing))

    # ------------------------------------------------------------
    # Weighted binary search demonstration.
    # ------------------------------------------------------------
    weights: List[float] = [random.random() + 0.1 for _ in sorted_arr]
    wbs_target_existing: int = target_existing
    wbs_target_missing: int = target_missing
    print("\n=== Weighted Binary Search ===")
    print("Weights:", [round(w, 2) for w in weights])
    print("Weighted Binary Search (existing):",
          weighted_binary_search(sorted_arr, wbs_target_existing, weights))
    print("Weighted Binary Search (missing):",
          weighted_binary_search(sorted_arr, wbs_target_missing, weights))

    # ------------------------------------------------------------
    # Sparse table search: range minimum query.
    # ------------------------------------------------------------
    rmq_arr: List[int] = [random.randint(0, 100) for _ in range(15)]
    L, R = 3, 10
    rmq_min = sparse_table_range_min_search(rmq_arr, L, R)
    print("\n=== Sparse Table Range-Min Search ===")
    print("Array:", rmq_arr)
    print(f"Range [{L}, {R}] minimum value:", rmq_min)


if __name__ == "__main__":
    main()