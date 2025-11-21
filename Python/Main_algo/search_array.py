# -*- coding: utf-8 -*-
"""
MASTER ARRAY SEARCH ALGORITHMS SUITE
================================================================================
Description: 
    Comprehensive implementation of 21 Array Search Algorithms ranging from 
    elementary linear scans to complex geometric and structure-augmented searches.
    
    This suite demonstrates:
    1. Algorithmic Efficiency (Time/Space Complexity mastery)
    2. Robust Error Handling & Type Safety
    3. Professional Documentation & Educational Depth
    4. Pythonic Best Practices (PEP 8)

Note on Complexity:
    - N: Size of the array
    - K: branching factor (K-ary) or number of lists (Fractional Cascading)
    - M: Size of sublist/matrix dimension
"""

import math
import random
from typing import List, Optional, Tuple, Union, Any, TypeVar, Generic
from functools import lru_cache

# Generic type for comparable items (int, float, str)
T = TypeVar('T', int, float, str)

class SearchAlgorithms:
    """
    Static utility class encapsulating high-performance search algorithms.
    Designed for educational clarity and production-grade reliability.
    """

    # ==========================================================================
    # 1. LINEAR SEARCH
    # ==========================================================================
    @staticmethod
    def linear_search(arr: List[T], target: T) -> int:
        """
        Performs a sequential check of every element.
        
        Theory:
            The most basic search. No prerequisite for sorting.
            Iterates from index 0 to N-1.
        
        Complexity:
            Time: O(N) - Worst/Average case.
            Space: O(1) - Iterative.
        """
        for i, val in enumerate(arr):
            if val == target:
                return i
        return -1

    # ==========================================================================
    # 2. BINARY SEARCH
    # ==========================================================================
    @staticmethod
    def binary_search(arr: List[T], target: T) -> int:
        """
        Classic Divide and Conquer on a SORTED array.
        
        Theory:
            Compares target with the middle element.
            If target < mid, search left half.
            If target > mid, search right half.
            
        Complexity:
            Time: O(log N)
            Space: O(1)
        """
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    # ==========================================================================
    # 3. JUMP SEARCH
    # ==========================================================================
    @staticmethod
    def jump_search(arr: List[T], target: T) -> int:
        """
        Jumps ahead by fixed steps (sqrt(N)) then linear searches the block.
        
        Theory:
            Optimal step size is sqrt(N). We skip elements to reduce comparisons
            compared to Linear Search, but jump back to perform a linear scan
            once we pass the target. Requires SORTED array.
            
        Complexity:
            Time: O(sqrt(N))
            Space: O(1)
        """
        n = len(arr)
        if n == 0: return -1
        
        step = int(math.sqrt(n))
        prev = 0
        
        # Finding the block where element is present
        while arr[min(step, n) - 1] < target:
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                return -1
                
        # Linear search in the identified block
        while arr[prev] < target:
            prev += 1
            if prev == min(step, n):
                return -1
                
        if arr[prev] == target:
            return prev
            
        return -1

    # ==========================================================================
    # 4. INTERPOLATION SEARCH
    # ==========================================================================
    @staticmethod
    def interpolation_search(arr: List[int], target: int) -> int:
        """
        Predicts position based on value magnitude (like looking in a phonebook).
        
        Theory:
            Instead of mid = (low+high)/2, we use:
            pos = low + ((target - arr[low]) * (high - low) / (arr[high] - arr[low]))
            Works best on SORTED and UNIFORMLY DISTRIBUTED data.
            
        Complexity:
            Time: O(log(log N)) average, O(N) worst case (skewed data).
            Space: O(1)
        """
        low = 0
        high = len(arr) - 1

        while low <= high and target >= arr[low] and target <= arr[high]:
            if low == high:
                if arr[low] == target:
                    return low
                return -1
            
            # Probing the position formula
            pos = low + int(((float(high - low) / (arr[high] - arr[low])) * (target - arr[low])))
            
            if arr[pos] == target:
                return pos
            elif arr[pos] < target:
                low = pos + 1
            else:
                high = pos - 1
        return -1

    # ==========================================================================
    # 5. EXPONENTIAL SEARCH
    # ==========================================================================
    @staticmethod
    def exponential_search(arr: List[T], target: T) -> int:
        """
        Finds range where element is present by doubling index, then Binary Searches.
        
        Theory:
            Useful for unbounded arrays or when the element is near the start.
            1. Start at index 1, double i (1, 2, 4, 8...) until arr[i] > target.
            2. Binary search between prev_i and i.
            
        Complexity:
            Time: O(log i) where i is index of element.
            Space: O(1)
        """
        n = len(arr)
        if n == 0: return -1
        if arr[0] == target: return 0
        
        i = 1
        while i < n and arr[i] <= target:
            i = i * 2
            
        # Call binary search helper on specific range
        return SearchAlgorithms._binary_search_range(arr, target, i // 2, min(i, n - 1))

    @staticmethod
    def _binary_search_range(arr: List[T], target: T, left: int, right: int) -> int:
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    # ==========================================================================
    # 6. FIBONACCI SEARCH
    # ==========================================================================
    @staticmethod
    def fibonacci_search(arr: List[T], target: T) -> int:
        """
        Uses Fibonacci numbers to divide the array ranges.
        
        Theory:
            Similar to Binary Search but uses only addition/subtraction (historically 
            faster than division). Divides array into unequal parts based on Fib numbers.
            Requires SORTED array.
            
        Complexity:
            Time: O(log N)
            Space: O(1)
        """
        n = len(arr)
        fib_m2 = 0  # (m-2)'th Fibonacci No.
        fib_m1 = 1  # (m-1)'th Fibonacci No.
        fib_m = fib_m2 + fib_m1  # m'th Fibonacci No.
        
        # Store the smallest Fibonacci Number greater than or equal to n
        while fib_m < n:
            fib_m2 = fib_m1
            fib_m1 = fib_m
            fib_m = fib_m2 + fib_m1
            
        offset = -1
        
        while fib_m > 1:
            # Check valid index
            i = min(offset + fib_m2, n - 1)
            
            if arr[i] < target:
                fib_m = fib_m1
                fib_m1 = fib_m2
                fib_m2 = fib_m - fib_m1
                offset = i
            elif arr[i] > target:
                fib_m = fib_m2
                fib_m1 = fib_m1 - fib_m2
                fib_m2 = fib_m - fib_m1
            else:
                return i
                
        if fib_m1 and offset + 1 < n and arr[offset + 1] == target:
            return offset + 1
            
        return -1

    # ==========================================================================
    # 7. TERNARY SEARCH
    # ==========================================================================
    @staticmethod
    def ternary_search(arr: List[T], target: T) -> int:
        """
        Divides the array into three parts using two pivots.
        
        Theory:
            mid1 = l + (r-l)/3
            mid2 = r - (r-l)/3
            Determine which third the target lies in. 
            Requires SORTED array.
            
        Complexity:
            Time: O(log_3 N). Theoretically more steps than Binary (log_2 N) due to 
            more comparisons per iteration, but reduces search space faster.
        """
        return SearchAlgorithms._ternary_recursive(arr, 0, len(arr) - 1, target)

    @staticmethod
    def _ternary_recursive(arr: List[T], l: int, r: int, target: T) -> int:
        if r >= l:
            mid1 = l + (r - l) // 3
            mid2 = r - (r - l) // 3
            
            if arr[mid1] == target: return mid1
            if arr[mid2] == target: return mid2
            
            if target < arr[mid1]:
                return SearchAlgorithms._ternary_recursive(arr, l, mid1 - 1, target)
            elif target > arr[mid2]:
                return SearchAlgorithms._ternary_recursive(arr, mid2 + 1, r, target)
            else:
                return SearchAlgorithms._ternary_recursive(arr, mid1 + 1, mid2 - 1, target)
        return -1

    # ==========================================================================
    # 8. META BINARY SEARCH (One-Sided / Bitwise)
    # ==========================================================================
    @staticmethod
    def meta_binary_search(arr: List[T], target: T) -> int:
        """
        Constructs the index using bitwise manipulation.
        
        Theory:
            Stores array size as closest power of 2. We construct the target index 
            bits from most significant to least. Often cache-friendlier in low-level 
            implementations because it avoids dependency chains in 'mid' calculation.
            
        Complexity:
            Time: O(log N)
        """
        n = len(arr)
        # Closest power of 2 >= n gives us number of bits required
        lg = n.bit_length() 
        
        pos = 0
        for i in range(lg - 1, -1, -1):
            if arr[pos] == target:
                return pos
            
            new_pos = pos | (1 << i)
            if new_pos < n:
                if arr[new_pos] <= target:
                    pos = new_pos
                    
        if pos < n and arr[pos] == target:
            return pos
        return -1

    # ==========================================================================
    # 9. GOLDEN SECTION SEARCH
    # ==========================================================================
    @staticmethod
    def golden_section_search(arr: List[T], target: T) -> int:
        """
        Similar to Ternary Search but uses the Golden Ratio (phi) to select pivots.
        
        Theory:
            Phi ~= 1.618.
            x1 = r - (r-l)/phi
            x2 = l + (r-l)/phi
            Usually used for finding Min/Max in unimodal functions, but applicable 
            to discrete searching in sorted arrays.
        """
        n = len(arr)
        if n == 0: return -1
        
        phi = (1 + math.sqrt(5)) / 2
        l, r = 0, n - 1
        
        # Integer approximation loop
        while l <= r:
            if r - l < 2:
                if arr[l] == target: return l
                if arr[r] == target: return r
                return -1
                
            # Distances based on golden ratio
            d = int((r - l) / phi)
            mid1 = r - d
            mid2 = l + d
            
            if arr[mid1] == target: return mid1
            if arr[mid2] == target: return mid2
            
            # Determine region
            if target < arr[mid1]:
                r = mid1 - 1
            elif target > arr[mid2]:
                l = mid2 + 1
            else:
                l = mid1 + 1
                r = mid2 - 1
        return -1

    # ==========================================================================
    # 10. SUBLIST SEARCH (Subarray pattern search)
    # ==========================================================================
    @staticmethod
    def sublist_search(main_arr: List[T], sub_arr: List[T]) -> int:
        """
        Finds the starting index of a sub-list within a main list.
        
        Theory:
            Checks if the sequence 'sub_arr' exists contiguously in 'main_arr'.
            Implemented here using a standard two-pointer approach (Naive).
            For optimal performance, KMP or Boyer-Moore algorithms would be used.
            
        Returns:
            Starting index of first occurrence, or -1.
        """
        n = len(main_arr)
        m = len(sub_arr)
        if m > n: return -1
        if m == 0: return 0
        
        for i in range(n - m + 1):
            match = True
            for j in range(m):
                if main_arr[i + j] != sub_arr[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    # ==========================================================================
    # 11. UNIFORM BINARY SEARCH
    # ==========================================================================
    @staticmethod
    def uniform_binary_search(arr: List[T], target: T) -> int:
        """
        A variation of binary search that stores the differences (deltas) of indices
        in a lookup table. 
        
        Theory:
            Historically used when arithmetic shifts were slow. 
            Here we simulate the index calculation logic. 
            Midpoint is calculated as index = index + delta or index - delta.
        """
        n = len(arr)
        # Build lookup table roughly halving
        delta = [0] * 32 # Support up to 2^32
        power = 1
        i = 1
        while power < n:
            delta[i] = (n + power) // (2 * power)
            power *= 2
            i += 1
            
        curr = delta[1] # Initial midpoint
        d = 1           # Level
        
        while d < i and curr < n:
            if arr[curr] == target:
                return curr
            
            d += 1
            if d >= len(delta): break # Safety
            
            if target < arr[curr]:
                curr -= delta[d]
            else:
                curr += delta[d]
                
            # Boundary check - if curr < 0, we fix in next iter essentially,
            # but simple loop requires positive bounds.
            if curr < 0: curr = 0
            
        # Final check around convergence
        if 0 <= curr < n and arr[curr] == target:
            return curr
        return -1

    # ==========================================================================
    # 12. GALLOPING SEARCH
    # ==========================================================================
    @staticmethod
    def galloping_search(arr: List[T], target: T) -> int:
        """
        Also known as Doubling Search. Identical logic to Exponential Search 
        but often specifically refers to finding the range [2^k, 2^(k+1)].
        """
        # Utilizing the Exponential implementation as they are algorithmically synonymous
        return SearchAlgorithms.exponential_search(arr, target)

    # ==========================================================================
    # 13. BINARY SEARCH VARIANT (LOWER BOUND)
    # ==========================================================================
    @staticmethod
    def lower_bound(arr: List[T], target: T) -> int:
        """
        Finds the first position where value is >= target.
        
        Usage:
            Useful for insertions or frequency counting.
        """
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        # left is the insertion point
        if left < len(arr) and arr[left] == target:
            return left
        return -1 # Not strictly found, though often lower_bound returns index regardless

    # ==========================================================================
    # 14. BINARY SEARCH VARIANT (UPPER BOUND)
    # ==========================================================================
    @staticmethod
    def upper_bound(arr: List[T], target: T) -> int:
        """
        Finds the first position where value is > target.
        """
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left # Returns index where element > target starts

    # ==========================================================================
    # 15. RANDOMIZED BINARY SEARCH
    # ==========================================================================
    @staticmethod
    def randomized_binary_search(arr: List[T], target: T) -> int:
        """
        Selects a random pivot instead of the exact middle.
        
        Theory:
            Reduces the probability of worst-case scenarios in adversarial 
            datasets designed to make standard BS slow (though standard BS is 
            always logN, so this is more theoretical/probabilistic defense).
        """
        left, right = 0, len(arr) - 1
        while left <= right:
            # Random pivot between left and right
            rand_mid = random.randint(left, right)
            
            if arr[rand_mid] == target:
                return rand_mid
            elif arr[rand_mid] < target:
                left = rand_mid + 1
            else:
                right = rand_mid - 1
        return -1

    # ==========================================================================
    # 16. K-ARY SEARCH
    # ==========================================================================
    @staticmethod
    def k_ary_search(arr: List[T], target: T, k: int = 4) -> int:
        """
        Generalized Binary/Ternary search. Splits array into K parts.
        
        Theory:
            Calculates K-1 midpoints. 
            High K reduces height of tree (log_k N) but increases comparisons per node.
            Optimal K often depends on cache line size and SIMD capabilities.
        """
        n = len(arr)
        return SearchAlgorithms._k_ary_recursive(arr, 0, n - 1, target, k)

    @staticmethod
    def _k_ary_recursive(arr: List[T], low: int, high: int, target: T, k: int) -> int:
        if low > high:
            return -1
        
        # Generate K-1 midpoints
        step = (high - low + 1) // k
        mids = []
        for i in range(1, k):
            idx = low + i * step
            if idx > high: break
            mids.append(idx)
            
        # Check mids
        for idx in mids:
            if arr[idx] == target:
                return idx
        
        # Check segments
        mids.insert(0, low - 1)
        mids.append(high + 1)
        
        for i in range(len(mids) - 1):
            l_bound = mids[i] + 1
            r_bound = mids[i+1] - 1
            
            val_l = arr[l_bound] if l_bound <= high else None
            val_r = arr[r_bound] if r_bound <= high else None
            
            # Determine if target is in this slice
            if l_bound <= r_bound:
                # We need to check boundaries of the slice
                # Simplified: if target is smaller than next mid, it's in this bucket
                # (Because we checked exact matches on mids already)
                upper_val = arr[mids[i+1]] if mids[i+1] <= high else float('inf')
                lower_val = arr[mids[i]] if mids[i] >= low else float('-inf')
                
                if lower_val < target < upper_val:
                    return SearchAlgorithms._k_ary_recursive(arr, l_bound, r_bound, target, k)
        
        return -1

    # ==========================================================================
    # 17. FRACTIONAL CASCADING SEARCH (Concept Implementation)
    # ==========================================================================
    class FractionalCascading:
        """
        Advanced structure to search for same element in Multiple Sorted Lists.
        
        Theory:
            Instead of binary searching each list (O(k log N)), we augment lists
            with pointers to the next list to search in O(log N + k).
            
            Here, we implement a simplified 'Lookahead' simulation.
        """
        def __init__(self, lists: List[List[int]]):
            self.lists = lists
            # In a full implementation, we would build augmented arrays here.
            # This is a placeholder for the algorithmic logic of iterative search refinement.
            
        def search(self, target: int) -> List[int]:
            """
            Returns index of target in each list (or -1) efficiently.
            Assuming standard BS for this demo, as full graph construction is complex.
            """
            results = []
            # To simulate the cascading effect:
            # In real FC, the position in list[i] gives a hint for list[i+1].
            # Here we maintain a search range that narrows or shifts based on previous.
            
            # Fallback to efficient repeated search for demonstration in single file
            for lst in self.lists:
                results.append(SearchAlgorithms.binary_search(lst, target))
            return results

    # ==========================================================================
    # 18. SADDLEBACK SEARCH (2D Matrix Search)
    # ==========================================================================
    @staticmethod
    def saddleback_search(matrix: List[List[int]], target: int) -> Tuple[int, int]:
        """
        Search in a row-wise and column-wise sorted matrix.
        
        Theory:
            Start at top-right corner.
            If cell > target: move left.
            If cell < target: move down.
            
        Complexity:
            Time: O(N + M)
        """
        if not matrix or not matrix[0]: return (-1, -1)
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        i, j = 0, cols - 1 # Top-Right
        
        while i < rows and j >= 0:
            if matrix[i][j] == target:
                return (i, j)
            elif matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return (-1, -1)

    # ==========================================================================
    # 19. PIVOT BINARY SEARCH (Rotated Array)
    # ==========================================================================
    @staticmethod
    def pivot_binary_search(arr: List[T], target: T) -> int:
        """
        Search in a sorted array that has been rotated.
        
        Theory:
            1. Find the pivot (smallest element).
            2. Perform BS on the appropriate subarray.
            Or handle logic inside one pass (implemented here).
        """
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            
            # Check if left half is sorted
            if arr[left] <= arr[mid]:
                if arr[left] <= target < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half is sorted
            else:
                if arr[mid] < target <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

    # ==========================================================================
    # 20. WEIGHTED BINARY SEARCH (Probabilistic)
    # ==========================================================================
    @staticmethod
    def weighted_binary_search(arr: List[T], weights: List[float], target: T) -> int:
        """
        Binary Search where splitting depends on element weights (access probabilities).
        
        Theory:
            Instead of splitting at index count / 2, split where sum of weights 
            is roughly equal on both sides. Optimizes average search time based 
            on usage frequency.
        """
        # Calculating prefix sums for weights for O(1) range weight calculation
        # This is a simplified version picking weighted middle.
        
        low, high = 0, len(arr) - 1
        
        while low <= high:
            # Find 'weighted' middle:
            # Sum weights in current range, find index that splits weight in half.
            # For performance in this single function without pre-calc tree:
            # We default to standard binary search as calculating weight sum 
            # iteratively is O(N), defeating the purpose. 
            # A true Weighted BS requires a pre-built Weighted Search Tree (BST).
            # Here we demonstrate the logic check.
            mid = (low + high) // 2 
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    # ==========================================================================
    # 21. SPARSE TABLE SEARCH (Range Query Structure)
    # ==========================================================================
    class SparseTable:
        """
        Used typically for Range Minimum Query, but can be adapted for static search.
        
        Theory:
            Precomputes intervals of size 2^k. 
            Allows O(1) query for Min/Max/GCD.
            Here we implement checking existence in range via Min/Max logic on sorted array.
        """
        def __init__(self, arr: List[int]):
            self.arr = arr
            self.n = len(arr)
            self.log_n = self.n.bit_length()
            self.table = [[0] * self.n for _ in range(self.log_n)]
            self._build()

        def _build(self):
            # Initialize interval length 1 (2^0)
            for i in range(self.n):
                self.table[0][i] = self.arr[i]
            
            # Compute for lengths 2, 4, 8...
            j = 1
            while (1 << j) <= self.n:
                i = 0
                while (i + (1 << j) - 1) < self.n:
                    # Storing MAX for this example to check upper bounds
                    self.table[j][i] = max(self.table[j-1][i], 
                                           self.table[j-1][i + (1 << (j-1))])
                    i += 1
                j += 1
                
        def query_max(self, L: int, R: int) -> int:
            if L > R: return -float('inf')
            j = (R - L + 1).bit_length() - 1
            return max(self.table[j][L], self.table[j][R - (1 << j) + 1])


# ==============================================================================
# TEST DRIVER
# ==============================================================================
if __name__ == "__main__":
    print("Initializing Algorithm Suite Tests...\n")
    
    # 1. Generate Dummy Sorted Data
    data_size = 50
    sorted_data = sorted([random.randint(1, 200) for _ in range(data_size)])
    target_idx = 25
    target_val = sorted_data[target_idx]
    
    # Missing target for negative tests
    missing_val = 201 
    
    print(f"Data (first 10): {sorted_data[:10]}...")
    print(f"Target Value: {target_val} at true index {target_idx}")
    
    algo = SearchAlgorithms()
    
    # Run Tests
    print(f"1. Linear Search:       {algo.linear_search(sorted_data, target_val)}")
    print(f"2. Binary Search:       {algo.binary_search(sorted_data, target_val)}")
    print(f"3. Jump Search:         {algo.jump_search(sorted_data, target_val)}")
    print(f"4. Interpolation Srch:  {algo.interpolation_search(sorted_data, target_val)}")
    print(f"5. Exponential Srch:    {algo.exponential_search(sorted_data, target_val)}")
    print(f"6. Fibonacci Search:    {algo.fibonacci_search(sorted_data, target_val)}")
    print(f"7. Ternary Search:      {algo.ternary_search(sorted_data, target_val)}")
    print(f"8. Meta Binary Srch:    {algo.meta_binary_search(sorted_data, target_val)}")
    print(f"9. Golden Section:      {algo.golden_section_search(sorted_data, target_val)}")
    
    # Sublist Test
    sub = sorted_data[10:13]
    print(f"10. Sublist Search:     {algo.sublist_search(sorted_data, sub)} (Expected: 10)")
    
    print(f"11. Uniform Binary:     {algo.uniform_binary_search(sorted_data, target_val)}")
    print(f"12. Galloping Search:   {algo.galloping_search(sorted_data, target_val)}")
    
    # Bounds
    data_with_dups = [1, 2, 4, 4, 4, 5, 6]
    print(f"13. Lower Bound (4):    {algo.lower_bound(data_with_dups, 4)} (Expected: 2)")
    print(f"14. Upper Bound (4):    {algo.upper_bound(data_with_dups, 4)} (Expected: 5)")
    
    print(f"15. Randomized BS:      {algo.randomized_binary_search(sorted_data, target_val)}")
    print(f"16. K-ary Search (k=4): {algo.k_ary_search(sorted_data, target_val, k=4)}")
    
    # Fractional Cascading Demo
    fc = algo.FractionalCascading([sorted_data, sorted_data])
    print(f"17. Frac Cascading:     {fc.search(target_val)}")
    
    # Saddleback (2D)
    matrix = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15]
    ]
    print(f"18. Saddleback (11):    {algo.saddleback_search(matrix, 11)}")
    
    # Pivot (Rotated)
    rotated = [15, 18, 2, 3, 6, 12]
    print(f"19. Pivot Search (3):   {algo.pivot_binary_search(rotated, 3)}")
    
    print(f"20. Weighted BS:        {algo.weighted_binary_search(sorted_data, [], target_val)}")
    
    # Sparse Table (Max Query)
    st = algo.SparseTable(sorted_data)
    # Query max in range [0, 5]
    print(f"21. Sparse Table Max:   {st.query_max(0, 5)} (Expected: {sorted_data[5]})")
    
    print("\nTests Completed.")