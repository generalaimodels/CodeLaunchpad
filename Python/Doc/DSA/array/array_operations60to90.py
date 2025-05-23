class ArrayOperations:
    """
    A class containing various array operations and algorithms.
    """

    def max_subarray_sum(self, arr):
        """
        Finds the maximum subarray sum using Kadane's Algorithm.

        Parameters:
            arr (list): List of integers.

        Returns:
            int: The maximum subarray sum.
        """
        if not arr:
            return 0
        max_current = max_global = arr[0]
        for num in arr[1:]:
            max_current = max(num, max_current + num)
            max_global = max(max_global, max_current)
        return max_global

    def min_subarray_sum(self, arr):
        """
        Finds the minimum subarray sum.

        Parameters:
            arr (list): List of integers.

        Returns:
            int: The minimum subarray sum.
        """
        if not arr:
            return 0
        min_current = min_global = arr[0]
        for num in arr[1:]:
            min_current = min(num, min_current + num)
            min_global = min(min_global, min_current)
        return min_global

    def find_subarray_with_given_sum(self, arr, target):
        """
        Finds a contiguous subarray that sums to a given target.

        Parameters:
            arr (list): List of integers.
            target (int): The target sum.

        Returns:
            tuple: The start and end indices of the subarray.
        """
        sum_dict = {}
        current_sum = 0
        for i, num in enumerate(arr):
            current_sum += num
            if current_sum == target:
                return (0, i)
            if (current_sum - target) in sum_dict:
                return (sum_dict[current_sum - target] + 1, i)
            sum_dict[current_sum] = i
        return None  # No subarray found

    def quicksort_partition(self, arr, low, high):
        """
        Partition function used in QuickSort.

        Parameters:
            arr (list): List of elements.
            low (int): Starting index.
            high (int): Ending index.

        Returns:
            int: Partition index.
        """
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                # Swap arr[i] and arr[j]
                arr[i], arr[j] = arr[j], arr[i]
        # Swap arr[i+1] and arr[high] (or pivot)
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def even_odd_partition(self, arr):
        """
        Rearranges the array so that even numbers come before odd numbers.

        Parameters:
            arr (list): List of integers.

        Returns:
            list: Rearranged list with even numbers before odd numbers.
        """
        left = 0
        right = len(arr) - 1
        while left < right:
            while left < right and arr[left] % 2 == 0:
                left += 1
            while left < right and arr[right] % 2 != 0:
                right -= 1
            if left < right:
                arr[left], arr[right] = arr[right], arr[left]
        return arr

    def positive_negative_partition(self, arr):
        """
        Groups positive and negative numbers separately.

        Parameters:
            arr (list): List of integers.

        Returns:
            list: Rearranged list with negative numbers before positive numbers.
        """
        left = 0
        right = len(arr) - 1
        while left <= right:
            while left <= right and arr[left] < 0:
                left += 1
            while left <= right and arr[right] >= 0:
                right -= 1
            if left < right:
                arr[left], arr[right] = arr[right], arr[left]
        return arr

    def kth_smallest_element(self, arr, k):
        """
        Finds the kth smallest element using QuickSelect algorithm.

        Parameters:
            arr (list): List of elements.
            k (int): The kth position to find.

        Returns:
            The kth smallest element.
        """
        if k < 1 or k > len(arr):
            raise ValueError("k is out of bounds")
        return self.quickselect(arr, 0, len(arr) - 1, k - 1)

    def kth_largest_element(self, arr, k):
        """
        Finds the kth largest element.

        Parameters:
            arr (list): List of elements.
            k (int): The kth position to find.

        Returns:
            The kth largest element.
        """
        if k < 1 or k > len(arr):
            raise ValueError("k is out of bounds")
        return self.quickselect(arr, 0, len(arr) - 1, len(arr) - k)

    def quickselect(self, arr, low, high, k):
        """
        QuickSelect algorithm to find the kth smallest element.

        Parameters:
            arr (list): List of elements.
            low (int): Starting index.
            high (int): Ending index.
            k (int): The kth index to find.

        Returns:
            The kth smallest element.
        """
        if low == high:
            return arr[low]

        pivot_index = self.partition(arr, low, high)

        if k == pivot_index:
            return arr[k]
        elif k < pivot_index:
            return self.quickselect(arr, low, pivot_index - 1, k)
        else:
            return self.quickselect(arr, pivot_index + 1, high, k)

    def partition(self, arr, low, high):
        """
        Partition function used in QuickSelect.

        Parameters:
            arr (list): List of elements.
            low (int): Starting index.
            high (int): Ending index.

        Returns:
            int: Partition index.
        """
        pivot = arr[high]
        i = low
        for j in range(low, high):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        arr[i], arr[high] = arr[high], arr[i]
        return i

    def prefix_sum_array(self, arr):
        """
        Computes the prefix sum array.

        Parameters:
            arr (list): List of numbers.

        Returns:
            list: Prefix sum array.
        """
        prefix_sum = [0] * len(arr)
        total = 0
        for i, num in enumerate(arr):
            total += num
            prefix_sum[i] = total
        return prefix_sum

    def suffix_sum_array(self, arr):
        """
        Computes the suffix sum array.

        Parameters:
            arr (list): List of numbers.

        Returns:
            list: Suffix sum array.
        """
        suffix_sum = [0] * len(arr)
        total = 0
        for i in reversed(range(len(arr))):
            total += arr[i]
            suffix_sum[i] = total
        return suffix_sum

    def difference_array(self, arr):
        """
        Computes the difference array.

        Parameters:
            arr (list): List of numbers.

        Returns:
            list: Difference array.
        """
        diff_array = [0] * len(arr)
        diff_array[0] = arr[0]
        for i in range(1, len(arr)):
            diff_array[i] = arr[i] - arr[i - 1]
        return diff_array

    def cumulative_sum(self, arr):
        """
        Computes the cumulative sum of the array.

        Parameters:
            arr (list): List of numbers.

        Returns:
            list: Cumulative sum array.
        """
        cum_sum = []
        total = 0
        for num in arr:
            total += num
            cum_sum.append(total)
        return cum_sum

    def cumulative_product(self, arr):
        """
        Computes the cumulative product of the array.

        Parameters:
            arr (list): List of numbers.

        Returns:
            list: Cumulative product array.
        """
        cum_prod = []
        total = 1
        for num in arr:
            total *= num
            cum_prod.append(total)
        return cum_prod

    def running_average(self, arr):
        """
        Computes the running average of the array.

        Parameters:
            arr (list): List of numbers.

        Returns:
            list: Running average array.
        """
        running_avg = []
        total = 0
        for i, num in enumerate(arr, 1):
            total += num
            running_avg.append(total / i)
        return running_avg

    def count_inversions(self, arr):
        """
        Counts the number of inversions in the array using merge sort.

        Parameters:
            arr (list): List of numbers.

        Returns:
            int: Number of inversions.
        """
        if arr:
            temp_arr = arr.copy()
            return self._merge_sort(arr, temp_arr, 0, len(arr) - 1)
        else:
            return 0

    def _merge_sort(self, arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count += self._merge_sort(arr, temp_arr, left, mid)
            inv_count += self._merge_sort(arr, temp_arr, mid + 1, right)
            inv_count += self._merge(arr, temp_arr, left, mid, right)
        return inv_count

    def _merge(self, arr, temp_arr, left, mid, right):
        i = left
        j = mid + 1
        k = left
        inv_count = 0

        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                temp_arr[k] = arr[j]
                inv_count += (mid - i + 1)
                j += 1
            k += 1

        while i <= mid:
            temp_arr[k] = arr[i]
            i += 1
            k += 1

        while j <= right:
            temp_arr[k] = arr[j]
            j += 1
            k += 1

        for idx in range(left, right + 1):
            arr[idx] = temp_arr[idx]

        return inv_count

    def is_sorted(self, arr):
        """
        Checks if the array is sorted.

        Parameters:
            arr (list): List of numbers.

        Returns:
            bool: True if sorted, False otherwise.
        """
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:
                return False
        return True

    def bounds_validation(self, index, size):
        """
        Ensures index is within valid limits.

        Parameters:
            index (int): Index to check.
            size (int): Size of the array.

        Returns:
            bool: True if index is valid, False otherwise.
        """
        return 0 <= index < size

    def out_of_bounds_check(self, index, size):
        """
        Detects and handles illegal index access.

        Parameters:
            index (int): Index to check.
            size (int): Size of the array.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index >= size:
            raise IndexError("Index out of bounds")

    def static_memory_allocation(self, size):
        """
        Allocates a fixed-size array.

        Parameters:
            size (int): Size of the array.

        Returns:
            list: Array initialized with zeros.
        """
        return [0] * size

    def dynamic_memory_allocation(self, size):
        """
        Allocates an array with size determined at runtime.

        Parameters:
            size (int): Size of the array.

        Returns:
            list: Array initialized with zeros.
        """
        return [0] * size

    def memory_deallocation(self, arr):
        """
        Frees the memory used by dynamic arrays.

        Parameters:
            arr (list): Array to deallocate.

        Returns:
            None
        """
        del arr

    def multi_dimensional_array_initialization(self, dimensions, default_value=0):
        """
        Creates a multi-dimensional array with default values.

        Parameters:
            dimensions (list): Dimensions of the array, e.g., [3, 4] for a 3x4 array.
            default_value: The default value to initialize.

        Returns:
            list: Multi-dimensional array.
        """
        if len(dimensions) == 1:
            return [default_value] * dimensions[0]
        else:
            return [self.multi_dimensional_array_initialization(dimensions[1:], default_value) for _ in range(dimensions[0])]

    def multi_dimensional_array_traversal(self, arr):
        """
        Iterates over multi-dimensional array elements.

        Parameters:
            arr (list): Multi-dimensional array.

        Returns:
            None
        """
        def traverse(array, indices):
            if isinstance(array, list):
                for i, elem in enumerate(array):
                    traverse(elem, indices + [i])
            else:
                print(f"Element at {indices}: {array}")
        traverse(arr, [])

    def multi_dimensional_indexing(self, arr, indices):
        """
        Accesses elements with multiple indices.

        Parameters:
            arr (list): Multi-dimensional array.
            indices (list): List of indices.

        Returns:
            The element at the given indices.
        """
        elem = arr
        for idx in indices:
            elem = elem[idx]
        return elem

    def matrix_transposition(self, matrix):
        """
        Transposes a matrix.

        Parameters:
            matrix (list of lists): The matrix to transpose.

        Returns:
            list of lists: Transposed matrix.
        """
        rows = len(matrix)
        cols = len(matrix[0])
        transposed = self.multi_dimensional_array_initialization([cols, rows])
        for i in range(rows):
            for j in range(cols):
                transposed[j][i] = matrix[i][j]
        return transposed

    def matrix_multiplication(self, matrix_a, matrix_b):
        """
        Multiplies two matrices.

        Parameters:
            matrix_a (list of lists): First matrix.
            matrix_b (list of lists): Second matrix.

        Returns:
            list of lists: Resultant matrix after multiplication.
        """
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])

        if cols_a != rows_b:
            raise ValueError("Cannot multiply matrices: incompatible dimensions.")

        result = self.multi_dimensional_array_initialization([rows_a, cols_b], 0)
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
        return result

    def flattening(self, arr):
        """
        Flattens a multi-dimensional array.

        Parameters:
            arr (list): Multi-dimensional array.

        Returns:
            list: Flattened array.
        """
        flat_list = []
        def flatten(element):
            if isinstance(element, list):
                for item in element:
                    flatten(item)
            else:
                flat_list.append(element)
        flatten(arr)
        return flat_list

    def reshaping(self, arr, new_dimensions):
        """
        Reshapes an array without changing its data.

        Parameters:
            arr (list): Original multi-dimensional array.
            new_dimensions (list): New dimensions.

        Returns:
            list: Reshaped array.
        """
        flat_list = self.flattening(arr)
        total_elements = 1
        for dim in new_dimensions:
            total_elements *= dim
        if total_elements != len(flat_list):
            raise ValueError("Cannot reshape array: total elements do not match.")
        return self._reshape_recursive(flat_list, new_dimensions)

    def _reshape_recursive(self, flat_list, dimensions):
        if len(dimensions) == 1:
            return [flat_list.pop(0) for _ in range(dimensions[0])]
        else:
            return [self._reshape_recursive(flat_list, dimensions[1:]) for _ in range(dimensions[0])]

def main():
    array_ops = ArrayOperations()

    # Test max_subarray_sum
    arr = [-2, -3, 4, -1, -2, 1, 5, -3]
    print("Max Subarray Sum:", array_ops.max_subarray_sum(arr))

    # Test min_subarray_sum
    arr = [3, -4, 2, -3, -1, 7, -5]
    print("Min Subarray Sum:", array_ops.min_subarray_sum(arr))

    # Test find_subarray_with_given_sum
    arr = [1, 4, 20, 3, 10, 5]
    target = 33
    result = array_ops.find_subarray_with_given_sum(arr, target)
    if result:
        print("Subarray with given sum found between indexes:", result)
    else:
        print("No subarray with given sum found.")

    # Test quicksort_partition
    arr = [10, 80, 30, 90, 40, 50, 70]
    partition_index = array_ops.quicksort_partition(arr, 0, len(arr) - 1)
    print("Array after partition:", arr)
    print("Partition Index:", partition_index)

    # Test even_odd_partition
    arr = [12, 17, 70, 15, 22, 65, 21, 90]
    print("Array after even-odd partition:", array_ops.even_odd_partition(arr.copy()))

    # Test positive_negative_partition
    arr = [12, -17, 70, -15, 22, -65, 21, -90]
    print("Array after positive-negative partition:", array_ops.positive_negative_partition(arr.copy()))

    # Test kth smallest element
    arr = [7, 10, 4, 3, 20, 15]
    k = 3
    print(f"{k}th smallest element is:", array_ops.kth_smallest_element(arr.copy(), k))

    # Test kth largest element
    k = 2
    print(f"{k}th largest element is:", array_ops.kth_largest_element(arr.copy(), k))

    # Test prefix sum array
    arr = [10, 20, 30, 40, 50]
    print("Prefix Sum Array:", array_ops.prefix_sum_array(arr))

    # Test suffix sum array
    print("Suffix Sum Array:", array_ops.suffix_sum_array(arr))

    # Test difference array
    print("Difference Array:", array_ops.difference_array(arr))

    # Test cumulative sum
    print("Cumulative Sum:", array_ops.cumulative_sum(arr))

    # Test cumulative product
    print("Cumulative Product:", array_ops.cumulative_product(arr))

    # Test running average
    print("Running Average:", array_ops.running_average(arr))

    # Test count inversions
    arr = [1, 20, 6, 4, 5]
    print("Number of inversions:", array_ops.count_inversions(arr.copy()))

    # Test is_sorted
    arr = [1, 2, 3, 4, 5]
    print("Array is sorted:", array_ops.is_sorted(arr))

    arr = [5, 4, 3, 2, 1]
    print("Array is sorted:", array_ops.is_sorted(arr))

    # Test multi-dimensional array initialization
    dimensions = [2, 3]
    multi_array = array_ops.multi_dimensional_array_initialization(dimensions, default_value=1)
    print("Multi-dimensional array:", multi_array)

    # Test multi-dimensional array traversal
    print("Traversing multi-dimensional array:")
    array_ops.multi_dimensional_array_traversal(multi_array)

    # Test matrix transposition
    matrix = [[1, 2, 3], [4, 5, 6]]
    transposed = array_ops.matrix_transposition(matrix)
    print("Transposed Matrix:", transposed)

    # Test matrix multiplication
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    multiplied_matrix = array_ops.matrix_multiplication(matrix_a, matrix_b)
    print("Matrix Multiplication Result:", multiplied_matrix)

    # Test flattening
    nested_list = [[1, [2, 3]], [4, 5], 6]
    print("Flattened array:", array_ops.flattening(nested_list))

    # Test reshaping
    arr = [1, 2, 3, 4, 5, 6]
    reshaped_array = array_ops.reshaping(arr, [2, 3])
    print("Reshaped Array:", reshaped_array)

if __name__ == "__main__":
    main()