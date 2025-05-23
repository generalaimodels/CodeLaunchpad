class ArrayOperations:
    """
    A class to perform various array operations including sorting, searching, and statistical calculations.
    """

    def __init__(self, size, default_value=0):
        """
        Initialize an array with a given size and default value.

        :param size: The size of the array.
        :param default_value: The default value for array elements.
        """
        self.size = size
        self.array = [default_value] * size

    def assign_values(self, values):
        """
        Assign values to the array elements.

        :param values: A list of values to assign.
        """
        min_length = min(self.size, len(values))
        for i in range(min_length):
            self.array[i] = values[i]

    def get_element(self, index):
        """
        Retrieve an element at a specific index.

        :param index: The index of the element to retrieve.
        :return: The element at the given index.
        """
        if 0 <= index < self.size:
            return self.array[index]
        else:
            raise IndexError("Index out of bounds.")

    def traverse_array(self):
        """
        Traverse and return all elements in the array.

        :return: A list of all array elements.
        """
        elements = []
        for i in range(self.size):
            elements.append(self.array[i])
        return elements

    def for_loop_iteration(self):
        """
        Use a for-loop to visit each element and return the elements.

        :return: A list of all array elements.
        """
        elements = []
        for element in self.array:
            elements.append(element)
        return elements

    def while_loop_iteration(self):
        """
        Use a while-loop to traverse the array and return the elements.

        :return: A list of all array elements.
        """
        elements = []
        index = 0
        while index < self.size:
            elements.append(self.array[index])
            index += 1
        return elements

    def recursive_traversal_helper(self, index, elements):
        """
        Helper method for recursive traversal.

        :param index: Current index in traversal.
        :param elements: List to collect elements.
        """
        if index >= self.size:
            return
        elements.append(self.array[index])
        self.recursive_traversal_helper(index + 1, elements)

    def recursive_traversal(self):
        """
        Recursively traverse and return all elements in the array.

        :return: A list of all array elements.
        """
        elements = []
        self.recursive_traversal_helper(0, elements)
        return elements

    def bubble_sort(self):
        """
        Sort the array using bubble sort algorithm.
        """
        n = self.size
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.array[j] > self.array[j + 1]:
                    # Swap
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]

    def selection_sort(self):
        """
        Sort the array using selection sort algorithm.
        """
        n = self.size
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.array[j] < self.array[min_idx]:
                    min_idx = j
            # Swap
            self.array[i], self.array[min_idx] = self.array[min_idx], self.array[i]

    def insertion_sort(self):
        """
        Sort the array using insertion sort algorithm.
        """
        for i in range(1, self.size):
            key = self.array[i]
            j = i - 1
            while j >= 0 and self.array[j] > key:
                self.array[j + 1] = self.array[j]
                j -= 1
            self.array[j + 1] = key

    def merge_sort(self):
        """
        Sort the array using merge sort algorithm.
        """
        self.array = self._merge_sort_recursive(self.array)

    def _merge_sort_recursive(self, arr):
        """
        Helper method for merge sort.

        :param arr: The array to sort.
        :return: Sorted array.
        """
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left_half = self._merge_sort_recursive(arr[:mid])
        right_half = self._merge_sort_recursive(arr[mid:])
        return self._merge(left_half, right_half)

    def _merge(self, left, right):
        """
        Merge two sorted arrays.

        :param left: Left half array.
        :param right: Right half array.
        :return: Merged sorted array.
        """
        merged = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1

        while i < len(left):
            merged.append(left[i])
            i += 1

        while j < len(right):
            merged.append(right[j])
            j += 1

        return merged

    def quick_sort(self):
        """
        Sort the array using quick sort algorithm.
        """
        self._quick_sort_recursive(0, self.size - 1)

    def _quick_sort_recursive(self, low, high):
        """
        Helper method for quick sort.

        :param low: Lower index.
        :param high: Higher index.
        """
        if low < high:
            pi = self._partition(low, high)
            self._quick_sort_recursive(low, pi - 1)
            self._quick_sort_recursive(pi + 1, high)

    def _partition(self, low, high):
        """
        Partition the array for quick sort.

        :param low: Lower index.
        :param high: Higher index.
        :return: Partition index.
        """
        pivot = self.array[high]
        i = low - 1
        for j in range(low, high):
            if self.array[j] < pivot:
                i += 1
                self.array[i], self.array[j] = self.array[j], self.array[i]
        self.array[i + 1], self.array[high] = self.array[high], self.array[i + 1]
        return i + 1

    def heap_sort(self):
        """
        Sort the array using heap sort algorithm.
        """
        n = self.size
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(n, i)
        # Extract elements
        for i in range(n - 1, 0, -1):
            self.array[i], self.array[0] = self.array[0], self.array[i]
            self._heapify(i, 0)

    def _heapify(self, n, i):
        """
        Heapify subtree rooted at index i.

        :param n: Size of heap.
        :param i: Root index.
        """
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and self.array[left] > self.array[largest]:
            largest = left

        if right < n and self.array[right] > self.array[largest]:
            largest = right

        if largest != i:
            self.array[i], self.array[largest] = self.array[largest], self.array[i]
            self._heapify(n, largest)

    def shell_sort(self):
        """
        Sort the array using shell sort algorithm.
        """
        n = self.size
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = self.array[i]
                j = i
                while j >= gap and self.array[j - gap] > temp:
                    self.array[j] = self.array[j - gap]
                    j -= gap
                self.array[j] = temp
            gap //= 2

    def counting_sort(self):
        """
        Sort the array using counting sort algorithm.
        """
        if self.size == 0:
            return
        max_value = self.array[0]
        for i in range(1, self.size):
            if self.array[i] > max_value:
                max_value = self.array[i]
        count = [0] * (max_value + 1)
        for num in self.array:
            count[num] += 1
        index = 0
        for i in range(len(count)):
            while count[i] > 0:
                self.array[index] = i
                index += 1
                count[i] -= 1

    def radix_sort(self):
        """
        Sort the array using radix sort algorithm.
        """
        if self.size == 0:
            return
        max_num = self.array[0]
        for num in self.array:
            if num > max_num:
                max_num = num
        exp = 1
        while max_num // exp > 0:
            self._counting_sort_for_radix(exp)
            exp *= 10

    def _counting_sort_for_radix(self, exp):
        """
        Counting sort used by radix sort.

        :param exp: Exponent value.
        """
        n = self.size
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = (self.array[i] // exp) % 10
            count[index] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        for i in range(n - 1, -1, -1):
            index = (self.array[i] // exp) % 10
            output[count[index] - 1] = self.array[i]
            count[index] -= 1
        for i in range(n):
            self.array[i] = output[i]

    def bucket_sort(self):
        """
        Sort the array using bucket sort algorithm.
        """
        if self.size == 0:
            return
        bucket_count = self.size
        max_value = self.array[0]
        for num in self.array:
            if num > max_value:
                max_value = num
        buckets = [[] for _ in range(bucket_count)]
        for num in self.array:
            index = int(num * bucket_count / (max_value + 1))
            buckets[index].append(num)
        index = 0
        for bucket in buckets:
            self._insertion_sort_bucket(bucket)
            for num in bucket:
                self.array[index] = num
                index += 1

    def _insertion_sort_bucket(self, bucket):
        """
        Insertion sort for sorting individual buckets.

        :param bucket: The bucket to sort.
        """
        for i in range(1, len(bucket)):
            key = bucket[i]
            j = i - 1
            while j >= 0 and bucket[j] > key:
                bucket[j + 1] = bucket[j]
                j -= 1
            bucket[j + 1] = key

    def binary_search(self, target):
        """
        Perform binary search for the target value.

        :param target: The value to search for.
        :return: Index of target if found, else -1.
        """
        left = 0
        right = self.size - 1
        while left <= right:
            mid = (left + right) // 2
            if self.array[mid] == target:
                return mid
            elif self.array[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def linear_search(self, target):
        """
        Perform linear search for the target value.

        :param target: The value to search for.
        :return: Index of target if found, else -1.
        """
        for i in range(self.size):
            if self.array[i] == target:
                return i
        return -1

    def find_minimum_element(self):
        """
        Find the minimum element in the array.

        :return: Minimum element.
        """
        if self.size == 0:
            return None
        min_value = self.array[0]
        for i in range(1, self.size):
            if self.array[i] < min_value:
                min_value = self.array[i]
        return min_value

    def find_maximum_element(self):
        """
        Find the maximum element in the array.

        :return: Maximum element.
        """
        if self.size == 0:
            return None
        max_value = self.array[0]
        for i in range(1, self.size):
            if self.array[i] > max_value:
                max_value = self.array[i]
        return max_value

    def sum_of_elements(self):
        """
        Calculate the sum of all elements in the array.

        :return: Sum of elements.
        """
        total = 0
        for num in self.array:
            total += num
        return total

    def average_calculation(self):
        """
        Calculate the average of the array elements.

        :return: Average value.
        """
        if self.size == 0:
            return 0
        return self.sum_of_elements() / self.size

    def median_finding(self):
        """
        Find the median of the array elements.

        :return: Median value.
        """
        sorted_array = self.array[:]
        sorted_array.sort()
        mid = self.size // 2
        if self.size % 2 == 0:
            return (sorted_array[mid - 1] + sorted_array[mid]) / 2
        else:
            return sorted_array[mid]

    def mode_finding(self):
        """
        Find the mode(s) of the array elements.

        :return: A list of mode(s).
        """
        frequency = {}
        max_freq = 0
        for num in self.array:
            frequency[num] = frequency.get(num, 0) + 1
            if frequency[num] > max_freq:
                max_freq = frequency[num]
        modes = [num for num, freq in frequency.items() if freq == max_freq]
        return modes

    def frequency_count(self):
        """
        Count the frequency of each element in the array.

        :return: Dictionary of element frequencies.
        """
        frequency = {}
        for num in self.array:
            frequency[num] = frequency.get(num, 0) + 1
        return frequency

    def remove_duplicates(self):
        """
        Remove duplicates from the array.

        :return: Array without duplicates.
        """
        unique_elements = []
        seen = set()
        for num in self.array:
            if num not in seen:
                seen.add(num)
                unique_elements.append(num)
        self.array = unique_elements
        self.size = len(self.array)

    def deep_copy_array(self):
        """
        Create a deep copy of the array.

        :return: A new array that is a copy of the current array.
        """
        copied_array = [num for num in self.array]
        return copied_array


# Dummy test case
if __name__ == "__main__":
    # Create an array of size 10 with default value 0
    array_ops = ArrayOperations(10)

    # Assign values
    array_ops.assign_values([5, 2, 9, 1, 5, 6, 9, 3, 2, 1])

    # Traverse array
    print("Original Array:")
    print(array_ops.traverse_array())

    # Sorting array
    array_ops.bubble_sort()
    print("Sorted Array using Bubble Sort:")
    print(array_ops.traverse_array())

    # Find minimum and maximum
    min_element = array_ops.find_minimum_element()
    max_element = array_ops.find_maximum_element()
    print(f"Minimum Element: {min_element}")
    print(f"Maximum Element: {max_element}")

    # Sum and average
    total = array_ops.sum_of_elements()
    average = array_ops.average_calculation()
    print(f"Sum of Elements: {total}")
    print(f"Average of Elements: {average}")

    # Median and mode
    median = array_ops.median_finding()
    modes = array_ops.mode_finding()
    print(f"Median of Elements: {median}")
    print(f"Mode of Elements: {modes}")

    # Remove duplicates
    array_ops.remove_duplicates()
    print("Array after removing duplicates:")
    print(array_ops.traverse_array())

    # Deep copy
    copied_array = array_ops.deep_copy_array()
    print("Copied Array:")
    print(copied_array)