class ArrayOperations:
    def __init__(self, size=0, default_value=0):
        """
        Array Initialization – Creating an array with default or specified values.
        """
        self.array = [default_value] * size
        self.size = size

    def declare_array(self, size):
        """
        Array Declaration – Declaring an array’s type and size.
        """
        self.array = [0] * size
        self.size = size

    def assign_values(self, values):
        """
        Array Assignment – Assigning values to array elements.
        """
        for i in range(min(len(values), self.size)):
            self.array[i] = values[i]

    def index_access(self, index):
        """
        Array Indexing – Accessing elements using their index.
        """
        if 0 <= index < self.size:
            return self.array[index]
        raise IndexError("Index out of bounds.")

    def element_access(self, position):
        """
        Array Element Access – Retrieving a specific element by its position.
        """
        return self.index_access(position)

    def traverse_array(self):
        """
        Array Traversal – Iterating over all elements in the array.
        """
        for element in self.array:
            print(element)

    def for_loop_iteration(self):
        """
        For‑loop Iteration – Using a for‑loop to visit each element.
        """
        for i in range(self.size):
            print(self.array[i])

    def while_loop_iteration(self):
        """
        While‑loop Iteration – Using a while‑loop to traverse the array.
        """
        i = 0
        while i < self.size:
            print(self.array[i])
            i += 1

    def recursive_traversal_helper(self, index):
        if index == self.size:
            return
        print(self.array[index])
        self.recursive_traversal_helper(index + 1)

    def recursive_traversal(self):
        """
        Recursive Traversal – Recursively processing array elements.
        """
        self.recursive_traversal_helper(0)

    def bubble_sort(self):
        """
        Bubble Sort – Sorting an array by repeatedly swapping adjacent elements.
        """
        n = self.size
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.array[j] > self.array[j + 1]:
                    # Swap
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]

    def selection_sort(self):
        """
        Selection Sort – Sorting by selecting the minimum element repeatedly.
        """
        n = self.size
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.array[min_idx] > self.array[j]:
                    min_idx = j
            # Swap
            self.array[i], self.array[min_idx] = self.array[min_idx], self.array[i]

    def insertion_sort(self):
        """
        Insertion Sort – Building the sorted array one element at a time.
        """
        for i in range(1, self.size):
            key = self.array[i]
            j = i - 1
            while j >=0 and key < self.array[j]:
                self.array[j + 1] = self.array[j]
                j -= 1
            self.array[j + 1] = key

    def merge_sort(self):
        """
        Merge Sort – A divide-and‑conquer recursive sort using merging.
        """
        self.array = self._merge_sort_recursive(self.array)

    def _merge_sort_recursive(self, arr):
        if len(arr) <=1:
            return arr
        mid = len(arr)//2
        left = self._merge_sort_recursive(arr[:mid])
        right = self._merge_sort_recursive(arr[mid:])
        return self._merge(left, right)

    def _merge(self, left, right):
        result = []
        i = j = 0
        while i<len(left) and j<len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i+=1
            else:
                result.append(right[j])
                j+=1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    def quick_sort(self):
        """
        Quick Sort – Sorting using a pivot and partitioning strategy.
        """
        self._quick_sort_recursive(0, self.size - 1)

    def _quick_sort_recursive(self, low, high):
        if low < high:
            pi = self._partition(low, high)
            self._quick_sort_recursive(low, pi -1)
            self._quick_sort_recursive(pi +1, high)

    def _partition(self, low, high):
        pivot = self.array[high]
        i = low -1
        for j in range(low, high):
            if self.array[j] <= pivot:
                i +=1
                self.array[i], self.array[j] = self.array[j], self.array[i]
        self.array[i+1], self.array[high] = self.array[high], self.array[i+1]
        return i +1

    def heap_sort(self):
        """
        Heap Sort – Sorting based on binary heap data structure.
        """
        n = self.size
        # Build max heap
        for i in range(n//2 -1, -1, -1):
            self._heapify(n, i)
        # Extract elements
        for i in range(n -1, 0, -1):
            self.array[i], self.array[0] = self.array[0], self.array[i]
            self._heapify(i, 0)

    def _heapify(self, n, i):
        largest = i
        l = 2*i +1
        r = 2*i +2
        if l < n and self.array[l] > self.array[largest]:
            largest = l
        if r < n and self.array[r] > self.array[largest]:
            largest = r
        if largest != i:
            self.array[i], self.array[largest] = self.array[largest], self.array[i]
            self._heapify(n, largest)

    def shell_sort(self):
        """
        Shell Sort – An improved insertion sort that allows swapping distant elements.
        """
        n = self.size
        gap = n//2
        while gap > 0:
            for i in range(gap, n):
                temp = self.array[i]
                j = i
                while j >= gap and self.array[j - gap] > temp:
                    self.array[j] = self.array[j - gap]
                    j -= gap
                self.array[j] = temp
            gap //=2

    def counting_sort(self):
        """
        Counting Sort – Sorting by counting occurrences of each value.
        """
        max_val = max(self.array)
        count = [0] * (max_val + 1)
        output = [0] * self.size

        for number in self.array:
            count[number] +=1

        for i in range(1, len(count)):
            count[i] += count[i -1]

        for number in reversed(self.array):
            output[count[number] -1] = number
            count[number] -=1

        self.array = output

    def radix_sort(self):
        """
        Radix Sort – Sorting numbers by processing individual digits.
        """
        max_val = max(self.array)
        exp =1
        while max_val // exp >0:
            self._counting_sort_exp(exp)
            exp *=10

    def _counting_sort_exp(self, exp):
        n = self.size
        output = [0]*n
        count = [0]*10

        for i in range(n):
            index = self.array[i] // exp
            count[(index)%10] +=1

        for i in range(1,10):
            count[i] += count[i -1]

        for i in range(n -1, -1, -1):
            index = self.array[i] // exp
            output[count[(index)%10] -1] = self.array[i]
            count[(index)%10] -=1

        self.array = output

    def bucket_sort(self):
        """
        Bucket Sort – Dividing elements into buckets and sorting each.
        """
        max_val = max(self.array)
        size = max_val / self.size
        buckets = [[] for _ in range(self.size)]

        for i in range(self.size):
            idx = int(self.array[i] / size)
            if idx != self.size:
                buckets[idx].append(self.array[i])
            else:
                buckets[self.size -1].append(self.array[i])

        for bucket in buckets:
            self._insertion_sort_bucket(bucket)

        result = []
        for bucket in buckets:
            result.extend(bucket)
        self.array = result

    def _insertion_sort_bucket(self, bucket):
        for i in range(1, len(bucket)):
            key = bucket[i]
            j = i -1
            while j >= 0 and bucket[j] > key:
                bucket[j+1] = bucket[j]
                j -=1
            bucket[j+1] = key

    def binary_search(self, target):
        """
        Binary Search – Quickly searching a sorted array by halving the search space.
        """
        left, right = 0, self.size -1
        while left <= right:
            mid = (left + right) //2
            if self.array[mid] == target:
                return mid
            elif self.array[mid] < target:
                left = mid +1
            else:
                right = mid -1
        return -1

    def linear_search(self, target):
        """
        Linear Search – Scanning each element until a match is found.
        """
        for i in range(self.size):
            if self.array[i] == target:
                return i
        return -1

    def find_minimum(self):
        """
        Find Minimum Element – Locating the smallest element in an array.
        """
        min_val = self.array[0]
        for i in range(1, self.size):
            if self.array[i] < min_val:
                min_val = self.array[i]
        return min_val

    def find_maximum(self):
        """
        Find Maximum Element – Locating the largest element in an array.
        """
        max_val = self.array[0]
        for i in range(1, self.size):
            if self.array[i] > max_val:
                max_val = self.array[i]
        return max_val

    def sum_of_elements(self):
        """
        Sum of Elements – Adding all the values in the array.
        """
        total = 0
        for num in self.array:
            total += num
        return total

    def average_calculation(self):
        """
        Average Calculation – Computing the mean value of array elements.
        """
        total = self.sum_of_elements()
        return total / self.size if self.size > 0 else 0

    def median_finding(self):
        """
        Median Finding – Determining the middle value (after sorting).
        """
        sorted_array = sorted(self.array)
        mid = self.size //2
        if self.size %2 ==0:
            return (sorted_array[mid -1] + sorted_array[mid]) /2
        else:
            return sorted_array[mid]

    def mode_finding(self):
        """
        Mode Finding – Identifying the most frequent element(s).
        """
        frequency = {}
        max_freq = 0
        modes = []
        for num in self.array:
            frequency[num] = frequency.get(num, 0) +1
            if frequency[num] > max_freq:
                max_freq = frequency[num]

        for num, freq in frequency.items():
            if freq == max_freq:
                modes.append(num)
        return modes

    def frequency_count(self):
        """
        Frequency Count – Counting how many times each value appears.
        """
        frequency = {}
        for num in self.array:
            frequency[num] = frequency.get(num, 0) +1
        return frequency

    def remove_duplicates(self):
        """
        Remove Duplicates – Eliminating repeated elements from the array.
        """
        unique_elements = []
        seen = set()
        for num in self.array:
            if num not in seen:
                seen.add(num)
                unique_elements.append(num)
        self.array = unique_elements
        self.size = len(self.array)

    def deep_copy(self):
        """
        Deep Copy of Array – Creating a new array with the same elements (independent copy).
        """
        new_array = self.array[:]
        return new_array


# Dummy test cases
if __name__ == "__main__":
    # Array Initialization
    array_ops = ArrayOperations(size=5, default_value=1)
    print("Initialized array:", array_ops.array)

    # Array Assignment
    array_ops.assign_values([4, 2, 5, 3, 1])
    print("Array after assignment:", array_ops.array)

    # Array Indexing
    print("Element at index 2:", array_ops.index_access(2))

    # Array Traversal
    print("Traversing array:")
    array_ops.traverse_array()

    # Bubble Sort
    array_ops.bubble_sort()
    print("Array after bubble sort:", array_ops.array)

    # Binary Search
    index = array_ops.binary_search(3)
    print("Index of element '3' after binary search:", index)

    # Sum of Elements
    total = array_ops.sum_of_elements()
    print("Sum of elements:", total)

    # Average Calculation
    average = array_ops.average_calculation()
    print("Average of elements:", average)

    # Find Minimum Element
    min_element = array_ops.find_minimum()
    print("Minimum element:", min_element)

    # Find Maximum Element
    max_element = array_ops.find_maximum()
    print("Maximum element:", max_element)

    # Remove Duplicates
    array_ops.assign_values([1, 2, 2, 3, 4, 4, 5])
    array_ops.size = 7  # Adjust size
    array_ops.remove_duplicates()
    print("Array after removing duplicates:", array_ops.array)

    # Deep Copy
    copied_array = array_ops.deep_copy()
    print("Deep copied array:", copied_array)