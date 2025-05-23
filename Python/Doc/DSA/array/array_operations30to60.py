"""
ArrayOperations.py

This module implements various array operations from scratch without using external modules.
The operations are encapsulated within the ArrayOperations class, adhering to PEP 8 standards.
Each method is optimized for time and space complexity, ensuring robustness and scalability.
"""

class ArrayOperations:
    """Class to perform various array operations."""

    def __init__(self, array=None):
        """Initialize the ArrayOperations with an optional array."""
        self.array = array if array is not None else []

    def shallow_copy(self):
        """Perform a shallow copy of the array."""
        return self.array  # Returns the reference to the same array

    def clone_array(self):
        """Clone the array (deep copy)."""
        return [item for item in self.array]

    def reverse_array(self):
        """Reverse the array in place."""
        n = len(self.array)
        for i in range(n // 2):
            self.array[i], self.array[n - i - 1] = self.array[n - i - 1], self.array[i]

    def merge_arrays(self, other_array):
        """Merge the current array with another array."""
        merged_array = [item for item in self.array]
        for item in other_array:
            merged_array.append(item)
        return merged_array

    def concatenate_arrays(self, other_array):
        """Concatenate the current array with another array."""
        return self.merge_arrays(other_array)

    def split_array(self, index):
        """Split the array at the given index into two subarrays."""
        if index < 0 or index > len(self.array):
            raise IndexError("Index out of bounds.")
        return self.array[:index], self.array[index:]

    def slice_array(self, start, end):
        """Slice the array to get elements from start to end indices."""
        if start < 0 or end > len(self.array) or start > end:
            raise IndexError("Invalid slice indices.")
        return [self.array[i] for i in range(start, end)]

    def insert_element(self, index, element):
        """Insert an element at the specified index."""
        if index < 0 or index > len(self.array):
            raise IndexError("Index out of bounds.")
        self.array += [None]
        for i in range(len(self.array) - 1, index, -1):
            self.array[i] = self.array[i - 1]
        self.array[index] = element

    def insert_subarray(self, index, subarray):
        """Insert a subarray into the array at the specified index."""
        if index < 0 or index > len(self.array):
            raise IndexError("Index out of bounds.")
        self.array += [None] * len(subarray)
        for i in range(len(self.array) - 1, index + len(subarray) - 1, -1):
            self.array[i] = self.array[i - len(subarray)]
        for i in range(len(subarray)):
            self.array[index + i] = subarray[i]

    def delete_element(self, index):
        """Delete an element at the specified index."""
        if index < 0 or index >= len(self.array):
            raise IndexError("Index out of bounds.")
        for i in range(index, len(self.array) - 1):
            self.array[i] = self.array[i + 1]
        self.array.pop()

    def delete_subarray(self, start, end):
        """Delete a sequence of elements from start to end indices."""
        if start < 0 or end > len(self.array) or start > end:
            raise IndexError("Invalid indices.")
        del_count = end - start
        for i in range(start, len(self.array) - del_count):
            self.array[i] = self.array[i + del_count]
        for _ in range(del_count):
            self.array.pop()

    def update_element(self, index, value):
        """Update the element at the specified index with a new value."""
        if index < 0 or index >= len(self.array):
            raise IndexError("Index out of bounds.")
        self.array[index] = value

    def batch_update(self, indices, values):
        """Update multiple elements simultaneously."""
        if len(indices) != len(values):
            raise ValueError("Indices and values must have the same length.")
        for idx, val in zip(indices, values):
            self.update_element(idx, val)

    def dynamic_resize(self, new_size, fill_value=None):
        """Adjust the array size at runtime."""
        current_size = len(self.array)
        if new_size < current_size:
            for _ in range(current_size - new_size):
                self.array.pop()
        else:
            self.array += [fill_value] * (new_size - current_size)

    def to_list(self):
        """Convert the array to a list structure."""
        return [item for item in self.array]

    @staticmethod
    def from_list(lst):
        """Build an array from a list."""
        return ArrayOperations([item for item in lst])

    def fill(self, value):
        """Set all elements to the specified value."""
        for i in range(len(self.array)):
            self.array[i] = value

    def clear(self):
        """Reset the array to an empty array."""
        self.array = []

    def replace(self, old_value, new_value):
        """Replace elements with a new value."""
        for i in range(len(self.array)):
            if self.array[i] == old_value:
                self.array[i] = new_value

    def swap_elements(self, index1, index2):
        """Swap the positions of two elements."""
        if index1 < 0 or index1 >= len(self.array) or index2 < 0 or index2 >= len(self.array):
            raise IndexError("Index out of bounds.")
        self.array[index1], self.array[index2] = self.array[index2], self.array[index1]

    def rotate_left(self, k=1):
        """Rotate the array to the left by k positions."""
        n = len(self.array)
        k = k % n
        self.array = self.array[k:] + self.array[:k]

    def rotate_right(self, k=1):
        """Rotate the array to the right by k positions."""
        n = len(self.array)
        k = k % n
        self.array = self.array[-k:] + self.array[:-k]

    def circular_rotate(self, k):
        """Rotate the array in a circular manner by k positions."""
        n = len(self.array)
        k = k % n
        if k < 0:
            k += n
        self.rotate_right(k)

    def fisher_yates_shuffle(self):
        """Randomly shuffle the array elements uniformly."""
        n = len(self.array)
        for i in range(n - 1, 0, -1):
            j = self.random_int(0, i)
            self.array[i], self.array[j] = self.array[j], self.array[i]

    def random_sampling_with_replacement(self, sample_size):
        """Select random elements with replacement."""
        samples = []
        n = len(self.array)
        for _ in range(sample_size):
            idx = self.random_int(0, n - 1)
            samples.append(self.array[idx])
        return samples

    def random_sampling_without_replacement(self, sample_size):
        """Select random elements without replacement."""
        if sample_size > len(self.array):
            raise ValueError("Sample size cannot be greater than array size.")
        temp_array = self.clone_array()
        samples = []
        for _ in range(sample_size):
            idx = self.random_int(0, len(temp_array) - 1)
            samples.append(temp_array[idx])
            temp_array[idx], temp_array[-1] = temp_array[-1], temp_array[idx]
            temp_array.pop()
        return samples

    def generate_permutations(self):
        """Generate all permutations of the array elements."""
        return self.permute(self.array)

    def generate_combinations(self, combination_size):
        """Generate all combinations of the array elements."""
        return self.combine(self.array, combination_size)

    def subarray_sum(self, start, end):
        """Compute the sum of a contiguous subarray."""
        if start < 0 or end > len(self.array) or start > end:
            raise IndexError("Invalid indices.")
        total = 0
        for i in range(start, end):
            total += self.array[i]
        return total

    # Helper methods
    @staticmethod
    def random_int(low, high):
        """Generate a random integer between low and high inclusive."""
        seed = ArrayOperations.seed_generator()
        return low + seed % (high - low + 1)

    @staticmethod
    def seed_generator():
        """Simple seed generator for random number generation."""
        # Using a simple linear congruential generator (LCG)
        seed = 123456789
        a = 1103515245
        c = 12345
        m = 2**31
        seed = (a * seed + c) % m
        return seed

    @staticmethod
    def permute(nums):
        """Generate all permutations of a list."""
        result = []

        def backtrack(start):
            if start == len(nums):
                result.append(nums[:])
            for i in range(start, len(nums)):
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]

        backtrack(0)
        return result

    @staticmethod
    def combine(nums, k):
        """Generate all combinations of size k from nums."""
        result = []

        def backtrack(start, comb):
            if len(comb) == k:
                result.append(comb[:])
                return
            for i in range(start, len(nums)):
                comb.append(nums[i])
                backtrack(i + 1, comb)
                comb.pop()

        backtrack(0, [])
        return result

# Dummy test cases
if __name__ == "__main__":
    # Initialize array
    arr_ops = ArrayOperations([1, 2, 3, 4, 5])

    # Shallow copy
    shallow_copied_array = arr_ops.shallow_copy()
    print("Shallow Copy:", shallow_copied_array)

    # Clone array
    cloned_array = arr_ops.clone_array()
    print("Cloned Array:", cloned_array)

    # Reverse array
    arr_ops.reverse_array()
    print("Reversed Array:", arr_ops.array)

    # Merge arrays
    merged_array = arr_ops.merge_arrays([6, 7, 8])
    print("Merged Array:", merged_array)

    # Concatenate arrays
    concatenated_array = arr_ops.concatenate_arrays([9, 10])
    print("Concatenated Array:", concatenated_array)

    # Split array
    left_subarray, right_subarray = arr_ops.split_array(2)
    print("Left Subarray:", left_subarray)
    print("Right Subarray:", right_subarray)

    # Slice array
    sliced_array = arr_ops.slice_array(1, 4)
    print("Sliced Array:", sliced_array)

    # Insert element
    arr_ops.insert_element(2, 99)
    print("After Insertion:", arr_ops.array)

    # Insert subarray
    arr_ops.insert_subarray(3, [100, 101])
    print("After Subarray Insertion:", arr_ops.array)

    # Delete element
    arr_ops.delete_element(4)
    print("After Deletion:", arr_ops.array)

    # Delete subarray
    arr_ops.delete_subarray(2, 4)
    print("After Subarray Deletion:", arr_ops.array)

    # Update element
    arr_ops.update_element(0, 55)
    print("After Element Update:", arr_ops.array)

    # Batch update
    arr_ops.batch_update([0, 1], [77, 88])
    print("After Batch Update:", arr_ops.array)

    # Dynamic resizing
    arr_ops.dynamic_resize(5, fill_value=0)
    print("After Dynamic Resizing:", arr_ops.array)

    # Convert to list
    list_version = arr_ops.to_list()
    print("Converted to List:", list_version)

    # Create from list
    new_arr_ops = ArrayOperations.from_list([9, 8, 7])
    print("New Array from List:", new_arr_ops.array)

    # Fill operation
    arr_ops.fill(1)
    print("After Fill Operation:", arr_ops.array)

    # Clear operation
    arr_ops.clear()
    print("After Clear Operation:", arr_ops.array)

    # Replace operation
    arr_ops = ArrayOperations([2, 3, 2, 4, 2])
    arr_ops.replace(2, 9)
    print("After Replace Operation:", arr_ops.array)

    # Swap elements
    arr_ops.swap_elements(1, 3)
    print("After Swapping Elements:", arr_ops.array)

    # Rotate left
    arr_ops.rotate_left(2)
    print("After Left Rotation:", arr_ops.array)

    # Rotate right
    arr_ops.rotate_right(1)
    print("After Right Rotation:", arr_ops.array)

    # Circular rotation
    arr_ops.circular_rotate(-3)
    print("After Circular Rotation:", arr_ops.array)

    # Fisher-Yates shuffle
    arr_ops = ArrayOperations([1, 2, 3, 4, 5])
    arr_ops.fisher_yates_shuffle()
    print("After Fisher-Yates Shuffle:", arr_ops.array)

    # Random sampling with replacement
    samples_with_replacement = arr_ops.random_sampling_with_replacement(3)
    print("Random Sampling With Replacement:", samples_with_replacement)

    # Random sampling without replacement
    samples_without_replacement = arr_ops.random_sampling_without_replacement(3)
    print("Random Sampling Without Replacement:", samples_without_replacement)

    # Generate permutations
    permutations = arr_ops.generate_permutations()
    print("Generated Permutations:", permutations)

    # Generate combinations
    combinations = arr_ops.generate_combinations(2)
    print("Generated Combinations:", combinations)

    # Subarray sum calculation
    subarray_sum = arr_ops.subarray_sum(1, 4)
    print("Subarray Sum:", subarray_sum)