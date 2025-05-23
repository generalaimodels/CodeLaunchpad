"""


Note: Some complex functions like FFT and DCT are simplified and may not be fully efficient or handle all edge cases. 
The serialization and compression functions use basic file operations and the built-in gzip module, 
which is acceptable as per the default Python library.

Feel free to run the code and observe the outputs from the test cases.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Array Operations Implemented from Scratch.

This module provides a class `ArrayOperations` that implements various array
operations without using external modules.

Author: OpenAI Assistant
Date: 2023-10
"""

from typing import List, Union
import math

class ArrayOperations:
    """
    A class that implements various array operations from scratch.
    """

    def __init__(self, array: List[float]):
        """
        Initialize with an array of numbers.
        """
        self.array = array

    def downcast_elements(self, dtype: str = 'int') -> List[Union[int, float]]:
        """
        Convert elements to a lower-precision type.

        Parameters
        ----------
        dtype : str
            The target data type ('int' or 'float').

        Returns
        -------
        List[Union[int, float]]
            The downcasted array.
        """
        if dtype == 'int':
            return [int(x) for x in self.array]
        elif dtype == 'float':
            return [float(x) for x in self.array]
        else:
            raise ValueError("Unsupported dtype. Use 'int' or 'float'.")

    def clip(self, min_value: float, max_value: float) -> List[float]:
        """
        Constrain values within a specified range.

        Parameters
        ----------
        min_value : float
            The minimum allowed value.
        max_value : float
            The maximum allowed value.

        Returns
        -------
        List[float]
            The clipped array.
        """
        return [max(min_value, min(x, max_value)) for x in self.array]

    def min_max_normalization(self) -> List[float]:
        """
        Scale values to a [0, 1] range.

        Returns
        -------
        List[float]
            The normalized array.
        """
        min_val = min(self.array)
        max_val = max(self.array)
        if max_val - min_val == 0:
            return [0 for _ in self.array]  # Avoid division by zero
        return [(x - min_val) / (max_val - min_val) for x in self.array]

    def z_score_normalization(self) -> List[float]:
        """
        Adjust values to have zero mean and unit variance.

        Returns
        -------
        List[float]
            The standardized array.
        """
        mean = self.mean()
        std_dev = self.std()
        if std_dev == 0:
            return [0 for _ in self.array]  # Avoid division by zero
        return [(x - mean) / std_dev for x in self.array]

    def mean(self) -> float:
        """
        Compute the arithmetic mean.

        Returns
        -------
        float
            The mean of the array.
        """
        return sum(self.array) / len(self.array)

    def median(self) -> float:
        """
        Find the median value.

        Returns
        -------
        float
            The median of the array.
        """
        sorted_array = sorted(self.array)
        n = len(sorted_array)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_array[mid - 1] + sorted_array[mid]) / 2
        else:
            return sorted_array[mid]

    def mode(self) -> List[float]:
        """
        Determine the most frequent element(s).

        Returns
        -------
        List[float]
            The mode(s) of the array.
        """
        frequency = {}
        for item in self.array:
            frequency[item] = frequency.get(item, 0) + 1
        max_freq = max(frequency.values())
        modes = [key for key, val in frequency.items() if val == max_freq]
        return modes

    def variance(self) -> float:
        """
        Measure the dispersion of elements.

        Returns
        -------
        float
            The variance of the array.
        """
        mean = self.mean()
        return sum((x - mean) ** 2 for x in self.array) / len(self.array)

    def std(self) -> float:
        """
        Calculate the spread of the data.

        Returns
        -------
        float
            The standard deviation of the array.
        """
        return math.sqrt(self.variance())

    def total_sum(self) -> float:
        """
        Sum all array elements.

        Returns
        -------
        float
            The total sum of the array.
        """
        return sum(self.array)

    def prod(self) -> float:
        """
        Multiply all elements together.

        Returns
        -------
        float
            The product of the array elements.
        """
        result = 1
        for x in self.array:
            result *= x
        return result

    def cumsum(self) -> List[float]:
        """
        Generate an array of running totals.

        Returns
        -------
        List[float]
            The cumulative sum of the array.
        """
        cumsum_array = []
        total = 0
        for x in self.array:
            total += x
            cumsum_array.append(total)
        return cumsum_array

    def cumprod(self) -> List[float]:
        """
        Generate an array of running products.

        Returns
        -------
        List[float]
            The cumulative product of the array.
        """
        cumprod_array = []
        total = 1
        for x in self.array:
            total *= x
            cumprod_array.append(total)
        return cumprod_array

    @staticmethod
    def dot_product(array1: List[float], array2: List[float]) -> float:
        """
        Compute the dot product between two arrays.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.

        Returns
        -------
        float
            The dot product.
        """
        if len(array1) != len(array2):
            raise ValueError("Arrays must be of the same length.")
        return sum(x * y for x, y in zip(array1, array2))

    @staticmethod
    def cross_product(array1: List[float], array2: List[float]) -> List[float]:
        """
        Calculate the vector cross product (in 3D).

        Parameters
        ----------
        array1 : List[float]
            The first array (3 elements).
        array2 : List[float]
            The second array (3 elements).

        Returns
        -------
        List[float]
            The cross product.
        """
        if len(array1) != 3 or len(array2) != 3:
            raise ValueError("Cross product is defined for 3-dimensional vectors.")
        a1, a2, a3 = array1
        b1, b2, b3 = array2
        return [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]

    @staticmethod
    def outer_product(array1: List[float], array2: List[float]) -> List[List[float]]:
        """
        Produce a matrix from two vectors.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.

        Returns
        -------
        List[List[float]]
            The outer product matrix.
        """
        return [[x * y for y in array2] for x in array1]

    @staticmethod
    def inner_product(array1: List[float], array2: List[float]) -> float:
        """
        Generalized inner product of two arrays.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.

        Returns
        -------
        float
            The inner product.
        """
        return ArrayOperations.dot_product(array1, array2)

    def norm(self, ord: int = 2) -> float:
        """
        Compute the magnitude of an array (vector norm).

        Parameters
        ----------
        ord : int
            The order of the norm (default is 2 for Euclidean norm).

        Returns
        -------
        float
            The vector norm.
        """
        if ord == 1:
            return sum(abs(x) for x in self.array)
        elif ord == 2:
            return math.sqrt(sum(x ** 2 for x in self.array))
        else:
            return sum(abs(x) ** ord for x in self.array) ** (1 / ord)

    @staticmethod
    def euclidean_distance(array1: List[float], array2: List[float]) -> float:
        """
        Compute the Euclidean distance between two arrays.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.

        Returns
        -------
        float
            The Euclidean distance.
        """
        if len(array1) != len(array2):
            raise ValueError("Arrays must be of the same length.")
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(array1, array2)))

    @staticmethod
    def manhattan_distance(array1: List[float], array2: List[float]) -> float:
        """
        Compute the Manhattan distance between two arrays.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.

        Returns
        -------
        float
            The Manhattan distance.
        """
        if len(array1) != len(array2):
            raise ValueError("Arrays must be of the same length.")
        return sum(abs(x - y) for x, y in zip(array1, array2))

    @staticmethod
    def correlation_coefficient(array1: List[float], array2: List[float]) -> float:
        """
        Measure the linear correlation between arrays.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.

        Returns
        -------
        float
            The correlation coefficient.
        """
        if len(array1) != len(array2):
            raise ValueError("Arrays must be of the same length.")

        n = len(array1)
        mean1 = sum(array1) / n
        mean2 = sum(array2) / n
        std1 = math.sqrt(sum((x - mean1) ** 2 for x in array1) / n)
        std2 = math.sqrt(sum((y - mean2) ** 2 for y in array2) / n)
        covariance = sum((x - mean1) * (y - mean2) for x, y in zip(array1, array2)) / n
        if std1 * std2 == 0:
            return 0  # Avoid division by zero
        return covariance / (std1 * std2)

    @staticmethod
    def covariance_matrix(data: List[List[float]]) -> List[List[float]]:
        """
        Calculate the covariance among array variables.

        Parameters
        ----------
        data : List[List[float]]
            A list of arrays (variables).

        Returns
        -------
        List[List[float]]
            The covariance matrix.
        """
        variables = len(data)
        n = len(data[0])
        means = [sum(variable) / n for variable in data]
        cov_matrix = []
        for i in range(variables):
            row = []
            for j in range(variables):
                cov = sum((data[i][k] - means[i]) * (data[j][k] - means[j]) for k in range(n)) / n
                row.append(cov)
            cov_matrix.append(row)
        return cov_matrix

    def histogram(self, bins: int) -> List[int]:
        """
        Bin array data to form a histogram.

        Parameters
        ----------
        bins : int
            The number of bins.

        Returns
        -------
        List[int]
            The histogram counts.
        """
        min_val = min(self.array)
        max_val = max(self.array)
        bin_size = (max_val - min_val) / bins
        histogram = [0] * bins
        for x in self.array:
            index = int((x - min_val) / bin_size)
            if index == bins:
                index -= 1  # Edge case where x == max_val
            histogram[index] += 1
        return histogram

    def data_binning(self, bins: int) -> List[int]:
        """
        Group continuous values into discrete bins.

        Parameters
        ----------
        bins : int
            The number of bins.

        Returns
        -------
        List[int]
            The bin indices for each element.
        """
        min_val = min(self.array)
        max_val = max(self.array)
        bin_size = (max_val - min_val) / bins
        bin_indices = []
        for x in self.array:
            index = int((x - min_val) / bin_size)
            if index == bins:
                index -= 1  # Edge case where x == max_val
            bin_indices.append(index)
        return bin_indices

    def outlier_detection(self, threshold: float = 3.0) -> List[float]:
        """
        Identify anomalous values in the array.

        Parameters
        ----------
        threshold : float
            The number of standard deviations to use.

        Returns
        -------
        List[float]
            The list of outliers.
        """
        mean = self.mean()
        std_dev = self.std()
        outliers = []
        for x in self.array:
            if abs(x - mean) > threshold * std_dev:
                outliers.append(x)
        return outliers

    def moving_average(self, window_size: int) -> List[float]:
        """
        Reduce noise by averaging over a sliding window.

        Parameters
        ----------
        window_size : int
            The size of the moving window.

        Returns
        -------
        List[float]
            The smoothed array.
        """
        n = len(self.array)
        smoothed = []
        for i in range(n - window_size + 1):
            window = self.array[i:i+window_size]
            smoothed.append(sum(window) / window_size)
        return smoothed

    def low_pass_filter(self, alpha: float) -> List[float]:
        """
        Filter high‑frequency components from array data.

        Parameters
        ----------
        alpha : float
            The smoothing factor between 0 and 1.

        Returns
        -------
        List[float]
            The filtered array.
        """
        filtered = [self.array[0]]  # Initialize with first element
        for x in self.array[1:]:
            filtered.append(alpha * x + (1 - alpha) * filtered[-1])
        return filtered

    def fft(self) -> List[complex]:
        """
        Perform Fast Fourier Transform.

        Returns
        -------
        List[complex]
            The FFT of the array.
        """
        n = len(self.array)
        if n <= 1:
            return [complex(self.array[0])]
        elif n % 2 != 0:
            raise ValueError("Size of array must be a power of 2 for FFT.")
        else:
            even = ArrayOperations(self.array[::2]).fft()
            odd = ArrayOperations(self.array[1::2]).fft()
            combined = [0] * n
            for k in range(n // 2):
                t = math.e ** complex(0, -2 * math.pi * k / n) * odd[k]
                combined[k] = even[k] + t
                combined[k + n // 2] = even[k] - t
            return combined

    def ifft(self) -> List[complex]:
        """
        Perform Inverse Fast Fourier Transform.

        Returns
        -------
        List[complex]
            The IFFT of the array.
        """
        n = len(self.array)
        conjugated = [x.conjugate() for x in self.array]
        fft_conj = ArrayOperations(conjugated).fft()
        return [x.conjugate() / n for x in fft_conj]

    def fft_shift(self) -> List[float]:
        """
        Shift the zero‑frequency component to the center.

        Returns
        -------
        List[float]
            The shifted array.
        """
        n = len(self.array)
        mid = n // 2
        return self.array[mid:] + self.array[:mid]

    def dct(self) -> List[float]:
        """
        Perform Discrete Cosine Transform.

        Returns
        -------
        List[float]
            The DCT of the array.
        """
        n = len(self.array)
        result = []
        for k in range(n):
            sum_val = 0
            for i in range(n):
                sum_val += self.array[i] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
            result.append(sum_val)
        return result

    def idct(self) -> List[float]:
        """
        Perform Inverse Discrete Cosine Transform.

        Returns
        -------
        List[float]
            The reconstructed array.
        """
        n = len(self.array)
        result = []
        for i in range(n):
            sum_val = self.array[0] / 2
            for k in range(1, n):
                sum_val += self.array[k] * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
            result.append(sum_val * (2 / n))
        return result

    def serialize(self, filename: str) -> None:
        """
        Save an array to disk in binary format.

        Parameters
        ----------
        filename : str
            The name of the file to save.
        """
        with open(filename, 'wb') as f:
            for num in self.array:
                f.write(num.to_bytes(8, byteorder='little', signed=True))

    @staticmethod
    def deserialize(filename: str) -> List[float]:
        """
        Load a saved array from disk.

        Parameters
        ----------
        filename : str
            The name of the file to load.

        Returns
        -------
        List[float]
            The loaded array.
        """
        array = []
        with open(filename, 'rb') as f:
            while True:
                bytes_read = f.read(8)
                if not bytes_read:
                    break
                num = int.from_bytes(bytes_read, byteorder='little', signed=True)
                array.append(num)
        return array

    def save_compressed(self, filename: str) -> None:
        """
        Save arrays in a compressed archive.

        Parameters
        ----------
        filename : str
            The name of the file to save.
        """
        import gzip
        with gzip.open(filename, 'wt') as f:
            f.write(','.join(map(str, self.array)))

    @staticmethod
    def load_compressed(filename: str) -> List[float]:
        """
        Extract arrays from compressed files.

        Parameters
        ----------
        filename : str
            The name of the file to load.

        Returns
        -------
        List[float]
            The loaded array.
        """
        import gzip
        with gzip.open(filename, 'rt') as f:
            content = f.read()
        return list(map(float, content.split(',')))

    @staticmethod
    def array_equal(array1: List[float], array2: List[float]) -> bool:
        """
        Check if two arrays are exactly equal.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.

        Returns
        -------
        bool
            True if arrays are equal, False otherwise.
        """
        return array1 == array2

    @staticmethod
    def allclose(array1: List[float], array2: List[float], tol: float = 1e-8) -> bool:
        """
        Check if two arrays are nearly equal within tolerance.

        Parameters
        ----------
        array1 : List[float]
            The first array.
        array2 : List[float]
            The second array.
        tol : float
            The tolerance value.

        Returns
        -------
        bool
            True if arrays are nearly equal, False otherwise.
        """
        if len(array1) != len(array2):
            return False
        return all(abs(a - b) <= tol for a, b in zip(array1, array2))

    def argmax(self) -> int:
        """
        Find the index of the maximum element.

        Returns
        -------
        int
            The index of the maximum element.
        """
        max_val = self.array[0]
        index = 0
        for i, x in enumerate(self.array):
            if x > max_val:
                max_val = x
                index = i
        return index

    def argmin(self) -> int:
        """
        Find the index of the minimum element.

        Returns
        -------
        int
            The index of the minimum element.
        """
        min_val = self.array[0]
        index = 0
        for i, x in enumerate(self.array):
            if x < min_val:
                min_val = x
                index = i
        return index

    def count_nonzero(self) -> int:
        """
        Count elements that are not zero.

        Returns
        -------
        int
            The number of non-zero elements.
        """
        return sum(1 for x in self.array if x != 0)

# Dummy test cases
def main():
    # Initialize an array
    arr = ArrayOperations([1, 2, 3, 4, 5])

    print("Original array:", arr.array)
    print("Downcast to int:", arr.downcast_elements('int'))
    print("Clipped array:", arr.clip(2, 4))
    print("Min-Max Normalization:", arr.min_max_normalization())
    print("Z-Score Normalization:", arr.z_score_normalization())
    print("Mean:", arr.mean())
    print("Median:", arr.median())
    print("Mode:", arr.mode())
    print("Variance:", arr.variance())
    print("Standard Deviation:", arr.std())
    print("Total Sum:", arr.total_sum())
    print("Product:", arr.prod())
    print("Cumulative Sum:", arr.cumsum())
    print("Cumulative Product:", arr.cumprod())

    # For operations requiring two arrays
    array1 = [1, 2, 3]
    array2 = [4, 5, 6]

    print("Dot Product:", ArrayOperations.dot_product(array1, array2))
    print("Cross Product:", ArrayOperations.cross_product(array1, array2))
    print("Outer Product:", ArrayOperations.outer_product(array1, array2))
    print("Inner Product:", ArrayOperations.inner_product(array1, array2))

    # Norm and distance calculations
    arr = ArrayOperations([3, 4])
    print("Norm (Euclidean):", arr.norm())
    print("Euclidean Distance:", ArrayOperations.euclidean_distance([1, 2], [4, 6]))
    print("Manhattan Distance:", ArrayOperations.manhattan_distance([1, 2], [4, 6]))

    # Statistical calculations
    array1 = [1, 2, 3, 4, 5]
    array2 = [2, 4, 6, 8, 10]
    print("Correlation Coefficient:", ArrayOperations.correlation_coefficient(array1, array2))
    print("Covariance Matrix:", ArrayOperations.covariance_matrix([array1, array2]))

    # Histogram and binning
    arr = ArrayOperations([1, 2, 2, 3, 4, 4, 5])
    print("Histogram:", arr.histogram(bins=5))
    print("Data Binning:", arr.data_binning(bins=5))

    # Outliers
    arr = ArrayOperations([1, 2, 2, 3, 4, 4, 100])
    print("Outliers:", arr.outlier_detection(threshold=2))

    # Smoothing
    arr = ArrayOperations([1, 2, 3, 4, 5])
    print("Moving Average:", arr.moving_average(window_size=3))
    print("Low-pass Filter:", arr.low_pass_filter(alpha=0.5))

    # FFT and related functions would require further implementation details
    # Skipping the actual execution due to complexity and the need for a power-of-two length array

    # Equality checks
    array1 = [1, 2, 3]
    array2 = [1, 2, 3]
    print("Array Equal:", ArrayOperations.array_equal(array1, array2))
    print("All Close:", ArrayOperations.allclose(array1, array2))

    # Argmax and Argmin
    arr = ArrayOperations([1, 3, 2, 5, 4])
    print("Argmax:", arr.argmax())
    print("Argmin:", arr.argmin())

    # Count non-zero
    arr = ArrayOperations([0, 1, 2, 0, 3])
    print("Count Non-zero:", arr.count_nonzero())

if __name__ == '__main__':
    main()