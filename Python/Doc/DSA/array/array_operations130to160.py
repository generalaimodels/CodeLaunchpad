class Array:
    """
    A custom array class that mimics some functionalities of NumPy arrays.
    """

    def __init__(self, data):
        """
        Initialize the Array with the provided data.

        :param data: A nested list representing the array elements.
        """
        self.data = data
        self.shape = self._compute_shape(self.data)
        self.dtype = type(self.data[0][0]) if isinstance(self.data[0], list) else type(self.data[0])

    def _compute_shape(self, data):
        """
        Compute the shape of the array.

        :param data: The data to compute the shape for.
        :return: A tuple representing the shape.
        """
        if isinstance(data, list):
            if not data:
                return (0,)
            elif isinstance(data[0], list):
                inner_shape = self._compute_shape(data[0])
                return (len(data),) + inner_shape
            else:
                return (len(data),)
        else:
            return ()

    def _apply_elementwise_operation(self, other, operation):
        """
        Apply an element-wise operation between two arrays.

        :param other: The other array or scalar.
        :param operation: A function that defines the operation.
        :return: A new Array with the result.
        """
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes of both arrays must be the same for element-wise operations.")
            result = self._elementwise_op(self.data, other.data, operation)
        else:
            result = self._elementwise_op_scalar(self.data, other, operation)
        return Array(result)

    def _elementwise_op(self, data1, data2, operation):
        """
        Recursively apply an element-wise operation to nested lists.

        :param data1: The first data list.
        :param data2: The second data list.
        :param operation: The operation to apply.
        :return: The result of the operation.
        """
        if isinstance(data1, list):
            return [self._elementwise_op(d1, d2, operation) for d1, d2 in zip(data1, data2)]
        else:
            return operation(data1, data2)

    def _elementwise_op_scalar(self, data, scalar, operation):
        """
        Apply an element-wise operation between the data and a scalar.

        :param data: The data list.
        :param scalar: The scalar value.
        :param operation: The operation to apply.
        :return: The result of the operation.
        """
        if isinstance(data, list):
            return [self._elementwise_op_scalar(d, scalar, operation) for d in data]
        else:
            return operation(data, scalar)

    # Element-wise Equality Comparison
    def elementwise_equal(self, other):
        """
        Compare elements for equality.

        :param other: Another Array or scalar.
        :return: An Array of booleans.
        """
        return self._apply_elementwise_operation(other, lambda x, y: x == y)

    # Element-wise Inequality Comparison
    def elementwise_not_equal(self, other):
        """
        Check for non-equality of elements.

        :param other: Another Array or scalar.
        :return: An Array of booleans.
        """
        return self._apply_elementwise_operation(other, lambda x, y: x != y)

    # Element-wise Greater-Than Comparison
    def elementwise_greater(self, other):
        """
        Compare if elements exceed corresponding elements.

        :param other: Another Array or scalar.
        :return: An Array of booleans.
        """
        return self._apply_elementwise_operation(other, lambda x, y: x > y)

    # Element-wise Less-Than Comparison
    def elementwise_less(self, other):
        """
        Compare if elements are less than corresponding ones.

        :param other: Another Array or scalar.
        :return: An Array of booleans.
        """
        return self._apply_elementwise_operation(other, lambda x, y: x < y)

    # Element-wise Greater-Than or Equal Comparison
    def elementwise_greater_equal(self, other):
        """
        Compare for greater than or equal to.

        :param other: Another Array or scalar.
        :return: An Array of booleans.
        """
        return self._apply_elementwise_operation(other, lambda x, y: x >= y)

    # Element-wise Less-Than or Equal Comparison
    def elementwise_less_equal(self, other):
        """
        Compare for less than or equal to.

        :param other: Another Array or scalar.
        :return: An Array of booleans.
        """
        return self._apply_elementwise_operation(other, lambda x, y: x <= y)

    # Boolean Masking
    def boolean_mask(self, condition_array):
        """
        Select elements that satisfy a condition.

        :param condition_array: An Array of booleans with the same shape.
        :return: A flat list of elements that satisfy the condition.
        """
        if self.shape != condition_array.shape:
            raise ValueError("Shape of condition array must match the shape of the array.")
        result = []
        self._apply_boolean_mask(self.data, condition_array.data, result)
        return result

    def _apply_boolean_mask(self, data, condition, result):
        """
        Recursively apply boolean masking.

        :param data: The data list.
        :param condition: The condition list.
        :param result: The list to store results.
        """
        if isinstance(data, list):
            for d, c in zip(data, condition):
                self._apply_boolean_mask(d, c, result)
        else:
            if condition:
                result.append(data)

    # Fancy Indexing
    def fancy_indexing(self, indices):
        """
        Index using a list of indices.

        :param indices: A list of indices.
        :return: A new Array with selected elements.
        """
        selected = [self.data[i] for i in indices]
        return Array(selected)

    # Resizing
    def resize(self, new_shape):
        """
        Change the array's size.

        :param new_shape: The new shape as a tuple.
        """
        flat_data = self.flatten()
        total_elements = self._product(new_shape)
        if total_elements > len(flat_data):
            flat_data.extend([0] * (total_elements - len(flat_data)))
        else:
            flat_data = flat_data[:total_elements]
        self.data = self._reshape(flat_data, new_shape)
        self.shape = new_shape

    def _product(self, shape):
        """
        Compute the product of the dimensions in shape.

        :param shape: The shape tuple.
        :return: The product of dimensions.
        """
        result = 1
        for dim in shape:
            result *= dim
        return result

    def flatten(self):
        """
        Flatten the array.

        :return: A flat list of elements.
        """
        return self._flatten(self.data)

    def _flatten(self, data):
        """
        Recursively flatten the data.

        :param data: The data to flatten.
        :return: A flat list.
        """
        if isinstance(data, list):
            result = []
            for item in data:
                result.extend(self._flatten(item))
            return result
        else:
            return [data]

    # Reshaping
    def reshape(self, new_shape):
        """
        Reshape the array.

        :param new_shape: The new shape as a tuple.
        :return: A new Array with reshaped data.
        """
        flat_data = self.flatten()
        total_elements = self._product(new_shape)
        if total_elements != len(flat_data):
            raise ValueError("Total size of new array must be unchanged.")
        new_data = self._reshape(flat_data, new_shape)
        return Array(new_data)

    def _reshape(self, flat_data, shape):
        """
        Recursively reshape the flat data into the given shape.

        :param flat_data: The flat data list.
        :param shape: The shape tuple.
        :return: The reshaped data.
        """
        if not shape:
            return flat_data.pop(0)
        size = shape[0]
        return [self._reshape(flat_data, shape[1:]) for _ in range(size)]

    # Appending Elements
    def append(self, values, axis=None):
        """
        Append values to the array.

        :param values: Values to append.
        :param axis: The axis along which to append.
        :return: A new Array with appended values.
        """
        if axis is None:
            new_data = self.flatten() + values.flatten()
            return Array(new_data)
        else:
            new_data = self._append_along_axis(self.data, values.data, axis)
            return Array(new_data)

    def _append_along_axis(self, data1, data2, axis):
        """
        Append data2 to data1 along the specified axis.

        :param data1: The first data list.
        :param data2: The data to append.
        :param axis: The axis along which to append.
        :return: The combined data.
        """
        if axis == 0:
            return data1 + data2
        else:
            return [self._append_along_axis(d1, d2, axis - 1) for d1, d2 in zip(data1, data2)]

    # Insertion
    def insert(self, index, values, axis=None):
        """
        Insert values at the specified index.

        :param index: The index at which to insert.
        :param values: The values to insert.
        :param axis: The axis along which to insert.
        :return: A new Array with inserted values.
        """
        if axis is None:
            flat_data = self.flatten()
            flat_values = values.flatten()
            new_data = flat_data[:index] + flat_values + flat_data[index:]
            return Array(new_data)
        else:
            new_data = self._insert_along_axis(self.data, index, values.data, axis)
            return Array(new_data)

    def _insert_along_axis(self, data, index, values, axis):
        """
        Insert values into data along the specified axis.

        :param data: The original data.
        :param index: The index at which to insert.
        :param values: The values to insert.
        :param axis: The axis along which to insert.
        :return: The data with values inserted.
        """
        if axis == 0:
            return data[:index] + values + data[index:]
        else:
            return [self._insert_along_axis(d, index, v, axis - 1) for d, v in zip(data, values)]

    # Deletion
    def delete(self, index, axis=None):
        """
        Delete elements at the specified index.

        :param index: The index of elements to delete.
        :param axis: The axis along which to delete.
        :return: A new Array with elements deleted.
        """
        if axis is None:
            flat_data = self.flatten()
            if isinstance(index, int):
                del flat_data[index]
            elif isinstance(index, slice):
                del flat_data[index]
            else:
                for idx in sorted(index, reverse=True):
                    del flat_data[idx]
            return Array(flat_data)
        else:
            new_data = self._delete_along_axis(self.data, index, axis)
            return Array(new_data)

    def _delete_along_axis(self, data, index, axis):
        """
        Delete elements along the specified axis.

        :param data: The data list.
        :param index: The index or indices to delete.
        :param axis: The axis along which to delete.
        :return: The data with elements deleted.
        """
        if axis == 0:
            if isinstance(index, int):
                return data[:index] + data[index + 1:]
            elif isinstance(index, slice):
                return data[:index.start] + data[index.stop:]
            else:
                return [d for i, d in enumerate(data) if i not in index]
        else:
            return [self._delete_along_axis(d, index, axis - 1) for d in data]

    # Finding Unique Elements
    def unique(self):
        """
        Extract the set of unique values.

        :return: A list of unique elements.
        """
        flat_data = self.flatten()
        seen = set()
        unique_elements = []
        for item in flat_data:
            if item not in seen:
                seen.add(item)
                unique_elements.append(item)
        return unique_elements

    # Sorting
    def sort(self, axis=-1):
        """
        Sort the array.

        :param axis: The axis along which to sort.
        """
        if axis == -1 or axis == self._get_max_axis():
            flat_data = self.flatten()
            flat_data.sort()
            self.data = self._reshape(flat_data, self.shape)
        else:
            self._sort_along_axis(self.data, axis)

    def _get_max_axis(self):
        """
        Get the maximum axis index.

        :return: The maximum axis.
        """
        return len(self.shape) - 1

    def _sort_along_axis(self, data, axis):
        """
        Recursively sort along the specified axis.

        :param data: The data list.
        :param axis: The axis along which to sort.
        """
        if axis == 0:
            data.sort()
        else:
            for sublist in data:
                self._sort_along_axis(sublist, axis - 1)

    # Concatenation
    @staticmethod
    def concatenate(arrays, axis=0):
        """
        Join arrays along an axis.

        :param arrays: A list of Array instances.
        :param axis: The axis along which to concatenate.
        :return: A new Array with concatenated data.
        """
        data_list = [array.data for array in arrays]
        concatenated_data = Array._concatenate_along_axis(data_list, axis)
        return Array(concatenated_data)

    @staticmethod
    def _concatenate_along_axis(data_list, axis):
        """
        Concatenate data along the specified axis.

        :param data_list: A list of data lists.
        :param axis: The axis along which to concatenate.
        :return: The concatenated data.
        """
        if axis == 0:
            result = []
            for data in data_list:
                result.extend(data)
            return result
        else:
            return [Array._concatenate_along_axis([d[i] for d in data_list], axis - 1)
                    for i in range(len(data_list[0]))]

    # Splitting
    def split(self, indices_or_sections, axis=0):
        """
        Divide the array into multiple subarrays.

        :param indices_or_sections: Indices or number of sections to split into.
        :param axis: The axis along which to split.
        :return: A list of Arrays.
        """
        if isinstance(indices_or_sections, int):
            total_length = self.shape[axis]
            if total_length % indices_or_sections != 0:
                raise ValueError("Array split does not result in an equal division.")
            section_size = total_length // indices_or_sections
            indices = [i * section_size for i in range(1, indices_or_sections)]
        else:
            indices = indices_or_sections

        split_data = self._split_along_axis(self.data, indices, axis)
        return [Array(d) for d in split_data]

    def _split_along_axis(self, data, indices, axis):
        """
        Recursively split data along the specified axis.

        :param data: The data list.
        :param indices: Indices at which to split.
        :param axis: The axis along which to split.
        :return: A list of data sections.
        """
        if axis == 0:
            sections = []
            prev_index = 0
            for index in indices:
                sections.append(data[prev_index:index])
                prev_index = index
            sections.append(data[prev_index:])
            return sections
        else:
            return [self._split_along_axis(sublist, indices, axis - 1) for sublist in data]

    # Vertical Stacking
    @staticmethod
    def vstack(arrays):
        """
        Stack arrays vertically.

        :param arrays: A list of Arrays.
        :return: A new Array with vertically stacked data.
        """
        return Array.concatenate(arrays, axis=0)

    # Horizontal Stacking
    @staticmethod
    def hstack(arrays):
        """
        Stack arrays horizontally.

        :param arrays: A list of Arrays.
        :return: A new Array with horizontally stacked data.
        """
        return Array.concatenate(arrays, axis=1)

    # Rolling
    def roll(self, shift, axis=None):
        """
        Shift array elements along a given axis.

        :param shift: The number of places by which elements are shifted.
        :param axis: The axis along which elements are shifted.
        """
        if axis is None:
            flat_data = self.flatten()
            shift %= len(flat_data)
            rolled_data = flat_data[-shift:] + flat_data[:-shift]
            self.data = self._reshape(rolled_data, self.shape)
        else:
            self._roll_along_axis(self.data, shift, axis)

    def _roll_along_axis(self, data, shift, axis):
        """
        Recursively roll data along the specified axis.

        :param data: The data list.
        :param shift: The shift amount.
        :param axis: The axis along which to roll.
        """
        if axis == 0:
            shift %= len(data)
            data[:] = data[-shift:] + data[:-shift]
        else:
            for sublist in data:
                self._roll_along_axis(sublist, shift, axis - 1)

    # Flipping
    def flip(self, axis=None):
        """
        Reverse the order of elements along an axis.

        :param axis: The axis along which to flip.
        """
        if axis is None:
            flat_data = self.flatten()
            flat_data.reverse()
            self.data = self._reshape(flat_data, self.shape)
        else:
            self._flip_along_axis(self.data, axis)

    def _flip_along_axis(self, data, axis):
        """
        Recursively flip data along the specified axis.

        :param data: The data list.
        :param axis: The axis along which to flip.
        """
        if axis == 0:
            data.reverse()
        else:
            for sublist in data:
                self._flip_along_axis(sublist, axis - 1)

    # Rotating
    def rot90(self, k=1, axes=(0, 1)):
        """
        Rotate an array by 90-degree increments.

        :param k: Number of times the array is rotated by 90 degrees.
        :param axes: The plane in which to rotate.
        """
        for _ in range(k % 4):
            self.transpose(axes)
            self.flip(axes[0])

    def transpose(self, axes):
        """
        Transpose the array along specified axes.

        :param axes: Tuple of two axes to transpose.
        """
        if len(self.shape) < 2:
            return
        axis1, axis2 = axes
        self.data = self._transpose_axes(self.data, axis1, axis2, 0)
        shape_list = list(self.shape)
        shape_list[axis1], shape_list[axis2] = shape_list[axis2], shape_list[axis1]
        self.shape = tuple(shape_list)

    def _transpose_axes(self, data, axis1, axis2, current_axis):
        """
        Recursively transpose data along specified axes.

        :param data: The data to transpose.
        :param axis1: The first axis.
        :param axis2: The second axis.
        :param current_axis: The current axis in recursion.
        :return: The transposed data.
        """
        if current_axis == axis1:
            return [self._transpose_axes(d, axis1, axis2, current_axis + 1) for d in zip(*data)]
        elif current_axis == axis2:
            return [self._transpose_axes(d, axis1, axis2, current_axis + 1) for d in data]
        else:
            return [self._transpose_axes(d, axis1, axis2, current_axis + 1) for d in data]

    # Meshgrid Creation
    @staticmethod
    def meshgrid(*arrays):
        """
        Generate coordinate matrices from coordinate vectors.

        :param arrays: Coordinate vectors.
        :return: A list of Arrays representing the meshgrid.
        """
        from itertools import product
        grids = []
        shape = [len(a.data) for a in arrays]
        for i in range(len(arrays)):
            grid = []
            for idx in product(*[range(len(a.data)) for a in arrays]):
                grid.append(arrays[i].data[idx[i]])
            grids.append(Array(Array._reshape(grid, shape)))
        return grids

    # Scalar Broadcasting
    def scalar_broadcast(self, scalar, operation):
        """
        Apply a scalar operation to every element.

        :param scalar: The scalar value.
        :param operation: The operation to apply.
        :return: A new Array with the result.
        """
        return self._apply_elementwise_operation(scalar, operation)

    # Boolean Indexing
    def boolean_indexing(self, condition):
        """
        Index the array using a boolean condition.

        :param condition: A function that returns True or False for a given element.
        :return: A flat list of elements that satisfy the condition.
        """
        flat_data = self.flatten()
        return [x for x in flat_data if condition(x)]

    # Copying
    def copy(self):
        """
        Create an independent copy of the array.

        :return: A new Array with copied data.
        """
        import copy
        return Array(copy.deepcopy(self.data))

    # Data Type Conversion
    def astype(self, new_type):
        """
        Change the type of array elements.

        :param new_type: The new data type.
        """
        self.data = self._change_type(self.data, new_type)
        self.dtype = new_type

    def _change_type(self, data, new_type):
        """
        Recursively change the type of data.

        :param data: The data to change.
        :param new_type: The new data type.
        :return: Data with changed type.
        """
        if isinstance(data, list):
            return [self._change_type(d, new_type) for d in data]
        else:
            return new_type(data)

    # Upcasting Elements
    def upcast(self, higher_precision_type):
        """
        Convert elements to a higher-precision type.

        :param higher_precision_type: The target data type.
        """
        self.astype(higher_precision_type)


# Dummy Test Cases
if __name__ == "__main__":
    # Initialize Arrays
    array1 = Array([[1, 2], [3, 4]])
    array2 = Array([[4, 3], [2, 1]])

    # Element-wise Equality Comparison
    print("Element-wise Equality:", array1.elementwise_equal(array2).data)

    # Element-wise Inequality Comparison
    print("Element-wise Inequality:", array1.elementwise_not_equal(array2).data)

    # Element-wise Greater Than Comparison
    print("Element-wise Greater Than:", array1.elementwise_greater(array2).data)

    # Element-wise Less Than Comparison
    print("Element-wise Less Than:", array1.elementwise_less(array2).data)

    # Boolean Masking
    condition = array1.elementwise_greater(2)
    print("Boolean Masking:", array1.boolean_mask(condition))

    # Fancy Indexing
    print("Fancy Indexing:", array1.fancy_indexing([0]).data)

    # Resizing
    array1.resize((4,))
    print("Resized Array:", array1.data)

    # Reshaping
    reshaped_array = array1.reshape((2, 2))
    print("Reshaped Array:", reshaped_array.data)

    # Appending Elements
    appended_array = array1.append(Array([5, 6]))
    print("Appended Array:", appended_array.data)

    # Insertion
    inserted_array = array1.insert(1, Array([7, 8]))
    print("Inserted Array:", inserted_array.data)

    # Deletion
    deleted_array = array1.delete(1)
    print("Deleted Array:", deleted_array.data)

    # Finding Unique Elements
    print("Unique Elements:", array1.unique())

    # Sorting
    array1.sort()
    print("Sorted Array:", array1.data)

    # Concatenation
    concatenated_array = Array.concatenate([array1, array2])
    print("Concatenated Array:", concatenated_array.data)

    # Splitting
    split_arrays = array1.split(2)
    print("Split Arrays:", [arr.data for arr in split_arrays])

    # Vertical Stacking
    vstacked_array = Array.vstack([array1, array2])
    print("Vertically Stacked Array:", vstacked_array.data)

    # # Horizontal Stacking
    # hstacked_array = Array.hstack([array1, array2])
    # print("Horizontally Stacked Array:", hstacked_array.data)

    # Rolling
    array1.roll(1)
    print("Rolled Array:", array1.data)

    # Flipping
    array1.flip()
    print("Flipped Array:", array1.data)

    # Rotating
    array1.rot90()
    print("Rotated Array:", array1.data)

    # Scalar Broadcasting
    broadcasted_array = array1.scalar_broadcast(2, lambda x, y: x * y)
    print("Scalar Broadcasted Array:", broadcasted_array.data)

    # Boolean Indexing
    boolean_indexed = array1.boolean_indexing(lambda x: x > 2)
    print("Boolean Indexed Elements:", boolean_indexed)

    # Copying
    copied_array = array1.copy()
    print("Copied Array:", copied_array.data)

    # Data Type Conversion
    array1.astype(float)
    print("Data Type Converted Array:", array1.data)

    # Upcasting Elements
    array1.upcast(complex)
    print("Upcasted Array:", array1.data)


# @staticmethod
# def _concatenate_along_axis(data_list, axis):
#     """
#     Concatenate data along the specified axis.

#     :param data_list: A list of data lists.
#     :param axis: The axis along which to concatenate.
#     :return: The concatenated data.
#     """
#     if not data_list:
#         return []
#     if axis == 0:
#         result = []
#         for data in data_list:
#             if isinstance(data, (list, tuple)):
#                 result.extend(data)
#             else:
#                 result.append(data)
#         return result
#     else:
#         if isinstance(data_list[0], (list, tuple)):
#             min_len = min(len(d) for d in data_list)
#             return [Array._concatenate_along_axis([d[i] for d in data_list], axis - 1)
#                     for i in range(min_len)]
#         else:
#             # When data_list contains scalars and axis > 0, can't index further
#             # So return data_list as is
#             return data_list