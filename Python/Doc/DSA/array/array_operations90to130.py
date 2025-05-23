class Matrix:
    """
    A class to represent a 2D matrix and perform various operations.
    """

    def __init__(self, data):
        """
        Initialize the Matrix with a 2D list.

        :param data: List of lists where each sublist represents a row.
        """
        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same number of columns.")
        self.data = [row[:] for row in data]
        self.rows = len(data)
        self.cols = len(data[0])

    def submatrix(self, start_row, end_row, start_col, end_col):
        """
        Extract a submatrix from the matrix.

        :param start_row: Starting row index.
        :param end_row: Ending row index.
        :param start_col: Starting column index.
        :param end_col: Ending column index.
        :return: New Matrix instance of the submatrix.
        """
        sub = [
            self.data[i][start_col:end_col]
            for i in range(start_row, end_row)
        ]
        return Matrix(sub)

    def diagonal_traversal(self):
        """
        Access the main diagonal of the matrix.
        """
        return [self.data[i][i] for i in range(min(self.rows, self.cols))]

    def anti_diagonal_traversal(self):
        """
        Access the anti-diagonal of the matrix.
        """
        return [self.data[i][self.cols - i - 1] for i in range(min(self.rows, self.cols))]

    def spiral_traversal(self):
        """
        Traverse the matrix in a spiral order from the outer edge inward.
        """
        result = []
        left, right = 0, self.cols - 1
        top, bottom = 0, self.rows - 1

        while left <= right and top <= bottom:
            # Top row
            for col in range(left, right + 1):
                result.append(self.data[top][col])
            top += 1

            # Right column
            for row in range(top, bottom + 1):
                result.append(self.data[row][right])
            right -= 1

            if top <= bottom:
                # Bottom row
                for col in range(right, left - 1, -1):
                    result.append(self.data[bottom][col])
                bottom -= 1

            if left <= right:
                # Left column
                for row in range(bottom, top - 1, -1):
                    result.append(self.data[row][left])
                left += 1

        return result

    def zigzag_traversal(self):
        """
        Traverse the matrix in a zigzag order.
        """
        result = []
        for i, row in enumerate(self.data):
            if i % 2 == 0:
                result.extend(row)
            else:
                result.extend(row[::-1])
        return result

    def diagonal_sum(self):
        """
        Calculate the sum of the main diagonal elements.
        """
        return sum(self.data[i][i] for i in range(min(self.rows, self.cols)))

    def boundary_sum(self):
        """
        Calculate the sum of the boundary elements.
        """
        if self.rows == 1:
            return sum(self.data[0])
        if self.cols == 1:
            return sum(row[0] for row in self.data)
        top = sum(self.data[0])
        bottom = sum(self.data[-1])
        middle = sum(
            self.data[i][0] + self.data[i][-1]
            for i in range(1, self.rows - 1)
        )
        return top + bottom + middle

    def insert_row(self, index, row):
        """
        Insert a new row into the matrix.

        :param index: Index at which to insert the row.
        :param row: List representing the new row.
        """
        if len(row) != self.cols:
            raise ValueError("Row length must be equal to the number of columns.")
        self.data.insert(index, row[:])
        self.rows += 1

    def delete_row(self, index):
        """
        Delete a row from the matrix.

        :param index: Index of the row to delete.
        """
        if self.rows == 0:
            raise IndexError("No rows to delete.")
        del self.data[index]
        self.rows -= 1

    def insert_column(self, index, column):
        """
        Insert a new column into the matrix.

        :param index: Index at which to insert the column.
        :param column: List representing the new column.
        """
        if len(column) != self.rows:
            raise ValueError("Column length must be equal to the number of rows.")
        for i in range(self.rows):
            self.data[i].insert(index, column[i])
        self.cols += 1

    def delete_column(self, index):
        """
        Delete a column from the matrix.

        :param index: Index of the column to delete.
        """
        if self.cols == 0:
            raise IndexError("No columns to delete.")
        for row in self.data:
            del row[index]
        self.cols -= 1

    def sort_rows(self):
        """
        Sort each row individually.
        """
        for row in self.data:
            row.sort()

    def sort_columns(self):
        """
        Sort each column individually.
        """
        for col_idx in range(self.cols):
            column = [self.data[row_idx][col_idx] for row_idx in range(self.rows)]
            column.sort()
            for row_idx in range(self.rows):
                self.data[row_idx][col_idx] = column[row_idx]

    @staticmethod
    def merge_matrices(matrix_a, matrix_b):
        """
        Merge two matrices into one larger matrix.

        :param matrix_a: First Matrix instance.
        :param matrix_b: Second Matrix instance.
        :return: New Matrix instance.
        """
        if matrix_a.rows != matrix_b.rows:
            raise ValueError("Matrices must have the same number of rows to merge.")
        merged_data = [matrix_a.data[i] + matrix_b.data[i] for i in range(matrix_a.rows)]
        return Matrix(merged_data)

    @staticmethod
    def horizontal_concatenation(matrix_a, matrix_b):
        """
        Horizontally concatenate two matrices.

        :param matrix_a: First Matrix instance.
        :param matrix_b: Second Matrix instance.
        :return: New Matrix instance.
        """
        return Matrix.merge_matrices(matrix_a, matrix_b)

    @staticmethod
    def vertical_concatenation(matrix_a, matrix_b):
        """
        Vertically concatenate two matrices.

        :param matrix_a: First Matrix instance.
        :param matrix_b: Second Matrix instance.
        :return: New Matrix instance.
        """
        if matrix_a.cols != matrix_b.cols:
            raise ValueError("Matrices must have the same number of columns to concatenate.")
        new_data = matrix_a.data + matrix_b.data
        return Matrix(new_data)

    def access_main_diagonal(self):
        """
        Extract the main diagonal elements.
        """
        return self.diagonal_traversal()

    def rotate_90_degrees(self, clockwise=True):
        """
        Rotate the matrix by 90 degrees.

        :param clockwise: Direction of rotation.
        :return: New Matrix instance.
        """
        if clockwise:
            rotated = [
                [self.data[self.rows - j - 1][i] for j in range(self.rows)]
                for i in range(self.cols)
            ]
        else:
            rotated = [
                [self.data[j][self.cols - i - 1] for j in range(self.rows)]
                for i in range(self.cols)
            ]
        return Matrix(rotated)

    def rotate_180_degrees(self):
        """
        Rotate the matrix by 180 degrees.
        """
        rotated = [row[::-1] for row in self.data[::-1]]
        return Matrix(rotated)

    def rotate_270_degrees(self, clockwise=True):
        """
        Rotate the matrix by 270 degrees.

        :param clockwise: Direction of rotation.
        :return: New Matrix instance.
        """
        return self.rotate_90_degrees(clockwise=not clockwise)

    def flip_horizontally(self):
        """
        Mirror the matrix along a vertical axis.
        """
        flipped = [row[::-1] for row in self.data]
        return Matrix(flipped)

    def flip_vertically(self):
        """
        Mirror the matrix along a horizontal axis.
        """
        flipped = self.data[::-1]
        return Matrix(flipped)

    def element_wise_operation(self, other, operation):
        """
        Perform element-wise operation with another matrix.

        :param other: Another Matrix instance.
        :param operation: Function that defines the operation.
        :return: New Matrix instance.
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must be of the same dimensions for element-wise operations.")
        new_data = [
            [operation(self.data[i][j], other.data[i][j]) for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(new_data)

    def element_wise_addition(self, other):
        """
        Element-wise addition with another matrix.
        """
        return self.element_wise_operation(other, lambda a, b: a + b)

    def element_wise_subtraction(self, other):
        """
        Element-wise subtraction with another matrix.
        """
        return self.element_wise_operation(other, lambda a, b: a - b)

    def element_wise_multiplication(self, other):
        """
        Element-wise multiplication with another matrix.
        """
        return self.element_wise_operation(other, lambda a, b: a * b)

    def element_wise_division(self, other):
        """
        Element-wise division with another matrix.
        """
        return self.element_wise_operation(other, lambda a, b: a / b if b != 0 else float('inf'))

    def element_wise_power(self, power):
        """
        Raise each element to a specified power.

        :param power: The exponent value.
        :return: New Matrix instance.
        """
        new_data = [
            [self.data[i][j] ** power for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(new_data)

    def element_wise_modulo(self, mod):
        """
        Apply modulo operation to each element.
        """
        new_data = [
            [self.data[i][j] % mod for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(new_data)

    def logical_and(self, other):
        """
        Perform a bit-wise AND across arrays.
        """
        return self.element_wise_operation(other, lambda a, b: a & b)

    def logical_or(self, other):
        """
        Perform a bit-wise OR across arrays.
        """
        return self.element_wise_operation(other, lambda a, b: a | b)

    def __str__(self):
        """
        String representation of the matrix.
        """
        return '\n'.join([' '.join(map(str, row)) for row in self.data])

    # Additional methods can be implemented similarly...

# Dummy test cases
def main():
    # Initialize matrices
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[5, 6], [7, 8]])

    # Submatrix Extraction
    submatrix = matrix1.submatrix(0, 2, 0, 2)
    print("Submatrix:")
    print(submatrix)

    # Diagonal Traversal
    diagonal = matrix1.diagonal_traversal()
    print("Main Diagonal:", diagonal)

    # Anti-Diagonal Traversal
    anti_diagonal = matrix1.anti_diagonal_traversal()
    print("Anti-Diagonal:", anti_diagonal)

    # Spiral Traversal
    spiral = matrix1.spiral_traversal()
    print("Spiral Traversal:", spiral)

    # Zigzag Traversal
    zigzag = matrix1.zigzag_traversal()
    print("Zigzag Traversal:", zigzag)

    # Diagonal Sum
    diag_sum = matrix1.diagonal_sum()
    print("Diagonal Sum:", diag_sum)

    # Boundary Sum
    boundary_sum = matrix1.boundary_sum()
    print("Boundary Sum:", boundary_sum)

    # Insert Row
    matrix1.insert_row(1, [9, 9])
    print("After Inserting Row:")
    print(matrix1)

    # Delete Row
    matrix1.delete_row(1)
    print("After Deleting Row:")
    print(matrix1)

    # Insert Column
    matrix1.insert_column(1, [8, 8])
    print("After Inserting Column:")
    print(matrix1)

    # Delete Column
    matrix1.delete_column(1)
    print("After Deleting Column:")
    print(matrix1)

    # Sort Rows
    matrix3 = Matrix([[3, 1], [4, 2]])
    matrix3.sort_rows()
    print("After Sorting Rows:")
    print(matrix3)

    # Sort Columns
    matrix3.sort_columns()
    print("After Sorting Columns:")
    print(matrix3)

    # Merge Two 2D Arrays
    merged_matrix = Matrix.merge_matrices(matrix1, matrix2)
    print("Merged Matrix:")
    print(merged_matrix)

    # Horizontal Concatenation
    horizontal_concat = Matrix.horizontal_concatenation(matrix1, matrix2)
    print("Horizontal Concatenation:")
    print(horizontal_concat)

    # Vertical Concatenation
    vertical_concat = Matrix.vertical_concatenation(matrix1, matrix2)
    print("Vertical Concatenation:")
    print(vertical_concat)

    # Rotate Matrix by 90°
    rotated_matrix_90 = matrix1.rotate_90_degrees()
    print("Rotated Matrix by 90°:")
    print(rotated_matrix_90)

    # Flip Matrix Horizontally
    flipped_horizontal = matrix1.flip_horizontally()
    print("Flipped Horizontally:")
    print(flipped_horizontal)

    # Element-wise Addition
    added_matrix = matrix1.element_wise_addition(matrix2)
    print("Element-wise Addition:")
    print(added_matrix)

    # Element-wise Subtraction
    subtracted_matrix = matrix1.element_wise_subtraction(matrix2)
    print("Element-wise Subtraction:")
    print(subtracted_matrix)

    # Element-wise Multiplication
    multiplied_matrix = matrix1.element_wise_multiplication(matrix2)
    print("Element-wise Multiplication:")
    print(multiplied_matrix)

    # Element-wise Division
    divided_matrix = matrix1.element_wise_division(matrix2)
    print("Element-wise Division:")
    print(divided_matrix)

    # Logical AND
    and_matrix = matrix1.logical_and(matrix2)
    print("Logical AND:")
    print(and_matrix)

    # Logical OR
    or_matrix = matrix1.logical_or(matrix2)
    print("Logical OR:")
    print(or_matrix)

if __name__ == "__main__":
    main()