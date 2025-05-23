/*******************************************************
 * Chapter 12: Divide and Conquer - Break and Rule 📚⚔️ *
 * Understanding Divide and Conquer through Examples   *

 *******************************************************/

 #include <iostream>
 #include <vector>
 #include <algorithm>
 #include <cmath>
 #include <climits>
 #include <cstring>
 
 // Example 1: Merge Sort Algorithm 📚⚔️➕
 /*
  * Merge Sort is a classic divide and conquer algorithm.
  * It divides the array into halves, recursively sorts them,
  * and then merges the sorted halves.
  */
 void merge(std::vector<int>& arr, int left, int mid, int right) {
     // Create temporary arrays ✨
     int n1 = mid - left + 1; // Size of left subarray
     int n2 = right - mid;    // Size of right subarray
 
     std::vector<int> L(n1), R(n2);
 
     // Copy data to temp arrays 📝
     for (int i = 0; i < n1; ++i)
         L[i] = arr[left + i]; // Left subarray
     for (int j = 0; j < n2; ++j)
         R[j] = arr[mid + 1 + j]; // Right subarray
 
     // Merge the temp arrays back into arr 🌟
     int i = 0, j = 0, k = left;
     while (i < n1 && j < n2) { // Compare and merge
         if (L[i] <= R[j]) {
             arr[k] = L[i];
             ++i;
         } else {
             arr[k] = R[j];
             ++j;
         }
         ++k;
     }
 
     // Copy any remaining elements of L[] 🌱
     while (i < n1) {
         arr[k] = L[i];
         ++i;
         ++k;
     }
 
     // Copy any remaining elements of R[] 🌱
     while (j < n2) {
         arr[k] = R[j];
         ++j;
         ++k;
     }
 }
 
 void mergeSort(std::vector<int>& arr, int left, int right) {
     if (left < right) { // Base case check 🛑
         int mid = left + (right - left) / 2; // Find the middle point 🧭
 
         // Recursively sort first and second halves 🌐
         mergeSort(arr, left, mid);
         mergeSort(arr, mid + 1, right);
 
         // Merge the sorted halves 🔗
         merge(arr, left, mid, right);
     }
 }
 
 // Example 2: Quick Sort Algorithm 📚⚔️🔄
 /*
  * Quick Sort picks an element as pivot and partitions the array around it.
  * It recursively sorts the partitions.
  */
 int partition(std::vector<int>& arr, int low, int high) {
     int pivot = arr[high]; // Pivot element 🎯
     int i = low - 1;       // Index of smaller element
 
     for (int j = low; j <= high - 1; ++j) {
         // If current element is smaller than or equal to pivot
         if (arr[j] <= pivot) {
             ++i; // Increment index of smaller element
             std::swap(arr[i], arr[j]); // Swap elements 🔄
         }
     }
     std::swap(arr[i + 1], arr[high]); // Place pivot in correct position
     return (i + 1); // Return partition index
 }
 
 void quickSort(std::vector<int>& arr, int low, int high) {
     if (low < high) { // Base case check 🛑
         int pi = partition(arr, low, high); // Partitioning index
 
         // Recursively sort elements before and after partition 🌀
         quickSort(arr, low, pi - 1);
         quickSort(arr, pi + 1, high);
     }
 }
 
 // Example 3: Binary Search Algorithm 🔎📚✂️
 /*
  * Binary Search finds the position of a target value within a sorted array.
  * It repeatedly divides the search interval in half.
  */
 int binarySearch(const std::vector<int>& arr, int left, int right, int x) {
     if (left <= right) { // Base case check 🛑
         int mid = left + (right - left) / 2; // Find the middle element
 
         // Check if x is present at mid
         if (arr[mid] == x)
             return mid; // Element found at mid 🎯
 
         // If x is smaller, ignore right half
         if (arr[mid] > x)
             return binarySearch(arr, left, mid - 1, x); // Search left 🌅
 
         // If x is larger, ignore left half
         return binarySearch(arr, mid + 1, right, x); // Search right 🌇
     }
 
     // Element is not present in array
     return -1; // Not found 🚫
 }
 
 // Example 4: Tower of Hanoi Puzzle 🗼⚔️🧩
 /*
  * Tower of Hanoi is a mathematical puzzle.
  * It consists of three rods and multiple disks of different sizes.
  * The objective is to move the entire stack of disks to another rod.
  */
 void towerOfHanoi(int n, char from_rod, char to_rod, char aux_rod) {
     if (n == 0) // Base case: no disks to move 🛑
         return;
     // Move n-1 disks from source to auxiliary rod
     towerOfHanoi(n - 1, from_rod, aux_rod, to_rod);
     // Move remaining disk from source to target rod
     std::cout << "Move disk " << n << " from rod " << from_rod
               << " to rod " << to_rod << std::endl;
     // Move n-1 disks from auxiliary to target rod
     towerOfHanoi(n - 1, aux_rod, to_rod, from_rod);
 }
 
 // Example 5: Maximum Subarray Sum (Divide and Conquer) 📈⚔️
 /*
  * Finds the subarray with the maximum sum.
  * This is a divide and conquer approach of Kadane's Algorithm.
  */
 int maxCrossingSum(const std::vector<int>& arr, int left, int mid, int right) {
     // Include elements on left of mid
     int sum = 0;
     int left_sum = INT_MIN;
     for (int i = mid; i >= left; --i) {
         sum += arr[i];
         if (sum > left_sum)
             left_sum = sum; // Update max left sum 🌟
     }
 
     // Include elements on right of mid
     sum = 0;
     int right_sum = INT_MIN;
     for (int i = mid + 1; i <= right; ++i) {
         sum += arr[i];
         if (sum > right_sum)
             right_sum = sum; // Update max right sum 🌟
     }
 
     // Return sum of elements on left and right of mid
     return left_sum + right_sum;
 }
 
 int maxSubArraySum(const std::vector<int>& arr, int left, int right) {
     // Base case: only one element
     if (left == right)
         return arr[left];
 
     int mid = (left + right) / 2;
 
     // Return maximum of three cases:
     // a) Max subarray sum in left half
     // b) Max subarray sum in right half
     // c) Max subarray sum crossing the mid
     int left_sum = maxSubArraySum(arr, left, mid);
     int right_sum = maxSubArraySum(arr, mid + 1, right);
     int cross_sum = maxCrossingSum(arr, left, mid, right);
 
     return std::max({left_sum, right_sum, cross_sum}); // Return max of three 📈
 }
 
 // Example 6: Counting Inversions in an Array 🔢⚔️
 /*
  * Inversions indicate how far the array is from being sorted.
  * The function counts the number of inversions using merge sort.
  */
 int mergeAndCount(std::vector<int>& arr, int left, int mid, int right) {
     // Left and right subarrays
     std::vector<int> leftArr(arr.begin() + left, arr.begin() + mid + 1);
     std::vector<int> rightArr(arr.begin() + mid + 1, arr.begin() + right + 1);
 
     int i = 0, j = 0, k = left, swaps = 0;
 
     // Merge and count inversions
     while (i < leftArr.size() && j < rightArr.size()) {
         if (leftArr[i] <= rightArr[j]) {
             arr[k++] = leftArr[i++];
         } else {
             arr[k++] = rightArr[j++];
             swaps += (mid + 1) - (left + i); // Count inversions 💥
         }
     }
 
     // Copy remaining elements
     while (i < leftArr.size())
         arr[k++] = leftArr[i++];
     while (j < rightArr.size())
         arr[k++] = rightArr[j++];
 
     return swaps; // Total inversions
 }
 
 int countInversions(std::vector<int>& arr, int left, int right) {
     int inv_count = 0;
     if (left < right) {
         int mid = (left + right) / 2;
 
         // Count inversions in left and right halves
         inv_count += countInversions(arr, left, mid);
         inv_count += countInversions(arr, mid + 1, right);
 
         // Merge two halves and count cross inversions
         inv_count += mergeAndCount(arr, left, mid, right);
     }
     return inv_count; // Total inversions 💯
 }
 
 // Example 7: Karatsuba Multiplication Algorithm ✖️⚔️
 /*
  * Efficient algorithm to multiply two large numbers.
  * It reduces the multiplication of two n-digit numbers to at most n^log2(3) ≈ n^1.585
  */
 unsigned long long karatsuba(unsigned long long x, unsigned long long y) {
     // Base case for small numbers
     if (x < 10 || y < 10)
         return x * y;
 
     // Calculate the size of numbers
     int n = std::max((int)log10(x)+1, (int)log10(y)+1);
     int m = n / 2;
 
     // Split the digit sequences
     unsigned long long high1 = x / pow(10, m);
     unsigned long long low1 = x % (unsigned long long)pow(10, m);
     unsigned long long high2 = y / pow(10, m);
     unsigned long long low2 = y % (unsigned long long)pow(10, m);
 
     // Recursively compute subproducts
     unsigned long long z0 = karatsuba(low1, low2);
     unsigned long long z1 = karatsuba((low1 + high1), (low2 + high2));
     unsigned long long z2 = karatsuba(high1, high2);
 
     // Combine the results
     return (z2 * pow(10, 2 * m)) + ((z1 - z2 - z0) * pow(10, m)) + z0; // Final product 🌟
 }
 
 // Example 8: Strassen's Matrix Multiplication Algorithm 🧮⚔️
 /*
  * More efficient algorithm for matrix multiplication.
  * It multiplies two matrices faster than the conventional algorithm.
  */
 void addMatrices(int** A, int** B, int** C, int size) {
     // Add matrices A and B, store in C
     for (int i = 0; i < size; ++i)
         for (int j = 0; j < size; ++j)
             C[i][j] = A[i][j] + B[i][j]; // Addition ➕
 }
 
 void subtractMatrices(int** A, int** B, int** C, int size) {
     // Subtract matrices B from A, store in C
     for (int i = 0; i < size; ++i)
         for (int j = 0; j < size; ++j)
             C[i][j] = A[i][j] - B[i][j]; // Subtraction ➖
 }
 
 void strassenMultiply(int** A, int** B, int** C, int size) {
     if (size == 1) {
         // Base case: single element multiplication
         C[0][0] = A[0][0] * B[0][0]; // Multiply 🌟
         return;
     }
 
     int newSize = size / 2;
     // Allocate memory for submatrices
     int** a11 = new int*[newSize];
     int** a12 = new int*[newSize];
     int** a21 = new int*[newSize];
     int** a22 = new int*[newSize];
     int** b11 = new int*[newSize];
     int** b12 = new int*[newSize];
     int** b21 = new int*[newSize];
     int** b22 = new int*[newSize];
     int** c11 = new int*[newSize];
     int** c12 = new int*[newSize];
     int** c21 = new int*[newSize];
     int** c22 = new int*[newSize];
     int** p1 = new int*[newSize];
     int** p2 = new int*[newSize];
     int** p3 = new int*[newSize];
     int** p4 = new int*[newSize];
     int** p5 = new int*[newSize];
     int** p6 = new int*[newSize];
     int** p7 = new int*[newSize];
     int** tempA = new int*[newSize];
     int** tempB = new int*[newSize];
 
     // Initialize submatrices
     for (int i = 0; i < newSize; ++i) {
         a11[i] = new int[newSize];
         a12[i] = new int[newSize];
         a21[i] = new int[newSize];
         a22[i] = new int[newSize];
         b11[i] = new int[newSize];
         b12[i] = new int[newSize];
         b21[i] = new int[newSize];
         b22[i] = new int[newSize];
         c11[i] = new int[newSize];
         c12[i] = new int[newSize];
         c21[i] = new int[newSize];
         c22[i] = new int[newSize];
         p1[i] = new int[newSize];
         p2[i] = new int[newSize];
         p3[i] = new int[newSize];
         p4[i] = new int[newSize];
         p5[i] = new int[newSize];
         p6[i] = new int[newSize];
         p7[i] = new int[newSize];
         tempA[i] = new int[newSize];
         tempB[i] = new int[newSize];
     }
 
     // Dividing matrices into submatrices
     for (int i = 0; i < newSize; ++i)
         for (int j = 0; j < newSize; ++j) {
             a11[i][j] = A[i][j]; // Top-left quadrant
             a12[i][j] = A[i][j + newSize]; // Top-right quadrant
             a21[i][j] = A[i + newSize][j]; // Bottom-left quadrant
             a22[i][j] = A[i + newSize][j + newSize]; // Bottom-right quadrant
 
             b11[i][j] = B[i][j];
             b12[i][j] = B[i][j + newSize];
             b21[i][j] = B[i + newSize][j];
             b22[i][j] = B[i + newSize][j + newSize];
         }
 
     // Calculating p1 to p7:
     addMatrices(a11, a22, tempA, newSize); // tempA = a11 + a22
     addMatrices(b11, b22, tempB, newSize); // tempB = b11 + b22
     strassenMultiply(tempA, tempB, p1, newSize); // p1 = (a11+a22)(b11+b22)
 
     addMatrices(a21, a22, tempA, newSize); // tempA = a21 + a22
     strassenMultiply(tempA, b11, p2, newSize); // p2 = (a21+a22)b11
 
     subtractMatrices(b12, b22, tempB, newSize); // tempB = b12 - b22
     strassenMultiply(a11, tempB, p3, newSize); // p3 = a11(b12 - b22)
 
     subtractMatrices(b21, b11, tempB, newSize); // tempB = b21 - b11
     strassenMultiply(a22, tempB, p4, newSize); // p4 = a22(b21 - b11)
 
     addMatrices(a11, a12, tempA, newSize); // tempA = a11 + a12
     strassenMultiply(tempA, b22, p5, newSize); // p5 = (a11+a12)b22
 
     subtractMatrices(a21, a11, tempA, newSize); // tempA = a21 - a11
     addMatrices(b11, b12, tempB, newSize); // tempB = b11 + b12
     strassenMultiply(tempA, tempB, p6, newSize); // p6 = (a21-a11)(b11+b12)
 
     subtractMatrices(a12, a22, tempA, newSize); // tempA = a12 - a22
     addMatrices(b21, b22, tempB, newSize); // tempB = b21 + b22
     strassenMultiply(tempA, tempB, p7, newSize); // p7 = (a12 - a22)(b21+b22)
 
     // Calculating c11, c12, c21, c22:
     addMatrices(p1, p4, tempA, newSize); // tempA = p1 + p4
     subtractMatrices(tempA, p5, tempB, newSize); // tempB = tempA - p5
     addMatrices(tempB, p7, c11, newSize); // c11 = tempB + p7
 
     addMatrices(p3, p5, c12, newSize); // c12 = p3 + p5
 
     addMatrices(p2, p4, c21, newSize); // c21 = p2 + p4
 
     addMatrices(p1, p3, tempA, newSize); // tempA = p1 + p3
     subtractMatrices(tempA, p2, tempB, newSize); // tempB = tempA - p2
     addMatrices(tempB, p6, c22, newSize); // c22 = tempB + p6
 
     // Combining quadrants into result matrix C
     for (int i = 0; i < newSize; ++i)
         for (int j = 0; j < newSize; ++j) {
             C[i][j] = c11[i][j];
             C[i][j + newSize] = c12[i][j];
             C[i + newSize][j] = c21[i][j];
             C[i + newSize][j + newSize] = c22[i][j];
         }
 
     // Deallocate memory (avoid memory leaks!) 🗑️
     for (int i = 0; i < newSize; ++i) {
         delete[] a11[i]; delete[] a12[i]; delete[] a21[i]; delete[] a22[i];
         delete[] b11[i]; delete[] b12[i]; delete[] b21[i]; delete[] b22[i];
         delete[] c11[i]; delete[] c12[i]; delete[] c21[i]; delete[] c22[i];
         delete[] p1[i]; delete[] p2[i]; delete[] p3[i]; delete[] p4[i];
         delete[] p5[i]; delete[] p6[i]; delete[] p7[i];
         delete[] tempA[i]; delete[] tempB[i];
     }
     delete[] a11; delete[] a12; delete[] a21; delete[] a22;
     delete[] b11; delete[] b12; delete[] b21; delete[] b22;
     delete[] c11; delete[] c12; delete[] c21; delete[] c22;
     delete[] p1; delete[] p2; delete[] p3; delete[] p4;
     delete[] p5; delete[] p6; delete[] p7;
     delete[] tempA; delete[] tempB;
 }
 
 // Example 9: Closest Pair of Points Problem 📍⚔️
 /*
  * Finds the pair of points with the smallest distance between them.
  * Uses divide and conquer to achieve O(n log n) time complexity.
  */
 struct Point {
     int x, y;
 };
 
 int compareX(const void* a, const void* b) {
     Point* p1 = (Point*)a, * p2 = (Point*)b;
     return p1->x - p2->x;
 }
 
 int compareY(const void* a, const void* b) {
     Point* p1 = (Point*)a, * p2 = (Point*)b;
     return p1->y - p2->y;
 }
 
 float dist(Point p1, Point p2) {
     return sqrt((p1.x - p2.x)*(p1.x - p2.x) +
                 (p1.y - p2.y)*(p1.y - p2.y)); // Distance formula 📏
 }
 
 float bruteForce(Point P[], int n) {
     float min = FLT_MAX;
     for (int i = 0; i < n; ++i)
         for (int j = i + 1; j < n; ++j)
             if (dist(P[i], P[j]) < min)
                 min = dist(P[i], P[j]); // Update min distance
     return min;
 }
 
 float stripClosest(Point strip[], int size, float d) {
     float min = d; // Initialize the minimum distance as d
 
     qsort(strip, size, sizeof(Point), compareY); // Sort strip according to Y coordinate 🧮
 
     for (int i = 0; i < size; ++i)
         for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < min; ++j)
             if (dist(strip[i], strip[j]) < min)
                 min = dist(strip[i], strip[j]); // Update min distance 🌟
 
     return min;
 }
 
 float closestUtil(Point P[], int n) {
     if (n <= 3)
         return bruteForce(P, n); // Base case: Use brute force for small n
 
     int mid = n / 2;
     Point midPoint = P[mid];
 
     // Consider the vertical line passing through the middle point
     float dl = closestUtil(P, mid); // Distance in left of middle point
     float dr = closestUtil(P + mid, n - mid); // Distance in right of middle point
 
     float d = std::min(dl, dr); // Find the smaller of two distances
 
     // Build an array strip[]
     Point strip[n];
     int j = 0;
     for (int i = 0; i < n; ++i)
         if (abs(P[i].x - midPoint.x) < d)
             strip[j++] = P[i];
 
     // Find the closest points in strip. Return the minimum of d and closest distance in strip
     return std::min(d, stripClosest(strip, j, d));
 }
 
 float closest(Point P[], int n) {
     qsort(P, n, sizeof(Point), compareX); // Sort points according to X coordinate 🧮
 
     // Use recursive function closestUtil() to find the smallest distance
     return closestUtil(P, n);
 }
 
 // Example 10: Exponentiation (Fast Powering) ⚡️⚔️
 /*
  * Computes a^n efficiently using divide and conquer approach.
  */
 long long power(long long a, long long n) {
     if (n == 0)
         return 1; // Base case: a^0 = 1
     long long half = power(a, n / 2);
     if (n % 2 == 0)
         return half * half; // If n is even
     else
         return a * half * half; // If n is odd
 }
 
 // Example 11: Find the Kth Smallest Element 🔢⚔️
 /*
  * Uses QuickSelect algorithm, which is a variation of QuickSort,
  * to find the kth smallest element in an unordered list.
  */
 int partitionK(std::vector<int>& arr, int left, int right) {
     int pivot = arr[right]; // Pivot element 🎯
     int i = left;
     for (int j = left; j < right; ++j) {
         if (arr[j] <= pivot) {
             std::swap(arr[i], arr[j]); // Swap elements 🔄
             ++i;
         }
     }
     std::swap(arr[i], arr[right]);
     return i; // Return partition index
 }
 
 int quickSelect(std::vector<int>& arr, int left, int right, int k) {
     if (left == right) // If the list contains only one element
         return arr[left]; // Base case 🛑
 
     int pi = partitionK(arr, left, right);
 
     int length = pi - left + 1;
 
     if (length == k) // The pivot value is the answer
         return arr[pi];
     else if (k < length)
         return quickSelect(arr, left, pi - 1, k); // Recur on the left subarray
     else
         return quickSelect(arr, pi + 1, right, k - length); // Recur on the right subarray
 }
 
 // Example 12: Matrix Chain Multiplication 🧮⚔️
 /*
  * Determines the most efficient way to multiply a chain of matrices.
  * Uses divide and conquer to find minimum number of multiplications.
  */
 int matrixChainOrder(int p[], int i, int j) {
     if (i == j)
         return 0; // Base case: Only one matrix 🛑
 
     int min = INT_MAX;
 
     // Place parentheses at different positions between first and last matrix
     for (int k = i; k < j; ++k) {
         // Cost = cost of splitting at k + cost of multiplying two parts
         int count = matrixChainOrder(p, i, k)
                     + matrixChainOrder(p, k + 1, j)
                     + p[i - 1] * p[k] * p[j];
 
         if (count < min)
             min = count; // Update minimum 🌟
     }
 
     return min;
 }
 
 // Example 13: The Skyline Problem 🌇⚔️
 /*
  * Given the locations and heights of buildings, output the skyline formed.
  * Uses divide and conquer to merge skylines of subproblems.
  */
 struct Strip {
     int left;
     int height;
 };
 
 std::vector<Strip> mergeSkylines(std::vector<Strip>& left, std::vector<Strip>& right) {
     int h1 = 0, h2 = 0;
     std::vector<Strip> merged;
     int i = 0, j = 0;
 
     while (i < left.size() && j < right.size()) {
         if (left[i].left < right[j].left) {
             int x1 = left[i].left;
             h1 = left[i].height;
             int maxH = std::max(h1, h2);
             merged.push_back({x1, maxH});
             ++i;
         } else {
             int x2 = right[j].left;
             h2 = right[j].height;
             int maxH = std::max(h1, h2);
             merged.push_back({x2, maxH});
             ++j;
         }
     }
 
     // Collect remaining strips
     while (i < left.size())
         merged.push_back(left[i++]);
     while (j < right.size())
         merged.push_back(right[j++]);
 
     return merged; // Merged skyline 🎆
 }
 
 std::vector<Strip> skyline(std::vector<std::vector<int>>& buildings, int l, int r) {
     if (l == r) { // Base case 🛑
         std::vector<Strip> sky;
         sky.push_back({buildings[l][0], buildings[l][2]});
         sky.push_back({buildings[l][1], 0});
         return sky;
     }
 
     int mid = (l + r) / 2;
 
     // Recurse for left and right halves
     std::vector<Strip> leftSky = skyline(buildings, l, mid);
     std::vector<Strip> rightSky = skyline(buildings, mid + 1, r);
 
     // Merge the two skylines
     return mergeSkylines(leftSky, rightSky);
 }
 
 // Example 14: Counting the Number of Ways to Parenthesize Expressions 🌐⚔️
 /*
  * Counts the number of ways to fully parenthesize expressions
  * so that the result is true.
  */
 int countParenthesizations(std::string symbols, std::string operators, int i, int j, bool isTrue) {
     if (i > j)
         return 0;
     if (i == j) {
         if (isTrue)
             return symbols[i] == 'T' ? 1 : 0;
         else
             return symbols[i] == 'F' ? 1 : 0;
     }
 
     int ways = 0;
 
     for (int k = i; k < j; ++k) {
         char op = operators[k];
 
         int lt = countParenthesizations(symbols, operators, i, k, true);
         int lf = countParenthesizations(symbols, operators, i, k, false);
         int rt = countParenthesizations(symbols, operators, k + 1, j, true);
         int rf = countParenthesizations(symbols, operators, k + 1, j, false);
 
         if (op == '&') {
             if (isTrue)
                 ways += lt * rt;
             else
                 ways += lt * rf + lf * rt + lf * rf;
         } else if (op == '|') {
             if (isTrue)
                 ways += lt * rt + lt * rf + lf * rt;
             else
                 ways += lf * rf;
         } else if (op == '^') {
             if (isTrue)
                 ways += lt * rf + lf * rt;
             else
                 ways += lt * rt + lf * rf;
         }
     }
 
     return ways;
 }
 
 // Example 15: Peak Finding in a 1D Array 🌄⚔️
 /*
  * Finds a peak element in an array using divide and conquer.
  * A peak element is greater than or equal to its neighbors.
  */
 int findPeakUtil(const std::vector<int>& arr, int low, int high, int n) {
     int mid = low + (high - low) / 2;
 
     // Check if mid is peak
     if ((mid == 0 || arr[mid - 1] <= arr[mid]) &&
         (mid == n - 1 || arr[mid + 1] <= arr[mid]))
         return mid; // Peak found 🎯
 
     // If left neighbor is greater, recur for left half
     else if (mid > 0 && arr[mid - 1] > arr[mid])
         return findPeakUtil(arr, low, mid - 1, n);
 
     // Else recur for right half
     else
         return findPeakUtil(arr, mid + 1, high, n);
 }
 
 int findPeak(const std::vector<int>& arr, int n) {
     return findPeakUtil(arr, 0, n - 1, n);
 }
 
 // Main function to demonstrate the examples 🚀
 int main() {
     // Example usage for Merge Sort
     std::vector<int> arr1 = {12, 11, 13, 5, 6, 7};
     mergeSort(arr1, 0, arr1.size() - 1);
     std::cout << "Sorted array using Merge Sort: ";
     for (int num : arr1)
         std::cout << num << " ";
     std::cout << std::endl;
 
     // Example usage for Quick Sort
     std::vector<int> arr2 = {10, 7, 8, 9, 1, 5};
     quickSort(arr2, 0, arr2.size() - 1);
     std::cout << "Sorted array using Quick Sort: ";
     for (int num : arr2)
         std::cout << num << " ";
     std::cout << std::endl;
 
     // Example usage for Binary Search
     int x = 10;
     int result = binarySearch(arr2, 0, arr2.size() - 1, x);
     if (result != -1)
         std::cout << "Element " << x << " found at index " << result << std::endl;
     else
         std::cout << "Element " << x << " not found" << std::endl;
 
     // Example usage for Tower of Hanoi
     int n = 3; // Number of disks
     std::cout << "Tower of Hanoi moves:" << std::endl;
     towerOfHanoi(n, 'A', 'C', 'B'); // A, B and C are names of rods
 
     // Example usage for Maximum Subarray Sum
     std::vector<int> arr3 = {-2, -5, 6, -2, -3, 1, 5, -6};
     int max_sum = maxSubArraySum(arr3, 0, arr3.size() - 1);
     std::cout << "Maximum Subarray Sum is " << max_sum << std::endl;
 
     // Example usage for Counting Inversions
     std::vector<int> arr4 = {1, 20, 6, 4, 5};
     int inv_count = countInversions(arr4, 0, arr4.size() - 1);
     std::cout << "Number of inversions are " << inv_count << std::endl;
 
     // Example usage for Karatsuba Multiplication
     unsigned long long num1 = 1234, num2 = 5678;
     unsigned long long product = karatsuba(num1, num2);
     std::cout << "Karatsuba Multiplication Result: " << product << std::endl;
 
     // Example usage for Exponentiation
     long long base = 2, exponent = 10;
     long long power_result = power(base, exponent);
     std::cout << base << "^" << exponent << " = " << power_result << std::endl;
 
     // Example usage for Find the Kth Smallest Element
     std::vector<int> arr5 = {7, 10, 4, 3, 20, 15};
     int k = 3;
     int kth_smallest = quickSelect(arr5, 0, arr5.size() - 1, k);
     std::cout << "Kth smallest element is " << kth_smallest << std::endl;
 
     // Example usage for Peak Finding
     std::vector<int> arr6 = {1, 3, 20, 4, 1, 0};
     int peak_index = findPeak(arr6, arr6.size());
     std::cout << "Peak element is at index " << peak_index << " with value " << arr6[peak_index] << std::endl;
 
     return 0; // End of program 🎉
 }