/*******************************************************
 * Chapter 13: Dynamic Programming - Remembering Solutions 🧠📝🚀🧩 *
 * Understanding Dynamic Programming through Examples       *
 * At least 15 examples with detailed explanations          *
 * All code in a single .cpp file                           *
 *******************************************************/

 #include <iostream>
 #include <vector>
 #include <algorithm>
 #include <cstring>
 #include <climits>
 #include <unordered_map>
 #include <numeric>
 
 // Example 1: Fibonacci Sequence using Memoization 🧠📝🔢
 /*
  * Computes Fibonacci numbers efficiently using memoization.
  * Stores computed values to avoid redundant calculations.
  */
 const int MAX_N = 1000; // Maximum size
 long long fib_memo[MAX_N]; // Memoization array
 
 long long fibonacci_memo(int n) {
     if (n <= 1)
         return n; // Base cases: fib(0) = 0, fib(1) = 1
     if (fib_memo[n] != -1)
         return fib_memo[n]; // Return cached value 🧠
     fib_memo[n] = fibonacci_memo(n - 1) + fibonacci_memo(n - 2); // Store result 📝
     return fib_memo[n];
 }
 
 // Example 2: Fibonacci Sequence using Tabulation 📊🔢
 /*
  * Computes Fibonacci numbers efficiently using tabulation.
  * Builds up solutions from base cases iteratively.
  */
 long long fibonacci_tab(int n) {
     std::vector<long long> fib_table(n + 1, 0);
     fib_table[0] = 0; // Base case: fib(0) = 0
     if (n > 0)
         fib_table[1] = 1; // Base case: fib(1) = 1
 
     for (int i = 2; i <= n; ++i) {
         fib_table[i] = fib_table[i - 1] + fib_table[i - 2]; // Build up table 📝
     }
 
     return fib_table[n];
 }
 
 // Example 3: Longest Common Subsequence (LCS) Length 🧬🧠📝
 /*
  * Finds length of LCS between two strings using DP.
  * Builds a table of solutions to subproblems.
  */
 int lcs_length(const std::string& X, const std::string& Y) {
     int m = X.length();
     int n = Y.length();
     std::vector<std::vector<int>> L(m + 1, std::vector<int>(n + 1, 0)); // DP table 📊
 
     for (int i = 0; i <= m; ++i) { // Build table in bottom-up manner
         for (int j = 0; j <= n; ++j) {
             if (i == 0 || j == 0)
                 L[i][j] = 0; // Base case 🛑
             else if (X[i - 1] == Y[j - 1])
                 L[i][j] = L[i - 1][j - 1] + 1; // Characters match 🔠
             else
                 L[i][j] = std::max(L[i - 1][j], L[i][j - 1]); // Max of left and top ⬆️⬅️
         }
     }
 
     return L[m][n]; // Length of LCS 🌟
 }
 
 // Example 4: Longest Common Subsequence (LCS) Print Sequence 🧬🧠📝
 /*
  * Finds and prints LCS between two strings using DP.
  * Reconstructs LCS from DP table.
  */
 std::string lcs_sequence(const std::string& X, const std::string& Y) {
     int m = X.length();
     int n = Y.length();
     // DP table 📊
     std::vector<std::vector<int>> L(m + 1, std::vector<int>(n + 1, 0));
 
     // Build the LCS table
     for (int i = 0; i <= m; ++i) {
         for (int j = 0; j <= n; ++j) {
             if (i == 0 || j == 0)
                 L[i][j] = 0; // Base case 🛑
             else if (X[i - 1] == Y[j - 1])
                 L[i][j] = L[i - 1][j - 1] + 1; // Match 🔠
             else
                 L[i][j] = std::max(L[i - 1][j], L[i][j - 1]); // Max ⬆️⬅️
         }
     }
 
     // Reconstruct LCS from table
     int index = L[m][n]; // Length of LCS
     std::string lcs(index, ' '); // Create string of length index
 
     int i = m, j = n;
     while (i > 0 && j > 0) {
         if (X[i - 1] == Y[j - 1]) {
             lcs[index - 1] = X[i - 1]; // Put character in lcs 🔠
             --i;
             --j;
             --index;
         } else if (L[i - 1][j] > L[i][j - 1])
             --i;
         else
             --j;
     }
 
     return lcs; // Return LCS sequence 🌟
 }
 
 // Example 5: Edit Distance (Levenshtein Distance) ✍️🔄🧠📝
 /*
  * Computes the minimum number of edits to convert one string to another.
  * Uses DP to build up solutions.
  */
 int editDistance(const std::string& str1, const std::string& str2) {
     int m = str1.length();
     int n = str2.length();
     // DP table 📊
     std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
 
     // Initialize base cases
     for (int i = 0; i <= m; ++i)
         dp[i][0] = i; // Deletion cost
     for (int j = 0; j <= n; ++j)
         dp[0][j] = j; // Insertion cost
 
     // Build dp table
     for (int i = 1; i <= m; ++i) {
         for (int j = 1; j <= n; ++j) {
             if (str1[i - 1] == str2[j - 1])
                 dp[i][j] = dp[i - 1][j - 1]; // Characters match 🔠
             else
                 dp[i][j] = 1 + std::min({dp[i - 1][j],    // Deletion
                                          dp[i][j - 1],    // Insertion
                                          dp[i - 1][j - 1] // Substitution
                                         }); // Minimum edit
         }
     }
 
     return dp[m][n]; // Minimum edits needed ✂️
 }
 
 // Example 6: 0/1 Knapsack Problem 🎒💰🧠📝
 /*
  * Finds the maximum value that can be put in a knapsack of capacity W.
  * Uses DP to build up solutions.
  */
 int knapsack(const std::vector<int>& wt, const std::vector<int>& val, int W) {
     int n = wt.size();
     // DP table 📊
     std::vector<std::vector<int>> dp(n + 1, std::vector<int>(W + 1, 0));
 
     // Build dp table
     for (int i = 0; i <= n; ++i) {
         for (int w = 0; w <= W; ++w) {
             if (i == 0 || w == 0)
                 dp[i][w] = 0; // Base case 🛑
             else if (wt[i - 1] <= w)
                 dp[i][w] = std::max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w]); // Include or exclude item
             else
                 dp[i][w] = dp[i - 1][w]; // Cannot include item
         }
     }
 
     return dp[n][W]; // Maximum value 💰
 }
 
 // Example 7: Rod Cutting Problem 🔩✂️💰🧠📝
 /*
  * Determines the maximum value obtainable by cutting up the rod and selling the pieces.
  * Uses DP to store intermediate results.
  */
 int rodCutting(const std::vector<int>& price, int n) {
     std::vector<int> dp(n + 1, 0); // DP table 📊
 
     // Build table dp[] in bottom-up manner
     for (int i = 1; i <= n; ++i) {
         int max_val = INT_MIN; // Initialize max value
         for (int j = 0; j < i; ++j)
             max_val = std::max(max_val, price[j] + dp[i - j - 1]);
         dp[i] = max_val; // Store maximum value 💰
     }
 
     return dp[n]; // Maximum obtainable value
 }
 
 // Example 8: Coin Change Problem 💰🪙🧠📝
 /*
  * Finds the minimum number of coins needed to make change for a given amount.
  * Uses DP to build up solutions.
  */
 int coinChange(const std::vector<int>& coins, int amount) {
     const int MAX = amount + 1;
     std::vector<int> dp(amount + 1, MAX); // DP table 📊
     dp[0] = 0; // Base case
 
     // Build dp table
     for (int i = 1; i <= amount; ++i) {
         for (int coin : coins) {
             if (coin <= i)
                 dp[i] = std::min(dp[i], dp[i - coin] + 1); // Take minimum coins
         }
     }
 
     return dp[amount] > amount ? -1 : dp[amount]; // Return -1 if no solution
 }
 
 // Example 9: Longest Increasing Subsequence (LIS) 📈🧠📝
 /*
  * Finds the length of the Longest Increasing Subsequence in an array.
  * Uses DP to build up solutions.
  */
 int longestIncreasingSubsequence(const std::vector<int>& nums) {
     if (nums.empty())
         return 0; // Empty sequence
 
     int n = nums.size();
     std::vector<int> dp(n, 1); // DP table 📊
 
     for (int i = 1; i < n; ++i) {
         for (int j = 0; j < i; ++j) {
             if (nums[i] > nums[j]) // Increasing
                 dp[i] = std::max(dp[i], dp[j] + 1); // Update dp[i]
         }
     }
 
     return *std::max_element(dp.begin(), dp.end()); // Length of LIS 📏
 }
 
 // Example 10: Matrix Chain Multiplication 🧮🧠📝
 /*
  * Determines the most efficient way to multiply a chain of matrices.
  * Uses DP to store minimum multiplication costs.
  */
 int matrixChainOrder(const std::vector<int>& p) {
     int n = p.size() - 1; // Number of matrices
     std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0)); // DP table 📊
 
     // L is chain length
     for (int L = 2; L <= n; ++L) {
         for (int i = 0; i < n - L + 1; ++i) {
             int j = i + L - 1;
             dp[i][j] = INT_MAX; // Initialize to infinity
             for (int k = i; k < j; ++k) {
                 // Cost of multiplying A[i..k] and A[k+1..j]
                 int cost = dp[i][k] + dp[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
                 dp[i][j] = std::min(dp[i][j], cost); // Minimum cost
             }
         }
     }
 
     return dp[0][n - 1]; // Minimum cost 💰
 }
 
 // Example 11: Subset Sum Problem 🎯🧠📝
 /*
  * Determines if there is a subset of the given set with sum equal to given sum.
  * Uses DP to build up solutions.
  */
 bool subsetSum(const std::vector<int>& set, int sum) {
     int n = set.size();
     std::vector<std::vector<bool>> dp(n + 1, std::vector<bool>(sum + 1, false)); // DP table 📊
 
     // Initialize
     for (int i = 0; i <= n; ++i)
         dp[i][0] = true; // Sum 0 is always possible with empty subset
 
     // Build dp table
     for (int i = 1; i <= n; ++i) {
         for (int s = 1; s <= sum; ++s) {
             if (set[i - 1] <= s)
                 dp[i][s] = dp[i - 1][s] || dp[i - 1][s - set[i - 1]]; // Include or exclude
             else
                 dp[i][s] = dp[i - 1][s]; // Cannot include
         }
     }
 
     return dp[n][sum]; // Return whether sum is possible
 }
 
 // Example 12: Unique Paths in a Grid 🚶🧠📝
 /*
  * Counts the number of unique paths from top-left to bottom-right in a grid.
  * Uses DP to build up the number of ways.
  */
 int uniquePaths(int m, int n) {
     std::vector<std::vector<int>> dp(m, std::vector<int>(n, 1)); // Initialize DP table with 1s 📊
 
     for (int i = 1; i < m; ++i) { // Start from cell (1,1)
         for (int j = 1; j < n; ++j) {
             dp[i][j] = dp[i - 1][j] + dp[i][j - 1]; // Ways from top and left cells
         }
     }
 
     return dp[m - 1][n - 1]; // Total unique paths to bottom-right cell 🚩
 }
 
 // Example 13: Minimum Path Sum in a Grid 📉🧠📝
 /*
  * Finds a path from top-left to bottom-right which minimizes the sum of all numbers along its path.
  * Uses DP to build up minimum sums.
  */
 int minPathSum(const std::vector<std::vector<int>>& grid) {
     if (grid.empty())
         return 0;
 
     int m = grid.size();
     int n = grid[0].size();
     std::vector<std::vector<int>> dp = grid; // DP table 📊
 
     // Initialize first row and first column
     for (int i = 1; i < m; ++i)
         dp[i][0] += dp[i - 1][0]; // From top cell
     for (int j = 1; j < n; ++j)
         dp[0][j] += dp[0][j - 1]; // From left cell
 
     // Build dp table
     for (int i = 1; i < m; ++i) {
         for (int j = 1; j < n; ++j)
             dp[i][j] += std::min(dp[i - 1][j], dp[i][j - 1]); // Minimum of top and left
     }
 
     return dp[m - 1][n - 1]; // Minimum path sum 💰
 }
 
 // Example 14: Partition Equal Subset Sum ⚖️🧠📝
 /*
  * Determines if the array can be partitioned into two subsets with equal sum.
  * Uses DP to build up solutions.
  */
 bool canPartition(const std::vector<int>& nums) {
     int sum = std::accumulate(nums.begin(), nums.end(), 0);
 
     if (sum % 2 != 0)
         return false; // Cannot partition odd sum
 
     int target = sum / 2;
     std::vector<bool> dp(target + 1, false); // DP table 📊
     dp[0] = true; // Base case
 
     for (int num : nums) {
         for (int i = target; i >= num; --i) {
             dp[i] = dp[i] || dp[i - num]; // Include or exclude
         }
     }
 
     return dp[target]; // Return whether target sum is possible
 }
 
 // Example 15: Boolean Parenthesization Problem 🎭🧠📝
 /*
  * Counts the number of ways we can parenthesize the expression so that the value evaluates to true.
  * Uses DP with memoization to store results.
  */
 int countWaysUtil(const std::string& symbols, const std::string& operators, int i, int j, bool isTrue, std::unordered_map<std::string, int>& memo) {
     if (i > j)
         return 0;
     if (i == j) {
         if (isTrue)
             return symbols[i] == 'T' ? 1 : 0;
         else
             return symbols[i] == 'F' ? 1 : 0;
     }
 
     std::string key = std::to_string(i) + "_" + std::to_string(j) + "_" + (isTrue ? "T" : "F");
     if (memo.find(key) != memo.end())
         return memo[key]; // Return cached value 🧠
 
     int ways = 0;
 
     for (int k = i; k < j; ++k) {
         int lt = countWaysUtil(symbols, operators, i, k, true, memo);
         int lf = countWaysUtil(symbols, operators, i, k, false, memo);
         int rt = countWaysUtil(symbols, operators, k + 1, j, true, memo);
         int rf = countWaysUtil(symbols, operators, k + 1, j, false, memo);
 
         char op = operators[k];
 
         if (op == '&') {
             if (isTrue)
                 ways += lt * rt;
             else
                 ways += lt * rf + lf * rt + lf * rf;
         }
         else if (op == '|') {
             if (isTrue)
                 ways += lt * rt + lt * rf + lf * rt;
             else
                 ways += lf * rf;
         }
         else if (op == '^') {
             if (isTrue)
                 ways += lt * rf + lf * rt;
             else
                 ways += lt * rt + lf * rf;
         }
     }
 
     memo[key] = ways; // Store result 📝
     return ways;
 }
 
 int countWays(const std::string& symbols, const std::string& operators) {
     int n = symbols.length();
     std::unordered_map<std::string, int> memo; // Memoization map 🧠
     return countWaysUtil(symbols, operators, 0, n - 1, true, memo);
 }
 
 // Main function to demonstrate the examples 🚀
 int main() {
     // Example usage for Fibonacci using Memoization
     std::memset(fib_memo, -1, sizeof(fib_memo)); // Initialize memoization array
     int fib_n = 40;
     std::cout << "Fibonacci number " << fib_n << " using Memoization is " << fibonacci_memo(fib_n) << std::endl;
 
     // Example usage for Fibonacci using Tabulation
     std::cout << "Fibonacci number " << fib_n << " using Tabulation is " << fibonacci_tab(fib_n) << std::endl;
 
     // Example usage for Longest Common Subsequence
     std::string X = "AGGTAB";
     std::string Y = "GXTXAYB";
     std::cout << "Length of LCS is " << lcs_length(X, Y) << std::endl;
     std::cout << "LCS sequence is " << lcs_sequence(X, Y) << std::endl;
 
     // Example usage for Edit Distance
     std::string str1 = "sunday";
     std::string str2 = "saturday";
     std::cout << "Edit Distance between " << str1 << " and " << str2 << " is " << editDistance(str1, str2) << std::endl;
 
     // Example usage for 0/1 Knapsack
     std::vector<int> wt = {1, 3, 4, 5};
     std::vector<int> val = {1, 4, 5, 7};
     int W = 7;
     std::cout << "Maximum value in Knapsack is " << knapsack(wt, val, W) << std::endl;
 
     // Example usage for Rod Cutting
     std::vector<int> price = {2, 5, 7, 8};
     int rod_length = 5;
     std::cout << "Maximum profit from Rod Cutting is " << rodCutting(price, rod_length) << std::endl;
 
     // Example usage for Coin Change
     std::vector<int> coins = {1, 2, 5};
     int amount = 11;
     int coin_result = coinChange(coins, amount);
     if (coin_result != -1)
         std::cout << "Minimum coins needed: " << coin_result << std::endl;
     else
         std::cout << "No solution exists to make amount " << amount << std::endl;
 
     // Example usage for Longest Increasing Subsequence
     std::vector<int> nums = {10,9,2,5,3,7,101,18};
     std::cout << "Length of Longest Increasing Subsequence is " << longestIncreasingSubsequence(nums) << std::endl;
 
     // Example usage for Matrix Chain Multiplication
     std::vector<int> p = {1, 2, 3, 4};
     std::cout << "Minimum number of multiplications is " << matrixChainOrder(p) << std::endl;
 
     // Example usage for Subset Sum
     std::vector<int> set = {3, 34, 4, 12, 5, 2};
     int sum = 9;
     if (subsetSum(set, sum))
         std::cout << "Found a subset with given sum " << sum << std::endl;
     else
         std::cout << "No subset with given sum " << sum << std::endl;
 
     // Example usage for Unique Paths in Grid
     int m = 3, n = 7;
     std::cout << "Number of unique paths is " << uniquePaths(m, n) << std::endl;
 
     // Example usage for Minimum Path Sum in Grid
     std::vector<std::vector<int>> grid = {{1,3,1},{1,5,1},{4,2,1}};
     std::cout << "Minimum path sum is " << minPathSum(grid) << std::endl;
 
     // Example usage for Partition Equal Subset Sum
     std::vector<int> nums_partition = {1, 5, 11, 5};
     if (canPartition(nums_partition))
         std::cout << "Can partition into equal subset sum" << std::endl;
     else
         std::cout << "Cannot partition into equal subset sum" << std::endl;
 
     // Example usage for Boolean Parenthesization Problem
     std::string symbols = "TTFT";
     std::string operators = "|&^";
     std::cout << "Number of ways to parenthesize expression to get true: " << countWays(symbols, operators) << std::endl;
 
     return 0; // End of program 🎉
 }