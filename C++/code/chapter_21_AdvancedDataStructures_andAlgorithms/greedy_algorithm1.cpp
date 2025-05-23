// Greedy Algorithms in C++ 🧮🚀

// Author: Advanced C++ Coder 👨‍💻
// Purpose: Demonstrate Greedy Algorithms - Making Change & Activity Selection

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// ============================================================
// Concept 1: Making Change 💰✅
// Goal: Minimize the number of coins for a given amount
// ============================================================

// Function to make change using Greedy Algorithm 💰✅
void makeChange(int amount, vector<int>& denominations) {
    // Sort denominations in descending order 🪙⬇️
    sort(denominations.rbegin(), denominations.rend());
    vector<int> coinsUsed;

    for (int coin : denominations) {
        // While we can use the coin 🪙
        while (amount >= coin) {
            amount -= coin;          // Subtract coin from amount ➖
            coinsUsed.push_back(coin); // Record coin used 📝
        }
    }

    // Output the result
    cout << "Coins used to make change: ";
    for (int coin : coinsUsed) {
        cout << coin << " ";
    }
    cout << endl;
}

// Examples for Making Change 💰✅
void testMakingChange() {
    cout << "=== Making Change Examples 💰✅ ===" << endl;

    // Example 1️⃣
    int amount1 = 93;
    vector<int> denominations1 = {1, 5, 10, 25};
    cout << "Amount: " << amount1 << ", Denominations: 1,5,10,25" << endl;
    makeChange(amount1, denominations1);

    // Example 2️⃣
    int amount2 = 37;
    vector<int> denominations2 = {1, 2, 5, 10, 20};
    cout << "\nAmount: " << amount2 << ", Denominations: 1,2,5,10,20" << endl;
    makeChange(amount2, denominations2);

    // Example 3️⃣
    int amount3 = 99;
    vector<int> denominations3 = {1, 5, 10, 25, 50};
    cout << "\nAmount: " << amount3 << ", Denominations: 1,5,10,25,50" << endl;
    makeChange(amount3, denominations3);

    // Example 4️⃣ (Edge Case: No denominations)
    int amount4 = 50;
    vector<int> denominations4 = {};
    cout << "\nAmount: " << amount4 << ", Denominations: (None)" << endl;
    makeChange(amount4, denominations4);

    // Example 5️⃣ (Edge Case: Amount is zero)
    int amount5 = 0;
    vector<int> denominations5 = {1, 5, 10};
    cout << "\nAmount: " << amount5 << ", Denominations: 1,5,10" << endl;
    makeChange(amount5, denominations5);

    // Example 6️⃣ (Non-standard denominations)
    int amount6 = 65;
    vector<int> denominations6 = {1, 7, 23};
    cout << "\nAmount: " << amount6 << ", Denominations: 1,7,23" << endl;
    makeChange(amount6, denominations6);

    // Example 7️⃣ (Greedy fails to find optimal)
    int amount7 = 6;
    vector<int> denominations7 = {1, 3, 4};
    cout << "\nAmount: " << amount7 << ", Denominations: 1,3,4" << endl;
    makeChange(amount7, denominations7);

    // Example 8️⃣
    int amount8 = 100;
    vector<int> denominations8 = {1, 20, 50};
    cout << "\nAmount: " << amount8 << ", Denominations: 1,20,50" << endl;
    makeChange(amount8, denominations8);

    // Example 9️⃣
    int amount9 = 1;
    vector<int> denominations9 = {2, 5, 10};
    cout << "\nAmount: " << amount9 << ", Denominations: 2,5,10" << endl;
    makeChange(amount9, denominations9);

    // Example 🔟
    int amount10 = 58;
    vector<int> denominations10 = {1, 5, 10, 25};
    cout << "\nAmount: " << amount10 << ", Denominations: 1,5,10,25" << endl;
    makeChange(amount10, denominations10);

    // Example 1️⃣1️⃣
    int amount11 = 47;
    vector<int> denominations11 = {1, 3, 5, 9};
    cout << "\nAmount: " << amount11 << ", Denominations: 1,3,5,9" << endl;
    makeChange(amount11, denominations11);

    // Example 1️⃣2️⃣
    int amount12 = 74;
    vector<int> denominations12 = {1, 7, 20, 50};
    cout << "\nAmount: " << amount12 << ", Denominations: 1,7,20,50" << endl;
    makeChange(amount12, denominations12);

    // Example 1️⃣3️⃣
    int amount13 = 15;
    vector<int> denominations13 = {1, 5, 12};
    cout << "\nAmount: " << amount13 << ", Denominations: 1,5,12" << endl;
    makeChange(amount13, denominations13);

    // Example 1️⃣4️⃣
    int amount14 = 83;
    vector<int> denominations14 = {1, 10, 25, 50};
    cout << "\nAmount: " << amount14 << ", Denominations: 1,10,25,50" << endl;
    makeChange(amount14, denominations14);

    // Example 1️⃣5️⃣
    int amount15 = 23;
    vector<int> denominations15 = {2, 5, 10};
    cout << "\nAmount: " << amount15 << ", Denominations: 2,5,10" << endl;
    makeChange(amount15, denominations15);
}

// ============================================================
// Concept 2: Activity Selection 🗓️🥇
// Goal: Maximize the number of non-overlapping activities
// ============================================================

// Activity structure 🗓️
struct Activity {
    int start;  // Start time 🕒
    int finish; // Finish time 🕕
};

// Compare activities by finish time 📅
bool activityCompare(Activity s1, Activity s2) {
    return (s1.finish < s2.finish);
}

// Function to select maximum number of activities 🗓️✅
void activitySelection(vector<Activity>& activities) {
    // Sort activities based on finish time 📅⬆️
    sort(activities.begin(), activities.end(), activityCompare);

    vector<Activity> selectedActivities;

    // The first activity always gets selected ✅
    selectedActivities.push_back(activities[0]);
    int lastFinishTime = activities[0].finish;

    // Iterate through the rest of the activities
    for (unsigned int i = 1; i < activities.size(); i++) {
        // If this activity starts after or at the finish of last selected
        if (activities[i].start >= lastFinishTime) {
            selectedActivities.push_back(activities[i]); // Select activity 📝
            lastFinishTime = activities[i].finish;       // Update last finish time ⏱️
        }
    }

    // Output the result
    cout << "Selected activities (start, finish): ";
    for (Activity act : selectedActivities) {
        cout << "(" << act.start << "," << act.finish << ") ";
    }
    cout << endl;
}

// Examples for Activity Selection 🗓️✅
void testActivitySelection() {
    cout << "\n=== Activity Selection Examples 🗓️✅ ===" << endl;

    // Example 1️⃣
    vector<Activity> activities1 = {{1, 3}, {2, 5}, {4, 7}, {1, 8}, {5, 9}, {8, 10}};
    cout << "Activities: (1,3), (2,5), (4,7), (1,8), (5,9), (8,10)" << endl;
    activitySelection(activities1);

    // Example 2️⃣
    vector<Activity> activities2 = {{0, 6}, {3, 4}, {1, 2}, {5, 7}, {8, 9}, {5, 9}};
    cout << "\nActivities: (0,6), (3,4), (1,2), (5,7), (8,9), (5,9)" << endl;
    activitySelection(activities2);

    // Example 3️⃣
    vector<Activity> activities3 = {{10, 20}, {12, 25}, {20, 30}};
    cout << "\nActivities: (10,20), (12,25), (20,30)" << endl;
    activitySelection(activities3);

    // Example 4️⃣ (All overlapping activities)
    vector<Activity> activities4 = {{1, 4}, {2, 5}, {3, 6}};
    cout << "\nActivities: (1,4), (2,5), (3,6)" << endl;
    activitySelection(activities4);

    // Example 5️⃣ (Non-overlapping activities)
    vector<Activity> activities5 = {{1, 2}, {3, 4}, {5, 6}};
    cout << "\nActivities: (1,2), (3,4), (5,6)" << endl;
    activitySelection(activities5);

    // Example 6️⃣
    vector<Activity> activities6 = {{6, 7}, {2, 4}, {8, 12}, {3, 5}, {0, 1}};
    cout << "\nActivities: (6,7), (2,4), (8,12), (3,5), (0,1)" << endl;
    activitySelection(activities6);

    // Example 7️⃣
    vector<Activity> activities7 = {{1, 4}, {3, 5}, {0, 6}, {5, 7}, {8, 9}, {5, 9}, {6, 10}};
    cout << "\nActivities: (1,4), (3,5), (0,6), (5,7), (8,9), (5,9), (6,10)" << endl;
    activitySelection(activities7);

    // Example 8️⃣
    vector<Activity> activities8 = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    cout << "\nActivities: (1,2), (2,3), (3,4), (4,5)" << endl;
    activitySelection(activities8);

    // Example 9️⃣
    vector<Activity> activities9 = {{7, 9}, {0, 10}, {4, 5}, {8, 9}, {4, 10}};
    cout << "\nActivities: (7,9), (0,10), (4,5), (8,9), (4,10)" << endl;
    activitySelection(activities9);

    // Example 🔟
    vector<Activity> activities10 = {{2, 14}, {3, 5}, {7, 9}, {12, 16}, {0, 6}};
    cout << "\nActivities: (2,14), (3,5), (7,9), (12,16), (0,6)" << endl;
    activitySelection(activities10);

    // Example 1️⃣1️⃣
    vector<Activity> activities11 = {{1, 4}, {2, 3}, {3, 5}, {7, 8}, {5, 7}, {4, 6}};
    cout << "\nActivities: (1,4), (2,3), (3,5), (7,8), (5,7), (4,6)" << endl;
    activitySelection(activities11);

    // Example 1️⃣2️⃣
    vector<Activity> activities12 = {{6, 10}, {1, 3}, {2, 5}, {8, 11}, {9, 12}, {3, 7}};
    cout << "\nActivities: (6,10), (1,3), (2,5), (8,11), (9,12), (3,7)" << endl;
    activitySelection(activities12);

    // Example 1️⃣3️⃣
    vector<Activity> activities13 = {{5, 9}, {1, 2}, {3, 4}, {0, 6}, {5, 7}, {8, 9}};
    cout << "\nActivities: (5,9), (1,2), (3,4), (0,6), (5,7), (8,9)" << endl;
    activitySelection(activities13);

    // Example 1️⃣4️⃣
    vector<Activity> activities14 = {{1, 3}, {2, 5}, {4, 6}, {6, 7}, {5, 8}};
    cout << "\nActivities: (1,3), (2,5), (4,6), (6,7), (5,8)" << endl;
    activitySelection(activities14);

    // Example 1️⃣5️⃣
    vector<Activity> activities15 = {{0, 2}, {1, 4}, {3, 5}, {4, 6}, {5, 7}, {7, 9}};
    cout << "\nActivities: (0,2), (1,4), (3,5), (4,6), (5,7), (7,9)" << endl;
    activitySelection(activities15);
}

int main() {
    // Test Making Change 💰✅
    testMakingChange();

    // Test Activity Selection 🗓️✅
    testActivitySelection();

    return 0; // Program executed successfully ✅
}