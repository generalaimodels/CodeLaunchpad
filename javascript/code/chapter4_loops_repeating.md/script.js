// Chapter 4: Loops - Repeating Actions

console.log("🚀 Chapter 4: Loops - Repeating Actions 🚀");

// --------------------------------------------------------------
// 1. `for` Loops
// --------------------------------------------------------------

console.log("\n--- 1. `for` Loops ---");

// Example 1: Basic `for` loop to print numbers from 0 to 4
console.log("\n--- Example 1: Basic `for` loop ---");
for (let i = 0; i < 5; i++) {
    console.log("Iteration number: " + i); // Output the current iteration number
}
// Output:
// Iteration number: 0
// Iteration number: 1
// Iteration number: 2
// Iteration number: 3
// Iteration number: 4

// Explanation of `for` loop structure
console.log("\n--- Explanation of `for` loop structure ---");
console.log(`
for (initialization; condition; increment/decrement) {
    // Code to execute repeatedly 🔄
}
`);

console.log(`
- **Initialization**: Runs once at the loop's start. Often sets up a counter (e.g., \`let i = 0\`). 🏁
- **Condition**: Checked before each iteration. Loop continues if TRUE ✅, stops if FALSE ❌ (e.g., \`i < 5\`).
- **Increment/Decrement**: Updates the counter after each iteration (e.g., \`i++\` increases i by 1). ⬆️⬇️
`);

// Table to illustrate `for` loop execution flow
console.log("\n--- `for` loop execution flow table ---");
console.table([
    { Iteration: 1, i: 0, Condition: "i < 5 (true)", Output: "Iteration number: 0", Action: "i++ (i becomes 1)" },
    { Iteration: 2, i: 1, Condition: "i < 5 (true)", Output: "Iteration number: 1", Action: "i++ (i becomes 2)" },
    { Iteration: 3, i: 2, Condition: "i < 5 (true)", Output: "Iteration number: 2", Action: "i++ (i becomes 3)" },
    { Iteration: 4, i: 3, Condition: "i < 5 (true)", Output: "Iteration number: 3", Action: "i++ (i becomes 4)" },
    { Iteration: 5, i: 4, Condition: "i < 5 (true)", Output: "Iteration number: 4", Action: "i++ (i becomes 5)" },
    { Iteration: 6, i: 5, Condition: "i < 5 (false)", Output: "Loop stops", Action: "Loop terminates" }
]);

// --------------------------------------------------------------
// 2. `while` Loops
// --------------------------------------------------------------

console.log("\n--- 2. `while` Loops ---");

// Example 1: Basic `while` loop counting to 3
console.log("\n--- Example 1: Basic `while` loop ---");
let counter = 0; // Initialize counter outside the loop
while (counter < 3) {
    console.log("Count is: " + counter); // Output current count
    counter++; // Increment counter inside the loop - VERY IMPORTANT!
}
// Output:
// Count is: 0
// Count is: 1
// Count is: 2

// Explanation of `while` loop structure
console.log("\n--- Explanation of `while` loop structure ---");
console.log(`
while (condition) {
    // Code to execute repeatedly 🔄
    // Update the condition inside the loop! ⚠️
}
`);

console.log(`
- **Condition**: Checked BEFORE each iteration. Loop continues if TRUE ✅, stops if FALSE ❌ (e.g., \`counter < 3\`).
- **Update Condition**: It's CRUCIAL to update something inside the loop to eventually make the condition FALSE and avoid infinite loops! ♾️
`);

// Table to illustrate `while` loop execution flow
console.log("\n--- `while` loop execution flow table ---");
console.table([
    { Iteration: 1, counter: 0, Condition: "counter < 3 (true)", Output: "Count is: 0", Action: "counter++ (counter becomes 1)" },
    { Iteration: 2, counter: 1, Condition: "counter < 3 (true)", Output: "Count is: 1", Action: "counter++ (counter becomes 2)" },
    { Iteration: 3, counter: 2, Condition: "counter < 3 (true)", Output: "Count is: 2", Action: "counter++ (counter becomes 3)" },
    { Iteration: 4, counter: 3, Condition: "counter < 3 (false)", Output: "Loop stops", Action: "Loop terminates" }
]);

// --------------------------------------------------------------
// 3. `do...while` Loops
// --------------------------------------------------------------

console.log("\n--- 3. `do...while` Loops ---");

// Example 1: Basic `do...while` loop counting down from 5
console.log("\n--- Example 1: Basic `do...while` loop ---");
let num = 5; // Initialize number
do {
    console.log("Number is: " + num); // Output current number
    num--; // Decrement number inside the loop
} while (num > 0); // Condition checked AFTER the first execution
// Output:
// Number is: 5
// Number is: 4
// Number is: 3
// Number is: 2
// Number is: 1

// Explanation of `do...while` loop structure
console.log("\n--- Explanation of \`do...while\` loop structure ---");
console.log(`
do {
    // Code to execute repeatedly 🔄 (at least once!)
    // Update the condition inside the loop! ⚠️
} while (condition);
`);

console.log(`
- **Code Execution First**: The code block inside \`do\` runs **at least once**, NO MATTER WHAT! 🚀
- **Condition Check After**: The \`condition\` is checked **AFTER** the first execution and after each subsequent iteration. Loop continues if TRUE ✅, stops if FALSE ❌ (e.g., \`num > 0\`).
- **Update Condition**: Just like \`while\`, update condition inside to avoid infinite loops! ♾️
`);

// Table to illustrate `do...while` loop execution flow
console.log("\n--- \`do...while\` loop execution flow table ---");
console.table([
    { Iteration: 1, num: 5, Output: "Number is: 5", ConditionCheck: "Condition checked AFTER", Condition: "num > 0 (true)", Action: "num-- (num becomes 4)" },
    { Iteration: 2, num: 4, Output: "Number is: 4", ConditionCheck: "Condition checked AFTER", Condition: "num > 0 (true)", Action: "num-- (num becomes 3)" },
    { Iteration: 3, num: 3, Output: "Number is: 3", ConditionCheck: "Condition checked AFTER", Condition: "num > 0 (true)", Action: "num-- (num becomes 2)" },
    { Iteration: 4, num: 2, Output: "Number is: 2", ConditionCheck: "Condition checked AFTER", Condition: "num > 0 (true)", Action: "num-- (num becomes 1)" },
    { Iteration: 5, num: 1, Output: "Number is: 1", ConditionCheck: "Condition checked AFTER", Condition: "num > 0 (true)", Action: "num-- (num becomes 0)" },
    { Iteration: 6, num: 0, Output: "First execution always happens", ConditionCheck: "Condition checked AFTER", Condition: "num > 0 (false)", Action: "Loop terminates" }
]);

// --------------------------------------------------------------
// 4. Loop Control: `break` and `continue`
// --------------------------------------------------------------

console.log("\n--- 4. Loop Control: \`break\` and \`continue\` ---");

// --- `break` statement ---
console.log("\n--- \`break\` statement ---");
console.log("Exits the loop immediately, no matter the condition. 🚪");
console.log("\nExample with `break`:");
for (let i = 0; i < 10; i++) {
    if (i === 5) {
        console.log("Breaking out of the loop when i is 5! 🛑");
        break; // Exit loop when i is 5
    }
    console.log("i is: " + i);
}
// Output:
// i is: 0
// i is: 1
// i is: 2
// i is: 3
// i is: 4
// Breaking out of the loop when i is 5! 🛑

// --- `continue` statement ---
console.log("\n--- \`continue\` statement ---");
console.log("Skips the current iteration and jumps to the next one. ⏭️");
console.log("\nExample with `continue`:");
for (let i = 0; i < 5; i++) {
    if (i === 2) {
        console.log("Skipping iteration for i = 2! ➡️");
        continue; // Skip current iteration when i is 2
    }
    console.log("i is: " + i);
}
// Output:
// i is: 0
// i is: 1
// Skipping iteration for i = 2! ➡️
// i is: 3
// i is: 4

// --------------------------------------------------------------
// 5. Example from instructions (`for` loop)
// --------------------------------------------------------------

console.log("\n--- 5. Example from instructions (`for` loop) ---");
console.log("Example from your instructions:");
for (let i = 0; i < 5; i++) {
    console.log("Instruction example - i: " + i);
}
// Output:
// Instruction example - i: 0
// Instruction example - i: 1
// Instruction example - i: 2
// Instruction example - i: 3
// Instruction example - i: 4

// --------------------------------------------------------------
// 6. Expected Outcome
// --------------------------------------------------------------

console.log("\n--- 6. Expected Outcome ---");
console.log(`
You should now be able to:
- Use \`for\`, \`while\`, and \`do...while\` loops to repeat code blocks. ✅
- Understand when each loop type is most suitable. 🤔
- Control loops with \`break\` and \`continue\`. 🕹️
- Write efficient code for repetitive tasks. 🚀
`);

console.log("\n🎉 Chapter 4 Complete! You're looping like a pro! 🔄 Ready for functions next? 🚀");



// Chapter 4: Loops - Repeating Actions - Advanced Nested Conditions Example

console.log("🚀 Chapter 4: Loops - Repeating Actions - Advanced Nested Conditions Example 🚀");

// --------------------------------------------------------------
// Advanced Example: Skill Challenge Game with Nested Loops and Complex Conditions
// --------------------------------------------------------------

console.log("\n--- Advanced Example: Skill Challenge Game with Nested Loops & Conditions ---");

// Scenario: Let's simulate a Skill Challenge Game 🎮 with multiple levels and challenges.
// We'll use nested loops and various conditions to control the game flow and user experience.

// Game Settings
const maxLevels = 5; // ⬆️ Maximum number of game levels
const challengesPerLevel = 3; // 🎯 Number of challenges in each level
const userSkills = ["beginner", "intermediate", "advanced"]; // 🧑‍💻 Possible user skill levels
const challengeTypes = ["logic", "speed", "memory"]; // 🧠 Types of challenges

// Let's simulate a user
const currentUser = {
    name: "Alex",
    skillLevel: userSkills[0], // Starting with "beginner"
    score: 0,
    isProMember: true // 🌟 Initially not a pro member
};

console.log(`\n🎮 Welcome to the Skill Challenge Game, ${currentUser.name}! 🎮`);
console.log(`Skill Level: ${currentUser.skillLevel} | Pro Member: ${currentUser.isProMember ? "Yes 💎" : "No"} | Starting Score: ${currentUser.score}`);

// -------------------- Outer Loop: Levels of the Game --------------------
console.log("\n--- Starting Game Levels ---");
for (let level = 1; level <= maxLevels; level++) {
    console.log(`\n🏆 Level ${level} - Challenges Begin! 🏆`);

    // Nested Condition: Level Difficulty Adjustment based on Level Number
    let levelDifficultyFactor = 1 + (level - 1) * 0.2; // Difficulty increases by 20% each level
    console.log(`⚠️ Level Difficulty Factor: x${levelDifficultyFactor.toFixed(1)} - Challenges are getting harder!`);

    // -------------------- Inner Loop: Challenges within each Level --------------------
    for (let challengeNum = 1; challengeNum <= challengesPerLevel; challengeNum++) {
        console.log(`\n🎯 Challenge ${challengeNum} of Level ${level} - Get Ready! 🎯`);

        // Simulate challenge type selection (randomly chosen for simplicity)
        const challengeType = challengeTypes[Math.floor(Math.random() * challengeTypes.length)];
        console.log(`🔥 Challenge Type: ${challengeType.toUpperCase()} 🔥`);

        // -------------------- Nested Conditions for Challenge Logic --------------------

        // Condition 1: Skill-Based Challenge Skipping (Nested Condition with 'continue')
        if (level > 3 && currentUser.skillLevel === "beginner") {
            console.log(`🚫 Beginner skill level detected on Level ${level} Challenge ${challengeNum}. Skipping advanced challenge.`);
            continue; // Skip to the next challenge in the level
        }

        // Condition 2: Challenge Logic based on Challenge Type (Nested 'if...else if...else' with different condition types)
        let challengeScore = 0;
        if (challengeType === "logic") {
            // Logic Challenge - Score based on level difficulty and skill (Comparison and Arithmetic)
            challengeScore = Math.round(10 * levelDifficultyFactor * (userSkills.indexOf(currentUser.skillLevel) + 1));
            console.log(`🤔 Logic Challenge completed! Base score: ${challengeScore}`);

            // Nested Condition (Boolean and Comparison): Bonus for Pro Members on Logic Challenges
            if (currentUser.isProMember && level >= 3) {
                challengeScore += 15; // Pro member bonus on higher levels
                console.log(`💎 Pro Member Bonus (+15) for Level ${level} Logic Challenge!`);
            }

        } else if (challengeType === "speed") {
            // Speed Challenge - Score with a slight variation (Arithmetic)
            challengeScore = Math.round(8 * levelDifficultyFactor);
            console.log(`⚡ Speed Challenge completed! Base score: ${challengeScore}`);

            // Nested Condition (String Equality and Logical OR): Penalty for Beginners or Intermediate in Speed Challenges
            if (currentUser.skillLevel === "beginner" || currentUser.skillLevel === "intermediate") {
                challengeScore -= 5; // Slight penalty for lower skill levels
                console.log(`⚠️ Skill Level Adjustment (-5) for Speed Challenge.`);
                challengeScore = Math.max(0, challengeScore); // Ensure score doesn't go negative
            }

        } else if (challengeType === "memory") {
            // Memory Challenge - Score varies more (Arithmetic)
            challengeScore = Math.round(12 * levelDifficultyFactor * Math.random()); // Random factor for memory challenge
            console.log(`🧠 Memory Challenge completed! Score: ${challengeScore}`);

            // Nested Condition (Comparison): Extra bonus for high scores in Memory Challenges
            if (challengeScore > 10) {
                challengeScore += 7; // Bonus for excellent memory performance
                console.log(`🌟 Excellent Memory Bonus (+7)!`);
            }
        }

        // -------------------- Update User Score --------------------
        currentUser.score += challengeScore;
        console.log(`✅ Challenge ${challengeNum} Level ${level} Completed! Score Earned: ${challengeScore} | Total Score: ${currentUser.score}`);

        // -------------------- Loop Control with 'break' based on Score --------------------
        if (currentUser.score < -20) {
            console.log(`\n💔 Game Over! Score dropped below -20. Better luck next time! 💔`);
            break; // Exit the inner loop (challenges) and then outer loop (levels) due to game over condition
        }

        console.log(`--- Challenge ${challengeNum} of Level ${level} Finished ---`);
    } // Inner loop (Challenges) Ends

    if (currentUser.score < -20) {
        console.log(`(Game Over state triggered, level loop also terminated)`);
        break; // Exit the outer loop (levels) if game over in inner loop
    }

    console.log(`🎉 Level ${level} Completed! Total Score: ${currentUser.score} 🎉`);

    // Level Completion Bonus Condition (Comparison): Bonus points for completing each level
    if (level <= 3) {
        currentUser.score += 20; // Level completion bonus for first 3 levels
        console.log(`🎁 Level ${level} Completion Bonus (+20)! New Total Score: ${currentUser.score}`);
    } else {
        currentUser.score += 10; // Reduced bonus for later levels
        console.log(`🎁 Level ${level} Completion Bonus (+10)! New Total Score: ${currentUser.score}`);
    }

    // Pro-Membership Upgrade Condition (Comparison): Upgrade to pro member if score is high enough
    if (!currentUser.isProMember && currentUser.score >= 150) {
        currentUser.isProMember = true;
        console.log(`🌟 Congratulations! You've reached a score of 150+! Upgrading to Pro Membership! 💎`);
    }

    console.log(`--- Level ${level} Finished ---`);
} // Outer loop (Levels) Ends

console.log("\n--- Game Levels Completed ---");

// -------------------- Game Final Result --------------------
console.log(`\n🏁 Game Over! Final Score for ${currentUser.name}: ${currentUser.score} 🏁`);
console.log(`Final Skill Level: ${currentUser.skillLevel} | Pro Member Status: ${currentUser.isProMember ? "💎 Pro Member" : "Regular Member"}`);

if (currentUser.score >= 250) {
    console.log(`🌟 Excellent Performance! You are a Skill Master! 🏆`);
} else if (currentUser.score >= 100) {
    console.log(`👍 Good Job! Keep practicing to improve! 💪`);
} else {
    console.log(`🙂 Nice try! Don't give up, try again! 😊`);
}

console.log("\n--- Explanation of Advanced Nested Conditions and Loop Control in this Game ---");

console.log(`
In this Skill Challenge Game example, we showcased advanced use of nested loops and conditions:

1.  **Nested Loops (Outer and Inner Loops):**
    - **Outer 'for' loop (Levels):** Controls the progression through game levels (\`for (let level = 1; level <= maxLevels; level++)\`).
    - **Inner 'for' loop (Challenges):**  Manages challenges within each level (\`for (let challengeNum = 1; challengeNum <= challengesPerLevel; challengeNum++)\`).
    - Nested loops are essential for scenarios with hierarchical structures, like levels and challenges in a game, or processing data with multiple layers.

2.  **Nested Conditional Statements:**
    - We used 'if', 'else if', 'else' structures **within** the inner loop to implement game logic based on:
        - **Skill Level:** \`if (level > 3 && currentUser.skillLevel === "beginner") { continue; }\` - Skill-based skipping of advanced content.
        - **Challenge Type:** \`if (challengeType === "logic") { ... } else if (challengeType === "speed") { ... } else if (challengeType === "memory") { ... }\` - Different scoring and logic for each challenge type.
        - **Pro Membership:** \`if (currentUser.isProMember && level >= 3) { ... }\` - Pro member bonuses.
        - **User Score:** \`if (currentUser.score < -20) { break; }\` - Game over condition based on score.

3.  **Different Types of Conditions Linked Together:**
    - **Comparison Operators:** \`level > 3\`, \`currentUser.score >= 150\`, \`challengeScore > 10\`, \`level <= 3\` - For numerical comparisons like level number, score, etc.
    - **Equality Operators:** \`currentUser.skillLevel === "beginner"\`, \`challengeType === "logic"\` - For checking specific values like skill level or challenge type.
    - **Logical Operators:** \`currentUser.skillLevel === "beginner" || currentUser.skillLevel === "intermediate"\` (Logical OR), \`currentUser.isProMember && level >= 3\` (Logical AND) - Combining multiple conditions for more complex logic.
    - **Boolean Conditions:** \`if (currentUser.isProMember) { ... }\` - Directly checking a boolean property.

4.  **Loop Control Statements ('continue' and 'break'):**
    - **'continue':** \`if (level > 3 && currentUser.skillLevel === "beginner") { continue; }\` - Skips the current challenge and proceeds to the next challenge in the inner loop. Used for skill-based challenge skipping.
    - **'break':** \`if (currentUser.score < -20) { break; }\` - Immediately exits the **inner loop** (challenges) and then the **outer loop** (levels) when the game over condition is met. Used for early termination of loops based on dynamic conditions.

5.  **Realistic Scenario:**
    - The example simulates a more complex, game-like scenario, demonstrating how loops and nested conditions are used to create dynamic and interactive program flows.

This advanced example illustrates how nested loops and a variety of condition types are combined to create sophisticated program logic, handling complex scenarios and dynamic decision-making. It's a step closer to building real-world applications and games! 🚀
`);

console.log("\n🎉 Advanced Nested Loops and Conditions Example for Chapter 4 Complete! You're now a loop and condition master! 🚀");