// Chapter 3: Control Flow - Making Decisions

console.log("🚀 Chapter 3: Control Flow - Making Decisions 🚀");

// --------------------------------------------------------------
// 1. Conditional Statements: `if`, `else if`, `else`
// --------------------------------------------------------------

console.log("\n--- 1. Conditional Statements: if, else if, else ---");

// --- if statement ---
console.log("\n--- if statement ---");

// Example 1: Simple if condition
let isRaining = true; // 🌧️ Let's say it's raining
if (isRaining) {
    console.log("Take an umbrella ☔"); // Output if isRaining is true
}

// Example 2: if condition with comparison operator
let age = 25; // 👶 Age of a person
if (age >= 18) { // ✅ Check if age is 18 or older
    console.log("You are an adult 🧑"); // Output if age is 18 or older
}

// Example 3: if condition with logical operator
let has_Ticket = true; // 🎟️ Imagine you have a ticket for an event
let hasPass = true; // 🎫 Imagine you have a pass for the even
if (has_Ticket && hasPass) { // ✅ Check if both conditions are true
    console.log("You can enter the event! 🎉"); // Output if both are true
}



// --- if...else statement ---
console.log("\n--- if...else statement ---");

// Example 1: if-else for temperature
let temperatureInCelsius = 15; // 🌡️ Temperature in Celsius
if (temperatureInCelsius > 25) { // 🔥 Check if temperature is greater than 25
    console.log("It's a hot day ☀️"); // Output if temperature > 25
} else {
    console.log("It's not too hot 😌"); // Output if temperature is not > 25
}

// Example 2: if-else for even or odd number
let number = 7; // 🔢 Let's check this number
if (number % 2 === 0) { // 💯 Check if the remainder when divided by 2 is 0 (even)
    console.log(number + " is an even number 짝수"); // Output if number is even
} else {
    console.log(number + " is an odd number 홀수"); // Output if number is odd
}


// --- if...else if...else statement ---
console.log("\n--- if...else if...else statement ---");

// Example 1: Grading system
let studentScore = 75; // 📝 Student's score
if (studentScore >= 90) { // 🥇 Check for A grade
    console.log("Grade: A - Excellent! 🌟");
} else if (studentScore >= 80) { // 🥈 Check for B grade if A grade condition is false
    console.log("Grade: B - Good 👍");
} else if (studentScore >= 70) { // 🥉 Check for C grade if A and B grade conditions are false
    console.log("Grade: C - Average 😐");
} else if (studentScore >= 60) { // Check for D grade if A, B, C grade conditions are false
    console.log("Grade: D - Passable 😥");
} else { // If none of the above conditions are true
    console.log("Grade: F - Failed 😞");
}


// Example 2: Day of the week
let dayOfWeek = "Friday"; // 📅 Let's check the day
if (dayOfWeek === "Monday") { // 💼 Check if it's Monday
    console.log("It's Monday, start of the work week");
} else if (dayOfWeek === "Tuesday") { // Check if it's Tuesday
    console.log("It's Tuesday, second day of work");
} else if (dayOfWeek === "Wednesday") { // Check if it's Wednesday
    console.log("It's Wednesday, middle of the week 🐪");
} else if (dayOfWeek === "Thursday") { // Check if it's Thursday
    console.log("It's Thursday, almost Friday");
} else if (dayOfWeek === "Friday") { // 🎉 Check if it's Friday
    console.log("It's Friday, weekend is coming! 🥳");
} else { // If it's not Monday to Friday, it must be weekend
    console.log("It's weekend! Relax 😎");
}


// Table summarizing if, else if, else
console.table([
    { type: "if", description: "Execute code block if condition is TRUE ✅", example: "if (condition) { /* code */ }" },
    { type: "else", description: "Execute code block if IF condition is FALSE ❌", example: "if (condition) { /* code */ } else { /* code */ }" },
    { type: "else if", description: "Check multiple conditions sequentially ➡️", example: "if (cond1) { /* code */ } else if (cond2) { /* code */ }" }
]);


// --------------------------------------------------------------
// 2. Switch Statements
// --------------------------------------------------------------

console.log("\n--- 2. Switch Statements ---");

// Example 1: Days of the week using switch
let currentDay = "Wednesday"; // 🗓️ Let's check the day again

switch (currentDay) {
    case "Monday": // 💼 Case for Monday
        console.log("Today is Monday");
        break; // 🚪 Exit switch after executing this case
    case "Tuesday": // Case for Tuesday
        console.log("Today is Tuesday");
        break;
    case "Wednesday": // 🐪 Case for Wednesday
        console.log("Today is Wednesday, Hump Day!");
        break;
    case "Thursday": // Case for Thursday
        console.log("Today is Thursday");
        break;
    case "Friday": // 🎉 Case for Friday
        console.log("Today is Friday, TGIF!");
        break;
    case "Saturday": // 🥳 Case for Saturday
    case "Sunday":   // 😴 Case for Sunday (both Saturday and Sunday will execute this)
        console.log("Today is weekend");
        break;
    default: // ❓ Default case if no other case matches
        console.log("Invalid day");
}


// Example 2: Traffic light colors
let trafficLight = "yellow"; // 🚦 Current traffic light color

switch (trafficLight) {
    case "red": // 🔴 Case for red light
        console.log("Stop! 🛑");
        break;
    case "yellow": // 🟡 Case for yellow light
        console.log("Caution, get ready to stop or go ⚠️");
        break;
    case "green": // 🟢 Case for green light
        console.log("Go! ✅");
        break;
    default: // ❓ Default case for unknown color
        console.log("Unknown traffic light color 🚥");
}


// Example 3: Switch without break (Fall-through) - Be careful with this!
let fruitType = "banana"; // 🍌 Fruit type

switch (fruitType) {
    case "apple": // 🍎 Case for apple
        console.log("It's an apple");
        // No break here, will fall through to the next case
    case "banana": // 🍌 Case for banana
        console.log("It's a banana or apple!"); // This will be executed for both "apple" and "banana"
        break;
    case "orange": // 🍊 Case for orange
        console.log("It's an orange");
        break;
    default: // ❓ Default case
        console.log("Unknown fruit");
}
// In this example, if fruitType is "apple", both "It's an apple" and "It's a banana or apple!" will be logged.


// Important points about switch statements
console.log("\n--- Important points of switch statement ---");
console.table([
    { point: 1, description: "Evaluates an expression and matches its value to case clauses 🔍" },
    { point: 2, description: "Uses strict equality (===) for comparisons 💯" },
    { point: 3, description: "break statement prevents 'fall-through' to next case 🚪 (important!)" },
    { point: 4, description: "default case is executed if no case matches, like a final else ❓" },
    { point: 5, description: "Group multiple cases to execute the same code block, e.g., case 'Sat': case 'Sun': ... 👯" }
]);


// --------------------------------------------------------------
// 3. Example from previous chapter (Voting eligibility)
// --------------------------------------------------------------

console.log("\n--- 3. Example (Voting Eligibility) ---");

let personAge = 20; // 👶 Age of a person

if (personAge >= 18) { // ✅ Check if age is 18 or older
    console.log("Eligible to vote ✅"); // Output if eligible
} else {
    console.log("Not eligible to vote ❌"); // Output if not eligible
}

// Example with a different age
let personAge2 = 16; // 👶 Younger person's age

if (personAge2 >= 18) { // ✅ Check again for a younger person
    console.log("Eligible to vote ✅");
} else {
    console.log("Not eligible to vote ❌"); // Output will be 'Not eligible' for age 16
}


console.log("\n🎉 Chapter 3 Complete! Decision-making unlocked! 🧠 You're becoming a control flow ninja! 🚀");

// ============================================================================
console.log("Before going Loops  once Again learn:🚀 Next chapter: Loops - Repeating Code")

// Chapter 3: Control Flow - Making Decisions

console.log("🚀 Chapter 3: Control Flow - Making Decisions 🚀");

// --------------------------------------------------------------
// 1. Conditional Statements: `if`, `else if`, `else`
// --------------------------------------------------------------

console.log("\n--- 1. Conditional Statements: if, else if, else ---");

// --- if statement ---
console.log("\n--- if statement ---");

// Example 1: Basic if statement to check a condition
let hasTicket = true; // 🎟️ Imagine you have a ticket for an event

if (hasTicket) { // ✅ Condition: Do you have a ticket?
    console.log("You can enter the event! 🎉"); // Execute if hasTicket is true
}
// Output: You can enter the event! 🎉

// Example 2: if statement with a comparison
let currentHour = 14; // 🕒 Let's say it's 2 PM (14:00 in 24-hour format)

if (currentHour < 12) { // ⏰ Condition: Is the current hour less than 12 (noon)?
    console.log("Good morning! ☀️"); // Execute if currentHour is before noon
}

// --- if...else statement ---
console.log("\n--- if...else statement ---");

// Example 1: Deciding between two actions based on a condition
let isMember = false; // 👤 Are you a member?

if (isMember) { // ✅ Condition: Are you a member?
    console.log("Welcome, member! Enjoy your benefits! 🌟"); // Execute if isMember is true
} else {
    console.log("Join us to get exclusive benefits! 🎁"); // Execute if isMember is false
}
// Output: Join us to get exclusive benefits! 🎁


// Example 2: Checking temperature and deciding what to wear
let temperatureCelsius = 25; // 🌡️ Temperature in Celsius

if (temperatureCelsius > 30) { // 🔥 Condition: Is it hotter than 30 degrees?
    console.log("It's very hot! Wear light clothes 🩳 and stay hydrated 💧"); // Hot weather advice
} else {
    console.log("The weather is pleasant 😌. You can wear normal clothes 👕"); // Mild weather advice
}
// Output: The weather is pleasant 😌. You can wear normal clothes 👕


// --- if...else if...else statement ---
console.log("\n--- if...else if...else statement ---");

// Example 1: Determining discount based on purchase amount
let purchaseAmount = 120; // 💰 Total purchase amount

if (purchaseAmount >= 200) { // 💎 Condition 1: Purchase >= 200
    console.log("You get a 20% discount! 🤩"); // Highest discount
} else if (purchaseAmount >= 100) { // ✨ Condition 2: Purchase >= 100 (but less than 200 because condition 1 failed)
    console.log("You get a 10% discount! 😊"); // Medium discount
} else if (purchaseAmount >= 50) { // 🎉 Condition 3: Purchase >= 50 (and less than 100 and 200)
    console.log("You get a 5% discount! 👍"); // Small discount
} else { // 🙁 If none of the above conditions are true (purchase < 50)
    console.log("No discount for now. Spend more to get a discount! 😉"); // No discount
}
// Output: You get a 10% discount! 😊


// Example 2: Checking user role and showing different messages
let userRole = "editor"; // 🛡️ User role can be 'admin', 'editor', 'viewer'

if (userRole === "admin") { // 👑 Condition 1: Is user an admin?
    console.log("Welcome, Admin! You have full access. 🚀"); // Admin access
} else if (userRole === "editor") { // ✍️ Condition 2: Is user an editor? (if not admin)
    console.log("Welcome, Editor! You can create and edit content. 📝"); // Editor access
} else if (userRole === "viewer") { // 👀 Condition 3: Is user a viewer? (if not admin or editor)
    console.log("Welcome, Viewer! You can only view content. 📖"); // Viewer access
} else { // ❓ If userRole is none of the above or invalid
    console.log("Unknown role. Please contact support. 📞"); // Unknown role
}
// Output: Welcome, Editor! You can create and edit content. 📝


// Summarizing if, else if, else in a table
console.log("\n--- Summary of if, else if, else ---");
console.table([
    { type: "if", description: "Executes code block if condition is TRUE", syntax: "if (condition) { /* code */ }" },
    { type: "else", description: "Executes code block if IF condition is FALSE", syntax: "if (condition) { /* code */ } else { /* code */ }" },
    { type: "else if", description: "Checks multiple conditions in sequence; used after 'if' and before optional 'else'", syntax: "if (cond1) { /* code */ } else if (cond2) { /* code */ }" }
]);


// --------------------------------------------------------------
// 2. Switch Statements
// --------------------------------------------------------------

console.log("\n--- 2. Switch Statements ---");

// Example 1: Checking the day of the week using switch
let currentDayOfWeek = "Wednesday"; // 📅 Let's find out what day it is

switch (currentDayOfWeek) {
    case "Monday": // 💼 Case: if currentDayOfWeek is "Monday"
        console.log("It's Monday, let's get to work! 💼");
        break; // 🚪 Important: Exit the switch statement after this case
    case "Tuesday": // Case: if currentDayOfWeek is "Tuesday"
        console.log("It's Tuesday, keep the momentum going! 💪");
        break;
    case "Wednesday": // 🐪 Case: if currentDayOfWeek is "Wednesday"
        console.log("It's Wednesday, halfway through the week! 🐪");
        break;
    case "Thursday": // Case: if currentDayOfWeek is "Thursday"
        console.log("It's Thursday, almost Friday! 🤩");
        break;
    case "Friday": // 🎉 Case: if currentDayOfWeek is "Friday"
        console.log("It's Friday, weekend is near! 🎉");
        break;
    case "Saturday": // 🥳 Case: if currentDayOfWeek is "Saturday"
    case "Sunday":   // 😴 Case: if currentDayOfWeek is "Sunday" (both will execute the same code)
        console.log("It's the weekend, time to relax! 🥳");
        break;
    default: // ❓ Default case: if currentDayOfWeek doesn't match any of the cases
        console.log("Invalid day of the week. 🗓️");
}
// Output: It's Wednesday, halfway through the week! 🐪


// Example 2: Checking browser type
let browser = "Chrome"; // 🌐 Let's check the browser

switch (browser) {
    case "Chrome": // ⚙️ Case for Chrome
        console.log("You are using Chrome. ⚙️");
        break;
    case "Firefox": // 🦊 Case for Firefox
        console.log("You are using Firefox. 🦊");
        break;
    case "Safari": // 🍎 Case for Safari
        console.log("You are using Safari. 🍎");
        break;
    case "Edge":   // 🟦 Case for Edge
        console.log("You are using Edge. 🟦");
        break;
    default: // ❓ Default case for other browsers
        console.log("You are using an unknown browser. 🤔");
}
// Output: You are using Chrome. ⚙️


// Example 3: Switch case fall-through (without break) - Use carefully!
let fruit = "apple"; // 🍎 Let's check the fruit

switch (fruit) {
    case "apple": // 🍎 Case for apple
        console.log("It's an apple");
        // no break! execution falls through to the next case
    case "banana": // 🍌 Case for banana
        console.log("It's either an apple or a banana!"); // This line will execute for both "apple" and "banana"
        break; // break here to stop fall-through
    case "orange": // 🍊 Case for orange
        console.log("It's an orange");
        break;
    default: // ❓ Default case
        console.log("Unknown fruit type");
}
// Output for fruit="apple":
// It's an apple
// It's either an apple or a banana!


// Key points about switch statements summarized in a table
console.log("\n--- Key points of switch statements ---");
console.table([
    { point: 1, description: "Evaluates an expression once. 🔍" },
    { point: 2, description: "Matches the expression's value against multiple 'case' values. 🎯" },
    { point: 3, description: "Uses strict equality (===) for comparisons. 💯" },
    { point: 4, description: "'break' is essential to prevent fall-through to the next case. 🚪" },
    { point: 5, description: "'default' case is optional and runs if no 'case' matches. ❓" },
    { point: 6, description: "Group multiple cases to execute the same code block (e.g., for weekends). 👯" }
]);


// --------------------------------------------------------------
// 3. Example from previous chapter (Voting eligibility) - using if-else
// --------------------------------------------------------------

console.log("\n--- 3. Example (Voting Eligibility) - Revisited ---");

let personAgeForVoting = 20; // 👶 Age of a person

if (personAgeForVoting >= 18) { // ✅ Condition: Is age 18 or older?
    console.log("🎉 Yes, you are eligible to vote! ✅"); // Eligible to vote
} else {
    console.log("😞 Sorry, you are not eligible to vote yet. ❌"); // Not eligible to vote
}
// Output: 🎉 Yes, you are eligible to vote! ✅

// Example with a younger person
let youngPersonAge = 16; // 👶 Younger person's age

if (youngPersonAge >= 18) { // ✅ Condition check for younger person
    console.log("🎉 Yes, you are eligible to vote! ✅");
} else {
    console.log("😞 Sorry, you are not eligible to vote yet. ❌");
}
// Output: 😞 Sorry, you are not eligible to vote yet. ❌


console.log("\n🎉 Chapter 3 Complete! You're now a decision-making master in JavaScript! 🧠 On to the next chapter! 🚀");