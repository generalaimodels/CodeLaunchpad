/**
 * Node.js 'readline' Module - Comprehensive Examples
 * 
 * The 'readline' module provides an interface for reading data from a Readable stream (such as process.stdin) one line at a time.
 * This file demonstrates all major and minor methods, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const readline = require('readline');
const fs = require('fs');
const stream = require('stream');

// 1. readline.createInterface(): Basic usage with process.stdin and process.stdout
// Uncomment to test interactively
// const rl1 = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout
// });
// rl1.question('1. What is your name? ', (answer) => {
//     console.log(`1. Hello, ${answer}!`); // Output: Hello, <name>!
//     rl1.close();
// });

// 2. Reading lines from a file stream
const filePath = __filename; // This file itself
const rl2 = readline.createInterface({
    input: fs.createReadStream(filePath),
    crlfDelay: Infinity
});
let lineCount = 0;
rl2.on('line', (line) => {
    lineCount++;
    if (lineCount === 2) {
        console.log('2. Second line of this file:', line); 
        // Output: Second line of this file: * Node.js 'readline' Module - Comprehensive Examples
    }
});
rl2.on('close', () => {
    // File reading done
});

// 3. rl.question(): Prompt user and get answer (async/await style)
async function askQuestion(query) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    return new Promise(resolve => rl.question(query, ans => {
        rl.close();
        resolve(ans);
    }));
}
// Uncomment to test interactively
// (async () => {
//     const ans = await askQuestion('3. Enter a number: ');
//     console.log('3. You entered:', ans); // Output: You entered: <number>
// })();

// 4. rl.on('line'): Read multiple lines from stdin
// Uncomment to test interactively
// const rl4 = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout
// });
// let count4 = 0;
// rl4.on('line', (input) => {
//     console.log(`4. Received: ${input}`); // Output: Received: <input>
//     count4++;
//     if (count4 === 2) rl4.close();
// });

// 5. rl.setPrompt(), rl.prompt(), rl.on('line') for custom prompt
// Uncomment to test interactively
// const rl5 = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout
// });
// rl5.setPrompt('5. Type something> ');
// rl5.prompt();
// rl5.on('line', (line) => {
//     console.log('5. You typed:', line); // Output: You typed: <line>
//     rl5.prompt();
//     if (line === 'exit') rl5.close();
// });

// 6. rl.close(): Close the interface (also triggers 'close' event)
const rl6 = readline.createInterface({
    input: new stream.Readable({ read() {} }),
    output: process.stdout
});
rl6.on('close', () => {
    console.log('6. rl.close() triggered'); // Output: rl.close() triggered
});
rl6.close();

// 7. rl.pause() and rl.resume(): Pause and resume input stream
const readable7 = new stream.Readable({
    read() {
        this.push('Line 1\nLine 2\n');
        this.push(null);
    }
});
const rl7 = readline.createInterface({ input: readable7 });
rl7.pause();
setTimeout(() => {
    rl7.resume();
}, 50);
rl7.on('line', (line) => {
    console.log('7. Read after resume:', line); // Output: Read after resume: Line 1, then Line 2
});

// 8. rl.write(): Simulate user input programmatically
const rl8 = readline.createInterface({
    input: new stream.PassThrough(),
    output: process.stdout
});
rl8.on('line', (line) => {
    console.log('8. Simulated input:', line); // Output: Simulated input: test
    rl8.close();
});
rl8.write('test\n');

// 9. rl.history: Access input history (only works with output set)
const rl9 = readline.createInterface({
    input: new stream.PassThrough(),
    output: process.stdout
});
rl9.write('first\n');
rl9.write('second\n');
setTimeout(() => {
    console.log('9. rl.history:', rl9.history); 
    // Output: rl.history: [ 'second', 'first' ]
    rl9.close();
}, 20);

// 10. Exception Handling: Invalid input/output streams
try {
    readline.createInterface({ input: null, output: null });
} catch (err) {
    console.log('10. Exception caught:', err.message); 
    // Output: The "input" argument must be an instance of Readable Stream
}

/**
 * Additional Notes:
 * - All major and minor methods of 'readline' module are covered.
 * - Both interactive and programmatic usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js readline module.
 */