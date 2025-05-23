/**
 * Node.js child_process Module: Comprehensive Examples
 * 
 * This file demonstrates all major and minor methods of the Node.js child_process module.
 * Each example is self-contained, with clear code, comments, and expected output.
 * 
 * To run: `node <filename>.js`
 */

const child_process = require('child_process');
const path = require('path');

// 1. child_process.exec(command[, options][, callback])
// Executes a shell command, buffers output, invokes callback with results.
function exampleExec() {
    child_process.exec('echo Hello, World!', (error, stdout, stderr) => {
        console.log('exec stdout:', stdout.trim()); // Hello, World!
        // Expected output: exec stdout: Hello, World!
    });
}
exampleExec();

// 2. child_process.execSync(command[, options])
// Synchronously executes a shell command, returns stdout as Buffer or string.
function exampleExecSync() {
    const output = child_process.execSync('echo Sync Hello');
    console.log('execSync:', output.toString().trim()); // Sync Hello
    // Expected output: execSync: Sync Hello
}
exampleExecSync();

// 3. child_process.spawn(command[, args][, options])
// Launches a new process with a given command.
function exampleSpawn() {
    const ls = child_process.spawn('node', ['-v']);
    ls.stdout.on('data', (data) => {
        console.log('spawn stdout:', data.toString().trim()); // vXX.XX.X
        // Expected output: spawn stdout: vXX.XX.X (your Node.js version)
    });
    ls.stderr.on('data', (data) => {
        console.error('spawn stderr:', data.toString());
    });
    ls.on('close', (code) => {
        // console.log('spawn process exited with code', code);
    });
}
exampleSpawn();

// 4. child_process.spawnSync(command[, args][, options])
// Synchronously spawns a process, returns result object.
function exampleSpawnSync() {
    const result = child_process.spawnSync('node', ['-e', 'console.log("Sync Spawn")']);
    console.log('spawnSync:', result.stdout.toString().trim()); // Sync Spawn
    // Expected output: spawnSync: Sync Spawn
}
exampleSpawnSync();

// 5. child_process.fork(modulePath[, args][, options])
// Spawns a new Node.js process and invokes a module.
function exampleFork() {
    // Create a simple child script for demonstration
    const childScript = path.join(__dirname, 'child.js');
    require('fs').writeFileSync(childScript, 'process.send("Hello from child!");');
    const child = child_process.fork(childScript);
    child.on('message', (msg) => {
        console.log('fork message:', msg); // Hello from child!
        // Expected output: fork message: Hello from child!
        child.kill();
        require('fs').unlinkSync(childScript); // Clean up
    });
}
exampleFork();

// 6. child_process.execFile(file[, args][, options][, callback])
// Executes a file directly, without a shell.
function exampleExecFile() {
    child_process.execFile(process.execPath, ['-e', 'console.log("ExecFile")'], (error, stdout, stderr) => {
        console.log('execFile stdout:', stdout.trim()); // ExecFile
        // Expected output: execFile stdout: ExecFile
    });
}
exampleExecFile();

// 7. child_process.execFileSync(file[, args][, options])
// Synchronously executes a file directly.
function exampleExecFileSync() {
    const output = child_process.execFileSync(process.execPath, ['-e', 'console.log("ExecFileSync")']);
    console.log('execFileSync:', output.toString().trim()); // ExecFileSync
    // Expected output: execFileSync: ExecFileSync
}
exampleExecFileSync();

// 8. Handling stdin, stdout, stderr streams with spawn
function exampleSpawnStreams() {
    const child = child_process.spawn('node', ['-i']); // Start Node.js REPL
    child.stdin.write('console.log("Stream Example")\n');
    child.stdin.end('.exit\n');
    child.stdout.on('data', (data) => {
        if (data.toString().includes('Stream Example')) {
            console.log('spawn stream:', data.toString().trim());
            // Expected output: spawn stream: Stream Example
        }
    });
}
exampleSpawnStreams();

// 9. Sending and receiving messages with forked processes
function exampleForkMessaging() {
    // Create a child script for messaging
    const childScript = path.join(__dirname, 'childMsg.js');
    require('fs').writeFileSync(childScript, `
        process.on('message', (msg) => {
            process.send('Received: ' + msg);
        });
    `);
    const child = child_process.fork(childScript);
    child.on('message', (msg) => {
        console.log('fork messaging:', msg); // Received: ping
        // Expected output: fork messaging: Received: ping
        child.kill();
        require('fs').unlinkSync(childScript); // Clean up
    });
    child.send('ping');
}
exampleForkMessaging();

// 10. Error handling and exit codes
function exampleErrorHandling() {
    const child = child_process.spawn('node', ['-e', 'process.exit(5)']);
    child.on('close', (code) => {
        console.log('child exited with code', code); // 5
        // Expected output: child exited with code 5
    });
    child.on('error', (err) => {
        console.error('child process error:', err);
    });
}
exampleErrorHandling();

/**
 * Summary:
 * - All major and minor child_process methods are covered.
 * - Each example is self-contained and demonstrates expected behavior.
 * - For fork examples, temporary child scripts are created and cleaned up.
 * - Uncomment or modify values to see different results.
 */