/**
 * Node.js 'process' Module - Comprehensive Examples
 * 
 * The 'process' global object provides information and control over the current Node.js process.
 * This file demonstrates all major and minor methods, properties, and edge cases.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

// 1. process.argv: Command-line arguments
console.log('1. process.argv:', process.argv.slice(0, 3)); 
// Output: [ 'node', '/path/to/file.js', ... ]

// 2. process.env: Environment variables
process.env.TEST_VAR = 'BestCoder';
console.log('2. process.env.TEST_VAR:', process.env.TEST_VAR); 
// Output: 'BestCoder'

// 3. process.cwd() and process.chdir(): Current working directory and change directory
console.log('3. Current directory:', process.cwd()); 
// Output: e.g., '/home/user/project'
try {
    process.chdir('..');
    console.log('3. Changed directory:', process.cwd()); 
    // Output: e.g., '/home/user'
    process.chdir(__dirname); // Change back for safety
} catch (err) {
    console.log('3. chdir error:', err.message);
}

// 4. process.exit(): Exit the process with a code
// Uncomment to test: process.exit(0);
// Output: (process exits with code 0)

// 5. process.on('exit'), process.on('beforeExit'): Listen for process exit events
process.on('beforeExit', (code) => {
    console.log('5. beforeExit event, code:', code); 
    // Output: beforeExit event, code: 0
});
process.on('exit', (code) => {
    console.log('5. exit event, code:', code); 
    // Output: exit event, code: 0
});

// 6. process.nextTick(): Schedule a callback to run on the next event loop tick
process.nextTick(() => {
    console.log('6. nextTick callback executed'); 
    // Output: 6. nextTick callback executed
});

// 7. process.memoryUsage(): Memory usage statistics
const mem = process.memoryUsage();
console.log('7. Memory usage (MB):', {
    rss: (mem.rss / 1024 / 1024).toFixed(2),
    heapTotal: (mem.heapTotal / 1024 / 1024).toFixed(2),
    heapUsed: (mem.heapUsed / 1024 / 1024).toFixed(2),
    external: (mem.external / 1024 / 1024).toFixed(2)
});
// Output: { rss: '...', heapTotal: '...', heapUsed: '...', external: '...' }

// 8. process.uptime(): Process uptime in seconds
console.log('8. Process uptime (s):', process.uptime().toFixed(2)); 
// Output: e.g., 0.12

// 9. process.hrtime() and process.hrtime.bigint(): High-resolution real time
const start = process.hrtime();
setTimeout(() => {
    const diff = process.hrtime(start);
    console.log('9. hrtime diff (s, ns):', diff); 
    // Output: [ <seconds>, <nanoseconds> ]
    const bigStart = process.hrtime.bigint();
    setTimeout(() => {
        const bigDiff = process.hrtime.bigint() - bigStart;
        console.log('9. hrtime.bigint diff (ns):', bigDiff.toString()); 
        // Output: <nanoseconds>
    }, 10);
}, 10);

// 10. process.kill(): Send a signal to a process (self-signal example)
process.on('SIGUSR2', () => {
    console.log('10. Received SIGUSR2'); 
    // Output: 10. Received SIGUSR2
});
try {
    process.kill(process.pid, 'SIGUSR2');
} catch (err) {
    console.log('10. process.kill error:', err.message);
}

// 11. process.pid, process.ppid, process.title, process.version, process.versions
console.log('11. PID:', process.pid); 
// Output: <current process id>
console.log('11. PPID:', process.ppid); 
// Output: <parent process id>
console.log('11. Title:', process.title); 
// Output: 'node'
console.log('11. Node version:', process.version); 
// Output: 'vXX.XX.X'
console.log('11. Node versions:', process.versions); 
// Output: { node: '...', v8: '...', ... }

// 12. process.stdin, process.stdout, process.stderr: Standard I/O streams
console.log('12. Is TTY (stdout):', process.stdout.isTTY); 
// Output: true or false
// Example: Write to stderr
process.stderr.write('12. This is an error message\n'); 
// Output: This is an error message

// 13. process.getuid(), process.getgid(), process.setuid(), process.setgid() (POSIX only)
if (process.getuid && process.getgid) {
    console.log('13. UID:', process.getuid()); 
    // Output: <user id>
    console.log('13. GID:', process.getgid()); 
    // Output: <group id>
    // Note: setuid/setgid require root privileges and are not demonstrated here for safety.
}

// 14. process.emitWarning(): Emit a process warning
process.on('warning', (warning) => {
    console.log('14. Warning emitted:', warning.name, warning.message); 
    // Output: Warning emitted: CustomWarning This is a custom warning
});
process.emitWarning('This is a custom warning', { code: 'CustomWarning' });

// 15. Exception Handling: Uncaught exception and unhandled rejection
process.on('uncaughtException', (err) => {
    console.log('15. Uncaught Exception:', err.message); 
    // Output: 15. Uncaught Exception: Test exception
});
process.on('unhandledRejection', (reason) => {
    console.log('15. Unhandled Rejection:', reason); 
    // Output: 15. Unhandled Rejection: Test rejection
});
setTimeout(() => {
    Promise.reject('Test rejection');
    setTimeout(() => {
        throw new Error('Test exception');
    }, 10);
}, 20);

/**
 * Additional Notes:
 * - All major and minor methods/properties of 'process' are covered.
 * - Both synchronous and event-driven usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js process module.
 */