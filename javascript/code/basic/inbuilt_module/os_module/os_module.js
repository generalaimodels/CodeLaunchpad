/**
 * Node.js 'os' Module - Comprehensive Examples
 * 
 * The 'os' module provides operating system-related utility methods and properties.
 * This file demonstrates all major and minor methods, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const os = require('os');

// 1. os.arch(): Returns the operating system CPU architecture
console.log('1. CPU Architecture:', os.arch()); 
// Output: 'x64', 'arm', etc.

// 2. os.platform(): Returns the operating system platform
console.log('2. Platform:', os.platform()); 
// Output: 'linux', 'win32', 'darwin', etc.

// 3. os.cpus(): Returns an array of objects containing information about each logical CPU core
const cpus = os.cpus();
console.log('3. Number of CPU cores:', cpus.length); 
// Output: e.g., 8
console.log('3. First CPU info:', cpus[0]); 
// Output: { model: '...', speed: ..., times: { user: ..., nice: ..., sys: ..., idle: ..., irq: ... } }

// 4. os.totalmem() and os.freemem(): Returns total and free system memory in bytes
console.log('4. Total Memory (MB):', (os.totalmem() / 1024 / 1024).toFixed(2)); 
// Output: e.g., 16384.00
console.log('4. Free Memory (MB):', (os.freemem() / 1024 / 1024).toFixed(2)); 
// Output: e.g., 1024.00

// 5. os.uptime(): Returns the system uptime in seconds
console.log('5. System Uptime (seconds):', os.uptime()); 
// Output: e.g., 123456

// 6. os.hostname(): Returns the hostname of the operating system
console.log('6. Hostname:', os.hostname()); 
// Output: e.g., 'my-computer'

// 7. os.networkInterfaces(): Returns network interfaces info
const netIfs = os.networkInterfaces();
console.log('7. Network Interfaces:', netIfs); 
// Output: { lo: [ ... ], eth0: [ ... ], ... }

// 8. os.homedir(), os.tmpdir(): Returns home and temp directory paths
console.log('8. Home Directory:', os.homedir()); 
// Output: e.g., '/home/user'
console.log('8. Temp Directory:', os.tmpdir()); 
// Output: e.g., '/tmp'

// 9. os.type(), os.release(), os.version(): OS type, release, and version
console.log('9. OS Type:', os.type()); 
// Output: 'Linux', 'Darwin', 'Windows_NT'
console.log('9. OS Release:', os.release()); 
// Output: e.g., '5.15.0-1051-azure'
console.log('9. OS Version:', os.version()); 
// Output: e.g., '#59-Ubuntu SMP Wed Oct 11 18:49:16 UTC 2023'

// 10. os.userInfo(): Returns information about the current user
const userInfo = os.userInfo();
console.log('10. User Info:', userInfo); 
// Output: { uid: ..., gid: ..., username: '...', homedir: '...', shell: '...' }

// 11. os.endianness(): Returns the endianness of the CPU ('BE' or 'LE')
console.log('11. Endianness:', os.endianness()); 
// Output: 'LE' or 'BE'

// 12. os.loadavg(): Returns an array containing the 1, 5, and 15 minute load averages (0s on Windows)
console.log('12. Load Average:', os.loadavg()); 
// Output: [0.12, 0.34, 0.56] or [0, 0, 0] on Windows

// 13. os.constants: Useful OS constants (signals, errno, priorities, etc.)
console.log('13. OS Constants:', Object.keys(os.constants).slice(0, 5)); 
// Output: [ 'UV_UDP_REUSEADDR', 'dlopen', 'errno', 'signals', 'priority' ]

// 14. os.EOL: End-of-line marker for the current OS
console.log('14. OS EOL:', JSON.stringify(os.EOL)); 
// Output: '\n' (Linux/macOS), '\r\n' (Windows)

// 15. Exception Handling: Accessing a non-existent network interface
try {
    const nonExistent = netIfs['doesnotexist'][0];
    console.log('15. Non-existent interface:', nonExistent);
} catch (err) {
    console.log('15. Exception caught:', err.message); 
    // Output: Cannot read properties of undefined (reading '0')
}

/**
 * Additional Notes:
 * - All major and minor methods of 'os' module are covered.
 * - Both synchronous and property-based usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js os module.
 */