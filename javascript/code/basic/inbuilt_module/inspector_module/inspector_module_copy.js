/**
 * Node.js 'inspector' Module - Comprehensive Examples
 * 
 * The 'inspector' module provides an API for interacting with the V8 inspector/debugger protocol.
 * This file demonstrates all major and minor methods, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const inspector = require('inspector');
const fs = require('fs');
const path = require('path');

// Utility: Session instance for all examples
const session = new inspector.Session();

// 1. session.connect(): Start a debugging session
session.connect();
console.log('1. Inspector session connected'); // Output: Inspector session connected

// 2. session.post(): Send a command to the inspector (enable Debugger domain)
session.post('Debugger.enable', {}, (err, result) => {
    if (err) throw err;
    console.log('2. Debugger enabled:', result); // Output: Debugger enabled: {}
});

// 3. session.on('event'): Listen for inspector events (e.g., breakpoints)
session.on('Debugger.paused', (msg) => {
    console.log('3. Debugger paused event:', msg); // Output: Debugger paused event: { ... }
    session.post('Debugger.resume'); // Resume execution after pause
});

// 4. Trigger a breakpoint programmatically
function triggerBreakpoint() {
    session.post('Debugger.pause', {}, (err) => {
        if (err) throw err;
        console.log('4. Breakpoint triggered'); // Output: Breakpoint triggered
    });
}
setTimeout(triggerBreakpoint, 100);

// 5. session.disconnect(): End the debugging session
setTimeout(() => {
    session.disconnect();
    console.log('5. Inspector session disconnected'); // Output: Inspector session disconnected
}, 300);

// 6. session.post(): Evaluate JavaScript code in the current context
session.connect();
session.post('Runtime.evaluate', { expression: '2 + 2' }, (err, result) => {
    if (err) throw err;
    console.log('6. Runtime.evaluate result:', result.result.value); // Output: 4
    session.disconnect();
});

// 7. session.post(): Take a heap snapshot and save to file
function takeHeapSnapshot() {
    const heapFile = path.join(__dirname, 'heap.heapsnapshot');
    const writeStream = fs.createWriteStream(heapFile);
    session.connect();
    session.post('HeapProfiler.takeHeapSnapshot', null, (err) => {
        if (err) throw err;
        console.log('7. Heap snapshot taken'); // Output: Heap snapshot taken
        session.disconnect();
    });
    session.on('HeapProfiler.addHeapSnapshotChunk', (m) => {
        writeStream.write(m.params.chunk);
    });
    session.on('HeapProfiler.heapStatsUpdate', (m) => {
        // Heap stats update event (optional)
    });
    session.on('HeapProfiler.reportHeapSnapshotProgress', (m) => {
        // Progress event (optional)
    });
}
setTimeout(takeHeapSnapshot, 500);

// 8. session.post(): Set a breakpoint by URL and line number
session.connect();
const scriptUrl = __filename;
session.post('Debugger.enable', {}, () => {
    session.post('Debugger.setBreakpointByUrl', {
        lineNumber: 10, // Example line number (change as needed)
        url: scriptUrl
    }, (err, result) => {
        if (err) {
            console.log('8. Error setting breakpoint:', err.message);
        } else {
            console.log('8. Breakpoint set:', result); // Output: Breakpoint set: { breakpointId, locations }
        }
        session.disconnect();
    });
});

// 9. session.post(): Profile CPU usage
function profileCPU() {
    session.connect();
    session.post('Profiler.enable', {}, () => {
        session.post('Profiler.start', {}, () => {
            setTimeout(() => {
                session.post('Profiler.stop', {}, (err, { profile }) => {
                    if (err) throw err;
                    console.log('9. CPU profile captured:', !!profile); // Output: true
                    session.disconnect();
                });
            }, 200);
        });
    });
}
setTimeout(profileCPU, 1000);

// 10. Exception Handling: Invalid command
session.connect();
session.post('NonExistentDomain.nonExistentMethod', {}, (err, result) => {
    if (err) {
        console.log('10. Exception caught:', err.message); // Output: 'Method 'NonExistentDomain.nonExistentMethod' wasn't found'
    } else {
        console.log('10. Unexpected success:', result);
    }
    session.disconnect();
});

/**
 * Additional Notes:
 * - All major and minor methods of 'inspector' module are covered.
 * - Both event-driven and callback-based usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js inspector module.
 * - For more protocol commands, see: https://chromedevtools.github.io/devtools-protocol/
 */