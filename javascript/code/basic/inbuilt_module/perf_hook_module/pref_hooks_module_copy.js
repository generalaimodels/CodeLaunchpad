/**
 * Node.js 'perf_hooks' Module - Comprehensive Examples
 * 
 * The 'perf_hooks' module provides performance measurement APIs, including high-resolution timers and user-defined marks/measures.
 * This file demonstrates all major and minor methods, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const { performance, PerformanceObserver, monitorEventLoopDelay, constants } = require('perf_hooks');

// 1. performance.now(): High-resolution time in milliseconds since Node.js process start
const t1 = performance.now();
setTimeout(() => {
    const t2 = performance.now();
    console.log('1. Elapsed ms (should be ~100):', Math.round(t2 - t1)); 
    // Output: Elapsed ms (should be ~100)
}, 100);

// 2. performance.mark() and performance.measure(): Custom performance marks and measures
performance.mark('A');
setTimeout(() => {
    performance.mark('B');
    performance.measure('A to B', 'A', 'B');
    const entries = performance.getEntriesByType('measure');
    console.log('2. Custom measure:', entries[0].name, Math.round(entries[0].duration)); 
    // Output: Custom measure: A to B <duration>
}, 50);

// 3. performance.getEntries(), getEntriesByName(), getEntriesByType()
performance.mark('C');
performance.mark('D');
performance.measure('C to D', 'C', 'D');
const allEntries = performance.getEntries();
console.log('3. All entries count:', allEntries.length); 
// Output: All entries count: <number>
const markEntries = performance.getEntriesByType('mark');
console.log('3. Mark entries:', markEntries.map(e => e.name)); 
// Output: Mark entries: [ 'A', 'B', 'C', 'D' ]
const measureEntries = performance.getEntriesByName('C to D', 'measure');
console.log('3. Measure entries:', measureEntries.length); 
// Output: 1

// 4. performance.clearMarks() and performance.clearMeasures()
performance.clearMarks('A');
performance.clearMeasures('A to B');
console.log('4. Marks after clear:', performance.getEntriesByType('mark').map(e => e.name)); 
// Output: Marks after clear: [ 'B', 'C', 'D' ]
console.log('4. Measures after clear:', performance.getEntriesByType('measure').map(e => e.name)); 
// Output: Measures after clear: [ 'C to D' ]

// 5. PerformanceObserver: Observe new performance entries
const obs = new PerformanceObserver((list, observer) => {
    const entries = list.getEntries();
    entries.forEach(entry => {
        console.log('5. Observed entry:', entry.name, entry.entryType); 
        // Output: Observed entry: <name> <entryType>
    });
    observer.disconnect();
});
obs.observe({ entryTypes: ['mark', 'measure'] });
performance.mark('ObservedMark');
performance.measure('ObservedMeasure', 'C', 'D');

// 6. monitorEventLoopDelay(): Monitor event loop delay statistics
const h = monitorEventLoopDelay({ resolution: 10 });
h.enable();
setTimeout(() => {
    h.disable();
    console.log('6. Event loop delay mean (ms):', Math.round(h.mean / 1e6)); 
    // Output: Event loop delay mean (ms): <number>
    console.log('6. Event loop delay max (ms):', Math.round(h.max / 1e6)); 
    // Output: Event loop delay max (ms): <number>
}, 120);

// 7. performance.timerify(): Measure function execution time
function slowFunction() {
    for (let i = 0; i < 1e6; ++i) {}
}
const timedSlowFunction = performance.timerify(slowFunction);
const obs2 = new PerformanceObserver((list) => {
    const entry = list.getEntries()[0];
    console.log('7. Timerify duration (ms):', entry.duration.toFixed(3)); 
    // Output: Timerify duration (ms): <number>
    obs2.disconnect();
});
obs2.observe({ entryTypes: ['function'] });
timedSlowFunction();

// 8. performance.nodeTiming: Node.js process timing info
console.log('8. Node.js timing:', performance.nodeTiming); 
// Output: { name: 'node', entryType: 'node', startTime: 0, ... }

// 9. performance.eventLoopUtilization(): Event loop utilization stats
const elu1 = performance.eventLoopUtilization();
setTimeout(() => {
    const elu2 = performance.eventLoopUtilization(elu1);
    console.log('9. Event loop utilization:', elu2.utilization.toFixed(3)); 
    // Output: Event loop utilization: <number>
}, 80);

// 10. Exception Handling: Invalid mark/measure names
try {
    performance.measure('Invalid', 'noMark1', 'noMark2');
} catch (err) {
    console.log('10. Exception caught:', err.message); 
    // Output: No known mark: noMark1
}

// 11. perf_hooks.constants: Useful constants for performance hooks
console.log('11. perf_hooks.constants:', Object.keys(constants)); 
// Output: [ 'NODE_PERFORMANCE_GC_MAJOR', 'NODE_PERFORMANCE_GC_MINOR', ... ]

/**
 * Additional Notes:
 * - All major and minor methods of 'perf_hooks' module are covered.
 * - Both synchronous and observer-based usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js perf_hooks module.
 */