/**
 * Node.js 'domain' Module (DEPRECATED): Comprehensive Usage Examples
 * 
 * The 'domain' module provides a way to handle multiple different IO operations as a single group.
 * If any of the event emitters or callbacks registered to a domain emit an error, 
 * that error will be routed to the domain's error event, rather than causing the program to crash.
 * 
 * Note: The 'domain' module is deprecated and should not be used in new code. 
 * Use try/catch, async_hooks, or proper error handling instead.
 * 
 * This file demonstrates all major and minor methods, properties, and use-cases of the domain module.
 * Each example is self-contained, includes expected output in comments, and covers exceptions.
 * 
 * Author: The Best Coder in the World
 */

const domain = require('domain');
const EventEmitter = require('events');

// 1. Creating a Domain using domain.create()
const d1 = domain.create();
console.log('1. Domain created:', d1 instanceof domain.Domain); 
// Expected: 1. Domain created: true

// 2. Handling Errors with domain.on('error')
d1.on('error', (err) => {
    console.log('2. Domain error caught:', err.message); 
    // Expected: 2. Domain error caught: Test error in domain
});
d1.run(() => {
    throw new Error('Test error in domain');
});

// 3. Adding an EventEmitter to a Domain with domain.add()
const d2 = domain.create();
const emitter = new EventEmitter();
d2.add(emitter);
d2.on('error', (err) => {
    console.log('3. Error from emitter caught by domain:', err.message); 
    // Expected: 3. Error from emitter caught by domain: Emitter error
});
emitter.emit('error', new Error('Emitter error'));

// 4. Removing an EventEmitter from a Domain with domain.remove()
const d3 = domain.create();
const emitter2 = new EventEmitter();
d3.add(emitter2);
d3.remove(emitter2);
d3.on('error', (err) => {
    console.log('4. Should not be called:', err.message);
});
try {
    emitter2.emit('error', new Error('Error after remove'));
} catch (err) {
    console.log('4. Error not caught by domain:', err.message); 
    // Expected: 4. Error not caught by domain: Error after remove
}

// 5. Using domain.run(fn[, ...args]) for synchronous and asynchronous code
const d4 = domain.create();
d4.on('error', (err) => {
    console.log('5. Async error caught by domain:', err.message); 
    // Expected: 5. Async error caught by domain: Async error
});
d4.run(() => {
    setTimeout(() => {
        throw new Error('Async error');
    }, 10);
});

// 6. Using domain.bind(callback) to wrap a function
const d5 = domain.create();
d5.on('error', (err) => {
    console.log('6. Error in bound function caught:', err.message); 
    // Expected: 6. Error in bound function caught: Bound error
});
const boundFn = d5.bind((a, b) => {
    throw new Error('Bound error');
});
try {
    boundFn(1, 2);
} catch (err) {
    // Should not reach here
}

// 7. Using domain.intercept(callback) to handle error-first callbacks
const d6 = domain.create();
d6.on('error', (err) => {
    console.log('7. Intercepted error caught:', err.message); 
    // Expected: 7. Intercepted error caught: Callback error
});
const interceptedFn = d6.intercept((data) => {
    console.log('7. This will not be called if error is present');
});
interceptedFn(new Error('Callback error'), null);

// 8. Accessing process.domain inside a domain context
const d7 = domain.create();
d7.on('error', (err) => {
    console.log('8. process.domain is d7:', process.domain === d7); 
    // Expected: 8. process.domain is d7: true
});
d7.run(() => {
    throw new Error('Check process.domain');
});

// 9. Nested domains: inner domain error handling
const d8 = domain.create();
const d9 = domain.create();
d8.on('error', (err) => {
    console.log('9. Outer domain caught:', err.message); 
    // Expected: 9. Outer domain caught: Inner domain error
});
d9.on('error', (err) => {
    console.log('9. Inner domain caught:', err.message); 
    // Expected: 9. Inner domain caught: Inner domain error
});
d8.run(() => {
    d9.run(() => {
        throw new Error('Inner domain error');
    });
});

// 10. Exception: Error thrown outside any domain is uncaught
try {
    setTimeout(() => {
        throw new Error('Uncaught error');
    }, 20);
} catch (err) {
    // This will not catch the error, as it's outside any domain
    console.log('10. This will not be called');
}
process.on('uncaughtException', (err) => {
    console.log('10. Uncaught exception handler:', err.message); 
    // Expected: 10. Uncaught exception handler: Uncaught error
});

/**
 * Summary:
 * - Covered: domain.create, domain.run, domain.add, domain.remove, domain.bind, domain.intercept, process.domain, nested domains, and exception handling.
 * - Each example is self-contained and demonstrates a unique aspect of the domain module.
 * - All expected outputs are provided in comments for clarity.
 * - Note: The domain module is deprecated and should not be used in new code.
 */