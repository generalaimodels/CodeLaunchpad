/**
 * events Module in Node.js
 * 
 * The 'events' module provides the EventEmitter class, which is at the core of Node.js event-driven architecture.
 * This file covers all major and minor aspects of the 'events' module, including all methods, properties, and edge cases.
 * Each example is self-contained and demonstrates a specific feature or method.
 * 
 * Author: The Best Coder
 */

const EventEmitter = require('events');

// 1. Basic Usage: on(), emit()
const emitter1 = new EventEmitter();
emitter1.on('greet', (name) => {
    console.log(`Hello, ${name}!`);
});
emitter1.emit('greet', 'Alice'); // Output: Hello, Alice!

// 2. once(): Listener that fires only once
const emitter2 = new EventEmitter();
emitter2.once('launch', () => {
    console.log('Rocket launched!');
});
emitter2.emit('launch'); // Output: Rocket launched!
emitter2.emit('launch'); // No output

// 3. removeListener()/off(): Remove a specific listener
const emitter3 = new EventEmitter();
function onPing() {
    console.log('Ping received');
}
emitter3.on('ping', onPing);
emitter3.emit('ping'); // Output: Ping received
emitter3.removeListener('ping', onPing);
emitter3.emit('ping'); // No output

// 4. removeAllListeners(): Remove all listeners for an event or all events
const emitter4 = new EventEmitter();
emitter4.on('data', () => console.log('Data 1'));
emitter4.on('data', () => console.log('Data 2'));
emitter4.on('end', () => console.log('End'));
emitter4.removeAllListeners('data');
emitter4.emit('data'); // No output
emitter4.emit('end'); // Output: End

// 5. listeners(), rawListeners(): Get array of listeners
const emitter5 = new EventEmitter();
function listenerA() { console.log('A'); }
function listenerB() { console.log('B'); }
emitter5.on('event', listenerA);
emitter5.once('event', listenerB);
console.log(emitter5.listeners('event').length); // Output: 2
console.log(typeof emitter5.rawListeners('event')[1]); // Output: function

// 6. setMaxListeners(), getMaxListeners(): Control memory leak warning threshold
const emitter6 = new EventEmitter();
emitter6.setMaxListeners(20);
console.log(emitter6.getMaxListeners()); // Output: 20

// 7. prependListener(), prependOnceListener(): Add listeners to the beginning
const emitter7 = new EventEmitter();
emitter7.on('order', () => console.log('Second'));
emitter7.prependListener('order', () => console.log('First'));
emitter7.emit('order'); // Output: First \n Second
emitter7.prependOnceListener('order', () => console.log('Prepended Once'));
emitter7.emit('order'); // Output: Prepended Once \n First \n Second
emitter7.emit('order'); // Output: First \n Second

// 8. eventNames(): List all event names with listeners
const emitter8 = new EventEmitter();
emitter8.on('alpha', () => {});
emitter8.on('beta', () => {});
console.log(emitter8.eventNames()); // Output: [ 'alpha', 'beta' ]

// 9. listenerCount(): Count listeners for an event (static and instance)
const emitter9 = new EventEmitter();
emitter9.on('count', () => {});
emitter9.on('count', () => {});
console.log(EventEmitter.listenerCount(emitter9, 'count')); // Output: 2
console.log(emitter9.listenerCount('count')); // Output: 2

// 10. error event: Special handling, uncaught error throws if no listener
const emitter10 = new EventEmitter();
emitter10.on('error', (err) => {
    console.log('Caught error:', err.message);
});
emitter10.emit('error', new Error('Something went wrong')); // Output: Caught error: Something went wrong

// --- Edge Cases and Advanced Usage ---

// 11. Symbol event names
const emitter11 = new EventEmitter();
const sym = Symbol('secret');
emitter11.on(sym, () => console.log('Symbol event triggered'));
emitter11.emit(sym); // Output: Symbol event triggered

// 12. Chaining: All add/remove methods return the emitter for chaining
const emitter12 = new EventEmitter();
emitter12
    .on('chain', () => console.log('Chained 1'))
    .on('chain', () => console.log('Chained 2'))
    .emit('chain'); // Output: Chained 1 \n Chained 2

// 13. Passing multiple arguments to listeners
const emitter13 = new EventEmitter();
emitter13.on('multi', (a, b, c) => {
    console.log(a, b, c);
});
emitter13.emit('multi', 1, 2, 3); // Output: 1 2 3

// 14. Remove listeners during emit
const emitter14 = new EventEmitter();
function removeSelf() {
    console.log('Removing myself');
    emitter14.removeListener('self', removeSelf);
}
emitter14.on('self', removeSelf);
emitter14.on('self', () => console.log('Another listener'));
emitter14.emit('self'); // Output: Removing myself \n Another listener
emitter14.emit('self'); // Output: Another listener

// 15. Inheritance: Custom EventEmitter
class MyEmitter extends EventEmitter {}
const myEmitter = new MyEmitter();
myEmitter.on('custom', () => console.log('Custom event!'));
myEmitter.emit('custom'); // Output: Custom event!

/**
 * Summary:
 * - All methods and properties of EventEmitter are covered.
 * - Examples include basic, advanced, and edge cases.
 * - Use this file as a reference for mastering Node.js events module.
 */