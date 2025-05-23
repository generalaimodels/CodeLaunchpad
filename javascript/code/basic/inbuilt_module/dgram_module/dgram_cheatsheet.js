/**
 * dgram Module in Node.js
 * 
 * The 'dgram' module provides an implementation of UDP datagram sockets.
 * This file covers all major and minor methods, properties, and use-cases of the dgram module.
 * Each example is self-contained and includes expected output in comments.
 * 
 * Author: The Best Coder in the World
 */

const dgram = require('dgram');

// 1. Creating a UDP Socket (udp4)
/**
 * dgram.createSocket(type[, callback])
 * type: 'udp4' or 'udp6'
 * callback: function(msg, rinfo) - called on 'message' event
 */
const socket1 = dgram.createSocket('udp4');
console.log('1. UDP4 Socket created'); // Expected: UDP4 Socket created

// 2. Creating a UDP Socket (udp6)
const socket2 = dgram.createSocket('udp6');
console.log('2. UDP6 Socket created'); // Expected: UDP6 Socket created

// 3. Binding a Socket to a Port
socket1.bind(41234, () => {
    console.log('3. Socket1 bound to port 41234'); // Expected: Socket1 bound to port 41234
});

// 4. Listening for Messages
socket1.on('message', (msg, rinfo) => {
    console.log(`4. Received message: ${msg} from ${rinfo.address}:${rinfo.port}`);
    // Expected: Received message: Hello UDP from 127.0.0.1:PORT
});

// 5. Sending a Message
const message = Buffer.from('Hello UDP');
socket1.on('listening', () => {
    const address = socket1.address();
    dgram.createSocket('udp4').send(message, 0, message.length, address.port, '127.0.0.1', (err) => {
        if (!err) {
            console.log('5. Message sent'); // Expected: Message sent
        }
    });
});

// 6. Using send() with Offset and Length
const msg2 = Buffer.from('Partial Message Example');
socket1.send(msg2, 8, 7, 41234, '127.0.0.1', (err) => {
    if (!err) {
        console.log('6. Partial message sent'); // Expected: Partial message sent
    }
    // This will send 'Message' (bytes 8-14)
});

// 7. Using close() to Close the Socket
setTimeout(() => {
    socket1.close(() => {
        console.log('7. Socket1 closed'); // Expected: Socket1 closed
    });
}, 1000);

// 8. Handling 'error' Event
socket2.on('error', (err) => {
    console.log(`8. Socket2 error: ${err.message}`); // Expected: Socket2 error: <error message>
    socket2.close();
});
socket2.bind(9999, '::1', () => {
    // Intentionally send to an invalid address to trigger error
    socket2.send('Test', 0, 4, 9999, 'invalid_address');
});

// 9. Using setBroadcast() to Enable Broadcast
const socket3 = dgram.createSocket('udp4');
socket3.bind(() => {
    socket3.setBroadcast(true);
    console.log('9. Broadcast enabled on socket3'); // Expected: Broadcast enabled on socket3
    socket3.close();
});

// 10. Using setMulticastTTL(), setMulticastLoopback(), addMembership(), dropMembership()
const socket4 = dgram.createSocket('udp4');
const MULTICAST_ADDR = '239.255.255.250';
const PORT = 1900;

socket4.bind(PORT, () => {
    socket4.setMulticastTTL(128);
    socket4.setMulticastLoopback(true);
    socket4.addMembership(MULTICAST_ADDR);
    console.log('10. Multicast settings applied and membership added'); // Expected: Multicast settings applied and membership added

    // Send a multicast message
    socket4.send('Multicast Hello', 0, 15, PORT, MULTICAST_ADDR, (err) => {
        if (!err) {
            console.log('10. Multicast message sent'); // Expected: Multicast message sent
        }
        // Drop membership after sending
        socket4.dropMembership(MULTICAST_ADDR);
        console.log('10. Multicast membership dropped'); // Expected: Multicast membership dropped
        socket4.close();
    });
});

/**
 * Additional: Using ref() and unref()
 * These methods control the socket's participation in the Node.js event loop.
 */
const socket5 = dgram.createSocket('udp4');
socket5.bind(() => {
    socket5.unref();
    console.log('Additional: socket5 unref() called'); // Expected: socket5 unref() called
    socket5.ref();
    console.log('Additional: socket5 ref() called'); // Expected: socket5 ref() called
    socket5.close();
});

/**
 * Additional: Using address() to Get Socket Address Info
 */
const socket6 = dgram.createSocket('udp4');
socket6.bind(0, () => {
    const addr = socket6.address();
    console.log(`Additional: socket6 address: ${addr.address}:${addr.port}`); // Expected: socket6 address: 0.0.0.0:PORT
    socket6.close();
});

/**
 * Additional: Using setMulticastInterface()
 * Sets the outgoing interface for multicast packets.
 */
const socket7 = dgram.createSocket('udp4');
socket7.bind(() => {
    try {
        socket7.setMulticastInterface('0.0.0.0');
        console.log('Additional: Multicast interface set to 0.0.0.0'); // Expected: Multicast interface set to 0.0.0.0
    } catch (e) {
        console.log('Additional: setMulticastInterface error:', e.message);
    }
    socket7.close();
});

/**
 * Additional: Using setTTL()
 * Sets the IP Time To Live for outgoing packets.
 */
const socket8 = dgram.createSocket('udp4');
socket8.bind(() => {
    socket8.setTTL(64);
    console.log('Additional: TTL set to 64'); // Expected: TTL set to 64
    socket8.close();
});

/**
 * Additional: Using socket.type property
 */
const socket9 = dgram.createSocket('udp4');
console.log(`Additional: socket9 type is ${socket9.type}`); // Expected: socket9 type is udp4
socket9.close();

/**
 * Additional: Using socket.remoteAddress (after connect)
 */
const socket10 = dgram.createSocket('udp4');
socket10.connect(41234, '127.0.0.1', () => {
    console.log(`Additional: socket10 remote address: ${JSON.stringify(socket10.remoteAddress())}`); // Expected: {address: '127.0.0.1', family: 'IPv4', port: 41234}
    socket10.close();
});

/**
 * Exception Handling: Trying to send after close
 */
const socket11 = dgram.createSocket('udp4');
socket11.bind(() => {
    socket11.close();
    try {
        socket11.send('Test', 0, 4, 41234, '127.0.0.1', (err) => {
            if (err) {
                console.log('Exception: Error sending after close:', err.message); // Expected: Error sending after close: Not running
            }
        });
    } catch (e) {
        console.log('Exception: Caught error:', e.message);
    }
});

/**
 * Summary:
 * - Covered: createSocket, bind, send, close, setBroadcast, setMulticastTTL, setMulticastLoopback, addMembership, dropMembership, setMulticastInterface, setTTL, ref, unref, address, type, remoteAddress, error handling.
 * - Each example is self-contained and demonstrates a unique aspect of the dgram module.
 */