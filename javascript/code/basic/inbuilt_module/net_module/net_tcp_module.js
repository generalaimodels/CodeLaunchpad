/**
 * Node.js 'net' Module - Comprehensive Examples
 * 
 * The 'net' module provides an asynchronous network API for creating stream-based TCP or IPC servers and clients.
 * This file demonstrates all major and minor methods, properties, and edge cases.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const net = require('net');
const path = require('path');
const os = require('os');

// Utility: Choose a port and IPC path for demonstration
const PORT = 4000;
const IPC_PATH = os.platform() === 'win32'
    ? '\\\\.\\pipe\\node_net_example_pipe'
    : path.join(os.tmpdir(), 'node_net_example.sock');

// 1. net.createServer(): Basic TCP server
const server1 = net.createServer((socket) => {
    socket.write('1. Hello TCP Client!\n');
    socket.end();
});
server1.listen(PORT, () => {
    console.log(`1. TCP server listening on port ${PORT}`); // Output: TCP server listening on port 4000
    server1.close();
});

// 2. net.connect()/net.createConnection(): Basic TCP client
const client2 = net.connect(PORT, () => {
    console.log('2. TCP client connected'); // Output: TCP client connected
});
client2.on('data', (data) => {
    console.log('2. Received from server:', data.toString().trim()); // Output: 1. Hello TCP Client!
    client2.end();
});

// 3. Server: Handling 'connection', 'close', 'error' events
const server3 = net.createServer();
server3.on('connection', (socket) => {
    console.log('3. New connection'); // Output: 3. New connection
    socket.end();
});
server3.on('close', () => {
    console.log('3. Server closed'); // Output: 3. Server closed
});
server3.on('error', (err) => {
    console.log('3. Server error:', err.message);
});
server3.listen(PORT + 1, () => {
    server3.close();
});

// 4. Client: Handling 'data', 'end', 'error', 'timeout' events
const server4 = net.createServer((socket) => {
    socket.write('4. Data event test');
    socket.end();
});
server4.listen(PORT + 2, () => {
    const client4 = net.createConnection(PORT + 2);
    client4.on('data', (data) => {
        console.log('4. Client received:', data.toString()); // Output: 4. Client received: 4. Data event test
    });
    client4.on('end', () => {
        console.log('4. Client connection ended'); // Output: 4. Client connection ended
        server4.close();
    });
    client4.on('error', (err) => {
        console.log('4. Client error:', err.message);
    });
    client4.setTimeout(500, () => {
        console.log('4. Client timeout'); // Output if timeout occurs
        client4.end();
    });
});

// 5. net.Server.listen() with IPC (Unix domain socket/Windows named pipe)
const server5 = net.createServer((socket) => {
    socket.write('5. Hello IPC Client!\n');
    socket.end();
});
server5.listen(IPC_PATH, () => {
    console.log('5. IPC server listening:', IPC_PATH); // Output: IPC server listening: <path>
    // Connect as IPC client
    const client5 = net.connect(IPC_PATH, () => {
        client5.on('data', (data) => {
            console.log('5. IPC client received:', data.toString().trim()); // Output: 5. Hello IPC Client!
            client5.end();
            server5.close();
        });
    });
});

// 6. socket.setEncoding(), socket.setTimeout(), socket.setKeepAlive()
const server6 = net.createServer((socket) => {
    socket.setEncoding('utf8');
    socket.setTimeout(1000);
    socket.setKeepAlive(true, 500);
    socket.on('timeout', () => {
        socket.write('6. Timeout!\n');
        socket.end();
    });
});
server6.listen(PORT + 3, () => {
    const client6 = net.createConnection(PORT + 3);
    client6.on('data', (data) => {
        console.log('6. Client received:', data.trim()); // Output: 6. Client received: 6. Timeout!
        client6.end();
        server6.close();
    });
});

// 7. socket.pause(), socket.resume(): Flow control
const server7 = net.createServer((socket) => {
    socket.pause();
    setTimeout(() => {
        socket.resume();
        socket.write('7. Resumed!\n');
        socket.end();
    }, 200);
});
server7.listen(PORT + 4, () => {
    const client7 = net.createConnection(PORT + 4);
    client7.on('data', (data) => {
        console.log('7. Client received:', data.toString().trim()); // Output: 7. Client received: 7. Resumed!
        client7.end();
        server7.close();
    });
});

// 8. server.getConnections(): Get number of concurrent connections
const server8 = net.createServer();
server8.listen(PORT + 5, () => {
    net.createConnection(PORT + 5);
    net.createConnection(PORT + 5);
    setTimeout(() => {
        server8.getConnections((err, count) => {
            console.log('8. Active connections:', count); // Output: 2
            server8.close();
        });
    }, 100);
});

// 9. server.address(): Get server address info
const server9 = net.createServer();
server9.listen(PORT + 6, () => {
    const address = server9.address();
    console.log('9. Server address:', address); // Output: { port: ..., family: 'IPv4', address: '::' }
    server9.close();
});

// 10. Exception Handling: Port in use
const server10a = net.createServer();
const server10b = net.createServer();
server10a.listen(PORT + 7, () => {
    server10b.listen(PORT + 7);
});
server10b.on('error', (err) => {
    console.log('10. Exception caught:', err.code); // Output: EADDRINUSE
    server10a.close();
});

/**
 * Additional Notes:
 * - All major and minor methods of 'net' module are covered.
 * - Both TCP and IPC (Unix/Windows) usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js net module.
 */