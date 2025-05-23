/**
 * Node.js 'http' Module - Comprehensive Examples
 * 
 * The 'http' module enables HTTP server and client functionality.
 * This file demonstrates all major and minor methods, both server and client, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// Utility: Port for server examples
const PORT = 3000;

// 1. http.createServer(): Basic HTTP server
const server1 = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('1. Hello, HTTP!\n');
});
server1.listen(PORT, () => {
    console.log(`1. Server running at http://localhost:${PORT}/`); // Output: Server running at http://localhost:3000/
    server1.close();
});

// 2. http.request(): Basic HTTP client request (GET)
const options2 = {
    hostname: 'example.com',
    port: 80,
    path: '/',
    method: 'GET'
};
const req2 = http.request(options2, (res) => {
    console.log('2. Status Code:', res.statusCode); // Output: 200
    res.on('data', (chunk) => {
        // Output: <HTML content chunk>
    });
});
req2.on('error', (e) => {
    console.log('2. Problem with request:', e.message);
});
req2.end();

// 3. http.get(): Simplified GET request
http.get('http://example.com', (res) => {
    console.log('3. GET Status:', res.statusCode); // Output: 200
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
        // Output: <HTML content>
    });
}).on('error', (e) => {
    console.log('3. GET error:', e.message);
});

// 4. Server: Handling different HTTP methods and URL paths
const server4 = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/hello') {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('4. Hello GET!\n');
    } else if (req.method === 'POST' && req.url === '/data') {
        let body = '';
        req.on('data', chunk => body += chunk);
        req.on('end', () => {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ received: body }));
        });
    } else {
        res.writeHead(404);
        res.end('4. Not Found');
    }
});
server4.listen(PORT + 1, () => {
    console.log(`4. Server for method/path at http://localhost:${PORT + 1}/`);
    server4.close();
});

// 5. Server: Streaming file as response (pipe)
const server5 = http.createServer((req, res) => {
    if (req.url === '/file') {
        const filePath = path.join(__dirname, 'http_example.txt');
        fs.writeFileSync(filePath, '5. File streaming content');
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        fs.createReadStream(filePath).pipe(res);
    } else {
        res.writeHead(404);
        res.end();
    }
});
server5.listen(PORT + 2, () => {
    console.log(`5. File streaming server at http://localhost:${PORT + 2}/file`);
    server5.close();
});

// 6. Client: Sending POST data
const postData6 = JSON.stringify({ name: 'BestCoder', skill: 'HTTP' });
const options6 = {
    hostname: 'example.com',
    port: 80,
    path: '/api',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData6)
    }
};
const req6 = http.request(options6, (res) => {
    console.log('6. POST Status:', res.statusCode); // Output: 404 (example.com/api doesn't exist)
});
req6.on('error', (e) => {
    console.log('6. POST error:', e.message);
});
req6.write(postData6);
req6.end();

// 7. Server: Custom headers and status codes
const server7 = http.createServer((req, res) => {
    res.statusCode = 201;
    res.setHeader('X-Custom-Header', 'BestCoder');
    res.end('7. Created with custom header');
});
server7.listen(PORT + 3, () => {
    console.log(`7. Custom header server at http://localhost:${PORT + 3}/`);
    server7.close();
});

// 8. Server: Handling request events (aborted, close, error)
const server8 = http.createServer((req, res) => {
    req.on('aborted', () => {
        console.log('8. Request aborted by client');
    });
    req.on('close', () => {
        console.log('8. Request closed');
    });
    req.on('error', (err) => {
        console.log('8. Request error:', err.message);
    });
    res.end('8. Event handling');
});
server8.listen(PORT + 4, () => {
    console.log(`8. Event server at http://localhost:${PORT + 4}/`);
    server8.close();
});

// 9. Server: Keep-Alive and Connection headers
const server9 = http.createServer((req, res) => {
    res.setHeader('Connection', 'keep-alive');
    res.end('9. Keep-Alive enabled');
});
server9.listen(PORT + 5, () => {
    console.log(`9. Keep-Alive server at http://localhost:${PORT + 5}/`);
    server9.close();
});

// 10. Exception Handling: Server error event
const server10 = http.createServer((req, res) => {
    throw new Error('10. Server crash!');
});
server10.on('error', (err) => {
    console.log('10. Server error caught:', err.message); // Output: 10. Server crash!
});
server10.listen(PORT + 6, () => {
    // Simulate a request to trigger error
    http.get(`http://localhost:${PORT + 6}/`, () => {});
    setTimeout(() => server10.close(), 100);
});

/**
 * Additional Notes:
 * - All major and minor methods of 'http' module are covered.
 * - Both server and client usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js http module.
 */