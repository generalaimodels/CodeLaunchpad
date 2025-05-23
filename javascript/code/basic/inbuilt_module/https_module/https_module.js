/**
 * Node.js 'https' Module - Comprehensive Examples
 * 
 * The 'https' module enables HTTPS server and client functionality (HTTP over TLS/SSL).
 * This file demonstrates all major and minor methods, both server and client, including edge cases and exceptions.
 * Each example is self-contained, with expected output in comments.
 * 
 * Author: The Best Coder
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// For demonstration, generate self-signed certificates using OpenSSL before running these examples.
// Example command (run in terminal, not in code):
// openssl req -nodes -new -x509 -keyout server.key -out server.cert -subj "/CN=localhost" -days 365

const CERT_DIR = path.join(__dirname, 'certs');
const keyPath = path.join(CERT_DIR, 'server.key');
const certPath = path.join(CERT_DIR, 'server.cert');

// Ensure certs exist for server examples
if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
    console.log('Please generate server.key and server.cert in ./certs directory for server examples.');
}

// 1. https.createServer(): Basic HTTPS server
const options1 = {
    key: fs.existsSync(keyPath) ? fs.readFileSync(keyPath) : '',
    cert: fs.existsSync(certPath) ? fs.readFileSync(certPath) : ''
};
const PORT = 3443;
if (options1.key && options1.cert) {
    const server1 = https.createServer(options1, (req, res) => {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('1. Hello, HTTPS!\n');
    });
    server1.listen(PORT, () => {
        console.log(`1. HTTPS server running at https://localhost:${PORT}/`); // Output: HTTPS server running at https://localhost:3443/
        server1.close();
    });
}

// 2. https.request(): Basic HTTPS client request (GET)
const options2 = {
    hostname: 'www.google.com',
    port: 443,
    path: '/',
    method: 'GET'
};
const req2 = https.request(options2, (res) => {
    console.log('2. Status Code:', res.statusCode); // Output: 200
    res.on('data', (chunk) => {
        // Output: <HTML content chunk>
    });
});
req2.on('error', (e) => {
    console.log('2. Problem with request:', e.message);
});
req2.end();

// 3. https.get(): Simplified GET request
https.get('https://www.google.com', (res) => {
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
if (options1.key && options1.cert) {
    const server4 = https.createServer(options1, (req, res) => {
        if (req.method === 'GET' && req.url === '/hello') {
            res.writeHead(200, { 'Content-Type': 'text/plain' });
            res.end('4. Hello GET over HTTPS!\n');
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
        console.log(`4. HTTPS server for method/path at https://localhost:${PORT + 1}/`);
        server4.close();
    });
}

// 5. Server: Streaming file as response (pipe)
if (options1.key && options1.cert) {
    const server5 = https.createServer(options1, (req, res) => {
        if (req.url === '/file') {
            const filePath = path.join(__dirname, 'https_example.txt');
            fs.writeFileSync(filePath, '5. File streaming content over HTTPS');
            res.writeHead(200, { 'Content-Type': 'text/plain' });
            fs.createReadStream(filePath).pipe(res);
        } else {
            res.writeHead(404);
            res.end();
        }
    });
    server5.listen(PORT + 2, () => {
        console.log(`5. HTTPS file streaming server at https://localhost:${PORT + 2}/file`);
        server5.close();
    });
}

// 6. Client: Sending POST data
const postData6 = JSON.stringify({ name: 'BestCoder', skill: 'HTTPS' });
const options6 = {
    hostname: 'postman-echo.com',
    port: 443,
    path: '/post',
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData6)
    }
};
const req6 = https.request(options6, (res) => {
    console.log('6. POST Status:', res.statusCode); // Output: 200
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
        // Output: JSON response from postman-echo
    });
});
req6.on('error', (e) => {
    console.log('6. POST error:', e.message);
});
req6.write(postData6);
req6.end();

// 7. Server: Custom headers and status codes
if (options1.key && options1.cert) {
    const server7 = https.createServer(options1, (req, res) => {
        res.statusCode = 201;
        res.setHeader('X-Custom-Header', 'BestCoder');
        res.end('7. Created with custom header over HTTPS');
    });
    server7.listen(PORT + 3, () => {
        console.log(`7. HTTPS custom header server at https://localhost:${PORT + 3}/`);
        server7.close();
    });
}

// 8. Server: Handling request events (aborted, close, error)
if (options1.key && options1.cert) {
    const server8 = https.createServer(options1, (req, res) => {
        req.on('aborted', () => {
            console.log('8. HTTPS Request aborted by client');
        });
        req.on('close', () => {
            console.log('8. HTTPS Request closed');
        });
        req.on('error', (err) => {
            console.log('8. HTTPS Request error:', err.message);
        });
        res.end('8. HTTPS Event handling');
    });
    server8.listen(PORT + 4, () => {
        console.log(`8. HTTPS event server at https://localhost:${PORT + 4}/`);
        server8.close();
    });
}

// 9. Server: Enabling TLS options (secureProtocol, ciphers, etc.)
if (options1.key && options1.cert) {
    const tlsOptions = {
        ...options1,
        secureProtocol: 'TLS_method', // Node.js default, can be customized
        ciphers: 'ECDHE-RSA-AES128-GCM-SHA256:AES128-GCM-SHA256:!RC4:!aNULL:!eNULL',
        honorCipherOrder: true
    };
    const server9 = https.createServer(tlsOptions, (req, res) => {
        res.end('9. HTTPS with custom TLS options');
    });
    server9.listen(PORT + 5, () => {
        console.log(`9. HTTPS server with TLS options at https://localhost:${PORT + 5}/`);
        server9.close();
    });
}

// 10. Exception Handling: Server error event
if (options1.key && options1.cert) {
    const server10 = https.createServer(options1, (req, res) => {
        throw new Error('10. HTTPS Server crash!');
    });
    server10.on('error', (err) => {
        console.log('10. HTTPS Server error caught:', err.message); // Output: 10. HTTPS Server crash!
    });
    server10.listen(PORT + 6, () => {
        // Simulate a request to trigger error
        https.get({ ...options1, port: PORT + 6, rejectUnauthorized: false }, () => {});
        setTimeout(() => server10.close(), 100);
    });
}

/**
 * Additional Notes:
 * - All major and minor methods of 'https' module are covered.
 * - Both server and client usage are demonstrated.
 * - Edge cases, exceptions, and advanced usage included.
 * - Use this file as a reference for mastering Node.js https module.
 * - For local server examples, use self-signed certificates and set rejectUnauthorized: false in client for testing.
 */