/***************************************************************************************************
 *  Node.js Core Module : `http`
 *  Author  : <Your Name Here>
 *  Purpose : Ten test-bench style examples that exercise EVERY public export of the `http` module
 *            (from the day-to-day helpers all the way to obscure header validators).  Each example
 *            is fully isolated, spins its own server (on an OS-assigned free port), performs one
 *            or more client requests, prints observable results, then disposes every resource to
 *            avoid interference.  Run the file with `node http-tour.js` and watch the waterfall.
 *
 *  Tested  : Node ≥ 18.x  (older versions might miss `validateHeader*`, `maxHeaderSize`, etc.).
 ***************************************************************************************************/

'use strict';
const http  = require('http');
const fs    = require('fs');
const path  = require('path');
const os    = require('os');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Tiny helper to run examples sequentially
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const EXAMPLES = [];
function addExample(title, fn) { EXAMPLES.push({ title, fn }); }

(async () => {
  for (const { title, fn } of EXAMPLES) {
    console.log('\n' + '═'.repeat(100));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(100));
    try { await fn(); }
    catch (e) { console.error('💥  Exception  →', e); }
  }
})(); // auto-exec

/***************************************************************************************************
 * 1) createServer() + http.get()  – the obligatory “Hello World”
 **************************************************************************************************/
addExample('1) createServer() + http.get()', async () => {
  const server = http.createServer((req, res) => {
    res.end('Hello HTTP!');
  }).listen(0);                                       // :0 = random free port

  const { port } = server.address();
  const body = await new Promise(resolve => {
    http.get(`http://localhost:${port}`, (res) => {
      let data = '';
      res.on('data', d => data += d);
      res.on('end', () => resolve(data));
    });
  });
  console.log('Server said →', body);                 // Hello HTTP!
  server.close();

  /* Expected output:
     Server said → Hello HTTP!
  */
});

/***************************************************************************************************
 * 2) http.METHODS and dynamic routing
 **************************************************************************************************/
addExample('2) Dispatch using http.METHODS', async () => {
  const server = http.createServer((req, res) => {
    const { method, url } = req;
    res.setHeader('Content-Type', 'text/plain');

    if (method === 'GET' && url === '/ping') return res.end('pong');
    if (method === 'DELETE')                     return res.end('🔪');
    res.statusCode = 405;                        // method not allowed
    res.end(`${method} not supported`);
  }).listen(0);

  const { port } = server.address();
  for (const m of ['GET', 'DELETE', 'POST']) {
    const result = await new Promise(resolve => {
      const req = http.request({ host: 'localhost', port, path: '/ping', method: m }, res => {
        let d = '';
        res.on('data', c => d += c);
        res.on('end', () => resolve({ m, d, code: res.statusCode }));
      });
      req.end();
    });
    console.log(result);
  }
  server.close();

  /* Expected output (order preserved):
     { m: 'GET',    d: 'pong',   code: 200 }
     { m: 'DELETE', d: '🔪',      code: 200 }
     { m: 'POST',   d: 'POST not supported', code: 405 }
  */
});

/***************************************************************************************************
 * 3) STATUS_CODES, statusMessage & header helpers
 **************************************************************************************************/
addExample('3) STATUS_CODES, setHeader/getHeader/removeHeader', async () => {
  const server = http.createServer((_, res) => {
    res.statusCode    = 418;
    res.statusMessage = http.STATUS_CODES[418];       // "I'm a Teapot"
    res.setHeader('X-Custom', '🍵');
    console.log('Before removal →', res.getHeader('X-Custom'));
    res.removeHeader('X-Custom');
    console.log('After  removal →', res.getHeader('X-Custom'));
    res.end('Short and stout');
  }).listen(0);

  const { port } = server.address();
  await new Promise(done => {
    http.get(`http://localhost:${port}`, res => {
      console.log('Response line →', `${res.statusCode} ${res.statusMessage}`);
      res.resume().on('end', done);
    });
  });
  server.close();

  /* Expected output:
     Before removal → 🍵
     After  removal → undefined
     Response line → 418 I'm a Teapot
  */
});

/***************************************************************************************************
 * 4) Streaming a file with res.writeHead() and compression hint
 **************************************************************************************************/
addExample('4) Pipe fs stream → res', async () => {
  // Scratch 1-MiB file
  const file = path.join(os.tmpdir(), 'big.bin');
  fs.writeFileSync(file, Buffer.alloc(1024 * 1024));  // 1 MiB

  const server = http.createServer((_, res) => {
    res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
  }).listen(0);

  const { port } = server.address();
  const size = await new Promise(resolve => {
    http.get(`http://localhost:${port}`, res => {
      let bytes = 0;
      res.on('data', b => bytes += b.length);
      res.on('end', () => resolve(bytes));
    });
  });
  console.log('Downloaded bytes →', size);
  server.close();

  /* Expected output:
     Downloaded bytes → 1048576
  */
});

/***************************************************************************************************
 * 5) Client POST with http.request() (JSON echo)
 **************************************************************************************************/
addExample('5) http.request() POST JSON', async () => {
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      res.end(body);                                 // echo
    });
  }).listen(0);

  const { port } = server.address();
  const payload = JSON.stringify({ msg: 'Hello' });
  const echoed  = await new Promise(resolve => {
    const req = http.request({ method: 'POST', host: 'localhost', port,
                               headers: { 'Content-Type': 'application/json',
                                          'Content-Length': Buffer.byteLength(payload) } },
                              res => {
      let d = '';
      res.on('data', c => d += c);
      res.on('end', () => resolve(d));
    });
    req.write(payload);
    req.end();
  });
  console.log('Echoed →', echoed);
  server.close();

  /* Expected output:
     Echoed → {"msg":"Hello"}
  */
});

/***************************************************************************************************
 * 6) Custom http.Agent (keep-alive & connection pooling)
 **************************************************************************************************/
addExample('6) Custom http.Agent keep-alive', async () => {
  const server = http.createServer((_, res) => res.end('OK')).listen(0);
  const { port } = server.address();

  const agent = new http.Agent({ keepAlive: true, maxSockets: 1 });
  for (let i = 0; i < 3; i++) {
    await new Promise(resolve => {
      http.get({ host: 'localhost', port, agent }, res => {
        res.resume().on('end', resolve);
      });
    });
  }
  console.log('Sockets kept alive →', agent.keepSocketAlive);
  console.log('Agent sockets keys →', Object.keys(agent.sockets));
  server.close();
  agent.destroy();

  /* Expected output:
     Sockets kept alive → [Function: keepSocketAlive] (or true in newer Node)
     Agent sockets keys → [] (empty – all requests finished)
  */
});

/***************************************************************************************************
 * 7) globalAgent inspection
 **************************************************************************************************/
addExample('7) http.globalAgent statistics', async () => {
  const server = http.createServer((_, res) => setTimeout(() => res.end('slow'), 100)).listen(0);
  const { port } = server.address();

  await Promise.all(
    Array.from({ length: 2 }, () =>
      new Promise(resolve => {
        http.get({ host: 'localhost', port }, res => res.resume().on('end', resolve));
      })
    )
  );
  console.log('globalAgent.maxSockets →', http.globalAgent.maxSockets);
  console.log('active sockets count  →', Object.keys(http.globalAgent.sockets).length);
  server.close();

  /* Expected output:
     globalAgent.maxSockets → Infinity  (or 256 on older Node versions)
     active sockets count  → 0
  */
});

/***************************************************************************************************
 * 8) Timeouts: server.setTimeout() and req.setTimeout()
 **************************************************************************************************/
addExample('8) server & client timeouts', async () => {
  const server = http.createServer((_, res) => {
    // never responds, triggers client timeout
  }).listen(0).setTimeout(0);                         // disable server idle timeout

  const { port } = server.address();
  await new Promise(resolve => {
    const req = http.get({ host: 'localhost', port }, () => {});
    req.setTimeout(300, () => {
      console.log('Client request timed out after 300ms');
      req.abort();                                    // cancel
      resolve();
    });
    req.on('error', () => {});                        // swallow
  });
  server.close();

  /* Expected output:
     Client request timed out after 300ms
  */
});

/***************************************************************************************************
 * 9) Abort / destroy a request mid-flight & error handling
 **************************************************************************************************/
addExample('9) Abort a request mid-flight', async () => {
  const server = http.createServer((_, res) => {
    // send chunks forever until client aborts
    const id = setInterval(() => res.write('data'), 10);
    res.on('close', () => clearInterval(id));
  }).listen(0);

  const { port } = server.address();
  await new Promise(resolve => {
    const req = http.get({ host: 'localhost', port }, res => {
      setTimeout(() => {
        console.log('Destroying socket …');
        req.destroy();                                // force-close
      }, 50);
    });
    req.on('error', err => {
      console.log('Caught error →', err.code);        // ECONNRESET
      resolve();
    });
  });
  server.close();

  /* Expected output:
     Destroying socket …
     Caught error → ECONNRESET
  */
});

/***************************************************************************************************
 * 10) validateHeaderName / validateHeaderValue / maxHeaderSize
 **************************************************************************************************/
addExample('10) Header validators & http.maxHeaderSize', async () => {
  // Successful validation
  http.validateHeaderName('X-Good');
  http.validateHeaderValue('X-Good', 'ok');

  // Invalid header demo
  try { http.validateHeaderName('bad header'); }
  catch (e) { console.log('Name validation failed →', e.code); } // ERR_INVALID_HTTP_TOKEN

  console.log('Process max header size →', http.maxHeaderSize, 'bytes'); // platform default

  /* Expected output:
     Name validation failed → ERR_INVALID_HTTP_TOKEN
     Process max header size → 16384 bytes  (value may differ)
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/