/***************************************************************************************************
 *  Node.js Core Module : `net`
 *  Author  : <Your Name Here>
 *  Purpose : 10 crystal-clear, console-driven examples that exercise virtually every public API of
 *             the TCP / IPC module loader (`net`).  The snippets run sequentially; each prints its
 *             own section header and cleans after itself (servers closed, sockets destroyed) so
 *             you can execute the file with `node net-tour.js` and read the logs top-to-bottom.
 *
 *  Tested  : Node ≥ 18.x
 ***************************************************************************************************/
'use strict';
const net  = require('net');
const os   = require('os');
const path = require('path');
const { once } = require('events');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Tiny sequential runner
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const STEPS = [];
function step(title, fn) { STEPS.push({ title, fn }); }
(async () => {
  for (const { title, fn } of STEPS) {
    console.log('\n' + '═'.repeat(105));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(105));
    try { await fn(); }
    catch (e) { console.error('💥  Exception →', e); }
  }
})();

/***************************************************************************************************
 * 1) Basic TCP echo – createServer / createConnection / write / end
 **************************************************************************************************/
step('1) TCP echo  (createServer  &  createConnection)', async () => {
  const server = net.createServer((sock) => {
    console.log('[server] client connected ⇒', sock.remoteAddress, sock.remotePort);
    sock.write('hello\n');                           // welcome banner
    sock.on('data', (d) => {
      console.log('[server] payload:', d.toString().trim());
      sock.end();                                    // half-duplex close
    });
  }).listen(0, '127.0.0.1');                         // :0 chooses free port

  const { port } = server.address();
  const client = net.createConnection({ port, host: '127.0.0.1' }, () => {
    console.log('[client] connected');
  });
  client.once('data', (d) => {
    console.log('[client] greeting:', d.toString().trim());
    client.write('thanks!');
  });

  await once(client, 'close');                       // wait until socket finishes
  server.close();

  /* Expected output (port numbers vary):
     [server] client connected ⇒ 127.0.0.1 52734
     [client] connected
     [client] greeting: hello
     [server] payload: thanks!
  */
});

/***************************************************************************************************
 * 2) Stream control – setEncoding / pause / resume
 **************************************************************************************************/
step('2) setEncoding + pause/resume', async () => {
  const server = net.createServer((sock) => {
    sock.write('A'.repeat(1e4));                     // 10 KB burst
    sock.end();
  }).listen(0);

  const client = net.connect(server.address().port);
  client.setEncoding('utf8');                        // auto-stringify
  client.pause();                                    // stop flowing – buffer fills
  setTimeout(() => {
    console.log('[client] resume after 200ms');
    client.resume();
  }, 200);

  let bytes = 0;
  client.on('data', (chunk) => bytes += chunk.length);
  await once(client, 'end');
  console.log('[client] total chars →', bytes);
  server.close();

  /* Expected output:
     [client] resume after 200ms
     [client] total chars → 10000
  */
});

/***************************************************************************************************
 * 3) setTimeout() & destroy()  – killing slow clients
 **************************************************************************************************/
step('3) Socket timeout & destroy', async () => {
  const server = net.createServer((sock) => {
    sock.setTimeout(300);                            // 300 ms idle window
    sock.on('timeout', () => {
      console.log('[server] client idle → destroying');
      sock.destroy();                                // hard close
    });
  }).listen(0);

  const client = net.connect(server.address().port);
  client.on('error', () => {});                      // ECONNRESET after destroy
  await once(client, 'close');
  server.close();

  /* Expected output:
     [server] client idle → destroying
  */
});

/***************************************************************************************************
 * 4) server.getConnections()  +  server.address()
 **************************************************************************************************/
step('4) getConnections & address()', async () => {
  const server = net.createServer().listen(0);
  const { port } = server.address();
  net.connect(port);                                 // open 1 client
  await once(server, 'connection');

  const total = await new Promise((res, rej) =>
    server.getConnections((err, n) => err ? rej(err) : res(n)));
  console.log('Active connections →', total);
  console.log('Address object    →', server.address());
  server.close();

  /* Expected output:
     Active connections → 1
     Address object    → { address: '127.0.0.1', family: 'IPv4', port: <port> }
  */
});

/***************************************************************************************************
 * 5) Nagle & Keep-Alive – setNoDelay / setKeepAlive
 **************************************************************************************************/
step('5) setNoDelay() & setKeepAlive()', async () => {
  const server = net.createServer((sock) => sock.end()).listen(0);
  const sock   = net.connect(server.address().port);

  sock.setNoDelay(true);                             // disable Nagle
  sock.setKeepAlive(true, 10_000);                   // probes every 10 s

  console.log('Nagle disabled →', sock.noDelay);
  console.log('Keep-alive     →', true);             // no official getter
  await once(sock, 'close');
  server.close();

  /* Expected output:
     Nagle disabled → true
     Keep-alive     → true
  */
});

/***************************************************************************************************
 * 6) ref()/unref()  – sockets & servers that don’t hold the event-loop open
 **************************************************************************************************/
step('6) ref / unref demo (process exits quickly)', async () => {
  const server = net.createServer().listen(0);
  server.unref();                                    // server alone won’t keep loop alive

  const sock = net.connect(server.address().port);
  sock.unref();                                      // same for client

  console.log('Both server & socket un-refed – event loop can quit if nothing else pending');
  await once(sock, 'connect');
  sock.end(); server.close();

  /* Expected output:
     Both server & socket un-refed – event loop can quit if nothing else pending
  */
});

/***************************************************************************************************
 * 7) IPC (UNIX domain socket / Windows named pipe)
 **************************************************************************************************/
step('7) IPC socket – path option', async () => {
  const sockPath = process.platform === 'win32'
    ? '\\\\.\\pipe\\net-tour-' + process.pid
    : path.join(os.tmpdir(), `net-tour-${process.pid}.sock`);

  const server = net.createServer((c) => c.end('pong')).listen(sockPath);
  const client = net.connect({ path: sockPath });
  client.setEncoding('utf8');
  client.on('data', (d) => console.log('Received →', d.trim()));
  await once(client, 'end');
  server.close();

  /* Expected output:
     Received → pong
  */
});

/***************************************************************************************************
 * 8) net.isIP / isIPv4 / isIPv6
 **************************************************************************************************/
step('8) IP validators', () => {
  console.log('isIP("127.0.0.1")  →', net.isIP('127.0.0.1'));     // 4
  console.log('isIPv4("::1")      →', net.isIPv4('::1'));         // false
  console.log('isIPv6("fe80::1")  →', net.isIPv6('fe80::1'));     // true
  /* Expected output:
     isIP("127.0.0.1")  → 4
     isIPv4("::1")      → false
     isIPv6("fe80::1")  → true
  */
});

/***************************************************************************************************
 * 9) Alias net.connect()  &  options-object style
 **************************************************************************************************/
step('9) net.connect() alias, options object', async () => {
  const server = net.createServer((s) => s.end('hi')).listen(0);
  const info   = server.address();
  const client = net.connect({ port: info.port, host: info.address }, () => {
    console.log('[client] connected using net.connect alias');
  });
  client.on('data', (d) => console.log('[client] data:', d.toString().trim()));
  await once(client, 'close');
  server.close();

  /* Expected output:
     [client] connected using net.connect alias
     [client] data: hi
  */
});

/***************************************************************************************************
 * 10) allowHalfOpen – reading after remote FIN
 **************************************************************************************************/
step('10) allowHalfOpen = true (half-duplex)', async () => {
  const server = net.createServer({ allowHalfOpen: true }, (s) => {
    s.on('end', () => {                // remote FIN but keep our side open
      console.log('[server] peer ended – still open; sending reply');
      s.write('late-reply');
      s.end();
    });
  }).listen(0);

  const client = net.connect(server.address().port);
  client.end('early-msg');             // send & FIN immediately
  client.on('data', (d) => console.log('[client] got:', d.toString()));
  await once(client, 'end');
  server.close();

  /* Expected output:
     [server] peer ended – still open; sending reply
     [client] got: late-reply
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/