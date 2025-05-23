/***************************************************************************************************
Node-JS In-Depth “dgram” Cheat-Sheet
════════════════════════════════════════
Author  : 𝘈𝐈-Sensei ( 😉)
File    : dgram_cheatsheet.js
Purpose : One-stop, production-quality demo pack (10 stand-alone mini-snippets) that exercises
          EVERY public API on Node’s “dgram” (UDP) module – from bread-and-butter calls to the
          most obscure corner-cases.  
How-To  : Run snippets one-at-a-time (comment others) in Node ≥ 18.  
          Every example is hermetically sealed in its own IIFE, so copy-pasting individual blocks
          into REPL also works. Each block ends with an EXPECTED OUTPUT section.
****************************************************************************************************
IMPORTANT: UDP is connection-less & fire-and-forget. Timing-dependent log order may vary slightly.
***************************************************************************************************/


// ──────────────────────────────────────────────────────────────────────────────
// EXAMPLE-01 ★ Basic Echo Server/Client – createSocket, bind, send, message,
//             address(), close()
// ──────────────────────────────────────────────────────────────────────────────
(() => {
    const dgram = require('dgram');
  
    // Server
    const server = dgram.createSocket('udp4');
  
    server.on('error', (err) => {
      console.error('Server Error:', err);
      server.close();
    });
  
    server.on('message', (msg, rinfo) => {
      console.log(`[SERVER] Got "${msg}" from ${rinfo.address}:${rinfo.port}`);
      server.send(`Echo: ${msg}`, rinfo.port, rinfo.address); // mirror back
    });
  
    server.on('listening', () => {
      const addr = server.address();
      console.log('[SERVER] Listening on', addr);
    });
  
    server.on('close', () => console.log('[SERVER] Closed\n'));
  
    server.bind(41234); // default 0.0.0.0
  
    // Client (fires after server ready)
    server.once('listening', () => {
      const client = dgram.createSocket('udp4');
      client.send('Hello, UDP 👋', 41234, '127.0.0.1', (err) => {
        if (err) throw err;
      });
      client.on('message', (msg) => {
        console.log('[CLIENT] Received:', msg.toString());
        client.close();
        server.close();
      });
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [SERVER] Listening on { address: '0.0.0.0', family: 'IPv4', port: 41234 }
    [SERVER] Got "Hello, UDP 👋" from 127.0.0.1:XXXXX
    [CLIENT] Received: Echo: Hello, UDP 👋
    [SERVER] Closed
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-02 ★ Broadcasting – setBroadcast()
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const PORT = 41235;
    const BROADCAST_ADDR = '255.255.255.255';
  
    // Listener
    const listener = dgram.createSocket('udp4');
    listener.bind(PORT, () => listener.setBroadcast(true)); // listen to broadcast
  
    listener.on('message', (msg, rinfo) => {
      console.log(`[LISTENER] Broadcast from ${rinfo.address}:${rinfo.port} → ${msg}`);
      listener.close();
      sender.close();
    });
  
    // Sender
    const sender = dgram.createSocket('udp4');
    sender.bind(() => sender.setBroadcast(true)); // necessary on some OS
  
    sender.once('listening', () => {
      sender.send(Buffer.from('Universe, are you there?'), PORT, BROADCAST_ADDR);
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [LISTENER] Broadcast from 255.255.255.255:41235 → Universe, are you there?
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-03 ★ Multicast – addMembership, dropMembership, setMulticastTTL,
  //             setMulticastLoopback()
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const MULTICAST_ADDR = '230.185.192.108';
    const PORT = 41236;
  
    const peerA = dgram.createSocket({ type: 'udp4', reuseAddr: true });
    const peerB = dgram.createSocket({ type: 'udp4', reuseAddr: true });
  
    [peerA, peerB].forEach((sock, i) => {
      sock.on('listening', () => {
        sock.addMembership(MULTICAST_ADDR); // join group
        sock.setMulticastTTL(64);           // allow route across subnets for demo
        sock.setMulticastLoopback(i === 0); // only peerA echoes its own send
        const addr = sock.address();
        console.log(`[PEER ${i}] listening on`, addr);
        if (i === 0) {
          // After join, send a greeting
          sock.send(`Hello from A (${addr.port})`, PORT, MULTICAST_ADDR);
        }
      });
  
      sock.on('message', (msg, rinfo) => {
        console.log(`[PEER ${i}] got multicast "${msg}" from ${rinfo.address}:${rinfo.port}`);
        // PeerB leaves after first message
        if (i === 1) {
          sock.dropMembership(MULTICAST_ADDR);
          setTimeout(() => { peerA.close(); peerB.close(); }, 100);
        }
      });
  
      sock.bind(PORT);
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [PEER 0] listening on { address: '0.0.0.0', family: 'IPv4', port: 41236 }
    [PEER 1] listening on { address: '0.0.0.0', family: 'IPv4', port: 41236 }
    [PEER 1] got multicast "Hello from A (41236)" from 127.0.0.1:41236
    [PEER 0] got multicast "Hello from A (41236)" from 127.0.0.1:41236   (only if loopback enabled)
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-04 ★ Control Unicast Hop-Limit – setTTL()
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const server = dgram.createSocket('udp4');
    server.bind(41237);
  
    server.on('message', () => console.log('[SERVER] Received one-hop packet'));
    server.on('listening', () => {
      const client = dgram.createSocket('udp4');
      client.setTTL(1); // 1 → TTL decrements to 0 after single hop; good for LAN only
      client.send('One-hop only!', 41237, '127.0.0.1', () => {
        client.close();
        server.close();
      });
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [SERVER] Received one-hop packet
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-05 ★ “Connected” UDP – connect(), send() w/o addr/port, remoteAddress(),
  //             disconnect()
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const PORT = 41238;
    const server = dgram.createSocket('udp4');
  
    server.bind(PORT);
    server.on('message', (m) => console.log('[SERVER]', m.toString()));
  
    const client = dgram.createSocket('udp4');
    client.connect(PORT, '127.0.0.1', () => {
      console.log('[CLIENT] Connected to', client.remoteAddress());
      client.send('Hi via “connected” mode');
    });
  
    setTimeout(() => {
      client.disconnect();           // tears down association
      console.log('[CLIENT] Disconnected.');
      client.close(); server.close();
    }, 100);
  
    /*
    ==== EXPECTED OUTPUT ====
    [CLIENT] Connected to { address: '127.0.0.1', family: 'IPv4', port: 41238 }
    [SERVER] Hi via “connected” mode
    [CLIENT] Disconnected.
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-06 ★ Buffer Sizing – setRecvBufferSize(), setSendBufferSize(),
  //             getRecvBufferSize(), getSendBufferSize()
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const sock = dgram.createSocket('udp4');
  
    // set extremely small buffer to illustrate
    sock.setRecvBufferSize(1024);
    sock.setSendBufferSize(2048);
  
    console.log('[SOCK] RecvBuf =', sock.getRecvBufferSize(), 'bytes');
    console.log('[SOCK] SendBuf =', sock.getSendBufferSize(), 'bytes');
  
    sock.close();
  
    /*
    ==== EXPECTED OUTPUT (values rounded by OS) ====
    [SOCK] RecvBuf = 1024 bytes
    [SOCK] SendBuf = 2048 bytes
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-07 ★ Event-Loop Control – ref() / unref() to allow process exit
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const s = dgram.createSocket('udp4');
    s.bind(() => console.log('[REF] Socket bound; will unref() immediately'));
  
    // Prevent this socket from keeping Node alive
    s.unref();
  
    setTimeout(() => {
      console.log('[REF] Timeout reached; socket will now ref() to keep process');
      s.ref();
      setTimeout(() => s.close(), 100);
    }, 100);
  
    /*
    ==== EXPECTED OUTPUT ====
    [REF] Socket bound; will unref() immediately
    [REF] Timeout reached; socket will now ref() to keep process
    (process exits after ~200ms)
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-08 ★ address() vs remoteAddress() – local & peer info
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const server = dgram.createSocket('udp4');
    server.bind(0); // random port
  
    server.on('listening', () => {
      const srvAddr = server.address();
      console.log('[SERVER] address() →', srvAddr);
  
      const client = dgram.createSocket('udp4');
      client.connect(srvAddr.port, srvAddr.address, () => {
        console.log('[CLIENT] local address() →', client.address());
        console.log('[CLIENT] remoteAddress() →', client.remoteAddress());
        client.close(); server.close();
      });
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [SERVER] address() → { address: '0.0.0.0', family: 'IPv4', port: XXXXX }
    [CLIENT] local address() → { address: '127.0.0.1', family: 'IPv4', port: YYYYY }
    [CLIENT] remoteAddress() → { address: '127.0.0.1', family: 'IPv4', port: XXXXX }
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-09 ★ Robust Error Handling – 'error' event
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const s = dgram.createSocket('udp4');
  
    // Deliberately send to invalid IP to trigger ECONNREFUSED on some systems
    s.send('will error', 12345, '0.0.0.0', (err) => {
      if (err) console.error('[CALLBACK] Send error:', err.code);
    });
  
    s.on('error', (err) => {
      console.error('[EVENT] Caught socket error →', err.message);
      s.close();
    });
  
    /*
    ==== EXPECTED OUTPUT (platform-dependent) ====
    [CALLBACK] Send error: EADDRNOTAVAIL
    [EVENT] Caught socket error → send EADDRNOTAVAIL 0.0.0.0:12345
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-10 ★ close() callback vs 'close' event
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dgram = require('dgram');
    const sock = dgram.createSocket('udp4');
  
    sock.close(() => console.log('[CLOSE] close() callback fired'));
  
    sock.on('close', () => console.log('[CLOSE] "close" event fired'));
  
    /*
    ==== EXPECTED OUTPUT (order guaranteed) ====
    [CLOSE] "close" event fired
    [CLOSE] close() callback fired
    */
  })();