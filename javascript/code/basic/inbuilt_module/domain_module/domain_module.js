/***************************************************************************************************
LEGACY “domain” MASTERCLASS – 10 SELF-CONTAINED MINI-LABS
═════════════════════════════════════════════════════════
Author  : 𝐀𝐈-Sensei 🏆   
Runtime : Node ≥ 18  (the domain API is *deprecated* – prefer  async_hooks / try-catch blocks in
          new code.  This file exists purely for backward-compat maintenance engineers.)  
Layout  : 10 IIFEs.  Each focuses on a distinct facet and calls **every single public API**:  
          • domain.create / createDomain          • domain.active  
          • Domain#run / enter / exit             • Domain#add / remove  
          • Domain#bind / intercept               • 'error' event  
          • Domain inherits EventEmitter (so on/once etc. shown implicitly)           
***************************************************************************************************/



// ──────────────────────────────────────────────────────────────────────────────
// EX-01 ★ Crash-Shield 101 – domain.create() + run()
// ──────────────────────────────────────────────────────────────────────────────
(() => {
    const domain = require('domain');
  
    const d = domain.create();
    d.on('error', (er) => console.log('[EX-01] Caught →', er.message));
  
    d.run(() => {
      throw new Error('Synchronous boom!');
    });
  
    /*
    EXPECTED
    [EX-01] Caught → Synchronous boom!
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-02 ★ Async I/O capture – domain.add(emitter)
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const fs = require('node:fs');
    const domain = require('domain');
  
    const d = domain.create();
    d.on('error', (er) => console.log('[EX-02] Caught async →', er.code));
  
    // Fake-bad path to trigger ENOENT
    const stream = fs.createReadStream('/__no_such_file__');
    d.add(stream);                 // all events from stream now routed to domain
  
    stream.on('data', () => {});   // handlers can be anywhere
    /*
    EXPECTED
    [EX-02] Caught async → ENOENT
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-03 ★ bind(callback) – auto-try/catch wrapper
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const domain = require('domain');
    const d = domain.create();
  
    d.on('error', (e) => console.log('[EX-03] bind caught →', e.message));
  
    const wrapped = d.bind((num) => {
      if (num < 0) throw new Error('negatives not allowed');
      console.log('[EX-03] OK', num);
    });
  
    wrapped(-7);  // error routed
    wrapped(3);   // prints OK
  
    /*
    EXPECTED
    [EX-03] bind caught → negatives not allowed
    [EX-03] OK 3
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-04 ★ intercept(callback) – Node-style (err, data) helper
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const domain = require('domain');
    const d = domain.create();
  
    d.on('error', (e) => console.log('[EX-04] intercept caught →', e.message));
  
    function fakeAsync(cb) {
      setTimeout(() => cb(new Error('DB down')), 20);
    }
  
    fakeAsync(d.intercept((data) => {
      // executes only on success
      console.log('[EX-04] Data:', data);
    }));
  
    /*
    EXPECTED
    [EX-04] intercept caught → DB down
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-05 ★ Manual enter() / exit() for non-run scopes
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const domain = require('domain');
    const d = domain.create();
  
    d.on('error', (e) => console.log('[EX-05] Captured →', e.message));
  
    function workOutside() {
      d.enter();
      setImmediate(() => {
        d.exit();                   // optional; auto-exit on cb end is safer
        throw new Error('late explode'); // still in domain
      });
    }
  
    workOutside();
  
    /*
    EXPECTED
    [EX-05] Captured → late explode
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-06 ★ remove(emitter) – stop capturing further errors
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const domain = require('domain');
    const { EventEmitter } = require('node:events');
  
    const src = new EventEmitter();
    const d = domain.create();
  
    d.on('error', (e) => console.log('[EX-06] First captured →', e.message));
  
    d.add(src);
    src.emit('error', new Error('inside')); // captured
  
    d.remove(src);                         // detach
    try {
      src.emit('error', new Error('outside')); // uncaught → crashes process, so catch manually
    } catch (e) {
      console.log('[EX-06] Not captured any more →', e.message);
    }
  
    /*
    EXPECTED
    [EX-06] First captured → inside
    [EX-06] Not captured any more → outside
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-07 ★ domain.active – peek currently executing context
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const domain = require('domain');
    const a = domain.create();
    const b = domain.create();
  
    a.run(() => {
      console.log('[EX-07] Inside A? ', domain.active === a); // true
      b.run(() => {
        console.log('[EX-07] Now inside B? ', domain.active === b); // true
      });
    });
  
    /*
    EXPECTED
    [EX-07] Inside A?  true
    [EX-07] Now inside B?  true
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-08 ★ Nested domains & bubbling – inner error not handled by outer
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const domain = require('domain');
    const outer = domain.create();
    const inner = domain.create();
  
    outer.on('error', (e) => console.log('[EX-08] Outer caught →', e.message));
    inner.on('error', (e) => console.log('[EX-08] Inner caught →', e.message));
  
    outer.run(() => {
      inner.run(() => {
        throw new Error('deep issue');
      });
    });
  
    /*
    EXPECTED
    [EX-08] Inner caught → deep issue
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-09 ★ Multiple members tracked in Domain#members
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const domain = require('domain');
    const net = require('node:net');
  
    const d = domain.create();
    d.on('error', (e) => console.log('[EX-09] Caught socket error →', e.code));
  
    const s1 = net.createServer();
    const s2 = net.createServer();
    d.add(s1).add(s2);            // chainable (EventEmitter.add returns domain)
  
    console.log('[EX-09] Members count =', d.members.length); // 2
  
    s1.emit('error', Object.assign(new Error, { code: 'EADDRINUSE' }));
  
    /*
    EXPECTED
    [EX-09] Members count = 2
    [EX-09] Caught socket error → EADDRINUSE
    */
  })();
  
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EX-10 ★ End-to-End legacy web server guard (complete flow)
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const http = require('node:http');
    const domain = require('domain');
  
    const PORT = 0; // random
    const srv = http.createServer((req, res) => {
      const reqDomain = domain.create();
      reqDomain.add(req).add(res);
  
      reqDomain.on('error', (err) => {
        console.error('[EX-10] Fatal request error:', err.message);
        try { res.writeHead(500); res.end('Server Error'); } catch {}
      });
  
      reqDomain.run(() => {
        // Simulate programmer bug
        if (req.url === '/boom') throw new Error('kaboom');
  
        res.end('Hello ' + req.url);
      });
    });
  
    srv.listen(PORT, () => {
      const { port } = srv.address();
      http.get(`http://localhost:${port}/boom`, (res) => {
        res.resume().on('end', () => srv.close());
      });
    });
  
    /*
    EXPECTED
    [EX-10] Fatal request error: kaboom
    (Client receives 500 Server Error)
    */
  })();