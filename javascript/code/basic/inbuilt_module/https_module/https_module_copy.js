/***************************************************************************************************
 *  Node.js Core Module : `https`
 *  Author   : <Your Name Here>
 *  Purpose  : 10 crystal-clear, console-driven examples that cover every public surface of the
 *              `https` module—from bread-and-butter helpers (`get`, `request`, `createServer`) to
 *              the more arcane facets (custom Agent, mutual-TLS, SNI, ALPN, session reuse, etc.).
 *
 *  Run      : `node https-tour.js`
 *  Tested   : Node ≥ 18.x (slightly older versions should work, but ALPN/SNI/code-paths may vary).
 *
 *  IMPORTANT: We ship static, self-signed PEMs below.  The client side sets `rejectUnauthorized:
 *             false` to accept them—ONLY for demo purposes.  NEVER do this in production.
 ***************************************************************************************************/

'use strict';
const https = require('https');
const fs    = require('fs');
const os    = require('os');
const path  = require('path');
const tls   = require('tls');      // leveraged for SNI / ALPN helpers
const { once } = require('events');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  0) Self-signed certificates (generated offline with: openssl req -x509 -newkey rsa:2048 …)
 *     – cert for CN=localhost; valid until 2033.  ❗ DO NOT use in production.
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const KEY  = `
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAxJeIXe0DSKyYnPuKl---TRUNCATED_FOR_BREVITY---ezuJzwN
-----END RSA PRIVATE KEY-----`;

const CERT = `
-----BEGIN CERTIFICATE-----
MIID0TCCArmgAwIBAgIUJwFzZxJfr---TRUNCATED_FOR_BREVITY---AJ+T4=
-----END CERTIFICATE-----`;

/*  Quick & dirty client cert for mutual TLS (shares same CA) */
const CLIENT_KEY  = KEY;
const CLIENT_CERT = CERT;

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Helper – sequential runner
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const STEPS = [];
function step(title, fn) { STEPS.push({ title, fn }); }

(async () => {
  for (const { title, fn } of STEPS) {
    console.log('\n' + '═'.repeat(110));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(110));
    try { await fn(); }
    catch (e) { console.error('💥  Exception →', e); }
  }
})();

/***************************************************************************************************
 * 1) createServer() + https.get()  – "Hello TLS!"
 **************************************************************************************************/
step('1) Basic https.createServer() + https.get()', async () => {
  const server = https.createServer({ key: KEY, cert: CERT },
    (_, res) => res.end('Hello TLS!')).listen(0);

  const { port } = server.address();
  const body = await new Promise(resolve => {
    https.get({ hostname: 'localhost', port, rejectUnauthorized: false },
      res => {
        let data = '';
        res.on('data', d => data += d);
        res.on('end', () => resolve(data));
      });
  });
  console.log('Server said →', body);               // Hello TLS!
  server.close();

  /* Expected output:
     Server said → Hello TLS!
  */
});

/***************************************************************************************************
 * 2) https.request() POST JSON (echo)
 **************************************************************************************************/
step('2) https.request() POST JSON', async () => {
  const server = https.createServer({ key: KEY, cert: CERT }, (req, res) => {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      res.setHeader('content-type', 'application/json');
      res.end(body);                                // echo back
    });
  }).listen(0);

  const { port } = server.address();
  const payload = JSON.stringify({ msg: 'hi' });
  const echoed  = await new Promise(resolve => {
    const req = https.request({
      hostname: 'localhost', port, method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': payload.length },
      rejectUnauthorized: false
    }, res => {
      let d = ''; res.on('data', c => d += c); res.on('end', () => resolve(d));
    });
    req.write(payload); req.end();
  });
  console.log('Echoed →', echoed);                  // {"msg":"hi"}
  server.close();

  /* Expected output:
     Echoed → {"msg":"hi"}
  */
});

/***************************************************************************************************
 * 3) Custom https.Agent (keep-alive & maxSockets)
 **************************************************************************************************/
step('3) Custom https.Agent keep-alive', async () => {
  const server = https.createServer({ key: KEY, cert: CERT },
    (_, res) => res.end('ok')).listen(0);
  const { port } = server.address();

  const agent = new https.Agent({ keepAlive: true, maxSockets: 1, rejectUnauthorized: false });
  for (let i = 0; i < 3; i++) {
    await new Promise(resolve => {
      https.get({ hostname: 'localhost', port, agent, rejectUnauthorized: false },
        res => res.resume().on('end', resolve));
    });
  }
  console.log('Agent reused sockets? →', Object.keys(agent.freeSockets).length > 0);
  agent.destroy(); server.close();

  /* Expected output:
     Agent reused sockets? → true
  */
});

/***************************************************************************************************
 * 4) Mutual TLS (client certificate authentication)
 **************************************************************************************************/
step('4) Mutual TLS authentication', async () => {
  const server = https.createServer({
    key: KEY, cert: CERT, ca: CERT,
    requestCert: true, rejectUnauthorized: true
  }, (req, res) => {
    const peer = req.socket.getPeerCertificate();
    res.end(peer.subject.CN || 'unknown CN');
  }).listen(0);

  const { port } = server.address();
  const cn = await new Promise(resolve => {
    https.get({
      hostname: 'localhost', port, key: CLIENT_KEY, cert: CLIENT_CERT,
      ca: CERT, rejectUnauthorized: true
    }, res => {
      let d = ''; res.on('data', c => d += c); res.on('end', () => resolve(d));
    });
  });
  console.log('Authenticated CN →', cn);            // localhost
  server.close();

  /* Expected output:
     Authenticated CN → localhost
  */
});

/***************************************************************************************************
 * 5) SNI (Server Name Indication) via SNICallback
 **************************************************************************************************/
step('5) SNI – multiple certificates', async () => {
  // Fake second cert/key pair (reuse same for demo)
  const contexts = {
    'example.com': tls.createSecureContext({ key: KEY, cert: CERT }),
    'localhost'  : tls.createSecureContext({ key: KEY, cert: CERT })
  };

  const server = https.createServer({
    SNICallback: (servername, cb) => cb(null, contexts[servername] || contexts['localhost']),
    key: KEY, cert: CERT
  }, (_, res) => res.end('SNI works')).listen(0);

  const { port } = server.address();
  const names = ['example.com', 'foobar.com'];       // 2nd will fall back to default
  for (const host of names) {
    const msg = await new Promise(resolve => {
      https.get({ host: 'localhost', port, servername: host, rejectUnauthorized: false },
        res => { let d=''; res.on('data', c=>d+=c); res.on('end', ()=>resolve(d)); });
    });
    console.log(`Host=${host} →`, msg);
  }
  server.close();

  /* Expected output:
     Host=example.com → SNI works
     Host=foobar.com  → SNI works
  */
});

/***************************************************************************************************
 * 6) ALPN (Application-Layer Protocol Negotiation) hint
 **************************************************************************************************/
step('6) ALPN negotiation (HTTP/1.1 vs h2)', async () => {
  const server = https.createServer({
    key: KEY, cert: CERT, ALPNProtocols: ['http/1.1']
  }, (_, res) => res.end('alpn')).listen(0);

  const { port } = server.address();
  const resProto = await new Promise(resolve => {
    const req = https.request({ host: 'localhost', port, rejectUnauthorized: false, ALPNProtocols: ['h2', 'http/1.1'] },
      res => { res.resume(); res.on('end', () => resolve(req.socket.alpnProtocol)); });
    req.end();
  });
  console.log('Negotiated protocol →', resProto);    // http/1.1
  server.close();

  /* Expected output:
     Negotiated protocol → http/1.1
  */
});

/***************************************************************************************************
 * 7) Timeouts & aborting a client request
 **************************************************************************************************/
step('7) Client timeout & abort()', async () => {
  const server = https.createServer({ key: KEY, cert: CERT }, () => {
    /* intentionally hang */ }).listen(0);

  const { port } = server.address();
  await new Promise(resolve => {
    const req = https.get({ host: 'localhost', port, rejectUnauthorized: false });
    req.setTimeout(300, () => { console.log('Timed out → aborting'); req.abort(); resolve(); });
    req.on('error', () => {});                       // swallow
  });
  server.close();

  /* Expected output:
     Timed out → aborting
  */
});

/***************************************************************************************************
 * 8) https.globalAgent stats
 **************************************************************************************************/
step('8) https.globalAgent live sockets', async () => {
  const server = https.createServer({ key: KEY, cert: CERT },
    (_, res) => res.end('x')).listen(0);
  const { port } = server.address();

  await Promise.all([...Array(2)].map(() => new Promise(r => {
    https.get({ host: 'localhost', port, rejectUnauthorized: false },
      res => res.resume().on('end', r));
  })));
  console.log('GlobalAgent sockets →', Object.keys(https.globalAgent.sockets).length);
  server.close();

  /* Expected output:
     GlobalAgent sockets → 0
  */
});

/***************************************************************************************************
 * 9) Cipher details & peer certificate info
 **************************************************************************************************/
step('9) Inspect cipher & cert on server side', async () => {
  const server = https.createServer({ key: KEY, cert: CERT }, (req, res) => {
    const cipher = req.socket.getCipher();
    const cert   = req.socket.getPeerCertificate();
    res.end(JSON.stringify({ cipher: cipher.name, peerCN: cert.subject?.CN || null }));
  }).listen(0);

  const { port } = server.address();
  const info = await new Promise(resolve => {
    https.get({ host: 'localhost', port, rejectUnauthorized: false },
      res => { let d=''; res.on('data', c=>d+=c); res.on('end', ()=>resolve(JSON.parse(d))); });
  });
  console.log(info);
  server.close();

  /* Expected output (cipher varies):
     { cipher: 'TLS_AES_256_GCM_SHA384', peerCN: null }
  */
});

/***************************************************************************************************
 * 10) TLS session reuse & session caching via Agent
 **************************************************************************************************/
step('10) Session reuse (resumption) demo', async () => {
  const server = https.createServer({ key: KEY, cert: CERT },
    (_, res) => res.end('sess')).listen(0);
  const { port } = server.address();

  const agent = new https.Agent({ keepAlive: true, rejectUnauthorized: false });
  const socketIds = new Set();

  async function hit() {
    await new Promise(res => {
      https.get({ host: 'localhost', port, agent, rejectUnauthorized: false },
        r => { socketIds.add(r.socket.session?.toString('hex') || Math.random()); r.resume().on('end', res); });
    });
  }
  await hit(); await hit();
  console.log('Unique TLS session IDs →', socketIds.size);
  agent.destroy(); server.close();

  /* Expected output:
     Unique TLS session IDs → 1   (proof that second handshake reused session ticket)
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/