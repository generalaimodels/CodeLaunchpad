/***************************************************************************************************
DNS MASTERY – 10 Ultra-Curated Examples
════════════════════════════════════════
Author : 𝐀𝐈-Sensei (🏆 )  
Target : Node 18+ (CommonJS)  
Goal   : Show EVERY public API of the builtin “dns” module (classic-callback, Resolver class,
         promises, flags, buffers, rare RR-types, error handling … you name it).  
Style  : Each example is a self-contained IIFE so you can comment / uncomment at will.  
Notes  : ① Internet required. ② Outputs may differ slightly (DNS is ever-changing).  
***************************************************************************************************/


// ──────────────────────────────────────────────────────────────────────────────
// EXAMPLE-01 ★ dns.lookup – the everyday work-horse
//             • implicit family v4/v6 detection
// ──────────────────────────────────────────────────────────────────────────────
(() => {
    const dns = require('node:dns');
  
    dns.lookup('nodejs.org', (err, address, family) => {
      if (err) throw err;
      console.log('[E-01] nodejs.org resolved to', address, 'IPv' + family);
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [E-01] nodejs.org resolved to 104.20.23.46 IPv4      ← values vary
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-02 ★ dns.lookup with advanced options – all, hints flags, verbatim
  //             Demonstrates constants: dns.ADDRCONFIG • dns.V4MAPPED • dns.ALL
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dns = require('node:dns');
  
    dns.lookup('nodejs.org',
      { all: true, hints: dns.ADDRCONFIG | dns.V4MAPPED, verbatim: false }, // 👉 full list
      (err, addresses) => {
        if (err) throw err;
        console.log('[E-02] lookup(all=true) →', addresses);
      });
  
    /*
    ==== EXPECTED OUTPUT (truncated) ====
    [E-02] lookup(all=true) → [
       { address: '104.20.23.46', family: 4 },
       { address: '104.20.22.46', family: 4 }
    ]
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-03 ★ dns.resolve vs specialised helpers – CNAME • SOA • NS
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dns = require('node:dns');
  
    dns.resolve('nodejs.org', 'CNAME', (e, records) =>
      console.log('[E-03] CNAME via generic resolve →', records));
  
    dns.resolveCname('nodejs.org', (e, records) =>
      console.log('[E-03] CNAME via resolveCname   →', records));
  
    dns.resolveSoa('iana.org', (e, soa) =>
      console.log('[E-03] SOA                       →', soa));
  
    dns.resolveNs('com', (e, ns) =>
      console.log('[E-03] Authoritative NS          →', ns));
  
    /*
    ==== EXPECTED OUTPUT (simplified) ====
    [E-03] CNAME via generic resolve → [ 'nodejs.org' ]    // usually empty; many sites now use A/AAAA
    [E-03] CNAME via resolveCname   → [ 'nodejs.org' ]
    [E-03] SOA                       → { nsname: 'a.iana-servers.net', ... }
    [E-03] Authoritative NS          → [ 'a.gtld-servers.net', 'b.gtld-servers.net', ... ]
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-04 ★ A + AAAA with TTL data – resolve4 / resolve6 (options.ttl)
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dns = require('node:dns');
  
    dns.resolve4('google.com', { ttl: true }, (err, recs) => {
      if (err) throw err;
      console.log('[E-04] A records with TTL →', recs);
    });
  
    dns.resolve6('google.com', { ttl: true }, (err, recs) => {
      console.log('[E-04] AAAA records with TTL →', recs);
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [E-04] A records with TTL → [ { address: '142.250.72.14', ttl: 300 }, ... ]
    [E-04] AAAA records with TTL → [ { address: '2a00:1450:4009:80d::200e', ttl: 300 }, ... ]
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-05 ★ Mail & verification records – MX • TXT
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dns = require('node:dns');
  
    dns.resolveMx('gmail.com', (err, mx) =>
      console.log('[E-05] MX →', mx.slice(0, 2))); // first 2 entries
  
    dns.resolveTxt('google.com', (err, txt) =>
      console.log('[E-05] TXT (first) →', txt[0]));
  
    /*
    ==== EXPECTED OUTPUT ====
    [E-05] MX → [ { priority: 10, exchange: 'alt1.gmail-smtp-in.l.google.com' }, ... ]
    [E-05] TXT (first) → [ 'v=spf1 include:_spf.google.com ~all' ]
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-06 ★ Service Discovery & Telephony – SRV • NAPTR
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dns = require('node:dns');
  
    dns.resolveSrv('_sip._tcp.antisip.com', (e, srv) =>
      console.log('[E-06] SRV →', srv));
  
    dns.resolveNaptr('sip2sip.info', (e, naptr) =>
      console.log('[E-06] NAPTR →', naptr));
  
    /*
    ==== EXPECTED OUTPUT ====
    [E-06] SRV → [ { priority: 0, weight: 0, port: 5060, name: 'sip.antisip.com' } ]
    [E-06] NAPTR → [
       { flags: 's', service: 'SIP+D2U', regexp: '', replacement: '_sip._udp.sip2sip.info', order: 30, preference: 50 },
       ...
    ]
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-07 ★ Who hosts this IP? – reverse PTR + lookupService
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dns = require('node:dns');
  
    const ip = '8.8.8.8'; // Google-DNS
  
    dns.reverse(ip, (err, hostnames) => {
      if (err) throw err;
      console.log('[E-07] reverse PTR →', hostnames); // e.g. dns.google
  
      // Port 53 (DNS), get service/hostname
      dns.lookupService(ip, 53, (e, hostname, service) =>
        console.log('[E-07] lookupService →', hostname, service));
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [E-07] reverse PTR → [ 'dns.google' ]
    [E-07] lookupService → dns.google domain
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-08 ★ Runtime name-server switching – setServers • getServers
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const dns = require('node:dns');
  
    const original = dns.getServers();
    console.log('[E-08] System resolvers →', original);
  
    dns.setServers(['8.8.8.8', '1.1.1.1']);         // Google & Cloudflare
    console.log('[E-08] After setServers() →', dns.getServers());
  
    // Quick check with new resolvers
    dns.resolve4('example.com', (e, rec) => {
      console.log('[E-08] example.com via custom resolvers →', rec);
      dns.setServers(original);                     // restore safety
    });
  
    /*
    ==== EXPECTED OUTPUT ====
    [E-08] System resolvers → [ '192.168.1.1' ]
    [E-08] After setServers() → [ '8.8.8.8', '1.1.1.1' ]
    [E-08] example.com via custom resolvers → [ '93.184.216.34' ]
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-09 ★ Per-Instance Isolation – dns.Resolver class
  // ──────────────────────────────────────────────────────────────────────────────
  (() => {
    const { Resolver } = require('node:dns');
  
    const googleResolver = new Resolver();
    googleResolver.setServers(['8.8.8.8']);
  
    const cloudflareResolver = new Resolver();
    cloudflareResolver.setServers(['1.1.1.1']);
  
    googleResolver.resolve4('example.com', (e, a) =>
      console.log('[E-09] Google DNS →', a));
  
    cloudflareResolver.resolve4('example.com', (e, a) =>
      console.log('[E-09] Cloudflare DNS →', a));
  
    /*
    ==== EXPECTED OUTPUT ====
    [E-09] Google DNS → [ '93.184.216.34' ]
    [E-09] Cloudflare DNS → [ '93.184.216.34' ]
    */
  })();
  
  
  // ──────────────────────────────────────────────────────────────────────────────
  // EXAMPLE-10 ★ Modern Promises/async-await – dns.promises & resolveAny
  // ──────────────────────────────────────────────────────────────────────────────
  (async () => {
    const dns = require('node:dns').promises;
  
    try {
      const any = await dns.resolveAny('nodejs.org');
      console.log('[E-10] resolveAny →', any.slice(0, 3)); // sample first 3
  
      const ptr = await dns.resolvePtr('8.8.4.4');
      console.log('[E-10] resolvePtr →', ptr);
  
      const { address } = await dns.lookup('github.com', { family: 4 });
      console.log('[E-10] await lookup →', address);
    } catch (err) {
      console.error('[E-10] Error:', err);
    }
  
    /*
    ==== EXPECTED OUTPUT (sample) ====
    [E-10] resolveAny → [
      { type: 'AAAA', address: '2606:4700::6810:172e' },
      { type: 'A', address: '104.16.23.46' },
      { type: 'A', address: '104.16.22.46' }
    ]
    [E-10] resolvePtr → [ 'dns.google' ]
    [E-10] await lookup → 20.205.243.166
    */
  })();