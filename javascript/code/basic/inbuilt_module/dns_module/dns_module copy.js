/**
 * Node.js 'dns' Module: Comprehensive Usage Examples
 * 
 * The 'dns' module provides name resolution functions for DNS (Domain Name System).
 * This file demonstrates all major and minor methods, properties, and use-cases of the dns module.
 * Each example is self-contained, includes expected output in comments, and covers exceptions.
 * 
 * Author: The Best Coder in the World
 */

const dns = require('dns');

// 1. dns.lookup(hostname[, options], callback)
// Resolves a hostname (e.g., 'nodejs.org') into the first found A or AAAA record.
dns.lookup('nodejs.org', (err, address, family) => {
    if (err) throw err;
    console.log('1. dns.lookup:', address, family); 
    // Expected: 1. dns.lookup: <IP_ADDRESS> 4 or 6
});

// 2. dns.lookup with options {all: true}
// Returns all resolved addresses.
dns.lookup('nodejs.org', { all: true }, (err, addresses) => {
    if (err) throw err;
    console.log('2. dns.lookup all:', addresses); 
    // Expected: 2. dns.lookup all: [ { address: '...', family: 4 }, ... ]
});

// 3. dns.resolve(hostname[, rrtype], callback)
// Resolves a hostname into an array of the specified record types (A, AAAA, MX, TXT, etc.).
dns.resolve('nodejs.org', 'A', (err, addresses) => {
    if (err) throw err;
    console.log('3. dns.resolve A:', addresses); 
    // Expected: 3. dns.resolve A: [ 'IP_ADDRESS1', 'IP_ADDRESS2', ... ]
});

// 4. dns.resolve4(hostname[, options], callback)
// Resolves IPv4 addresses for a hostname.
dns.resolve4('nodejs.org', (err, addresses) => {
    if (err) throw err;
    console.log('4. dns.resolve4:', addresses); 
    // Expected: 4. dns.resolve4: [ 'IP_ADDRESS1', ... ]
});

// 5. dns.resolve6(hostname[, options], callback)
// Resolves IPv6 addresses for a hostname.
dns.resolve6('nodejs.org', (err, addresses) => {
    if (err) throw err;
    console.log('5. dns.resolve6:', addresses); 
    // Expected: 5. dns.resolve6: [ 'IPv6_ADDRESS1', ... ]
});

// 6. dns.resolveMx(hostname, callback)
// Resolves mail exchange records for a hostname.
dns.resolveMx('gmail.com', (err, addresses) => {
    if (err) throw err;
    console.log('6. dns.resolveMx:', addresses); 
    // Expected: 6. dns.resolveMx: [ { exchange: '...', priority: ... }, ... ]
});

// 7. dns.resolveTxt(hostname, callback)
// Resolves text records for a hostname.
dns.resolveTxt('google.com', (err, records) => {
    if (err) throw err;
    console.log('7. dns.resolveTxt:', records); 
    // Expected: 7. dns.resolveTxt: [ [ 'v=spf1 ...' ], ... ]
});

// 8. dns.reverse(ip, callback)
// Reverse resolves an IP address to hostnames.
dns.reverse('8.8.8.8', (err, hostnames) => {
    if (err) throw err;
    console.log('8. dns.reverse:', hostnames); 
    // Expected: 8. dns.reverse: [ 'dns.google' ]
});

// 9. dns.resolveSrv(hostname, callback)
// Resolves SRV records for a hostname.
dns.resolveSrv('_sip._tcp.google.com', (err, addresses) => {
    if (err) {
        console.log('9. dns.resolveSrv: No SRV records or error:', err.code); 
        // Expected: 9. dns.resolveSrv: No SRV records or error: ENODATA or similar
    } else {
        console.log('9. dns.resolveSrv:', addresses); 
        // Expected: 9. dns.resolveSrv: [ { priority: ..., weight: ..., port: ..., name: '...' }, ... ]
    }
});

// 10. dns.resolveSoa(hostname, callback)
// Resolves SOA (Start of Authority) record for a hostname.
dns.resolveSoa('google.com', (err, record) => {
    if (err) throw err;
    console.log('10. dns.resolveSoa:', record); 
    // Expected: 10. dns.resolveSoa: { nsname: '...', hostmaster: '...', serial: ..., ... }
});

// 11. dns.resolveNs(hostname, callback)
// Resolves name server records for a hostname.
dns.resolveNs('nodejs.org', (err, addresses) => {
    if (err) throw err;
    console.log('11. dns.resolveNs:', addresses); 
    // Expected: 11. dns.resolveNs: [ 'ns1.pXX.dynect.net', ... ]
});

// 12. dns.resolveCname(hostname, callback)
// Resolves canonical name records for a hostname.
dns.resolveCname('www.google.com', (err, addresses) => {
    if (err) {
        console.log('12. dns.resolveCname: No CNAME or error:', err.code); 
        // Expected: 12. dns.resolveCname: No CNAME or error: ENODATA or similar
    } else {
        console.log('12. dns.resolveCname:', addresses); 
        // Expected: 12. dns.resolveCname: [ '...' ]
    }
});

// 13. dns.resolvePtr(hostname, callback)
// Resolves PTR records (mainly for reverse DNS).
dns.resolvePtr('8.8.8.8.in-addr.arpa', (err, addresses) => {
    if (err) {
        console.log('13. dns.resolvePtr: No PTR or error:', err.code); 
        // Expected: 13. dns.resolvePtr: No PTR or error: ENOTFOUND or similar
    } else {
        console.log('13. dns.resolvePtr:', addresses); 
        // Expected: 13. dns.resolvePtr: [ '...' ]
    }
});

// 14. dns.resolveNaptr(hostname, callback)
// Resolves NAPTR records.
dns.resolveNaptr('sip2sip.info', (err, records) => {
    if (err) {
        console.log('14. dns.resolveNaptr: No NAPTR or error:', err.code); 
        // Expected: 14. dns.resolveNaptr: No NAPTR or error: ENODATA or similar
    } else {
        console.log('14. dns.resolveNaptr:', records); 
        // Expected: 14. dns.resolveNaptr: [ { flags: '...', service: '...', ... }, ... ]
    }
});

// 15. dns.getServers()
// Returns an array of IP addresses as strings that are being used for name resolution.
console.log('15. dns.getServers:', dns.getServers()); 
// Expected: 15. dns.getServers: [ '8.8.8.8', '8.8.4.4', ... ] (system dependent)

// 16. dns.setServers(servers)
// Sets the IP addresses of the servers to be used for DNS resolution.
dns.setServers(['8.8.8.8', '8.8.4.4']);
console.log('16. dns.setServers: Set to Google DNS'); 
// Expected: 16. dns.setServers: Set to Google DNS

// 17. dns.promises API: Using async/await with dns.promises
(async () => {
    try {
        const addresses = await dns.promises.resolve4('nodejs.org');
        console.log('17. dns.promises.resolve4:', addresses); 
        // Expected: 17. dns.promises.resolve4: [ 'IP_ADDRESS1', ... ]
    } catch (err) {
        console.error('17. dns.promises.resolve4 error:', err);
    }
})();

// 18. Exception Handling: Invalid Hostname
dns.lookup('invalid.hostname.example', (err, address, family) => {
    if (err) {
        console.log('18. Exception: dns.lookup error:', err.code); 
        // Expected: 18. Exception: dns.lookup error: ENOTFOUND
    }
});

// 19. dns.lookupService(address, port, callback)
// Resolves an IP address and port to a hostname and service.
dns.lookupService('8.8.8.8', 53, (err, hostname, service) => {
    if (err) throw err;
    console.log('19. dns.lookupService:', hostname, service); 
    // Expected: 19. dns.lookupService: <hostname> domain
});

// 20. dns.Resolver class: Custom DNS resolver instance
const resolver = new dns.Resolver();
resolver.setServers(['1.1.1.1']);
resolver.resolve4('nodejs.org', (err, addresses) => {
    if (err) throw err;
    console.log('20. dns.Resolver.resolve4:', addresses); 
    // Expected: 20. dns.Resolver.resolve4: [ 'IP_ADDRESS1', ... ]
});

/**
 * Summary:
 * - Covered: All major and minor methods of the dns module, including lookup, resolve, reverse, get/setServers, promises API, Resolver class, and error handling.
 * - Each example is self-contained and demonstrates a unique aspect of the dns module.
 * - All expected outputs are provided in comments for clarity.
 */