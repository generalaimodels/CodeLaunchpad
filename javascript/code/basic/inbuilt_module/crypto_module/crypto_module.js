/***************************************************************************************************
* File  : crypto-ultimate-cheatsheet.js
* Author: 
* Why   : 10 laser-focused, self-contained examples that walk through the **entire** `crypto` API
*         spectrum – from bread-and-butter hashing to rarely-touched ECC curves & WebCrypto.
* Node  : ≥ 18 (for WebCrypto + RSA-OAEP SHA-256)
* Run   : `node crypto-ultimate-cheatsheet.js`
* Style : ES-2023, 2-space indent, strict-mode, zero external deps.
***************************************************************************************************/
'use strict';
const crypto = require('crypto');

/***************************************************************************************************
* Helper ↓ prints section titles uniformly
***************************************************************************************************/
let section = 0;
const title = (txt) => console.log(`\n${'='.repeat(6)}  EX-${++section} ► ${txt}\n`);

/***************************************************************************************************
* EX-1 ► One-shot & streaming hashing
*   API : createHash(), hash.update(), hash.digest(), getHashes()
***************************************************************************************************/
(() => {
  title('Hashing – SHA-256 / MD5');

  console.log(' available algorithms:', crypto.getHashes().slice(0, 5), '…');

  const oneShot = crypto.createHash('sha256').update('hello world').digest('hex');
  console.log(' sha256("hello world") =', oneShot);
  // Expected (constant):
  // sha256("hello world") = b94d27b9934d3e08a52e52d7da7dab...

  // Streaming example (incremental chunks)
  const streamHash = crypto.createHash('md5');
  ['hel', 'lo ', 'wor', 'ld'].forEach(chunk => streamHash.update(chunk));
  console.log(' md5("hello world")   =', streamHash.digest('base64'));
  // Expected: XrY7u+Ae7tCTyyK7j1rNww==
})();

/***************************************************************************************************
* EX-2 ► HMAC & constant-time comparison
*   API : createHmac(), timingSafeEqual(), randomBytes()
***************************************************************************************************/
(() => {
  title('HMAC + timingSafeEqual');

  const key = crypto.randomBytes(32);
  const sign = (msg) => crypto.createHmac('sha256', key).update(msg).digest();

  const macA = sign('payload');
  const macB = sign('payload');
  console.log(' macA.equals(macB)          :', macA.equals(macB));                      // true
  console.log(' timingSafeEqual(macA,macB) :',
    crypto.timingSafeEqual(macA, macB));                                                // true

  const macBad = sign('tampered');
  console.log(' safe compare w/ bad mac     :',
    crypto.timingSafeEqual(macA, macBad)); // throws if lengths differ – they don’t.
  // Expected:
  // first 2 logs true, last log false (values differ)
})();

/***************************************************************************************************
* EX-3 ► Secure randomness helpers
*   API : randomBytes(), randomFillSync(), randomInt(), randomUUID()
***************************************************************************************************/
(() => {
  title('Random data / ints / UUID');

  console.log(' randomBytes(8)   :', crypto.randomBytes(8).toString('hex')); // unpredictable
  const buf = Buffer.alloc(4);
  crypto.randomFillSync(buf);
  console.log(' randomFillSync   :', buf);                                   // random buffer

  console.log(' randomInt(1,10)  :', crypto.randomInt(1, 10));               // 1..9
  console.log(' randomUUID()     :', crypto.randomUUID());                   // RFC-4122
})();

/***************************************************************************************************
* EX-4 ► Password-based key derivation
*   API : pbkdf2Sync(), scryptSync(), pbkdf2()
***************************************************************************************************/
(() => {
  title('PBKDF2 / scrypt');

  const password = 'Tr0ub4dor&3';
  const salt     = crypto.randomBytes(16);

  const key1 = crypto.pbkdf2Sync(password, salt, 100_000, 32, 'sha512');
  console.log(' PBKDF2 key (hex) :', key1.toString('hex').slice(0, 32), '…');

  const key2 = crypto.scryptSync(password, salt, 32);
  console.log(' scrypt  key (hex):', key2.toString('hex').slice(0, 32), '…');

  // async PBKDF2 for completeness
  crypto.pbkdf2(password, salt, 1e5, 32, 'sha512', (err, derived) =>
    console.log(' async PBKDF2 done? err=', !!err, 'len=', derived.length));
})();

/***************************************************************************************************
* EX-5 ► Symmetric encryption (AES-256-CBC) w/ createSecretKey()
*   API : createSecretKey(), createCipheriv(), createDecipheriv()
***************************************************************************************************/
(() => {
  title('AES-256-CBC encryption/decryption');

  const key = crypto.createSecretKey(crypto.randomBytes(32));
  const iv  = crypto.randomBytes(16);

  const plaintext  = 'Confidential ‑ ' + Date.now();
  const cipher     = crypto.createCipheriv('aes-256-cbc', key, iv);
  const ciphertext = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);

  const decipher   = crypto.createDecipheriv('aes-256-cbc', key, iv);
  const decrypted  = Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString();

  console.log(' encrypted (base64):', ciphertext.toString('base64').slice(0, 24), '…');
  console.log(' decrypted          :', decrypted);
  // Expected: decrypted equals original plaintext
})();

/***************************************************************************************************
* EX-6 ► Authenticated encryption (AES-256-GCM)
*   API : cipher.getAuthTag(), decipher.setAuthTag()
***************************************************************************************************/
(() => {
  title('AES-256-GCM (auth tag)');

  const key = crypto.randomBytes(32);
  const iv  = crypto.randomBytes(12); // 96-bit IV recommended for GCM
  const pt  = 'Message w/ integrity';

  const c = crypto.createCipheriv('aes-256-gcm', key, iv);
  const ct = Buffer.concat([c.update(pt, 'utf8'), c.final()]);
  const tag = c.getAuthTag();

  const d = crypto.createDecipheriv('aes-256-gcm', key, iv);
  d.setAuthTag(tag);
  const dt = Buffer.concat([d.update(ct), d.final()]).toString();

  console.log(' auth tag (hex)  :', tag.toString('hex'));
  console.log(' decrypt okay    :', dt === pt); // true
})();

/***************************************************************************************************
* EX-7 ► RSA key-pair, sign / verify
*   API : generateKeyPairSync(), sign(), verify(), createPrivateKey()
***************************************************************************************************/
(() => {
  title('RSA-PSS SHA-256 signing');

  const { privateKey, publicKey } = crypto.generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicExponent: 0x10001,
  });

  const data = Buffer.from('Important document');

  const sig = crypto.sign('sha256', data, privateKey);
  const ok  = crypto.verify('sha256', data, publicKey, sig);

  console.log(' signature len:', sig.length);
  console.log(' verify passed:', ok); // true

  // show import via createPrivateKey / createPublicKey (PEM -> KeyObject)
  const privObj = crypto.createPrivateKey(privateKey.export({ type:'pkcs8', format:'pem' }));
  console.log(' privateKey type:', privObj.type); // 'private'
})();

/***************************************************************************************************
* EX-8 ► RSA-OAEP encryption / decryption
*   API : publicEncrypt(), privateDecrypt(), constants.RSA_PKCS1_OAEP_PADDING
***************************************************************************************************/
(() => {
  title('RSA-OAEP encrypt/decrypt');

  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
  const secret = Buffer.from('🔐 TOP SECRET 🔐');

  const enc = crypto.publicEncrypt(
    { key: publicKey, padding: crypto.constants.RSA_PKCS1_OAEP_PADDING },
    secret
  );

  const dec = crypto.privateDecrypt(
    { key: privateKey, padding: crypto.constants.RSA_PKCS1_OAEP_PADDING },
    enc
  );

  console.log(' ciphertext bytes:', enc.length);
  console.log(' round-trip match:', secret.equals(dec)); // true
})();

/***************************************************************************************************
* EX-9 ► Elliptic-Curve Diffie-Hellman (ECDH) & classic DH groups
*   API : createECDH(), getDiffieHellman()
***************************************************************************************************/
(() => {
  title('ECDH (P-256) + built-in DH group "modp15"');

  // -------- ECDH --------
  const alice = crypto.createECDH('prime256v1');
  alice.generateKeys();
  const bob   = crypto.createECDH('prime256v1');
  bob.generateKeys();

  const secretA = alice.computeSecret(bob.getPublicKey());
  const secretB = bob.computeSecret(alice.getPublicKey());
  console.log(' ECDH secrets equal ? ', secretA.equals(secretB)); // true

  // -------- preset Diffie-Hellman group --------
  const dh1 = crypto.getDiffieHellman('modp15'); // 3072-bit MODP group
  dh1.generateKeys();
  console.log(' DH prime bits     :', dh1.getPrime().length * 8);
})();

/***************************************************************************************************
* EX-10 ► WebCrypto (globalThis.crypto) + subtle.digest()
*   API : webcrypto.subtle, subtle.digest(), subtle.generateKey() (*browser-compatible*)
***************************************************************************************************/
(async () => {
  title('WebCrypto subtle.digest (SHA-384)');

  const text   = new TextEncoder().encode('web-crypto ♥');
  const digest = await crypto.webcrypto.subtle.digest('SHA-384', text);
  console.log(' SHA-384 bytes:', Buffer.from(digest).toString('hex').slice(0, 32), '…');

  // Quick demonstration of generating an AES-GCM key via WebCrypto
  const wcKey = await crypto.webcrypto.subtle.generateKey(
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  );
  console.log(' WebCrypto key type:', wcKey.type); // 'secret'
})();

/***************************************************************************************************
* 🎯  End of file – you now wield virtually every crypto primitive available in Node.js. Experiment,
*     tweak, and hack responsibly! 🚀
***************************************************************************************************/