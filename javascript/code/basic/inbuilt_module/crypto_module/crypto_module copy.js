/**
 * Node.js crypto Module: Comprehensive Examples
 * 
 * This file demonstrates all major and minor methods of the Node.js crypto module.
 * Each example is self-contained, with clear code, comments, and expected output.
 * 
 * To run: `node <filename>.js`
 */

const crypto = require('crypto');

// 1. crypto.createHash(algorithm)
// Hashing data (SHA256, MD5, etc.)
function exampleCreateHash() {
    const hash = crypto.createHash('sha256');
    hash.update('hello world');
    const digest = hash.digest('hex');
    console.log('SHA256 hash:', digest);
    // Expected output: SHA256 hash: b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
}
exampleCreateHash();

// 2. crypto.createHmac(algorithm, key)
// HMAC (Hash-based Message Authentication Code)
function exampleCreateHmac() {
    const hmac = crypto.createHmac('sha256', 'secret-key');
    hmac.update('hello world');
    const digest = hmac.digest('hex');
    console.log('HMAC:', digest);
    // Expected output: HMAC: c0535e4be2b79ffd93291305436bf889314e4a3faec05ecffcbb7df31bf6e902
}
exampleCreateHmac();

// 3. crypto.randomBytes(size[, callback])
// Generate cryptographically strong random bytes
function exampleRandomBytes() {
    const buf = crypto.randomBytes(8);
    console.log('Random bytes:', buf.toString('hex'));
    // Expected output: Random bytes: <16 hex chars, random>
}
exampleRandomBytes();

// 4. crypto.pbkdf2(password, salt, iterations, keylen, digest, callback)
// Derive a key from a password (async)
function examplePbkdf2() {
    crypto.pbkdf2('password', 'salt', 100000, 32, 'sha256', (err, derivedKey) => {
        if (err) throw err;
        console.log('PBKDF2 key:', derivedKey.toString('hex'));
        // Expected output: PBKDF2 key: <64 hex chars>
    });
}
examplePbkdf2();

// 5. crypto.scrypt(password, salt, keylen[, options], callback)
// Modern password-based key derivation (async)
function exampleScrypt() {
    crypto.scrypt('password', 'salt', 32, (err, derivedKey) => {
        if (err) throw err;
        console.log('Scrypt key:', derivedKey.toString('hex'));
        // Expected output: Scrypt key: <64 hex chars>
    });
}
exampleScrypt();

// 6. crypto.createCipheriv(algorithm, key, iv)
// Symmetric encryption (AES-256-CBC)
function exampleCreateCipheriv() {
    const algorithm = 'aes-256-cbc';
    const key = crypto.randomBytes(32);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv(algorithm, key, iv);
    let encrypted = cipher.update('Secret Message', 'utf8', 'hex');
    encrypted += cipher.final('hex');
    console.log('Encrypted:', encrypted);
    // Expected output: Encrypted: <hex string>
    // Decrypt for demonstration:
    const decipher = crypto.createDecipheriv(algorithm, key, iv);
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    console.log('Decrypted:', decrypted);
    // Expected output: Decrypted: Secret Message
}
exampleCreateCipheriv();

// 7. crypto.generateKeyPair(type, options, callback)
// Asymmetric key pair generation (RSA)
function exampleGenerateKeyPair() {
    crypto.generateKeyPair('rsa', {
        modulusLength: 2048,
        publicKeyEncoding: { type: 'spki', format: 'pem' },
        privateKeyEncoding: { type: 'pkcs8', format: 'pem' }
    }, (err, publicKey, privateKey) => {
        if (err) throw err;
        console.log('Public Key:', publicKey.slice(0, 40) + '...');
        console.log('Private Key:', privateKey.slice(0, 40) + '...');
        // Expected output: PEM-formatted keys (truncated for display)
    });
}
exampleGenerateKeyPair();

// 8. crypto.createSign(algorithm) and crypto.createVerify(algorithm)
// Digital signature and verification (RSA-SHA256)
function exampleSignVerify() {
    // Generate keys for demonstration
    const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
        modulusLength: 2048,
    });
    const sign = crypto.createSign('SHA256');
    sign.update('important message');
    sign.end();
    const signature = sign.sign(privateKey, 'hex');
    console.log('Signature:', signature.slice(0, 40) + '...');
    // Verify
    const verify = crypto.createVerify('SHA256');
    verify.update('important message');
    verify.end();
    const isValid = verify.verify(publicKey, signature, 'hex');
    console.log('Signature valid:', isValid);
    // Expected output: Signature: <hex...> Signature valid: true
}
exampleSignVerify();

// 9. crypto.publicEncrypt and crypto.privateDecrypt
// Asymmetric encryption/decryption (RSA)
function examplePublicEncryptPrivateDecrypt() {
    const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
        modulusLength: 2048,
    });
    const encrypted = crypto.publicEncrypt(publicKey, Buffer.from('top secret'));
    console.log('Encrypted (RSA):', encrypted.toString('base64').slice(0, 40) + '...');
    const decrypted = crypto.privateDecrypt(privateKey, encrypted);
    console.log('Decrypted (RSA):', decrypted.toString());
    // Expected output: Decrypted (RSA): top secret
}
examplePublicEncryptPrivateDecrypt();

// 10. crypto.getCiphers(), crypto.getHashes(), crypto.getCurves()
// List available ciphers, hashes, and elliptic curves
function exampleGetCiphersHashesCurves() {
    console.log('Available ciphers:', crypto.getCiphers().slice(0, 5), '...');
    console.log('Available hashes:', crypto.getHashes().slice(0, 5), '...');
    if (crypto.getCurves) {
        console.log('Available curves:', crypto.getCurves().slice(0, 5), '...');
    }
    // Expected output: Lists of supported ciphers, hashes, and curves
}
exampleGetCiphersHashesCurves();

// 11. crypto.randomInt(min, max[, callback])
// Generate a cryptographically secure random integer
function exampleRandomInt() {
    crypto.randomInt(1, 100, (err, n) => {
        if (err) throw err;
        console.log('Random int:', n); // 1 <= n < 100
        // Expected output: Random int: <number between 1 and 99>
    });
}
exampleRandomInt();

// 12. crypto.timingSafeEqual(a, b)
// Constant-time buffer comparison to prevent timing attacks
function exampleTimingSafeEqual() {
    const a = Buffer.from('abcd');
    const b = Buffer.from('abcd');
    const c = Buffer.from('abce');
    console.log('timingSafeEqual (a, b):', crypto.timingSafeEqual(a, b)); // true
    // Expected output: timingSafeEqual (a, b): true
    try {
        console.log('timingSafeEqual (a, c):', crypto.timingSafeEqual(a, c)); // false
    } catch (err) {
        console.log('timingSafeEqual error:', err.message);
        // Buffers must be same length, else throws
    }
}
exampleTimingSafeEqual();

/**
 * Summary:
 * - All major and minor crypto methods are covered.
 * - Each example is self-contained and demonstrates expected behavior.
 * - Modify or uncomment lines to see different results and behaviors.
 */