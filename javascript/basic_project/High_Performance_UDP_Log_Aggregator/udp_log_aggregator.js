// udp_log_aggregator.js

const dgram = require('dgram');
const zlib = require('zlib');
const https = require('https');
const { setTimeout, clearTimeout } = require('timers');
const crypto = require('crypto');
const EventEmitter = require('events');

// Configuration
const CONFIG = {
    UDP_PORT: 5140,
    UDP_HOST: '0.0.0.0',
    BATCH_MAX_SIZE: 1000,
    BATCH_MAX_MS: 100,
    DEDUP_CACHE_SIZE: 100000,
    STORAGE_ENDPOINT: {
        hostname: 's3.example.com',
        port: 443,
        path: '/logs/upload',
        method: 'POST',
        headers: {
            'Content-Type': 'application/gzip'
        }
    },
    MAX_CONCURRENT_UPLOADS: 4,
    UPLOAD_RETRY_MS: 2000,
    UPLOAD_MAX_RETRIES: 10
};

// Deduplication cache using a fixed-size LRU
class LRUCache {
    constructor(limit) {
        this.limit = limit;
        this.map = new Map();
    }
    has(key) {
        if (!this.map.has(key)) return false;
        // Move to end (most recently used)
        const value = this.map.get(key);
        this.map.delete(key);
        this.map.set(key, value);
        return true;
    }
    set(key, value) {
        if (this.map.has(key)) this.map.delete(key);
        this.map.set(key, value);
        if (this.map.size > this.limit) {
            // Remove least recently used
            const firstKey = this.map.keys().next().value;
            this.map.delete(firstKey);
        }
    }
}

// Event emitter for process monitoring
class AggregatorEvents extends EventEmitter {}
const events = new AggregatorEvents();

// Batch manager
class BatchManager {
    constructor(onBatchReady) {
        this.entries = [];
        this.timer = null;
        this.onBatchReady = onBatchReady;
        this.lock = false;
    }
    add(entry) {
        this.entries.push(entry);
        if (this.entries.length === 1) {
            this.timer = setTimeout(() => this.flush('timeout'), CONFIG.BATCH_MAX_MS);
        }
        if (this.entries.length >= CONFIG.BATCH_MAX_SIZE) {
            this.flush('size');
        }
    }
    flush(reason) {
        if (this.lock) return;
        this.lock = true;
        if (this.timer) clearTimeout(this.timer);
        if (this.entries.length === 0) {
            this.lock = false;
            return;
        }
        const batch = this.entries;
        this.entries = [];
        this.timer = null;
        events.emit('batch_flush', { count: batch.length, reason });
        this.onBatchReady(batch);
        this.lock = false;
    }
}

// Upload queue with back-pressure
class UploadQueue {
    constructor() {
        this.queue = [];
        this.active = 0;
    }
    enqueue(batch, attempt = 0) {
        this.queue.push({ batch, attempt });
        this.process();
    }
    process() {
        while (this.active < CONFIG.MAX_CONCURRENT_UPLOADS && this.queue.length > 0) {
            const { batch, attempt } = this.queue.shift();
            this.active++;
            this.upload(batch, attempt)
                .then(() => {
                    this.active--;
                    events.emit('upload_success', { size: batch.length });
                    this.process();
                })
                .catch((err) => {
                    this.active--;
                    if (attempt < CONFIG.UPLOAD_MAX_RETRIES) {
                        events.emit('upload_retry', { error: err.message, attempt: attempt + 1 });
                        setTimeout(() => this.enqueue(batch, attempt + 1), CONFIG.UPLOAD_RETRY_MS);
                    } else {
                        events.emit('upload_failed', { error: err.message });
                    }
                    this.process();
                });
        }
    }
    upload(batch, attempt) {
        return new Promise((resolve, reject) => {
            const json = Buffer.from(JSON.stringify(batch));
            zlib.gzip(json, (err, compressed) => {
                if (err) return reject(err);
                const req = https.request(CONFIG.STORAGE_ENDPOINT, (res) => {
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        res.resume();
                        resolve();
                    } else {
                        res.resume();
                        reject(new Error(`HTTP ${res.statusCode}`));
                    }
                });
                req.on('error', reject);
                req.write(compressed);
                req.end();
            });
        });
    }
}

// Main aggregator
class UDPAggregator {
    constructor() {
        this.dedup = new LRUCache(CONFIG.DEDUP_CACHE_SIZE);
        this.batchManager = new BatchManager(this.handleBatch.bind(this));
        this.uploadQueue = new UploadQueue();
        this.server = dgram.createSocket('udp4');
        this.setupServer();
        this.setupEvents();
    }
    setupServer() {
        this.server.on('message', (msg, rinfo) => {
            let log;
            try {
                log = JSON.parse(msg.toString());
            } catch (e) {
                events.emit('parse_error', { error: e.message, from: rinfo.address });
                return;
            }
            if (!log.id) {
                // If no ID, generate a hash as fallback
                log.id = crypto.createHash('sha256').update(msg).digest('hex');
            }
            if (this.dedup.has(log.id)) {
                events.emit('dedup_discard', { id: log.id });
                return;
            }
            this.dedup.set(log.id, true);
            this.batchManager.add(log);
            events.emit('log_received', { id: log.id, from: rinfo.address });
        });
        this.server.on('error', (err) => {
            events.emit('udp_error', { error: err.message });
        });
        this.server.bind(CONFIG.UDP_PORT, CONFIG.UDP_HOST, () => {
            events.emit('udp_listening', { port: CONFIG.UDP_PORT, host: CONFIG.UDP_HOST });
        });
    }
    handleBatch(batch) {
        this.uploadQueue.enqueue(batch);
    }
    setupEvents() {
        // Developers can listen to these events for monitoring
        events.on('udp_listening', (info) => {
            console.log(`[INFO] UDP server listening on ${info.host}:${info.port}`);
        });
        events.on('log_received', (info) => {
            // Optionally log: console.log(`[RECV] Log ${info.id} from ${info.from}`);
        });
        events.on('dedup_discard', (info) => {
            // Optionally log: console.log(`[DEDUP] Discarded duplicate log ${info.id}`);
        });
        events.on('parse_error', (info) => {
            console.error(`[ERROR] Failed to parse log from ${info.from}: ${info.error}`);
        });
        events.on('batch_flush', (info) => {
            console.log(`[BATCH] Flushed ${info.count} logs due to ${info.reason}`);
        });
        events.on('upload_success', (info) => {
            console.log(`[UPLOAD] Successfully uploaded batch of ${info.size} logs`);
        });
        events.on('upload_retry', (info) => {
            console.warn(`[RETRY] Upload failed (${info.error}), retrying (attempt ${info.attempt})`);
        });
        events.on('upload_failed', (info) => {
            console.error(`[FAILED] Upload failed after max retries: ${info.error}`);
        });
        events.on('udp_error', (info) => {
            console.error(`[UDP ERROR] ${info.error}`);
        });
    }
}

// Start aggregator
new UDPAggregator();