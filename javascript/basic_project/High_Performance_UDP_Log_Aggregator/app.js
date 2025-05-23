/******************************************************************************************
 * High-Performance UDP Log Aggregator
 * Author  : 
 * Runtime : Node.js ≥ 16
 * Modules : dgram, zlib, https, timers, crypto, events
 *
 * FEATURES
 *  • Listens for JSON log records over UDP
 *  • De-duplicates records via configurable TTL cache
 *  • Batches logs (≤100 ms OR ≤1000 records)
 *  • Gzip-compresses each batch
 *  • Asynchronously uploads over HTTPS with bounded concurrency
 *  • Back-pressure: pauses UDP intake when upload queue is saturated
 *  • Emits detailed runtime metrics for observability
 ******************************************************************************************/

/* -------------------------------- CONFIGURATION --------------------------------------- */
const CONFIG = Object.freeze({
    UDP_PORT              : 5140,                  // Listening port
    UDP_HOST              : '0.0.0.0',            // Bind address
    BATCH_MAX_MS          : 100,                  // Flush interval (ms)
    BATCH_MAX_COUNT       : 1000,                 // Flush size
    DEDUP_TTL_MS          : 60_000,               // De-duplication window
    UPLOAD_ENDPOINT       : 'https://logs.example.com/upload',
    UPLOAD_CONCURRENCY    : 4,                    // Parallel HTTPS uploads
    BACKPRESSURE_MAX_QUEUE: 100,                  // Pause intake when queued uploads exceed
    METRICS_INTERVAL_MS   : 5_000                 // Console metrics frequency
});

/* -------------------------------- DEPENDENCIES --------------------------------------- */
const dgram  = require('dgram');
const zlib   = require('zlib');
const https  = require('https');
const timers = require('timers');
const crypto = require('crypto');
const {EventEmitter} = require('events');

/* -------------------------------- HELPER UTILITIES ----------------------------------- */
function now() { return Date.now(); }

function safeJsonParse(buf) {
    try { return JSON.parse(buf); }
    catch { return null; }
}

/* --------------------------- DEDUPLICATION CACHE ------------------------------------- */
class DedupCache {
    constructor(ttl) {
        this.ttl  = ttl;
        this.store = new Map();            // id -> timestamp
        // periodic clean-up
        timers.setInterval(() => this._sweep(), ttl);
    }

    /**
     * @param {string} id – Unique log identifier
     * @returns {boolean} true if fresh (not duplicate)
     */
    checkAndInsert(id) {
        const ts = now();
        if (this.store.has(id)) return false;
        this.store.set(id, ts);
        return true;
    }

    _sweep() {
        const expiry = now() - this.ttl;
        for (const [id, ts] of this.store) if (ts < expiry) this.store.delete(id);
    }
}

/* --------------------------- BATCH MANAGER ------------------------------------------- */
class BatchManager extends EventEmitter {
    constructor(maxMs, maxCount) {
        super();
        this.maxMs    = maxMs;
        this.maxCount = maxCount;
        this.buffer   = [];
        this.timer    = null;
    }

    add(record) {
        this.buffer.push(record);
        if (!this.timer) this.timer = timers.setTimeout(() => this.flush(), this.maxMs);
        if (this.buffer.length >= this.maxCount) this.flush();
    }

    flush() {
        if (this.timer) { timers.clearTimeout(this.timer); this.timer = null; }
        if (this.buffer.length === 0) return;
        const batch = this.buffer;
        this.buffer = [];
        this.emit('batch', batch);
    }
}

/* --------------------------- UPLOADER WITH BACKPRESSURE ------------------------------ */
class Uploader extends EventEmitter {
    constructor(endpoint, concurrency, maxQueue) {
        super();
        this.endpoint     = new URL(endpoint);
        this.concurrency  = concurrency;
        this.maxQueue     = maxQueue;
        this.inFlight     = 0;
        this.queue        = [];
    }

    enqueue(payload) {
        this.queue.push(payload);
        this._drain();
        // emit back-pressure signals
        if (this.queue.length >= this.maxQueue) this.emit('pressure', true);
    }

    _drain() {
        while (this.inFlight < this.concurrency && this.queue.length) {
            const payload = this.queue.shift();
            this.inFlight++;
            this._upload(payload).finally(() => {
                this.inFlight--;
                if (this.queue.length < this.maxQueue) this.emit('pressure', false);
                this._drain();
            });
        }
    }

    _upload(gzBuffer) {
        return new Promise((resolve) => {
            const options = {
                hostname: this.endpoint.hostname,
                port    : this.endpoint.port || 443,
                path    : this.endpoint.pathname,
                method  : 'POST',
                headers : {
                    'Content-Type'   : 'application/gzip',
                    'Content-Length' : gzBuffer.length,
                    'Content-Encoding': 'gzip'
                },
                timeout : 10_000
            };

            const req = https.request(options, (res) => {
                res.on('data', ()=>{});
                res.on('end', ()=>resolve());
            });

            req.on('error', (err)=>{ 
                console.error('Upload error:', err.message);
                // simple retry strategy: re-queue
                this.queue.push(gzBuffer);
                resolve();
            });

            req.write(gzBuffer);
            req.end();
        });
    }
}

/* --------------------------- MAIN AGGREGATOR CLASS ---------------------------------- */
class UDPAggregator {
    constructor(cfg = CONFIG) {
        this.cfg       = cfg;
        this.dedup     = new DedupCache(cfg.DEDUP_TTL_MS);
        this.batcher   = new BatchManager(cfg.BATCH_MAX_MS, cfg.BATCH_MAX_COUNT);
        this.uploader  = new Uploader(cfg.UPLOAD_ENDPOINT,
                                      cfg.UPLOAD_CONCURRENCY,
                                      cfg.BACKPRESSURE_MAX_QUEUE);

        this.socket    = dgram.createSocket('udp4');
        this.metrics   = {
            received: 0, duplicated: 0, batched: 0,
            uploaded: 0, paused: false
        };

        /* Wire events */
        this.batcher.on('batch', (records) => this._handleBatch(records));
        this.uploader.on('pressure', (isHigh) => this._handlePressure(isHigh));

        this._startMetricsTicker();
    }

    start() {
        this.socket.on('message', (msg) => this._onMessage(msg));
        this.socket.on('error', (err) => console.error('Socket error:', err));
        this.socket.bind(this.cfg.UDP_PORT, this.cfg.UDP_HOST, () =>
            console.log(`UDP listening on ${this.cfg.UDP_HOST}:${this.cfg.UDP_PORT}`));
    }

    /* ------------------------- private methods -------------------------------------- */
    _onMessage(buf) {
        this.metrics.received++;
        const json = safeJsonParse(buf);
        if (!json || !json.id) return;      // malformed
        if (!this.dedup.checkAndInsert(json.id)) { this.metrics.duplicated++; return; }
        this.batcher.add(json);
    }

    _handleBatch(records) {
        this.metrics.batched += records.length;
        zlib.gzip(Buffer.from(JSON.stringify(records)), (err, gz) => {
            if (err) return console.error('Gzip error:', err);
            this.uploader.enqueue(gz);
            this.metrics.uploaded += records.length;
        });
    }

    _handlePressure(isHigh) {
        if (isHigh && !this.metrics.paused) {
            this.socket.pause();
            this.metrics.paused = true;
            console.warn('Intake paused due to back-pressure');
        } else if (!isHigh && this.metrics.paused) {
            this.socket.resume();
            this.metrics.paused = false;
            console.info('Intake resumed');
        }
    }

    _startMetricsTicker() {
        timers.setInterval(() => {
            console.table({
                Received  : this.metrics.received,
                Duplicated: this.metrics.duplicated,
                Batched   : this.metrics.batched,
                Uploaded  : this.metrics.uploaded,
                QueueSize : this.uploader.queue.length,
                InFlight  : this.uploader.inFlight,
                Paused    : this.metrics.paused
            });
        }, this.cfg.METRICS_INTERVAL_MS);
    }
}

/* --------------------------------- ENTRY POINT --------------------------------------- */
if (require.main === module) {
    const aggregator = new UDPAggregator();
    aggregator.start();
}