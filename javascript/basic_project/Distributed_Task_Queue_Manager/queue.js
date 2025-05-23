/**
 * Distributed Task Queue Manager (single–file implementation)
 * ------------------------------------------------------------
 * Core modules used: http, cluster, os, fs, events
 *
 * Features
 * 1. HTTP API
 *      • POST   /jobs           → submit job  {type, payload}
 *      • GET    /jobs/:id       → fetch per-job status
 *      • GET    /status         → aggregate queue metrics
 * 2. In-memory queue with disk persistence fallback (queue.json)
 * 3. Automatic worker-pool sizing = #logical CPUs
 * 4. Idle-worker job dispatching, progress tracking, retry on failure
 * 5. Robust process monitoring (worker crashes, master shutdown)
 *
 * NOTE: This file is self-contained; run with “node queue.js”.
 */

const http    = require('http');
const cluster = require('cluster');
const os      = require('os');
const fs      = require('fs');
const EventEmitter = require('events');

const PORT         = process.env.PORT || 8080;
const PERSIST_FILE = process.env.PERSIST_FILE || './queue.json';
const MAX_RETRY    = 3;               // default retry limit per job
const SAVE_DEBOUNCE_MS = 200;         // throttle disk writes

/* ---------- Shared Utilities ------------------------------------------------ */

const uuid = (()=>{                 // tiny UUID v4 replacement
  const rnd = () => Math.random().toString(16).slice(2);
  return () => `${rnd()}${rnd()}`.slice(0,32);
})();

function now() { return new Date().toISOString(); }

/* ---------- Job & Queue Data Structures ------------------------------------ */

class Job {
  constructor({type, payload}) {
    this.id         = uuid();
    this.type       = type;
    this.payload    = payload;
    this.status     = 'queued';   // queued | running | failed | done
    this.result     = null;
    this.error      = null;
    this.attempts   = 0;
    this.maxAttempts= MAX_RETRY;
    this.createdAt  = now();
    this.updatedAt  = now();
  }
}

class Queue extends EventEmitter {
  constructor() {
    super();
    this.pending  = [];                 // FIFO queue
    this.running  = new Map();          // jobId → worker.id
    this.failed   = new Map();          // jobId → Job
    this.done     = new Map();          // jobId → Job
    this._saveTimer = null;
    this._loadFromDisk();
  }

  enqueue(job) {
    this.pending.push(job);
    this._bump(job);
    this._scheduleSave();
    this.emit('enqueue');
  }

  markRunning(job, workerId) {
    job.status  = 'running';
    job.attempts+= 1;
    job.updatedAt = now();
    this.running.set(job.id, {job, workerId});
    this._scheduleSave();
  }

  markDone(job, result) {
    job.status    = 'done';
    job.result    = result;
    job.updatedAt = now();
    this.running.delete(job.id);
    this.done.set(job.id, job);
    this._scheduleSave();
  }

  markFailed(job, error) {
    job.error     = error;
    job.updatedAt = now();
    this.running.delete(job.id);

    if (job.attempts < job.maxAttempts) {
      job.status   = 'queued';
      this.pending.unshift(job);          // immediate retry (front of queue)
      this.emit('enqueue');
    } else {
      job.status   = 'failed';
      this.failed.set(job.id, job);
    }
    this._scheduleSave();
  }

  /* Persistence ------------------------------------------------------------- */

  _save() {
    const snapshot = {
      pending  : this.pending,
      running  : [...this.running.values()].map(v=>v.job),
      failed   : [...this.failed.values()],
      done     : [...this.done.values()]
    };
    try { fs.writeFileSync(PERSIST_FILE, JSON.stringify(snapshot, null, 2)); }
    catch (e) { console.error('Persist-error:', e); }
  }
  _scheduleSave() {
    clearTimeout(this._saveTimer);
    this._saveTimer = setTimeout(()=>this._save(), SAVE_DEBOUNCE_MS);
  }
  _loadFromDisk() {
    if (!fs.existsSync(PERSIST_FILE)) return;
    try {
      const data = JSON.parse(fs.readFileSync(PERSIST_FILE, 'utf8'));
      (data.pending  || []).forEach(j=>this.pending.push(Object.assign(new Job({}), j)));
      (data.failed   || []).forEach(j=>this.failed.set(j.id, Object.assign(new Job({}), j)));
      (data.done     || []).forEach(j=>this.done.set(j.id,   Object.assign(new Job({}), j)));
      // running jobs from previous crash revert to 'queued'
      (data.running  || []).forEach(j=>{
        j.status='queued';
        this.pending.unshift(Object.assign(new Job({}), j));
      });
      console.log(`[master] Restored queue: ${this.pending.length} pending, ${this.failed.size} failed, ${this.done.size} done`);
    } catch(e) {
      console.error('Corrupted queue persistence, starting fresh.', e);
    }
  }
}

/* ---------- Worker-Side Logic ---------------------------------------------- */

function workerMain() {
  process.on('message', async msg => {
    if (msg.cmd !== 'run') return;
    const job = msg.job;
    try {
      const result = await performJob(job);     // CPU/IO heavy task
      process.send({cmd:'done', id:job.id, result});
    } catch(err) {
      process.send({cmd:'fail', id:job.id, error: err.message || String(err)});
    }
  });

  /* Demo task executor (replace with real logic) */
  function performJob(job) {
    return new Promise((res, rej) => {
      try {
        // Example: simulate CPU-bound work via busy loop
        const t0 = Date.now();
        /* naive Fibonacci to burn CPU */
        const fib = n => n<=1? n : fib(n-1)+fib(n-2);
        fib(25);                               // tune for your CPU
        const elapsed = Date.now() - t0;
        res({elapsed});
      } catch(e) { rej(e); }
    });
  }
}

/* ---------- Master-Side Logic ---------------------------------------------- */

async function masterMain() {
  const queue   = new Queue();
  const numCPUs = os.cpus().length;

  /* Spawn workers ----------------------------------------------------------- */
  const workers = new Map();
  function spawnWorker() {
    const w = cluster.fork();
    workers.set(w.id, {worker:w, busy:false});
    w.on('exit', (code, signal) => {
      console.error(`[master] Worker ${w.id} died (${signal||code}). Respawning.`);
      workers.delete(w.id);
      spawnWorker();
    });
    w.on('message', msg => {
      if (!msg || !msg.cmd) return;
      const record = queue.running.get(msg.id);
      if (!record) return;   // job lost?
      const job = record.job;
      if (msg.cmd === 'done') {
        queue.markDone(job, msg.result);
        workers.get(w.id).busy = false;
      } else if (msg.cmd === 'fail') {
        queue.markFailed(job, msg.error);
        workers.get(w.id).busy = false;
      }
      dispatch();   // maybe schedule next
    });
  }
  for (let i=0;i<numCPUs;i++) spawnWorker();

  /* Job Dispatch ------------------------------------------------------------ */
  function dispatch() {
    for (const [id, meta] of workers) {
      if (meta.busy) continue;
      const job = queue.pending.shift();
      if (!job) break;
      meta.busy = true;
      queue.markRunning(job, id);
      meta.worker.send({cmd:'run', job});
    }
  }
  queue.on('enqueue', dispatch);

  /* HTTP Server ------------------------------------------------------------- */
  const server = http.createServer((req, res) => {
    const {method, url} = req;
    // CORS / JSON headers
    res.setHeader('Content-Type','application/json');
    res.setHeader('Access-Control-Allow-Origin','*');
    if (method==='OPTIONS') { res.writeHead(204); return res.end(); }

    /* POST /jobs ----------------------------------------------------------- */
    if (method==='POST' && url==='/jobs') {
      let body='';
      req.on('data', chunk=> body+=chunk);
      req.on('end', ()=>{
        try {
          const {type, payload} = JSON.parse(body||'{}');
          if (!type) throw new Error('Missing type');
          const job = new Job({type, payload});
          queue.enqueue(job);
          res.writeHead(202);
          res.end(JSON.stringify({id:job.id}));
        } catch(e) {
          res.writeHead(400);
          res.end(JSON.stringify({error:e.message}));
        }
      });
      return;
    }

    /* GET /jobs/:id -------------------------------------------------------- */
    if (method==='GET' && url.startsWith('/jobs/')) {
      const id = url.split('/')[2];
      const job = queue.running.get(id)?.job
               || queue.pending.find(j=>j.id===id)
               || queue.failed.get(id)
               || queue.done.get(id);
      if (!job) { res.writeHead(404); return res.end(JSON.stringify({error:'not_found'})); }
      res.end(JSON.stringify(job));
      return;
    }

    /* GET /status ---------------------------------------------------------- */
    if (method==='GET' && url==='/status') {
      const data = {
        queued : queue.pending.length,
        running: queue.running.size,
        failed : queue.failed.size,
        done   : queue.done.size,
        workers: {
          total : workers.size,
          busy  : [...workers.values()].filter(w=>w.busy).length
        }
      };
      res.end(JSON.stringify(data));
      return;
    }

    /* Default 404 ---------------------------------------------------------- */
    res.writeHead(404);
    res.end(JSON.stringify({error:'route_not_found'}));
  });

  server.listen(PORT, ()=> console.log(`[master] HTTP listening on ${PORT}`));

  /* Graceful shutdown ------------------------------------------------------ */
  const shutdown = () => {
    console.log('\n[master] Shutting down...');
    server.close(()=>console.log('[master] HTTP closed'));
    queue._save();
    for (const {worker} of workers.values()) worker.kill();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);

  /* Initial dispatch for persisted jobs */
  dispatch();
}

/* ---------- Entrypoint ----------------------------------------------------- */

if (cluster.isPrimary) {
  masterMain().catch(e=>{ console.error(e); process.exit(1); });
} else {
  workerMain();
}