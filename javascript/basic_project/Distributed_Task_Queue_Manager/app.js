// distributed_task_queue_manager.js

const http = require('http');
const os = require('os');
const fs = require('fs');
const path = require('path');
const cluster = require('cluster');
const { EventEmitter } = require('events');
const { fork } = require('child_process');

// --- Configuration ---
const PORT = 8080;
const PERSISTENCE_FILE = path.join(__dirname, 'job_queue.json');
const MAX_RETRIES = 3;

// --- Job Queue Implementation ---
class PersistentJobQueue extends EventEmitter {
    constructor() {
        super();
        this.queue = [];
        this.running = new Map(); // jobId -> job
        this.failed = new Map();  // jobId -> job
        this.completed = new Map(); // jobId -> job
        this.loadFromDisk();
    }

    enqueue(job) {
        this.queue.push(job);
        this.persist();
        this.emit('jobEnqueued', job);
    }

    dequeue() {
        const job = this.queue.shift();
        this.persist();
        return job;
    }

    requeue(job) {
        this.queue.unshift(job);
        this.persist();
        this.emit('jobRequeued', job);
    }

    markRunning(job) {
        this.running.set(job.id, job);
        this.emit('jobRunning', job);
    }

    markCompleted(job) {
        this.running.delete(job.id);
        this.completed.set(job.id, job);
        this.emit('jobCompleted', job);
    }

    markFailed(job) {
        this.running.delete(job.id);
        this.failed.set(job.id, job);
        this.emit('jobFailed', job);
    }

    getStatus() {
        return {
            queued: this.queue.length,
            running: this.running.size,
            failed: this.failed.size,
            completed: this.completed.size,
            queue: this.queue.map(j => j.id),
            runningJobs: Array.from(this.running.keys()),
            failedJobs: Array.from(this.failed.keys()),
            completedJobs: Array.from(this.completed.keys())
        };
    }

    persist() {
        try {
            fs.writeFileSync(PERSISTENCE_FILE, JSON.stringify({
                queue: this.queue,
                failed: Array.from(this.failed.values())
            }, null, 2));
        } catch (err) {
            // Log error, but do not throw
            console.error('Persistence error:', err);
        }
    }

    loadFromDisk() {
        if (fs.existsSync(PERSISTENCE_FILE)) {
            try {
                const data = JSON.parse(fs.readFileSync(PERSISTENCE_FILE, 'utf-8'));
                this.queue = data.queue || [];
                for (const job of (data.failed || [])) {
                    this.failed.set(job.id, job);
                }
            } catch (err) {
                // Corrupted file, start fresh
                this.queue = [];
                this.failed = new Map();
            }
        }
    }
}

// --- Job ID Generator ---
function generateJobId() {
    return 'job_' + Date.now() + '_' + Math.random().toString(36).slice(2, 10);
}

// --- Worker Process Logic ---
function workerProcess() {
    process.on('message', async (msg) => {
        if (msg.type === 'processJob') {
            const job = msg.job;
            let result, error;
            try {
                // Simulate CPU-intensive work (replace with real logic)
                result = await simulateJob(job.data);
                process.send({ type: 'jobCompleted', jobId: job.id, result });
            } catch (err) {
                error = err.message || 'Unknown error';
                process.send({ type: 'jobFailed', jobId: job.id, error });
            }
        }
    });

    async function simulateJob(data) {
        // Simulate CPU work (e.g., image processing)
        await new Promise(res => setTimeout(res, 1000 + Math.random() * 2000));
        if (Math.random() < 0.15) throw new Error('Simulated job failure');
        return { processed: true, input: data };
    }
}

// --- Master Process Logic ---
if (cluster.isMaster) {
    // --- State ---
    const jobQueue = new PersistentJobQueue();
    const numWorkers = os.cpus().length;
    const workers = [];
    const workerStatus = new Map(); // workerId -> { idle: true/false, jobId: string|null }

    // --- Worker Management ---
    function spawnWorkers() {
        for (let i = 0; i < numWorkers; i++) {
            const worker = cluster.fork();
            workers.push(worker);
            workerStatus.set(worker.id, { idle: true, jobId: null });

            worker.on('message', (msg) => {
                if (msg.type === 'jobCompleted') {
                    const job = jobQueue.running.get(msg.jobId);
                    if (job) {
                        job.result = msg.result;
                        jobQueue.markCompleted(job);
                    }
                    workerStatus.get(worker.id).idle = true;
                    workerStatus.get(worker.id).jobId = null;
                    dispatchJobs();
                } else if (msg.type === 'jobFailed') {
                    const job = jobQueue.running.get(msg.jobId);
                    if (job) {
                        job.retries = (job.retries || 0) + 1;
                        if (job.retries > MAX_RETRIES) {
                            job.error = msg.error;
                            jobQueue.markFailed(job);
                        } else {
                            jobQueue.requeue(job);
                        }
                    }
                    workerStatus.get(worker.id).idle = true;
                    workerStatus.get(worker.id).jobId = null;
                    dispatchJobs();
                }
            });

            worker.on('exit', (code, signal) => {
                // Respawn worker if it dies
                workerStatus.delete(worker.id);
                const newWorker = cluster.fork();
                workers.push(newWorker);
                workerStatus.set(newWorker.id, { idle: true, jobId: null });
            });
        }
    }

    // --- Job Dispatching ---
    function dispatchJobs() {
        for (const worker of workers) {
            const status = workerStatus.get(worker.id);
            if (status && status.idle && jobQueue.queue.length > 0) {
                const job = jobQueue.dequeue();
                jobQueue.markRunning(job);
                status.idle = false;
                status.jobId = job.id;
                worker.send({ type: 'processJob', job });
            }
        }
    }

    // --- HTTP API ---
    const server = http.createServer(async (req, res) => {
        if (req.method === 'POST' && req.url === '/submit') {
            let body = '';
            req.on('data', chunk => body += chunk);
            req.on('end', () => {
                try {
                    const data = JSON.parse(body);
                    const job = {
                        id: generateJobId(),
                        data,
                        status: 'queued',
                        retries: 0,
                        created: Date.now()
                    };
                    jobQueue.enqueue(job);
                    dispatchJobs();
                    res.writeHead(202, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ jobId: job.id }));
                } catch (err) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Invalid JSON' }));
                }
            });
        } else if (req.method === 'GET' && req.url.startsWith('/status')) {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(jobQueue.getStatus()));
        } else if (req.method === 'GET' && req.url.startsWith('/job/')) {
            const jobId = req.url.split('/').pop();
            let job = jobQueue.running.get(jobId) || jobQueue.failed.get(jobId) || jobQueue.completed.get(jobId);
            if (!job) {
                job = jobQueue.queue.find(j => j.id === jobId);
            }
            if (job) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(job));
            } else {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Job not found' }));
            }
        } else {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Not found' }));
        }
    });

    // --- Event Logging (for process monitoring) ---
    jobQueue.on('jobEnqueued', job => {
        console.log(`[QUEUE] Job enqueued: ${job.id}`);
    });
    jobQueue.on('jobRunning', job => {
        console.log(`[WORKER] Job running: ${job.id}`);
    });
    jobQueue.on('jobCompleted', job => {
        console.log(`[COMPLETE] Job completed: ${job.id}`);
    });
    jobQueue.on('jobFailed', job => {
        console.log(`[FAILED] Job failed: ${job.id} (retries: ${job.retries})`);
    });
    jobQueue.on('jobRequeued', job => {
        console.log(`[RETRY] Job requeued: ${job.id} (retry: ${job.retries})`);
    });

    // --- Startup ---
    spawnWorkers();
    server.listen(PORT, () => {
        console.log(`Distributed Task Queue Manager running on port ${PORT}`);
        console.log(`Worker processes: ${numWorkers}`);
    });

    // --- Graceful Shutdown ---
    process.on('SIGINT', () => {
        jobQueue.persist();
        process.exit(0);
    });

} else {
    // --- Worker Process Entry Point ---
    workerProcess();
}