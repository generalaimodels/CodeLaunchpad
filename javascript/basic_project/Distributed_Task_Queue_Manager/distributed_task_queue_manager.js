/*  Distributed Task Queue Manager
    --------------------------------
    One-file Node.js implementation.

    Features:
    • HTTP API (submit jobs, inspect metrics, job status)
    • In-memory queue with disk-persistence fallback
    • Auto-spawns N (= CPU cores) workers via cluster
    • Dynamic job dispatch, progress tracking, retries
    • Resilient to worker/master crashes (queue reload)               */

    'use strict';

    const http      = require('http');
    const cluster   = require('cluster');
    const os        = require('os');
    const fs        = require('fs');
    const path      = require('path');
    const events    = require('events');
    const { v4: uuid } = require('crypto').randomUUID ? { v4: () => crypto.randomUUID() } : require('uuid'); // Node ≥19 has crypto.randomUUID
    
    /* ---------- Configuration ---------- */
    const PORT              = process.env.PORT || 8080;
    const QUEUE_FILE        = path.join(__dirname, 'queue.json');
    const MAX_RETRIES       = 3;
    const PERSIST_INTERVAL  = 1000;          // ms – debounce persistence
    const JOB_TIMEOUT_MS    = 15 * 60e3;     // 15 min safety net
    
    /* ---------- Shared Job Model ---------- */
    class Job {
        constructor(payload) {
            this.id         = uuid();
            this.payload    = payload;       // arbitrary JSON sent by client
            this.status     = 'queued';      // queued|running|completed|failed
            this.result     = null;          // worker-returned data or error
            this.retries    = 0;
            this.createdAt  = Date.now();
            this.startedAt  = null;
            this.endedAt    = null;
        }
    }
    
    /* ============ MASTER PROCESS ============ */
    if (cluster.isMaster) {
    
        /* ----- State ----- */
        const eventBus   = new events.EventEmitter();
        const jobQueue   = [];              // FIFO queue of Job IDs
        const jobs       = new Map();       // id -> Job
        const idleWorkers= new Set();       // worker.id values
        let persistTimer = null;
    
        /* ----- Helpers ----- */
        const enqueueJob = job => {
            jobs.set(job.id, job);
            jobQueue.push(job.id);
            schedulePersist();
            dispatch();                     // attempt immediate dispatch
        };
    
        const schedulePersist = () => {
            if (persistTimer) return;
            persistTimer = setTimeout(() => {
                persistTimer = null;
                persistToDisk();
            }, PERSIST_INTERVAL);
        };
    
        const persistToDisk = () => {
            const snapshot = {
                jobs:[...jobs.values()],
                queue: jobQueue.slice()
            };
            fs.writeFile(QUEUE_FILE, JSON.stringify(snapshot), err => {
                if (err) console.error('⛔ Persist error:', err);
            });
        };
    
        const restoreFromDisk = () => {
            try {
                if (!fs.existsSync(QUEUE_FILE)) return;
                const data = JSON.parse(fs.readFileSync(QUEUE_FILE,'utf8'));
                data.jobs.forEach(j => jobs.set(j.id, j));
                jobQueue.push(...data.queue);
                console.log(`🔄 Restored ${data.queue.length} queued job(s) from disk`);
            } catch (e) { console.error('⛔ Restore failed:', e); }
        };
    
        const metrics = () => {
            let queued=0, running=0, failed=0, completed=0;
            for (const j of jobs.values()) {
                switch(j.status){
                    case 'queued':   queued++;   break;
                    case 'running':  running++;  break;
                    case 'failed':   failed++;   break;
                    case 'completed':completed++;break;
                }
            }
            return {queued,running,failed,completed,total:jobs.size,workers:Object.keys(cluster.workers).length,idle:idleWorkers.size};
        };
    
        /* ----- Dispatch Engine ----- */
        const dispatch = () => {
            while (jobQueue.length && idleWorkers.size) {
                const jobId  = jobQueue.shift();
                const job    = jobs.get(jobId);
                if (!job || job.status!=='queued') continue;
    
                const workerId = idleWorkers.values().next().value;
                const worker   = cluster.workers[workerId];
                if (!worker) { idleWorkers.delete(workerId); jobQueue.unshift(jobId); continue; }
    
                idleWorkers.delete(workerId);
    
                job.status    = 'running';
                job.startedAt = Date.now();
                schedulePersist();
    
                worker.send({cmd:'run', job});
                // timeout guard
                job._timeout = setTimeout(()=>{
                    console.log(`⏰ Job ${job.id} timed out on worker ${workerId}`);
                    worker.kill(); // force restart
                }, JOB_TIMEOUT_MS);
            }
        };
    
        /* ----- Cluster Setup ----- */
        restoreFromDisk();
        const cpuCount = os.cpus().length || 1;
        console.log(`🚀 Master ${process.pid} spawning ${cpuCount} workers`);
        for (let i=0;i<cpuCount;i++) cluster.fork();
    
        cluster.on('online', worker=>{
            idleWorkers.add(worker.id);
            dispatch();
        });
    
        cluster.on('exit', (worker,code,signal)=>{
            console.log(`⚠️  Worker ${worker.id} exited (${signal||code}); spawning replacement`);
            idleWorkers.delete(worker.id);
            cluster.fork();
        });
    
        cluster.on('message', (worker,msg)=>{
            const {cmd, jobId, result, error} = msg;
            if (cmd!=='done' && cmd!=='error') return;
    
            const job = jobs.get(jobId);
            if (!job) return;
            clearTimeout(job._timeout);
    
            if (cmd==='done') {
                job.status   = 'completed';
                job.result   = result;
                job.endedAt  = Date.now();
            } else { // error
                job.retries += 1;
                if (job.retries<=MAX_RETRIES){
                    console.log(`🔁 Retry ${job.retries}/${MAX_RETRIES} for job ${jobId}`);
                    job.status='queued';
                    job.result=null;
                    job.startedAt=job.endedAt=null;
                    jobQueue.push(jobId);
                } else {
                    job.status  = 'failed';
                    job.result  = error;
                    job.endedAt = Date.now();
                }
            }
            idleWorkers.add(worker.id);
            schedulePersist();
            dispatch();
        });
    
        /* ----- HTTP API ----- */
        const server = http.createServer((req,res)=>{
            // basic routing
            if (req.method==='POST' && req.url==='/job') {
                let body='';
                req.on('data',d=>body+=d);
                req.on('end',()=>{
                    try {
                        const payload = body?JSON.parse(body):{};
                        const job = new Job(payload);
                        enqueueJob(job);
                        res.writeHead(202,{'Content-Type':'application/json'});
                        res.end(JSON.stringify({id:job.id}));
                    }catch(e){
                        res.writeHead(400);
                        res.end(JSON.stringify({error:'Invalid JSON'}));
                    }
                });
            }
            else if (req.method==='GET' && req.url.startsWith('/job/')) {
                const id = req.url.split('/')[2];
                const job = jobs.get(id);
                if (!job){ res.writeHead(404); res.end(); return; }
                res.writeHead(200,{'Content-Type':'application/json'});
                res.end(JSON.stringify(job));
            }
            else if (req.method==='GET' && req.url==='/status') {
                res.writeHead(200,{'Content-Type':'application/json'});
                res.end(JSON.stringify(metrics()));
            }
            else { res.writeHead(404); res.end(); }
        }).listen(PORT, ()=>console.log(`🌐 Listening on :${PORT}`));
    
        /* ----- Graceful Shutdown ----- */
        const shutdown = ()=>{ console.log('\n💾 Persisting before shutdown'); persistToDisk(); process.exit(0); };
        process.on('SIGINT',shutdown);
        process.on('SIGTERM',shutdown);
    }
    
    /* ============ WORKER PROCESS ============ */
    else {
        console.log(`⚙️  Worker ${process.pid} online`);
        process.on('message', async msg=>{
            if (msg.cmd!=='run') return;
            const job = msg.job;
            try {
                const result = await simulateHeavyComputation(job.payload);
                process.send({cmd:'done', jobId:job.id, result});
            } catch (err){
                process.send({cmd:'error', jobId:job.id, error:err.message||'unknown'});
            }
        });
    
        /* ----- demo heavy work (replace with real logic) ----- */
        function simulateHeavyComputation(payload){
            return new Promise((resolve,reject)=>{
                const duration = Math.round(1000+Math.random()*4000);   // 1‒5 s
                const fail = Math.random()<0.2;                         // 20 % failure
                setTimeout(()=>{
                    if (fail) reject(new Error('Random failure')); 
                    else resolve({echo:payload, took:duration});
                }, duration);
            });
        }
    }