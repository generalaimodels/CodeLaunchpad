/***************************************************************************************************
* File    : cluster-complete-cheatsheet.js
* Purpose : 10 laser-focused, self-contained demos that exercise EVERY public API exposed by Node.js
*           “cluster” — from bread-and-butter forking to the least-used scheduling knobs.
* Run     : `node cluster-complete-cheatsheet.js` (Node ≥ 16).  
* Style   : ES-2023, strict mode, top-tier readability, rich inline docs + expected output.
***************************************************************************************************/
'use strict';
const cluster        = require('cluster');
const { fork: cFork} = require('child_process');   // used by the orchestrator (root runner)
const path           = require('path');
const THIS_FILE      = __filename;

/***************************************************************************************************
* ───────────────────────────────────────── ROOT ORCHESTRATOR ─────────────────────────────────────
* The root (“director”) process starts a **fresh** Node.js instance for every numbered example in
* order to keep each cluster demo isolated (one cluster per process by design).
***************************************************************************************************/
if (!process.env.CLUSTER_EXAMPLE) {
  // ░░ root process ░░
  (async () => {
    for (let i = 1; i <= 10; ++i) {
      console.log(`\n=========  EXAMPLE ${i}  =========`);
      await runChildExample(i);
    }
    console.log('\n🏁  All cluster examples completed.\n');
  })();
  return; // prevent fall-through to example logic
}

/***************************************************************************************************
* Helper used by the orchestrator
***************************************************************************************************/
function runChildExample(num) {
  return new Promise((resolve) => {
    const child = cFork(THIS_FILE, [], {
      stdio : 'inherit',
      env   : { ...process.env, CLUSTER_EXAMPLE: String(num) }
    });
    child.on('exit', resolve);
  });
}

/***************************************************************************************************
* ───────────────────────────────────────── PER-EXAMPLE EXECUTION ─────────────────────────────────
* Exactly ONE of the following switch branches runs inside each spawned process (env-selected).
***************************************************************************************************/
switch (Number(process.env.CLUSTER_EXAMPLE)) {
  case 1: return ex1_basicFork();
  case 2: return ex2_ipc();
  case 3: return ex3_schedulingPolicy();
  case 4: return ex4_setupPrimary();
  case 5: return ex5_gracefulDisconnect();
  case 6: return ex6_forceKill();
  case 7: return ex7_connectedDeadFlags();
  case 8: return ex8_listeningEvent();
  case 9: return ex9_workersMap();
  case 10:return ex10_workerCtx();
  default: throw new Error('Unknown example id');
}

/***************************************************************************************************
*  EX-1 ─ Basic fork, isPrimary / isWorker, cluster.fork
***************************************************************************************************
* Expected output (≈):
*   [P] Forked worker #1 (pid 12345)
*   [P] Forked worker #2 (pid 12346)
*   [W1] Hello from worker 1  (isWorker=true)
*   [W2] Hello from worker 2  (isWorker=true)
*   [P] All workers exited – demo finished
***************************************************************************************************/
function ex1_basicFork() {
  if (cluster.isPrimary) {
    let exited = 0;
    // Fork two workers
    for (let i = 1; i <= 2; ++i) {
      const w = cluster.fork({ IDX: i });
      console.log(`[P] Forked worker #${i} (pid ${w.process.pid})`);
      w.on('exit', () => {
        if (++exited === 2) {
          console.log('[P] All workers exited – demo finished');
          process.exit(0);
        }
      });
    }
  } else {
    console.log(`[W${process.env.IDX}] Hello from worker ${process.env.IDX}  (isWorker=${cluster.isWorker})`);
    process.exit(0);
  }
}

/***************************************************************************************************
*  EX-2 ─ IPC messaging: worker.send / process.on('message')
***************************************************************************************************
* Expected output (≈):
*   [P] sending: ping
*   [W] got ping
*   [W] sending pong
*   [P] got pong
***************************************************************************************************/
function ex2_ipc() {
  if (cluster.isPrimary) {
    const worker = cluster.fork();
    worker.on('online', () => {
      console.log('[P] sending: ping');
      worker.send('ping');
    });
    worker.on('message', (msg) => {
      console.log('[P] got', msg);
      process.exit(0);
    });
  } else {
    process.on('message', (msg) => {
      console.log('[W] got', msg);
      console.log('[W] sending pong');
      process.send('pong');
      process.exit(0);
    });
  }
}

/***************************************************************************************************
*  EX-3 ─ schedulingPolicy, SCHED_RR vs SCHED_NONE
***************************************************************************************************
* Expected output (≈):
*   [P] Scheduling policy set to SCHED_NONE (value=1)
*   [P] worker exited – demo done
***************************************************************************************************/
function ex3_schedulingPolicy() {
  if (cluster.isPrimary) {
    // Use least-used constant: SCHED_NONE
    cluster.schedulingPolicy = cluster.SCHED_NONE;
    console.log(`[P] Scheduling policy set to SCHED_NONE (value=${cluster.schedulingPolicy})`);
    const w = cluster.fork();
    w.on('exit', () => {
      console.log('[P] worker exited – demo done');
      process.exit(0);
    });
  } else {
    console.log('[W] running under SCHED_NONE – just exiting');
    process.exit(0);
  }
}

/***************************************************************************************************
*  EX-4 ─ setupPrimary (a.k.a. setupMaster) + cluster.settings
***************************************************************************************************
* Expected output (≈):
*   [P] settings.exec: … cluster-complete-cheatsheet.js
*   [P] settings.args:  --example4-child
*   [W] argv[2]        : --example4-child
***************************************************************************************************/
function ex4_setupPrimary() {
  if (cluster.isPrimary) {
    cluster.setupPrimary({
      exec : __filename,            // file workers will run
      args : ['--example4-child'],  // extra argv visible to worker
      silent: false
    });
    console.log('[P] settings.exec:', cluster.settings.exec);
    console.log('[P] settings.args: ', cluster.settings.args.join(' '));

    const w = cluster.fork();
    w.on('exit', () => process.exit(0));
  } else {
    console.log('[W] argv[2]        :', process.argv[2]);
    process.exit(0);
  }
}

/***************************************************************************************************
*  EX-5 ─ Graceful restart via worker.disconnect()
***************************************************************************************************
* Expected output (≈):
*   [P] worker online
*   [P] ask for graceful shutdown
*   [W] got disconnect – cleanup done
*   [P] worker exited after disconnect: true
***************************************************************************************************/
function ex5_gracefulDisconnect() {
  if (cluster.isPrimary) {
    const w = cluster.fork();
    w.on('online', () => {
      console.log('[P] worker online');
      setTimeout(() => {
        console.log('[P] ask for graceful shutdown');
        w.disconnect();                        // triggers 'disconnect' event in worker
      }, 200);
    });
    w.on('exit', () => {
      console.log('[P] worker exited after disconnect:', w.exitedAfterDisconnect);
      process.exit(0);
    });
  } else {
    process.on('disconnect', () => {
      console.log('[W] got disconnect – cleanup done');
      process.exit(0);
    });
    // keep worker alive until disconnect arrives
    setInterval(() => {}, 1000);
  }
}

/***************************************************************************************************
*  EX-6 ─ Force kill a worker with signal
***************************************************************************************************
* Expected output (≈):
*   [P] sending SIGTERM
*   [P] exit code:null  signal:SIGTERM
***************************************************************************************************/
function ex6_forceKill() {
  if (cluster.isPrimary) {
    const w = cluster.fork();
    w.on('online', () => {
      console.log('[P] sending SIGTERM');
      w.kill('SIGTERM');
    });
    w.on('exit', (code, signal) => {
      console.log('[P] exit code:', code, ' signal:', signal);
      process.exit(0);
    });
  } else {
    // endless loop until killed
    setInterval(() => {}, 1000);
  }
}

/***************************************************************************************************
*  EX-7 ─ isConnected() & isDead()
***************************************************************************************************
* Expected output (≈):
*   [P] connected? true
*   [P] after exit → isDead()? true
***************************************************************************************************/
function ex7_connectedDeadFlags() {
  if (cluster.isPrimary) {
    const w = cluster.fork();
    w.on('online', () => console.log('[P] connected?', w.isConnected()));
    w.on('exit', () => {
      console.log('[P] after exit → isDead()? ', w.isDead());
      process.exit(0);
    });
    // let worker exit quickly
    w.send('quit');
  } else {
    process.on('message', () => process.exit(0));
  }
}

/***************************************************************************************************
*  EX-8 ─ 'listening' event (shared server port)
***************************************************************************************************
* Expected output (≈):
*   [P] worker 1 listening on 127.0.0.1:<random>
*   [P] shutdown
***************************************************************************************************/
function ex8_listeningEvent() {
  const http = require('http');

  if (cluster.isPrimary) {
    const w = cluster.fork({ IDX: 1 });
    cluster.on('listening', (_, { address, port }) => {
      console.log(`[P] worker 1 listening on ${address}:${port}`);
      w.kill(); // stop demo quickly
    });
    w.on('exit', () => {
      console.log('[P] shutdown');
      process.exit(0);
    });
  } else {
    const server = http.createServer(() => {}).listen(0, '127.0.0.1');
    // worker exits when master kills it
  }
}

/***************************************************************************************************
*  EX-9 ─ Enumerating cluster.workers map
***************************************************************************************************
* Expected output (≈):
*   [P] current workers: [1,2,3]
***************************************************************************************************/
function ex9_workersMap() {
  if (cluster.isPrimary) {
    const total = 3;
    let launched = 0, exited = 0;
    cluster.on('online', () => {
      if (++launched === total) {
        console.log('[P] current workers:', Object.keys(cluster.workers).map(Number));
        for (const id in cluster.workers) cluster.workers[id].kill();
      }
    });
    cluster.on('exit', () => {
      if (++exited === total) process.exit(0);
    });
    for (let i = 0; i < total; ++i) cluster.fork();
  } else {
    setInterval(() => {}, 1000);
  }
}

/***************************************************************************************************
*  EX-10 ─ Inside a worker: cluster.worker & its properties
***************************************************************************************************
* Expected output (≈):
*   [W] my id: 1  pid: 12345  connected:true
***************************************************************************************************/
function ex10_workerCtx() {
  if (cluster.isPrimary) {
    const w = cluster.fork();
    w.on('exit', () => process.exit(0));
  } else {
    console.log(`[W] my id: ${cluster.worker.id}  pid: ${process.pid}  connected:${cluster.worker.isConnected()}`);
    process.exit(0);
  }
}