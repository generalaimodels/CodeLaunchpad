/**
 * Node.js cluster Module: Comprehensive Examples
 * 
 * This file demonstrates all major and minor methods and properties of the Node.js cluster module.
 * Each example is self-contained, with clear code, comments, and expected output.
 * 
 * To run: `node <filename>.js`
 * Note: Some examples require running as a master process and will fork workers.
 */

const cluster = require('cluster');
const http = require('http');
const os = require('os');

if (cluster.isMaster) {
    // 1. cluster.isMaster, cluster.isPrimary, cluster.isWorker
    console.log('Is master:', cluster.isMaster); // true
    console.log('Is primary:', cluster.isPrimary); // true (since Node.js v16+)
    console.log('Is worker:', cluster.isWorker); // false
    // Expected output: Is master: true, Is primary: true, Is worker: false

    // 2. cluster.fork([env])
    // Forks a new worker process.
    const worker1 = cluster.fork({ WORKER_ID: 1 });
    const worker2 = cluster.fork({ WORKER_ID: 2 });

    // 3. cluster.workers
    // An object containing all active worker objects, indexed by id.
    setTimeout(() => {
        console.log('Active workers:', Object.keys(cluster.workers)); // [ '1', '2' ] (ids may vary)
        // Expected output: Active workers: [ '1', '2' ]
    }, 500);

    // 4. cluster.on('fork', callback)
    cluster.on('fork', (worker) => {
        console.log(`Worker ${worker.id} forked`);
        // Expected output: Worker 1 forked, Worker 2 forked
    });

    // 5. cluster.on('online', callback)
    cluster.on('online', (worker) => {
        console.log(`Worker ${worker.id} is online`);
        // Expected output: Worker 1 is online, Worker 2 is online
    });

    // 6. cluster.on('exit', callback)
    cluster.on('exit', (worker, code, signal) => {
        console.log(`Worker ${worker.id} exited with code ${code} and signal ${signal}`);
        // Expected output: Worker X exited with code Y and signal Z
    });

    // 7. cluster.on('listening', callback)
    cluster.on('listening', (worker, address) => {
        console.log(`Worker ${worker.id} is listening on ${address.address}:${address.port}`);
        // Expected output: Worker X is listening on 127.0.0.1:PORT
    });

    // 8. cluster.setupMaster([settings])
    // Sets up master settings before forking workers.
    cluster.setupMaster({
        exec: __filename, // This file
        args: [],
        silent: false
    });

    // 9. cluster.settings
    // Shows current cluster settings.
    console.log('Cluster settings:', cluster.settings);
    // Expected output: Cluster settings: { exec: ..., args: [], ... }

    // 10. cluster.disconnect([callback])
    // Gracefully disconnects all workers.
    setTimeout(() => {
        cluster.disconnect(() => {
            console.log('All workers disconnected');
            // Expected output: All workers disconnected
        });
    }, 2000);

    // 11. Sending messages to workers (worker.send)
    setTimeout(() => {
        for (const id in cluster.workers) {
            cluster.workers[id].send({ cmd: 'greet', id });
        }
    }, 1000);

    // 12. cluster.SCHED_NONE and cluster.SCHED_RR
    // Scheduling policies (default is SCHED_RR on most platforms)
    console.log('Scheduling policy:', cluster.SCHED_RR === cluster.settings.schedulingPolicy ? 'Round Robin' : 'None');
    // Expected output: Scheduling policy: Round Robin (or None)

} else {
    // Worker process code

    // 13. cluster.worker
    // The current worker object.
    console.log('Worker id:', cluster.worker.id); // 1 or 2
    // Expected output: Worker id: 1 (or 2)

    // 14. process.env in worker
    console.log('Worker env:', process.env.WORKER_ID); // 1 or 2
    // Expected output: Worker env: 1 (or 2)

    // 15. worker.on('message', callback)
    process.on('message', (msg) => {
        if (msg.cmd === 'greet') {
            console.log(`Worker ${cluster.worker.id} received greeting from master`);
            // Expected output: Worker X received greeting from master
        }
    });

    // 16. worker.send(message[, sendHandle[, options]][, callback])
    if (cluster.worker.id === 1) {
        process.send({ reply: 'Hello master, from worker 1' });
    }

    // 17. Sharing server ports between workers
    // All workers can share the same port.
    const server = http.createServer((req, res) => {
        res.writeHead(200);
        res.end(`Handled by worker ${cluster.worker.id}\n`);
    });
    server.listen(0, '127.0.0.1'); // OS assigns a random port

    // 18. worker.disconnect()
    // Worker can disconnect itself.
    setTimeout(() => {
        if (cluster.worker.id === 2) {
            cluster.worker.disconnect();
        }
    }, 1500);

    // 19. worker.isConnected(), worker.isDead()
    setTimeout(() => {
        console.log(`Worker ${cluster.worker.id} isConnected:`, cluster.worker.isConnected());
        console.log(`Worker ${cluster.worker.id} isDead:`, cluster.worker.isDead());
        // Expected output: isConnected: true/false, isDead: true/false
    }, 1700);

    // 20. Exception handling in worker
    process.on('uncaughtException', (err) => {
        console.error(`Worker ${cluster.worker.id} uncaught exception:`, err.message);
        // Expected output: Worker X uncaught exception: <error message>
    });
    // Uncomment to test:
    // if (cluster.worker.id === 1) throw new Error('Test exception in worker');
}

/**
 * Summary:
 * - All major and minor cluster methods, properties, and events are covered.
 * - Each example is self-contained and demonstrates expected behavior.
 * - Master and worker code are both included in the same file.
 * - Modify or uncomment lines to see different results and behaviors.
 */