/**************************************************************************************************
 * Chapter 15 | Node.js Advanced & Microservices
 * -----------------------------------------------------------------------------------------------
 * ONE self‑contained .js playground. 6 sections × 5 concise, illustrative examples each.
 * Run in Node ≥16; external deps are lazy‑loaded under try/catch to stay optional.
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION EVL — Event Loop Internals & libuv                      */
/*───────────────────────────────────────────────────────────────────*/

/* EVL‑Example‑1:  nextTick > micro‑task before promise then */
process.nextTick(() => console.log('EVL‑1 nextTick'));
Promise.resolve().then(() => console.log('EVL‑1 promise'));
setImmediate(() => console.log('EVL‑1 immediate'));

/* EVL‑Example‑2:  I/O phase illustration */
const fs = require('fs');
fs.readFile(__filename, () => console.log('EVL‑2 I/O callback after poll'));

/* EVL‑Example‑3:  CPU‑bound blocking (event‑loop delay) */
const start = Date.now();
while (Date.now() - start < 50); // block 50 ms
setTimeout(() => console.log('EVL‑3 timer fired +', Date.now() - start, 'ms'), 0);

/* EVL‑Example‑4:  libuv thread‑pool size tuning */
console.log('EVL‑4 UV_THREADPOOL_SIZE =', process.env.UV_THREADPOOL_SIZE ?? 4);

/* EVL‑Example‑5:  Measuring event‑loop lag */
const lag = require('perf_hooks').monitorEventLoopDelay({ resolution: 10 });
lag.enable();
setTimeout(() => { lag.disable(); console.log('EVL‑5 lag mean', lag.mean / 1e6, 'ms'); }, 200);

/*───────────────────────────────────────────────────────────────────*/
/* SECTION CLU — Clustering & Worker Threads                       */
/*───────────────────────────────────────────────────────────────────*/

const cluster = require('cluster');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

/* CLU‑Example‑1:  Fork workers equal to CPU cores */
if (cluster.isPrimary && !process.env.NO_CLUSTER) {
  require('os').cpus().forEach(() => cluster.fork({ NO_CLUSTER: 1 }));
  cluster.on('exit', id => console.log('CLU‑1 worker died', id.process.pid));
}

/* CLU‑Example‑2:  Worker thread simple computation */
if (isMainThread) {
  new Worker(__filename, { workerData: 5, env: { SKIP_WT: true } });
} else if (!process.env.SKIP_WT) {
  parentPort.postMessage(workerData ** 2);
  parentPort.on('message', d => console.log('CLU‑2 got', d));
}

/* CLU‑Example‑3:  Broadcast to all cluster workers */
if (cluster.isPrimary) cluster.on('online', w => w.send('ping'));
else process.on('message', m => console.log('CLU‑3 worker', process.pid, m));

/* CLU‑Example‑4:  Worker pool pattern */
function runTask(n) {
  return new Promise(res => {
    const w = new Worker(`const {parentPort}=require('worker_threads');parentPort.postMessage(${n}*2)`, { eval: true });
    w.on('message', res);
  });
}
runTask(21).then(r => console.log('CLU‑4 worker result', r));

/* CLU‑Example‑5:  Graceful shutdown */
process.on('SIGINT', () => {
  for (const id in cluster.workers) cluster.workers[id].kill();
  console.log('CLU‑5 graceful exit'); process.exit();
});

/*───────────────────────────────────────────────────────────────────*/
/* SECTION STM — Streams, Buffers & Backpressure                   */
/*───────────────────────────────────────────────────────────────────*/

const { Readable, Transform, pipeline } = require('stream');

/* STM‑Example‑1:  Custom Readable stream */
const src = new Readable({ read() { this.push(Buffer.from('data\n')); this.push(null); } });

/* STM‑Example‑2:  Transform uppercase */
const upper = new Transform({ transform(c, _, cb) { cb(null, c.toString().toUpperCase()); } });

/* STM‑Example‑3:  Backpressure via highWaterMark */
const slowDest = new Transform({ highWaterMark: 8, transform(c, _, cb){ setTimeout(()=>cb(null,c),50);} });

pipeline(src, upper, slowDest, err => console.log('STM‑3 done', !!err));

/* STM‑Example‑4:  Buffer concat & length */
const buf = Buffer.concat([Buffer.from('a'), Buffer.from('b')]);
console.log('STM‑4 buffer len', buf.length);

/* STM‑Example‑5:  Stream pipeline promise util */
require('stream/promises').pipeline(
  fs.createReadStream(__filename),
  new Transform({ transform(c, _, cb){ cb(null,c); } }),
  fs.createWriteStream('copy.tmp')
).then(() => fs.unlinkSync('copy.tmp'));

/*───────────────────────────────────────────────────────────────────*/
/* SECTION PROC — Child Processes & IPC                            */
/*───────────────────────────────────────────────────────────────────*/

const { spawn, execFile, fork, exec } = require('child_process');

/* PROC‑Example‑1:  spawn ls */
spawn(process.platform === 'win32' ? 'cmd' : 'ls', ['/c', '.'].slice(0, process.platform === 'win32' ? 2 : 1))
  .stdout.on('data', d => console.log('PROC‑1 ls out', d.toString()));

/* PROC‑Example‑2:  execFile node -v */
execFile(process.execPath, ['-v'], (_, o) => console.log('PROC‑2 node', o.trim()));

/* PROC‑Example‑3:  fork child for computation */
const child = fork(__filename, ['child']);
if (process.argv[2] === 'child') process.send(42);
child.on('message', m => console.log('PROC‑3 parent got', m));

/* PROC‑Example‑4:  exec shell command with timeout */
exec('sleep 1 && echo done', { timeout: 500 }, (e, o) => console.log('PROC‑4 timeout?', !!e, o));

/* PROC‑Example‑5:  IPC between cluster workers via process.send above (reuse) */

/*───────────────────────────────────────────────────────────────────*/
/* SECTION MSC — Building Microservices (NestJS, Fastify)          */
/*───────────────────────────────────────────────────────────────────*/

/* MSC‑Example‑1:  Fastify hello world */
(async () => {
  try {
    const fastify = (await import('fastify')).default();
    fastify.get('/ping', async () => 'pong');
    await fastify.listen({ port: 5000 });
    console.log('MSC‑1 fastify on 5000');
  } catch {}
})();

/* MSC‑Example‑2:  Fastify plugin pattern */
async function utilPlugin (f, opts) { f.decorate('util', ()=>'x'); }
(async () => {
  try { const fastify =(await import('fastify')).default(); fastify.register(utilPlugin);
    await fastify.ready(); console.log('MSC‑2 util =>', fastify.util()); } catch{}
})();

/* MSC‑Example‑3:  NestJS controller skeleton */
const nestCtrl = `
@Controller('cats')
export class CatsController {
  @Get() findAll(){ return 'MSC‑3 cats'; }
}
`; console.log('MSC‑3 Nest snippet lines', nestCtrl.trim().split('\n').length);

/* MSC‑Example‑4:  Fastify‑RabbitMQ microservice link (pseudo) */
const rabbitCfg = { url:'amqp://localhost', queue:'jobs' };
console.log('MSC‑4 Rabbit cfg', rabbitCfg.queue);

/* MSC‑Example‑5:  Health check endpoint */
(async () => {
  try { const fastify=(await import('fastify')).default(); fastify.get('/health',()=>({ok:true}));
    await fastify.listen({port:5050}); } catch{}
})();

/*───────────────────────────────────────────────────────────────────*/
/* SECTION RPC — GraphQL & gRPC Servers                            */
/*───────────────────────────────────────────────────────────────────*/

/* RPC‑Example‑1:  Apollo Server quick start */
(async () => {
  try {
    const { ApolloServer, gql } = await import('@apollo/server');
    const typeDefs = gql`type Query{hello:String}`;
    const resolvers = { Query:{ hello: ()=>'RPC‑1 hi' } };
    const srv = new ApolloServer({ typeDefs, resolvers });
    await srv.start(); console.log('RPC‑1 Apollo started');
  } catch{}
})();

/* RPC‑Example‑2:  GraphQL schema SDL string */
const sdl = `type Todo{id:ID! text:String!} type Query{todos:[Todo]}`;
console.log('RPC‑2 SDL length', sdl.length);

/* RPC‑Example‑3:  GraphQL subscription payload */
const subMsg = { data:{ onMessage:{ id:1, body:'hi' } } };
console.log('RPC‑3 sub payload keys', Object.keys(subMsg.data.onMessage));

/* RPC‑Example‑4:  gRPC proto snippet */
const proto = `
syntax="proto3";
service Greeter{ rpc SayHello (HelloReq) returns (HelloRes); }
message HelloReq{ string name=1;}
message HelloRes{ string msg=1;}
`;
console.log('RPC‑4 proto lines', proto.trim().split('\n').length);

/* RPC‑Example‑5:  gRPC server implementation (optional deps) */
(async () => {
  try {
    const grpc = await import('@grpc/grpc-js'); const protoLoader = await import('@grpc/proto-loader');
    const pkgDef = protoLoader.loadSync('/tmp/greeter.proto', {}); const obj = grpc.loadPackageDefinition(pkgDef);
    const server = new grpc.Server(); server.addService(obj.Greeter.service, { SayHello:(c,cb)=>cb(null,{msg:'hi'}) });
    server.bindAsync('0.0.0.0:7000', grpc.ServerCredentials.createInsecure(), ()=>server.start());
    console.log('RPC‑5 gRPC up 7000');
  } catch{}
})();