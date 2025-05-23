/* SECTION EL — Event Loop Internals & libuv */

/* EL‑1: process.nextTick vs Promise vs setImmediate vs setTimeout */
(function EL1() {
    console.log('EL1 start');
    setTimeout(() => console.log('timeout'), 0);
    setImmediate(() => console.log('immediate'));
    Promise.resolve().then(() => console.log('promise'));
    process.nextTick(() => console.log('nextTick'));
    console.log('EL1 end');
  })();
  
  /* EL‑2: fs.readFile callback ordering */
  (function EL2() {
    const fs = require('fs');
    fs.readFile(__filename, () => console.log('EL2 fs.readFile'));
    setImmediate(() => console.log('EL2 setImmediate'));
    process.nextTick(() => console.log('EL2 nextTick'));
  })();
  
  /* EL‑3: Offloading to libuv threadpool via crypto.pbkdf2 */
  (function EL3() {
    const crypto = require('crypto');
    console.time('EL3 pbkdf2');
    crypto.pbkdf2('a','salt', 100000, 64, 'sha512', () => {
      console.timeEnd('EL3 pbkdf2');
    });
    console.log('EL3 scheduling pbkdf2');
  })();
  
  /* EL‑4: setImmediate inside I/O callback vs timers */
  (function EL4() {
    const fs = require('fs');
    fs.open(__filename, () => {
      setTimeout(() => console.log('EL4 timeout in I/O'), 0);
      setImmediate(() => console.log('EL4 immediate in I/O'));
    });
  })();
  
  /* EL‑5: nextTick recursion and starvation caution */
  (function EL5() {
    let i = 0;
    function recur() {
      if (i++ < 3) {
        process.nextTick(recur);
        console.log('EL5 tick', i);
      }
    }
    recur();
  })();
  
  
  /* SECTION CL — Clustering & Worker Threads */
  
  /* CL‑1: Basic cluster.fork */
  (function CL1() {
    const cluster = require('cluster');
    if (cluster.isMaster) {
      console.log('CL1 Master PID', process.pid);
      cluster.fork();
    } else {
      console.log('CL1 Worker PID', process.pid);
    }
  })();
  
  /* CL‑2: Handling worker exit & respawn */
  (function CL2() {
    const cluster = require('cluster');
    if (cluster.isMaster) {
      const w = cluster.fork();
      w.on('exit', (code, sig) => {
        console.log(`CL2 Worker died (${code},${sig}); respawning`);
        cluster.fork();
      });
      w.kill();
    }
  })();
  
  /* CL‑3: Worker Threads basic */
  (function CL3() {
    const { Worker, isMainThread, parentPort, threadId } = require('worker_threads');
    if (isMainThread) {
      const w = new Worker(__filename);
      w.on('message', msg => console.log('CL3 from worker:', msg));
    } else {
      parentPort.postMessage(`Hello from thread ${threadId}`);
    }
  })();
  
  /* CL‑4: Passing data to Worker and back */
  (function CL4() {
    const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
    if (isMainThread) {
      const w = new Worker(__filename, { workerData: { num: 42 } });
      w.on('message', msg => console.log('CL4 processed:', msg));
    } else {
      parentPort.postMessage(workerData.num * 2);
    }
  })();
  
  /* CL‑5: SharedArrayBuffer usage */
  (function CL5() {
    const { Worker, isMainThread, parentPort } = require('worker_threads');
    if (isMainThread) {
      const sab = new SharedArrayBuffer(4);
      const arr = new Int32Array(sab);
      const w = new Worker(__filename, { workerData: sab });
      w.on('message', () => console.log('CL5 value:', Atomics.load(arr, 0)));
    } else {
      const { workerData } = require('worker_threads');
      const arr = new Int32Array(workerData);
      Atomics.store(arr, 0, 7);
      parentPort.postMessage('done');
    }
  })();
  
  
  /* SECTION SB — Streams, Buffers & Backpressure */
  
  const { Readable, Writable, Transform, pipeline } = require('stream');
  
  /* SB‑1: Readable from array */
  (function SB1() {
    const data = ['a','b','c'];
    const r = Readable.from(data);
    r.on('data', chunk => console.log('SB1 chunk', chunk.toString()));
  })();
  
  /* SB‑2: Writable that logs and enforces backpressure */
  (function SB2() {
    const w = new Writable({
      write(chunk, enc, cb) {
        console.log('SB2 write', chunk.toString());
        setTimeout(cb, 100); // slow consumer
      }
    });
    const r = Readable.from(['1','2','3','4']);
    r.pipe(w);
  })();
  
  /* SB‑3: Transform uppercase */
  (function SB3() {
    const t = new Transform({
      transform(chunk, enc, cb) {
        cb(null, chunk.toString().toUpperCase());
      }
    });
    Readable.from(['x','y','z'])
      .pipe(t)
      .on('data', d => console.log('SB3', d.toString()));
  })();
  
  /* SB‑4: File stream with backpressure handling */
  (function SB4() {
    const fs = require('fs');
    const rs = fs.createReadStream(__filename, { highWaterMark: 16 });
    rs.on('data', chunk => {
      console.log('SB4 chunk len', chunk.length);
      rs.pause();
      setTimeout(() => rs.resume(), 50);
    });
  })();
  
  /* SB‑5: Buffer manipulation */
  (function SB5() {
    const buf = Buffer.from('hello world');
    console.log('SB5 slice', buf.slice(0,5).toString());
    console.log('SB5 concat', Buffer.concat([buf, Buffer.from('!!!')]).toString());
  })();
  
  
  /* SECTION CP — Child Processes & IPC */
  
  const { spawn, exec, fork } = require('child_process');
  
  /* CP‑1: spawn external command */
  (function CP1() {
    const ls = spawn('echo',['hello']);
    ls.stdout.on('data', d => console.log('CP1 stdout', d.toString()));
  })();
  
  /* CP‑2: exec with callback */
  (function CP2() {
    exec('node -v', (err, stdout) => {
      if (err) console.error('CP2 exec error', err);
      else console.log('CP2 node version', stdout.trim());
    });
  })();
  
  /* CP‑3: fork a module and message */
  (function CP3() {
    const child = fork(__filename, ['child']);
    if (process.argv[2] === 'child') {
      process.send('CP3 hello from child');
      process.exit();
    } else {
      child.on('message', msg => console.log(msg));
    }
  })();
  
  /* CP‑4: IPC with error handling */
  (function CP4() {
    const child = fork(__filename, ['fail'], { silent: true });
    if (process.argv[2] === 'fail') {
      throw new Error('CP4 intentional');
    } else {
      child.on('error', e => console.error('CP4 child error', e.message));
    }
  })();
  
  /* CP‑5: piping stdio */
  (function CP5() {
    const child = spawn('node',['-e',"console.error('err'); console.log('out')"], { stdio: ['ignore','pipe','pipe'] });
    child.stdout.on('data', d => console.log('CP5 out', d.toString()));
    child.stderr.on('data', d => console.error('CP5 err', d.toString()));
  })();
  
  
  /* SECTION MS — Microservices with Fastify & NestJS */
  
  /* MS‑1: Fastify basic GET */
  (function MS1() {
    const fastify = require('fastify')();
    fastify.get('/ping', async () => ({ pong: true }));
    fastify.listen(3000, err => { if (err) throw err; console.log('MS1 Fastify ping at 3000'); });
  })();
  
  /* MS‑2: Fastify POST with schema validation */
  (function MS2() {
    const fastify = require('fastify')();
    fastify.post('/echo', {
      schema: { body: { type:'object', properties:{msg:{type:'string'}}, required:['msg'] } }
    }, async (req) => ({ echoed: req.body.msg }));
    fastify.listen(3001, err => { if (err) throw err; console.log('MS2 Fastify echo'); });
  })();
  
  /* MS‑3: NestJS minimal app */
  (function MS3() {
    const { NestFactory } = require('@nestjs/core');
    const { Module, Controller, Get } = require('@nestjs/common');
    @Controller()
    class AppController {
      @Get('hello') hello() { return 'Hello Nest'; }
    }
    @Module({ imports:[], controllers:[AppController] })
    class AppModule {}
    (async () => {
      const app = await NestFactory.create(AppModule);
      await app.listen(3002);
      console.log('MS3 NestJS running');
    })();
  })();
  
  /* MS‑4: Fastify plugin usage */
  (function MS4() {
    const fastify = require('fastify')();
    fastify.register(require('fastify-cors'), { origin:'*' });
    fastify.get('/time', () => ({ time: Date.now() }));
    fastify.listen(3003);
  })();
  
  /* MS‑5: NestJS provider injection */
  (function MS5() {
    const { NestFactory } = require('@nestjs/core');
    const { Module, Injectable, Controller, Get } = require('@nestjs/common');
    @Injectable() class MyService { get() { return 'data'; } }
    @Controller() class MyCtrl { constructor(svc){this.svc=svc;} @Get('data') get(){return this.svc.get()} }
    @Module({ providers:[MyService], controllers:[MyCtrl] }) class MyMod {}
    (async()=> (await NestFactory.create(MyMod)).listen(3004))();
    console.log('MS5 NestJS DI ready');
  })();
  
  
  /* SECTION GQ — GraphQL & gRPC Servers */
  
  /* GQ‑1: Apollo GraphQL server */
  (function GQ1() {
    const { ApolloServer, gql } = require('apollo-server');
    const typeDefs = gql`type Query{hello:String}`;
    const resolvers = { Query:{hello:()=> 'world'} };
    const server = new ApolloServer({ typeDefs, resolvers });
    server.listen(4000).then(({ url }) => console.log(`GQ1 running at ${url}`));
  })();
  
  /* GQ‑2: GraphQL schema with resolver args */
  (function GQ2() {
    const { ApolloServer, gql } = require('apollo-server');
    const typeDefs = gql`
      type Query{add(x:Int!,y:Int!):Int}
    `;
    const resolvers = { Query:{ add:(_, {x,y}) => x+y }};
    new ApolloServer({ typeDefs, resolvers }).listen(4001);
  })();
  
  /* GQ‑3: GraphQL subscription using Apollo */
  (function GQ3() {
    const { ApolloServer, gql, PubSub } = require('apollo-server');
    const pubsub = new PubSub();
    const typeDefs = gql`
      type Query{_:Boolean}
      type Subscription{time:String}
    `;
    const resolvers = {
      Subscription:{ time:{ subscribe:()=> { setInterval(()=>pubsub.publish('TICK',{time:new Date().toISOString()}),1000); return pubsub.asyncIterator('TICK'); } } }
    };
    new ApolloServer({ typeDefs, resolvers }).listen(4002);
  })();
  
  /* GQ‑4: gRPC server */
  (function GQ4() {
    const grpc = require('@grpc/grpc-js');
    const protoLoader = require('@grpc/proto-loader');
    const packageDef = protoLoader.loadSync(__dirname+'/service.proto', {});
    const pkg = grpc.loadPackageDefinition(packageDef).service;
    const server = new grpc.Server();
    server.addService(pkg.MyService.service, { ping: (call, cb) => cb(null, { msg: 'pong' }) });
    server.bindAsync('0.0.0.0:50051', grpc.ServerCredentials.createInsecure(), () => server.start());
    console.log('GQ4 gRPC listening 50051');
  })();
  
  /* GQ‑5: gRPC client call */
  (function GQ5() {
    const grpc = require('@grpc/grpc-js');
    const protoLoader = require('@grpc/proto-loader');
    const def = protoLoader.loadSync(__dirname+'/service.proto', {});
    const Client = grpc.loadPackageDefinition(def).service.MyService;
    const client = new Client('localhost:50051', grpc.credentials.createInsecure());
    client.ping({}, (err, res) => {
      if (err) console.error('GQ5 error', err);
      else console.log('GQ5 response', res.msg);
    });
  })();