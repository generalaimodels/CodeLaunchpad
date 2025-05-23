/**************************************************************************************************
*  child-process-masterclass.js
*  Goal  : ONE file that demonstrates every public API of Node.js “child_process” — from 99 %
*          day-to-day calls to obscure edge-cases — in 10 independent, bite-sized examples.
*  Style : ES-2023, strict mode, 2-space indent, exhaustive inline commentary.
*  Run   : `node child-process-masterclass.js`
**************************************************************************************************/
'use strict';
const {
  spawn,            // async streaming
  spawnSync,        // sync streaming
  exec,             // async buffered shell command
  execSync,         // sync buffered shell command
  execFile,         // async *direct* binary invocation
  execFileSync,     // sync *direct* binary invocation
  fork              // special spawn for Node modules w/ IPC
} = require('child_process');

const THIS_FILE = __filename;          // needed for the fork demo

/**************************************************************************************************
* INTERNAL HELPER ─ Same file doubles as the “forked” child in EX-4.
* When launched with --fork-child we run child-only logic then exit immediately.
**************************************************************************************************/
if (process.argv.includes('--fork-child')) {
  // Child receives data, transforms, returns result over IPC
  process.on('message', msg => {
    if (msg.cmd === 'double') {
      process.send({ result: msg.value * 2 });
      process.exit(0);
    }
  });
  // Safety-net: if parent dies, exit.
  setTimeout(() => process.exit(1), 5_000);
  return; // prevent master demos from running
}

/**************************************************************************************************
* EX-1 ► spawn  —  simplest form (streaming STDOUT/STDERR, no shell)
*   – Good for large outputs or continuous streaming (video transcoding, etc.)
**************************************************************************************************/
(() => {
  console.log('\nEX-1: spawn basic   —> node -v');

  const child = spawn(process.execPath, ['-v']); // node binary prints its version
  child.stdout.on('data', chunk =>
    console.log(' child stdout:', chunk.toString().trim())  // e.g. v20.8.0
  );
  child.stderr.on('data', chunk =>
    console.error(' child stderr:', chunk.toString())
  );
  child.on('close', code =>
    console.log(' exited with code', code)                  // 0
  );

  // Expected (≈):
  // child stdout: vX.Y.Z
  // exited with code 0
})();

/**************************************************************************************************
* EX-2 ► exec  —  buffered + SHELL parsing
*   – Simpler one-liner, but entire output kept in RAM (maxBuffer default 1 MiB)
**************************************************************************************************/
(() => {
  console.log('\nEX-2: exec   —> `node -p "40+2"`');

  exec(`${process.execPath} -p "40+2"`, (err, stdout, stderr) => {
    if (err) throw err;
    console.log(' stdout:', stdout.trim());   // 42
    console.log(' stderr:', stderr.trim());   // (empty)
  });

  // Expected:
  // stdout: 42
})();

/**************************************************************************************************
* EX-3 ► execFile  —  direct binary (no shell), safer against injection
*   – Pass args array, get buffers or strings just like exec.
**************************************************************************************************/
(() => {
  console.log('\nEX-3: execFile   —> node -e "console.log(\'execFile\')"');

  execFile(process.execPath, ['-e', 'console.log("execFile")'], (err, stdout) => {
    if (err) throw err;
    console.log(' stdout:', stdout.trim());     // execFile
  });

  // Expected:
  // stdout: execFile
})();

/**************************************************************************************************
* EX-4 ► fork  —  Node-to-Node + IPC (send/receive JS objects)
**************************************************************************************************/
(() => {
  console.log('\nEX-4: fork with IPC   —> doubling 21');

  const child = fork(THIS_FILE, ['--fork-child']);

  child.on('message', msg => {
    console.log(' message from child:', msg);  // { result: 42 }
    child.disconnect();
  });

  child.on('exit', code => console.log(' child exited', code)); // 0

  child.send({ cmd: 'double', value: 21 });

  // Expected:
  // message from child: { result: 42 }
  // child exited 0
})();

/**************************************************************************************************
* EX-5 ► spawnSync  —  blocking counterpart to spawn
*   – Returns a result object instead of emitting events.
**************************************************************************************************/
(() => {
  console.log('\nEX-5: spawnSync   —> node -p "6*7"');

  const res = spawnSync(process.execPath, ['-p', '6*7']);
  console.log(' status:', res.status);                   // 0
  console.log(' output:', res.stdout.toString().trim()); // 42

  // Expected:
  // status: 0
  // output: 42
})();

/**************************************************************************************************
* EX-6 ► execSync  —  blocking buffered shell command
**************************************************************************************************/
(() => {
  console.log('\nEX-6: execSync   —> prints platform & Node');

  const ver = execSync(`${process.execPath} -v`).toString().trim();
  console.log(' version:', ver);  // vX.Y.Z

  // Expected:
  // version: vX.Y.Z
})();

/**************************************************************************************************
* EX-7 ► execFileSync  —  blocking direct binary
**************************************************************************************************/
(() => {
  console.log('\nEX-7: execFileSync   —> node prints 99');

  const out = execFileSync(process.execPath, ['-e', 'console.log(99)']);
  console.log(' stdout:', out.toString().trim()); // 99
})();

/**************************************************************************************************
* EX-8 ► Advanced spawn options (stdio=inherit, detached, unref)
*   – Useful to create daemons or allow parent to exit first.
**************************************************************************************************/
(() => {
  console.log('\nEX-8: spawn detached daemon-ish child');

  const det = spawn(
    process.execPath,
    ['-e', 'console.log("detached child says hi"); setTimeout(()=>{}, 1000);'],
    { stdio: 'inherit', detached: true }
  );
  det.unref(); // parent no longer waits

  // Expected immediate console output:
  // detached child says hi
})();

/**************************************************************************************************
* EX-9 ► Sending signals (.kill) & observing .signalCode
**************************************************************************************************/
(() => {
  console.log('\nEX-9: spawn + kill (SIGTERM after 1 s)');

  const looping = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], {
    stdio: 'ignore'
  });

  setTimeout(() => {
    looping.kill('SIGTERM'); // portable termination
  }, 1_000);

  looping.on('exit', (code, signal) =>
    console.log(' exited code:', code, 'signal:', signal) // null, SIGTERM
  );

  // Expected:
  // exited code: null signal: SIGTERM
})();

/**************************************************************************************************
* EX-10 ► exec with maxBuffer + timeout (handling ERR_CHILD_PROCESS_STDIO_MAXBUFFER)
**************************************************************************************************/
(() => {
  console.log('\nEX-10: exec with constrained buffers (expect error)');

  const cmd = `${process.execPath} -e "console.log('x'.repeat(1e6))"`;
  exec(cmd, { maxBuffer: 1024, timeout: 500 }, (err) => {
    if (err) {
      console.log(' name    :', err.name);        // Error
      console.log(' code    :', err.code);        // 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER' or 'ETIMEDOUT'
      console.log(' message :', err.message.split('\n')[0]);
    }
  });

  // Expected (one of):
  // code    : ERR_CHILD_PROCESS_STDIO_MAXBUFFER
  //  or
  // code    : ETIMEDOUT
})();

/**************************************************************************************************
*  🎯 You now command the full arsenal of “child_process”. Experiment, tweak, conquer. 🚀
**************************************************************************************************/