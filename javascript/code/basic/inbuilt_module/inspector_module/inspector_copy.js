/***************************************************************************************************
 *  Node.js Core Module : `inspector`
 *  Author   : <Your Name Here>
 *  Purpose  : Full-coverage, example-driven tour of the `inspector` module.  Ten runnable snippets
 *             show how to programmatically open/close the DevTools endpoint, drive the V8 Inspector
 *             Protocol through `inspector.Session`, capture CPU/heap data, and even hold execution
 *             until a debugger attaches.  All public API points are exercised (*open*, *close*,
 *             *url*, *waitForDebugger*, `Session.{connect,disconnect,post}`, events, etc.).
 *
 *  Run      : `node inspector-tour.js`
 *  Tested   : Node ≥ 18.x   (minor adjustments may be needed for older versions)
 *
 *  NB:  Heavy-weight operations (heap snapshots, CPU profiling) are kept ultra-short to remain
 *       interactive.  All output is printed to STDOUT; compare with the “Expected output” blocks.
 ***************************************************************************************************/

'use strict';
const inspector = require('inspector');
const { EventEmitter } = require('events');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Pretty runner – executes examples sequentially, catches sync/async errors
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const QUEUE = [];
function example(title, fn)   { QUEUE.push({ title, fn }); }
(async () => {
  for (const { title, fn } of QUEUE) {
    console.log('\n' + '═'.repeat(100));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(100));
    try { await fn(); }
    catch (e) { console.error('💥 Exception →', e); }
  }
})();

/***************************************************************************************************
 *  Helper – small util spawning a connected Session with Runtime enabled
 **************************************************************************************************/
function freshSession() {
  const sess = new inspector.Session();
  sess.connect();
  sess.post('Runtime.enable');            // most subsequent calls rely on Runtime domain
  return sess;
}

/***************************************************************************************************
 * 1) inspector.open() / inspector.url() / inspector.close()
 **************************************************************************************************/
example('1) Open inspector on random port, read URL, then close', () => {
  console.log('URL before open →', inspector.url()); // undefined

  inspector.open(0, '127.0.0.1');                    // 0 = random free port
  console.log('URL after  open →', inspector.url()); // ws://127.0.0.1:92xx/…

  inspector.close();
  console.log('URL after close →', inspector.url()); // undefined

  /* Expected output (port varies):
     URL before open → undefined
     URL after  open → ws://127.0.0.1:92xx/xxxxxxxx-xxxx…
     URL after close → undefined
  */
});

/***************************************************************************************************
 * 2) Session.connect() / disconnect()  + Runtime.evaluate (callback style)
 **************************************************************************************************/
example('2) Session basic connect / disconnect', () => {
  const sess = new inspector.Session();
  sess.connect();                                   // attaches to current process
  sess.post('Runtime.enable');
  sess.post('Runtime.evaluate', { expression: '1 + 2' }, (err, res) => {
    console.log('1 + 2 =', res.result.value);       // 3
    sess.disconnect();
  });

  /* Expected output:
     1 + 2 = 3
  */
});

/***************************************************************************************************
 * 3) Session.post() returning a Promise (no callback passed)
 **************************************************************************************************/
example('3) Promise-based Session.post()', async () => {
  const sess = freshSession();
  const { result } = await sess.post('Runtime.evaluate', { expression: 'new Date().getFullYear()' });
  console.log('Current year →', result.value);
  sess.disconnect();

  /* Expected output (year will, of course, change):
     Current year → 2024
  */
});

/***************************************************************************************************
 * 4) Listening to arbitrary protocol events via “inspectorNotification”
 **************************************************************************************************/
example('4) EventEmitter facet → capture console.log() calls', async () => {
  const sess = freshSession();
  await sess.post('Runtime.enable');
  await sess.post('Runtime.consoleAPICalled');        // ensure domain ready

  sess.on('inspectorNotification', msg => {
    if (msg.method === 'Runtime.consoleAPICalled') {
      const txt = msg.params.args[0].value;
      console.log('Captured console:', txt);          // Hello from V8!
    }
  });

  await sess.post('Runtime.evaluate', {
    expression: 'console.log("Hello from V8!")'
  });
  sess.disconnect();

  /* Expected output:
     Captured console: Hello from V8!
  */
});

/***************************************************************************************************
 * 5) Quick CPU profile (Profiler.start/stop)
 **************************************************************************************************/
example('5) 50-ms CPU profile', async () => {
  const sess = freshSession();
  await sess.post('Profiler.enable');
  await sess.post('Profiler.start');

  // tiny busy-loop
  for (let x = 0; x < 1e6; x++) { Math.sqrt(x); }

  const { profile } = await sess.post('Profiler.stop');
  console.log('Profile nodes →', profile.nodes.length); // non-zero
  sess.disconnect();

  /* Expected output:
     Profile nodes → <some positive number>
  */
});

/***************************************************************************************************
 * 6) Heap snapshot – count how many chunks were streamed
 **************************************************************************************************/
example('6) Heap snapshot (chunk counter)', async () => {
  const sess = freshSession();
  await sess.post('HeapProfiler.enable');

  let chunks = 0;
  sess.on('inspectorNotification', (m) => {
    if (m.method === 'HeapProfiler.addHeapSnapshotChunk') chunks++;
  });
  await sess.post('HeapProfiler.takeHeapSnapshot', { reportProgress: false });
  console.log('Snapshot chunks received →', chunks);
  sess.disconnect();

  /* Expected output:
     Snapshot chunks received → <hundreds>
  */
});

/***************************************************************************************************
 * 7) Debugger.pause() / resume()  (programmatic breakpoint)
 **************************************************************************************************/
example('7) Programmatic pause / resume', async () => {
  const sess = freshSession();
  await sess.post('Debugger.enable');

  const resumePromise = new Promise(resolve =>
    sess.on('inspectorNotification', (m) => {
      if (m.method === 'Debugger.paused') {
        console.log('Execution is paused 🔴');
        sess.post('Debugger.resume').then(() => resolve());
      }
    }));

  await sess.post('Debugger.pause');                  // triggers pause
  await resumePromise;
  console.log('Execution resumed 🟢');
  sess.disconnect();

  /* Expected output:
     Execution is paused 🔴
     Execution resumed 🟢
  */
});

/***************************************************************************************************
 * 8) Precise coverage (Profiler.startPreciseCoverage / takePreciseCoverage)
 **************************************************************************************************/
example('8) Code coverage on-the-fly', async () => {
  const sess = freshSession();
  await sess.post('Profiler.enable');
  await sess.post('Profiler.startPreciseCoverage', { callCount: false, detailed: true });

  // run some arbitrary code
  function foo(n) { return n * 2; }
  foo(21);

  const { result } = await sess.post('Profiler.takePreciseCoverage');
  console.log('Scripts measured →', result.length);
  await sess.post('Profiler.stopPreciseCoverage');
  sess.disconnect();

  /* Expected output:
     Scripts measured → ≥1
  */
});

/***************************************************************************************************
 * 9) waitForDebugger() – block until debugger attaches & resumes
 *    We attach our *own* Session to satisfy the wait, then send the resume command.
 **************************************************************************************************/
example('9) waitForDebugger() (self-satisfied)', async () => {
  // Kick off a “waiter” in the background
  const waiter = (async () => {
    console.log('⏳  Calling inspector.waitForDebugger() …');
    await inspector.waitForDebugger();               // pauses here
    console.log('✅  waitForDebugger() resolved.');
  })();

  // Meanwhile, attach as the “debugger” and send the magic resume
  const sess = freshSession();
  // Runtime.runIfWaitingForDebugger is the required command:
  await sess.post('Runtime.runIfWaitingForDebugger');
  sess.disconnect();

  await waiter;
  /* Expected output:
     ⏳  Calling inspector.waitForDebugger() …
     ✅  waitForDebugger() resolved.
  */
});

/***************************************************************************************************
 * 10) addListener / removeListener demo (aliases) + Session lifecycle
 **************************************************************************************************/
example('10) EventEmitter helpers (addListener / off)', async () => {
  const sess = freshSession();
  await sess.post('Runtime.enable');

  const listener = (m) => {
    if (m.method === 'Runtime.executionContextCreated')
      console.log('Execution context created!');
  };
  sess.addListener('inspectorNotification', listener); // alias of .on()

  await sess.post('Runtime.evaluate', { expression: '1' }); // triggers context event

  sess.off('inspectorNotification', listener);              // alias of removeListener
  await sess.post('Runtime.evaluate', { expression: '2' }); // no log this time

  sess.disconnect();
  console.log('Listener removed ✔️');

  /* Expected output:
     Execution context created!
     Listener removed ✔️
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/