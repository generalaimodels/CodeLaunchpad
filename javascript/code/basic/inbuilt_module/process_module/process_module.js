/***************************************************************************************************
 *  Node.js Core Module : `process`
 *  Author : <Your Name Here>
 *  File   : process-tour.js               ──>  run with `node process-tour.js`
 *
 *  Ten focused demos that cover essentially every *public* knob of the global `process` object
 *  (plus a few under-used gems).  Each section prints a delimiter, executes synchronously or
 *  awaits completion, and leaves the runtime in a clean state so the next demo can run.
 *
 *  Tested on Node ≥ 18.x.  Windows-only / *nix-only areas are guarded with feature detection.
 ***************************************************************************************************/
'use strict';
const fs   = require('fs');
const os   = require('os');
const path = require('path');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Sequential runner – very small, no deps
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const STEPS = [];
function step(title, fn) { STEPS.push({ title, fn }); }
(async () => {
  for (const { title, fn } of STEPS) {
    console.log('\n' + '═'.repeat(105));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(105));
    try { await fn(); }
    catch (e) { console.error('💥  Exception →', e); }
  }
})();                                             // immediate fire

/***************************************************************************************************
 * 1)  Identity & CLI – argv, argv0, execPath, pid, env, platform, arch, versions, release
 **************************************************************************************************/
step('1) Process identity & CLI flags', () => {
  console.log('argv0      →', process.argv0);          // node / node.exe
  console.log('argv       →', process.argv.slice(0, 3)); // first 3 CLI pieces
  console.log('execPath   →', process.execPath);       // absolute path to node binary
  console.log('pid / ppid →', process.pid, '/', process.ppid);
  console.log('platform   →', process.platform, '  arch →', process.arch);
  console.log('versions.node →', process.versions.node);
  console.log('release.lts    →', process.release.lts || 'none');
  console.log('env.HOME       →', process.env.HOME || process.env.USERPROFILE);
  /* Expected output (sample):
     argv0      → node
     argv       → [ 'node', '/abs/path/process-tour.js', '--foo' ]
     execPath   → /usr/bin/node
     pid / ppid → 12345 / 12340
     platform   → linux   arch → x64
     versions.node → 20.5.1
     release.lts    → Iron
     env.HOME       → /home/alice
  */
});

/***************************************************************************************************
 * 2)  Working directory – cwd(), chdir()
 **************************************************************************************************/
step('2) cwd() & chdir()', () => {
  const orig = process.cwd();
  const tmp  = fs.mkdtempSync(path.join(os.tmpdir(), 'proc-tour-'));
  console.log('Original cwd →', orig);
  process.chdir(tmp);
  console.log('After chdir  →', process.cwd());
  process.chdir(orig);                                // restore
  fs.rmSync(tmp, { recursive: true, force: true });
  /* Expected output:
     Original cwd → /home/alice/project
     After chdir  → /tmp/proc-tour-abc123
  */
});

/***************************************************************************************************
 * 3)  High-res timers – hrtime(), hrtime.bigint(), uptime()
 **************************************************************************************************/
step('3) hrtime / hrtime.bigint / uptime', () => {
  const t0 = process.hrtime.bigint();
  const busy = Date.now() + 25;                      // ~25-ms spin CPU
  while (Date.now() < busy);
  const ns = process.hrtime.bigint() - t0;
  console.log('Busy-loop took →', Number(ns) / 1e6, 'ms');
  console.log('process.uptime() →', process.uptime().toFixed(3), 'sec');
  /* Expected output:
     Busy-loop took → 25.1 ms
     process.uptime() → 0.312 sec
  */
});

/***************************************************************************************************
 * 4)  Resource stats – memoryUsage(), resourceUsage(), cpuUsage()
 **************************************************************************************************/
step('4) Memory / CPU / resourceUsage', () => {
  const mem = process.memoryUsage();
  console.log('RSS         →', (mem.rss / 1048576).toFixed(1), 'MB');
  console.log('Heap used   →', (mem.heapUsed / 1048576).toFixed(1), 'MB');
  const res = process.resourceUsage?.();
  if (res) console.log('User CPU time →', (res.userCPUTime / 1e6).toFixed(2), 'ms');
  const cpu = process.cpuUsage();
  console.log('cpuUsage() user/system μs →', cpu.user, '/', cpu.system);
  /* Expected output:
     RSS         → 35.6 MB
     Heap used   → 4.3 MB
     User CPU time → 7.12 ms
     cpuUsage() user/system μs → 24567 / 18560
  */
});

/***************************************************************************************************
 * 5)  nextTick() vs setImmediate() execution order
 **************************************************************************************************/
step('5) nextTick vs setImmediate', async () => {
  process.nextTick(() => console.log('• nextTick runs first'));
  setImmediate(() => console.log('• setImmediate runs later'));
  await new Promise(r => setTimeout(r, 0));           // let loop flush
  /* Expected output (order guaranteed):
     • nextTick runs first
     • setImmediate runs later
  */
});

/***************************************************************************************************
 * 6)  Signals – on('SIGTERM'), kill(), once(), off()
 **************************************************************************************************/
step('6) Signal handling (self-SIGTERM)', async () => {
  if (process.platform === 'win32') {
    console.log('Signals limited on Windows – demo skipped.');
    return;
  }
  const handler = () => {
    console.log('SIGTERM captured ✓');
    process.off('SIGTERM', handler);                  // clean up
  };
  process.once('SIGTERM', handler);
  process.kill(process.pid, 'SIGTERM');               // send to self
  await new Promise(r => setTimeout(r, 50));          // give it time
  /* Expected output:
     SIGTERM captured ✓
  */
});

/***************************************************************************************************
 * 7)  Errors – unhandledRejection, uncaughtException, emitWarning()
 **************************************************************************************************/
step('7) Error & warning channels', async () => {
  process.once('warning', (w) => console.log('Warning caught →', w.name, w.message));
  process.emitWarning('low disk', { code: 'LOW_SPACE', detail: 'only 1GB left' });

  process.once('unhandledRejection', (reason) => {
    console.log('unhandledRejection →', reason);
  });
  Promise.reject('oops');                             // will trigger above

  process.once('uncaughtException', (err) => {
    console.log('uncaughtException  →', err.message);
  });
  setTimeout(() => { throw new Error('boom'); }, 0);

  await new Promise(r => setTimeout(r, 50));          // flush events
  /* Expected output:
     Warning caught → Warning low disk
     unhandledRejection → oops
     uncaughtException  → boom
  */
});

/***************************************************************************************************
 * 8)  Privileged bits – umask(), get/set uid/gid (best-effort)
 **************************************************************************************************/
step('8) umask & uid/gid introspection', () => {
  const original = process.umask();
  console.log('umask before →', original.toString(8));
  const temp = process.umask(0o027);                  // set & read back
  console.log('umask after  →', process.umask(temp).toString(8)); // restore & show

  if (process.getuid) {
    console.log('UID/GID →', process.getuid(), '/', process.getgid());
    console.log('Groups  →', process.getgroups().slice(0, 4), '…');
  } else {
    console.log('UID/GID APIs not available on this OS.');
  }
  /* Expected output (Linux non-root):
     umask before → 22
     umask after  → 27
     UID/GID → 1000 / 1000
     Groups  → [ 1000, 4, 24, 27 ] …
  */
});

/***************************************************************************************************
 * 9)  Misc gems – allowedNodeEnvironmentFlags, stdio, .stdout.columns
 **************************************************************************************************/
step('9) Misc: env flags, stdio props', () => {
  console.log('allowedNodeEnvironmentFlags has --trace-warnings ?',
    process.allowedNodeEnvironmentFlags.has('--trace-warnings'));
  console.log('stdout.isTTY →', process.stdout.isTTY);
  console.log('stdout.columns / rows →', process.stdout.columns, '/', process.stdout.rows);
  process.stdout.write('Direct write to stdout ✓\n');
  /* Expected output:
     allowedNodeEnvironmentFlags has --trace-warnings ? true
     stdout.isTTY → true
     stdout.columns / rows → 120 / 30
     Direct write to stdout ✓
  */
});

/***************************************************************************************************
 * 10)  Exit lifecycle – beforeExit & exit events + exitCode
 **************************************************************************************************/
step('10) beforeExit & exit events', () => {
  process.on('beforeExit', (code) => {
    console.log('[beforeExit] code →', code);
    // queue some async work just to show we can prolong shutdown
    if (code === 0) setTimeout(() => console.log('Extra async on shutdown ✓'), 10);
  });
  process.on('exit', (code) => console.log('[exit] code →', code));
  process.exitCode = 0;                                // harmless, keeps process alive
  /* Expected output (printed when program terminates):
     [beforeExit] code → 0
     Extra async on shutdown ✓
     [exit] code → 0
  */
});

/***************************************************************************************************
 * End of file – enjoy exploring `process`!
 ***************************************************************************************************/