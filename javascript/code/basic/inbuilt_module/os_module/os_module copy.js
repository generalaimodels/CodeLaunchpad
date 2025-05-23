/***************************************************************************************************
 *  Node.js Core Module : `os`
 *  Author  : <Your Name Here>
 *  File    : os-tour.js   (run with `node os-tour.js`)
 *
 *  10 hands-on, console-driven examples covering EVERY public export of the `os` module—common to
 *  exotic.  Each snippet is self-contained, prints a separator, and leaves no side-effects.  Approx.
 *  outputs are shown in comments (values will differ by host machine).
 *
 *  Node ≥ 18.x  (for os.machine / os.availableParallelism)    – older versions still work minus
 *                                                              those two APIs.
 ***************************************************************************************************/
'use strict';
const os = require('os');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Pretty sequential runner
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const CASES = [];
function ex(title, fn) { CASES.push({ title, fn }); }
(async () => {
  for (const { title, fn } of CASES) {
    console.log('\n' + '═'.repeat(90));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(90));
    await fn();
  }
})();

/***************************************************************************************************
 * 1)  Identity – platform(), type(), arch(), release(), version(), machine()
 **************************************************************************************************/
ex('1) OS identity & kernel info', () => {
  console.log('type()      →', os.type());        // Linux / Darwin / Windows_NT
  console.log('platform()  →', os.platform());    // linux / darwin / win32
  console.log('arch()      →', os.arch());        // x64 / arm64 / ia32
  console.log('release()   →', os.release());     // 6.2.0-39-generic …
  console.log('version()   →', os.version());     // #53~22.04.1-Ubuntu SMP …
  if (os.machine) console.log('machine()   →', os.machine()); // x86_64 / armv7l
  /* Expected output (sample):
     type()      → Linux
     platform()  → linux
     arch()      → x64
     release()   → 5.15.0-76-generic
     version()   → #83-Ubuntu SMP Fri…
     machine()   → x86_64
  */
});

/***************************************************************************************************
 * 2)  CPU & load – cpus(), availableParallelism(), loadavg(), uptime()
 **************************************************************************************************/
ex('2) CPU topology & system load', () => {
  const cores = os.cpus();
  console.log('CPU count (cpus().length) →', cores.length);
  console.log('Model of core[0]          →', cores[0].model);
  console.log('availableParallelism()    →', os.availableParallelism?.());
  console.log('loadavg() [1m,5m,15m]     →', os.loadavg());
  console.log('uptime() (sec)            →', os.uptime());
  /* Expected output:
     CPU count (cpus().length) → 12
     Model of core[0]          → Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
     availableParallelism()    → 12
     loadavg() [1m,5m,15m]     → [ 0.43, 0.35, 0.31 ]
     uptime() (sec)            → 53240
  */
});

/***************************************************************************************************
 * 3)  Memory – totalmem(), freemem()  (plus % free)
 **************************************************************************************************/
ex('3) Memory statistics', () => {
  const toMB = (b) => Math.round(b / 1048576) + ' MB';
  const total = os.totalmem();
  const free  = os.freemem();
  console.log('totalmem() →', toMB(total));
  console.log('freemem()  →', toMB(free));
  console.log('% free     →', (free / total * 100).toFixed(1) + '%');
  /* Expected output:
     totalmem() → 15924 MB
     freemem()  → 9241 MB
     % free     → 58.0%
  */
});

/***************************************************************************************************
 * 4)  Network – networkInterfaces()
 **************************************************************************************************/
ex('4) Network interfaces & IPv4 list', () => {
  const nets = os.networkInterfaces();
  for (const [name, addrs] of Object.entries(nets)) {
    addrs.filter(a => a.family === 'IPv4').forEach(a =>
      console.log(`${name} → ${a.address} (${a.internal ? 'internal' : 'external'})`));
  }
  /* Expected output (sample):
     lo         → 127.0.0.1 (internal)
     enp0s31f6  → 192.168.1.10 (external)
  */
});

/***************************************************************************************************
 * 5)  Paths & newlines – homedir(), tmpdir(), devNull, EOL
 **************************************************************************************************/
ex('5) Home, tmp, devNull & EOL', () => {
  console.log('homedir()  →', os.homedir());      // /home/user
  console.log('tmpdir()   →', os.tmpdir());       // /tmp
  console.log('devNull    →', os.devNull);        // /dev/null or \\.\NUL
  console.log('EOL byte(s)→', JSON.stringify(os.EOL)); // "\n" or "\r\n"
  /* Expected output:
     homedir()  → /home/alice
     tmpdir()   → /tmp
     devNull    → /dev/null
     EOL byte(s)→ "\n"
  */
});

/***************************************************************************************************
 * 6)  User information – userInfo()
 **************************************************************************************************/
ex('6) Current user info', () => {
  const ui = os.userInfo();
  console.log(ui); // { uid: 1000, gid: 1000, username: 'alice', homedir: '/home/alice', shell: '/bin/bash' }
  /* Expected output (structure):
     {
       uid: 1000,
       gid: 1000,
       username: 'alice',
       homedir: '/home/alice',
       shell: '/usr/bin/zsh'
     }
  */
});

/***************************************************************************************************
 * 7)  Endianness & constants
 **************************************************************************************************/
ex('7) Endianness and OS constants', () => {
  console.log('endianness() →', os.endianness()); // LE / BE
  console.log('signals.SIGINT value →', os.constants.signals.SIGINT);
  console.log('errno.EADDRINUSE     →', os.constants.errno.EADDRINUSE);
  /* Expected output:
     endianness() → LE
     signals.SIGINT value → 2
     errno.EADDRINUSE     → -98
  */
});

/***************************************************************************************************
 * 8)  Process priority – getPriority() / setPriority()
 *      (May require privileges; falls back gracefully.)
 **************************************************************************************************/
ex('8) getPriority & setPriority', () => {
  const pid = process.pid;
  const before = os.getPriority(pid);
  try {
    os.setPriority(pid, before + 1);                  // lower priority by 1
  } catch (e) {
    console.log('setPriority failed →', e.code);      // likely EPERM on Windows / unprivileged
  }
  const after = os.getPriority(pid);
  console.log(`Priority was ${before} → now ${after}`);
  /* Expected output (non-priv. user):
     setPriority failed → EPERM
     Priority was 0 → now 0
     (Root could succeed, so numbers would differ.)
  */
});

/***************************************************************************************************
 * 9)  Using constants.signals to handle termination gracefully
 **************************************************************************************************/
ex('9) Handling SIGTERM via os.constants.signals', async () => {
  const SIG = os.constants.signals.SIGTERM;
  console.log('SIGTERM numeric value →', SIG);
  // Quick demo: schedule self-signal then await.
  setTimeout(() => process.kill(process.pid, SIG), 200);
  await new Promise(res => process.once('SIGTERM', () => {
    console.log('SIGTERM caught – graceful shutdown ✓');
    res();
  }));
  /* Expected output:
     SIGTERM numeric value → 15
     SIGTERM caught – graceful shutdown ✓
  */
});

/***************************************************************************************************
 * 10)  Misc curiosities – hrtime.bigint(), getPriority(other-pid) w/ try/catch
 **************************************************************************************************/
ex('10) hi-res timers & remote priority probe', () => {
  const t0 = process.hrtime.bigint();
  // Busy op
  for (let i = 0; i < 1e6; i++) Math.sqrt(i);
  const nsElapsed = process.hrtime.bigint() - t0;
  console.log('Computation took →', Number(nsElapsed / 1_000_000n), 'ms');

  // getPriority for PID 1 (likely systemd / launchd)
  try {
    console.log('Priority of PID 1 →', os.getPriority(1));
  } catch {
    console.log('Cannot query priority of PID 1 (permission denied)');
  }
  /* Expected output:
     Computation took → 15 ms
     Priority of PID 1 → 0   (or message if denied)
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/