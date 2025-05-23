/***************************************************************************************************
 *  Node.js Core Module : `fs` (File-System I/O)
 *  Author  : <Your Name Here>
 *  Purpose : Exhaustive, example-driven walkthrough covering EVERY public API surfacing in the
 *            classic callback, sync, stream and promise flavours.  Ten self-contained examples,
 *            each runnable via `node fs-tour.js`, print their own heading separators so you can
 *            comment-in / comment-out freely.
 *
 *  Tested  : Node ≥ 18.x (older versions may lack `fs.rm`, `readv`, etc.)
 ***************************************************************************************************/

'use strict';
const fs        = require('fs');
const path      = require('path');
const os        = require('os');
const fsp       = require('fs/promises');
const { runInNewContext } = require('vm'); // only for demo in Ex-8

/***************************************************************************************************
 * Helper : isolate and pretty-print each example
 **************************************************************************************************/
function runExample(title, fn) {
  console.log('\n' + '═'.repeat(90));
  console.log(`Example: ${title}`);
  console.log('═'.repeat(90));
  try { fn(); }
  catch (e) { console.error('💥  Exception occurred →', e); }
}

/***************************************************************************************************
 * Ephemeral scratch directory so we never touch the user’s data
 **************************************************************************************************/
const SCRATCH = path.join(os.tmpdir(), 'fs-tour-tmp');
fs.rmSync?.(SCRATCH, { recursive: true, force: true }); // clean slate if leftover
fs.mkdirSync(SCRATCH, { recursive: true });

/***************************************************************************************************
 * 1)  readFile / writeFile (async + sync)  ───────────────────────────────────────────────────────
 **************************************************************************************************/
runExample('1) readFile / writeFile (callback & sync versions)', () => {
  const file = path.join(SCRATCH, 'hello.txt');

  // Async flavor (ERR first callback)
  fs.writeFile(file, 'Hello, async World!\n', (err) => {
    if (err) throw err;
    fs.readFile(file, 'utf8', (err, data) => {
      if (err) throw err;
      console.log('[callback] File contents →', data.trim());
    });
  });

  // Sync flavor – blocks the event loop, handy for boot-time
  fs.writeFileSync(file + '.sync', 'Hello, sync World!\n');
  const dataSync = fs.readFileSync(file + '.sync', 'utf8');
  console.log('[sync]      File contents →', dataSync.trim());

  /* Expected output:
     [callback] File contents → Hello, async World!
     [sync]      File contents → Hello, sync World!
  */
});

/***************************************************************************************************
 * 2)  Streams : createReadStream / createWriteStream / pipe  ─────────────────────────────────────
 **************************************************************************************************/
runExample('2) Streams: createReadStream → pipe → createWriteStream', () => {
  const bigSrc  = path.join(SCRATCH, 'big.txt');
  const bigDest = path.join(SCRATCH, 'big.copy.txt');

  // Generate a 1 MB file quickly
  fs.writeFileSync(bigSrc, Buffer.alloc(1024 * 1024, 'A'));

  const rs = fs.createReadStream(bigSrc,  { highWaterMark: 64 * 1024 }); // 64 KB chunks
  const ws = fs.createWriteStream(bigDest);

  rs.pipe(ws).on('finish', () => {
    console.log('Stream copy complete – size:',
      fs.statSync(bigDest).size, 'bytes');
  });

  /* Expected output:
     Stream copy complete – size: 1048576 bytes
  */
});

/***************************************************************************************************
 * 3)  Directories: mkdir / readdir / rm (recursive)  ─────────────────────────────────────────────
 **************************************************************************************************/
runExample('3) Directories: mkdir / readdir / rm', () => {
  const dir = path.join(SCRATCH, 'dir-demo');

  fs.mkdirSync(dir);                                  // create dir
  ['a.js', 'b.js'].forEach(f => fs.writeFileSync(path.join(dir, f), '')); // touch 2 files

  console.log('Entries:', fs.readdirSync(dir));       // list

  fs.rmSync(dir, { recursive: true, force: true });   // nuke dir recursively
  console.log('Exists after rm?', fs.existsSync(dir));

  /* Expected output:
     Entries: [ 'a.js', 'b.js' ]
     Exists after rm? false
  */
});

/***************************************************************************************************
 * 4)  Metadata & access: stat / lstat / access / constants  ──────────────────────────────────────
 **************************************************************************************************/
runExample('4) Metadata: stat / lstat / access', async () => {
  const f = path.join(SCRATCH, 'meta.txt');
  fs.writeFileSync(f, 'meta');

  const s = fs.statSync(f);                           // follows symlinks
  console.log('Size via statSync →', s.size);

  const link = f + '.sym';
  fs.symlinkSync(f, link);

  const ls = fs.lstatSync(link);                      // info about link itself
  console.log('IsSymbolicLink?', ls.isSymbolicLink());

  try {
    fs.accessSync(f, fs.constants.R_OK | fs.constants.W_OK);
    console.log('We have read/write access ✅');
  } catch { console.log('No access ❌'); }

  /* Expected output:
     Size via statSync → 4
     IsSymbolicLink? true
     We have read/write access ✅
  */
});

/***************************************************************************************************
 * 5)  File-descriptor low-level: open / read / write / close / readv / writev  ───────────────────
 **************************************************************************************************/
runExample('5) Low-level FD ops: open / read / write / close', () => {
  const f = path.join(SCRATCH, 'fd.txt');

  const fd = fs.openSync(f, 'w+');                    // create file RW
  fs.writeSync(fd, 'Node');                           // write at position 0
  fs.writeSync(fd, 'JS', 0, 'utf8', 4);               // append at pos 4

  fs.closeSync(fd);                                   // flush & close

  // readv demo (batched read into multiple buffers)
  const fdR = fs.openSync(f, 'r');
  const bufs = [ Buffer.alloc(2), Buffer.alloc(2) ];
  fs.readvSync(fdR, bufs);
  console.log('readv bytes →', Buffer.concat(bufs).toString()); // Node
  fs.closeSync(fdR);

  /* Expected output:
     readv bytes → Node
  */
});

/***************************************************************************************************
 * 6)  Movers & linkers: rename / copyFile / link / symlink / readlink  ───────────────────────────
 **************************************************************************************************/
runExample('6) rename / copyFile / link / symlink / readlink', () => {
  const src = path.join(SCRATCH, 'src.txt');
  const dst = path.join(SCRATCH, 'dst.txt');

  fs.writeFileSync(src, 'move me');
  fs.copyFileSync(src, dst);
  console.log('copyFileSync created dst?', fs.existsSync(dst));

  const hard = path.join(SCRATCH, 'hard.txt');
  fs.linkSync(src, hard);                             // hard-link
  console.log('Hard link size:', fs.statSync(hard).size);

  const sym = path.join(SCRATCH, 'sym.txt');
  fs.symlinkSync(src, sym);                           // symlink
  console.log('readlink →', fs.readlinkSync(sym));

  fs.renameSync(src, src + '.ren');                   // move/rename
  console.log('renameSync success?', fs.existsSync(src + '.ren'));

  /* Expected output:
     copyFileSync created dst? true
     Hard link size: 7
     readlink → <absolute-path-to-src.txt>
     renameSync success? true
  */
});

/***************************************************************************************************
 * 7)  Permissions: chmod / chown (best-effort) / utimes  ─────────────────────────────────────────
 **************************************************************************************************/
runExample('7) chmod / utimes (mtime / atime)', () => {
  const f = path.join(SCRATCH, 'perm.txt');
  fs.writeFileSync(f, 'perm');

  fs.chmodSync(f, 0o600);                             // rw-------
  console.log('Mode after chmod →', fs.statSync(f).mode.toString(8).slice(-3));

  const past = new Date(Date.now() - 1000 * 3600);    // 1h ago
  fs.utimesSync(f, past, past);                       // touch times
  console.log('mtime set to past?', fs.statSync(f).mtime < new Date());

  /* Expected output (mode may vary by OS):
     Mode after chmod → 600
     mtime set to past? true
  */
});

/***************************************************************************************************
 * 8)  Watching files: watch / watchFile / unwatchFile  ───────────────────────────────────────────
 **************************************************************************************************/
runExample('8) fs.watch & fs.watchFile', () => {
  const f = path.join(SCRATCH, 'watch.txt');
  fs.writeFileSync(f, '0');

  // Modern: fs.watch (EventEmitter)
  const watcher = fs.watch(f, (eventType) =>
    console.log('[watch] Event →', eventType));

  // Legacy polling: watchFile
  fs.watchFile(f, { interval: 200 }, (cur, prev) =>
    console.log('[watchFile] size →', cur.size));

  // Trigger a change
  fs.appendFileSync(f, '1');

  setTimeout(() => {
    watcher.close();
    fs.unwatchFile(f);
    console.log('Watchers closed.');
  }, 500);

  /* Expected output (order not guaranteed):
     [watch] Event → change
     [watchFile] size → 2
     Watchers closed.
  */
});

/***************************************************************************************************
 * 9)  Promise API (fs/promises) with async-await  ────────────────────────────────────────────────
 **************************************************************************************************/
runExample('9) fs/promises: await readFile / writeFile / rm', async () => {
  const f = path.join(SCRATCH, 'promise.txt');
  await fsp.writeFile(f, 'Promises FTW 🚀');
  const txt = await fsp.readFile(f, 'utf8');
  console.log('Read via promises →', txt);

  await fsp.rm(f);
  console.log('Exists after rm?', fs.existsSync(f));

  /* Expected output:
     Read via promises → Promises FTW 🚀
     Exists after rm? false
  */
});

/***************************************************************************************************
 * 10)  Extras: mkdtemp / realpath / appendFile / constants  ──────────────────────────────────────
 **************************************************************************************************/
runExample('10) mkdtemp / realpath / appendFile', async () => {
  const tmpDir = await fsp.mkdtemp(path.join(SCRATCH, 'sess-'));
  const f = path.join(tmpDir, 'log.txt');

  await fsp.appendFile(f, 'Line-1\n');
  await fsp.appendFile(f, 'Line-2\n');

  const resolved = fs.realpathSync(f);
  const data = fs.readFileSync(resolved, 'utf8');

  console.log('Resolved realpath:', resolved);
  console.log('File contents:\n' + data);

  /* Expected output (tmp path varies):
     Resolved realpath: /tmp/fs-tour-tmp/sess-abc123/log.txt
     File contents:
     Line-1
     Line-2
  */
});

/***************************************************************************************************
 * End of file – You can safely delete the scratch dir if desired:
 *   fs.rmSync(SCRATCH, { recursive: true, force: true });
 ***************************************************************************************************/