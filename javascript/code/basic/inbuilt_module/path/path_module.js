/***************************************************************************************************
 *  Node.js Core Module : `path`
 *  Author : 
 *  File   : path-tour.js           ──>  run with `node path-tour.js`
 *
 *  Mission
 *  ───────
 *  Deliver 10 ultra-concise yet complete examples that walk through EVERY public export exposed by
 *  the `path` module (including the `posix` / `win32` sub-objects and OS-specific quirks such as
 *  `toNamespacedPath`).  Each example is isolated in its own function, prints a section header, and
 *  leaves the global state untouched.
 *
 *  Node ≥ 18.x   (but everything except `toNamespacedPath` exists way earlier)
 ***************************************************************************************************/
'use strict';
const path = require('path');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Mini runner – executes the demos sequentially for deterministic stdout
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const DEMOS = [];
function demo(title, fn) { DEMOS.push({ title, fn }); }
(async () => {
  for (const { title, fn } of DEMOS) {
    console.log('\n' + '═'.repeat(85));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(85));
    await fn();
  }
})();

/***************************************************************************************************
 * 1) join()   vs.   resolve()   – day-to-day path building
 **************************************************************************************************/
demo('1) join() vs resolve()', () => {
  console.log('join       →', path.join('usr', 'local', 'bin'));      // usr/local/bin
  console.log('resolve    →', path.resolve('usr', 'local', 'bin'));   // /current/dir/usr/local/bin
  console.log('resolve(/) →', path.resolve('/etc', '../var'));        // /var
  /* Expected output (cwd dependant):
     join       → usr/local/bin
     resolve    → /home/alice/project/usr/local/bin
     resolve(/) → /var
  */
});

/***************************************************************************************************
 * 2) dirname()  basename()  extname()
 **************************************************************************************************/
demo('2) dirname / basename / extname', () => {
  const fp = '/opt/apps/server/index.html';
  console.log('dirname () →', path.dirname(fp));          // /opt/apps/server
  console.log('basename() →', path.basename(fp));         // index.html
  console.log('extname () →', path.extname(fp));          // .html
});

/***************************************************************************************************
 * 3) parse() ⇄ format()  – round-trip components
 **************************************************************************************************/
demo('3) parse() ⇄ format()', () => {
  const parsed = path.parse('/var/log/syslog.1');
  console.log('parse() →', parsed);
  const rebuilt = path.format(parsed);
  console.log('format() →', rebuilt);                     // same as original
  /* Expected output (structure):
     parse() → { root: '/', dir: '/var/log', base: 'syslog.1', ext: '.1', name: 'syslog' }
     format() → /var/log/syslog.1
  */
});

/***************************************************************************************************
 * 4) normalize() – squash “..”, “.” and duplicate separators
 **************************************************************************************************/
demo('4) normalize()', () => {
  console.log(path.normalize('/foo//bar/../baz//'));      // /foo/baz/
  console.log(path.normalize('C:\\temp\\\\foo\\..\\'));   // Windows-style cleanup
});

/***************************************************************************************************
 * 5) relative() – compute how to get from A to B
 **************************************************************************************************/
demo('5) relative()', () => {
  const from = '/data/projects';
  const to   = '/data/photos/2024/vacation.jpg';
  console.log(path.relative(from, to));                   // ../photos/2024/vacation.jpg
});

/***************************************************************************************************
 * 6) isAbsolute() + sep + delimiter
 **************************************************************************************************/
demo('6) isAbsolute / sep / delimiter', () => {
  console.log('isAbsolute("/etc")  →', path.isAbsolute('/etc'));      // true (POSIX)
  console.log('isAbsolute("etc")   →', path.isAbsolute('etc'));       // false
  console.log('OS path separator   →', JSON.stringify(path.sep));     // "/" or "\\"
  console.log('PATH env delimiter  →', JSON.stringify(path.delimiter)); // ":" or ";"
});

/***************************************************************************************************
 * 7) posix   vs.   win32   – cross-platform handling
 **************************************************************************************************/
demo('7) path.posix & path.win32', () => {
  const p = 'C:\\Program Files\\node\\node.exe';
  console.log('basename(win32) →', path.win32.basename(p));           // node.exe
  const q = '/usr/local/bin/node';
  console.log('basename(posix) →', path.posix.basename(q));           // node
});

/***************************************************************************************************
 * 8) toNamespacedPath()  (Windows-only API – safe no-op elsewhere)
 **************************************************************************************************/
demo('8) toNamespacedPath()', () => {
  const p = 'C:\\Temp\\file.txt';
  const ns = path.toNamespacedPath ? path.toNamespacedPath(p) : p;
  console.log('Namespace path →', ns);
  /* Expected output (Windows):
     Namespace path → \\?\C:\Temp\file.txt
     (POSIX → same as input)
  */
});

/***************************************************************************************************
 * 9) dirname() edge-case: strip trailing “/” repeatedly
 **************************************************************************************************/
demo('9) dirname edge-cases', () => {
  let dir = '/a/b/c/';
  while (dir && dir !== '/') {
    console.log('dir →', dir);
    dir = path.dirname(dir);
  }
  /* Expected output:
     dir → /a/b/c/
     dir → /a/b
     dir → /a
  */
});

/***************************************************************************************************
 * 10) Weird extension rules – basename(file,".ext") & extname double-dots
 **************************************************************************************************/
demo('10) basename with ext filter & multi-part extname', () => {
  const file = 'archive.tar.gz';
  console.log('extname()     →', path.extname(file));               // .gz
  console.log('basename(*,.gz)→', path.basename(file, '.gz'));      // archive.tar
  console.log('basename(*,.tar.gz) (manual) →',
    path.basename(file).replace(/\.tar\.gz$/, ''));                 // archive
  /* Expected output:
     extname()     → .gz
     basename(*,.gz)→ archive.tar
     basename(*,.tar.gz) (manual) → archive
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/