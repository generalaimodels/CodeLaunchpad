/**
 * console-methods-showcase.js
 * ---------------------------------------------------------------
 * A one-stop, production-ready reference that demonstrates EVERY
 * public (and several internal) Node.js `console` method.  
 * Two concise, real-world examples are provided per method.
 *
 * 
 */

'use strict';
const fs   = require('fs');
const path = require('path');

/*───────────────────────────── helpers ─────────────────────────────*/
const banner = title =>
  console.log('\n' + '—'.repeat(14) + ` ${title} ` + '—'.repeat(14));

/*======================================================================
  1. console.log()
 ======================================================================*/
banner('console.log');
console.log('EX1 → Server listening on :%d', 3000);
console.log('EX2 → User data %o', { id: 7, name: 'Grace Hopper' });

/*======================================================================
  2. console.warn()
 ======================================================================*/
banner('console.warn');
console.warn('EX1 → ⚠️  Feature %s will be removed in v4', 'legacy-auth');
try { JSON.parse('{'); } catch { console.warn('EX2 → Using defaults'); }

/*======================================================================
  3. console.dir()
 ======================================================================*/
banner('console.dir');
const nested = { a: 1, b: { c: [2, 3] } };
console.dir(nested, { depth: 0 });                    // EX1
console.dir(nested, { depth: null, colors: true });   // EX2

/*======================================================================
  4. console.time / timeLog / timeEnd
 ======================================================================*/
banner('console.time|timeLog|timeEnd');
console.time('hash');
require('crypto').createHash('sha256').update('data').digest('hex');
console.timeLog('hash', 'EX1 → midway');
console.timeEnd('hash');

(async () => {                                        // EX2
  console.time('api');
  await new Promise(r => setTimeout(r, 250));
  console.timeEnd('api');
})();

/*======================================================================
  5. console.trace()
 ======================================================================*/
banner('console.trace');
function f1() { f2(); }
function f2() { console.trace('EX1 → call stack'); }
f1();

(function recur(n) { if (!n) return; console.trace(`EX2 → depth ${n}`); recur(--n);} )(2);

/*======================================================================
  6. console.assert()
 ======================================================================*/
banner('console.assert');
console.assert(1 === 1, 'This will NOT print');                // EX1
console.assert(false===true, 'EX2 → Assertion failed %o', { code:500 });

/*======================================================================
  7. console.clear()
 ======================================================================*/
banner('console.clear');
console.log('EX1 → about to clear in 1s');                      // visible
setTimeout(() => { console.clear(); console.log('EX2 → cleared'); }, 1000);

/*======================================================================
  8. console.count / countReset
 ======================================================================*/
banner('console.count|countReset');
['GET','GET','POST'].forEach(m => console.count(m));            // EX1
console.countReset('GET');
console.count('GET');                                           // EX2

/*======================================================================
  9. console.group / groupEnd / groupCollapsed
 ======================================================================*/
banner('console.group|groupEnd|groupCollapsed');
console.group('Startup');                                       // EX1
console.log('Loading modules…');
console.groupEnd();

console.groupCollapsed('Hidden');                               // EX2
console.debug('secret=123');
console.groupEnd();

/*======================================================================
 10. console.table()
 ======================================================================*/
banner('console.table');
console.table([{ id:1, status:'ok' },{ id:2, status:'fail' }]); // EX1
console.table({ Alice:91, Bob:88, Carol:95 });                  // EX2

/*======================================================================
 11. console.debug()
 ======================================================================*/
banner('console.debug');
console.debug('EX1 → Debug level message');
console.debug('EX2 → Current env %s', process.env.NODE_ENV);

/*======================================================================
 12. console.info()
 ======================================================================*/
banner('console.info');
console.info('EX1 → ℹ️  Build finished');
console.info('EX2 → Memory %f MB', process.memoryUsage().rss/1e6);

/*======================================================================
 13. console.dirxml()
 ======================================================================*/
banner('console.dirxml');
console.dirxml({ tag:'div', children:[{ tag:'span', text:'Hi' }]}); // EX1
console.dirxml('<root/>');                                           // EX2

/*======================================================================
 14. console.error()
 ======================================================================*/
// banner('console.error');
console.error('EX1 → 💥 Fatal: %s', 'disk full');
// try { fs.readFileSync('nope'); } catch (e) { console.error('EX2', e); }

/*======================================================================
 15. console.profile / profileEnd / timeStamp
        (visible in Chrome DevTools / Node inspector)
 ======================================================================*/
banner('console.profile|profileEnd|timeStamp');
console.profile('calc');
for (let i=0;i<1e5;i++) Math.sqrt(i);
console.profileEnd('calc');
console.timeStamp('profile done');

/*======================================================================
 16. console.context()
 ======================================================================*/
banner('console.context');
const ctx1 = console.context({ reqId:'xyz' });
ctx1.log('EX1 → contextual log');
console.context({ user:42 }).warn('EX2 → with user ctx');

/*======================================================================
 17. console.Console  (custom transports)
 ======================================================================*/
banner('console.Console');
const out = fs.createWriteStream(path.join(__dirname,'app.log'));
const err = fs.createWriteStream(path.join(__dirname,'app.err'));
const fileCon = new console.Console({ stdout: out, stderr: err });
fileCon.log('EX1 → Persisted');
fileCon.error('EX2 → Error persisted');

/*======================================================================
 18. console.createTask()  (experimental)
 ======================================================================*/
// banner('console.createTask');
// const t = console.createTask('backup');
// t.start(); t.log('EX1 → running'); t.end();
// const t2 = console.createTask('migrate');
// t2.start(); t2.error('EX2 → failed'); t2.end();

/*======================================================================
 19. Internal fields & handlers
      _stdout, _stderr, _stdoutErrorHandler, _stderrErrorHandler,
      _ignoreErrors, _times
 ======================================================================*/
banner('Internal helpers');

// _stdout / _stderr direct
console._stdout.write('EX1 → direct STDOUT write\n');
console._stderr.write('EX2 → direct STDERR write\n');

// _times inspection
console.time('tmp');
setTimeout(() => { console.timeEnd('tmp'); console.log(console._times); }, 50);

// custom error handlers
// console._stdoutErrorHandler = err => fs.appendFileSync('stdout.fails', err.message+'\n');
// console._stderrErrorHandler = err => fs.appendFileSync('stderr.fails', err.message+'\n');
console._ignoreErrors = true;  // toggle global silence

/*======================================================================
 20. DONE
 ======================================================================*/
banner('DONE');
process.on('beforeExit', ()=>console.log('✅  All console method demos complete'));