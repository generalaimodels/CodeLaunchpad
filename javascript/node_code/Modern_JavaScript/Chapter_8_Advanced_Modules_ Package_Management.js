/**************************************************************************************************
 * Chapter 8 | Advanced Modules & Package Management
 * ------------------------------------------------------------------
 * ONE single .js file – no external assets. Modern Node (≥16) ready.
 * 5 domains × ≥5 concise, runnable examples each.
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION ESM — ES Modules vs CommonJS                            */
/*───────────────────────────────────────────────────────────────────*/

/* ESM‑Example‑1:  Loading CommonJS with `require` */
(function () {
    const { Module } = require('module');
    const src = 'module.exports = x => x * 2;';
    const path = require('path').join(__dirname, 'tmp‑cjs.js');
    require('fs').writeFileSync(path, src);
    const double = require(path);
    console.log('ESM‑1 (CJS):', double(4));
    require('fs').unlinkSync(path);
  })();
  
  /* ESM‑Example‑2:  Loading an ES module via dynamic import (data‑URL) */
  (async () => {
    const mod = await import('data:text/javascript,export const id=7;');
    console.log('ESM‑2 (ESM):', mod.id);
  })();
  
  /* ESM‑Example‑3:  Hoisting vs. runtime evaluation */
  (function () {
    console.log('ESM‑3 order A');
    require('fs');               // executed now
    // import 'fs';               // (would be hoisted, error in .js CJS context)
    console.log('order B');
  })();
  
  /* ESM‑Example‑4:  Named & default exports interoperability */
  (async () => {
    const code = 'export default 1; export const x=2;';
    const m = await import('data:text/javascript,' + encodeURIComponent(code));
    console.log('ESM‑4:', m.default, m.x);
  })();
  
  /* ESM‑Example‑5:  __filename vs. import.meta.url */
  (function () {
    console.log('ESM‑5:', '__filename =', __filename);
    import('url').then(u => console.log('import.meta.url =', import.meta.url));
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION DYN — Dynamic `import()`, Tree Shaking                  */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* DYN‑Example‑1:  Late‑loaded heavy dependency */
  (async () => {
    if (Math.random() > 0.5) {
      const { default: zlib } = await import('zlib');
      console.log('DYN‑1 zlib loaded:', typeof zlib.deflate);
    }
  })();
  
  /* DYN‑Example‑2:  Conditional locale import */
  (async () => {
    const locale = 'fr';
    const dict = await import(`data:text/javascript,export default {hi:"salut"};`);
    console.log('DYN‑2:', dict.default.hi);
  })();
  
  /* DYN‑Example‑3:  Top‑level await with remote data‑url */
  (async () => {
    const mod = await import('data:text/javascript,export const n=42;');
    console.log('DYN‑3 top‑level await:', mod.n);
  })();
  
  /* DYN‑Example‑4:  Simulated tree‑shaking (sideEffects flag) */
  (function () {
    const pkg = { name: 'lib', sideEffects: false };
    const lib = {
      used: () => 1,
      unused: () => { console.log('should be dropped'); }
    };
    console.log('DYN‑4 tree‑shake keeps:', lib.used());
  })();
  
  /* DYN‑Example‑5:  Webpack magic‑comment chunk name */
  (async () => {
    await import(/* webpackChunkName:"analytics" */ 'data:text/javascript,export default 0');
    console.log('DYN‑5 chunk loaded');
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION PKG — package.json & Semantic Versioning                */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* PKG‑Example‑1:  Minimal package.json object validation */
  (function () {
    const pkg = { name: 'app', version: '1.0.0', main: 'index.js' };
    const mandatory = ['name', 'version'];
    const ok = mandatory.every(k => k in pkg);
    console.log('PKG‑1 valid:', ok);
  })();
  
  /* PKG‑Example‑2:  SemVer comparator */
  (function () {
    const cmp = (a, b) => a.localeCompare(b, undefined, { numeric: true });
    console.log('PKG‑2 1.2.10 > 1.2.2 ?', cmp('1.2.10', '1.2.2') > 0);
  })();
  
  /* PKG‑Example‑3:  Range ^ & ~ tester */
  (function () {
    const satisfy = (range, v) => {
      const [maj, min] = v.split('.');
      if (range.startsWith('^')) return range.slice(1).split('.')[0] === maj;
      if (range.startsWith('~')) return v.startsWith(range.slice(1, range.lastIndexOf('.')));
    };
    console.log('PKG‑3:', satisfy('^1.2.0', '1.4.9'), satisfy('~1.2.0', '1.3.0'));
  })();
  
  /* PKG‑Example‑4:  "exports" conditional resolution */
  (function () {
    const exportsField = { '.': { import: './esm.js', require: './cjs.js' } };
    const mode = 'require';
    console.log('PKG‑4 selected entry:', exportsField['.'][mode]);
  })();
  
  /* PKG‑Example‑5:  Optional dependency failure handling */
  (function () {
    const optional = ['fsevents'];
    try { require(optional[0]); }
    catch { console.log('PKG‑5 optional dep missing, continuing'); }
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION MONO — Monorepos (Lerna, Nx)                            */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* MONO‑Example‑1:  lerna.json skeleton */
  (function () {
    const lerna = { version: 'independent', packages: ['packages/*'] };
    console.log('MONO‑1 lerna:', lerna);
  })();
  
  /* MONO‑Example‑2:  Yarn workspaces declaration */
  (function () {
    const rootPkg = { workspaces: ['packages/a', 'packages/b'] };
    console.log('MONO‑2 workspaces:', rootPkg.workspaces.join(', '));
  })();
  
  /* MONO‑Example‑3:  Nx target executor mock */
  (function () {
    const run = (proj, target) => console.log(`nx run ${proj}:${target}`);
    run('api', 'build');
  })();
  
  /* MONO‑Example‑4:  Fixed vs. independent versioning outputs */
  (function () {
    const pkgs = ['a', 'b'];
    const bumpFixed = v => pkgs.map(n => `${n}@${v}`);
    console.log('MONO‑4 fixed:', bumpFixed('2.0.0'));
  })();
  
  /* MONO‑Example‑5:  Local package linking simulation */
  (function () {
    const graph = { a: ['b'], b: [] };
    const order = [];
    const visit = p => { graph[p].forEach(visit); order.push(p); };
    visit('a');
    console.log('MONO‑5 build order:', order);
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION LOCK — Yarn, PNPM & Lockfile Strategies                 */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* LOCK‑Example‑1:  Parsing yarn.lock line */
  (function () {
    const line = '"left-pad@^1.0.0":';
    const pkg = line.match(/"(.+)@/)[1];
    console.log('LOCK‑1 pkg:', pkg);
  })();
  
  /* LOCK‑Example‑2:  pnpm store path calculation */
  (function () {
    const hash = v => require('crypto').createHash('sha1').update(v).digest('hex').slice(0, 2);
    const loc = (name, ver) => `.pnpm/${hash(name)}/${name}@${ver}`;
    console.log('LOCK‑2 store path:', loc('lodash', '4.17.21'));
  })();
  
  /* LOCK‑Example‑3:  npm‑lock shrinkwrap vs. package‑lock flag */
  (function () {
    const type = file => file === 'npm-shrinkwrap.json' ? 'frozen' : 'generated';
    console.log('LOCK‑3:', type('package-lock.json'));
  })();
  
  /* LOCK‑Example‑4:  Deterministic installs (`npm ci`) */
  (function () {
    const ci = lock => lock ? 'fast & reproducible' : 'cannot run';
    console.log('LOCK‑4 npm ci:', ci(true));
  })();
  
  /* LOCK‑Example‑5:  Integrity checksum validation */
  (function () {
    const integrity = (data, sum) =>
      require('crypto').createHash('sha512').update(data).digest('hex') === sum;
    const data = 'hello';
    const sum  = require('crypto').createHash('sha512').update(data).digest('hex');
    console.log('LOCK‑5 checksum ok:', integrity(data, sum));
  })();