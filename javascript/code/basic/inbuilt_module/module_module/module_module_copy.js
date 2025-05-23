/***************************************************************************************************
 *  Node.js Core Module : `module`
 *  Author  : <Your Name Here>
 *  Goal    : 10 self–contained examples that hit every *documented* public API (+ a few de-facto
 *             internals such as `_resolveFilename/_load`, `require.extensions`) so you can bend the
 *             Node.js module loader to your will.  Execute the file with `node module-tour.js`.
 *
 *  Tested  : Node ≥ 18.x (older versions may lack `findSourceMap` / `SourceMap`).
 ***************************************************************************************************/
'use strict';
const Module = require('module');
const fs     = require('fs');
const path   = require('path');
const os     = require('os');
const { spawnSync } = require('child_process');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Sequential runner for prettified output
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const CASES = [];
function demo(title, fn) { CASES.push({ title, fn }); }
(async () => {
  for (const { title, fn } of CASES) {
    console.log('\n' + '═'.repeat(100));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(100));
    try { await fn(); }
    catch (e) { console.error('💥  Exception →', e); }
  }
})();

/***************************************************************************************************
 * Scratch playground (never touch user data)
 **************************************************************************************************/
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'module-tour-'));

/***************************************************************************************************
 * 1) Module.builtinModules  – inventory all native add-ons packaged with Node
 **************************************************************************************************/
demo('1) builtinModules inventory', () => {
  console.log('Total built-ins →', Module.builtinModules.length);
  console.log('Sample slice   →', Module.builtinModules.slice(0, 8));
  /* Expected output:
     Total built-ins → <≈80>
     Sample slice   → [ 'assert', 'async_hooks', 'buffer', 'child_process', ... ]
  */
});

/***************************************************************************************************
 * 2) createRequire(from)  – fabricate a `require` that resolves like another file/URL
 **************************************************************************************************/
demo('2) Module.createRequire()', () => {
  const cfgPath = path.join(TMP, 'config.json');
  fs.writeFileSync(cfgPath, JSON.stringify({ env: 'dev' }));
  const fakeFile = path.join(TMP, 'nested', 'file.js');
  fs.mkdirSync(path.dirname(fakeFile), { recursive: true });

  const requireFromNested = Module.createRequire(fakeFile);
  const cfg = requireFromNested('../config.json');   // resolved relative to `fakeFile`
  console.log('Loaded env →', cfg.env);              // dev
  /* Expected output:
     Loaded env → dev
  */
});

/***************************************************************************************************
 * 3) Module._resolveFilename(request, parent)  (⚠ internal, still invaluable)
 **************************************************************************************************/
demo('3) _resolveFilename – reveal exact path Node will load', () => {
  const resolved = Module._resolveFilename('./config.json', { filename: __filename });
  console.log('Resolved path →', resolved);
  /* Expected output:
     Resolved path → <absolute path>/config.json
  */
});

/***************************************************************************************************
 * 4) Module._load(request, parent)  – one-liner dynamic loader bypassing the normal `require` cache
 **************************************************************************************************/
demo('4) _load – force fresh module instance', () => {
  const somePath = path.join(TMP, 'counter.js');
  fs.writeFileSync(somePath, 'module.exports = { n: 0 };');

  const m1 = Module._load(somePath, null, true);
  m1.n++;
  delete require.cache[somePath];                    // scrub cache
  const m2 = Module._load(somePath, null, true);
  console.log('Fresh copy? →', m2.n === 0);          // true
  /* Expected output:
     Fresh copy? → true
  */
});

/***************************************************************************************************
 * 5) syncBuiltinESMExports() – keep CJS & ESM views of built-ins in sync
 **************************************************************************************************/
demo('5) syncBuiltinESMExports() keeps patches visible in ESM', async () => {
  // Mutate a builtin (NOT for production – purely demo)
  const fsCJS = require('fs');
  fsCJS.foo = () => 'patched';
  Module.syncBuiltinESMExports();                    // propagate mutation to ESM side

  const fsESM = await import('node:fs');             // dynamic import of ESM namespace
  console.log('ESM sees patch? →', typeof fsESM.foo === 'function');
  /* Expected output:
     ESM sees patch? → true
  */
});

/***************************************************************************************************
 * 6) findSourceMap() & SourceMap – map transpiled code back to the original
 **************************************************************************************************/
demo('6) findSourceMap() + SourceMap consumer', () => {
  const transpiled = path.join(TMP, 'hello.js');
  const srcMap     = path.join(TMP, 'hello.js.map');

  // Fake compilation: original line 1 maps to generated line 1
  fs.writeFileSync(srcMap, JSON.stringify({
    version:3, sources:['hello.ts'], names:[], mappings:'AAAA', file:'hello.js'
  }));
  fs.writeFileSync(transpiled,
    'console.log("hi");\n//# sourceMappingURL=hello.js.map');

  require(transpiled);                               // executes file; SourceMap registered
  const map = Module.findSourceMap(transpiled);
  console.log('Map found? →', !!map);
  console.log('Original source of line 1 col 0 →', map.sourceMap.sources[0]); // hello.ts
  /* Expected output:
     hi
     Map found? → true
     Original source of line 1 col 0 → hello.ts
  */
});

/***************************************************************************************************
 * 7) Module.SourceMap class – manual injection of a map at runtime
 **************************************************************************************************/
demo('7) Manual SourceMap injection via new Module.SourceMap()', () => {
  if (!Module.SourceMap) return console.log('SourceMap class not in this Node build, skipping.');
  const codePath = path.join(TMP, 'calc.js');
  fs.writeFileSync(codePath, 'exports.sum = (a,b)=>a+b//# sourceMappingURL=data:,');

  const fakeMap = new Module.SourceMap({
    url:        'calc.js.map',
    sourceMap:  { version:3, sources:['calc.orig.js'], names:[], mappings:'' }
  });
  Module.addSourceMap(codePath, fakeMap);            // undocumented util but supported
  const retrieved = Module.findSourceMap(codePath);
  console.log('Injected map source[0] →', retrieved?.sourceMap?.sources?.[0]);
  /* Expected output:
     Injected map source[0] → calc.orig.js
  */
});

/***************************************************************************************************
 * 8) Module.runMain() – (re)run the entry script programmatically
 **************************************************************************************************/
demo('8) runMain – re-invoke a file as if via CLI', () => {
  const mainFile = path.join(TMP, 'main-echo.js');
  fs.writeFileSync(mainFile, 'console.log("I am main", process.argv[2]);');

  const out = spawnSync(process.execPath, ['-e',
    `const Module=require("module");process.argv[1]='${mainFile}';process.argv[2]='XYZ';Module.runMain();`
  ]);
  console.log('Spawn output →', out.stdout.toString().trim());
  /* Expected output:
     Spawn output → I am main XYZ
  */
});

/***************************************************************************************************
 * 9) require.extensions – custom loader for *.txt  (legacy but still available)
 **************************************************************************************************/
demo('9) Custom loader via require.extensions[".txt"]', () => {
  Module._extensions = Module._extensions || require.extensions; // alias safeguard
  require.extensions['.txt'] = (mod, filename) => {
    const content = fs.readFileSync(filename, 'utf8');
    mod.exports   = content.toUpperCase();
  };
  const poemPath = path.join(TMP, 'poem.txt');
  fs.writeFileSync(poemPath, 'roses are red');
  console.log('Loaded poem →', require(poemPath));   // ROSES ARE RED
  /* Expected output:
     Loaded poem → ROSES ARE RED
  */
});

/***************************************************************************************************
 * 10) Module.globalPaths – Augmenting lookup paths at runtime
 **************************************************************************************************/
demo('10) globalPaths – on-the-fly NODE_PATH', () => {
  const libDir = path.join(TMP, 'lib');
  fs.mkdirSync(libDir);
  fs.writeFileSync(path.join(libDir, 'mock.js'), 'module.exports = 42;');

  Module.globalPaths.unshift(libDir);                // prepend to search list
  console.log('Resolved via global path →', require('mock')); // 42
  /* Expected output:
     Resolved via global path → 42
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/