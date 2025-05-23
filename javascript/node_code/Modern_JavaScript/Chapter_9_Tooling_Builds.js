/**************************************************************************************************
 * Chapter 8 | Advanced Modules & Package Management
 * -----------------------------------------------------------------------------------------------
 * ONE self‑contained .js (ESM) file — 5 sections × ≥5 examples each.
 * Requires Node ≥16 with "type":"module" or .mjs extension.
 **************************************************************************************************/

import { createRequire } from 'module';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION M1 ── ES Modules vs CommonJS                                                       */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

// M1‑1: Named & default exports (ESM)
// File: esm-module.mjs
// export const pi = 3.1415;
// export function area(r){ return pi * r * r; }
// export default class Circle { constructor(r){ this.r=r } }

// M1‑2: CommonJS exports
// File: cjs-module.js
// module.exports = { pi:3.14, area(r){ return this.pi * r * r } };

// M1‑3: Importing CommonJS in ESM via createRequire
const require_ = createRequire(import.meta.url);
const cjs = require_('./cjs-module.js');
console.log('M1‑3:', cjs.pi, cjs.area(2));

// M1‑4: Importing ESM in ESM (static & dynamic)
import Circle, { pi as piE, area as areaE } from './esm-module.mjs';
console.log('M1‑4 static:', new Circle(2).r, piE, areaE(2));
(async()=>{
  const mod = await import('./esm-module.mjs');
  console.log('M1‑4 dynamic:', mod.default.name);
})();

// M1‑5: __dirname & __filename equivalents
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
console.log('M1‑5:', __dirname, __filename);
// Exception: static __dirname unavailable in ESM without above workaround

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION M2 ── Dynamic import() & Tree Shaking                                            */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

// M2‑1: Dynamic import with error handling
;(async()=>{
  try {
    const { area } = await import('./esm-module.mjs');
    console.log('M2‑1:', area(3));
  } catch(e){
    console.error('M2‑1 import failed:', e.message);
  }
})();

// M2‑2: Conditional import
const legacyMode = false;
if (legacyMode) {
  await import('./legacy.js').then(m=>m.init());
} else {
  await import('./modern.js').then(m=>m.init());
}

// M2‑3: Webpack magic comments for chunk naming
import(/* webpackChunkName:"utils", webpackMode:"lazy-once" */ './utils.js')
  .then(utils=>console.log('M2‑3 utils:', Object.keys(utils)));

// M2‑4: Tree‑shaking demonstration
// File: lib.js
// export function used(){ return 'used' }
// export function unused(){ return 'unused' }
// Bundler drops `unused()` automatically when only `used()` is imported

// M2‑5: Vite import.meta.glob for code‑splitting
const modules = import.meta.glob('./modules/*.js');
for (const path in modules) {
  modules[path]().then(m=>console.log('M2‑5 loaded', path));
}

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION M3 ── package.json & Semantic Versioning                                        */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

// M3‑1: Semantic version fields
const pkg1 = { name:"app", version:"1.2.3" };      // MAJOR.MINOR.PATCH
console.log('M3‑1:', pkg1.version);

// M3‑2: Dependency range specifiers
const pkg2 = {
  dependencies:{
    "libA":"^2.0.0",    // >=2.0.0 <3.0.0
    "libB":"~1.5.2",    // >=1.5.2 <1.6.0
    "libC":">=0.3.0",   // >=0.3.0
    "libD":"2.1.x",     // >=2.1.0 <2.2.0
    "libE":"*"          // any version
  }
};
console.log('M3‑2 libA range:', pkg2.dependencies.libA);

// M3‑3: Scripts & cross‑platform env
const pkg3 = {
  scripts:{
    start:"node index.js",
    test:"jest",
    build:"cross-env NODE_ENV=production webpack"
  }
};

// M3‑4: Conditional exports field
const pkg4 = {
  exports:{
    ".":{ import:"./esm.js", require:"./cjs.js" },
    "./feature":{ node:"./feat.node.js", default:"./feat.js" }
  }
};
console.log('M3‑4 exports:', Object.keys(pkg4.exports));

// M3‑5: type/module/main interplay
const pkg5 = { type:"module", main:"cjs-entry.js", module:"esm-entry.js" };
console.log('M3‑5 type:', pkg5.type);

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION M4 ── Monorepos (Lerna, Nx)                                                       */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

// M4‑1: lerna.json (fixed vs independent versions)
const lernaFixed = { packages:["packages/*"], version:"1.0.0" };
const lernaIndep= { packages:["packages/*"], version:"independent" };
console.log('M4‑1 fixed:', lernaFixed.version, 'indep:', lernaIndep.version);

// M4‑2: Yarn/PNPM workspaces in root package.json
const rootPkg = { private:true, workspaces:["packages/*"] };
console.log('M4‑2 workspaces:', rootPkg.workspaces);

// M4‑3: Nx workspace.json snippet
const nxWorkspace = {
  projects:{
    app:{ root:"apps/app", targets:{ build:{ executor:"@nrwl/node:build" } } },
    lib:{ root:"libs/lib", tags:["scope:shared"] }
  },
  implicitDependencies:{ "tsconfig.base.json":["*"] }
};
console.log('M4‑3 projects:', Object.keys(nxWorkspace.projects));

// M4‑4: Programmatically list packages folder
const pkgDirs = fs.readdirSync('packages', { withFileTypes:true })
  .filter(d=>d.isDirectory()).map(d=>d.name);
console.log('M4‑4 packages:', pkgDirs);

// M4‑5: Running Lerna bootstrap via child_process
exec('npx lerna bootstrap', (err, stdout)=> {
  if (err) console.error('M4‑5 error:', err.message);
  else console.log('M4‑5 bootstrap:', stdout.split('\n')[0]);
});

/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION M5 ── Yarn, PNPM & Lockfile Strategies                                          */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

// M5‑1: Lockfile types
console.log('M5‑1: yarn uses yarn.lock, npm uses package-lock.json, pnpm uses pnpm-lock.yaml');

// M5‑2: Yarn resolutions to override transitive deps
const pkgResolutions = {
  dependencies:{ foo:"1.0.0" },
  resolutions:{ "bar":"2.2.3" }
};
console.log('M5‑2 resolutions:', pkgResolutions.resolutions);

// M5‑3: pnpm hoisting config (.npmrc)
const npmrc = 'node-linker=hoisted\n';
fs.writeFileSync('.npmrc', npmrc);
console.log('M5‑3 .npmrc written');

// M5‑4: npm ci vs install in CI
exec('npm ci --prefer-offline', (e,o)=>{
  console.log('M5‑4 npm ci', e?e.message:'success');
});

// M5‑5: Lockfile commit strategy
console.log('M5‑5: Always commit lockfiles; enforce with CI checks (e.g. `npm ci` failures on diff`).');