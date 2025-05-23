/****************************************************************************************
* Chapter-4 | Memory Management & Performance                                            *
* THIS SINGLE FILE CONTAINS 30 WELL-COMMENTED, SELF-CONTAINED EXAMPLES (5 PER SECTION). *
* TO RUN NODE-SPECIFIC SNIPPETS THAT REQUIRE GC ACCESS, USE: `node --expose-gc main.js`  *
****************************************************************************************/

/*────────────────────────────────────────────────────────────────────────────────────────
1. GARBAGE COLLECTION ALGORITHMS
   (Mark-&-Sweep + Generational GC) – 5 Examples
────────────────────────────────────────────────────────────────────────────────────────*/

/* 1-A MARK & SWEEP (SIMULATION) – Graph traversal marks reachable nodes, unmarks are swept */
(function markSweepSimulation() {
    class Node { constructor(id) { this.id = id; this.refs = []; } }
    const A = new Node('A'), B = new Node('B'), C = new Node('C');
    A.refs.push(B);  B.refs.push(C);          // A→B→C  (root: A)
    let roots = [A];                          // GC root set
    // Step-1: mark
    const marked = new Set();
    (function mark(obj) {
      if (!obj || marked.has(obj)) return;
      marked.add(obj);
      obj.refs.forEach(mark);
    })(roots[0]);
    // Step-2: sweep (anything unmarked is collected)
    [A, B, C].forEach(n => {
      if (!marked.has(n)) console.log(`Swept ${n.id}`);
    });
  })();
  
  /* 1-B MARK & SWEEP (REAL GC TRIGGERED BY LOSING REFERENCE) */
  (function realMarkSweep() {
    let big = new Array(1e6).fill('*'); // allocate ~8MB
    console.log('big allocated');       // reachable
    big = null;                         // unreachable → will be swept
    if (global.gc) { global.gc(); console.log('manual GC'); }
  })();
  
  /* 1-C GENERATIONAL GC – MANY SHORT-LIVED OBJECTS (YOUNG GEN) */
  (function generationalShortLived() {
    for (let i = 0; i < 1e4; i++) { const tmp = {i}; } // die quickly
    if (global.gc) { global.gc(); }
    // Most survived 0-1 collections; cheap minor GC cycles handle them.
  })();
  
  /* 1-D GENERATIONAL GC – LONG-LIVED OBJECT PROMOTION */
  (function generationalLongLived() {
    const survivors = [];
    for (let i = 0; i < 5; i++) survivors.push({i}); // stay referenced
    // After N minor collections, the 5 objs move to old gen; major GC needed
  })();
  
  /* 1-E WEAK REFERENCES (WeakMap + FinalizationRegistry) – AVOIDING RETENTION */
  (function weakRefs() {
    const cache = new WeakMap();
    const registry = new FinalizationRegistry(id => console.log(`GC ->`, id));
    (function() {
      let obj = {payload: 'heavy'};
      cache.set(obj, 'cached');
      registry.register(obj, 'heavy-obj');
    })(); // obj goes out of scope here
    if (global.gc) { global.gc(); }
  })();
  
  /*────────────────────────────────────────────────────────────────────────────────────────
  2. IDENTIFYING & FIXING MEMORY LEAKS – 5 Patterns
  ────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* 2-A ACCIDENTAL GLOBAL */
  function leakGlobal() {
    leaked = new Array(1e6);  // 'leaked' becomes a global
  }
  function fixGlobal() {
    const notLeaked = new Array(1e6);
  }
  
  /* 2-B DETACHED DOM NODE (BROWSER) */
  (function domDetachLeak() {
    // <div id="container"><span id="child"></span></div>
    const container = document.getElementById('container');
    let orphan = document.getElementById('child');
    container.removeChild(orphan); // orphan detached BUT still referenced
    // FIX:
    orphan = null; // allow GC
  })();
  
  /* 2-C CLOSURE CAPTURING LARGE OBJ */
  function closureLeak() {
    const large = new Array(1e6);
    return () => console.log(large.length); // leak
  }
  function fixClosureLeak() {
    const length = (function() {
      const size = 1e6;
      return () => console.log(size);
    })(); // only primitive captured
  }
  
  /* 2-D FORGOTTEN EVENT LISTENER */
  function listenerLeak(btn) {
    function handler() { console.log('click'); }
    btn.addEventListener('click', handler);
    // FIX:
    return () => btn.removeEventListener('click', handler);
  }
  
  /* 2-E UNBOUNDED CACHE */
  class LRUCache {
    constructor(limit = 1000) { this.limit = limit; this.map = new Map(); }
    get(k) { const v = this.map.get(k); if (v) { this.map.delete(k); this.map.set(k, v); } return v; }
    set(k, v) {
      if (this.map.size >= this.limit) this.map.delete(this.map.keys().next().value);
      this.map.set(k, v);
    }
  }
  
  /*────────────────────────────────────────────────────────────────────────────────────────
  3. PROFILING & BENCHMARKING (DEVTOOLS / NODE) – 5 Examples
  ────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* 3-A console.time */
  console.time('loop');
  for (let i = 0; i < 1e6; i++);
  console.timeEnd('loop');
  
  /* 3-B performance.now (Browser / Node >= 8.5) */
  const {performance} = require('perf_hooks');
  const t0 = performance.now(); for (let i = 0; i < 1e6; i++); console.log('Δms', performance.now() - t0);
  
  /* 3-C Node PerformanceObserver */
  const {PerformanceObserver} = require('perf_hooks');
  const obs = new PerformanceObserver(list => console.log(list.getEntries()[0]));
  obs.observe({entryTypes: ['function']});
  function heavy() { for (let i = 0; i < 1e7; i++); }
  performance.timerify(heavy)();
  
  /* 3-D Chrome DevTools mark/measure */
  performance.mark('A');
  // ...code
  performance.mark('B');
  performance.measure('A→B', 'A', 'B');
  console.log(performance.getEntriesByName('A→B')[0].duration);
  
  /* 3-E Benchmark.js (3rd-party) */
  const Benchmark = require('benchmark');
  new Benchmark.Suite()
    .add('string#concat', () => 'a' + 'b')
    .add('string#template', () => `a${'b'}`)
    .on('cycle', e => console.log(String(e.target)))
    .run();
  
  /*────────────────────────────────────────────────────────────────────────────────────────
  4. OPTIMIZING LOOPS & RECURSION – 5 Cases
  ────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* 4-A PRECOMPUTE LENGTH */
  function sum1(arr) { let s = 0; for (let i = 0, len = arr.length; i < len; i++) s += arr[i]; }
  
  /* 4-B FOR-OF VS CLASSIC FOR */
  function sum2(arr) { let s = 0; for (const n of arr) s += n; } // cleaner but slower
  
  /* 4-C MAP VS MANUAL LOOP */
  const squaredFast = arr => {
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] ** 2;
  };
  
  /* 4-D TAIL RECURSION ELIMINATION */
  function factIter(n) { let res = 1; while (n > 1) res *= n--; return res; }
  
  /* 4-E MEMOIZED RECURSION */
  const fib = (() => {
    const memo = new Map([[0,0],[1,1]]);
    return function f(n){ if(memo.has(n)) return memo.get(n); const val = f(n-1)+f(n-2); memo.set(n,val); return val; };
  })();
  
  /*────────────────────────────────────────────────────────────────────────────────────────
  5. REFLOWS/REPAINTS & DOM BATCH UPDATES – 5 Examples
  ────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* 5-A READ-WRITE COALESCING */
  function efficientStyle(el) {
    const height = el.offsetHeight;     // READ
    requestAnimationFrame(() => { el.style.height = (height + 10) + 'px'; }); // WRITE
  }
  
  /* 5-B DOCUMENT FRAGMENT */
  function addMany() {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 1000; i++) {
      const li = document.createElement('li');
      li.textContent = i;
      frag.appendChild(li);
    }
    document.getElementById('list').appendChild(frag);
  }
  
  /* 5-C CSS CLASS TOGGLE INSTEAD OF INLINE STYLE */
  function toggleState(el, on) { el.classList[on ? 'add' : 'remove']('active'); }
  
  /* 5-D VIRTUAL SCROLLING (SIMPLIFIED) */
  function renderVisible(container, items, itemHeight) {
    const scrollTop = container.scrollTop;
    const start = Math.floor(scrollTop / itemHeight);
    const end = start + Math.ceil(container.clientHeight / itemHeight);
    container.innerHTML = ''; // single reflow
    for (let i = start; i <= end; i++) {
      const div = document.createElement('div');
      div.textContent = items[i];
      div.style.height = itemHeight + 'px';
      container.appendChild(div);
    }
  }
  
  /* 5-E RESIZE OBSERVER TO DEFER HANDLING */
  const ro = new ResizeObserver(entries => {
    requestAnimationFrame(() => {
      entries.forEach(entry => console.log('resized', entry.contentRect));
    });
  });
  ro.observe(document.body);
  
  /*────────────────────────────────────────────────────────────────────────────────────────
  6. CODE SPLITTING & LAZY LOADING – 5 Examples
  ────────────────────────────────────────────────────────────────────────────────────────*/
  
  /* 6-A DYNAMIC IMPORT */
  async function loadChart() {
    const {default: Chart} = await import('./Chart.js');
    new Chart();
  }
  
  /* 6-B CONDITIONAL IMPORT */
  if (location.pathname === '/admin') import('./adminPanel.js');
  
  /* 6-C PREFETCH LINK + IMPORT (HTML + JS) */
  /*
  <link rel="prefetch" href="analytics.js">
  */
  async function loadAnalytics() { await import(/* webpackPrefetch: true */ './analytics.js'); }
  
  /* 6-D REACT.LAZY */
  import React, { Suspense } from 'react';
  const Settings = React.lazy(() => import('./Settings.jsx'));
  function App() { return (<Suspense fallback={<span>…</span>}><Settings/></Suspense>); }
  
  /* 6-E WEBPACK MAGIC COMMENTS – CHUNK NAME */
  import(/* webpackChunkName: "vendor-lodash" */ 'lodash')
    .then(_ => console.log('lodash loaded'));