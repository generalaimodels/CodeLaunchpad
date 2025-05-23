/**************************************************************************************************
 *  Chapter 4 | Memory Management & Performance
 *  -----------------------------------------------------------------------------------------------
 *  This single .js file is a self‑contained “mini‑playground” that demonstrates, documents,
 *  and stress‑tests the core ideas behind efficient memory management and performance tuning
 *  in modern JavaScript engines (V8, SpiderMonkey, JavaScriptCore, etc.).
 *
 *  Structure
 *  =========
 *  1.  Garbage‑Collection Algorithms (Mark‑&‑Sweep, Generational GC) …………………  Section GC
 *  2.  Identifying & Fixing Memory Leaks …………………………………………………………… Section LEAKS
 *  3.  Profiling & Benchmarking (DevTools + micro‑benchmarks) ……………………… Section PROF
 *  4.  Optimizing Loops & Recursion …………………………………………………………………… Section LOOPS
 *  5.  Reflows / Repaints & DOM Batch Updates …………………………………………………… Section DOM
 *  6.  Code‑Splitting & Lazy Loading ………………………………………………………………… Section SPLIT
 *
 *  Each section contains ≥5 concrete, runnable examples.  Open this file in DevTools,
 *  copy/paste per‑section into the console or run under Node (DOM‑specific parts need browser).
 *
 *  NOTE:  Everything lives in a single file as required.                                      ✨
 **************************************************************************************************/


/*───────────────────────────────────────────────────────────────────────────────────────────────*/
/* SECTION GC ── Garbage‑Collection Algorithms (Mark‑&‑Sweep, Generational)                    */
/*───────────────────────────────────────────────────────────────────────────────────────────────*/

/**
 *  GC‑Example‑1:  Simple Mark‑&‑Sweep illustration.
 *  ------------------------------------------------
 *  After obj is set to null, it becomes unreachable; next GC cycle will
 *  mark it “white” and collect (sweep) it.
 */
(function gcExample1() {
    let obj = { huge: new Array(1e6).fill('*') };  // ~1MB
    console.log('GC‑1: allocated');
    obj = null;                                    // Eligible for collection
  })();
  
  /**
   *  GC‑Example‑2:  Closure retaining memory (illustrates why Mark‑&‑Sweep matters).
   */
  (function gcExample2() {
    function makeRetainer() {
      const big = new Array(1e6).fill('#');
      return () => big;             // big is captured; NOT collectible
    }
    const leak = makeRetainer();    // Still in scope
    console.log('GC‑2: big is retained, cannot be swept');
    // leak(); // uncomment to prove reference still alive
  })();
  
  /**
   *  GC‑Example‑3:  Generational GC – survive multiple minor GCs, then promoted to old space.
   *  ----------------------------------------------------------------------------------------
   *  V8 promotes objects that live long (>2 minor collections) to “old space”.
   */
  (function gcExample3() {
    let persistent = {};
    for (let i = 0; i < 3; ++i) {
      // Allocate garbage to trigger minor collections
      const junk = new Array(1e5).fill(i);
    }
    // `persistent` has survived; now in old space (engine‑dependent).
    console.log('GC‑3: object likely promoted to old‑generation');
  })();
  
  /**
   *  GC‑Example‑4:  WeakMap prevents memory retention (Mark‑&‑Sweep skips Weak refs).
   */
  (function gcExample4() {
    const cache = new WeakMap();
    let key = {};
    cache.set(key, { payload: new Array(1e6) });
    console.log('GC‑4: cached');
    key = null;                    // Value now unreachable, GC can reclaim both key & value
  })();
  
  /**
   *  GC‑Example‑5:  FinalizationRegistry to observe collection (spec‑compliant).
   */
  (function gcExample5() {
    if (typeof FinalizationRegistry === 'function') {
      const registry = new FinalizationRegistry(() => console.log('GC‑5: object collected'));
      let obj = {};
      registry.register(obj, 'some‑token');
      obj = null;
    }
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION LEAKS ── Identifying & Fixing Memory Leaks                                           */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /**
   *  LEAK‑Example‑1:  Accidental globals.
   */
  (function leaksExample1() {
    function bad() {
      // Missing `let/const/var` -> becomes global
      leakyObj = new Array(1e6);
    }
    bad();
    // window.leakyObj (browser) or global.leakyObj (Node) leaks until manually removed.
  })();
  
  /**
   *  LEAK‑Example‑2:  Detached DOM nodes kept in array.
   */
  (function leaksExample2() {
    const cache = [];
    for (let i = 0; i < 5; ++i) {
      const div = document.createElement('div');
      div.textContent = i;
      document.body.appendChild(div);
      document.body.removeChild(div); // Detached from DOM
      cache.push(div);                // Still referenced -> leak
    }
    console.log('Leak‑2: Detached nodes cached =', cache.length);
  })();
  
  /**
   *  LEAK‑Example‑3:  Forgotten timers / intervals.
   */
  (function leaksExample3() {
    const id = setInterval(() => console.log('still alive'), 1e4);
    // clearInterval(id); // Uncomment to fix leak
  })();
  
  /**
   *  LEAK‑Example‑4:  Event listeners on long‑lived objects.
   */
  (function leaksExample4() {
    const btn = document.createElement('button');
    btn.textContent = 'Click';
    document.body.appendChild(btn);
    function onClick() { console.log('clicked'); }
    btn.addEventListener('click', onClick);
    setTimeout(() => {
      btn.removeEventListener('click', onClick); // Proper cleanup
      document.body.removeChild(btn);
    }, 5_000);
  })();
  
  /**
   *  LEAK‑Example‑5:  Unbounded in‑memory cache with Map.
   */
  (function leaksExample5() {
    const cache = new Map();
    fetch('/api/data')
      .then(r => r.json())
      .then(data => cache.set(Date.now(), data)); // Without eviction => memory grow forever
    // Solution: LRU or TTL eviction
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION PROF ── Profiling & Benchmarking                                                     */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /**
   *  PROF‑Example‑1:  console.time / timeEnd micro‑benchmark.
   */
  (function profExample1() {
    console.time('concat');
    let s = '';
    for (let i = 0; i < 1e5; ++i) s += 'a';
    console.timeEnd('concat'); // Compare vs template below
  })();
  
  /**
   *  PROF‑Example‑2:  performance.now() high‑resolution timing.
   */
  (function profExample2() {
    const t0 = performance.now();
    new Array(1e7).fill(0).map(Math.random);
    const t1 = performance.now();
    console.log(`Map cost = ${(t1 - t0).toFixed(2)} ms`);
  })();
  
  /**
   *  PROF‑Example‑3:  Flamechart via DevTools Performance panel.
   *  -----------------------------------------------------------
   *  Run heavyComputation(), then record in DevTools to inspect call‑stack.
   */
  function heavyComputation() {
    for (let i = 0; i < 200; ++i) {
      Fibonacci.recursive(30);
    }
  }
  const Fibonacci = {
    recursive(n) { return n < 2 ? n : this.recursive(n - 1) + this.recursive(n - 2); }
  };
  
  /**
   *  PROF‑Example‑4:  Memory snapshot comparison.
   *  --------------------------------------------
   *  Call leakGenerator() repeatedly, capture heap snapshots, diff them.
   */
  function leakGenerator() {
    window.__heap = window.__heap || [];
    window.__heap.push(new Array(1e6));
  }
  
  /**
   *  PROF‑Example‑5:  Node.js --prof & --inspect example (run in Node).
   *      node --inspect --prof ./thisFile.js
   *      node --prof-process isolate-*.log
   */
  (function profExample5() {
    if (typeof process !== 'undefined' && process.argv.includes('--demo-prof')) {
      for (let i = 0; i < 1e6; ++i) Math.sqrt(i);
    }
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION LOOPS ── Optimizing Loops & Recursion                                                */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /**
   *  LOOPS‑Example‑1:  Cache length in for‑loop.
   */
  (function loopsExample1() {
    const arr = Array.from({ length: 1e5 }, (_, i) => i);
    for (let i = 0, len = arr.length; i < len; ++i) { /* … */ }
  })();
  
  /**
   *  LOOPS‑Example‑2:  Prefer for…of vs forEach (avoids callback allocation).
   */
  (function loopsExample2() {
    const arr = Array(1e5).fill(0);
    console.time('forEach');
    arr.forEach((_, i) => arr[i]++);
    console.timeEnd('forEach');
  
    console.time('for‑of');
    for (const [i, v] of arr.entries()) arr[i] = v + 1;
    console.timeEnd('for‑of');
  })();
  
  /**
   *  LOOPS‑Example‑3:  Tail‑call optimization (spec mandated, but not all engines support).
   */
  (function loopsExample3() {
    'use strict';
    function factorial(n, acc = 1) {
      if (n <= 1) return acc;
      return factorial(n - 1, n * acc); // TCO expected
    }
    console.log('TCO fact(5)=', factorial(5));
  })();
  
  /**
   *  LOOPS‑Example‑4:  Memoization to avoid repeated recursion.
   */
  (function loopsExample4() {
    const memo = [0, 1];
    function fib(n) {
      if (memo[n] != null) return memo[n];
      return memo[n] = fib(n - 1) + fib(n - 2);
    }
    console.log('Memoized fib(40)=', fib(40));
  })();
  
  /**
   *  LOOPS‑Example‑5:  Unrolling small loops (sometimes micro‑win).
   */
  (function loopsExample5() {
    function unrolled(src) {
      const len = src.length;
      let acc0 = 0, acc1 = 0;
      for (let i = 0; i < len; i += 2) {
        acc0 += src[i];
        acc1 += src[i + 1];
      }
      return acc0 + acc1;
    }
    console.log('Sum=', unrolled(Array(1e6).fill(1)));
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION DOM ── Reflows / Repaints & DOM Batch Updates                                        */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /**
   *  DOM‑Example‑1:  Triggering vs batching style mutations.
   */
  (function domExample1() {
    const box = document.createElement('div');
    document.body.appendChild(box);
  
    // BAD: multiple reflows
    box.style.width = '100px';
    box.style.height = '100px';
    box.style.marginLeft = '10px';
  
    // GOOD: use cssText or class toggle (1 reflow)
    box.style.cssText = 'width:100px;height:100px;margin-left:10px;';
  })();
  
  /**
   *  DOM‑Example‑2:  Read then write (avoid interleave).
   */
  (function domExample2() {
    const el = document.body;
    const height = el.clientHeight;   // Read layout once
    // Many writes afterwards
    for (let i = 0; i < 5; ++i) el.style.paddingTop = `${height / 10}px`;
  })();
  
  /**
   *  DOM‑Example‑3:  DocumentFragment batching.
   */
  (function domExample3() {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 1000; ++i) {
      const li = document.createElement('li');
      li.textContent = i;
      frag.appendChild(li);
    }
    document.body.appendChild(frag); // Single reflow
  })();
  
  /**
   *  DOM‑Example‑4:  requestAnimationFrame for visual updates.
   */
  (function domExample4() {
    const ball = document.createElement('div');
    ball.style.cssText = 'position:fixed;top:0;left:0;width:20px;height:20px;background:red;';
    document.body.appendChild(ball);
  
    let t0;
    function animate(ts) {
      if (!t0) t0 = ts;
      const x = Math.min(ts - t0, 500);
      ball.style.transform = `translateX(${x}px)`;
      if (x < 500) requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  })();
  
  /**
   *  DOM‑Example‑5:  ResizeObserver batching.
   */
  (function domExample5() {
    const target = document.body;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) console.log('Size changed', entry.contentRect);
    });
    ro.observe(target);
    // Resize window to see batched callbacks
  })();
  
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  /* SECTION SPLIT ── Code Splitting & Lazy Loading                                               */
  /*───────────────────────────────────────────────────────────────────────────────────────────────*/
  
  /**
   *  SPLIT‑Example‑1:  Dynamic import() on user action.
   */
  document.addEventListener('click', async () => {
    const { default: showDialog } = await import('./dialog.js');
    showDialog();
  });
  
  /**
   *  SPLIT‑Example‑2:  IntersectionObserver for lazy images.
   */
  (function splitExample2() {
    const imgs = document.querySelectorAll('img[data-src]');
    const io = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.src = e.target.dataset.src;
          io.unobserve(e.target);
        }
      }
    });
    imgs.forEach(img => io.observe(img));
  })();
  
  /**
   *  SPLIT‑Example‑3:  React.lazy + Suspense (pseudo‑code inside JS file).
   */
  // const Chart = React.lazy(() => import('./Chart.jsx'));
  // function Dashboard() {
  //   return (
  //     <Suspense fallback={<Spinner/>}>
  //       <Chart/>
  //     </Suspense>
  //   );
  // }
  
  /**
   *  SPLIT‑Example‑4:  Preload chunk on hover (predictive).
   */
  (function splitExample4() {
    const btn = document.querySelector('#heavyFeature');
    btn?.addEventListener('pointerenter', () => import('./heavy-feature.js'));
  })();
  
  /**
   *  SPLIT‑Example‑5:  Webpack magic comments to name chunks.
   */
  // import(/* webpackChunkName: "admin-panel" */ './admin.js').then(initAdmin);
  
  /*──────────────────────────── End of Chapter 4 examples ───────────────────────────────────────*/