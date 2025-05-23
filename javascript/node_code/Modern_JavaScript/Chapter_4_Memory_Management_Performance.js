// Chapter 4: Memory Management & Performance
// ===========================================
// 1. Garbage Collection Algorithms (Mark & Sweep, Generational GC)
// ----------------------------------------------------------------

// Example 1: Simple object allocation and dereferencing (Mark & Sweep)
(function example1() {
    let obj = { a: 1 };
    console.log('Example1 before GC:', obj);
    obj = null; // obj is now unreachable -> eligible for GC
  })();
  
  // Example 2: Short‑lived objects in a loop (Generational hypothesis)
  (function example2() {
    for (let i = 0; i < 1e5; i++) {
      let temp = { idx: i }; // allocated in "young" generation
      // temp goes out of scope each iteration -> reclaimed quickly
    }
    console.log('Example2 done: Many short-lived objects created');
  })();
  
  // Example 3: Large array dereferenced (Mark & Sweep)
  (function example3() {
    let bigArray = new Array(1e6).fill('x');
    console.log('Example3 size before GC:', bigArray.length);
    bigArray = null; // entire large array becomes unreachable
  })();
  
  // Example 4: Circular references are handled by Mark & Sweep
  (function example4() {
    function Node() {
      this.next = null;
    }
    let a = new Node(), b = new Node();
    a.next = b;
    b.next = a;
    // Even though a & b reference each other, if we drop both, they get collected
    a = null; b = null;
    console.log('Example4: circular nodes dereferenced');
  })();
  
  // Example 5: Using WeakRef to avoid retaining in old generation
  (function example5() {
    let obj = { value: 'I might vanish' };
    const weak = new WeakRef(obj);
    console.log('Example5 before nullify:', weak.deref()?.value);
    obj = null; // only weak holds it -> eligible for GC even if referenced
  })();
  
  
  
  // 2. Identifying & Fixing Memory Leaks
  // -------------------------------------
  
  // Example 1: Global variable leak
  (function leak1() {
    leaked = { name: 'global' }; // forgot var/let/const
    // Fix: use let/const to scope properly
  })();
  
  // Example 2: Forgotten timers
  (function leak2() {
    let count = 0;
    const id = setInterval(() => { count++; }, 1000);
    // Later, clearInterval(id) when no longer needed
    clearInterval(id);
  })();
  
  // Example 3: Detached DOM nodes
  (function leak3() {
    const div = document.createElement('div');
    document.body.appendChild(div);
    document.body.removeChild(div);
    // If we still hold `div` references elsewhere, it leaks.
    // Fix: nullify references: div = null;
  })();
  
  // Example 4: Closures holding large contexts
  (function leak4() {
    function createHeavy() {
      const big = new Array(1e5).fill('*');
      return () => console.log(big[0]);
    }
    const fn = createHeavy();
    // If fn is no longer used, set fn = null to free `big`.
  })();
  
  // Example 5: Event listeners on removed elements
  (function leak5() {
    const btn = document.createElement('button');
    document.body.appendChild(btn);
    const handler = () => console.log('clicked');
    btn.addEventListener('click', handler);
    document.body.removeChild(btn);
    // btn and handler still in memory -> remove listener first:
    // btn.removeEventListener('click', handler);
  })();
  
  
  
  // 3. Profiling & Benchmarking (DevTools)
  // ---------------------------------------
  
  // Example 1: console.time / console.timeEnd
  (function bench1() {
    console.time('bench1');
    for (let i = 0; i < 1e6; i++) { Math.sqrt(i); }
    console.timeEnd('bench1');
  })();
  
  // Example 2: performance.now()
  (function bench2() {
    const t0 = performance.now();
    for (let i = 0; i < 1e6; i++) { Math.log(i + 1); }
    const t1 = performance.now();
    console.log('bench2:', (t1 - t0).toFixed(2), 'ms');
  })();
  
  // Example 3: DevTools CPU profiler (manual step)
  // In Chrome DevTools: open Performance panel → click Record → run function → Stop → analyze flame chart.
  
  function heavyTask() {
    let sum = 0;
    for (let i = 0; i < 5e7; i++) sum += i;
    return sum;
  }
  // Call heavyTask() while recording in DevTools to see hotspots.
  
  // Example 4: performance.mark & performance.measure
  (function bench4() {
    performance.mark('start');
    [...Array(1e5)].map((_, i) => i * 2);
    performance.mark('end');
    performance.measure('mapTransform', 'start', 'end');
    console.log(performance.getEntriesByName('mapTransform'));
  })();
  
  // Example 5: Memory profiling (manual step)
  // In Chrome DevTools: open Memory panel → take Heap snapshot → allocate → take another → diff to find leaks.
  
  
  
  // 4. Optimizing Loops & Recursion
  // --------------------------------
  
  // Example 1: for vs forEach
  (function loop1() {
    const arr = Array.from({ length: 1e6 }, (_, i) => i);
    console.time('for'); for (let i = 0; i < arr.length; i++); console.timeEnd('for');
    console.time('forEach'); arr.forEach(() => {}); console.timeEnd('forEach');
  })();
  
  // Example 2: while is sometimes faster than for
  (function loop2() {
    const n = 1e6;
    console.time('for'); for (let i = 0; i < n; i++); console.timeEnd('for');
    console.time('while'); let j = 0; while (j++ < n); console.timeEnd('while');
  })();
  
  // Example 3: loop unrolling
  (function loop3() {
    const n = 1e6;
    console.time('normal');
    for (let i = 0; i < n; i++) Math.sqrt(i);
    console.timeEnd('normal');
  
    console.time('unrolled');
    for (let i = 0; i < n; i += 4) {
      Math.sqrt(i); Math.sqrt(i+1); Math.sqrt(i+2); Math.sqrt(i+3);
    }
    console.timeEnd('unrolled');
  })();
  
  // Example 4: tail recursion (where supported)
  (function loop4() {
    function factorial(n, acc = 1) {
      return n === 0 ? acc : factorial(n - 1, acc * n);
    }
    console.log('factorial(10):', factorial(10));
  })();
  
  // Example 5: memoization for recursive fib
  (function loop5() {
    const memo = {};
    function fib(n) {
      if (n < 2) return n;
      if (memo[n]) return memo[n];
      return memo[n] = fib(n-1) + fib(n-2);
    }
    console.log('fib(30):', fib(30));
  })();
  
  
  
  // 5. Reflows/Repaints & DOM Batch Updates
  // ----------------------------------------
  
  // Example 1: Multiple style writes vs batch
  (function dom1() {
    const el = document.createElement('div');
    document.body.appendChild(el);
    console.time('multiple');
    el.style.width = '100px';
    el.style.height = '100px';
    el.style.background = 'red';
    console.timeEnd('multiple');
  
    console.time('batch');
    el.style.cssText = 'width:100px;height:100px;background:red';
    console.timeEnd('batch');
  })();
  
  // Example 2: Reading layout triggers reflow
  (function dom2() {
    const el = document.body;
    el.style.padding = '10px';
    console.time('forcedReflow');
    const h = el.offsetHeight; // forced reflow here
    el.style.margin = '10px';
    console.timeEnd('forcedReflow');
  })();
  
  // Example 3: Use DocumentFragment for many inserts
  (function dom3() {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 1000; i++) {
      const li = document.createElement('li');
      li.textContent = `Item ${i}`;
      frag.appendChild(li);
    }
    console.time('fragment');
    document.body.appendChild(frag);
    console.timeEnd('fragment');
  })();
  
  // Example 4: requestAnimationFrame batching
  (function dom4() {
    const box = document.createElement('div');
    document.body.appendChild(box);
    let x = 0;
    function animate() {
      x += 5;
      box.style.transform = `translateX(${x}px)`;
      if (x < 200) requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  })();
  
  // Example 5: classList.toggle vs style property
  (function dom5() {
    const el = document.createElement('div');
    document.body.appendChild(el);
    console.time('style');
    el.style.display = 'none';
    el.style.display = 'block';
    console.timeEnd('style');
  
    console.time('class');
    el.classList.add('hidden');
    el.classList.remove('hidden');
    console.timeEnd('class');
  })();
  
  
  
  // 6. Code Splitting & Lazy Loading
  // ---------------------------------
  
  // Example 1: Dynamic import (ESM)
  (async function cs1() {
    // assume './module.js' exports function greet()
    const { greet } = await import('./module.js');
    greet();
  })();
  
  // Example 2: Conditional load
  (function cs2() {
    if (window.innerWidth < 600) {
      import('./mobile.js').then(m => m.initMobileUI());
    } else {
      import('./desktop.js').then(m => m.initDesktopUI());
    }
  })();
  
  // Example 3: Lazy loading on user interaction
  (function cs3() {
    document.getElementById('loadBtn').addEventListener('click', async () => {
      const mod = await import('./heavyFeature.js');
      mod.run();
    });
  })();
  
  // Example 4: Webpack magic comments for chunk names
  (async function cs4() {
    const mod = await import(
      /* webpackChunkName: "analytics" */ './analytics.js'
    );
    mod.trackPage();
  })();
  
  // Example 5: Simulated route-based loader
  const routes = {
    '/home': () => import('./home.js'),
    '/about': () => import('./about.js'),
  };
  function navigate(path) {
    routes[path]().then(m => m.render());
  }
  navigate('/home');