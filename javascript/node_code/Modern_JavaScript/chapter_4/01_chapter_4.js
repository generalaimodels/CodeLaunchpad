// Chapter 4 | Memory Management & Performance

// 1. Garbage Collection Algorithms
//    - Mark & Sweep
//    - Generational GC

// Example 1: Mark & Sweep (Conceptual Simulation)
function markAndSweepSimulation() {
    let memory = [
        { id: 1, marked: false },
        { id: 2, marked: false },
        { id: 3, marked: false }
    ];
    let roots = [memory[0], memory[2]];

    // Mark phase
    roots.forEach(obj => obj.marked = true);

    // Sweep phase
    memory = memory.filter(obj => obj.marked);

    return memory; // Only objects referenced by roots remain
}

// Example 2: Generational GC (Conceptual Simulation)
function generationalGCSimulation() {
    let youngGen = [{ id: 'a' }, { id: 'b' }];
    let oldGen = [{ id: 'x' }];

    // Simulate promotion
    let survived = youngGen.filter(obj => Math.random() > 0.5);
    oldGen = oldGen.concat(survived);
    youngGen = youngGen.filter(obj => !survived.includes(obj));

    return { youngGen, oldGen };
}

// Example 3: Forcing GC in Node.js (Not recommended in production)
function forceGCExample() {
    if (global.gc) {
        let arr = new Array(1e6).fill(0);
        arr = null;
        global.gc(); // Run with --expose-gc
    }
}

// Example 4: Memory Retention via Closures (GC can't collect)
function closureMemoryLeak() {
    let leak = [];
    return function() {
        leak.push(new Array(1e6).fill(0));
    };
}

// Example 5: Weak References (ES2021 WeakRef)
function weakRefExample() {
    let obj = { data: 123 };
    let weak = new WeakRef(obj);
    obj = null; // Now eligible for GC
    // weak.deref() may return undefined if GC has run
    return weak;
}

// 2. Identifying & Fixing Memory Leaks

// Example 1: Event Listener Leak
function eventListenerLeak() {
    const btn = document.createElement('button');
    function handler() { /* ... */ }
    btn.addEventListener('click', handler);
    // Not removing handler causes leak if btn is removed from DOM
    btn.removeEventListener('click', handler); // Fix
}

// Example 2: Detached DOM Nodes
function detachedDOMLeak() {
    let div = document.createElement('div');
    document.body.appendChild(div);
    document.body.removeChild(div);
    // If references to div remain, memory leak occurs
    div = null; // Fix
}

// Example 3: Global Variable Leak
function globalVariableLeak() {
    leak = new Array(1e6).fill(0); // Implicit global
    // Fix: use let/const/var
}

// Example 4: Closures Holding References
function closureLeak() {
    let largeObj = { data: new Array(1e6).fill(0) };
    return function() {
        // largeObj is never released
        return largeObj.data[0];
    };
    // Fix: nullify largeObj when not needed
}

// Example 5: Caching Unbounded Data
function cacheLeak() {
    const cache = {};
    function addToCache(key, value) {
        cache[key] = value;
    }
    // Fix: Use Map with size limit or WeakMap for object keys
}

// 3. Profiling & Benchmarking (DevTools)

// Example 1: Chrome DevTools Memory Snapshot
function memorySnapshotExample() {
    // Open DevTools > Memory > Take snapshot
    let arr = [];
    for (let i = 0; i < 1e5; i++) arr.push({ i });
    // Analyze retained objects
}

// Example 2: Performance Profiling
function performanceProfileExample() {
    console.time('loop');
    for (let i = 0; i < 1e6; i++) {}
    console.timeEnd('loop');
}

// Example 3: Heap Profiler
function heapProfilerExample() {
    // Open DevTools > Memory > Heap snapshot
    let leak = [];
    setInterval(() => leak.push(new Array(1e4)), 1000);
    // Observe growing heap
}

// Example 4: Timeline Recording
function timelineRecordingExample() {
    // DevTools > Performance > Record
    let arr = [];
    for (let i = 0; i < 1e6; i++) arr.push(i);
    // Stop recording, analyze JS heap and CPU usage
}

// Example 5: Benchmarking with Performance API
function performanceAPIExample() {
    performance.mark('start');
    let sum = 0;
    for (let i = 0; i < 1e6; i++) sum += i;
    performance.mark('end');
    performance.measure('sum', 'start', 'end');
    console.log(performance.getEntriesByName('sum'));
}

// 4. Optimizing Loops & Recursion

// Example 1: Loop Unrolling
function loopUnrollingExample(arr) {
    let sum = 0, i = 0, len = arr.length;
    for (; i + 3 < len; i += 4) {
        sum += arr[i] + arr[i+1] + arr[i+2] + arr[i+3];
    }
    for (; i < len; i++) sum += arr[i];
    return sum;
}

// Example 2: Avoiding Expensive Operations in Loops
function expensiveOperationLoop(arr) {
    // Bad: arr.length recalculated each iteration
    for (let i = 0; i < arr.length; i++) {}
    // Good:
    for (let i = 0, len = arr.length; i < len; i++) {}
}

// Example 3: Tail Recursion Optimization (not supported in all JS engines)
function tailRecursionSum(n, acc = 0) {
    if (n === 0) return acc;
    return tailRecursionSum(n - 1, acc + n);
}

// Example 4: Memoization to Optimize Recursion
function fibMemo(n, memo = {}) {
    if (n <= 1) return n;
    if (memo[n]) return memo[n];
    return memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
}

// Example 5: Using Array Methods for Performance
function arrayMethodPerformance(arr) {
    // forEach/map/filter are often optimized
    return arr.filter(x => x % 2 === 0).map(x => x * 2);
}

// 5. Reflows/Repaints & DOM Batch Updates

// Example 1: Triggering Reflow
function triggerReflow() {
    const el = document.createElement('div');
    document.body.appendChild(el);
    el.style.width = '100px';
    // Reading offsetWidth forces reflow
    const width = el.offsetWidth;
    el.style.width = '200px';
}

// Example 2: Batch DOM Updates with DocumentFragment
function batchDOMUpdates() {
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 1000; i++) {
        const div = document.createElement('div');
        frag.appendChild(div);
    }
    document.body.appendChild(frag);
}

// Example 3: Avoid Layout Thrashing
function layoutThrashing() {
    const el = document.getElementById('test');
    // Bad: alternating reads/writes
    for (let i = 0; i < 100; i++) {
        el.style.width = `${i}px`;
        const w = el.offsetWidth;
    }
    // Good: batch writes, then reads
}

// Example 4: Using requestAnimationFrame for Smooth Updates
function smoothUpdate() {
    function update() {
        // DOM update here
        requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// Example 5: CSS Class Toggle vs. Inline Styles
function classToggleVsInline() {
    const el = document.getElementById('test');
    // Better: toggle class for batch style changes
    el.classList.add('active');
    // Worse: multiple inline style changes
    el.style.width = '100px';
    el.style.height = '100px';
}

// 6. Code Splitting & Lazy Loading

// Example 1: Dynamic Import (ES2020)
async function dynamicImportExample() {
    const { add } = await import('./math.js');
    return add(2, 3);
}

// Example 2: Webpack Code Splitting (Comment for illustration)
// import(/* webpackChunkName: "lodash" */ 'lodash').then(_ => { /* ... */ });

// Example 3: Lazy Loading Images
function lazyLoadImages() {
    const imgs = document.querySelectorAll('img[data-src]');
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.src = entry.target.dataset.src;
                observer.unobserve(entry.target);
            }
        });
    });
    imgs.forEach(img => observer.observe(img));
}

// Example 4: React.lazy for Component Splitting
// const LazyComponent = React.lazy(() => import('./LazyComponent'));

// Example 5: Loading Scripts on Demand
function loadScriptOnDemand(src, callback) {
    const script = document.createElement('script');
    script.src = src;
    script.onload = callback;
    document.head.appendChild(script);
}