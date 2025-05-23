// Chapter 4 | Memory Management & Performance

// 1. Garbage Collection Algorithms
//    - Mark & Sweep
//    - Generational GC

// Example 1: Mark & Sweep (Conceptual Simulation)
function markAndSweepSimulation() {
    let memory = [
        { id: 1, marked: false, ref: [2] },
        { id: 2, marked: false, ref: [3] },
        { id: 3, marked: false, ref: [] },
        { id: 4, marked: false, ref: [] }, // Unreachable
    ];
    function mark(objId) {
        let obj = memory.find(o => o.id === objId);
        if (obj && !obj.marked) {
            obj.marked = true;
            obj.ref.forEach(mark);
        }
    }
    // Root is object 1
    mark(1);
    // Sweep
    memory = memory.filter(obj => obj.marked);
    return memory.map(obj => obj.id); // [1,2,3]
}

// Example 2: Generational GC (Conceptual Simulation)
function generationalGCSimulation() {
    let youngGen = [{ id: 'a' }, { id: 'b' }];
    let oldGen = [];
    // Promote 'a' to old generation after surviving GC
    youngGen = youngGen.filter(obj => obj.id !== 'a');
    oldGen.push({ id: 'a' });
    // Collect garbage in youngGen
    youngGen = youngGen.filter(obj => obj.id !== 'b');
    return { youngGen, oldGen }; // {youngGen: [], oldGen: [{id:'a'}]}
}

// Example 3: Forcing Garbage Collection (Node.js only, for demonstration)
function forceGCExample() {
    if (global.gc) {
        let arr = new Array(1e6).fill(0);
        arr = null;
        global.gc(); // Run with: node --expose-gc file.js
        return 'GC triggered';
    }
    return 'GC not available';
}

// Example 4: Memory Pressure and GC
function memoryPressureGC() {
    let arr = [];
    for (let i = 0; i < 1e5; i++) {
        arr.push({ data: new Array(1000).fill(i) });
    }
    arr = null; // Eligible for GC
    // GC will reclaim memory in next cycle
    return 'Memory released for GC';
}

// Example 5: Weak References (ES2021+)
function weakRefExample() {
    let obj = { data: 123 };
    let weak = new WeakRef(obj);
    obj = null; // Now only weakly referenced
    // After GC, weak.deref() may return undefined
    return typeof weak.deref();
}

// 2. Identifying & Fixing Memory Leaks

// Example 1: Global Variable Leak
function globalLeak() {
    leak = []; // Missing 'let' or 'var' creates global variable
    for (let i = 0; i < 1000; i++) leak.push(i);
    // Fix: use 'let leak = []'
    return typeof leak;
}

// Example 2: Closures Holding References
function closureLeak() {
    let big = new Array(1e6).fill(0);
    function leaky() { return big; }
    // Fix: set big = null when not needed
    big = null;
    return 'Leak fixed by nullifying reference';
}

// Example 3: Detached DOM Nodes
function domLeak() {
    let div = document.createElement('div');
    document.body.appendChild(div);
    document.body.removeChild(div);
    // Still referenced
    window.leakDiv = div;
    // Fix: window.leakDiv = null;
    return 'Leak fixed by removing external reference';
}

// Example 4: Event Listener Leak
function eventListenerLeak() {
    let btn = document.createElement('button');
    function handler() { /* ... */ }
    btn.addEventListener('click', handler);
    // Not removing listener before removing element
    btn.removeEventListener('click', handler); // Fix
    return 'Listener removed';
}

// Example 5: Caching Leak
function cacheLeak() {
    let cache = {};
    function addToCache(key, value) { cache[key] = value; }
    addToCache('big', new Array(1e6).fill(0));
    // Fix: Use WeakMap for object keys or clear cache
    cache = {};
    return 'Cache cleared';
}

// 3. Profiling & Benchmarking (DevTools)

// Example 1: Performance.now() Benchmark
function benchmarkLoop() {
    const start = performance.now();
    let sum = 0;
    for (let i = 0; i < 1e6; i++) sum += i;
    const end = performance.now();
    return end - start; // ms
}

// Example 2: console.time/console.timeEnd
function timeExample() {
    console.time('myTimer');
    let arr = [];
    for (let i = 0; i < 1e5; i++) arr.push(i);
    console.timeEnd('myTimer');
    return 'Timing complete';
}

// Example 3: Chrome DevTools Memory Snapshot
// Steps (not code): 
// 1. Open DevTools > Memory tab
// 2. Take Heap Snapshot before/after operation
// 3. Analyze retained objects

// Example 4: Performance Profiling (DevTools)
// Steps (not code):
// 1. Open DevTools > Performance tab
// 2. Record while running code
// 3. Analyze call stacks, scripting, rendering

// Example 5: Custom Benchmark Function
function customBenchmark(fn, iterations = 1000) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) fn();
    return performance.now() - start;
}

// 4. Optimizing Loops & Recursion

// Example 1: Loop Unrolling
function unrolledLoop(arr) {
    let sum = 0, i = 0, len = arr.length;
    for (; i + 3 < len; i += 4) {
        sum += arr[i] + arr[i+1] + arr[i+2] + arr[i+3];
    }
    for (; i < len; i++) sum += arr[i];
    return sum;
}

// Example 2: Avoiding Expensive Operations in Loops
function optimizedLoop(arr) {
    let len = arr.length, sum = 0;
    for (let i = 0; i < len; i++) sum += arr[i];
    return sum;
}

// Example 3: Tail Recursion (if supported)
function tailRecursionSum(n, acc = 0) {
    if (n === 0) return acc;
    return tailRecursionSum(n - 1, acc + n);
}

// Example 4: Memoization to Optimize Recursion
function fibMemo(n, memo = {}) {
    if (n <= 1) return n;
    if (memo[n]) return memo[n];
    return memo[n] = fibMemo(n-1, memo) + fibMemo(n-2, memo);
}

// Example 5: Using Array Methods for Performance
function arrayReduceSum(arr) {
    return arr.reduce((a, b) => a + b, 0);
}

// 5. Reflows/Repaints & DOM Batch Updates

// Example 1: Minimize Layout Thrashing
function layoutThrash() {
    let el = document.body;
    let width = el.offsetWidth; // Forces reflow
    el.style.width = (width + 10) + 'px';
    // Fix: batch reads/writes separately
    return 'Batched DOM access';
}

// Example 2: DocumentFragment for Batch DOM Updates
function batchDOMUpdate() {
    let frag = document.createDocumentFragment();
    for (let i = 0; i < 100; i++) {
        let div = document.createElement('div');
        div.textContent = i;
        frag.appendChild(div);
    }
    document.body.appendChild(frag);
    return 'Batch update complete';
}

// Example 3: requestAnimationFrame for Visual Updates
function animateBox(box) {
    function move() {
        box.style.left = (parseInt(box.style.left || 0) + 1) + 'px';
        if (parseInt(box.style.left) < 100) requestAnimationFrame(move);
    }
    requestAnimationFrame(move);
    return 'Animation started';
}

// Example 4: CSS Class Toggle Instead of Inline Styles
function toggleClass(el) {
    el.classList.add('active');
    // Instead of multiple style changes
    return 'Class toggled';
}

// Example 5: Avoid Synchronous Layout Queries in Loops
function avoidSyncLayout(elements) {
    // BAD: for (let el of elements) { el.offsetHeight; el.style.color = 'red'; }
    // GOOD:
    let heights = Array.from(elements, el => el.offsetHeight);
    elements.forEach(el => el.style.color = 'red');
    return 'Synchronous layout avoided';
}

// 6. Code Splitting & Lazy Loading

// Example 1: Dynamic Import (ES2020+)
async function loadModule() {
    const module = await import('./myModule.js');
    return module.default();
}

// Example 2: Lazy Loading Images
function lazyLoadImage(img) {
    img.src = img.dataset.src;
    return 'Image loaded lazily';
}

// Example 3: IntersectionObserver for Lazy Loading
function setupLazyLoad(img) {
    let observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.src = entry.target.dataset.src;
                observer.unobserve(entry.target);
            }
        });
    });
    observer.observe(img);
    return 'Observer set';
}

// Example 4: Webpack Code Splitting (Comment for context)
// import(/* webpackChunkName: "chunkA" */ './chunkA').then(module => module.doSomething());

// Example 5: Conditional Loading of Features
function loadFeature(condition) {
    if (condition) {
        import('./feature.js').then(module => module.init());
    }
    return 'Feature loaded conditionally';
}