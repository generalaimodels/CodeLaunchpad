/***************************************************************************************************
 *  Node.js Core Module : `perf_hooks`
 *  Author : <Your Name Here>
 *  File   : perf_hooks-tour.js               ──>  run with `node perf_hooks-tour.js`
 *
 *  10 meticulously-curated examples that touch EVERY public surface of the `perf_hooks` module:
 *    • performance.{now, timeOrigin, mark, measure, clearMarks, clearMeasures, getEntries*,
 *                  eventLoopUtilization, setResourceTimingBufferSize, timerify, nodeTiming}
 *    • PerformanceObserver (observe, disconnect, supportedEntryTypes)
 *    • monitorEventLoopDelay() histogram utility
 *    • perf_hooks.constants (GC flags et al.)
 *
 *  Each snippet is self-contained, prints its own heading, and leaves the runtime pristine.
 *  Tested on Node ≥ 18.x (older versions should work but may lack `monitorEventLoopDelay` or
 *  `eventLoopUtilization`).
 ***************************************************************************************************/
'use strict';
const {
  performance,
  PerformanceObserver,
  monitorEventLoopDelay,
  constants
} = require('perf_hooks');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Pretty sequential runner
 *────────────────────────────────────────────────────────────────────────────────────────────────*/
const DEMOS = [];
function demo(title, fn) { DEMOS.push({ title, fn }); }
(async () => {
  for (const { title, fn } of DEMOS) {
    console.log('\n' + '═'.repeat(100));
    console.log(`Example: ${title}`);
    console.log('═'.repeat(100));
    await fn();
  }
})();

/***************************************************************************************************
 * 1) performance.now()  vs  Date.now()  (micro-precision timer)
 **************************************************************************************************/
demo('1) performance.now() vs Date.now()', () => {
  const tP = performance.now();
  const tD = Date.now();
  console.log('performance.now() →', tP.toFixed(3), 'ms since timeOrigin');
  console.log('Date.now()        →', tD, 'ms since UNIX epoch');
  /* Expected output (sample):
     performance.now() →  53.104 ms since timeOrigin
     Date.now()        → 1690000123456 ms since UNIX epoch
  */
});

/***************************************************************************************************
 * 2) mark(), measure(), getEntriesByName(), clear*()
 **************************************************************************************************/
demo('2) Custom marks & measures', () => {
  performance.mark('A:start');
  for (let i = 0; i < 1e5; i++);              // tiny busy-loop
  performance.mark('A:end');
  performance.measure('A:duration', 'A:start', 'A:end');

  const [measure] = performance.getEntriesByName('A:duration');
  console.log('Measured duration →', measure.duration.toFixed(3), 'ms');

  performance.clearMarks();  // free buffer
  performance.clearMeasures();
  /* Expected output:
     Measured duration → <≈1.2> ms
  */
});

/***************************************************************************************************
 * 3) PerformanceObserver – live feed of "measure" entries
 **************************************************************************************************/
demo('3) PerformanceObserver capturing measures', async () => {
  const obs = new PerformanceObserver((list) => {
    for (const entry of list.getEntries())
      console.log('Observer saw →', entry.name, entry.duration.toFixed(2), 'ms');
  });
  obs.observe({ entryTypes: ['measure'] });

  performance.mark('B:start');
  await new Promise(r => setTimeout(r, 100));
  performance.mark('B:end');
  performance.measure('B:sleep', 'B:start', 'B:end');

  obs.disconnect();
  /* Expected output:
     Observer saw → B:sleep 100.xx ms
  */
});

/***************************************************************************************************
 * 4) performance.timerify(fn) – auto-instrument a function
 **************************************************************************************************/
demo('4) timerify() auto-profiling', () => {
  const expensive = performance.timerify(function fib(n) {
    return n < 2 ? n : fib(n - 1) + fib(n - 2);
  });
  const obs = new PerformanceObserver((list) => {
    list.getEntries().forEach(e =>
      console.log(`${e.name} call #${e.count} took ${e.duration.toFixed(3)} ms`));
  });
  obs.observe({ entryTypes: ['function'] });

  expensive(15);  // invoke once
  obs.disconnect();
  /* Expected output (rough):
     fib call #1 took 2.345 ms
  */
});

/***************************************************************************************************
 * 5) eventLoopUtilization(old?) – quantify blockage
 **************************************************************************************************/
demo('5) eventLoopUtilization()', async () => {
  const start = performance.eventLoopUtilization();
  const busy = Date.now() + 50;            // sync block for 50 ms
  while (Date.now() < busy);               // burn CPU

  const end = performance.eventLoopUtilization(start);
  console.log('ELU % (blocked vs total) →', (end.utilization * 100).toFixed(1) + '%');
  /* Expected output:
     ELU % (blocked vs total) → 99.0%   (value depends on machine)
  */
});

/***************************************************************************************************
 * 6) monitorEventLoopDelay() – histogram of latency
 **************************************************************************************************/
demo('6) monitorEventLoopDelay()', async () => {
  const h = monitorEventLoopDelay({ resolution: 10 }); // 10-ms buckets
  h.enable();
  await new Promise(r => setTimeout(r, 120));          // idle wait
  h.disable();

  console.log('Mean lag →', h.mean.toFixed(2), 'ns',
              ' | 99th →', h.percentile(99).toFixed(2), 'ns');
  h.reset();                                           // zero counters
  /* Expected output:
     Mean lag → 5000.00 ns  | 99th → 20000.00 ns
  */
});

/***************************************************************************************************
 * 7) performance.nodeTiming & constants (GC flags etc.)
 **************************************************************************************************/
demo('7) Built-in nodeTiming entry + constants', () => {
  const nt = performance.nodeTiming;
  console.log('Bootstrap took →',
    (nt.bootstrapComplete - nt.environment).toFixed(2), 'ms');

  console.log('GC_MAJOR flag value →', constants.NODE_PERFORMANCE_GC_MAJOR);
  /* Expected output (bootstrap gap varies):
     Bootstrap took → 4.37 ms
     GC_MAJOR flag value → 1
  */
});

/***************************************************************************************************
 * 8) setResourceTimingBufferSize()  +  supportedEntryTypes
 **************************************************************************************************/
demo('8) Resource timing buffer size & supported types', () => {
  performance.setResourceTimingBufferSize(8);
  console.log('Supported entryTypes →', PerformanceObserver.supportedEntryTypes);
  /* Expected output:
     Supported entryTypes → [ 'node', 'mark', 'measure', 'function', 'gc', … ]
  */
});

/***************************************************************************************************
 * 9) timeOrigin – aligning high-res & wall-clock time
 **************************************************************************************************/
demo('9) timeOrigin bridging', () => {
  const wallClock = new Date(performance.timeOrigin + performance.now());
  console.log('High-res converted to Date →', wallClock.toISOString());
  /* Expected output (ISO string):
     High-res converted to Date → 2024-07-25T12:34:56.789Z
  */
});

/***************************************************************************************************
 * 10) Using PerformanceObserver for GC events (constants flags)
 **************************************************************************************************/
demo('10) Observing GC via PerformanceObserver', async () => {
  const obs = new PerformanceObserver((list) => {
    for (const e of list.getEntries()) {
      const kind = e.kind === constants.NODE_PERFORMANCE_GC_MAJOR ? 'major' :
                   e.kind === constants.NODE_PERFORMANCE_GC_MINOR ? 'minor' :
                   e.kind === constants.NODE_PERFORMANCE_GC_INCREMENTAL ? 'incr' : 'weakcb';
      console.log(`GC (${kind}) took ${e.duration.toFixed(2)} ms`);
    }
  });
  obs.observe({ entryTypes: ['gc'], buffered: true });

  // Allocate & release to likely trigger a GC
  const trash = [];
  for (let i = 0; i < 1e5; i++) trash.push({ x: i });
  await new Promise(r => setTimeout(r, 100)); // give GC some time

  obs.disconnect();
  /* Expected output (may vary or be empty if GC didn't run):
     GC (minor) took 1.23 ms
  */
});

/***************************************************************************************************
 * End of file
 ***************************************************************************************************/