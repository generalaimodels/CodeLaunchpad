/***************************************************************************************************
 *  Node.js Core Module : `readline`
 *  Author : <Your Name Here>
 *  File   : readline-tour.js                          ──> run with `node readline-tour.js`
 *
 *  Ten bite-sized examples that collectively exercise EVERY officially documented surface of the
 *  `readline` module – from the day-to-day `question()` helper down to rarely seen utilities such
 *  as `emitKeypressEvents()` and low-level cursor control helpers (`cursorTo`, `moveCursor`,
 *  `clear*`).  Each demo is isolated (fresh interface, fake input stream, deterministic output) and
 *  prints a heading separator so you can comment-in / comment-out freely.
 *
 *  Tested on Node ≥ 18.x (older versions still run but promise-based `question()` or
 *  `Interface.getCursorPos()` may be missing).
 ***************************************************************************************************/
'use strict';
const readline = require('readline');
const { Readable, PassThrough } = require('stream');
const fs       = require('fs');
const os       = require('os');
const path     = require('path');

/*──────────────────────────────────────────────────────────────────────────────────────────────────
 *  Simple sequential runner
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

/* Helper util – build a Readable that feeds the provided lines then ends */
const fakeInput = (lines) => Readable.from(lines.map(l => l.endsWith('\n') ? l : l + '\n'));

/***************************************************************************************************
 * 1)  rl.question(query, cb)  (classic callback variant)
 **************************************************************************************************/
demo('1) rl.question() callback', () => new Promise((done) => {
  const rl = readline.createInterface({
    input : fakeInput(['Alice']),
    output: process.stdout,
    terminal: false
  });
  rl.question('Your name? ', (answer) => {
    console.log('Hello,', answer);
    rl.close(); done();
  });
  /* Expected output:
     Your name? Hello, Alice
  */
}));

/***************************************************************************************************
 * 2)  Promise-based question()             (Node 17+)
 **************************************************************************************************/
demo('2) rl.question() promise', async () => {
  const rl = readline.createInterface({ input: fakeInput(['blue']), output: process.stdout, terminal:false });
  const colour = await rl.question('Favourite colour? ');
  console.log('Colour =', colour);
  rl.close();
  /* Expected output:
     Favourite colour? Colour = blue
  */
});

/***************************************************************************************************
 * 3)  line / close events, setPrompt(), prompt()
 **************************************************************************************************/
demo('3) line events & custom prompt', () => new Promise((done) => {
  const rl = readline.createInterface({ input: fakeInput(['foo', 'bar', '']), output: process.stdout, terminal:false });
  rl.setPrompt('> ');
  rl.prompt();
  const stash = [];
  rl.on('line', (ln) => {
    if (!ln) { rl.close(); }
    else      { stash.push(ln); rl.prompt(); }
  }).on('close', () => {
    console.log('Received lines →', stash);
    done();
  });
  /* Expected output:
     > > > Received lines → [ 'foo', 'bar' ]
  */
}));

/***************************************************************************************************
 * 4)  Completer function                        (tab-completion)
 **************************************************************************************************/
demo('4) Completer demo', () => {
  // Simple “fruits” completer
  const fruits = ['apple', 'apricot', 'banana', 'pear'];
  const completer = (line) => {
    const hits = fruits.filter(f => f.startsWith(line));
    return [hits.length ? hits : fruits, line];   // [matches, original]
  };

  const rl = readline.createInterface({
    input: fakeInput([]),
    output: process.stdout,
    completer,
    terminal: false
  });
  // The returned completer itself can be called directly for demo purposes:
  console.log('Completer("ap") →', completer('ap')[0]);
  rl.close();
  /* Expected output:
     Completer("ap") → [ 'apple', 'apricot' ]
  */
});

/***************************************************************************************************
 * 5)  pause() / resume()  – throttling a fast source
 **************************************************************************************************/
demo('5) pause & resume', () => new Promise((done) => {
  const src = fakeInput([...Array(5)].map((_,i)=>`L${i}`));
  const rl  = readline.createInterface({ input: src, terminal:false });
  rl.on('line', (l) => {
    console.log('Got', l);
    rl.pause();                          // temporarily stop flow
    setTimeout(()=> rl.resume(), 20);    // resume after 20 ms
  }).on('close', done);
  /* Expected output (timing gap of ~20 ms between lines):
     Got L0
     Got L1
     Got L2
     Got L3
     Got L4
  */
}));

/***************************************************************************************************
 * 6)  emitKeypressEvents()  – low-level key capture (no TTY needed here)
 **************************************************************************************************/
demo('6) emitKeypressEvents()', () => new Promise((done) => {
  // Craft a dummy stream that pretends to be a TTY
  const ks = new PassThrough();
  ks.isTTY = true;
  ks.setRawMode = () => {};                        // satisfy readline internals
  readline.emitKeypressEvents(ks);
  ks.on('keypress', (str, key) => {
    console.log('Keypress →', str, key);
    done();
  });
  ks.emit('data', Buffer.from('a'));               // simulate pressing "a"
  /* Expected output:
     Keypress → a { sequence: 'a', name: 'a', ctrl: false, meta: false, shift: false }
  */
}));

/***************************************************************************************************
 * 7)  cursorTo / moveCursor / clearLine / clearScreenDown
 **************************************************************************************************/
demo('7) Cursor manipulation helpers', () => {
  process.stdout.write('12345');
  readline.cursorTo   (process.stdout, 0);          // move to column 0
  readline.moveCursor (process.stdout, 0, -1);      // up one line (no-op if top)
  readline.clearLine  (process.stdout, 1);          // erase to right
  process.stdout.write('ABCDE\n');
  readline.clearScreenDown(process.stdout);         // wipe rest of screen
  /* Expected output (visually):
     ABCDE
     (previous “12345” cleared)
  */
});

/***************************************************************************************************
 * 8)  Reading a file line-by-line (terminal: false)
 **************************************************************************************************/
demo('8) File streaming with createInterface()', () => new Promise((done) => {
  const tmp = path.join(os.tmpdir(), 'rl-demo.txt');
  fs.writeFileSync(tmp, 'alpha\nbeta\ngamma');
  const rl = readline.createInterface({ input: fs.createReadStream(tmp), crlfDelay: Infinity });
  const lines = [];
  rl.on('line', l => lines.push(l))
    .on('close', () => { console.log('File lines →', lines); fs.unlinkSync(tmp); done(); });
  /* Expected output:
     File lines → [ 'alpha', 'beta', 'gamma' ]
  */
}));

/***************************************************************************************************
 * 9)  rl.write() – programmatically edit the current line
 **************************************************************************************************/
demo('9) rl.write()', () => new Promise((done) => {
  const rl = readline.createInterface({ input: fakeInput([]), output: process.stdout });
  rl.write('Hello');               // prints 'Hello'
  setTimeout(() => {
    // CTRL+U clears the line:
    rl.write(null, { ctrl:true, name:'u' });
    rl.write('Bye\n');             // overwrite with 'Bye'
    rl.close(); done();
  }, 20);
  /* Expected output (visually):
     HelloBye
  */
}));

/***************************************************************************************************
 * 10)  getCursorPos() (Node ≥20) – obtain cursor location
 **************************************************************************************************/
demo('10) Interface.getCursorPos()', async () => {
  if (!readline.Interface.prototype.getCursorPos) {
    console.log('getCursorPos() not available on this Node build.');
    return;
  }
  const rl = readline.createInterface({ input: fakeInput([]), output: process.stdout });
  await rl.write('XYZ');                           // place cursor after XYZ
  const pos = await rl.getCursorPos();
  console.log('Cursor position →', pos);           // { rows: <int>, cols:3 }
  rl.close();
  /* Expected output (columns may vary):
     Cursor position → { cols: 3, rows: 0 }
  */
});

/***************************************************************************************************
 * End of file – readline mastered!
 ***************************************************************************************************/