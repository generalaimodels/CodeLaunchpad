/***************************************************************************\
|* 0. UTILITY HELPERS (indentation, banner printer)                        *|
\***************************************************************************/
const util = require('util');
let TAB = 0;
const IND = '  ';
const ts  = () => new Date().toISOString().slice(11, 23);
const pad = () => IND.repeat(TAB);
function banner(title) { console.log(`\n${'='.repeat(80)}\n>>> ${title}\n${'='.repeat(80)}`); }
function log(...a)     { console.log(`${ts()} ${pad()}${a.map(x => typeof x === 'string' ? x : util.inspect(x,{depth:3,colors:true})).join(' ')}`);}
function in_()         { TAB++; }
function out_()        { TAB = Math.max(0, TAB-1); }

/* 0A – basic banner & log */
function util0A() { banner('0A – hello utilities'); log('utility loaded'); }

/* 0B – indentation demo */
function util0B() { banner('0B – indentation'); log('lvl0'); in_(); log('lvl1'); in_(); log('lvl2'); out_(); out_(); }

/* 0C – util.inspect colours */
function util0C() { banner('0C – inspect'); log({ foo: 1, arr: [1,2,3] }); }

/* 0D – dynamic indent inside loop */
function util0D() { banner('0D – dynamic indent loop'); for(let i=0;i<3;i++){ in_(); log('depth',i); } TAB=0; }

/* 0E – utilities done */
function util0E() { banner('0E – utilities end'); }

/***************************************************************************\
|* 1. BASIC LIFO BEHAVIOR                                                  *|
\***************************************************************************/

/* 1A – simple call chain */
function lifo1A() {
  banner('1A – simple call chain');
  function A(){log('A enter'); B(); log('A exit');}
  function B(){log('  B enter'); C(); log('  B exit');}
  function C(){log('    C leaf');}
  A();
}

/* 1B – manual stack with array push/pop */
function lifo1B() {
  banner('1B – manual push/pop');
  const stack=[];
  function push(v){stack.push(v);log('push',v,'stack',stack);}
  function pop(){const v=stack.pop();log('pop',v,'stack',stack);}
  push(1);push(2);pop();push(3);pop();pop();
}

/* 1C – for‑loop illustrating LIFO exit */
function lifo1C() {
  banner('1C – loop calls');
  function loop(n){
    if(n===0){log('bottom');return;}
    log('call',n); loop(n-1); log('return',n);
  }
  loop(3);
}

/* 1D – value propagation through stack */
function lifo1D() {
  banner('1D – return propagation');
  function inc(x){return x+1;}
  function dbl(x){return inc(x)*2;}
  log('result', dbl(3));
}

/* 1E – LIFO with thrown error */
function lifo1E() {
  banner('1E – throw unwinds');
  function inner(){throw new Error('boom');}
  function outer(){inner();}
  try{outer();}catch(e){log('caught',e.message);}
}

/***************************************************************************\
|* 2. NESTED INVOCATION CHAIN                                              *|
\***************************************************************************/

/* 2A – depth 4 nested */
function nest2A() {
  banner('2A – depth 4');
  function a(){log('a');b();}
  function b(){log(' b');c();}
  function c(){log('  c');d();}
  function d(){log('   d');}
  a();
}

/* 2B – functional composition */
function nest2B() {
  banner('2B – composition');
  const add = x=>y=>x+y;
  const inc = add(1);
  const dbl = x=>x*2;
  const composed = x=>dbl(inc(x));
  log('compose(3)=', composed(3));
}

/* 2C – method chain (fluent API) */
function nest2C() {
  banner('2C – fluent API');
  const obj={
    v:0,
    add(n){this.v+=n;return this;},
    mul(n){this.v*=n;return this;},
    show(){log('value',this.v);}
  };
  obj.add(2).mul(5).show();
}

/* 2D – promise chain (sync values) */
function nest2D() {
  banner('2D – promise chain');
  Promise.resolve(3).then(x=>x+1).then(x=>x*2).then(log);
}

/* 2E – middleware style chain */
function nest2E() {
  banner('2E – middleware chain');
  const middleware=[(ctx,n)=>{ctx.push('a');n();},
                    (ctx,n)=>{ctx.push('b');n();},
                    (ctx,n)=>{ctx.push('c');}];
  function run(ctx){ let i=-1; const next=()=>{middleware[++i]&&middleware[i](ctx,next);} ; next(); }
  const ctx=[]; run(ctx); log('ctx',ctx);
}

/***************************************************************************\
|* 3. RECURSION & UNWINDING ORDER                                          *|
\***************************************************************************/

/* 3A – factorial with trace */
function rec3A() {
  banner('3A – factorial trace');
  function fact(n){ log('enter',n); const r=n<=1?1:n*fact(n-1); log('exit',n,'=',r); return r; }
  fact(4);
}

/* 3B – fibonacci memo */
function rec3B() {
  banner('3B – fib memo');
  const memo={0:0,1:1};
  function fib(n){ if(memo[n]!=null)return memo[n]; memo[n]=fib(n-1)+fib(n-2); return memo[n];}
  log('fib 8 =', fib(8), 'memo keys',Object.keys(memo));
}

/* 3C – recursive tree traversal */
function rec3C() {
  banner('3C – tree traversal');
  const tree={v:1,l:{v:2,l:null,r:null},r:{v:3,l:null,r:{v:4}}};
  function dfs(node){ if(!node)return; log(node.v); dfs(node.l); dfs(node.r);}
  dfs(tree);
}

/* 3D – mutual recursion (even/odd) */
function rec3D() {
  banner('3D – mutual recursion even/odd');
  const isEven=n=>n===0||isOdd(n-1);
  const isOdd =n=>n!==0&&isEven(n-1);
  log('5 even?',isEven(5),'4 even?',isEven(4));
}

/* 3E – recursion depth counter */
function rec3E() {
  banner('3E – depth counter');
  function depth(n,max){ if(n===max)return n; return depth(n+1,max); }
  log('depth', depth(0,10));
}

/***************************************************************************\
|* 4. STACK‑OVERFLOW DEMONSTRATION (commented)                             *|
\***************************************************************************/

/* 4A – naive unbounded recursion (DON’T CALL) */
function overflow4A() {
  // function boom(){ return boom(); }
  // boom(); // would crash
  banner('4A – commented overflow demo');
}

/* 4B – incremental depth until RangeError */
function overflow4B() {
  banner('4B – RangeError catch');
  let i=0;
  (function recurse(){ try{ i++; recurse(); }catch(e){ log('depth reached',i,e.message);} })();
}

/* 4C – stack size probe with safe cap */
function overflow4C() {
  banner('4C – probe depth 5k');
  let depth=0;
  function f(){ depth++; if(depth<5000)f(); }
  try{f();}catch(e){log('caught',e);}
  log('max depth',depth);
}

/* 4D – iterative replacement avoiding overflow */
function overflow4D() {
  banner('4D – iterative fibonacci');
  function fibIter(n){ let a=0,b=1; for(let i=0;i<n;i++){ [a,b]=[b,a+b]; } return a;}
  log('fib 1000 computed without stack overflow length', fibIter(1000).toString().length);
}

/* 4E – trampoline to avoid deep recursion */
function overflow4E() {
  banner('4E – trampoline factorial');
  const trampoline=fn=>(...args)=>{ let res=fn(...args); while(typeof res==='function'){ res=res(); } return res; };
  const factT=(n,acc=1)=> n<=1 ? acc : ()=>factT(n-1,n*acc);
  const safeFact=trampoline(factT);
  log('fact 20 =', safeFact(20));
}

/***************************************************************************\
|* 5. ASYNC CALLBACKS VS. SYNCHRONOUS STACK                                *|
\***************************************************************************/

/* 5A – setTimeout callback after sync stack empty */
function async5A() {
  banner('5A – timeout order');
  setTimeout(()=>log('timeout'),0);
  log('sync end');
}

/* 5B – Promise microtask before timeout */
function async5B() {
  banner('5B – microtask vs macrotask');
  Promise.resolve().then(()=>log('microtask then'));
  setTimeout(()=>log('macrotask'),0);
}

/* 5C – I/O callback vs sync */
function async5C() {
  banner('5C – fs.readFile callback');
  require('fs').readFile(__filename, ()=>log('I/O done'));
  log('after readFile call');
}

/* 5D – event loop starvation illustration */
function async5D() {
  banner('5D – compute blocks callbacks');
  setTimeout(()=>log('should delay'),0);
  const t=Date.now(); while(Date.now()-t<50){} // block 50ms
}

/* 5E – async/await unwrap stack */
async function async5E() {
  banner('5E – async/await stack simplification');
  async function foo(){ await Promise.resolve(); throw new Error('async err'); }
  try{ await foo(); }catch(e){ log(e.stack.split('\n').slice(0,3).join('\n')); }
}

/***************************************************************************\
|* 6. ERROR STACK TRACES & AUTOMATIC UNWINDING                             *|
\***************************************************************************/

/* 6A – capture stack */
function err6A() {
  banner('6A – new Error stack');
  const e=new Error('whoops'); log(e.stack.split('\n')[1]);
}

/* 6B – propagate and catch */
function err6B() {
  banner('6B – propagate');
  function inner(){throw new Error('bang');}
  function outer(){inner();}
  try{outer();}catch(e){log(e.stack);}
}

/* 6C – async throw loses sync frames */
function err6C() {
  banner('6C – async throw');
  setTimeout(()=>{ try{throw new Error('async');}catch(e){log('async stack',e.stack);} },0);
}

/* 6D – unhandledRejection demo */
function err6D() {
  banner('6D – unhandledRejection');
  process.once('unhandledRejection',r=>log('caught',r));
  Promise.reject('rej');
}

/* 6E – automatic unwinding with finally */
function err6E() {
  banner('6E – finally after throw');
  try{throw new Error('boom');}finally{log('finally always runs');}
}

/***************************************************************************\
|* 7. TRY / FINALLY & GUARANTEED UNWINDING                                 *|
\***************************************************************************/

/* 7A – file descriptor cleanup */
function tf7A() {
  banner('7A – fs.open finally');
  const fs=require('fs'); let fd;
  try{ fd=fs.openSync(__filename,'r'); const buf=Buffer.alloc(10); fs.readSync(fd,buf,0,10,0); }
  finally{ if(fd!=null){ fs.closeSync(fd); log('fd closed'); } }
}

/* 7B – lock release */
function tf7B() {
  banner('7B – fake lock');
  let locked=true;
  try{ if(!locked)throw new Error('not locked'); log('doing work'); }
  finally{ locked=false; log('lock released'); }
}

/* 7C – try/finally with return */
function tf7C() {
  banner('7C – return in try');
  function f(){ try{ return 1; } finally{ log('still runs'); } }
  log('f returns',f());
}

/* 7D – nested try/finally */
function tf7D() {
  banner('7D – nested finally order');
  try{ log('outer try'); try{ log('inner try'); } finally{ log('inner finally'); } }
  finally{ log('outer finally'); }
}

/* 7E – rethrow after finally */
function tf7E() {
  banner('7E – rethrow');
  try{ try{ throw new Error('err'); } finally{ log('cleanup'); } }
  catch(e){ log('caught outside'); }
}

/***************************************************************************\
|* 8. TAIL CALL OPTIMIZATION (TCO) NOTE + TRAMPOLINE POLYFILL              *|
\***************************************************************************/

/* 8A – normal recursion blows with big n */
function tco8A() {
  banner('8A – naive sum recursion');
  function sum(n,acc=0){ if(n===0)return acc; return sum(n-1,acc+n); }
  try{ sum(1e4); }catch(e){ log('stack overflow risk'); }
}

/* 8B – trampoline helper */
function tco8B() {
  banner('8B – trampoline sum');
  const tramp=fn=>(...a)=>{ let r=fn(...a); while(typeof r==='function') r=r(); return r; };
  const sumT=(n,acc=0)=> n===0 ? acc : ()=>sumT(n-1,acc+n);
  const safe=tramp(sumT);
  log('sum 10000 =', safe(10000));
}

/* 8C – CPS (continuation‑passing) */
function tco8C() {
  banner('8C – CPS factorial');
  const factCPS=(n,k)=> n===0 ? k(1) : factCPS(n-1,x=>k(n*x));
  factCPS(5, r=>log('fact 5',r));
}

/* 8D – iterative tail‑rec replacement */
function tco8D() {
  banner('8D – iterative tail recursion');
  function sumIter(n){ let acc=0; while(n)acc+=n--; return acc;}
  log('sumIter 1e6', sumIter(1e6));
}

/* 8E – generator based trampoline */
function tco8E() {
  banner('8E – generator trampoline');
  function* factG(n,acc=1){ while(true){ if(n<=1) return acc; acc*=n; n--; yield; } }
  function run(gen){ let it=gen; while(!it.next().done){} return it.return ? it.return().value : undefined; }
  log('factG 10', run(factG(10)));
}

/***************************************************************************\
|* MAIN – RUN EVERYTHING SEQUENTIALLY                                      *|
\***************************************************************************/
async function runAll(){
  const fns=[
    util0A,util0B,util0C,util0D,util0E,
    lifo1A,lifo1B,lifo1C,lifo1D,lifo1E,
    nest2A,nest2B,nest2C,nest2D,nest2E,
    rec3A,rec3B,rec3C,rec3D,rec3E,
    overflow4A,overflow4B,overflow4C,overflow4D,overflow4E,
    async5A,async5B,async5C,async5D,async5E,
    err6A,err6B,err6C,err6D,err6E,
    tf7A,tf7B,tf7C,tf7D,tf7E,
    tco8A,tco8B,tco8C,tco8D,tco8E
  ];
  for(const fn of fns){
    await new Promise(r=>setTimeout(r,20));
    await fn();
  }
}
runAll();