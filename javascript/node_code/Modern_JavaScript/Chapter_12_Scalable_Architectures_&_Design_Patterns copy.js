/**************************************************************************************************
 * Chapter 12 | Scalable Architectures & Design Patterns
 * -----------------------------------------------------------------------------------------------
 * Single‑file reference playground. 5 sections × 5 concise, runnable examples each.
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION MVC — MVC, MVVM, Flux & Redux                           */
/*───────────────────────────────────────────────────────────────────*/

/* MVC‑Example‑1:  Classic MVC (Vanilla JS) */
(function () {
    class Model { constructor() { this.data = 0; this.subs = []; } inc(){ this.data++; this.notify(); }
      subscribe(fn){ this.subs.push(fn);} notify(){ this.subs.forEach(fn=>fn(this.data)); } }
    class View { constructor(ctrl){ this.btn=document.createElement('button');this.btn.textContent='Click';
      this.out=document.createElement('span');document.body.append(this.btn,this.out);
      this.btn.onclick=()=>ctrl.handleClick(); }
      render(v){ this.out.textContent=' '+v; } }
    class Controller { constructor(m,v){ this.m=m; this.v=v; m.subscribe(v.render.bind(v)); }
      handleClick(){ this.m.inc(); } }
    new Controller(new Model(), new View());
  })();
  
  /* MVC‑Example‑2:  MVVM via Proxy (auto‑sync) */
  (function () {
    const vm = new Proxy({ text: 'hello' }, { set(o,k,v){ o[k]=v; document.querySelectorAll(`[data-bind=${k}]`)
      .forEach(el=>el.textContent=v); return true;}});
    const span=document.createElement('span'); span.dataset.bind='text'; document.body.appendChild(span);
    setTimeout(()=>vm.text='world',1000);
  })();
  
  /* MVC‑Example‑3:  Flux dispatcher skeleton */
  (function () {
    const dispatcher = (()=>{ const handlers={};return{register(a,h){handlers[a]=h;},
      dispatch(a,p){handlers[a]?.(p);} };})();
    dispatcher.register('ADD', p => console.log('MVC‑3 action ADD', p));
    dispatcher.dispatch('ADD',{x:1});
  })();
  
  /* MVC‑Example‑4:  Minimal Redux store */
  (function () {
    const createStore = (reducer, state) => ({
      getState:()=>state,
      dispatch: a => {state=reducer(state,a); listeners.forEach(l=>l());},
      subscribe:l=>listeners.push(l)}), listeners=[];
    const reducer = (s={c:0},a)=>a.type==='INC'?{c:s.c+1}:s;
    const store=createStore(reducer);
    store.subscribe(()=>console.log('MVC‑4 count',store.getState().c));
    store.dispatch({type:'INC'});
  })();
  
  /* MVC‑Example‑5:  Redux middleware logger */
  (function () {
    const applyMW = (store, mw) => { const d=store.dispatch; store.dispatch=mw(store)(d); };
    const store={ state:0, dispatch:a=>{ if(a.type==='INC')store.state++;}, getState:()=>store.state };
    const logger=s=>n=>a=>{console.log('prev',s.getState());n(a);console.log('next',s.getState());};
    applyMW(store, logger); store.dispatch({type:'INC'});
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION OBS — Observer, Pub/Sub & Event Sourcing                */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* OBS‑Example‑1:  Observer pattern class */
  class Subject {
    #subs = new Set();
    subscribe(fn){ this.#subs.add(fn);} unsubscribe(fn){this.#subs.delete(fn);}
    next(v){ this.#subs.forEach(fn=>fn(v)); }
  }
  const subj = new Subject();
  subj.subscribe(v=>console.log('OBS‑1',v)); subj.next(42);
  
  /* OBS‑Example‑2:  Pub/Sub with topics */
  (function () {
    const bus={}; const sub=(t,h)=>(bus[t]=bus[t]||[]).push(h);
    const pub=(t,d)=>(bus[t]||[]).forEach(h=>h(d));
    sub('news',d=>console.log('OBS‑2',d)); pub('news','event happened');
  })();
  
  /* OBS‑Example‑3:  RxJS observable (browser/node w/ esm.run) */
  (async () => {
    try {
      const { Observable } = await import('https://esm.run/rxjs?bundle');
      Observable.interval=ms=>new Observable(s=>{let i=0; const id=setInterval(()=>s.next(i++),ms);return()=>clearInterval(id);});
      Observable.interval(500).subscribe(v=>{ if(v>2)this.unsubscribe; console.log('OBS‑3',v); });
    } catch{}
  })();
  
  /* OBS‑Example‑4:  Event Sourcing append‑only log */
  (function () {
    const log=[]; const apply=(state,e)=>{switch(e.type){case'ADD':state+=e.v;break;}return state;};
    const dispatch=e=>{log.push(e); state=apply(state,e);} ; let state=0;
    dispatch({type:'ADD',v:5}); dispatch({type:'ADD',v:3});
    const replay=log.reduce(apply,0); console.log('OBS‑4 replay',replay);
  })();
  
  /* OBS‑Example‑5:  Event emitter using Node built‑in */
  (function () {
    const { EventEmitter } = require('events');
    const ee=new EventEmitter();
    ee.on('tick',v=>console.log('OBS‑5 tick',v));
    ee.emit('tick',Date.now());
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION FSM — CQRS, State Machines & Finite Automata            */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* FSM‑Example‑1:  CQRS command + query separation */
  (function () {
    const store={items:[]};
    const Commands={add:p=>store.items.push(p)};
    const Queries={count:()=>store.items.length};
    Commands.add('a'); Commands.add('b');
    console.log('FSM‑1 count',Queries.count());
  })();
  
  /* FSM‑Example‑2:  Simple finite automaton for binary multiple of 3 */
  (function () {
    const trans=[[0,1],[2,0],[1,2]]; let state=0;
    const accept=s=>{state=0; for(const c of s)state=trans[state][c]; return state===0;};
    console.log('FSM‑2 "110" multiple3?',accept('110'));
  })();
  
  /* FSM‑Example‑3:  State machine library (xstate lite) */
  (async () => {
    try {
      const { createMachine, interpret } = await import('https://esm.run/xstate@4?bundle');
      const machine=createMachine({id:'toggle',initial:'off',states:{off:{on:{TOGGLE:'on'}},on:{on:{TOGGLE:'off'}}}});
      const service=interpret(machine).onTransition(s=>console.log('FSM‑3',s.value)).start();
      service.send('TOGGLE'); service.send('TOGGLE');
    } catch{}
  })();
  
  /* FSM‑Example‑4:  Saga pattern for long‑running txn */
  (function () {
    const saga=async()=>{try{await step1();await step2();}catch{await compensate();}}
    const step1=()=>Promise.resolve(); const step2=()=>Promise.reject();
    const compensate=()=>console.log('FSM‑4 rollback');
    saga();
  })();
  
  /* FSM‑Example‑5:  Event‑driven microservice command bus */
  (function () {
    const bus={}; bus.exec=(cmd)=>bus[cmd.type]?.(cmd);
    bus['EmailSend']=c=>console.log('FSM‑5 email to',c.to);
    bus.exec({type:'EmailSend',to:'x@y'});
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION MFE — Micro Frontends & Module Federation               */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* MFE‑Example‑1:  Webpack Module Federation runtime import */
  (async () => {
    /* eslint-disable no-undef */
    // __webpack_init_sharing__('default'); const container=window.remoteApp; await container.init(__webpack_share_scopes__.default);
    // const mod=await container.get('./Widget'); const Widget=mod().default;
    console.log('MFE‑1 federation stub executed');
  })();
  
  /* MFE‑Example‑2:  Custom element as contract */
  class UserCard extends HTMLElement {
    connectedCallback(){ this.textContent=`User: ${this.getAttribute('name')}`; }
  }
  customElements.define('user-card',UserCard);
  console.log('MFE‑2 custom element registered');
  
  /* MFE‑Example‑3:  iframe integration handshake */
  (function () {
    const child= new MessageChannel();
    child.port1.onmessage=e=>console.log('MFE‑3 host got',e.data);
    child.port2.postMessage('hello from micro‑frontend');
  })();
  
  /* MFE‑Example‑4:  Shared global state via BroadcastChannel */
  (function () {
    const bc=new BroadcastChannel('global');
    bc.onmessage=e=>console.log('MFE‑4 bc',e.data);
    bc.postMessage({type:'INC'});
  })();
  
  /* MFE‑Example‑5:  Import maps for version isolation */
  (function () {
    const script=document.createElement('script'); script.type='importmap';
    script.textContent=JSON.stringify({imports:{react:'https://cdn.skypack.dev/react@18'}});
    document.currentScript?.after(script);
    console.log('MFE‑5 importmap injected');
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION SRV — Serverless Functions & BFF Patterns               */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* SRV‑Example‑1:  AWS Lambda handler */
  const lambda = async (evt) => ({ statusCode:200, body:JSON.stringify({ msg:'SRV‑1 hi' }) });
  lambda({}).then(r=>console.log(r.body));
  
  /* SRV‑Example‑2:  Azure Functions context demo */
  const azureFunc = (context, req) => { context.res={body:'SRV‑2 ok'}; context.done(); };
  azureFunc({ done:()=>console.log('SRV‑2 done') }, {});
  
  /* SRV‑Example‑3:  Google Cloud Function HTTP */
  const gcf = (req,res)=>{ res.json({g:'SRV‑3'}); };
  console.log('SRV‑3 gcf defined params:', gcf.length);
  
  /* SRV‑Example‑4:  BFF route aggregating microservices */
  (function () {
    const fetchUser=()=>Promise.resolve({name:'Ada'}); const fetchOrders=()=>Promise.resolve([1,2]);
    const bff=async()=>({...(await fetchUser()),orders:await fetchOrders()});
    bff().then(r=>console.log('SRV‑4 BFF result',r));
  })();
  
  /* SRV‑Example‑5:  Edge function (Cloudflare) */
  addEventListener?.('fetch',e=>e.respondWith(new Response('SRV‑5 edge',{headers:{'Content-Type':'text/plain'}})));