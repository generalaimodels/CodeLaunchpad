/****************************************************************************************
 * Chapter 12 | Scalable Architectures & Design Patterns
 * Single-file demo: 5 topics × ≥5 examples each.
 ****************************************************************************************/

/* SECTION MVR — MVC, MVVM, Flux & Redux */

/* MVR‑1: Simple MVC */
(function MVC1() {
    class Model { constructor(){ this.data = ''; } setData(v){ this.data = v; } getData(){ return this.data; } }
    class View {
      constructor() {
        this.onInput = null;
        this.input = document.createElement('input');
        this.display = document.createElement('div');
        document.body.append(this.input, this.display);
        this.input.addEventListener('input', e => this.onInput && this.onInput(e.target.value));
      }
      update(v){ this.display.textContent = v; }
    }
    class Controller {
      constructor(m, v){
        v.onInput = val => { m.setData(val); v.update(m.getData()); };
      }
    }
    new Controller(new Model(), new View());
  })();
  
  /* MVR‑2: MVVM w/ two‑way binding */
  (function MVVM2() {
    class ViewModel {
      constructor() {
        this._text = '';
        this.onChange = null;
        Object.defineProperty(this, 'text', {
          get: () => this._text,
          set: v => { this._text = v; this.onChange && this.onChange(v); }
        });
      }
    }
    const vm = new ViewModel();
    const input = document.createElement('input'), span = document.createElement('span');
    document.body.append(input, span);
    vm.onChange = v => span.textContent = v;
    input.addEventListener('input', e => vm.text = e.target.value);
  })();
  
  /* MVR‑3: Flux basic dispatcher/store */
  (function Flux3() {
    class Dispatcher {
      constructor(){ this.callbacks = []; }
      register(cb){ return this.callbacks.push(cb) - 1; }
      dispatch(action){ this.callbacks.forEach(cb => cb(action)); }
    }
    const dispatcher = new Dispatcher();
    const Store = (() => {
      let state = 0;
      const listeners = [];
      dispatcher.register(({ type, payload }) => {
        if(type==='inc') state += payload;
        listeners.forEach(l=>l(state));
      });
      return { subscribe: l=>listeners.push(l) };
    })();
    Store.subscribe(s=>console.log('Flux3 state:', s));
    dispatcher.dispatch({ type:'inc', payload:5 });
  })();
  
  /* MVR‑4: Redux createStore */
  (function Redux4() {
    function createStore(reducer){
      let state, listeners = [];
      const getState = () => state;
      const dispatch = action => {
        state = reducer(state, action);
        listeners.forEach(l=>l());
        return action;
      };
      const subscribe = l=>listeners.push(l);
      dispatch({}); // init
      return { getState, dispatch, subscribe };
    }
    const reducer = (s=0, a)=> a.type==='add'?s+a.payload:s;
    const store = createStore(reducer);
    store.subscribe(()=>console.log('Redux4 state:', store.getState()));
    store.dispatch({ type:'add', payload:3 });
  })();
  
  /* MVR‑5: Redux w/ middleware (logger) */
  (function Redux5() {
    const applyMiddleware = (store, ...mws) => {
      let dispatch = store.dispatch;
      mws.slice().reverse().forEach(mw => { dispatch = mw(store)(dispatch); });
      return Object.assign({}, store, { dispatch });
    };
    const logger = store => next => action => {
      console.log('Redux5 prev:', store.getState());
      const result = next(action);
      console.log('Redux5 next:', store.getState());
      return result;
    };
    const reducer = (s=0,a)=>a.type==='add'?s+a.payload:s;
    const store = applyMiddleware(createStore(reducer), logger);
    store.dispatch({ type:'add', payload:7 });
    function createStore(r){let s,l=[];return{getState:()=>s,dispatch:a=>(s=r(s,a),l.forEach(fn=>fn()),a),subscribe:fn=>l.push(fn)}}  
  })();
  
  /* SECTION OBS — Observer, Pub/Sub & Event Sourcing */
  
  /* OBS‑1: Observer pattern */
  (function OBS1() {
    class Subject {
      constructor(){ this.observers = []; }
      attach(o){ this.observers.push(o); }
      detach(o){ this.observers = this.observers.filter(x=>x!==o); }
      notify(data){ this.observers.forEach(o=>o.update(data)); }
    }
    class Observer { constructor(id){ this.id=id; } update(d){ console.log(`OBS1 ${this.id}:`, d); } }
    const subj = new Subject();
    const o1 = new Observer(1), o2 = new Observer(2);
    subj.attach(o1); subj.attach(o2);
    subj.notify('hello'); subj.detach(o2); subj.notify('world');
  })();
  
  /* OBS‑2: Pub/Sub bus */
  (function OBS2() {
    const bus = {
      topics: {},
      subscribe(t, cb){ (this.topics[t] = this.topics[t]||[]).push(cb); },
      publish(t, data){ (this.topics[t]||[]).forEach(cb=>cb(data)); }
    };
    bus.subscribe('evt', d=>console.log('OBS2 got', d));
    bus.publish('evt', { a:1 });
  })();
  
  /* OBS‑3: Node.js EventEmitter */
  (function OBS3() {
    const { EventEmitter } = require('events');
    const ee = new EventEmitter();
    ee.on('ping', () => console.log('OBS3 pong'));
    ee.emit('ping');
  })();
  
  /* OBS‑4: Event Sourcing example */
  (function OBS4() {
    const events = [];
    const emit = e => events.push(e);
    const apply = state => events.reduce((s,e)=>{
      if(e.type==='deposit') s.balance += e.amount;
      if(e.type==='withdraw') s.balance -= e.amount;
      return s;
    }, state);
    emit({ type:'deposit', amount:100 });
    emit({ type:'withdraw', amount:30 });
    console.log('OBS4 balance:', apply({ balance:0 }).balance);
  })();
  
  /* OBS‑5: Minimal Observable */
  (function OBS5() {
    const Observable = producer => ({
      subscribe: observer => producer(observer),
      map: fn => Observable(o => this.subscribe({ next: v=>o.next(fn(v)) }))
    });
    const obs = Observable(o => { o.next(1); o.next(2); o.next(3); });
    obs.subscribe({ next: v=>console.log('OBS5 val', v) });
  })();
  
  /* SECTION CQS — CQRS, State Machines & Finite Automata */
  
  /* CQS‑1: CQRS separation */
  (function CQS1() {
    const state = { count:0 };
    const commandHandlers = {
      increment: payload => { state.count += payload; }
    };
    const queryHandlers = {
      getCount: () => state.count
    };
    commandHandlers.increment(4);
    console.log('CQS1 count:', queryHandlers.getCount());
  })();
  
  /* CQS‑2: Traffic light FSM */
  (function CQS2() {
    const states = { RED:'GREEN', GREEN:'YELLOW', YELLOW:'RED' };
    let current = 'RED';
    function next() {
      current = states[current];
      console.log('CQS2:', current);
    }
    next(); next(); next();
  })();
  
  /* CQS‑3: FSM with guards */
  (function CQS3() {
    const transitions = {
      OFF: { turnOn: 'ON' },
      ON: { turnOff: 'OFF' }
    };
    let state = 'OFF';
    function dispatch(event) {
      const next = transitions[state][event];
      if(!next) throw Error(`Invalid event ${event} in state ${state}`);
      state = next;
      console.log('CQS3:', state);
    }
    dispatch('turnOn'); dispatch('turnOff');
  })();
  
  /* CQS‑4: Door automaton */
  (function CQS4() {
    const fsm = {
      closed: { open:'open', lock:'locked' },
      open:   { close:'closed' },
      locked: { unlock:'closed' }
    };
    let s='closed';
    ['open','close','lock','unlock'].forEach(e=>{
      if(fsm[s][e]) s=fsm[s][e];
      console.log('CQS4', e, '=>', s);
    });
  })();
  
  /* CQS‑5: State machine runner */
  (function CQS5() {
    function createMachine(config){
      let state = config.initial;
      return { send: evt => {
        const next = config.states[state].on[evt];
        if(!next) throw Error(`No transition for ${evt} in ${state}`);
        state = next; console.log('CQS5 state:', state);
      }};
    }
    const m = createMachine({
      initial:'idle',
      states:{
        idle:{ on:{ start:'running' } },
        running:{ on:{ stop:'idle' } }
      }
    });
    m.send('start'); m.send('stop');
  })();
  
  /* SECTION MFED — Micro Frontends & Module Federation */
  
  /* MFED‑1: Dynamic Web Component loader */
  (function MFED1() {
    async function loadComponent(url, tag) {
      await import(url);
      console.log(`MFED1 loaded ${tag}`);
    }
    // usage: loadComponent('/remote-component.js','remote-comp');
  })();
  
  /* MFED‑2: Webpack Module Federation host config */
  const mfHostConfig = {
    name: 'hostApp',
    remotes: { remoteApp: 'remoteApp@http://localhost:3001/remoteEntry.js' },
    shared: ['react','react-dom']
  };
  console.log('MFED2 host config:', mfHostConfig);
  
  /* MFED‑3: Module Federation remote config */
  const mfRemoteConfig = {
    name: 'remoteApp',
    filename: 'remoteEntry.js',
    exposes: { './Button': './src/Button' },
    shared: []
  };
  console.log('MFED3 remote config:', mfRemoteConfig);
  
  /* MFED‑4: Dynamic remote loading */
  (function MFED4() {
    const script = document.createElement('script');
    script.src = 'http://localhost:3001/remoteEntry.js';
    script.onload = () => {
      __webpack_init_sharing__('default');
      __webpack_share_scopes__.default = {};
      __webpack_init_sharing__('default').then(()=>{
        __webpack_get_script__('remoteApp/Button')
          .then(factory=>{
            const Button = factory();
            document.body.append(new Button());
          });
      });
    };
    document.head.append(script);
  })();
  
  /* MFED‑5: SystemJS microfrontend */
  (function MFED5() {
    System.import('https://cdn.example.com/mf-app.js')
      .then(m=> console.log('MFED5 loaded', m))
      .catch(e=> console.error(e));
  })();
  
  /* SECTION SLS — Serverless Functions & BFF Patterns */
  
  /* SLS‑1: AWS Lambda handler */
  exports.lambdaHandler = async (event, context) => {
    try {
      const name = event.queryStringParameters?.name || 'world';
      return { statusCode:200, body:JSON.stringify({ msg:`Hello, ${name}` }) };
    } catch (e) {
      return { statusCode:500, body:'Internal Error' };
    }
  };
  
  /* SLS‑2: Azure Function */
  module.exports.azureFunction = async function (context, req) {
    try {
      context.res = { body: `Hello ${req.query.name||'Azure'}` };
    } catch (e) {
      context.log.error(e);
      context.res = { status:500, body:'Error' };
    }
  };
  
  /* SLS‑3: Google Cloud Function */
  exports.gcpFunction = (req, res) => {
    try {
      res.status(200).send(`Hi ${req.query.name||'GCP'}`);
    } catch (e) {
      res.status(500).send('Error');
    }
  };
  
  /* SLS‑4: BFF Gateway with Express */
  (function BFF4() {
    const express = require('express');
    const axios = require('axios');
    const app = express();
    app.get('/api/aggregate', async (req, res) => {
      try {
        const [u, o] = await Promise.all([
          axios.get('http://users.service/users'),
          axios.get('http://orders.service/orders')
        ]);
        res.json({ users: u.data, orders: o.data });
      } catch (e) {
        res.status(502).send('Bad Gateway');
      }
    });
    // app.listen(4000);
  })();
  
  /* SLS‑5: Serverless Offline (Express wrap) */
  (function SLS5() {
    const serverless = require('serverless-http');
    const express = require('express');
    const app = express();
    app.get('/ping', (req, res) => res.send('pong'));
    exports.handler = serverless(app);
  })();