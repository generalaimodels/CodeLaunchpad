/**************************************************************************************************
 * Chapter 13 | Real‑Time & Collaborative Apps
 * -----------------------------------------------------------------------------------------------
 * Single‑file playground. 5 sections × 5 concise, illustrative examples each.
 * Works in browser; Node parts guarded (require / dynamic import in try‑catch).
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION WS — WebSockets & Socket.io                             */
/*───────────────────────────────────────────────────────────────────*/

/* WS‑Example‑1:  Native browser WebSocket echo */
(function () {
    if (typeof WebSocket === 'function') {
      const ws = new WebSocket('wss://echo.websocket.events');
      ws.onopen  = () => ws.send('WS‑1 hello');
      ws.onmessage = e => console.log('WS‑1 echo:', e.data);
    }
  })();
  
  /* WS‑Example‑2:  Node ‘ws’ echo server */
  (function () {
    try {
      const { WebSocketServer } = require('ws');
      const wss = new WebSocketServer({ port: 8080 });
      wss.on('connection', c => c.on('message', m => c.send(m)));
      console.log('WS‑2 ws server on :8080');
    } catch {}
  })();
  
  /* WS‑Example‑3:  Socket.io server (Node) */
  (function () {
    try {
      const http = require('http').createServer();
      const io   = require('socket.io')(http);
      io.on('connection', s => s.emit('news', 'WS‑3 hi client'));
      http.listen(3000, () => console.log('WS‑3 socket.io :3000'));
    } catch {}
  })();
  
  /* WS‑Example‑4:  Socket.io browser client */
  (async () => {
    if (typeof io === 'undefined') {
      await import('https://cdn.socket.io/4.7.2/socket.io.min.js');
    }
    if (typeof io === 'function') {
      const s = io('http://localhost:3000');
      s.on('news', d => console.log('WS‑4 received', d));
      s.emit('ping', Date.now());
    }
  })();
  
  /* WS‑Example‑5:  Reconnection with exponential back‑off */
  (function () {
    let retry = 0;
    const connect = () => {
      const ws = new WebSocket('wss://echo.websocket.events');
      ws.onerror = () => {};
      ws.onclose = () => {
        const delay = Math.min(2 ** retry * 100, 10_000);
        console.log('WS‑5 reconnect in', delay);
        setTimeout(connect, delay); retry++;
      };
      ws.onopen = () => (retry = 0, ws.send('re‑connected'));
    };
    if (typeof WebSocket === 'function') connect();
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION SSE — Server‑Sent Events                                */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* SSE‑Example‑1:  EventSource client */
  (function () {
    if (typeof EventSource === 'function') {
      const es = new EventSource('/stream');
      es.onmessage = e => console.log('SSE‑1 data', e.data);
    }
  })();
  
  /* SSE‑Example‑2:  Express SSE endpoint (Node) */
  (function () {
    try {
      const express = require('express'); const app = express();
      app.get('/stream', (req, res) => {
        res.set({ 'Content-Type':'text/event-stream', 'Cache-Control':'no-cache', Connection:'keep-alive' });
        let i = 0; const id = setInterval(()=>res.write(`data: ping ${++i}\n\n`), 1000);
        req.on('close', ()=>clearInterval(id));
      });
      app.listen(4000, () => console.log('SSE‑2 /stream on 4000'));
    } catch {}
  })();
  
  /* SSE‑Example‑3:  Auto‑retry after disconnect */
  (function () {
    if (typeof EventSource === 'function') {
      const es = new EventSource('/stream', { withCredentials:false });
      es.onerror = () => console.log('SSE‑3 reconnecting …');
    }
  })();
  
  /* SSE‑Example‑4:  Custom event types */
  (function () {
    if (typeof EventSource === 'function') {
      const es = new EventSource('/stream');
      es.addEventListener('alert', e => console.log('SSE‑4 alert:', e.data));
    }
  })();
  
  /* SSE‑Example‑5:  Closing EventSource gracefully */
  (function () {
    if (typeof EventSource === 'function') {
      const es = new EventSource('/stream');
      setTimeout(()=>{ es.close(); console.log('SSE‑5 closed'); }, 5_000);
    }
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION RTC — WebRTC Peer‑to‑Peer                               */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* RTC‑Example‑1:  Create offer */
  (async () => {
    if (typeof RTCPeerConnection === 'function') {
      const pc = new RTCPeerConnection();
      const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
      console.log('RTC‑1 SDP length', offer.sdp.length);
    }
  })();
  
  /* RTC‑Example‑2:  DataChannel send/receive */
  (function () {
    if (typeof RTCPeerConnection === 'function') {
      const a = new RTCPeerConnection(), b = new RTCPeerConnection();
      const ch = a.createDataChannel('chat');
      b.ondatachannel = e => e.channel.onmessage = m => console.log('RTC‑2 got', m.data);
      a.onicecandidate = e => e.candidate && b.addIceCandidate(e.candidate);
      b.onicecandidate = e => e.candidate && a.addIceCandidate(e.candidate);
      a.createOffer().then(o=>a.setLocalDescription(o).then(()=>b.setRemoteDescription(o))
        .then(()=>b.createAnswer()).then(a2=>b.setLocalDescription(a2).then(()=>a.setRemoteDescription(a2)));
      setTimeout(()=>ch.send('hi'),500);
    }
  })();
  
  /* RTC‑Example‑3:  ICE candidate logging */
  (function () {
    if (typeof RTCPeerConnection === 'function') {
      const pc = new RTCPeerConnection();
      pc.onicecandidate = e => console.log('RTC‑3 ICE', e.candidate?.candidate);
      pc.createDataChannel('x'); pc.createOffer().then(o=>pc.setLocalDescription(o));
    }
  })();
  
  /* RTC‑Example‑4:  STUN/TURN configuration */
  (function () {
    if (typeof RTCPeerConnection === 'function') {
      const cfg = { iceServers:[{ urls:'stun:stun.l.google.com:19302' }] };
      try { new RTCPeerConnection(cfg); console.log('RTC‑4 STUN ok'); } catch(e){ console.error(e); }
    }
  })();
  
  /* RTC‑Example‑5:  Screen share getDisplayMedia */
  (async () => {
    if (navigator.mediaDevices?.getDisplayMedia) {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({ video:true });
        console.log('RTC‑5 screenshare tracks', stream.getTracks().length);
        stream.getTracks().forEach(t=>t.stop());
      } catch(e){ console.log('RTC‑5 denied'); }
    }
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION CRDT — CRDTs & Operational Transforms                   */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* CRDT‑Example‑1:  Grow‑only Counter (G‑Counter) */
  (function () {
    class GCounter {
      constructor(id){ this.id=id; this.c={}; }
      inc(n=1){ this.c[this.id]=(this.c[this.id]||0)+n; }
      merge(o){ Object.entries(o.c).forEach(([k,v])=>this.c[k]=Math.max(this.c[k]||0,v)); }
      value(){ return Object.values(this.c).reduce((a,b)=>a+b,0); }
    }
    const a=new GCounter('a'), b=new GCounter('b');
    a.inc(); b.inc(2); a.merge(b); b.merge(a);
    console.log('CRDT‑1 values',a.value(),b.value());
  })();
  
  /* CRDT‑Example‑2:  RGA‑like text insert */
  (function () {
    const doc=[]; const insert=(idx,char,id)=>doc.splice(idx,0,{char,id});
    insert(0,'H',1); insert(1,'i',2); console.log('CRDT‑2 text',doc.map(x=>x.char).join(''));
  })();
  
  /* CRDT‑Example‑3:  Automerge shared document */
  (async () => {
    try {
      const Automerge = await import('https://esm.run/automerge@2?bundle');
      let doc1=Automerge.init(); doc1=Automerge.change(doc1,d=>d.items=['a']);
      const binary = Automerge.save(doc1);
      let doc2=Automerge.load(binary);
      doc2=Automerge.change(doc2,d=>d.items.push('b'));
      const merged=Automerge.merge(doc1,doc2);
      console.log('CRDT‑3 merged items',merged.items.join(','));
    } catch {}
  })();
  
  /* CRDT‑Example‑4:  OT transform for concurrent insert/delete */
  (function () {
    const transform = (opA, opB) => {
      if (opA.type==='insert' && opB.type==='delete' && opA.pos>=opB.pos) opA.pos--;
      return opA;
    };
    const A={type:'insert',pos:3,char:'x'}, B={type:'delete',pos:2};
    console.log('CRDT‑4 transformed',transform({...A},B));
  })();
  
  /* CRDT‑Example‑5:  Yjs + websocket provider */
  (async () => {
    try {
      const Y = await import('https://esm.run/yjs?bundle');
      const { WebsocketProvider } = await import('https://esm.run/y-websocket?bundle');
      const ydoc=new Y.Doc(); const provider=new WebsocketProvider('wss://demos.yjs.dev','room',ydoc);
      const ytext=ydoc.getText('msg'); ytext.observe(e=>console.log('CRDT‑5 update',ytext.toString()));
      ytext.insert(0,'hello');
    } catch {}
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION OFF — Offline‑First & Synchronization Strategies        */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* OFF‑Example‑1:  LocalStorage cache fall‑back */
  (function () {
    const key='data'; const fetchData=()=>Promise.resolve('remote');
    (async ()=>{ const cached=localStorage.getItem(key);
      const data=cached||await fetchData(); console.log('OFF‑1 data',data); localStorage.setItem(key,data); })();
  })();
  
  /* OFF‑Example‑2:  Service Worker install caching */
  (async () => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').then(() => console.log('OFF‑2 SW registered'));
    }
  })();
  
  /* OFF‑Example‑3:  IndexedDB queue for offline requests */
  (function () {
    const enqueue=async(req)=>{ const db=await idb.openDB('queue',1,{upgrade(db){db.createObjectStore('outbox',{autoIncrement:true});}});
      db.add('outbox',req); };
    // idb omitted for brevity; use https://github.com/jakearchibald/idb
    console.log('OFF‑3 enqueue stub');
  })();
  
  /* OFF‑Example‑4:  Background Sync API */
  (async () => {
    if ('serviceWorker' in navigator && 'SyncManager' in window) {
      const reg = await navigator.serviceWorker.ready;
      await reg.sync.register('sync-outbox');
      console.log('OFF‑4 sync registered');
    }
  })();
  
  /* OFF‑Example‑5:  Last‑write‑wins conflict resolution */
  (function () {
    const local={val:1,ts:Date.now()}, remote={val:2,ts:Date.now()+100};
    const merged = remote.ts > local.ts ? remote : local;
    console.log('OFF‑5 resolved value',merged.val);
  })();