/****************************************************************************************
 * Chapter 13 | Real Time & Collaborative Apps
 * Single-file .js: 5 sections × ≥5 examples each
 ****************************************************************************************/

/* SECTION WS — WebSockets & Socket.io */

// WS‑Example‑1: Simple WebSocket echo server (Node.js)
(function WS1_Server() {
    const WebSocket = require('ws');
    const wss = new WebSocket.Server({ port: 8080 });
    wss.on('connection', ws => {
      ws.on('message', msg => ws.send(`Echo: ${msg}`));
      ws.send('Welcome');
    });
  })();
  
  // WS‑Example‑2: WebSocket client (browser)
  (function WS2_Client() {
    try {
      const ws = new WebSocket('ws://localhost:8080');
      ws.onopen = () => ws.send('Hello');
      ws.onmessage = e => console.log('WS msg:', e.data);
      ws.onerror = e => console.error('WS error:', e);
    } catch(e) { console.error('WS2 init error:', e); }
  })();
  
  // WS‑Example‑3: Socket.io basic server
  (function SIO3_Server() {
    const io = require('socket.io')(3000, { cors: { origin: '*' } });
    io.on('connection', sock => {
      sock.emit('greet', 'Hello Client');
      sock.on('message', msg => sock.emit('reply', `Server got ${msg}`));
    });
  })();
  
  // WS‑Example‑4: Socket.io rooms & broadcasting
  (function SIO4_Rooms() {
    const io = require('socket.io')(3001);
    io.on('connection', sock => {
      sock.join('room1');
      io.to('room1').emit('joined', sock.id);
      sock.on('message', msg => io.to('room1').emit('broadcast', { id: sock.id, msg }));
    });
  })();
  
  // WS‑Example‑5: Reconnect logic with exponential backoff (client)
  (function WS5_Reconnect() {
    let url = 'ws://localhost:8080', ws, retries = 0;
    function connect() {
      ws = new WebSocket(url);
      ws.onopen = () => { retries = 0; console.log('WS5 open'); };
      ws.onmessage = e => console.log('WS5 msg:', e.data);
      ws.onclose = () => {
        const delay = Math.min(1000 * 2 ** retries, 30000);
        setTimeout(connect, delay);
        retries++;
      };
      ws.onerror = err => console.error('WS5 err:', err);
    }
    connect();
  })();
  
  /* SECTION SSE — Server Sent Events */
  
  // SSE‑Example‑1: SSE endpoint (Express)
  (function SSE1_Server() {
    const express = require('express');
    const app = express();
    app.get('/sse', (req, res) => {
      res.set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
      res.write('retry: 10000\n');
      let id = 0;
      const iv = setInterval(() => {
        res.write(`id: ${id++}\ndata: ${JSON.stringify({ time: Date.now() })}\n\n`);
      }, 1000);
      req.on('close', () => clearInterval(iv));
    });
    app.listen(4000);
  })();
  
  // SSE‑Example‑2: SSE client (browser)
  (function SSE2_Client() {
    try {
      const es = new EventSource('http://localhost:4000/sse');
      es.onopen = () => console.log('SSE open');
      es.onmessage = e => console.log('SSE msg:', e.data);
      es.onerror = e => console.error('SSE error', e);
    } catch(e) { console.error('SSE2 init error:', e); }
  })();
  
  // SSE‑Example‑3: Keep‑alive comment ping
  (function SSE3_KeepAlive() {
    const express = require('express');
    const app = express();
    app.get('/keep', (req, res) => {
      res.set({ 'Content-Type': 'text/event-stream' });
      res.write(': ping\n\n');
      const iv = setInterval(() => res.write(': ping\n\n'), 15000);
      req.on('close', () => clearInterval(iv));
    });
    app.listen(4001);
  })();
  
  // SSE‑Example‑4: Reconnection with Last‑Event‑ID
  (function SSE4_Client() {
    const es = new EventSource('/sse-restore');
    es.onmessage = e => console.log('SSE4 data', e.data);
    es.onerror = () => console.error('SSE4 lost');
  })();
  
  // SSE‑Example‑5: Named event and JSON parsing
  (function SSE5_CustomEvent() {
    const es = new EventSource('/sse-events');
    es.addEventListener('update', e => {
      try {
        const obj = JSON.parse(e.data);
        console.log('SSE5 update', obj);
      } catch(err) {
        console.error('SSE5 parse error', err);
      }
    });
  })();
  
  /* SECTION RTC — WebRTC Peer to Peer */
  
  // RTC‑Example‑1: Create RTCPeerConnection
  (function RTC1_Create() {
    const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
    console.log('RTC1 state:', pc.connectionState);
  })();
  
  // RTC‑Example‑2: Offer/Answer exchange (pseudo with signaling)
  (function RTC2_OfferAnswer() {
    const pc = new RTCPeerConnection();
    pc.onicecandidate = e => e.candidate && sendSignal({ ice: e.candidate });
    pc.createOffer().then(o => pc.setLocalDescription(o))
      .then(() => sendSignal({ sdp: pc.localDescription }))
      .catch(err => console.error('RTC2 offer error', err));
    receiveSignal(async msg => {
      try {
        if (msg.sdp) {
          await pc.setRemoteDescription(msg.sdp);
          if (msg.sdp.type === 'offer') {
            const ans = await pc.createAnswer();
            await pc.setLocalDescription(ans);
            sendSignal({ sdp: pc.localDescription });
          }
        }
        if (msg.ice) await pc.addIceCandidate(msg.ice);
      } catch(e) { console.error('RTC2 answer error', e); }
    });
    function sendSignal(m) { /* send via WebSocket or HTTP */ }
    function receiveSignal(cb) { /* hook from your signaling channel */ }
  })();
  
  // RTC‑Example‑3: DataChannel usage
  (function RTC3_DataChannel() {
    const pc = new RTCPeerConnection();
    const dc = pc.createDataChannel('chat');
    dc.onopen = () => dc.send('hello peer');
    dc.onmessage = e => console.log('RTC3 msg:', e.data);
    pc.ondatachannel = ev => {
      const receive = ev.channel;
      receive.onmessage = e => console.log('RTC3 recv:', e.data);
    };
  })();
  
  // RTC‑Example‑4: Media stream (getUserMedia)
  (function RTC4_Media() {
    navigator.mediaDevices.getUserMedia({ audio: true, video: true })
      .then(stream => {
        const video = document.createElement('video');
        video.srcObject = stream; video.autoplay = true; video.muted = true;
        document.body.appendChild(video);
      })
      .catch(e => console.error('RTC4 media error:', e));
  })();
  
  // RTC‑Example‑5: ICE connection state changes
  (function RTC5_ICEState() {
    const pc = new RTCPeerConnection();
    pc.oniceconnectionstatechange = () => {
      console.log('RTC5 ICE state:', pc.iceConnectionState);
      if (pc.iceConnectionState === 'failed') pc.restartIce();
    };
  })();
  
  /* SECTION CRDT — CRDTs & Operational Transforms */
  
  // CRDT‑Example‑1: LWW-Register (Last‑Write‑Wins)
  (function CRDT1_LWW() {
    class LWW {
      constructor() { this.ts = 0; this.value = null; }
      set(val, ts = Date.now()) {
        if (ts >= this.ts) { this.value = val; this.ts = ts; }
      }
      get() { return this.value; }
    }
    const r = new LWW();
    r.set('A', 100); r.set('B', 50);
    console.log('CRDT1', r.get()); // 'A'
  })();
  
  // CRDT‑Example‑2: G‑Counter (grow‑only counter)
  (function CRDT2_GCounter() {
    class GCounter {
      constructor(id) { this.id = id; this.counts = {}; }
      inc() { this.counts[this.id] = (this.counts[this.id] || 0) + 1; }
      value() { return Object.values(this.counts).reduce((a, b) => a + b, 0); }
      merge(other) {
        Object.entries(other.counts).forEach(([k, v]) => {
          this.counts[k] = Math.max(this.counts[k] || 0, v);
        });
      }
    }
    const a = new GCounter('A'), b = new GCounter('B');
    a.inc(); b.inc(); b.inc();
    a.merge(b);
    console.log('CRDT2', a.value()); // 3
  })();
  
  // CRDT‑Example‑3: OR‑Set (Observed‑Remove Set)
  (function CRDT3_ORS() {
    class ORSet {
      constructor() { this.adds = new Map(); this.removes = new Map(); }
      add(val, ts) { this.adds.set(val + '@' + ts, { val, ts }); }
      remove(val, ts) { this.removes.set(val + '@' + ts, { val, ts }); }
      value() {
        const res = new Set();
        for (const { val, ts } of this.adds.values()) {
          const removed = [...this.removes.values()].some(r => r.val === val && r.ts >= ts);
          if (!removed) res.add(val);
        }
        return [...res];
      }
    }
    const s = new ORSet();
    s.add('x', 1); s.remove('x', 2);
    console.log('CRDT3', s.value()); // []
  })();
  
  // CRDT‑Example‑4: Automerge basic doc
  (async function CRDT4_AutoMerge() {
    const Automerge = require('automerge');
    let doc1 = Automerge.from({ items: [] });
    let doc2 = Automerge.clone(doc1);
    doc1 = Automerge.change(doc1, d => d.items.push('a'));
    const merged = Automerge.merge(doc1, doc2);
    console.log('CRDT4', merged.items); // ['a']
  })();
  
  // CRDT‑Example‑5: OT transform simple text
  (function CRDT5_OT() {
    function transform(op1, op2) {
      // op: { pos, str, type:'insert'|'delete' }
      if (op1.type === 'insert' && op2.type === 'insert' && op1.pos <= op2.pos) {
        op2.pos += op1.str.length;
      }
      return op2;
    }
    const op1 = { pos: 0, str: 'A', type: 'insert' };
    const op2 = { pos: 0, str: 'B', type: 'insert' };
    console.log('CRDT5', transform(op1, op2).pos); // 1
  })();
  
  /* SECTION OFF — Offline First & Synchronization */
  
  // OFF‑Example‑1: Queue requests in localStorage
  (function OFF1_Queue() {
    function enqueue(req) {
      const q = JSON.parse(localStorage.getItem('queue') || '[]');
      q.push(req);
      localStorage.setItem('queue', JSON.stringify(q));
    }
    function sync() {
      const q = JSON.parse(localStorage.getItem('queue') || '[]');
      while (q.length) {
        const r = q.shift();
        fetch(r.url, r).catch(() => q.unshift(r));
      }
      localStorage.setItem('queue', JSON.stringify(q));
    }
    window.addEventListener('online', sync);
  })();
  
  // OFF‑Example‑2: IndexedDB cache fallback
  (function OFF2_IDB() {
    const dbReq = indexedDB.open('cacheDB', 1);
    dbReq.onupgradeneeded = e => e.target.result.createObjectStore('responses', { keyPath: 'url' });
    function cacheFetch(url) {
      return fetch(url)
        .then(res => res.clone().json())
        .then(data => {
          dbReq.result.transaction('responses', 'readwrite')
            .objectStore('responses')
            .put({ url, data });
          return data;
        })
        .catch(() => dbReq.result.transaction('responses')
          .objectStore('responses')
          .get(url)
          .then(r => r && r.data));
    }
    cacheFetch('/api/data').then(console.log);
  })();
  
  // OFF‑Example‑3: Service Worker intercept & cache
  (function OFF3_SW() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js');
    }
    // sw.js:
    /*
    self.addEventListener('fetch', e => {
      e.respondWith(
        caches.open('v1').then(cache =>
          cache.match(e.request).then(r =>
            r || fetch(e.request).then(resp => {
              cache.put(e.request, resp.clone());
              return resp;
            })
          )
        )
      );
    });
    */
  })();
  
  // OFF‑Example‑4: Background sync (Service Worker)
  (function OFF4_BGSync() {
    // In Service Worker:
    /*
    self.addEventListener('sync', e => {
      if (e.tag === 'syncQueue') e.waitUntil(syncQueue());
    });
    */
    // Registration:
    navigator.serviceWorker.ready.then(sw => sw.sync.register('syncQueue'));
  })();
  
  // OFF‑Example‑5: Conflict resolution merge strategy
  (function OFF5_Merge() {
    function mergeClientServer(local, remote) {
      const map = new Map();
      [...local, ...remote]
        .sort((a, b) => a.ts - b.ts)
        .forEach(op => map.set(op.id, op));
      return Array.from(map.values());
    }
    const local = [{ id: 1, ts: 1 }], remote = [{ id: 1, ts: 2 }];
    console.log('OFF5 merge', mergeClientServer(local, remote));
  })();