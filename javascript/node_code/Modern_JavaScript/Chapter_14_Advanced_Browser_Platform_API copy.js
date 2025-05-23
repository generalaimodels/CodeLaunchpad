/**************************************************************************************************
 * Chapter 14 | Advanced Browser & Platform APIs
 * -----------------------------------------------------------------------------------------------
 * Single‑file playground. 5 sections × 5 concise, runnable examples each (feature‑gated).
 **************************************************************************************************/

/*───────────────────────────────────────────────────────────────────*/
/* SECTION SW — Service Workers & Cache API                        */
/*───────────────────────────────────────────────────────────────────*/

/* SW‑Example‑1:  Register service worker */
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').then(r => console.log('SW‑1 scope', r.scope));
  }
  
  /* SW‑Example‑2:  Basic install event in /sw.js
  self.addEventListener('install', e => {
    e.waitUntil(caches.open('v1').then(c => c.addAll(['/','/index.css'])));
  });
  */
  
  /* SW‑Example‑3:  Fetch handler cache‑first
  self.addEventListener('fetch', e => {
    e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
  });
  */
  
  /* SW‑Example‑4:  Cache invalidation */
  async function clearOldCaches(current = 'v2') {
    const names = await caches.keys();
    await Promise.all(names.filter(n => n !== current).map(n => caches.delete(n)));
    console.log('SW‑4 deleted', names.length - 1);
  }
  
  /* SW‑Example‑5:  Background Sync trigger */
  if ('serviceWorker' in navigator && 'SyncManager' in window) {
    navigator.serviceWorker.ready.then(r => r.sync.register('sync‑tag'));
  }
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION IDB — IndexedDB & LocalForage                           */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* IDB‑Example‑1:  Open DB + object store */
  const idbOpen = indexedDB.open('appDB', 1);
  idbOpen.onupgradeneeded = e => e.target.result.createObjectStore('kv');
  idbOpen.onsuccess = () => console.log('IDB‑1 opened');
  
  /* IDB‑Example‑2:  Read/Write helper */
  function idbSet(key, val) {
    const tx = idbOpen.result.transaction('kv', 'readwrite');
    tx.objectStore('kv').put(val, key);
    return tx.complete;
  }
  
  /* IDB‑Example‑3:  Get helper */
  function idbGet(key) {
    return new Promise(r => {
      const req = idbOpen.result.transaction('kv').objectStore('kv').get(key);
      req.onsuccess = () => r(req.result);
    });
  }
  
  /* IDB‑Example‑4:  localForage usage */
  (async () => {
    try {
      const localforage = await import('https://esm.run/localforage?bundle');
      await localforage.default.setItem('lfKey', 'val');
      console.log('IDB‑4 localForage', await localforage.default.getItem('lfKey'));
    } catch {}
  })();
  
  /* IDB‑Example‑5:  Cursor iteration */
  function idbIter() {
    const req = idbOpen.result.transaction('kv').objectStore('kv').openCursor();
    req.onsuccess = e => {
      const cur = e.target.result;
      if (cur) { console.log('IDB‑5', cur.key, cur.value); cur.continue(); }
    };
  }
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION WASM — WebAssembly & JS Interop                          */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* WASM‑Example‑1:  Instantiate minimal module */
  const wasmBytes = new Uint8Array([
    0,97,115,109,1,0,0,0,1,4,1,96,1,127,1,127,3,2,1,0,7,7,1,3,add,0,0,10,9,1,7,0,20,0,11
  ]);
  WebAssembly.instantiate(wasmBytes).then(({instance}) =>
    console.log('WASM‑1 add(3,4)=', instance.exports.add(3,4)));
  
  /* WASM‑Example‑2:  Shared memory */
  (async () => {
    const memMod = new WebAssembly.Memory({ initial:1, maximum:1, shared:true });
    console.log('WASM‑2 shared bytes:', memMod.buffer.byteLength);
  })();
  
  /* WASM‑Example‑3:  Import JS function into WASM */
  (async () => {
    const bytes = new Uint8Array([0,97,115,109,1,0,0,0,1,6,1,96,0,0,2,7,1,2,js,2,log,0,0,3,2,1,0,7,
      5,1,1,f,0,0,10,6,1,4,0,16,0,11]);
    const imports = { js:{ log: () => console.log('WASM‑3 from JS') } };
    const { instance } = await WebAssembly.instantiate(bytes, imports);
    instance.exports.f();
  })();
  
  /* WASM‑Example‑4:  Fetch & compile streaming */
  (async () => {
    if (WebAssembly.instantiateStreaming) {
      const resp = fetch('hello.wasm').catch(()=>null);
      if (resp) WebAssembly.instantiateStreaming(resp, {}).then(() => console.log('WASM‑4 streamed'));
    }
  })();
  
  /* WASM‑Example‑5:  Passing array between JS & WASM (memory view) */
  (async () => {
    const bytes=[0,97,115,109,1,0,0,0,5,4,1,3,0,1,6,1,0,7,9,1,3,sum,0,0,10,11,1,9,0,32,0,32,1,106,11];
    const mem=new WebAssembly.Memory({initial:1});
    const {instance}=await WebAssembly.instantiate(new Uint8Array(bytes),{});
    console.log('WASM‑5 sum(5,6)=',instance.exports.sum(5,6));
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION GPU — WebGPU / WebGL2                                   */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* GPU‑Example‑1:  WebGPU adapter query */
  (async () => {
    if (navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      console.log('GPU‑1 adapter found?', !!adapter);
    }
  })();
  
  /* GPU‑Example‑2:  WebGPU device & queue */
  (async () => {
    if (navigator.gpu) {
      const device = await (await navigator.gpu.requestAdapter()).requestDevice();
      console.log('GPU‑2 default queue label', device.queue.label);
    }
  })();
  
  /* GPU‑Example‑3:  WebGL2 context creation */
  (function () {
    const canvas=document.createElement('canvas'); const gl=canvas.getContext('webgl2');
    console.log('GPU‑3 WebGL2 supported?', !!gl);
  })();
  
  /* GPU‑Example‑4:  Compile WebGL shader */
  (function () {
    const c=document.createElement('canvas'); const gl=c.getContext('webgl2');
    if (!gl) return;
    const vs=gl.createShader(gl.VERTEX_SHADER); gl.shaderSource(vs,'void main(){}'); gl.compileShader(vs);
    console.log('GPU‑4 shader status', gl.getShaderParameter(vs, gl.COMPILE_STATUS));
  })();
  
  /* GPU‑Example‑5:  WebGPU buffer mapping */
  (async () => {
    if (navigator.gpu) {
      const device=await (await navigator.gpu.requestAdapter()).requestDevice();
      const buf=device.createBuffer({size:4,usage:GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC,mappedAtCreation:true});
      new Uint32Array(buf.getMappedRange())[0]=123; buf.unmap(); console.log('GPU‑5 buffer written');
    }
  })();
  
  /*───────────────────────────────────────────────────────────────────*/
  /* SECTION MED — Media Capture, Streams & File System Access      */
  /*───────────────────────────────────────────────────────────────────*/
  
  /* MED‑Example‑1:  getUserMedia video */
  (async () => {
    if (navigator.mediaDevices?.getUserMedia) {
      try { const s=await navigator.mediaDevices.getUserMedia({video:true});
        console.log('MED‑1 tracks', s.getTracks().length); s.getTracks().forEach(t=>t.stop()); } catch{}
    }
  })();
  
  /* MED‑Example‑2:  MediaRecorder save blob */
  (async () => {
    if (window.MediaRecorder && navigator.mediaDevices?.getUserMedia) {
      const s=await navigator.mediaDevices.getUserMedia({audio:true});
      const rec=new MediaRecorder(s); const chunks=[];
      rec.ondataavailable=e=>chunks.push(e.data);
      rec.start(); setTimeout(()=>{rec.stop(); rec.onstop=()=>console.log('MED‑2 blob', new Blob(chunks));},1000);
    }
  })();
  
  /* MED‑Example‑3:  File System Access API */
  (async () => {
    if ('showOpenFilePicker' in window) {
      // const [handle] = await showOpenFilePicker();
      console.log('MED‑3 file picker available');
    }
  })();
  
  /* MED‑Example‑4:  Stream to <video> element */
  (function () {
    const vid=document.createElement('video'); vid.autoplay=true; document.body.appendChild(vid);
    navigator.mediaDevices?.getUserMedia({video:true}).then(s=>vid.srcObject=s).catch(()=>vid.remove());
  })();
  
  /* MED‑Example‑5:  Picture‑in‑Picture request */
  (function () {
    const video=document.querySelector('video');
    if (video?.requestPictureInPicture) {
      video.requestPictureInPicture().then(() => console.log('MED‑5 PiP entered')).catch(()=>{});
    }
  })();