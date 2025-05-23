/****************************************************************************************
 * Chapter 14 | Advanced Browser & Platform APIs
 * Single-file .js with ≥5 examples/section.
 ****************************************************************************************/

/* SECTION SW — Service Workers & Cache API */

// SW‑1: Register Service Worker
(function SW1_Register() {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js')
        .then(r=>console.log('SW1 registered:', r.scope))
        .catch(e=>console.error('SW1 reg failed:', e));
    }
  })();
  
  // SW‑2: Install & precache (sw.js)
  const swCode = `self.addEventListener('install', e=>{
    const cacheName='v1';
    e.waitUntil(caches.open(cacheName).then(c=>c.addAll([
      '/','/offline.html','/styles.css','/app.js'
    ])));
    self.skipWaiting();
  });`;
  const swBlob = new Blob([swCode], { type:'application/javascript' });
  navigator.serviceWorker.register(URL.createObjectURL(swBlob));
  
  // SW‑3: Fetch event runtime caching
  ;(function SW3_Fetch() {
    const code = `
  self.addEventListener('fetch', e=>{
    e.respondWith(
      caches.match(e.request).then(r=>r||fetch(e.request))
    );
  });`;
    const blob = new Blob([code],{type:'application/javascript'});
    navigator.serviceWorker.register(URL.createObjectURL(blob));
  })();
  
  // SW‑4: Activate & cleanup old caches
  ;(function SW4_Activate() {
    const code=`self.addEventListener('activate', e=>{
    const keep=['v1'];
    e.waitUntil(
      caches.keys().then(keys=>Promise.all(
        keys.map(k=>keep.includes(k)?null:caches.delete(k))
      ))
    );
  });`;
    navigator.serviceWorker.register(URL.createObjectURL(new Blob([code],{type:'application/javascript'})));
  })();
  
  // SW‑5: Offline fallback
  ;(function SW5_Offline() {
    const code=`self.addEventListener('fetch', e=>{
    e.respondWith(
      fetch(e.request).catch(()=>caches.match('/offline.html'))
    );
  });`;
    navigator.serviceWorker.register(URL.createObjectURL(new Blob([code],{type:'application/javascript'})));
  })();
  
  /* SECTION IDB — IndexedDB & LocalForage */
  
  // IDB‑1: Open DB & add record
  ;(function IDB1_OpenAdd() {
    const req = indexedDB.open('db1',1);
    req.onupgradeneeded = ()=>req.result.createObjectStore('store',{keyPath:'id'});
    req.onsuccess = ()=> {
      const tx = req.result.transaction('store','readwrite');
      tx.objectStore('store').add({id:1,name:'Alice'});
      tx.oncomplete = ()=>console.log('IDB1 added');
    };
  })();
  
  // IDB‑2: Get record
  ;(function IDB2_Get() {
    const req = indexedDB.open('db1');
    req.onsuccess = ()=> {
      const s=req.result.transaction('store').objectStore('store');
      s.get(1).onsuccess = e=>console.log('IDB2 got',e.target.result);
    };
  })();
  
  // IDB‑3: Update & delete
  ;(function IDB3_UpdateDelete() {
    const req = indexedDB.open('db1');
    req.onsuccess = ()=> {
      const tx=req.result.transaction('store','readwrite');
      const os=tx.objectStore('store');
      os.put({id:1,name:'Bob'});
      os.delete(1);
      tx.oncomplete = ()=>console.log('IDB3 update/delete done');
    };
  })();
  
  // IDB‑4: Cursor iteration
  ;(function IDB4_Cursor() {
    const req=indexedDB.open('db2',1);
    req.onupgradeneeded = ()=>req.result.createObjectStore('s',{autoIncrement:true});
    req.onsuccess = ()=>{
      const os=req.result.transaction('s','readwrite').objectStore('s');
      [10,20,30].forEach(v=>os.add(v));
      os.openCursor().onsuccess=e=>{
        const c=e.target.result;
        if(c){console.log('IDB4 val',c.value); c.continue();}
      };
    };
  })();
  
  // IDB‑5: LocalForage usage (requires localforage lib)
  ;(function IDB5_LocalForage() {
    if (!window.localforage) return console.warn('LocalForage missing');
    localforage.config({name:'lfDB'});
    localforage.setItem('key','value').then(()=>localforage.getItem('key'))
      .then(v=>console.log('IDB5 LF got',v))
      .catch(e=>console.error(e));
  })();
  
  /* SECTION WASM — WebAssembly & JS Interop */
  
  // WASM‑1: Fetch & instantiate
  ;(async function WASM1_Fetch() {
    try {
      const resp = await fetch('module.wasm');
      const buf = await resp.arrayBuffer();
      const mod = await WebAssembly.instantiate(buf);
      console.log('WASM1 result', mod.instance.exports.add(2,3));
    } catch(e) { console.error('WASM1 error',e); }
  })();
  
  // WASM‑2: Streaming compile
  ;(async function WASM2_Stream() {
    if (WebAssembly.instantiateStreaming) {
      const { module, instance } = await WebAssembly.instantiateStreaming(
        fetch('module.wasm'),{}
      );
      console.log('WASM2 mul', instance.exports.mul(4,5));
    }
  })();
  
  // WASM‑3: Memory read/write
  ;(async function WASM3_Memory() {
    const { instance } = await WebAssembly.instantiateStreaming(fetch('mem.wasm'),{});
    const mem = new Uint32Array(instance.exports.memory.buffer);
    console.log('WASM3 before', mem[0]);
    mem[0]=42;
    instance.exports.read(); // uses updated mem
  })();
  
  // WASM‑4: JS import into WASM
  ;(async function WASM4_Import() {
    const imports = { env: { log: x=>console.log('WASM4 log',x) }};
    const { instance } = await WebAssembly.instantiateStreaming(fetch('imp.wasm'),imports);
    instance.exports.callLog(7);
  })();
  
  // WASM‑5: Error handling
  ;(async function WASM5_Error() {
    try {
      await WebAssembly.instantiateStreaming(fetch('bad.wasm'),{});
    } catch(e) {
      console.error('WASM5 inst failed:', e);
    }
  })();
  
  /* SECTION GPU — WebGPU & WebGL2 */
  
  // GPU‑1: WebGL2 init & clear
  ;(function GPU1_GL2() {
    const c=document.createElement('canvas');
    document.body.appendChild(c);
    const gl=c.getContext('webgl2');
    if(!gl) return console.error('GPU1 no WebGL2');
    gl.clearColor(0,0.5,0.5,1); gl.clear(gl.COLOR_BUFFER_BIT);
    console.log('GPU1 cleared');
  })();
  
  // GPU‑2: Shader compile & program link
  ;(function GPU2_Shader() {
    const c=document.createElement('canvas');document.body.append(c);
    const gl=c.getContext('webgl2');
    const vs=`#version 300 es
  in vec4 p;void main(){gl_Position=p;}`;
    const fs=`#version 300 es
  precision highp float;out vec4 o;void main(){o=vec4(1,0,0,1);}`;
    function compile(src,type){const s=gl.createShader(type);
      gl.shaderSource(s,src);gl.compileShader(s);
      if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(s));
      return s;
    }
    const prog=gl.createProgram();
    gl.attachShader(prog,compile(vs,gl.VERTEX_SHADER));
    gl.attachShader(prog,compile(fs,gl.FRAGMENT_SHADER));
    gl.linkProgram(prog);
    gl.useProgram(prog);
    console.log('GPU2 program linked');
  })();
  
  // GPU‑3: Draw triangle
  ;(function GPU3_Tri() {
    const c=document.createElement('canvas');document.body.append(c);
    const gl=c.getContext('webgl2');
    const vs=`#version 300 es
  in vec2 a;void main(){gl_Position=vec4(a,0,1);}`;
    const fs=`#version 300 es
  precision mediump float;out vec4 o;void main(){o=vec4(0,1,0,1);}`;
    function s(src,t){const sh=gl.createShader(t);
      gl.shaderSource(sh,src);gl.compileShader(sh);
      return sh;
    }
    const p=gl.createProgram();
    gl.attachShader(p,s(vs,gl.VERTEX_SHADER));
    gl.attachShader(p,s(fs,gl.FRAGMENT_SHADER));
    gl.linkProgram(p);gl.useProgram(p);
    const buf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buf);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([
      0,1,-1,-1,1,-1
    ]),gl.STATIC_DRAW);
    const loc=gl.getAttribLocation(p,'a');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
    gl.drawArrays(gl.TRIANGLES,0,3);
    console.log('GPU3 triangle drawn');
  })();
  
  // GPU‑4: WebGPU compute (if supported)
  ;(async function GPU4_WebGPU() {
    if (!navigator.gpu) return console.warn('GPU4 no WebGPU');
    const ad=await navigator.gpu.requestAdapter();
    const dev=await ad.requestDevice();
    const buf=dev.createBuffer({size:4*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const code=`@compute @workgroup_size(1)
  fn main(@builtin(global_invocation_id)i:vec3<u32>,
    @group(0) @binding(0) buf:ptr<storage,u32>) {
    (*buf)=42u;
  }`;
    const mod=dev.createShaderModule({code});
    const pipeline=dev.createComputePipeline({
      compute:{module:mod,entryPoint:'main'}
    });
    const bind=dev.createBindGroup({
      layout:pipeline.getBindGroupLayout(0),
      entries:[{binding:0,resource:{buffer:buf}}]
    });
    const cmd=dev.createCommandEncoder();
    const pass=cmd.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatch(1);pass.end();
    dev.queue.submit([cmd.finish()]);
    console.log('GPU4 compute submitted');
  })();
  
  // GPU‑5: Texture upload & draw (WebGL2)
  ;(function GPU5_Tex() {
    const img=new Image();
    img.src='tex.png'; img.onload=()=>{
      const c=document.createElement('canvas');document.body.append(c);
      const gl=c.getContext('webgl2');
      const tex=gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D,tex);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);
      gl.generateMipmap(gl.TEXTURE_2D);
      console.log('GPU5 texture loaded');
    };
  })();
  
  /* SECTION MEDIA — Media Capture, Streams & File System Access */
  
  // MEDIA‑1: getUserMedia audio/video
  ;(function MEDIA1_UM() {
    navigator.mediaDevices.getUserMedia({ video:true,audio:true })
      .then(s=>{const v=document.createElement('video');
        v.srcObject=s;v.autoplay=true;v.muted=true;
        document.body.append(v);
        console.log('MEDIA1 stream active');
      }).catch(e=>console.error('MEDIA1 fail',e));
  })();
  
  // MEDIA‑2: enumerateDevices
  ;(function MEDIA2_Enum() {
    navigator.mediaDevices.enumerateDevices()
      .then(devs=>devs.forEach(d=>console.log('MEDIA2',d.kind,d.label)))
      .catch(e=>console.error(e));
  })();
  
  // MEDIA‑3: Record with MediaRecorder
  ;(function MEDIA3_Record() {
    navigator.mediaDevices.getUserMedia({ audio:true })
      .then(stream=>{
        const rec=new MediaRecorder(stream);
        const chunks=[];
        rec.ondataavailable=e=>chunks.push(e.data);
        rec.onstop=()=>{
          const blob=new Blob(chunks);
          console.log('MEDIA3 recorded',blob.size);
        };
        rec.start();
        setTimeout(()=>rec.stop(),3000);
      });
  })();
  
  // MEDIA‑4: Pipe capture to canvas
  ;(function MEDIA4_Canvas() {
    navigator.mediaDevices.getUserMedia({video:true})
      .then(stream=>{
        const video=document.createElement('video');
        const canvas=document.createElement('canvas');
        document.body.append(video,canvas);
        video.srcObject=stream;video.play();
        video.onplay=()=>{
          const ctx=canvas.getContext('2d');
          canvas.width=video.videoWidth;canvas.height=video.videoHeight;
          setInterval(()=>ctx.drawImage(video,0,0),100);
        };
      }).catch(e=>console.error(e));
  })();
  
  // MEDIA‑5: File System Access API
  ;(async function MEDIA5_FS() {
    if (!window.showOpenFilePicker) return console.warn('MEDIA5 FS not supported');
    try {
      const [fh]=await showOpenFilePicker();
      const wf=await fh.createWritable();
      await wf.write('Hello FileSystem');
      await wf.close();
      console.log('MEDIA5 wrote file');
    } catch(e) {
      console.error('MEDIA5 error',e);
    }
  })();