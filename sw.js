// BMY Groove Lab offline worker: keeps the app itself on the device so it opens without internet.
// Songs, stems and Perform takes are in IndexedDB; the drum-separator parts in the app's own model cache.
// The page is fetched fresh when online (so updates arrive); every versioned file (?b= build, ?v= kit) is
// kept once fetched. Bump Q with each build (same string as the ?b= on the module imports).
const Q = '?b=20261128e';
const SHELL = 'gl-shell' + Q, LIBS = 'gl-libs-v1', KITS = 'gl-kits-v5';
const MODULES = ['app', 'adtof', 'bank', 'beats', 'db', 'demucs', 'bmyt', 'denoise', 'dsp', 'oaf', 'perform', 'playable', 'synth', 'transcribe'];
const PRECACHE = ['./', 'index.html', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/apple-touch-icon.png', 'js/rec-worklet.js',
  ...MODULES.map(m => `js/${m}.js${Q}`)];

self.addEventListener('install', e => {
  e.waitUntil((async () => {
    const c = await caches.open(SHELL);
    await Promise.all(PRECACHE.map(u => c.add(new Request(u, { cache: 'reload' })).catch(() => {})));
    const l = await caches.open(LIBS);                         // the transcriber (1.8 MB), so Re-transcribe works offline
    if (!(await l.match('models/adtof.onnx'))) await l.add('models/adtof.onnx').catch(() => {});
    await self.skipWaiting();
  })());
});
self.addEventListener('activate', e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith('gl-shell') && k !== SHELL) await caches.delete(k);
    await self.clients.claim();
  })());
});

async function cacheFirst(req, name) {
  const c = await caches.open(name), hit = await c.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok || res.type === 'opaque') c.put(req, res.clone()).catch(() => {});
  return res;
}
async function staleWhileRevalidate(req) {      // unversioned files (worklet, icons): quick from cache, refreshed behind
  const c = await caches.open(SHELL), hit = await c.match(req);
  const fresh = fetch(req).then(res => { if (res.ok) c.put(req, res.clone()).catch(() => {}); return res; }).catch(() => null);
  return hit || (await fresh) || Response.error();
}
async function networkFirst(req) {
  const c = await caches.open(SHELL);
  try {
    const res = await Promise.race([fetch(req), new Promise((_, rej) => setTimeout(() => rej(new Error('slow')), 4000))]);
    if (res.ok) c.put('index.html', res.clone()).catch(() => {});
    return res;
  } catch (err) {
    return (await c.match('index.html')) || (await c.match('./')) || Response.error();
  }
}

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET' || req.headers.has('range')) return;
  const u = new URL(req.url);
  if (u.origin === location.origin) {
    if (req.mode === 'navigate' || /\/(index\.html)?$/.test(u.pathname)) return e.respondWith(networkFirst(req));
    if (/\/speedtest\.html$/.test(u.pathname)) return;
    if (u.pathname.includes('/kit/')) return e.respondWith(cacheFirst(req, KITS));
    if (u.pathname.includes('/models/')) return e.respondWith(cacheFirst(req, LIBS));
    return e.respondWith(u.search ? cacheFirst(req, SHELL) : staleWhileRevalidate(req));   // js (?b=build) / worklet, icons, manifest
  }
  if (u.hostname === 'cdn.jsdelivr.net') return e.respondWith(cacheFirst(req, LIBS));   // ONNX Runtime + Demucs helpers (versioned URLs)
  // everything else (Hugging Face model parts: the app keeps those in its own cache) goes straight to the network
});
