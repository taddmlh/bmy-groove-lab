// Drum separation: htdemucs split into 21 half-precision parts (monteslu/htdemucs-web-onnx, MIT),
// run one part at a time on WebGPU. Pre/post-processing from demucs-web (MIT).
// Proven on iPad Pro 12.9" (4th gen), iPadOS 27: ~0.46x realtime.
const SPLIT_BASE = 'https://huggingface.co/monteslu/htdemucs-web-onnx/resolve/main/';
const CACHE = 'bmy-groove-lab-models-v1';
const CPU_NODES = ['/ReduceMean', '/Sub', '/Pow', '/ReduceMean_1', '/Clip', '/Sqrt', '/Add', '/Div',
  '/ReduceMean_2', '/Sub_1', '/Pow_1', '/ReduceMean_3', '/Clip_1', '/Sqrt_1', '/Add_1', '/Div_1'];
const TS = 343980, BINS = 2048, FRAMES = 336, STRIDE = Math.floor(TS * 0.75);
const sleep = ms => new Promise(r => setTimeout(r, ms));

export const SEPARATOR_VERSION = 'lean-6';
let dw = null, model = null;

// lean: no buffer pooling, so graphics memory freed by one part is really given back before the next part
const LEAN_GPU = {};   // tried 'disabled' (breaks tiny buffers) and 'lazyRelease' (used more memory)
const sessOpts = (i, final, lean) => ({
  executionProviders: [{ name: 'webgpu', ...(i === 0 ? { forceCpuNodeNames: CPU_NODES } : {}), ...(lean ? LEAN_GPU : {}) }],
  graphOptimizationLevel: 'all',
  preferredOutputLocation: final || lean ? 'cpu' : 'gpu-buffer'   // lean: results go through normal memory, so a part can be closed safely
});

async function cachedBytes(url) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch (e) {}
  let hit = cache && await cache.match(url);
  if (!hit) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Download failed (${res.status}): ${url.split('/').pop()}`);
    if (!cache) return new Uint8Array(await res.arrayBuffer());
    await cache.put(url, res);
    hit = await cache.match(url);
  }
  return new Uint8Array(await hit.arrayBuffer());
}

export function demucsLoaded() { return !!model; }

/** lean = phone: keep only the downloaded parts; each part is opened just before it runs and closed straight after (much less memory). */
export async function loadDemucs(ort, onStatus = () => {}, { lean = false } = {}) {
  if (model && model.lean === lean) return model;
  model = null;
  if (!('gpu' in navigator)) throw new Error('This browser has no WebGPU, which drum separation needs (iOS/iPadOS 26 or later).');
  dw = dw || await import('https://cdn.jsdelivr.net/npm/demucs-web@1.0.2/src/processor.js');
  const man = JSON.parse(new TextDecoder().decode(await cachedBytes(SPLIT_BASE + 'htdemucs_split_manifest.json')));
  const n = man.pieces.length, pieces = [], lastUse = {};
  man.pieces.forEach((p, i) => p.inputs.forEach(k => { lastUse[k] = i; }));
  for (let i = 0; i < n; i++) {
    const p = man.pieces[i];
    onStatus(`${lean ? 'Checking' : 'Loading'} drum separator, part ${i + 1} of ${n}…`, i / n, `loading separator part ${i + 1}`);
    const final = p.outputs.includes(man.outputs.freq) || p.outputs.includes(man.outputs.time);
    if (lean) { await cachedBytes(SPLIT_BASE + p.file); pieces.push({ url: SPLIT_BASE + p.file, inputs: p.inputs, final, i }); continue; }
    const bytes = await cachedBytes(SPLIT_BASE + p.file);
    const session = await ort.InferenceSession.create(bytes, sessOpts(i, final));
    pieces.push({ session, inputs: p.inputs, final, i });
    await sleep(5);
  }
  model = { pieces, lastUse, freq: man.outputs.freq, time: man.outputs.time, lean };
  return model;
}

async function runSegment(ort, input, onPart = () => {}) {
  const m = model;
  const env = {
    mix: new ort.Tensor('float32', input.waveform, [1, 2, TS]),
    mag: new ort.Tensor('float32', input.magSpec, [1, 4, BINS, FRAMES])
  };
  const dev = ort.env.webgpu && ort.env.webgpu.device;
  for (let i = 0; i < m.pieces.length; i++) {
    const p = m.pieces[i];
    const feeds = {}; for (const k of p.inputs) feeds[k] = env[k];
    onPart(i + 1, m.pieces.length);
    if (m.lean) {
      let bytes = await cachedBytes(p.url);
      const session = await ort.InferenceSession.create(bytes, sessOpts(p.i, p.final, true)); bytes = null;
      Object.assign(env, await session.run(feeds));
      if (dev) { try { await dev.queue.onSubmittedWorkDone(); } catch (e) {} }
      try { await session.release(); } catch (e) {}
    } else Object.assign(env, await p.session.run(feeds));
    for (const k of p.inputs) if (m.lastUse[k] === i && k !== 'mix' && k !== 'mag') { try { env[k].dispose(); } catch (e) {} delete env[k]; }
    if (dev) { try { await dev.queue.onSubmittedWorkDone(); } catch (e) {} }
    await sleep(1);
  }
  return { freq: env[m.freq].data, time: env[m.time].data, dims: env[m.time].dims };
}

/** Drums only. L, R: Float32Array at 44.1 kHz. onProg(fraction). Returns {left, right}. */
export async function separateDrums(ort, L, R, onProg = () => {}, onPart = () => {}) {
  const total = L.length;
  const nSeg = Math.ceil(Math.max(0, total - TS) / STRIDE) + 1;
  const left = new Float32Array(total), right = new Float32Array(total), w = new Float32Array(total);
  let seg = 0;
  for (let start = 0; start < total; start += STRIDE) {
    const segLen = Math.min(TS, total - start);
    const sl = new Float32Array(TS), sr = new Float32Array(TS);
    sl.set(L.subarray(start, start + segLen)); sr.set(R.subarray(start, start + segLen));
    onPart(seg + 1, nSeg, 0, model.pieces.length);                          // 'part 0' = preparing the input
    const r = await runSegment(ort, dw.prepareModelInput(sl, sr), (i, n) => onPart(seg + 1, nSeg, i, n));
    const drums = dw.standaloneIspec(dw.standaloneMask(r.freq)[0], TS);   // track 0 = drums
    const ns = r.dims[3];
    for (let i = 0; i < segLen; i++) {
      const wi = Math.min(i / (STRIDE * 0.5), (segLen - i) / (STRIDE * 0.5), 1);
      left[start + i] += (r.time[i] + (drums.left[i] || 0)) * wi;
      right[start + i] += (r.time[ns + i] + (drums.right[i] || 0)) * wi;
      w[start + i] += wi;
    }
    onProg(++seg / nSeg);
    if (start + TS >= total) break;
  }
  for (let i = 0; i < total; i++) if (w[i] > 0) { left[i] /= w[i]; right[i] /= w[i]; }
  return { left, right };
}
