// Test rig: run Groove Lab's own transcriber code (js/adtof.js, js/bmyt.js) on WAV files in Node.
//   node transcribe.mjs <repo> <adtof|bmy> <out.json> <file.wav>... [--model path/to/bmy_drums.onnx]
// Needs onnxruntime-web (npm install onnxruntime-web@1.31.1) next to this script or on NODE_PATH.
// Output: { "<wav>": [ { time, cls, model? }, ... ] }, times in seconds from the start of the file.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const args = process.argv.slice(2);
const mi = args.indexOf('--model');
const modelPath = mi >= 0 ? args.splice(mi, 2)[1] : null;
const [repo, which, outFile, ...wavs] = args;
const require = createRequire(path.join(process.cwd(), 'x.js'));
const ort = require('onnxruntime-web');
ort.env.wasm.numThreads = 1;

// the app fetches the model's JSON; here it is a local file
globalThis.fetch = async (u) => {
  const p = String(u).replace(/\?.*$/, '');
  const f = path.isAbsolute(p) ? p : path.join(repo, p);
  return { ok: fs.existsSync(f), json: async () => JSON.parse(fs.readFileSync(f, 'utf8')) };
};

function readWav(file) {                              // PCM 16-bit or float 32, any channels -> mono Float32
  const b = fs.readFileSync(file);
  let p = 12, fmt = null, data = null;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4), n = b.readUInt32LE(p + 4);
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(p + 8), ch: b.readUInt16LE(p + 10), sr: b.readUInt32LE(p + 12), bits: b.readUInt16LE(p + 22) };
    if (id === 'data') data = b.subarray(p + 8, p + 8 + n);
    p += 8 + n + (n & 1);
  }
  if (fmt.sr !== 44100) throw new Error(`${file}: ${fmt.sr} Hz (need 44100)`);
  const bytes = fmt.bits / 8, frames = Math.floor(data.length / (bytes * fmt.ch)), x = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let s = 0;
    for (let c = 0; c < fmt.ch; c++) {
      const o = (i * fmt.ch + c) * bytes;
      s += fmt.tag === 3 ? data.readFloatLE(o) : data.readInt16LE(o) / 32768;
    }
    x[i] = s / fmt.ch;
  }
  return x;
}

function normalise(raw) {                             // exactly as app.js transcribeRegion: -1 dBFS peak, max +30 dB
  let peak = 0;
  for (const v of raw) peak = Math.max(peak, Math.abs(v));
  const gain = peak > 0 ? Math.min(31.6, 0.891 / peak) : 1;
  return gain === 1 ? raw : raw.map(v => v * gain);
}

const js = f => import(pathToFileURL(path.join(repo, 'js', f)).href);
const out = {};
if (which === 'adtof') {
  const { logSpec } = await js('dsp.js');
  const A = await js('adtof.js');
  await A.loadAdtof(ort, path.join(repo, 'models/adtof.onnx'));
  for (const w of wavs) {
    const mono = normalise(readWav(w));
    const { act, frames } = await A.activations(ort, mono, logSpec(mono));
    out[w] = A.hitsFrom(act, frames).map(h => ({ time: +h.time.toFixed(3), cls: h.cls }));
  }
} else {
  const B = await js('bmyt.js');
  await B.loadBmyt(ort, modelPath || path.join(repo, 'models/bmy_drums.onnx'));
  for (const w of wavs) {
    const mono = normalise(readWav(w));
    out[w] = (await B.bmytHits(ort, mono)).map(h => ({ time: +h.time.toFixed(3), cls: h.cls, model: h.model }));
  }
}
fs.writeFileSync(outFile, JSON.stringify(out));
console.log(`${which}: ${wavs.length} files -> ${outFile}`);
