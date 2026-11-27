// BMY's own drum transcriber (trained by training/BMY_Transcriber_Colab.ipynb on licence-clean data).
// models/bmy_drums.onnx + models/bmy_drums.json. Front end = training/bmy_transcriber.features():
// -1 dBFS peak, periodic Hann, 2048-point FFT, hop 441 (100 fps), centred + reflect padding, power,
// 128 HTK mel bands 30 Hz - 16 kHz (Slaney norm), ln(power + 1e-7).
import { fft } from './dsp.js?b=20261127k';

const SR = 44100, N = 2048, HOP = 441, NM = 128, K = N / 2 + 1, FMIN = 30, FMAX = 16000;
export const BMYT_FPS = SR / HOP;
// model classes -> Groove Lab hit classes (and fixed lanes where the model already knows them)
const MAP = {
  kick: { cls: 'kick' }, snare: { cls: 'snare' },
  tomHigh: { cls: 'tom', modelLane: 'tomHigh' }, tomMid: { cls: 'tom', modelLane: 'tomMid' }, tomFloor: { cls: 'tom', modelLane: 'tomFloor' },
  hhClosed: { cls: 'hihat', open: false }, hhOpen: { cls: 'hihat', open: true },
  crash: { cls: 'cymbal', kind: 'crash' }, ride: { cls: 'cymbal', kind: 'ride' },
  // percussion (models from 2026-11-27 on): Groove Lab puts the busiest three on One-shot A/B/C
  shaker: { cls: 'perc', perc: 'shaker' }, tambourine: { cls: 'perc', perc: 'tambourine' }, bell: { cls: 'perc', perc: 'bell' },
  handDrum: { cls: 'perc', perc: 'handDrum' }, wood: { cls: 'perc', perc: 'wood' },
};

const WIN = new Float64Array(N); for (let n = 0; n < N; n++) WIN[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / N);
let MEL = null;
function melBank() {
  if (MEL) return MEL;
  const hz2mel = f => 2595 * Math.log10(1 + f / 700), mel2hz = m => 700 * (Math.pow(10, m / 2595) - 1);
  const m0 = hz2mel(FMIN), m1 = hz2mel(FMAX), pts = [];
  for (let i = 0; i < NM + 2; i++) pts.push(mel2hz(m0 + (m1 - m0) * i / (NM + 1)));
  MEL = new Float32Array(NM * K);
  for (let m = 0; m < NM; m++) {
    const lo = pts[m], c = pts[m + 1], hi = pts[m + 2], norm = 2 / (hi - lo);
    for (let k = 0; k < K; k++) { const f = k * SR / N; MEL[m * K + k] = Math.max(0, Math.min((f - lo) / (c - lo), (hi - f) / (hi - c))) * norm; }
  }
  return MEL;
}

/** x: mono 44.1 kHz, already at a -1 dBFS peak. Returns { data: Float32Array [frames x 128], frames }. */
export function bmytFeatures(x) {
  const mel = melBank(), pad = N / 2, L = x.length;
  const at = i => { if (i < 0) i = -i; if (i >= L) i = 2 * (L - 1) - i; return x[Math.max(0, Math.min(L - 1, i))]; };
  const frames = 1 + Math.floor(L / HOP), data = new Float32Array(frames * NM);
  const re = new Float64Array(N), im = new Float64Array(N), pw = new Float64Array(K);
  for (let f = 0; f < frames; f++) {
    const s = f * HOP - pad;
    for (let n = 0; n < N; n++) { re[n] = at(s + n) * WIN[n]; im[n] = 0; }
    fft(re, im);
    for (let k = 0; k < K; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < NM; m++) {
      let v = 0; const row = m * K;
      for (let k = 0; k < K; k++) { const w = mel[row + k]; if (w) v += w * pw[k]; }
      data[f * NM + m] = Math.log(v + 1e-7);
    }
  }
  return { data, frames };
}

let sess = null, meta = null;
export const BMYT_V = 'kits1';        // bump when a retrained model replaces models/bmy_drums.* (the worker caches models)
export async function loadBmyt(ort, url = `models/bmy_drums.onnx?v=${BMYT_V}`) {
  if (!sess) {
    meta = await (await fetch(typeof url === 'string' ? url.replace(/\.onnx(\?|$)/, '.json$1') : `models/bmy_drums.json?v=${BMYT_V}`)).json();
    sess = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  }
  return { sess, meta };
}
export const bmytAvailable = async () => { try { const r = await fetch(`models/bmy_drums.json?v=${BMYT_V}`, { method: 'HEAD', cache: 'no-store' }); return r.ok; } catch (e) { return false; } };

/** Hits for x (mono, 44.1 kHz, -1 dBFS), times from 0: { time, cls, modelLane?, kind?, open?, p }. */
export async function bmytHits(ort, x, opt = {}) {      // opt.weak: also return peaks down to weak x threshold (flag weak)
  const { sess, meta } = await loadBmyt(ort);
  const { data, frames } = bmytFeatures(x);
  const out = await sess.run({ mel: new ort.Tensor('float32', data, [1, frames, NM]) });
  const p = out.onset.data, C = meta.classes.length, gap = meta.min_gap_frames || 3;
  const hits = [];
  meta.classes.forEach((name, c) => {
    const thr = meta.thresholds[c], lo = thr * (opt.weak || 1);
    let last = -1e9, lastP = 0;
    for (let f = 0; f < frames; f++) {
      const v = p[f * C + c];
      if (v < lo) continue;
      if (f > 0 && p[(f - 1) * C + c] > v) continue;
      if (f + 1 < frames && p[(f + 1) * C + c] >= v) continue;
      if (f - last < gap) { if (v > lastP) { const h = hits[hits.length - 1]; h.time = f / BMYT_FPS; h.p = v; h.conf = v / thr; h.weak = v < thr; last = f; lastP = v; } continue; }
      hits.push({ time: f / BMYT_FPS, ...(MAP[name] || { cls: 'perc', perc: name }), p: v, model: name, conf: v / thr, weak: v < thr });
      last = f; lastP = v;
    }
  });
  return hits.sort((a, b) => a.time - b.time);
}
