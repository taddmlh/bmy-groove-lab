// Magenta "OaF Drums" (Onsets & Frames trained on E-GMD; Apache 2.0 code, E-GMD CC BY 4.0) as a second transcriber,
// for comparing with ADTOF. models/oaf_drums.onnx = the onsets + velocity networks of the E-GMD checkpoint, converted
// (tools/oaf_export.py); output matches Magenta's TensorFlow model to ~1e-6.
// Front end = Magenta's training spectrogram: 44.1 kHz mono, 2048-point STFT (periodic Hann, centred, reflect padding),
// hop 512, 250 HTK mel bands from 30 Hz with Slaney area normalisation, power in dB (ref 1, floor 1e-10, top 80 dB).
import { fft } from './dsp.js?b=20261127i';

const SR = 44100, N = 2048, HOP = 512, NM = 250, K = N / 2 + 1;
export const OAF_FPS = SR / HOP;
// model outputs, in order: Magenta's 8-hit map bases 36, 38, 48, 46, 51, 53, 49
const OUT = [{ cls: 'kick' }, { cls: 'snare' }, { cls: 'tom' }, { cls: 'hihat' }, { cls: 'cymbal', kind: 'ride' }, { cls: 'cymbal', kind: 'ride' }, { cls: 'cymbal', kind: 'crash' }];

const WIN = new Float64Array(N); for (let n = 0; n < N; n++) WIN[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / N);
let MEL = null, MELF = null;
function melBank() {
  if (MEL) return MEL;
  const hz2mel = f => 2595 * Math.log10(1 + f / 700), mel2hz = m => 700 * (Math.pow(10, m / 2595) - 1);
  const m0 = hz2mel(30), m1 = hz2mel(SR / 2), pts = [];
  for (let i = 0; i < NM + 2; i++) pts.push(mel2hz(m0 + (m1 - m0) * i / (NM + 1)));
  MEL = new Float32Array(NM * K);
  MELF = new Float32Array(NM); for (let i = 0; i < NM; i++) MELF[i] = mel2hz(m0 + (m1 - m0) * i / (NM - 1));   // band frequencies as librosa.mel_frequencies (used by the hat rule)
  for (let m = 0; m < NM; m++) {
    const lo = pts[m], c = pts[m + 1], hi = pts[m + 2], norm = 2 / (hi - lo);
    for (let k = 0; k < K; k++) {
      const f = k * SR / N;
      const w = Math.max(0, Math.min((f - lo) / (c - lo), (hi - f) / (hi - c)));
      MEL[m * K + k] = w * norm;
    }
  }
  return MEL;
}

/** Log-mel frames for Magenta's model: { data: Float32Array [frames x 250] (dB), frames }. */
export function oafMel(x) {
  const mel = melBank(), pad = N / 2, L = x.length;
  const at = i => { if (i < 0) i = -i; if (i >= L) i = 2 * (L - 1) - i; return x[Math.max(0, Math.min(L - 1, i))]; };   // reflect
  const frames = 1 + Math.floor(L / HOP), data = new Float32Array(frames * NM);
  const re = new Float64Array(N), im = new Float64Array(N), pw = new Float64Array(K);
  let top = -Infinity;
  for (let f = 0; f < frames; f++) {
    const s = f * HOP - pad;
    for (let n = 0; n < N; n++) { re[n] = at(s + n) * WIN[n]; im[n] = 0; }
    fft(re, im);
    for (let k = 0; k < K; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < NM; m++) {
      let v = 0; const row = m * K;
      for (let k = 0; k < K; k++) { const w = mel[row + k]; if (w) v += w * pw[k]; }
      const db = 10 * Math.log10(Math.max(1e-10, v)); data[f * NM + m] = db; if (db > top) top = db;
    }
  }
  for (let i = 0; i < data.length; i++) if (data[i] < top - 80) data[i] = top - 80;
  return { data, frames };
}

let sess = null;
export async function loadOaf(ort, url = 'models/oaf_drums.onnx') {
  if (!sess) sess = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  return sess;
}

/**
 * Hits for x (mono, 44.1 kHz), times from 0. Each: { time, cls, kind?, oafVel (MIDI), p }.
 * hatFix: Magenta was trained on electronic kits and hears many real hi-hats as snare; a "snare" whose attack is mostly
 * highs (5 kHz+) with little body (150-1200 Hz) is turned back into a hi-hat (our own rule, threshold from Matt's stems).
 */
export async function oafHits(ort, x, { hatFix = true } = {}) {
  const s = await loadOaf(ort);
  const { data, frames } = oafMel(x);
  const out = await s.run({ mel: new ort.Tensor('float32', data, [1, frames, NM]) });
  const on = out.onset.data, vel = out.velocity.data, C = OUT.length;
  const hits = [];
  for (let c = 0; c < C; c++) {
    for (let f = 0; f < frames; f++) {
      const p = on[f * C + c], prev = f ? on[(f - 1) * C + c] : 0;
      if (p > 0.5 && prev <= 0.5) hits.push({ time: f / OAF_FPS, ...OUT[c], oafVel: Math.max(1, Math.min(127, Math.round(vel[f * C + c] * 127))), p, f });
    }
  }
  if (hatFix) {
    melBank();
    const lo = [], hi = []; for (let m = 0; m < NM; m++) { if (MELF[m] >= 150 && MELF[m] < 1200) lo.push(m); if (MELF[m] >= 5000) hi.push(m); }
    for (const h of hits) {
      if (h.cls !== 'snare') continue;
      let pl = 0, ph = 0, n = 0;
      for (let f = h.f; f < Math.min(frames, h.f + 3); f++, n++) {
        for (const m of lo) pl += Math.pow(10, data[f * NM + m] / 10) / lo.length;
        for (const m of hi) ph += Math.pow(10, data[f * NM + m] / 10) / hi.length;
      }
      h.hiLo = 10 * Math.log10(ph / pl);
      if (h.hiLo > -16.5) { h.cls = 'hihat'; h.fromSnare = true; }
    }
  }
  // one hit per drum per moment (ride + ride bell, or a re-classified hat on top of a real one)
  hits.sort((a, b) => a.time - b.time);
  const kept = [];
  for (const h of hits) if (!kept.some(k => k.cls === h.cls && (k.kind || '') === (h.kind || '') && Math.abs(k.time - h.time) < 0.03)) kept.push(h);
  kept.forEach(h => { delete h.f; });
  return kept;
}
