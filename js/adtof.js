// ADTOF drum transcription (Frame_RNN, 5 classes) via onnxruntime-web.
// Model: ADTOF-pytorch weights (xavriley/ADTOF-pytorch) exported to ONNX, parity 7e-7 vs PyTorch.
import { logSpec, NBANDS, FPS } from './dsp.js?b=20261127i';

export const CLASSES = ['kick', 'snare', 'tom', 'hihat', 'cymbal'];   // GM 35, 38, 47, 42, 49
export const THRESHOLDS = [0.22, 0.24, 0.32, 0.22, 0.30];

let session = null;
export async function loadAdtof(ort, url) {
  if (session) return session;
  session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
  return session;
}

/** Activations [frames x 5] for a mono 44.1 kHz signal. Long input is processed in overlapping chunks. */
export async function activations(ort, mono, spec) {
  const s = session;
  const { data, frames } = spec || logSpec(mono);
  const CH = 3000, OV = 200;                     // 30 s chunks, 2 s overlap each side
  const act = new Float32Array(frames * 5);
  for (let start = 0; start < frames; start += CH) {
    const a = Math.max(0, start - OV), b = Math.min(frames, start + CH + OV);
    const T = b - a;
    const input = new ort.Tensor('float32', data.subarray(a * NBANDS, b * NBANDS), [1, T, NBANDS, 1]);
    const out = (await s.run({ spec: input })).act.data;
    const keep0 = start, keep1 = Math.min(frames, start + CH);
    for (let f = keep0; f < keep1; f++) for (let c = 0; c < 5; c++) act[f * 5 + c] = out[(f - a) * 5 + c];
  }
  return { act, frames };
}

/** madmom NotePeakPickingProcessor as used by ADTOF (pre_avg .1, post_avg .01, pre_max .02, post_max .01, combine .02). */
export function pickPeaks(act, frames, cls, threshold) {
  const x = new Float32Array(frames);
  for (let f = 0; f < frames; f++) x[f] = act[f * 5 + cls];
  const preA = Math.round(0.1 * FPS), postA = Math.round(0.01 * FPS);
  const preM = Math.round(0.02 * FPS), postM = Math.round(0.01 * FPS);
  const proc = new Float32Array(frames);
  const n = preA + 1 + postA;
  for (let f = 0; f < frames; f++) {
    let s = 0;
    for (let k = -preA; k <= postA; k++) s += x[Math.min(frames - 1, Math.max(0, f + k))];
    proc[f] = Math.max(0, x[f] - s / n);
  }
  const peaks = [];
  for (let f = 0; f < frames; f++) {
    let mx = -Infinity;
    for (let k = -preM; k <= postM; k++) mx = Math.max(mx, proc[Math.min(frames - 1, Math.max(0, f + k))]);
    if (proc[f] >= mx && proc[f] >= threshold) peaks.push(f);
  }
  if (!peaks.length) return [];
  const comb = Math.max(1, Math.round(0.02 * FPS));
  const kept = [];
  let group = [peaks[0]];
  const flush = () => kept.push(group.reduce((b, i) => (proc[i] > proc[b] ? i : b), group[0]));
  for (let i = 1; i < peaks.length; i++) {
    if (peaks[i] - group[group.length - 1] <= comb) group.push(peaks[i]);
    else { flush(); group = [peaks[i]]; }
  }
  flush();
  return kept.map(f => ({ frame: f, time: f / FPS, strength: x[f] }));
}

/** All hits: [{time, cls: 'kick'|..., frame, strength}] sorted by time. */
export function hitsFrom(act, frames) {
  const out = [];
  CLASSES.forEach((name, c) => pickPeaks(act, frames, c, THRESHOLDS[c]).forEach(p => out.push({ ...p, cls: name })));
  return out.sort((a, b) => a.time - b.time);
}
