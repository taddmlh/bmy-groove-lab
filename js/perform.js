// Perform: clap / tap / beatbox over a click into BMY lanes. Pure analysis helpers (no DOM).
import { logSpec, NBANDS, BAND_FREQ, FPS } from './dsp.js?b=20261126i';

/**
 * Onsets in a mono 44.1 kHz recording: [{t (s), levelDb, feat (unit vector, 84 bands)}].
 * Spectral-flux peaks over an adaptive threshold, at least `gap` s apart.
 */
export function detectOnsets(mono, { gap = 0.06, sens = 1 } = {}) {
  const { data, frames } = logSpec(mono);
  const flux = new Float32Array(frames);
  for (let f = 1; f < frames; f++) {
    let s = 0;
    for (let b = 0; b < NBANDS; b++) { const d = data[f * NBANDS + b] - data[(f - 1) * NBANDS + b]; if (d > 0) s += d; }
    flux[f] = s;
  }
  const sorted = Float32Array.from(flux).sort(), med = sorted[Math.floor(frames / 2)] || 0, hi = sorted[Math.floor(frames * 0.995)] || 1;
  const out = [];
  const W = 15;   // 150 ms local window
  for (let f = 2; f < frames - 2; f++) {
    if (!(flux[f] >= flux[f - 1] && flux[f] > flux[f + 1])) continue;
    let loc = 0; for (let k = Math.max(0, f - W); k <= Math.min(frames - 1, f + W); k++) loc += flux[k]; loc /= 2 * W + 1;
    const thr = Math.max(med * 3, loc * 1.6, hi * 0.08) / sens;
    if (flux[f] < thr) continue;
    const t = f / FPS;
    if (out.length && t - out[out.length - 1].t < gap) { if (flux[f] > out[out.length - 1].flux) out.pop(); else continue; }
    // level: peak RMS over 30 ms from the onset
    const s0 = Math.max(0, Math.round((t - 0.005) * 44100)), s1 = Math.min(mono.length, s0 + 1323);
    let pk = 0; for (let i = s0; i < s1; i += 64) { let e = 0; for (let j = i; j < Math.min(s1, i + 256); j++) e += mono[j] * mono[j]; pk = Math.max(pk, e / 256); }
    // spectral fingerprint: 40 ms after the onset, mean removed, unit length
    const feat = new Float32Array(NBANDS);
    for (let k = 0; k < 4 && f + k < frames; k++) for (let b = 0; b < NBANDS; b++) feat[b] += data[(f + k) * NBANDS + b];
    let m = 0; for (let b = 0; b < NBANDS; b++) m += feat[b]; m /= NBANDS;
    let n = 0; for (let b = 0; b < NBANDS; b++) { feat[b] -= m; n += feat[b] * feat[b]; }
    n = Math.sqrt(n) || 1; for (let b = 0; b < NBANDS; b++) feat[b] /= n;
    out.push({ t, flux: flux[f], levelDb: 10 * Math.log10(pk + 1e-12), feat });
  }
  return out;
}

export const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export function centroid(feats) {
  const c = new Float32Array(NBANDS);
  for (const f of feats) for (let b = 0; b < NBANDS; b++) c[b] += f[b];
  let n = 0; for (let b = 0; b < NBANDS; b++) n += c[b] * c[b]; n = Math.sqrt(n) || 1;
  for (let b = 0; b < NBANDS; b++) c[b] /= n;
  return c;
}
/** Index of the taught sound nearest to this onset, and how sure (cosine). */
export function classify(feat, sounds) {
  let best = -1, bs = -2;
  sounds.forEach((s, i) => { if (!s.centroid) return; const c = cosine(feat, s.centroid); if (c > bs) { bs = c; best = i; } });
  return { i: best, score: bs };
}

/** BPM from clapped/tapped quarter notes (onset times in s). Folds into 60-180. */
export function tempoFromOnsets(ts) {
  if (ts.length < 3) return null;
  const iv = []; for (let i = 1; i < ts.length; i++) iv.push(ts[i] - ts[i - 1]);
  iv.sort((a, b) => a - b);
  let bpm = 60 / iv[Math.floor(iv.length / 2)];
  while (bpm < 60) bpm *= 2; while (bpm > 180) bpm /= 2;
  return Math.round(bpm * 10) / 10;
}

/** Click sounds: accent (beat 1) and normal, short pure tones so they can be told apart from claps. */
export function clickBuffers(ctx) {
  const mk = (freq, dur = 0.03) => {
    const n = Math.round(dur * ctx.sampleRate), b = ctx.createBuffer(1, n, ctx.sampleRate), d = b.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.sin(2 * Math.PI * freq * i / ctx.sampleRate) * Math.exp(-i / (n / 5)) * 0.8;
    return b;
  };
  return { accent: mk(2637), beat: mk(1760) };
}
export const CLICK_FREQS = [1760, 2637];

/** True if this onset looks like the click (energy concentrated around the click tones). */
export function looksLikeClick(feat) {
  // the click is a pure tone: its fingerprint peaks in 1-2 bands around 1.76/2.64 kHz
  let top = 0, at = -1; for (let b = 0; b < NBANDS; b++) if (feat[b] > top) { top = feat[b]; at = b; }
  const f = BAND_FREQ[at] || 0;
  return top > 0.45 && CLICK_FREQS.some(c => Math.abs(Math.log2(f / c)) < 0.12);
}
