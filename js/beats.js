// Tempo, beat phase and downbeat from a log-filtered spectrogram (100 fps). Constant tempo, 4/4.
import { NBANDS, BAND_FREQ, FPS } from './dsp.js';

function bandFlux(spec, frames, lo, hi) {
  const bands = [];
  for (let b = 0; b < NBANDS; b++) if (BAND_FREQ[b] >= lo && BAND_FREQ[b] < hi) bands.push(b);
  const out = new Float32Array(frames);
  for (let f = 1; f < frames; f++) {
    let s = 0;
    for (const b of bands) { const d = spec[f * NBANDS + b] - spec[(f - 1) * NBANDS + b]; if (d > 0) s += d; }
    out[f] = s;
  }
  // subtract a 0.5 s local mean, keep the positive part
  const w = 25, res = new Float32Array(frames);
  let acc = 0;
  for (let f = 0; f < frames; f++) {
    acc += out[f]; if (f - 2 * w - 1 >= 0) acc -= out[f - 2 * w - 1];
    const c = f - w;
    if (c >= 0) res[c] = Math.max(0, out[c] - acc / (2 * w + 1));
  }
  return res;
}

const sample = (env, t) => {     // linear interpolation, frames
  const i = Math.floor(t), fr = t - i;
  if (i < 0 || i + 1 >= env.length) return 0;
  return env[i] * (1 - fr) + env[i + 1] * fr;
};

/** -> { bpm, t0 } where t0 (seconds) is a downbeat. */
export function detectGrid(spec, frames) {
  const env = bandFlux(spec, frames, 30, 16000);
  const low = bandFlux(spec, frames, 30, 150);
  const mid = bandFlux(spec, frames, 180, 4000);

  // 1. coarse tempo: autocorrelation with a soft prior around 120 BPM
  const ac = L => { let s = 0; for (let f = L; f < frames; f++) s += env[f] * env[f - L]; return s / (frames - L); };
  let best = 0, bestL = 50;
  for (let L = 25; L <= 100; L++) {
    const bpm = 60 * FPS / L;
    const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120) / 0.9, 2));
    const sc = (ac(L) + 0.5 * ac(2 * L) + 0.25 * (L % 2 ? 0 : ac(L / 2))) * prior;
    if (sc > best) { best = sc; bestL = L; }
  }
  // 2. fine tempo + phase
  const coarse = 60 * FPS / bestL;
  let bb = { score: -1 };
  for (let bpm = coarse - 3; bpm <= coarse + 3; bpm += 0.05) {
    const P = 60 * FPS / bpm;
    for (let ph = 0; ph < P; ph += 0.5) {
      let s = 0, n = 0;
      for (let t = ph; t < frames; t += P) { s += sample(env, t); n++; }
      s /= n;
      if (s > bb.score) bb = { score: s, bpm, ph };
    }
  }
  // 3. downbeat: kick on 1 (and 3), snare on 2 and 4
  const P = 60 * FPS / bb.bpm;
  const sums = [0, 0, 0, 0].map(() => ({ low: 0, mid: 0 }));
  let k = 0;
  for (let t = bb.ph; t < frames; t += P, k++) { sums[k % 4].low += sample(low, t); sums[k % 4].mid += sample(mid, t); }
  let bestJ = 0, bestS = -Infinity;
  for (let j = 0; j < 4; j++) {
    const s = sums[j].low + 0.5 * sums[(j + 2) % 4].low + sums[(j + 1) % 4].mid + sums[(j + 3) % 4].mid
      - 0.5 * (sums[(j + 1) % 4].low + sums[(j + 3) % 4].low) - 0.5 * (sums[j].mid + sums[(j + 2) % 4].mid);
    if (s > bestS) { bestS = s; bestJ = j; }
  }
  let t0 = (bb.ph + bestJ * P) / FPS;
  const barSec = 4 * 60 / bb.bpm;
  while (t0 - barSec >= 0) t0 -= barSec;
  return { bpm: Math.round(bb.bpm * 100) / 100, t0 };
}

/** Bar start times covering [0, duration). */
export function barTimes(grid, duration) {
  const barSec = 4 * 60 / grid.bpm;
  let t = grid.t0;
  while (t > 0) t -= barSec;
  const out = [];
  for (; t < duration; t += barSec) out.push(t);
  return out;
}
