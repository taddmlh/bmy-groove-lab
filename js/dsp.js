// Shared DSP: FFT, ADTOF-compatible log-filtered spectrogram (port of adtof_pytorch/audio.py).
// Pure functions, no DOM: runs in the browser and in Node tests.

export const SR = 44100;
export const FPS = 100;
export const HOP = 441;          // round(44100 / 100)
export const NFFT = 2048;
const HALF = NFFT / 2;

// --- radix-2 complex FFT (in place) ------------------------------------------------------
const bitrev = new Uint16Array(NFFT);
{
  const bits = Math.log2(NFFT);
  for (let i = 0; i < NFFT; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    bitrev[i] = r;
  }
}
const cosT = new Float64Array(HALF), sinT = new Float64Array(HALF);
for (let i = 0; i < HALF; i++) { cosT[i] = Math.cos(-2 * Math.PI * i / NFFT); sinT[i] = Math.sin(-2 * Math.PI * i / NFFT); }

function fft(re, im) {
  for (let i = 0; i < NFFT; i++) {
    const j = bitrev[i];
    if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let size = 2; size <= NFFT; size <<= 1) {
    const half = size >> 1, step = NFFT / size;
    for (let start = 0; start < NFFT; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cosT[k * step], wi = sinT[k * step];
        const a = start + k, b = a + half;
        const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

// np.hanning (symmetric)
const win = new Float64Array(NFFT);
for (let n = 0; n < NFFT; n++) win[n] = 0.5 - 0.5 * Math.cos(2 * Math.PI * n / (NFFT - 1));

// --- madmom-style log filterbank, exactly as audio.py builds it ----------------------------
function buildFilterbank() {
  const freqs = [];
  const factor = Math.pow(2, 1 / 12);
  for (let f = 20; f <= 20000 * (1 + 1e-12); f *= factor) freqs.push(f);
  const fftFreqs = new Float64Array(HALF);
  for (let k = 0; k < HALF; k++) fftFreqs[k] = k * SR / NFFT;
  const raw = freqs.map(f => {
    let best = 0, bd = Infinity;
    for (let k = 0; k < HALF; k++) { const d = Math.abs(fftFreqs[k] - f); if (d < bd) { bd = d; best = k; } }
    return best;
  });
  const bins = [];
  let last = -1;
  for (const b of raw) if (b > last) { bins.push(b); last = b; }
  const n = bins.length - 2;
  const filters = [];
  for (let i = 0; i < n; i++) {
    const l = bins[i], c = bins[i + 1], r = bins[i + 2];
    const w = new Map();
    if (r - l < 2) { w.set(l, 1); }
    else {
      for (let b = l; b < c; b++) w.set(b, (b - l) / (c - l));
      w.set(c, 1);
      for (let b = c + 1; b < Math.min(r, HALF); b++) w.set(b, (r - b) / (r - c));
    }
    let sum = 0; for (const v of w.values()) sum += v;
    if (sum === 0) sum = 1;
    const idx = [...w.keys()], val = [...w.values()].map(v => v / sum);
    filters.push({ idx: Int32Array.from(idx), val: Float32Array.from(val) });
  }
  return filters;
}
export const FILTERS = buildFilterbank();
export const NBANDS = FILTERS.length;   // 84
export const BAND_FREQ = (() => {       // centre frequency of each band (for class-specific loudness)
  const out = new Float32Array(NBANDS);
  FILTERS.forEach((f, i) => { let best = 0, bv = -1; f.idx.forEach((k, j) => { if (f.val[j] > bv) { bv = f.val[j]; best = k; } }); out[i] = best * SR / NFFT; });
  return out;
})();

/**
 * Log-filtered magnitude spectrogram of a mono 44.1 kHz signal: Float32Array [frames * NBANDS],
 * frame-major. Matches librosa.stft(center=True, pad_mode='constant', hann) -> |.|[:1024] ->
 * filterbank -> log10(1 + x).
 */
export function logSpec(mono, onProgress) {
  const pad = NFFT / 2;
  const total = mono.length + 2 * pad;
  const frames = 1 + Math.floor((total - NFFT) / HOP);
  const out = new Float32Array(frames * NBANDS);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT), mag = new Float32Array(HALF);
  for (let f = 0; f < frames; f++) {
    const s0 = f * HOP - pad;
    for (let n = 0; n < NFFT; n++) {
      const i = s0 + n;
      re[n] = (i >= 0 && i < mono.length ? mono[i] : 0) * win[n];
      im[n] = 0;
    }
    fft(re, im);
    for (let k = 0; k < HALF; k++) mag[k] = Math.hypot(re[k], im[k]);
    const o = f * NBANDS;
    for (let b = 0; b < NBANDS; b++) {
      const flt = FILTERS[b];
      let s = 0;
      for (let j = 0; j < flt.idx.length; j++) s += flt.val[j] * mag[flt.idx[j]];
      out[o + b] = Math.log10(1 + s);
    }
    if (onProgress && (f & 255) === 0) onProgress(f / frames);
  }
  return { data: out, frames };
}

export function toMono(L, R) {
  const m = new Float32Array(L.length);
  for (let i = 0; i < L.length; i++) m[i] = 0.5 * (L[i] + (R ? R[i] : L[i]));
  return m;
}
