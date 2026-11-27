// Gentle hiss reduction for recordings: learns the steady background noise from the quietest moments
// (10th-percentile power per frequency), then turns down only that part of the sound (spectral gating),
// limited to `reduceDb` so music and drum transients are left alone.
import { fft, NFFT } from './dsp.js?b=20261127h';

const N = NFFT, H = N / 4, K = N / 2 + 1;
const W = new Float64Array(N); for (let n = 0; n < N; n++) W[n] = Math.sin(Math.PI * n / N);   // sqrt-Hann: analysis x synthesis = Hann

function frameSpec(x, start, re, im) {
  for (let n = 0; n < N; n++) { const i = start + n; re[n] = i >= 0 && i < x.length ? x[i] * W[n] : 0; im[n] = 0; }
  fft(re, im);
}

/** x: Float32Array (one channel). Returns { out, noiseDb } where noiseDb is the hiss level found. */
export function denoise(x, { reduceDb = 18, over = 1.5 } = {}) {
  const F = Math.ceil((x.length + N) / H), start0 = -N + H;
  const re = new Float64Array(N), im = new Float64Array(N);
  const pow = new Float32Array(F * K);
  for (let f = 0; f < F; f++) {
    frameSpec(x, start0 + f * H, re, im);
    for (let k = 0; k < K; k++) pow[f * K + k] = re[k] * re[k] + im[k] * im[k];
  }
  // noise profile: 10th percentile over time, per bin
  const noise = new Float32Array(K), col = new Float32Array(F);
  for (let k = 0; k < K; k++) { for (let f = 0; f < F; f++) col[f] = pow[f * K + k]; col.sort(); noise[k] = col[Math.floor(F * 0.1)] / 0.105; }   // hiss power fluctuates (exponential): its 10th percentile is ~0.105 x its mean
  let nsum = 0; for (let k = 0; k < K; k++) nsum += noise[k];
  const noiseDb = 10 * Math.log10(Math.max(1e-20, 2 * nsum / (N * N / 8)));       // rough level of the hiss
  const gmin = Math.pow(10, -reduceDb / 20), out = new Float32Array(x.length);
  const g = new Float32Array(K), prev = new Float32Array(K).fill(1);
  for (let f = 0; f < F; f++) {
    const s = start0 + f * H;
    frameSpec(x, s, re, im);
    for (let k = 0; k < K; k++) { const p = pow[f * K + k]; g[k] = Math.max(gmin, p > 0 ? 1 - over * noise[k] / p : gmin); }
    for (let k = 0; k < K; k++) {                         // smooth across frequency, and release slowly over time (no "watery" artefacts)
      const sm = (g[Math.max(0, k - 1)] + 2 * g[k] + g[Math.min(K - 1, k + 1)]) / 4;
      const v = Math.max(sm, prev[k] * 0.75); prev[k] = v;
      re[k] *= v; im[k] *= v; if (k > 0 && k < N / 2) { re[N - k] *= v; im[N - k] *= v; }
    }
    // inverse FFT: conj -> fft -> conj, / N
    for (let n = 0; n < N; n++) im[n] = -im[n];
    fft(re, im);
    for (let n = 0; n < N; n++) { const i = s + n; if (i >= 0 && i < out.length) out[i] += re[n] / N * W[n] / 2; }   // Hann at 75 % overlap sums to 2
  }
  return { out, noiseDb };
}
