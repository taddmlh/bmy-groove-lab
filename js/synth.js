// Tiny drum synth for auditioning a BMY-style pattern (not a kit: just enough to check timing).
import { VEL_MIDI } from './transcribe.js';
const SR = 44100;

const voices = {};
function voice(kind) {
  if (voices[kind]) return voices[kind];
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
  const dur = { kick: .35, snare: .22, rim: .12, hat: .05, ohat: .45, tomHigh: .35, tomMid: .4, tomFloor: .5, crash: 1.4, ride: .5, shot: .15 }[kind];
  const o = new Float32Array(Math.round(dur * SR));
  let lp = 0, ph = 0;
  for (let i = 0; i < o.length; i++) {
    const t = i / SR; let v = 0;
    switch (kind) {
      case 'kick': ph += 2 * Math.PI * (48 + 100 * Math.exp(-t * 32)) / SR; v = Math.sin(ph) * Math.exp(-t * 10); break;
      case 'snare': ph += 2 * Math.PI * 190 / SR; v = 0.45 * Math.sin(ph) * Math.exp(-t * 28) + 0.6 * rnd() * Math.exp(-t * 20); break;
      case 'rim': ph += 2 * Math.PI * 420 / SR; v = 0.6 * Math.sin(ph) * Math.exp(-t * 60) + 0.4 * rnd() * Math.exp(-t * 50); break;
      case 'hat': case 'ohat': { const n = rnd(); v = (n - lp) * 0.45 * Math.exp(-t * (kind === 'hat' ? 75 : 7)); lp = n; break; }
      case 'tomHigh': case 'tomMid': case 'tomFloor': {
        const f = { tomHigh: 190, tomMid: 140, tomFloor: 95 }[kind];
        ph += 2 * Math.PI * (f + 40 * Math.exp(-t * 25)) / SR; v = 0.8 * Math.sin(ph) * Math.exp(-t * 8); break;
      }
      case 'crash': case 'ride': { const n = rnd(); v = (n - lp) * (kind === 'crash' ? 0.4 : 0.25) * Math.exp(-t * (kind === 'crash' ? 2.6 : 6)); lp = n * 0.3 + lp * 0.7;
        if (kind === 'ride') { ph += 2 * Math.PI * 4300 / SR; v += 0.12 * Math.sin(ph) * Math.exp(-t * 9); } break; }
      case 'shot': ph += 2 * Math.PI * 880 / SR; v = 0.3 * Math.sin(ph) * Math.exp(-t * 20); break;
    }
    o[i] = v;
  }
  return (voices[kind] = o);
}
const LANE_VOICE = { kick: 'kick', snare: 'snare', snareAlt: 'rim', closedHat: 'hat', openHat: 'ohat', tomHigh: 'tomHigh',
  tomMid: 'tomMid', tomFloor: 'tomFloor', crash: 'crash', oneShotA: 'shot', oneShotB: 'shot', oneShotC: 'shot' };
const SPAN = { t4: 4, t8: 2, t16: 1 };

/** Hit times (in steps, 0..16) for a pattern, applying BMY swing and triplet words. */
export function patternEvents(pattern, swing) {
  const depth = swing && swing.type !== 'Off' ? swing.depth / 100 : 0;
  const swingStep = s => {
    if (!depth) return s;
    if (swing.type === '1-8' && s % 4 === 2) return s + depth * 2 / 3;
    if (swing.type === '1-16' && s % 2 === 1) return s + depth / 3;
    return s;
  };
  const ev = [];
  for (const c of pattern.cells) {
    const lane = c.lane === 'closedHat' && pattern.hatUseAlt ? 'ride' : LANE_VOICE[c.lane];
    if (c.ornament && SPAN[c.ornament]) {
      const span = SPAN[c.ornament], bits = c.tripletPattern || '111';
      const vels = [c.velocity, c.note2Velocity || c.velocity, c.note3Velocity || c.velocity];
      for (let k = 0; k < 3; k++) if (bits[k] === '1') ev.push({ voice: lane, step: c.step + k * span / 3, vel: vels[k] });
    } else ev.push({ voice: lane, step: swingStep(c.step), vel: c.velocity });
  }
  return ev;
}

/** One bar as an AudioBuffer (stereo) at bpm. */
export function renderBar(ctx, pattern, swing, bpm) {
  const stepSec = 60 / bpm / 4, len = Math.round(16 * stepSec * SR);
  const buf = ctx.createBuffer(2, len, SR);
  const L = buf.getChannelData(0), R = buf.getChannelData(1);
  for (const e of patternEvents(pattern, swing)) {
    const v = voice(e.voice), g = (VEL_MIDI[e.vel] || 82) / 127;
    const s0 = Math.round(e.step * stepSec * SR);
    for (let i = 0; i < v.length; i++) { const j = (s0 + i) % len; L[j] += v[i] * g; R[j] += v[i] * g; }
  }
  return buf;
}

/**
 * Region [a, b) of the song as an AudioBuffer: each item {start, stepSec, pattern} is a bar whose
 * pattern plays at its place in the song. Used to hear patterns in parallel with the song and stem.
 */
export function renderRegion(ctx, a, b, items, swing) {
  const len = Math.max(1, Math.round((b - a) * SR));
  const buf = ctx.createBuffer(2, len, SR);
  const L = buf.getChannelData(0), R = buf.getChannelData(1);
  for (const it of items) for (const e of patternEvents(it.pattern, swing)) {
    const t = it.start + e.step * it.stepSec - a;
    if (t < -0.01 || t >= b - a) continue;
    const v = voice(e.voice), g = (VEL_MIDI[e.vel] || 82) / 127, s0 = Math.round(t * SR);
    for (let i = 0; i < v.length; i++) { const j = s0 + i; if (j >= 0 && j < len) { L[j] += v[i] * g; R[j] += v[i] * g; } }
  }
  return buf;
}
