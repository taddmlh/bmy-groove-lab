// Tiny drum synth for auditioning a BMY-style pattern (not a kit: just enough to check timing).
import { VEL_MIDI } from './transcribe.js?b=20261127e';
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
    const kitLane = c.lane === 'closedHat' && pattern.hatUseAlt ? 'ride' : c.lane;
    const lane = c.lane === 'closedHat' && pattern.hatUseAlt ? 'ride' : LANE_VOICE[c.lane];
    if (c.ornament && SPAN[c.ornament]) {
      const span = SPAN[c.ornament], bits = c.tripletPattern || '111';
      const vels = [c.velocity, c.note2Velocity || c.velocity, c.note3Velocity || c.velocity];
      for (let k = 0; k < 3; k++) if (bits[k] === '1') ev.push({ voice: lane, lane: kitLane, step: c.step + k * span / 3, vel: vels[k] });
    } else if (c.ornament === 'double') {        // BMY: main x0.85, second stroke half a step later x0.95
      ev.push({ voice: lane, lane: kitLane, step: swingStep(c.step), vel: c.velocity, scale: 0.85 });
      ev.push({ voice: lane, lane: kitLane, step: swingStep(c.step) + 0.5, vel: c.velocity, scale: 0.95 });
    } else if (c.ornament === 'flam') {          // BMY: grace 22 ms before, x0.33
      ev.push({ voice: lane, lane: kitLane, step: swingStep(c.step), vel: c.velocity });
      ev.push({ voice: lane, lane: kitLane, step: swingStep(c.step), ms: -22, vel: c.velocity, scale: 0.33 });
    } else ev.push({ voice: lane, lane: kitLane, step: swingStep(c.step), vel: c.velocity });
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

// ---- BMY's built-in kits: GeneralUser GS drum programs, rendered per BMY velocity label (kit/[program/]kit.json)
const VELS = ['gh', 'sf', 'n1', 'n2', 'a1', 'a2'];
const KIT_VERSION = 5;   // bump when kits are re-rendered, so browsers fetch the new samples
/** The kits BMY offers (BMY_SoundFontKit.swift gsKits). Program 0 lives in kit/, the rest in kit/<program>/. */
export const KITS = [[0, 'Standard 1'], [1, 'Standard 2'], [2, 'Standard 3'], [8, 'Room'], [16, 'Power'], [24, 'Electronic'],
  [25, '808/909'], [26, 'Dance'], [32, 'Jazz'], [40, 'Brush'], [48, 'Orchestral'], [56, 'SFX']];
const loaded = new Map();     // program -> kit
let KIT = null, STD = null;   // current kit; Standard 1 (fills lanes a kit doesn't have)
async function fetchKit(ctx, program) {
  if (loaded.has(program)) return loaded.get(program);
  const base = program ? `kit/${program}/` : 'kit/';
  const man = await (await fetch(base + 'kit.json?v=' + KIT_VERSION)).json();
  const lanes = {};
  await Promise.all(Object.entries(man.notes).map(async ([lane, n]) => {
    const buf = await ctx.decodeAudioData(await (await fetch(base + n.file + '?v=' + KIT_VERSION)).arrayBuffer());
    const data = buf.getChannelData(0), k = buf.sampleRate / man.sr;
    lanes[lane] = { excl: n.exclusiveClass, layers: n.layers.map(([o, l]) => data.subarray(Math.round(o * k), Math.round((o + l) * k))) };
  }));
  const kit = { program, name: man.kit, lanes };
  loaded.set(program, kit);
  return kit;
}
/** Load (or switch to) a kit. Standard 1 is always loaded too, for lanes a kit lacks (e.g. SFX has no kick). */
export async function loadKit(ctx, program = 0) {
  STD = await fetchKit(ctx, 0);
  KIT = await fetchKit(ctx, +program || 0);
  return KIT;
}
export const kitProgram = () => (KIT ? KIT.program : 0);
export const kitLoaded = () => !!KIT;
/** Play one kit sound now (lane icon tap). */
export function audition(ctx, lane, vel = 'n2') {
  const snd = sound({ lane, voice: LANE_VOICE[lane] || 'shot', vel });
  const b = ctx.createBuffer(1, snd.data.length, SR); b.getChannelData(0).set(snd.data);
  const n = ctx.createBufferSource(); n.buffer = b;
  const g = ctx.createGain(); g.gain.value = snd.gain; n.connect(g); g.connect(ctx.destination); n.start();
}
const KIT_LANE = { closedHat: 'closedHat', openHat: 'openHat', snare: 'snare', snareAlt: 'snareAlt', kick: 'kick', crash: 'crash',
  tomHigh: 'tomHigh', tomMid: 'tomMid', tomFloor: 'tomFloor', oneShotA: 'oneShotA', oneShotB: 'oneShotB', oneShotC: 'oneShotC', ride: 'ride' };

/** Sample + choke class for one event (BMY kit if loaded, else the tiny synth). */
function sound(e) {
  let vel = e.vel, trim = 1;
  if (e.scale && e.scale !== 1) {              // scaled stroke: nearest BMY layer, then trim the level
    const v = (VEL_MIDI[e.vel] || 82) * e.scale;
    vel = VELS.reduce((b, l) => (Math.abs(VEL_MIDI[l] - v) < Math.abs(VEL_MIDI[b] - v) ? l : b), 'gh');
    trim = Math.min(1.5, Math.pow(v / VEL_MIDI[vel], 2));
  }
  if (KIT) {
    const k = KIT.lanes[KIT_LANE[e.lane]] || (STD && STD.lanes[KIT_LANE[e.lane]]);
    if (k) { const li = VELS.indexOf(vel); return { data: k.layers[li < 0 ? 3 : li], gain: trim, excl: k.excl }; }
  }
  return { data: voice(e.voice), gain: (VEL_MIDI[vel] || 82) / 127 * trim, excl: e.voice === 'hat' || e.voice === 'ohat' ? 1 : 0 };
}

/**
 * Region [a, b) of the song as an AudioBuffer: each item {start, stepSec, pattern} is a bar whose
 * pattern plays at its place in the song. Used to hear patterns in parallel with the song and stem.
 */
export function renderRegion(ctx, a, b, items, swing) {
  const len = Math.max(1, Math.round((b - a) * SR));
  const buf = ctx.createBuffer(2, len, SR);
  const L = buf.getChannelData(0), R = buf.getChannelData(1);
  const evs = [];
  for (const it of items) for (const e of patternEvents(it.pattern, swing)) {
    const t = it.start + e.step * it.stepSec + (e.ms || 0) / 1000 - a;
    if (t < -0.01 || t >= b - a) continue;
    evs.push({ s0: Math.round(t * SR), ...sound(e) });
  }
  evs.sort((x, y) => x.s0 - y.s0);
  const FADE = 300;                                          // ~7 ms choke fade
  evs.forEach((e, k) => {
    let n = e.data.length;
    if (e.excl) { const nx = evs.slice(k + 1).find(o => o.excl === e.excl); if (nx) n = Math.min(n, nx.s0 - e.s0 + FADE); }
    for (let i = 0; i < n; i++) {
      const j = e.s0 + i; if (j < 0) continue; if (j >= len) break;
      const f = e.excl && n < e.data.length && i > n - FADE ? (n - i) / FADE : 1;
      const v = e.data[i] * e.gain * f; L[j] += v; R[j] += v;
    }
  });
  return buf;
}
