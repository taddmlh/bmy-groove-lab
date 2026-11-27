// ADTOF hits + drum-stem spectrogram -> BMY-style one-bar patterns.
// Lanes and labels follow BMY_Kit.json / BMY_Velocity.json (see claude/BMY_GrooveExtractor_BankFormat.md).
import { NBANDS, BAND_FREQ, FPS } from './dsp.js?b=20261127k';

export const LANES = [   // BMY grid order, top to bottom
  { role: 'crash', name: 'Crash', color: '#d9b326' },
  { role: 'openHat', name: 'Open Hat', color: '#ec8a2c' },
  { role: 'closedHat', name: 'Closed Hat', color: '#ec8a2c' },
  { role: 'snare', name: 'Snare', color: '#3373e6' },
  { role: 'snareAlt', name: 'Rimshot', color: '#59a6f2' },
  { role: 'kick', name: 'Kick', color: '#a61a1a' },
  { role: 'tomHigh', name: 'Hi Tom', color: '#994dbf' },
  { role: 'tomMid', name: 'Mid Tom', color: '#994dbf' },
  { role: 'tomFloor', name: 'Floor Tom', color: '#994dbf' },
  { role: 'oneShotA', name: 'One-Shot A', color: '#33b34d' },
  { role: 'oneShotB', name: 'One-Shot B', color: '#33b34d' },
  { role: 'oneShotC', name: 'One-Shot C', color: '#33b34d' }
];
export const VEL = ['gh', 'sf', 'n1', 'n2', 'a1', 'a2'];
export const VEL_MIDI = { gh: 28, sf: 46, n1: 66, n2: 82, a1: 102, a2: 116 };

// ---- spectrogram helpers -----------------------------------------------------------------
const bandsIn = (lo, hi) => { const r = []; for (let b = 0; b < NBANDS; b++) if (BAND_FREQ[b] >= lo && BAND_FREQ[b] < hi) r.push(b); return r; };
const B = {
  kick: bandsIn(30, 150), snare: bandsIn(150, 5000), tom: bandsIn(60, 600),
  hihat: bandsIn(5000, 16000), cymbal: bandsIn(3000, 16000), toneRange: bandsIn(60, 500), perc: bandsIn(200, 16000)
};

// ---- percussion -> one-shot lanes -----------------------------------------------------------
// BMY's model hears five percussion types. The busiest three in the marked bars go to One-shot A/B/C, each tagged
// with a GM note (training/bmy_transcriber.py PERC_GM) so BMY can play a matching sound. Matt, 2026-11-27.
export const PERC = {
  shaker: { gm: 82, name: 'Shaker' }, tambourine: { gm: 54, name: 'Tambourine' }, bell: { gm: 56, name: 'Cowbell' },
  handDrum: { gm: 63, name: 'Conga' }, wood: { gm: 76, name: 'Woodblock' }
};
const SHOTS = ['oneShotA', 'oneShotB', 'oneShotC'];
// Hand drums and woodblocks come as high/low pairs in GM. A part that clearly uses two pitches is split into two
// units (two one-shot lanes), named by pitch: bongos above ~400 Hz, congas below.
const PITCHED = { handDrum: bandsIn(80, 1000), wood: bandsIn(300, 3000) };
function pairNames(perc, hiPitch) {
  if (perc === 'wood') return { high: { gm: 76, name: 'Woodblock hi' }, low: { gm: 77, name: 'Woodblock lo' }, all: { gm: 76, name: 'Woodblock' } };
  return hiPitch > 400
    ? { high: { gm: 60, name: 'Bongo hi' }, low: { gm: 61, name: 'Bongo lo' }, all: { gm: 60, name: 'Bongo' } }
    : { high: { gm: 63, name: 'Conga hi' }, low: { gm: 64, name: 'Conga lo' }, all: { gm: 63, name: 'Conga' } };
}
/** Units of percussion heard in these bars: [{key, perc, gm, name, count}], busiest first; sets h.percUnit. */
function percUnits(hits, bars) {
  const inBars = h => bars.some(b => h.time >= b.start - b.step / 4 && h.time < b.end - b.step / 4);
  const units = {};
  const add = (key, info, h) => { (units[key] = units[key] || { key, ...info, count: 0 }).count += inBars(h) ? 1 : 0; h.percUnit = key; };
  for (const perc of Object.keys(PERC)) {
    const hs = hits.filter(h => h.cls === 'perc' && h.perc === perc);
    if (!hs.length) continue;
    if (!PITCHED[perc] || hs.some(h => !h.pitch)) { hs.forEach(h => add(perc, { perc, ...PERC[perc] }, h)); continue; }
    const ps = hs.map(h => h.pitch).sort((a, b) => a - b);
    let cut = -1, r = 1.12;                                  // biggest pitch gap (> 2 semitones), 15 %+ of hits each side
    for (let i = 1; i < ps.length; i++) if (ps[i] / ps[i - 1] > r && i >= 0.15 * ps.length && ps.length - i >= 0.15 * ps.length) { r = ps[i] / ps[i - 1]; cut = i; }
    if (cut < 0) { const nm = pairNames(perc, ps[ps.length >> 1]).all; hs.forEach(h => add(perc, { perc, ...nm }, h)); continue; }
    const split = (ps[cut - 1] + ps[cut]) / 2, nm = pairNames(perc, ps[Math.floor((cut + ps.length) / 2)]);
    hs.forEach(h => { const part = h.pitch >= split ? 'high' : 'low'; add(`${perc}:${part}`, { perc, ...nm[part] }, h); });
  }
  const order = Object.keys(PERC);
  return Object.values(units).filter(u => u.count).sort((a, b) => b.count - a.count || order.indexOf(a.perc) - order.indexOf(b.perc));
}
/** -> { oneShotA: {perc, gm, name, count, key}, ... }: the three busiest percussion units in these bars. */
export function percAllocation(hits, bars) {
  return Object.fromEntries(percUnits(hits, bars).slice(0, 3).map((u, i) => [SHOTS[i], u]));
}
const lin = v => Math.pow(10, v) - 1;
function energy(spec, frames, f, bands) {
  if (f < 0 || f >= frames) return 0;
  let s = 0; for (const b of bands) s += lin(spec[f * NBANDS + b]); return s / bands.length;
}
function peakEnergy(spec, frames, f, bands) { let m = 0; for (let k = 0; k <= 2; k++) m = Math.max(m, energy(spec, frames, f + k, bands)); return m; }
function meanEnergy(spec, frames, f0, f1, bands) { let s = 0, n = 0; for (let f = f0; f <= f1; f++) { s += energy(spec, frames, f, bands); n++; } return s / n; }

/** Adds features to each hit: level (dB), decay ratios, tom pitch. specFrame0 = song frame of spec[0]. */
export function describeHits(hits, spec, frames, specFrame0) {
  const hats = hits.filter(x => x.cls === 'hihat').sort((a, b) => a.time - b.time);
  for (const h of hits) {
    const f = Math.round(h.time * FPS) - specFrame0;
    const bands = B[h.cls];
    const pk = peakEnergy(spec, frames, f, bands) + 1e-9;
    h.levelDb = 20 * Math.log10(pk);
    if (h.cls === 'hihat') {
      // open or closed = how long it rings. Fast hats (e.g. triplets) are hit again before the usual
      // 120-200 ms window: then measure only up to the next hat, with a stricter bar (it's earlier in the ring).
      // stop at the next hat (listed, or a re-attack seen in the hat bands, e.g. a quiet hit the transcriber skipped)
      const next = hats.find(x => x.time > h.time + 0.02);
      let gapF = next ? Math.round((next.time - h.time) * FPS) : 999;
      for (let k = 4; k < Math.min(gapF, 22); k++) {
        const e0 = energy(spec, frames, f + k - 1, B.hihat), e1 = energy(spec, frames, f + k, B.hihat);
        if (e1 > 1.6 * e0 && e1 > 0.15 * pk) { gapF = k; break; }
      }
      if (gapF > 21) h.decay = meanEnergy(spec, frames, f + 12, f + 20, B.hihat) / pk;
      else if (gapF - 2 > 7) h.decay = meanEnergy(spec, frames, f + 6, f + gapF - 2, B.hihat) / pk * (0.2 / 0.4);
      else h.decay = 0;                                     // re-hit within 90 ms: a closed hat
      if (hits.some(x => x.cls === 'snare' && Math.abs(x.time - h.time) < 0.03)) h.decay *= 0.5;   // a snare's ring sits in the hat bands too
    }
    if (h.cls === 'cymbal') h.decay = meanEnergy(spec, frames, f + 30, f + 45, B.cymbal) / pk;
    if (h.cls === 'tom' || (h.cls === 'perc' && PITCHED[h.perc])) {   // strongest band shortly after the hit
      let best = 0, bv = -1;
      for (const b of h.cls === 'tom' ? B.toneRange : PITCHED[h.perc]) { const e = meanEnergy(spec, frames, f + 3, f + 10, [b]); if (e > bv) { bv = e; best = b; } }
      h.pitch = BAND_FREQ[best];
    }
  }
  return hits;
}

// ---- lane assignment ---------------------------------------------------------------------
function tomLanes(hits) {
  const toms = hits.filter(h => h.cls === 'tom');
  const ps = [...new Set(toms.map(h => Math.round(h.pitch)))].sort((a, b) => a - b);
  const map = new Map();
  if (ps.length === 0) return map;
  // split at the biggest pitch gaps (ratio > 1.12), max 3 groups
  const gaps = [];
  for (let i = 1; i < ps.length; i++) if (ps[i] / ps[i - 1] > 1.12) gaps.push({ i, r: ps[i] / ps[i - 1] });
  gaps.sort((a, b) => b.r - a.r);
  const cuts = gaps.slice(0, 2).map(g => g.i).sort((a, b) => a - b);
  const groups = []; let s = 0;
  for (const c of [...cuts, ps.length]) { groups.push(ps.slice(s, c)); s = c; }
  let roles;
  if (groups.length === 1) {
    const p = groups[0][Math.floor(groups[0].length / 2)];
    roles = [p < 110 ? 'tomFloor' : p < 170 ? 'tomMid' : 'tomHigh'];
  } else roles = groups.length === 2 ? ['tomFloor', 'tomHigh'] : ['tomFloor', 'tomMid', 'tomHigh'];
  groups.forEach((g, i) => g.forEach(p => map.set(p, roles[i])));
  return map;
}

export function assignLanes(hits, bars) {
  const tm = tomLanes(hits);
  const shots = percAllocation(hits, bars), shotOf = {};
  for (const [lane, x] of Object.entries(shots)) shotOf[x.key] = lane;
  for (const h of hits) {
    if (h.fixedLane) continue;                // Perform: the lane was chosen (lane mode) or taught (beatbox)
    if (h.cls === 'kick') h.lane = 'kick';
    else if (h.cls === 'snare') h.lane = 'snare';
    else if (h.cls === 'tom') h.lane = h.modelLane || tm.get(Math.round(h.pitch)) || 'tomMid';          // BMY's model names the tom
    else if (h.cls === 'hihat') h.lane = h.open !== undefined ? (h.open ? 'openHat' : 'closedHat') : h.decay > 0.2 ? 'openHat' : 'closedHat';
    else if (h.cls === 'cymbal') h.lane = h.kind || 'crash';      // Magenta tells ride from crash; ADTOF doesn't
    else if (h.cls === 'perc') h.lane = shotOf[h.percUnit] || null;   // beyond the three busiest: no lane left
  }
  // Ride: 3+ cymbal hits in a bar are a ride pattern, not crashes.
  for (const bar of bars) {
    const cy = hits.filter(h => h.cls === 'cymbal' && !h.kind && h.time >= bar.start - bar.step / 4 && h.time < bar.end - bar.step / 4);
    if (cy.length >= 3) cy.forEach(h => { if (!h.fixedLane) h.lane = 'ride'; });
  }
  // Velocity: loudness against the median of the same instrument across the marked bars.
  const groups = {};
  hits.forEach(h => { if (h.lane) (groups[h.lane] = groups[h.lane] || []).push(h.levelDb); });
  const med = {};
  for (const [k, v] of Object.entries(groups)) { const s = [...v].sort((a, b) => a - b); med[k] = s[Math.floor(s.length / 2)]; }
  for (const h of hits) {
    if (!h.lane) continue;
    const d = h.levelDb - med[h.lane];
    h.vel = d > 4 ? 'a2' : d > 1.5 ? 'a1' : d > -2 ? 'n2' : d > -5 ? 'n1' : d > -10 ? 'sf' : 'gh';
  }
  return hits;
}

// ---- swing ---------------------------------------------------------------------------------
function gridFor(sw) {   // positions within a beat (in steps) for straight/8th/16th swing
  if (sw.type === '1-8') return [0, 1, 2 + (2 / 3) * sw.depth, 3];
  if (sw.type === '1-16') return [0, 1 + sw.depth / 3, 2, 3 + sw.depth / 3];
  return [0, 1, 2, 3];
}
const nearest = (q, grid) => { let bi = 0, bd = Infinity; grid.forEach((g, i) => { const d = Math.abs(q - g); if (d < bd) { bd = d; bi = i; } }); const d4 = Math.abs(q - 4); return d4 < bd ? { i: 4, d: d4 } : { i: bi, d: bd }; };

/** One swing setting for the whole bank (songs don't change swing between bars). */
export function detectSwing(hits, bars) {
  const qs = [];
  for (const bar of bars) for (const h of hits) {
    const p = (h.time - bar.start) / bar.step;
    if (p < -0.25 || p >= 15.75) continue;
    qs.push(((p % 4) + 4) % 4 > 3.5 ? ((p % 4) + 4) % 4 - 4 : ((p % 4) + 4) % 4);
  }
  if (qs.length < 4) return { type: 'Off', depth: 0 };
  const err = sw => qs.reduce((s, q) => s + nearest(q, gridFor(sw)).d, 0) / qs.length;
  let best = { sw: { type: 'Off', depth: 0 }, e: err({ type: 'Off', depth: 0 }) };
  for (const type of ['1-8', '1-16']) for (let d = 0.1; d <= 1.0001; d += 0.05) {
    const sw = { type, depth: d };
    const used = qs.filter(q => Math.abs(q - gridFor(sw)[type === '1-8' ? 2 : 1]) < 0.15 || Math.abs(q - gridFor(sw)[3]) < 0.15).length;
    const e = err(sw) + 0.015;                       // prefer straight unless swing clearly fits better
    if (used >= 2 && e < best.e * 0.8) best = { sw, e };
  }
  return { type: best.sw.type, depth: Math.round(best.sw.depth * 100) };
}

// ---- one bar -> cells ---------------------------------------------------------------------
const TRIP4 = [0, 4 / 3, 8 / 3], TRIP8 = [0, 2 / 3, 4 / 3];
const SPANS = { t4: 4, t8: 2, t16: 1, double: 1, flam: 1 };

/** bar: {start, end, step}. A hit belongs to this bar from 1/4 step before its downbeat to 1/4 step before the next
 * (so a 32nd stroke at 15.5 stays here, an early downbeat at 15.8 goes to the next bar). Returns {cells, hatUseAlt, notes:[]}. */
export function barToPattern(hits, bar, swing) {
  const sw = { type: swing.type, depth: swing.depth / 100 };
  const grid = gridFor(sw);
  const inBar = hits.filter(h => h.time >= bar.start - bar.step / 4 && h.time < bar.end - bar.step / 4)
    .map(h => ({ ...h, p: (h.time - bar.start) / bar.step }));
  const notes = [];
  const rideBar = inBar.filter(h => h.lane === 'ride').length >= 3;
  if (rideBar) {
    const hats = inBar.filter(h => h.lane === 'closedHat').length;
    if (hats) notes.push(`${hats} closed-hat hit(s) dropped: this bar plays ride on the hat lane`);
  }
  const lanes = {};
  for (const h of inBar) {
    if (!h.lane) continue;                                     // percussion beyond the three one-shot lanes
    let lane = h.lane === 'ride' ? 'closedHat' : h.lane;
    if (rideBar && h.lane === 'closedHat') continue;
    (lanes[lane] = lanes[lane] || []).push(h);
  }
  const cells = [];
  const velRank = v => VEL.indexOf(v);
  for (const [lane, hs] of Object.entries(lanes)) {
    const byBeat = [[], [], [], []];
    for (const h of hs) { const beat = Math.min(3, Math.max(0, Math.floor((h.p + 0.25) / 4))); byBeat[beat].push(h); }
    const taken = new Map();   // step -> cell
    const put = c => { const old = taken.get(c.step); if (!old || velRank(c.velocity) > velRank(old.velocity)) taken.set(c.step, c); };
    byBeat.forEach((bh, beat) => {
      if (!bh.length) return;
      const qs = bh.map(h => ({ h, q: h.p - beat * 4 }));
      const unexplained = qs.filter(x => nearest(x.q, grid).d > 0.2);
      // quarter-span triplet (3 over the beat)
      const d4 = x => Math.min(...TRIP4.map(t => Math.abs(x.q - t)));
      // a triplet needs 3 notes, or 2 notes sitting almost exactly on triplet positions (else it's more likely a double)
      if (unexplained.length && (qs.length >= 3 ? qs.every(x => d4(x) < 0.18) : qs.length === 2 && qs.every(x => d4(x) < 0.08))) {
        const slot = [null, null, null];
        qs.forEach(x => { const i = TRIP4.reduce((bi, t, k) => (Math.abs(x.q - t) < Math.abs(x.q - TRIP4[bi]) ? k : bi), 0); if (!slot[i] || velRank(x.h.vel) > velRank(slot[i].vel)) slot[i] = x.h; });
        cells.push(tripCell(lane, beat * 4, 't4', slot));
        return;
      }
      // eighth-span triplets (3 over each half-beat)
      for (const half of [0, 2]) {
        const hq = qs.filter(x => x.q >= half - 0.25 && x.q < half + 1.75);
        const un = hq.filter(x => nearest(x.q, grid).d > 0.2);
        const d8 = x => Math.min(...TRIP8.map(t => Math.abs(x.q - half - t)));
        if (un.length && (hq.length >= 3 ? hq.every(x => d8(x) < 0.15) : hq.length === 2 && hq.every(x => d8(x) < 0.06))) {
          const slot = [null, null, null];
          hq.forEach(x => { const i = TRIP8.reduce((bi, t, k) => (Math.abs(x.q - half - t) < Math.abs(x.q - half - TRIP8[bi]) ? k : bi), 0); if (!slot[i] || velRank(x.h.vel) > velRank(slot[i].vel)) slot[i] = x.h; });
          cells.push(tripCell(lane, beat * 4 + half, 't8', slot));
          hq.forEach(x => { x.used = true; });
        }
      }
      // ruffs: three strokes inside about one 16th (each 0.2-0.45 step apart) -> t16 on the first stroke's step
      const rest = qs.filter(x => !x.used).sort((x, y) => x.q - y.q);
      for (let k = 0; k + 2 < rest.length; k++) {
        const [x, y, z] = [rest[k], rest[k + 1], rest[k + 2]];
        if (x.used || y.used || z.used) continue;
        const g1 = y.q - x.q, g2 = z.q - y.q;
        if (g1 >= 0.2 && g1 <= 0.45 && g2 >= 0.2 && g2 <= 0.45 && z.q - x.q <= 0.85) {
          const step = Math.min(15, beat * 4 + nearest(x.q, grid).i);
          cells.push({ lane, step, velocity: x.h.vel, ornament: 't16', tripletPattern: '111', note2Velocity: y.h.vel, note3Velocity: z.h.vel });
          x.used = y.used = z.used = true;
          for (const [st, c] of [...taken]) if (st === step) taken.delete(st);
        }
      }
      // doubles (two strokes a 32nd apart) and flams (soft grace 8-45 ms before the main stroke)
      for (let k = 0; k + 1 < rest.length; k++) {
        const x = rest[k], y = rest[k + 1];
        if (x.used || y.used) continue;
        const d = y.q - x.q, ms = d * bar.step * 1000;
        if (ms >= 8 && ms <= 45 && (velRank(x.h.vel) < velRank(y.h.vel) || (x.h.levelDb || 0) < (y.h.levelDb || 0) - 3)) {
          y.orn = 'flam'; x.used = true;
        } else if (d >= 0.3 && d <= 0.7 && (nearest(y.q, grid).d > 0.2 || nearest(x.q, grid).i === nearest(y.q, grid).i)) {
          x.orn = 'double'; x.vel2 = y.h.vel; y.used = true;
        }
      }
      for (const x of qs) {
        if (x.used) continue;
        const n = nearest(x.q, grid);
        const step = Math.min(15, beat * 4 + n.i);   // a lone hit in 15.5-15.75 stays on 15 (the next bar starts at 15.75)
        if (step < 0) continue;
        if (cells.some(c => c.lane === lane && c.ornament && SPANS[c.ornament] > 1 && step >= c.step && step < c.step + SPANS[c.ornament])) continue;
        const c = { lane, step, velocity: x.h.vel, tripletPattern: '111' };
        if (x.orn) c.ornament = x.orn;
        put(c);
      }
    });
    cells.push(...taken.values());
  }
  // drop straight cells covered by a triplet in the same lane
  const multi = c => c.ornament === 't4' || c.ornament === 't8' || c.ornament === 't16';
  const out = cells.filter(c => multi(c) || !cells.some(t => multi(t) && t.lane === c.lane && c.step >= t.step && c.step < t.step + SPANS[t.ornament]));
  out.sort((a, b) => a.step - b.step || LANES.findIndex(l => l.role === a.lane) - LANES.findIndex(l => l.role === b.lane));
  return { cells: out, hatUseAlt: rideBar ? 1 : 0, notes };
}

function tripCell(lane, step, orn, slot) {
  const present = slot.map(s => (s ? '1' : '0')).join('');
  if (present === '100') return { lane, step, velocity: slot[0].vel, tripletPattern: '111' };   // just a straight hit
  const loud = slot.filter(Boolean).reduce((a, b) => (VEL.indexOf(b.vel) > VEL.indexOf(a.vel) ? b : a));
  const c = { lane, step, velocity: (slot[0] || loud).vel, ornament: orn, tripletPattern: present };
  if (slot[1]) c.note2Velocity = slot[1].vel;
  if (slot[2]) c.note3Velocity = slot[2].vel;
  return c;
}
