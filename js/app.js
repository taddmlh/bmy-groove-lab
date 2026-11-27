// BMY Groove Lab: song -> drum stem (Demucs) -> hits (ADTOF) -> BMY patterns -> .bmybank
import { logSpec, toMono, FPS, NBANDS, BAND_FREQ } from './dsp.js?b=20261127f';
import { loadAdtof, activations, hitsFrom, CLASSES } from './adtof.js?b=20261127f';
import { detectGrid, barTimes } from './beats.js?b=20261127f';
import { LANES, VEL, describeHits, assignLanes, detectSwing, barToPattern } from './transcribe.js?b=20261127f';
import { buildBank, bankFileName } from './bank.js?b=20261127f';
import { detectOnsets, centroid, classify, tempoFromOnsets, clickBuffers, looksLikeClick } from './perform.js?b=20261127f';
import { oafHits, loadOaf } from './oaf.js?b=20261127f';
import { bmytHits, loadBmyt, bmytAvailable, BMYT_V } from './bmyt.js?b=20261127f';
import { denoise } from './denoise.js?b=20261127f';
import { loadDemucs, separateDrums, SEPARATOR_VERSION } from './demucs.js?b=20261127f';
import { renderRegion, loadKit, audition, KITS } from './synth.js?b=20261127f';
import { putSong, getSong, putFile, getFile, hasFile, putStems, getStems, listSongs, deleteSong, askPersist } from './db.js?b=20261127f';

const ORT_VER = '1.31.1';
const SR = 44100;
const MAX_MARKS = 6;
const $ = id => document.getElementById(id);

// ---------------------------------------------------------------- log (survives reloads)
let prevLog = ''; try { prevLog = localStorage.getItem('bmyGLLog') || ''; } catch (e) {}
let curLog = '';
const log = m => {
  curLog += `${new Date().toLocaleTimeString()}  ${m}\n`;
  try { localStorage.setItem('bmyGLLog', curLog.slice(-20000)); } catch (e) {}
  $('log').textContent = (prevLog ? '— PREVIOUS PAGE LOAD —\n' + prevLog + '\n— THIS PAGE LOAD —\n' : '') + curLog;
};
const errText = e => { const m = (e && (e.message || String(e))) || 'unknown'; const st = e && e.stack && !e.stack.includes(m) ? '\n' + e.stack : ''; return m + st; };
const crumb = m => { try { localStorage.setItem('bmyGLStage', m); } catch (e) {} };
const crumbDone = () => { try { localStorage.removeItem('bmyGLStage'); } catch (e) {} };
let lastCrash = null; try { lastCrash = localStorage.getItem('bmyGLStage'); if (lastCrash && /^separating/.test(lastCrash)) localStorage.setItem('bmyGLSepFail', '1'); } catch (e) {}
const sepFailed = () => { try { return !!localStorage.getItem('bmyGLSepFail'); } catch (e) { return false; } };
window.addEventListener('error', e => log('PAGE ERROR: ' + (e.message || e.error)));
window.addEventListener('unhandledrejection', e => log('UNHANDLED: ' + (e.reason && (e.reason.stack || e.reason.message) || e.reason)));
const status = (m, frac) => { $('status').textContent = m; if (frac !== undefined) $('prog').style.width = Math.round(frac * 100) + '%'; };
const yieldUI = () => new Promise(r => setTimeout(r, 0));

// ---------------------------------------------------------------- state
const S = {
  name: '', dur: 0, mixBuf: null, mixMono: null, drumBuf: null, drumMono: null, extracted: [],
  mixSpec: null, grid: null, bars: [], marks: [], view: { start: 0, pps: 60 }, sel: null, playhead: 0,
  hits: [], swing: { type: 'Off', depth: 0 }, patterns: new Map(), tab: null, undo: [], src: 'mix', loop: true
};
let ort = null, actx = null;
const TEST_NO_DEMUCS = new URLSearchParams(location.search).has('nodemucs');   // test hook: audio is already drums only
if (TEST_NO_DEMUCS) window.GL = { S, get P() { return P; }, saveNow: () => saveNow(), gridChanged: n => gridChanged(n), get loop() { return nodes.mix && [nodes.mix.loopStart, nodes.mix.loopEnd]; }, get nodes() { return Object.keys(nodes); }, get gains() { return Object.fromEntries(Object.entries(gains).map(([k, g]) => [k, +g.gain.value.toFixed(2)])); }, get playing() { return playing; } };

async function getOrt() {
  if (ort) return ort;
  ort = await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
  ort.env.wasm.numThreads = 1;
  return ort;
}
function audioCtx() {
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  if (actx.state === 'suspended') actx.resume();
  return actx;
}

// ---------------------------------------------------------------- bars
const barSec = () => 240 / S.grid.bpm;
function rebuildBars() {
  const bt = barTimes(S.grid, S.dur);
  S.bars = bt.map((t, i) => ({ i, start: t, end: t + barSec(), step: barSec() / 16 }));
  S.marks = S.marks.filter(i => i < S.bars.length);
}
const barAt = t => S.bars.findIndex(b => t >= b.start && t < b.end);
const barNo = i => { const first = S.bars.findIndex(b => b.start >= -0.1); return i - first + 1; };   // first bar line in the song = bar 1 (a pickup is bar 0)

// ---------------------------------------------------------------- waveforms
const waveW = () => $('wMix').getBoundingClientRect().width || Math.max(300, $('detail').getBoundingClientRect().width || window.innerWidth - 16);
function canvasSize(c) {
  const dpr = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return { w, h, dpr };
}
const tToX = (t, w, dpr) => (t - S.view.start) * S.view.pps * dpr;
const xToT = (xCss) => S.view.start + xCss / S.view.pps;

function drawWave(c, mono, color, extracted) {
  const { w, h, dpr } = canvasSize(c), g = c.getContext('2d');
  g.clearRect(0, 0, w, h);
  if (!S.mixBuf) return;
  const viewEnd = S.view.start + w / dpr / S.view.pps;
  drawBarShading(g, w, h, dpr, viewEnd);
  if (extracted !== undefined && extracted.length === 0) {
    g.fillStyle = '#5b6070'; g.font = `${12 * dpr}px -apple-system,sans-serif`; g.textAlign = 'center';
    g.fillText('Mark bars, then Extract drums', w / 2, h / 2 + 4 * dpr);
  }
  if (mono) {
    g.fillStyle = color;
    const spp = SR / (S.view.pps * dpr);
    for (let x = 0; x < w; x++) {
      const t = S.view.start + x / dpr / S.view.pps;
      if (extracted && !extracted.some(([a, b]) => t >= a && t < b)) continue;
      const s0 = Math.floor(t * SR), s1 = Math.min(mono.length, Math.floor(s0 + Math.max(1, spp)));
      if (s0 < 0 || s0 >= mono.length) continue;
      let mn = 1, mx = -1;
      const stride = Math.max(1, Math.floor((s1 - s0) / 64));
      for (let s = s0; s < s1; s += stride) { const v = mono[s]; if (v < mn) mn = v; if (v > mx) mx = v; }
      const y0 = h / 2 - mx * h * 0.48, y1 = h / 2 - mn * h * 0.48;
      g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
    }
  }
  drawOverlays(g, w, h, dpr);
  if (c.id === 'wDrums' && S.sel && !S.locked && selIsBars()) drawShiftGrip(g, w, h, dpr);
}
// bottom handle on the drums lane: drag to move the whole bar (start and end together, tempo unchanged)
const SHIFT_H = 18;
function drawShiftGrip(g, w, h, dpr) {
  const x0 = tToX(S.sel.a, w, dpr), x1 = tToX(S.sel.b, w, dpr), y = h - (SHIFT_H - 2) * dpr, hh = (SHIFT_H - 5) * dpr;
  g.fillStyle = 'rgba(255,173,20,.9)';
  if (g.roundRect) { g.beginPath(); g.roundRect(x0, y, x1 - x0, hh, 4 * dpr); g.fill(); } else g.fillRect(x0, y, x1 - x0, hh);
  g.fillStyle = '#000'; const cx = (x0 + x1) / 2, cy = y + hh / 2;
  for (const dx of [-6, -2, 2, 6]) g.fillRect(cx + dx * dpr, y + 3 * dpr, dpr, hh - 6 * dpr);
  for (const [x, d] of [[cx - 16 * dpr, -1], [cx + 16 * dpr, 1]]) { g.beginPath(); g.moveTo(x + d * 5 * dpr, cy); g.lineTo(x, cy - 4 * dpr); g.lineTo(x, cy + 4 * dpr); g.fill(); }
}
function drawBarShading(g, w, h, dpr, viewEnd) {
  for (const b of S.bars) {
    if (b.end < S.view.start || b.start > viewEnd) continue;
    const x0 = tToX(b.start, w, dpr), x1 = tToX(b.end, w, dpr);
    if (S.marks.includes(b.i)) { g.fillStyle = 'rgba(38,199,255,.13)'; g.fillRect(x0, 0, x1 - x0, h); }
    g.fillStyle = '#3a3f4c'; g.fillRect(Math.round(x0), 0, Math.max(1, dpr), h);
    if (S.view.pps * barSec() / 4 > 14) {
      g.fillStyle = '#20232b';
      for (let k = 1; k < 4; k++) g.fillRect(Math.round(x0 + (x1 - x0) * k / 4), 0, Math.max(1, dpr * 0.5), h);
    }
  }
}
function drawOverlays(g, w, h, dpr, grips) {
  if (S.sel) {
    const x0 = tToX(S.sel.a, w, dpr), x1 = tToX(S.sel.b, w, dpr);
    g.fillStyle = 'rgba(255,173,20,.16)'; g.fillRect(x0, 0, x1 - x0, h);
    g.fillStyle = '#ffad14'; g.fillRect(x0, 0, (grips ? 2 : 1.5) * dpr, h); g.fillRect(x1 - (grips ? 2 : 1.5) * dpr, 0, (grips ? 2 : 1.5) * dpr, h);
    if (grips && !S.locked && selIsBars()) {
      const hw = 14 * dpr;
      g.fillRect(x0, 0, hw, h); g.fillRect(x1 - hw, 0, hw, h);
      g.fillStyle = '#000';
      for (const x of [x0 + 4 * dpr, x1 - 10 * dpr]) { g.fillRect(x, h * 0.3, dpr, h * 0.4); g.fillRect(x + 4 * dpr, h * 0.3, dpr, h * 0.4); }
    }
  }
  const px = tToX(S.playhead, w, dpr);
  g.fillStyle = '#ffffff'; g.fillRect(px, 0, Math.max(1, 1.5 * dpr), h);
}
function selIsBars() {
  if (!S.sel || !S.grid) return false;
  const a = barAt(S.sel.a + 0.001), b = barAt(S.sel.b - 0.001);
  return a >= 0 && b >= 0 && Math.abs(S.bars[a].start - S.sel.a) < 1e-6 && Math.abs(S.bars[b].end - S.sel.b) < 1e-6;
}
function drawRuler() {
  const c = $('ruler'), { w, h, dpr } = canvasSize(c), g = c.getContext('2d');
  g.clearRect(0, 0, w, h);
  if (!S.grid) return;
  const viewEnd = S.view.start + w / dpr / S.view.pps;
  g.font = `600 ${12 * dpr}px -apple-system,sans-serif`; g.textBaseline = 'middle';
  const every = Math.max(1, Math.ceil(36 / (barSec() * S.view.pps)));
  for (const b of S.bars) {
    if (b.end < S.view.start || b.start > viewEnd) continue;
    const x0 = tToX(b.start, w, dpr), x1 = tToX(b.end, w, dpr);
    const m = S.marks.indexOf(b.i);
    if (m >= 0) {
      g.fillStyle = b.i === S.tab ? '#1b6f8c' : '#0f3a4a'; g.fillRect(x0 + dpr, 2 * dpr, x1 - x0 - 2 * dpr, h - 4 * dpr);
      const gx = selIsBars() && Math.abs(S.sel.a - b.start) < 1e-6 ? 14 * dpr : 0;
      g.fillStyle = '#26c7ff'; g.beginPath(); g.arc(x0 + gx + 12 * dpr, h / 2, 8 * dpr, 0, 7); g.fill();
      g.fillStyle = '#000'; g.textAlign = 'center'; g.fillText(`${m + 1}`, x0 + gx + 12 * dpr, h / 2 + dpr); g.textAlign = 'left';
    }
    g.fillStyle = '#3a3f4c'; g.fillRect(Math.round(x0), 0, Math.max(1, dpr), h);
    if (barNo(b.i) % every === 0 || m >= 0) {
      g.fillStyle = m >= 0 ? '#e9f8ff' : '#8d929e';
      const label = `${barNo(b.i)}`;
      const off = (selIsBars() && Math.abs(S.sel.a - b.start) < 1e-6 ? 14 * dpr : 0) + (m >= 0 ? 25 * dpr : 5 * dpr);
      g.fillText(label, x0 + off, h / 2);
    }
  }
  drawOverlays(g, w, h, dpr, true);
}
function drawOverview() {
  const c = $('overview'), { w, h, dpr } = canvasSize(c), g = c.getContext('2d');
  g.clearRect(0, 0, w, h);
  if (!S.mixMono) return;
  if (!S.ovPeaks || S.ovPeaks.length !== w) {
    S.ovPeaks = new Float32Array(w);
    const spp = S.mixMono.length / w;
    for (let x = 0; x < w; x++) {
      let m = 0; const s0 = Math.floor(x * spp), s1 = Math.floor((x + 1) * spp);
      for (let s = s0; s < s1; s += 32) m = Math.max(m, Math.abs(S.mixMono[s]));
      S.ovPeaks[x] = m;
    }
  }
  g.fillStyle = '#4a5060';
  for (let x = 0; x < w; x++) { const a = S.ovPeaks[x] * h * 0.9; g.fillRect(x, (h - a) / 2, 1, Math.max(1, a)); }
  const k = w / S.dur;
  for (const i of S.marks) { const b = S.bars[i]; g.fillStyle = '#26c7ff'; g.fillRect(b.start * k, 0, Math.max(2, (b.end - b.start) * k), 4 * dpr); }
  const vw = canvasSize($('wMix')).w / (window.devicePixelRatio || 1) / S.view.pps;
  g.strokeStyle = '#ffad14'; g.lineWidth = 2 * dpr; g.strokeRect(S.view.start * k, 1, vw * k, h - 2);
  g.fillStyle = '#fff'; g.fillRect(S.playhead * k, 0, dpr, h);
}
function redraw() {
  drawOverview(); drawRuler();
  drawWave($('wMix'), S.mixMono, '#9aa3b5');
  drawWave($('wDrums'), S.drumMono, '#ec8a2c', S.extracted);
  $('selText').textContent = S.sel ? selLabel() : '';
}
function selLabel() {
  const a = barAt(S.sel.a + 0.001), b = barAt(S.sel.b - 0.001);
  const fmt = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`;
  return (a >= 0 && b >= 0 ? (a === b ? `Bar ${barNo(a)}` : `Bars ${barNo(a)}–${barNo(b)}`) + ' · ' : '') + `${fmt(S.sel.a)}–${fmt(S.sel.b)}`;
}
function clampView() {
  const vw = waveW() / S.view.pps;
  S.view.start = Math.max(-0.5, Math.min(S.view.start, S.dur - vw + 0.5));
}
window.addEventListener('resize', () => { S.ovPeaks = null; redraw(); renderGrid(); });

// ---------------------------------------------------------------- pointer handling (timeline)
const multi = { pts: new Map(), used: false, snap: null, last: null };   // two-finger scroll / pinch on the timeline
function pointerDrag(el, { down, move, up }) {
  let st = null;
  el.addEventListener('pointerdown', e => {
    if (multi.used || multi.pts.size > 1) { st = null; return; }
    el.setPointerCapture(e.pointerId); const r = el.getBoundingClientRect(); st = { x0: e.clientX, x: e.clientX - r.left, moved: false, r, id: e.pointerId }; down && down(st, e);
  });
  el.addEventListener('pointermove', e => {
    if (!st || e.pointerId !== st.id) return;
    if (multi.used) { st = null; return; }
    const dx = e.clientX - st.x0; if (Math.abs(dx) > 8) st.moved = true; st.dx = dx; st.x = e.clientX - st.r.left; move && move(st, e);
  });
  const end = e => { if (!st || e.pointerId !== st.id) return; const s0 = st; st = null; if (!multi.used) up && up(s0, e); };
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
}
{
  const det = $('detail');
  const centroid = () => { const p = [...multi.pts.values()]; return { x: (p[0].x + p[1].x) / 2, d: Math.max(20, Math.abs(p[0].x - p[1].x)) }; };
  det.addEventListener('pointerdown', e => {
    multi.pts.set(e.pointerId, { x: e.clientX - det.getBoundingClientRect().left });
    if (multi.pts.size === 1) multi.snap = { sel: S.sel && { ...S.sel }, grid: S.grid && { ...S.grid }, playhead: S.playhead };
    if (multi.pts.size === 2 && S.dur) {
      multi.used = true;
      // undo whatever the first finger started
      const sn = multi.snap;
      if (sn && S.grid && (sn.grid.t0 !== S.grid.t0 || sn.grid.bpm !== S.grid.bpm)) { Object.assign(S.grid, sn.grid); gridChanged(null, true); }
      if (sn) { S.sel = sn.sel; S.playhead = sn.playhead; }
      multi.last = centroid(); redraw();
    }
  }, true);
  det.addEventListener('pointermove', e => {
    if (!multi.pts.has(e.pointerId)) return;
    multi.pts.get(e.pointerId).x = e.clientX - det.getBoundingClientRect().left;
    if (!multi.used || multi.pts.size < 2) return;
    const c = centroid(), l = multi.last;
    const tAtC = S.view.start + l.x / S.view.pps;                         // time under the fingers
    S.view.pps = Math.max(4, Math.min(800, S.view.pps * c.d / l.d));      // pinch
    S.view.start = tAtC - c.x / S.view.pps;                               // pan, keeping that time under the fingers
    clampView(); multi.last = c; redraw();
  }, true);
  const lift = e => {
    multi.pts.delete(e.pointerId);
    if (multi.pts.size === 1 && multi.used) multi.last = null;
    if (multi.pts.size === 0) setTimeout(() => { if (!multi.pts.size) multi.used = false; }, 0);
  };
  det.addEventListener('pointerup', lift, true); det.addEventListener('pointercancel', lift, true);
  // trackpad / mouse wheel: scroll sideways, pinch or ctrl+wheel to zoom
  det.addEventListener('wheel', e => {
    if (!S.dur) return;
    e.preventDefault();
    if (e.ctrlKey) {
      const x = e.clientX - det.getBoundingClientRect().left, t = S.view.start + x / S.view.pps;
      S.view.pps = Math.max(4, Math.min(800, S.view.pps * Math.exp(-e.deltaY * 0.01)));
      S.view.start = t - x / S.view.pps;
    } else S.view.start += (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / S.view.pps;
    clampView(); redraw();
  }, { passive: false });
}
function handleHit(st) {          // top grips (ruler): left = bar start, right = bar end; each moves on its own
  if (!S.grid || S.locked || !selIsBars()) return false;
  const xa = (S.sel.a - S.view.start) * S.view.pps, xb = (S.sel.b - S.view.start) * S.view.pps;
  const n = Math.round((S.sel.b - S.sel.a) / barSec());
  if (Math.abs(st.x - xa) < 22) { Object.assign(st, { mode: 'start', a0: S.sel.a, b0: S.sel.b, n }); return true; }
  if (Math.abs(st.x - xb) < 22) { Object.assign(st, { mode: 'end', a0: S.sel.a, b0: S.sel.b, n }); return true; }
  return false;
}
function shiftHit(st, e) {        // bottom handle (drums lane): moves start and end together
  if (!S.grid || S.locked || !selIsBars()) return false;
  const y = e.clientY - st.r.top; if (y < st.r.height - SHIFT_H - 6) return false;
  const xa = (S.sel.a - S.view.start) * S.view.pps, xb = (S.sel.b - S.view.start) * S.view.pps;
  if (st.x < xa - 10 || st.x > xb + 10) return false;
  Object.assign(st, { mode: 'shift', a0: S.sel.a, xd: st.x, n: Math.round((S.sel.b - S.sel.a) / barSec()) });
  return true;
}
const setTempoFromLen = len => { S.grid.bpm = Math.max(30, Math.min(300, Math.round(240 / len * 100) / 100)); };
function handleMove(st) {
  if (st.mode === 'shift') {          // slide every bar line, tempo unchanged
    const a = Math.max(-barSec(), Math.min(S.dur, st.a0 + (st.x - st.xd) / S.view.pps));
    S.grid.t0 = a; gridChanged(null, true); S.sel = { a, b: a + st.n * barSec() }; redraw(); syncPlayback(true);
    status(`Bar moved: starts at ${a.toFixed(3)} s, ${S.grid.bpm.toFixed(2)} BPM — release to re-transcribe`);
  } else if (st.mode === 'start') {   // start follows the finger, end stays put: tempo follows
    const a = Math.min(st.b0 - 0.2 * st.n, xToT(st.x));
    setTempoFromLen((st.b0 - a) / st.n); S.grid.t0 = a;
    gridChanged(null, true); S.sel = { a, b: a + st.n * barSec() }; redraw(); syncPlayback(true);
    status(`Bar start ${a.toFixed(3)} s, ${S.grid.bpm.toFixed(2)} BPM — release to re-transcribe`);
  } else {                            // end follows the finger, start stays put: tempo follows
    const b = xToT(st.x), len = Math.max(0.2, b - st.a0) / st.n;
    setTempoFromLen(len); S.grid.t0 = st.a0;
    gridChanged(null, true); S.sel = { a: st.a0, b: st.a0 + st.n * barSec() }; redraw(); syncPlayback(true);
    status(`Bar end ${(st.a0 + st.n * barSec()).toFixed(3)} s, ${S.grid.bpm.toFixed(2)} BPM — release to re-transcribe`);
  }
}
function handleUp(st) {
  gridChanged(st.mode === 'shift' ? `Bar lines moved (tempo unchanged, ${S.grid.bpm.toFixed(2)} BPM).` : `Bar ${st.mode === 'start' ? 'start' : 'end'} set: ${S.grid.bpm.toFixed(2)} BPM.`);
}
// Ruler: tap = mark/unmark bar; drag = scroll; drag a grip = move the bar start / set the tempo
pointerDrag($('ruler'), {
  down: st => { st.v0 = S.view.start; st.mode = 'pan'; handleHit(st); },
  move: st => {
    if (!S.grid || !st.moved) return;
    if (st.mode !== 'pan') return handleMove(st);
    S.view.start = st.v0 - st.dx / S.view.pps; clampView(); redraw();
  },
  up: st => {
    if (!S.grid) return;
    if (st.moved) { if (st.mode !== 'pan') handleUp(st); return; }
    toggleMark(barAt(xToT(st.x)));
  }
});
// Waveforms: tap = select that bar (and move playhead); drag = select a range, snapped to bars
for (const id of ['wMix', 'wDrums']) pointerDrag($(id), {
  down: (st, e) => { st.t0 = xToT(st.x); st.mode = 'select'; if (id === 'wDrums') shiftHit(st, e); },
  move: st => {
    if (!S.grid || !st.moved) return;
    if (st.mode !== 'select') return handleMove(st);
    const a = Math.min(st.t0, xToT(st.x)), b = Math.max(st.t0, xToT(st.x));
    const ba = barAt(a), bb = barAt(b);
    S.sel = ba >= 0 && bb >= 0 ? { a: S.bars[ba].start, b: S.bars[bb].end } : { a: Math.max(0, a), b: Math.min(S.dur, b) };
    if (!playing) S.playhead = Math.max(0, S.sel.a);
    redraw();
  },
  up: st => {
    if (S.grid && st.moved && st.mode !== 'select') { handleUp(st); return; }
    if (S.grid && st.moved) { syncPlayback(); return; }
    if (!S.grid || st.moved) return;
    const t = xToT(st.x), i = barAt(t);
    S.playhead = Math.max(0, Math.min(S.dur, t));
    if (i >= 0) S.sel = { a: S.bars[i].start, b: S.bars[i].end };
    if (S.marks.includes(i)) { S.tab = i; renderTabs(); renderGrid(); }
    if (playing) startPlay(); else redraw();
  }
});
pointerDrag($('overview'), {
  down: st => moveOv(st), move: st => moveOv(st)
});
function moveOv(st) {
  if (!S.dur) return;
  const vw = waveW() / S.view.pps;
  S.view.start = st.x / st.r.width * S.dur - vw / 2; clampView(); redraw();
}
function zoom(f) {
  if (!S.dur) return;
  const w = waveW(), centre = S.view.start + w / S.view.pps / 2;
  S.view.pps = Math.max(4, Math.min(800, S.view.pps * f));
  S.view.start = centre - w / S.view.pps / 2; clampView(); redraw();
}
$('zoomIn').onclick = () => zoom(1.5);
$('zoomOut').onclick = () => zoom(1 / 1.5);

// ---------------------------------------------------------------- marks
function toggleMark(i) {
  if (i < 0) return;
  const k = S.marks.indexOf(i);
  if (k >= 0) { S.marks.splice(k, 1); if (S.tab === i) S.tab = null; }
  else {
    if (S.marks.length >= MAX_MARKS) { status(`A BMY bank holds 6 patterns: unmark a bar first.`); return; }
    S.marks.push(i); S.marks.sort((a, b) => a - b);
  }
  S.sel = { a: S.bars[i].start, b: S.bars[i].end }; S.playhead = Math.max(0, S.sel.a);
  afterMarksChanged();
}
function afterMarksChanged() {
  if (S.hits.length) buildPatterns(false);
  if (S.tab === null || !S.marks.includes(S.tab)) S.tab = S.marks.find(i => S.patterns.has(i)) ?? S.marks[0] ?? null;
  updateButtons(); renderTabs(); renderGrid(); redraw();
  syncPlayback(); scheduleSave();
}
$('markSel').onclick = () => {
  if (!S.sel) return;
  const a = barAt(S.sel.a + 0.001), b = barAt(S.sel.b - 0.001);
  if (a < 0 || b < 0) return;
  for (let i = a; i <= b; i++) if (!S.marks.includes(i)) { if (S.marks.length >= MAX_MARKS) { status('Marked the first bars only: a bank holds 6 patterns.'); break; } S.marks.push(i); }
  S.marks.sort((x, y) => x - y); afterMarksChanged();
};
$('clearMarks').onclick = () => { S.marks = []; S.tab = null; afterMarksChanged(); };
const covered = i => { const b = S.bars[i]; return !!b && S.extracted.some(([a, e]) => a <= Math.max(0, b.start - 0.3) && e >= Math.min(S.dur, b.end + 0.3)); };
function updateButtons() {
  const need = S.marks.filter(i => !covered(i)).length;
  $('extract').disabled = !S.mixBuf || !S.marks.length || busy;
  $('extract').textContent = !S.marks.length ? 'Extract drums' : need ? `Extract drums (${need} bar${need > 1 ? 's' : ''})` : 'Re-transcribe';
  $('export').disabled = $('exportFile').disabled = !S.marks.some(i => S.patterns.has(i)) || busy;
  $('undo').disabled = !S.undo.length;
}

// ---------------------------------------------------------------- open song (Files, Recent, Record)
const isVideo = f => /^video\//.test(f.type) || /\.(mp4|mov|m4v|webm)$/i.test(f.name);
$('file').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  e.target.value = '';
  const name = f.name.replace(/\.[^.]+$/, '');
  if (!isVideo(f)) { const bytes = await f.arrayBuffer(); await openSong({ key: `${f.name}|${f.size}`, name, bytes, type: f.type || 'audio' }); return; }
  // a video (e.g. a screen recording): take its sound, keep only that (as WAV), never the video itself
  stopPlay(); busy = true; updateButtons();
  try {
    status('Taking the sound from the video…', 0.02); crumb('video audio ' + name);
    let buf = null;
    if (!/[?&]vidrt\b/.test(location.search)) { try { buf = await audioCtx().decodeAudioData(await f.arrayBuffer()); log(`Video sound decoded directly (${f.type || 'video'})`); } catch (err) { log('Direct decode failed (' + (err.message || err) + '), playing the video through instead'); } }
    if (!buf) buf = await audioFromVideo(f, p => status(`Taking the sound from the video… ${Math.round(p * 100)}% (plays through once, silently)`, p));
    let pk = 0; for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i += 4) pk = Math.max(pk, Math.abs(d[i])); }
    if (pk < 0.001) throw new Error('that video has no sound (for screen recordings, the app’s sound must be playing while recording)');
    crumbDone(); busy = false;
    let nm = tidyName(name);
    try { const prev = await getSong(`${f.name}|${f.size}`); if (prev) nm = prev.name; else { const a = prompt('Name this song', nm); if (a && a.trim()) nm = a.trim(); } } catch (e) {}
    await openSong({ key: `${f.name}|${f.size}`, name: nm, bytes: wavBytes(buf), type: 'audio/wav', buf });
  } catch (err) {
    crumbDone(); busy = false; updateButtons(); status('Could not take the sound from that video: ' + (err.message || err)); log('ERROR video ' + errText(err));
  }
});
/** Fallback: play the video once (silently) through the audio graph and record what comes out. Real time. */
async function audioFromVideo(file, onProg) {
  const ctx = audioCtx(); try { await ctx.resume(); } catch (e) {}
  const url = URL.createObjectURL(file), v = document.createElement('video');
  v.src = url; v.playsInline = true; v.preload = 'auto'; v.setAttribute('playsinline', '');
  try {
    await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error('this browser cannot open that video')); });
    await ctx.audioWorklet.addModule('js/rec-worklet.js');
    const src = ctx.createMediaElementSource(v);
    const node = new AudioWorkletNode(ctx, 'bmy-recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain(); mute.gain.value = 0;
    src.connect(node); node.connect(mute); mute.connect(ctx.destination);
    const L = [], R = []; let frames = 0;
    node.port.onmessage = e => { const ch = e.data.ch; L.push(ch[0]); R.push(ch[1] || ch[0]); frames += ch[0].length; };
    await v.play();
    await new Promise((res, rej) => {
      const t = setInterval(() => { onProg(v.duration ? v.currentTime / v.duration : 0); }, 250);
      v.onended = () => { clearInterval(t); res(); };
      v.onerror = () => { clearInterval(t); rej(new Error('the video stopped playing')); };
    });
    node.port.postMessage('stop'); src.disconnect(); node.disconnect(); mute.disconnect();
    if (frames < SR * 0.2) throw new Error('no sound came out of the video');
    const buf = ctx.createBuffer(2, frames, SR);
    for (const [c, arr] of [[0, L], [1, R]]) { const d = buf.getChannelData(c); let o = 0; for (const a of arr) { d.set(a, o); o += a.length; } }
    log(`Video sound captured by playing it through: ${(frames / SR).toFixed(1)} s`);
    return buf;
  } finally { URL.revokeObjectURL(url); v.removeAttribute('src'); v.load(); }
}
/** Decode, then restore saved work for this song or analyse it fresh. */
async function openSong({ key, name, bytes, type, buf }) {
  phoneTab("song");
  stopPlay(); closeRecent();
  try {
    busy = true; updateButtons();
    status('Decoding the song…', 0.02); crumb('decoding ' + name);
    if (!buf) buf = await audioCtx().decodeAudioData(bytes.slice(0));
    S.locked = false;
    Object.assign(S, { key, name, dur: buf.duration, mixBuf: buf, drumBuf: null, drumMono: null, extracted: [],
      perf: null, hits: [], flux: [], patterns: new Map(), marks: [], tab: null, undo: [], sel: null, playhead: 0, ovPeaks: null, swing: { type: 'Off', depth: 0 } });
    S.mixMono = toMono(buf.getChannelData(0), buf.numberOfChannels > 1 ? buf.getChannelData(1) : null);
    $('songTitle').textContent = S.name; $('bankName').value = S.name;
    log(`Opened "${name}": ${buf.duration.toFixed(1)} s`);
    let saved = null;
    try { saved = await getSong(key); } catch (e) { log('Song memory unavailable: ' + e.message); }
    if (saved && saved.name && saved.name !== S.name) { S.name = saved.name; $('songTitle').textContent = S.name; $('bankName').value = S.name; }
    if (saved && saved.state) {
      await restoreState(saved.state);
      status(`Restored your work on "${S.name}": ${S.grid.bpm.toFixed(2)} BPM, ${S.marks.length} marked bar${S.marks.length === 1 ? '' : 's'}${S.patterns.size ? `, ${S.patterns.size} pattern${S.patterns.size > 1 ? 's' : ''}` : ''}.`, 1);
      log('Restored saved state for ' + key);
    } else {
      status('Finding the tempo and bar lines…', 0.1); await yieldUI();
      crumb('analysing tempo');
      S.mixSpec = await specAsync(S.mixMono, p => status('Finding the tempo and bar lines…', 0.1 + 0.85 * p));
      S.grid = detectGrid(S.mixSpec.data, S.mixSpec.frames); S.mixSpec = null;
      log(`Tempo ${S.grid.bpm} BPM, first bar line at ${S.grid.t0.toFixed(3)} s`);
      rebuildBars();
      { const w = waveW(); S.view.pps = Math.max(8, w / (barSec() * Math.max(3, Math.min(12, w / 110)))); }   // ~12 bars on iPad, ~3-4 on a phone
      S.view.start = Math.max(-0.2, S.grid.t0 - 0.2);
      status(`${S.grid.bpm} BPM. Tap a bar and play it. If beat 1 is wrong use ◀︎ Beat / Beat ▶︎, or use the orange handles on the selected bar: the top ones (bar-number strip) move the start or the end on its own; the bar under the drums lane moves the whole bar. Two fingers scroll and pinch-zoom. Then tap bar numbers to mark up to 6 bars.`, 1);
    }
    setBpmField();
    crumbDone();
    try { if (!(await hasFile(key))) await putFile({ key, bytes, type }); } catch (e) { log('Could not keep the audio file: ' + e.message); }
    askPersist();
    saveNow();
  } catch (err) {
    crumbDone(); status('Could not open that song: ' + err.message); log('ERROR ' + errText(err));
  } finally { busy = false; updateButtons(); showLock(); renderTabs(); renderGrid(); redraw(); }
}

// ---------------------------------------------------------------- song memory
function snapshot() {
  return {
    v: 1, grid: { ...S.grid }, locked: !!S.locked, marks: [...S.marks], tab: S.tab, swing: { ...S.swing },
    sel: S.sel && { ...S.sel }, view: { ...S.view }, mixer: S.mixer, bankName: $('bankName').value,
    extracted: S.extracted.map(r => [...r]),
    hits: S.hits.map(h => ({ time: h.time, cls: h.cls, strength: h.strength, levelDb: h.levelDb, decay: h.decay, pitch: h.pitch, ...(h.fixedLane ? { lane: h.lane, fixedLane: true } : {}) })),
    perf: S.perf || null,
    flux: (S.flux || []).map(r => ({ a: r.a, b: r.b, f0: r.f0, frames: r.frames, data: Array.from(r.data, v => Math.round(v * 100) / 100) })),
    patterns: [...S.patterns.entries()].map(([i, p]) => [i, { cells: p.cells, hatUseAlt: p.hatUseAlt, notes: p.notes || [], edited: !!p.edited }])
  };
}
async function restoreState(st) {
  S.grid = { ...st.grid }; rebuildBars();
  S.locked = !!st.locked; S.marks = (st.marks || []).filter(i => i < S.bars.length);
  S.swing = st.swing || { type: 'Off', depth: 0 };
  S.perf = st.perf || null;
  if (S.perf) {   // keep the recorded loop as a base layer so new overdubs add to it
    const sr = S.mixBuf.sampleRate, a0 = Math.round(0.5 * sr), n = Math.round(240 / S.perf.bpm * S.perf.bars * sr);
    Object.assign(P, { bpm: S.perf.bpm, bars: S.perf.bars, takes: S.perf.takes.map(t => ({ ...t, audio: null })), key: S.key, name: S.name, base: S.mixMono.slice(a0, a0 + n) });
  } else if (P.key && P.key !== S.key) Object.assign(P, { takes: [], key: null, base: null });
  S.hits = st.hits || []; S.flux = (st.flux || []).map(r => ({ ...r, data: Float32Array.from(r.data) })); S.patterns = new Map((st.patterns || []).filter(([i]) => i < S.bars.length));
  S.tab = S.patterns.has(st.tab) || S.marks.includes(st.tab) ? st.tab : (S.marks[0] ?? null);
  S.sel = st.sel || null; if (st.view) S.view = { ...st.view };
  if (st.bankName) $('bankName').value = st.bankName;
  const stems = await getStems(S.key).catch(() => null);
  if (S.perf) {                     // Perform song: the recording is the drums; layers' audio kept separately for Undo layer
    S.drumBuf = S.mixBuf; S.drumMono = S.mixMono; S.extracted = [[0, S.dur]];
    if (stems && stems.perfTakes) {
      const byId = new Map(stems.perfTakes.map(t => [t.id, t.audio]));
      P.takes.forEach(t => { t.audio = byId.get(t.id) || null; });
      P.base = stems.perfBase || (P.takes.every(t => t.audio) ? null : P.base);
    }
  } else if (stems && stems.regions && stems.regions.length) {
    S.drumBuf = audioCtx().createBuffer(2, S.mixBuf.length, SR); S.drumMono = new Float32Array(S.mixBuf.length);
    for (const r of stems.regions) {
      const s0 = Math.floor(r.a * SR);
      S.drumBuf.getChannelData(0).set(r.L.subarray(0, S.mixBuf.length - s0), s0);
      S.drumBuf.getChannelData(1).set(r.R.subarray(0, S.mixBuf.length - s0), s0);
      for (let i = 0; i < r.L.length && s0 + i < S.drumMono.length; i++) S.drumMono[s0 + i] = 0.5 * (r.L[i] + r.R[i]);
      S.extracted.push([r.a, r.b]);
    }
  }
}
let saveTimer = 0;
function scheduleSave() { clearTimeout(saveTimer); saveTimer = setTimeout(saveNow, 1200); }
async function saveNow() {
  clearTimeout(saveTimer);
  if (!S.key || !S.grid) return;
  try {
    await putSong({ key: S.key, name: S.name, updated: Date.now(), dur: S.dur, bpm: S.grid.bpm, patterns: S.patterns.size, marks: S.marks.length, state: snapshot() });
  } catch (e) { log('Save failed: ' + e.message); }
}
async function saveStems() {
  if (!S.key || !S.drumBuf) return;
  const L = S.drumBuf.getChannelData(0), R = S.drumBuf.getChannelData(1);
  const regions = S.extracted.map(([a, b]) => { const s0 = Math.floor(a * SR), s1 = Math.min(L.length, Math.floor(b * SR)); return { a, b, L: L.slice(s0, s1), R: R.slice(s0, s1) }; });
  try { await putStems({ key: S.key, regions }); } catch (e) { log('Could not keep the drum stems: ' + e.message); }
}

// ---------------------------------------------------------------- Recent songs
const ago = t => { const m = (Date.now() - t) / 60000; return m < 1 ? 'just now' : m < 60 ? `${Math.round(m)} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`; };
const esc = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function closeRecent() { $('recent').classList.add('hidden'); $('recentBtn').setAttribute('aria-expanded', 'false'); }
// ---------------------------------------------------------------- rename
const tidyName = n => {                                    // "ScreenRecording_11-26-2026 17-08-07_1" -> "Screen recording 26 Nov 17:08"
  const m = /^ScreenRecording_(\d\d)-(\d\d)-(\d{4}) (\d\d)-(\d\d)/.exec(n);
  if (!m) return n.replace(/[_]+/g, ' ').trim();
  const d = new Date(+m[3], +m[1] - 1, +m[2]);
  return `Screen recording ${d.getDate()} ${d.toLocaleString(undefined, { month: 'short' })} ${m[4]}:${m[5]}`;
};
function renameCurrent(nn) {
  nn = (nn || '').trim(); if (!nn || !S.key || nn === S.name) return;
  const old = S.name; S.name = nn; $('songTitle').textContent = nn;
  if (!$('bankName').value || $('bankName').value === old) $('bankName').value = nn;
  if (P.key === S.key) P.name = nn;
  saveNow(); log(`Renamed "${old}" to "${nn}"`);
}
$('songTitle').onclick = () => { if (!S.key) return; const nn = prompt('Rename this song', S.name); if (nn !== null) renameCurrent(nn); };
async function renameStored(key, nn) {
  nn = (nn || '').trim(); if (!nn) return false;
  if (key === S.key) { renameCurrent(nn); return true; }
  const rec = await getSong(key); if (!rec) return false;
  rec.name = nn; if (rec.state && (!rec.state.bankName || rec.state.bankName === rec.name)) rec.state.bankName = nn;
  await putSong(rec); return true;
}
$('recentBtn').onclick = async () => {
  if (!$('recent').classList.contains('hidden')) return closeRecent();
  let list = [];
  try { list = await listSongs(); } catch (e) { status('Song memory is not available in this browser.'); return; }
  const el = $('recent');
  el.innerHTML = `<h3>Recent songs</h3>` + (list.length ? list.map(r => `
    <div class="item" data-key="${esc(r.key)}">
      <div class="meta"><b>${esc(r.name)}</b><small>${r.bpm ? r.bpm.toFixed(1) + ' BPM · ' : ''}${r.marks || 0} marked · ${r.patterns || 0} pattern${r.patterns === 1 ? '' : 's'} · ${ago(r.updated)}</small></div>
      <button data-act="open">Open</button><button data-act="ren" title="Rename">✎</button><button data-act="del" title="Forget this song">✕</button>
    </div>`).join('') : '<div class="foot">No songs yet. Songs you open or record are kept here, with your work.</div>')
    + `<div class="foot">${standalone ? 'Kept in Groove Lab on this device.' : 'Kept in Safari on this device. Install Groove Lab (Add to Home Screen) so they’re kept for good; the installed app has its own song list.'}</div>`;
  el.querySelectorAll('.item').forEach(it => {
    const key = it.dataset.key;
    it.querySelector('[data-act=open]').onclick = async () => {
      const f = await getFile(key).catch(() => null), meta = list.find(r => r.key === key);
      if (!f) { status('The audio for that song is no longer stored. Open it from Files again.'); return; }
      await openSong({ key, name: meta.name, bytes: f.bytes, type: f.type });
    };
    it.querySelector('[data-act=ren]').onclick = async () => {
      const meta = list.find(r => r.key === key), nn = prompt('Rename this song', meta.name);
      if (nn === null || !nn.trim()) return;
      if (await renameStored(key, nn).catch(() => false)) { meta.name = nn.trim(); it.querySelector('.meta b').textContent = meta.name; }
    };
    const del = it.querySelector('[data-act=del]');
    del.onclick = async () => {
      if (del.textContent !== 'Forget?') { del.textContent = 'Forget?'; setTimeout(() => { del.textContent = '✕'; }, 3000); return; }
      await deleteSong(key); it.remove(); log('Forgot ' + key);
    };
  });
  el.classList.remove('hidden'); $('recentBtn').setAttribute('aria-expanded', 'true');
};
document.addEventListener('pointerdown', e => { if (!$('recent').classList.contains('hidden') && !$('recent').contains(e.target) && e.target !== $('recentBtn')) closeRecent(); });

// ---------------------------------------------------------------- input level meters (Record panel and Perform)
const dbOf = g => 20 * Math.log10(Math.max(1e-6, g));
const lvlPct = db => Math.max(0, Math.min(100, (db + 60) / 60 * 100));          // -60..0 dBFS across the bar
function runMeter(id, an, alive) {
  const el = $(id), txt = $(id + 'Db'), buf = new Float32Array(an.fftSize);
  let hold = -90, holdT = 0, clipT = 0;
  el.classList.remove('idle');
  const tick = () => {
    if (!alive()) { el.classList.add('idle'); el.classList.remove('clip'); txt.textContent = ''; txt.classList.remove('clip'); return; }
    an.getFloatTimeDomainData(buf);
    let pk = 0; for (let i = 0; i < buf.length; i++) { const v = Math.abs(buf[i]); if (v > pk) pk = v; }
    const db = dbOf(pk), now = performance.now();
    if (db > hold || now - holdT > 1500) { hold = db; holdT = now; }
    if (pk >= 0.98) clipT = now;
    const clip = now - clipT < 1500;
    el.firstChild.style.width = lvlPct(db) + '%';
    el.firstChild.style.background = db > -3 ? '#e5484d' : db > -12 ? '#f5c542' : '#3fbf6a';
    el.children[1].style.left = `calc(${lvlPct(hold)}% - 2px)`;
    el.classList.toggle('clip', clip); txt.classList.toggle('clip', clip);
    txt.textContent = clip ? 'CLIP' : hold < -59 ? '−∞ dB' : `${Math.round(hold)} dB`;
    requestAnimationFrame(tick);
  };
  tick();
}
const gainLabel = db => (db > 0 ? '+' : db < 0 ? '−' : '') + Math.abs(db) + ' dB';

// ---------------------------------------------------------------- Record from the audio input
// ● Record opens a panel: pick the input, watch the level, set the gain, then Start.
let rec = null, mon = null;
let recGainDb = 0; try { recGainDb = +(localStorage.getItem('bmyGLRecGain') || 0); } catch (e) {}
async function listInputs() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
    for (const id of ['inputSel', 'rsInput']) {
      const sel = $(id), cur = sel.value;
      sel.innerHTML = devs.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || 'Input ' + (i + 1))}</option>`).join('');
      if (cur) sel.value = cur;
    }
    $('inputSel').classList.add('hidden');
  } catch (e) {}
}
async function openInput() {
  closeInput();
  const ctx = audioCtx(), dev = $('rsInput').value;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: {
    deviceId: dev ? { exact: dev } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 2 } });
  await listInputs();
  const src = ctx.createMediaStreamSource(stream), gain = ctx.createGain(), an = ctx.createAnalyser();
  gain.gain.value = Math.pow(10, recGainDb / 20); an.fftSize = 2048;
  src.connect(gain); gain.connect(an);
  const m = { stream, src, gain, an, label: (stream.getAudioTracks()[0] || {}).label || 'input' };
  mon = m; runMeter('rsLvl', an, () => mon === m);
  return m;
}
function closeInput() {
  const m = mon; mon = null; if (!m) return;
  try { m.src.disconnect(); m.gain.disconnect(); } catch (e) {}
  m.stream.getTracks().forEach(t => t.stop());
}
function showRecSheet(on) {
  $('recSheet').classList.toggle('hidden', !on); $('rec').setAttribute('aria-expanded', String(on));
  if (!on && !rec) closeInput();
}
$('rec').onclick = async () => {
  if (rec) return stopRecording();
  if (!$('recSheet').classList.contains('hidden')) return showRecSheet(false);
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { status('This browser cannot record audio.'); return; }
  stopPlay(); closeRecent(); showRecSheet(true);
  $('rsGain').value = recGainDb; $('rsGainV').textContent = gainLabel(recGainDb);
  try { await openInput(); }
  catch (err) { showRecSheet(false); status('Could not open the input: ' + (err.name === 'NotAllowedError' ? 'microphone access was not allowed (Settings › Safari › Microphone).' : err.message)); log('ERROR input ' + errText(err)); }
};
$('rsInput').onchange = () => { if (!rec) openInput().catch(err => status('Could not open that input: ' + err.message)); };
$('rsGain').oninput = () => { recGainDb = +$('rsGain').value; $('rsGainV').textContent = gainLabel(recGainDb); if (mon) mon.gain.gain.value = Math.pow(10, recGainDb / 20); try { localStorage.setItem('bmyGLRecGain', recGainDb); } catch (e) {} };
$('rsClose').onclick = () => showRecSheet(false);
try { $('rsDenoise').checked = localStorage.getItem('bmyGLDenoise') !== '0'; } catch (e) {}
$('rsDenoise').onchange = () => { try { localStorage.setItem('bmyGLDenoise', $('rsDenoise').checked ? '1' : '0'); } catch (e) {} };
$('rsStart').onclick = () => rec ? stopRecording() : startRecording();
async function startRecording() {
  const ctx = audioCtx();
  try {
    const m = mon || await openInput();
    await ctx.audioWorklet.addModule('js/rec-worklet.js');
    const node = new AudioWorkletNode(ctx, 'bmy-recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain(); mute.gain.value = 0;
    m.gain.connect(node); node.connect(mute); mute.connect(ctx.destination);
    rec = { m, node, mute, chunks: [[], []], frames: 0, t0: performance.now(), peak: 0, label: m.label };
    node.port.onmessage = e => {
      const ch = e.data.ch; rec.chunks[0].push(ch[0]); rec.chunks[1].push(ch[1] || ch[0]); rec.frames += ch[0].length;
      let pk = 0; for (const v of ch[0]) pk = Math.max(pk, Math.abs(v)); rec.peak = Math.max(pk, rec.peak * 0.9);
      if (rec.frames > SR * 600) stopRecording();
    };
    $('rec').classList.add('on'); $('recMeter').classList.remove('hidden'); $('rsStart').classList.add('on'); $('rsInput').disabled = true;
    const tick = () => {
      if (!rec) return;
      const s = (performance.now() - rec.t0) / 1000, tm = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
      $('rec').textContent = `■ Stop ${tm}`; $('rsStart').textContent = `■ Stop recording ${tm}`;
      $('recMeter').firstChild.style.width = Math.min(100, Math.sqrt(rec.peak) * 100) + '%';
      $('recMeter').firstChild.style.background = rec.peak > 0.95 ? '#e5484d' : '#3fbf6a';
      requestAnimationFrame(tick);
    };
    tick();
    status(`Recording from ${rec.label}. Play the song, then tap Stop (up to 10 minutes).`);
    log(`Recording from ${rec.label}, gain ${gainLabel(recGainDb)}`);
  } catch (err) {
    rec = null; status('Could not start recording: ' + (err.name === 'NotAllowedError' ? 'microphone access was not allowed (Settings › Safari › Microphone).' : err.message));
    log('ERROR rec ' + errText(err));
  }
}
async function stopRecording() {
  const r = rec; rec = null; if (!r) return;
  try { r.node.port.postMessage('stop'); } catch (e) {}
  try { r.m.gain.disconnect(r.node); } catch (e) {} r.node.disconnect(); r.mute.disconnect();
  showRecSheet(false); closeInput();
  $('rsStart').classList.remove('on'); $('rsStart').textContent = '● Start recording'; $('rsInput').disabled = false;
  $('rec').classList.remove('on'); $('rec').textContent = phoneMQ.matches ? '● Rec' : '● Record'; $('recMeter').classList.add('hidden');
  if (r.frames < SR * 0.5) { status('Recording too short.'); return; }
  const ctx = audioCtx(), buf = ctx.createBuffer(2, r.frames, SR);
  for (let c = 0; c < 2; c++) { const d = buf.getChannelData(c); let o = 0; for (const ch of r.chunks[c]) { d.set(ch, o); o += ch.length; } }
  const when = new Date();
  const name = `Recording ${when.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })} ${when.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`;
  let pk = 0; for (let c = 0; c < 2; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) { const v = Math.abs(d[i]); if (v > pk) pk = v; } }
  log(`Recorded ${(r.frames / SR).toFixed(1)} s from ${r.label}, peak ${pk > 0 ? Math.round(dbOf(pk)) + ' dB' : 'silent'}, sample rate ${audioCtx().sampleRate}`);
  if (pk < 0.001) { status('That recording is silent: no sound reached the input. Check the meter in the Record panel before you start.'); return; }
  if ($('rsDenoise').checked) {
    status('Reducing hiss…'); await yieldUI();
    let nd = 0; for (let c = 0; c < 2; c++) { const r = denoise(buf.getChannelData(c)); buf.getChannelData(c).set(r.out); nd = r.noiseDb; }
    pk = 0; for (let c = 0; c < 2; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) { const v = Math.abs(d[i]); if (v > pk) pk = v; } }
    log(nd < -90 ? "Hiss reduction: no background hiss found" : `Reduced hiss (background was about ${Math.round(nd)} dB)`);
  }
  // normalise: loudest moment to -1 dBFS (boost at most +30 dB; never turn a loud clip down)
  const target = Math.pow(10, -1 / 20), boost = Math.min(Math.pow(10, 30 / 20), target / pk);
  if (boost > 1.05) {
    for (let c = 0; c < 2; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) d[i] *= boost; }
    log(`Normalised the recording: +${dbOf(boost).toFixed(1)} dB`);
  }
  await openSong({ key: 'rec-' + when.getTime(), name, bytes: wavBytes(buf), type: 'audio/wav', buf });
}
function wavBytes(buf) {
  const n = buf.length, ch = buf.numberOfChannels, out = new ArrayBuffer(44 + n * ch * 2), v = new DataView(out);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * ch * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, ch, true); v.setUint32(24, buf.sampleRate, true); v.setUint32(28, buf.sampleRate * ch * 2, true);
  v.setUint16(32, ch * 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, n * ch * 2, true);
  const chans = [...Array(ch)].map((_, c) => buf.getChannelData(c)); let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) { v.setInt16(o, Math.max(-1, Math.min(1, chans[c][i])) * 32767, true); o += 2; }
  return out;
}
navigator.mediaDevices && navigator.mediaDevices.addEventListener && navigator.mediaDevices.addEventListener('devicechange', listInputs);

async function specAsync(mono, onP) {   // logSpec in chunks so the page stays responsive
  const CH = SR * 20, parts = [];
  let frames = 0;
  for (let s = 0; s < mono.length; s += CH) {
    // compute each chunk with 1 s of context on both sides, keep the middle frames
    const a = Math.max(0, s - SR), b = Math.min(mono.length, s + CH + SR);
    const sp = logSpec(mono.subarray(a, b));
    const f0 = Math.round((s - a) / 441), f1 = f0 + Math.round((Math.min(mono.length, s + CH) - s) / 441);
    parts.push(sp.data.subarray(f0 * 84, Math.min(sp.frames, f1) * 84));
    onP(Math.min(1, (s + CH) / mono.length)); await yieldUI();
  }
  const total = parts.reduce((n, p) => n + p.length, 0), data = new Float32Array(total);
  let o = 0; for (const p of parts) { data.set(p, o); o += p.length; }
  frames = total / 84;
  return { data, frames };
}

// ---------------------------------------------------------------- tempo controls
function setBpmField() { $('bpm').value = S.grid ? S.grid.bpm.toFixed(2) : ''; }
const mid = i => (S.bars[i].start + S.bars[i].end) / 2;
function gridChanged(note, live) {
  const markT = S.marks.map(mid), tabT = S.tab !== null && S.bars[S.tab] ? mid(S.tab) : null;
  const selBar = S.sel ? barAt(S.sel.a + 0.001) : -1, selBarEnd = S.sel ? barAt(S.sel.b - 0.001) : -1;
  const selT = selBar >= 0 && S.sel.a === S.bars[selBar].start ? [mid(selBar), mid(selBarEnd)] : null;
  rebuildBars(); setBpmField();
  S.marks = [...new Set(markT.map(barAt).filter(i => i >= 0))].sort((a, b) => a - b);
  S.tab = tabT === null ? null : barAt(tabT);
  if (selT) { const a = barAt(selT[0]), b = barAt(selT[1]); if (a >= 0 && b >= 0) S.sel = { a: S.bars[a].start, b: S.bars[b].end }; }
  if (live) { redraw(); syncPlayback(true); return; }
  if (S.hits.length) { S.patterns = new Map(); buildPatterns(true); if (note) status(note); }
  else if (note) status(note);
  afterMarksChanged();
}
const anchor = () => (S.sel ? S.sel.a : S.playhead);
function setBpm(v) {
  if (!S.grid || !isFinite(v)) return;
  // keep the bar line nearest the selection where it is
  const a = anchor(), i = barAt(a), keep = i >= 0 ? S.bars[i].start : S.grid.t0;
  S.grid.bpm = Math.max(30, Math.min(300, Math.round(v * 100) / 100)); S.grid.t0 = keep;
  gridChanged('Tempo changed: patterns re-transcribed from the stored hits.');
}
$('bpmDn').onclick = () => S.grid && setBpm(S.grid.bpm - 0.1);
$('bpmUp').onclick = () => S.grid && setBpm(S.grid.bpm + 0.1);
$('bpm').addEventListener('change', () => setBpm(parseFloat($('bpm').value)));
$('half').onclick = () => S.grid && setBpm(S.grid.bpm / 2);
$('dbl').onclick = () => S.grid && setBpm(S.grid.bpm * 2);
$('nudgeL').onclick = () => { if (!S.grid) return; S.grid.t0 -= 0.01; gridChanged(); };
$('nudgeR').onclick = () => { if (!S.grid) return; S.grid.t0 += 0.01; gridChanged(); };
const GRID_CTRLS = ['bpmDn', 'bpm', 'bpmUp', 'half', 'dbl', 'beatL', 'beatR', 'nudgeL', 'nudgeR', 'barHere'];
function showLock() {
  $('lock').setAttribute('aria-pressed', !!S.locked);
  $('lock').textContent = S.locked ? '🔒 Locked' : 'Lock';
  for (const id of GRID_CTRLS) $(id).disabled = !!S.locked || !S.grid;
  $('lock').disabled = !S.grid;
}
$('lock').onclick = () => {
  S.locked = !S.locked; showLock(); redraw(); scheduleSave();
  status(S.locked ? `Tempo and bar lines locked at ${S.grid.bpm.toFixed(2)} BPM. Tap Locked to unlock.` : 'Tempo and bar lines unlocked.');
};
$('beatL').onclick = () => { if (!S.grid) return; S.grid.t0 -= 60 / S.grid.bpm; gridChanged('Bar lines moved one beat earlier.'); };
$('beatR').onclick = () => { if (!S.grid) return; S.grid.t0 += 60 / S.grid.bpm; gridChanged('Bar lines moved one beat later.'); };
$('barHere').onclick = () => { if (!S.grid) return; S.grid.t0 = S.playhead; gridChanged('Bar line moved to the playhead.'); };

// ---------------------------------------------------------------- extract + transcribe
let busy = false;
$('extract').onclick = async () => {
  if (busy || !S.marks.length) return;
  busy = true; updateButtons(); stopPlay();
  try {
    const o = await getOrt();
    const todo = S.marks.filter(i => !covered(i));
    if (todo.length) {
      crumb('loading separator');
      const lean = /iPhone|iPod/.test(navigator.userAgent) || phoneMQ.matches || /[?&]lean\b/.test(location.search);
      const skipSep = TEST_NO_DEMUCS || (lean && sepFailed() && confirm('Drum separation ran out of memory on this phone last time.\n\nOK: find the drums straight from the full song (quicker, a little less clean).\nCancel: try separating again.'));
      if (skipSep && !TEST_NO_DEMUCS) log('Skipping separation: finding drums in the full song');
      if (!skipSep) { await loadDemucs(o, (m, f, c) => { status(m, f * 0.3); if (c) crumb(c); }, { lean }); log(`Drum separator ready (${lean ? 'low-memory mode: one part at a time' : 'all parts loaded'})`); }
      // regions: each bar +-1 s, merged when close
      const regs = todo.map(i => [Math.max(0, S.bars[i].start - 1), Math.min(S.dur, S.bars[i].end + 1)]).sort((a, b) => a[0] - b[0]);
      const merged = [];
      for (const r of regs) { const l = merged[merged.length - 1]; if (l && r[0] - l[1] < 3) l[1] = Math.max(l[1], r[1]); else merged.push([...r]); }
      if (!S.drumBuf) {
        S.drumBuf = audioCtx().createBuffer(2, S.mixBuf.length, SR);
        S.drumMono = new Float32Array(S.mixBuf.length);
      }
      const total = merged.reduce((n, r) => n + r[1] - r[0], 0); let done = 0;
      const L = S.mixBuf.getChannelData(0), R = S.mixBuf.numberOfChannels > 1 ? S.mixBuf.getChannelData(1) : L;
      for (const [a, b] of merged) {
        const s0 = Math.floor(a * SR), s1 = Math.floor(b * SR);
        crumb(`separating ${a.toFixed(1)}-${b.toFixed(1)} s`);
        const t = performance.now();
        const d = skipSep ? { left: L.slice(s0, s1), right: R.slice(s0, s1) } : await separateDrums(o, L.subarray(s0, s1), R.subarray(s0, s1), p => status(`Separating drums… ${Math.round((done + p * (b - a)) / total * 100)}%`, 0.3 + 0.55 * (done + p * (b - a)) / total),
          (sg, ns, i, n) => { crumb(`separating ${a.toFixed(1)}-${b.toFixed(1)} s, segment ${sg}/${ns}, part ${i}/${n}`); if (lean) status(`Separating drums… segment ${sg} of ${ns}, part ${i} of ${n}`, 0.3 + 0.55 * (done + ((sg - 1) + i / n) / ns * (b - a)) / total); });
        if (!skipSep) { log(`Separated ${a.toFixed(1)}–${b.toFixed(1)} s in ${((performance.now() - t) / 1000).toFixed(1)} s`); try { localStorage.removeItem('bmyGLSepFail'); } catch (e) {} }
        S.drumBuf.getChannelData(0).set(d.left, s0); S.drumBuf.getChannelData(1).set(d.right, s0);
        for (let i = 0; i < d.left.length; i++) S.drumMono[s0 + i] = 0.5 * (d.left[i] + d.right[i]);
        S.extracted.push([a, b]); done += b - a;
        redraw();
        status('Finding hits…', 0.86 + 0.1 * done / total); await yieldUI();
        await transcribeRegion(o, a, b);
      }
    } else {                                  // Re-transcribe: run the drum transcriber again on the drums we already have
      for (const [a, b] of S.extracted) { status(`Finding hits ${a.toFixed(1)}–${b.toFixed(1)} s…`); await yieldUI(); await transcribeRegion(o, a, b); }
    }
    buildPatterns(!todo.length);
    S.tab = S.marks.find(i => S.patterns.has(i)) ?? null;
    crumbDone(); phoneTab('pattern');
    status(`Done. Swing ${S.swing.type === 'Off' ? 'off' : S.swing.type + ' ' + S.swing.depth + '%'}. Check each pattern (turn on Drums and Pattern to hear them together), then Export to BMY.`, 1);
  } catch (err) {
    let at = ''; try { at = localStorage.getItem('bmyGLStage') || ''; } catch (e) {}
    crumbDone(); status('Extraction failed: ' + (err.message || err)); log('ERROR during ' + at + ': ' + errText(err));
  } finally { busy = false; updateButtons(); renderTabs(); renderGrid(); redraw(); saveNow(); saveStems(); }
};
const FLUX_BANDS = Object.fromEntries(Object.entries({ kick: [30, 150], snare: [150, 5000], tom: [60, 600], hihat: [5000, 16000], cymbal: [3000, 16000] })
  .map(([k, [lo, hi]]) => [k, [...Array(NBANDS).keys()].filter(i => BAND_FREQ[i] >= lo && BAND_FREQ[i] < hi)]));
/** Second strokes of doubles: a clear level jump on the same drum 0.3-0.7 step after a hit. */
function doubleStrokes(bars) {
  const out = [];
  for (const h of S.hits) {
    const b = bars.find(x => h.time >= x.start - x.step / 4 && h.time < x.end - x.step / 4); if (!b) continue;
    if (!['snare', 'tom'].includes(h.cls)) continue;   // kick ring and hats/cymbals give false doubles
    const c = CLASSES.indexOf(h.cls), fr = (S.flux || []).find(r => h.time >= r.a && h.time < r.b); if (c < 0 || !fr) continue;
    const at = t => Math.round(t * FPS) - fr.f0, val = f => (f > 0 && f < fr.frames ? fr.data[f * 5 + c] : 0);
    const fh = at(h.time); let main = 0; for (let f = fh - 2; f <= fh + 2; f++) main = Math.max(main, val(f));
    const w0 = at(h.time + 0.3 * b.step), w1 = at(h.time + 0.7 * b.step), n0 = at(h.time + 0.15 * b.step), n1 = at(h.time + 0.85 * b.step);
    let pk = 0, pf = -1; for (let f = w0; f <= w1; f++) if (val(f) > pk) { pk = val(f); pf = f; }
    const around = []; for (let f = n0; f <= n1; f++) if (Math.abs(f - pf) > 1) around.push(val(f));
    around.sort((x, y) => x - y); const floor = around.length ? around[Math.floor(around.length / 2)] : 0;
    const t = (pf + fr.f0) / FPS;
    if (pf > 0 && pk > 0.12 * main && pk > 2.8 * floor + 0.25 && !S.hits.concat(out).some(o => o.cls === h.cls && Math.abs(o.time - t) < 0.3 * b.step))   // once only (two hits can see the same stroke)
      out.push({ time: t, cls: h.cls, lane: h.lane, vel: 'sf', levelDb: (h.levelDb || 0) - 10, ghost: true });
  }
  return out;
}
async function transcribeRegion(o, a, b) {
  crumb('transcribing');
  const model = transcriber();
  if (model === 'oaf') await loadOaf(o, 'models/oaf_drums.onnx'); else if (model === 'bmy') await loadBmyt(o); else await loadAdtof(o, 'models/adtof.onnx');   // loadBmyt(o): versioned URL, so a new model isn't served from cache
  const s0 = Math.floor(a * SR), s1 = Math.min(S.drumMono.length, Math.floor(b * SR));
  // normalise the stem to -1 dBFS peak (max +30 dB) for detection only; playback and velocities are unaffected
  const raw = S.drumMono.subarray(s0, s1);
  let peak = 0; for (let i = 0; i < raw.length; i++) { const v = Math.abs(raw[i]); if (v > peak) peak = v; }
  const gain = peak > 0 ? Math.min(31.6, 0.891 / peak) : 1;
  const mono = gain === 1 ? raw : raw.map(v => v * gain);
  log(`Stem ${a.toFixed(1)}–${b.toFixed(1)} s peaked at ${(20 * Math.log10(peak || 1e-9)).toFixed(1)} dBFS; normalised by ${(20 * Math.log10(gain)).toFixed(1)} dB for transcription`);
  const spec = logSpec(mono);
  let hits;
  if (model === 'bmy') {                  // BMY's own model (training/)
    const t = performance.now();
    hits = (await bmytHits(o, mono)).map(h => ({ ...h, time: h.time + s0 / SR }));
    log(`BMY transcriber: ${hits.length} hits in ${((performance.now() - t) / 1000).toFixed(1)} s`);
  } else if (model === 'oaf') {          // Magenta OaF Drums (licence-friendly candidate for BMY)
    const t = performance.now();
    hits = (await oafHits(o, mono)).map(h => ({ ...h, time: h.time + s0 / SR }));
    const fixed = hits.filter(h => h.fromSnare).length;
    log(`Magenta transcriber: ${hits.length} hits in ${((performance.now() - t) / 1000).toFixed(1)} s${fixed ? `; ${fixed} "snare" hit(s) re-read as hi-hat` : ''}`);
  } else {
    const { act, frames } = await activations(o, mono, spec);
    hits = hitsFrom(act, frames).map(h => ({ ...h, time: h.time + s0 / SR }));
  }
  describeHits(hits, spec.data, spec.frames, Math.round(s0 / SR * FPS));
  // per-drum level jumps (band spectral flux): finds quiet second strokes the transcriber misses
  const flux = new Float32Array(spec.frames * 5);
  CLASSES.forEach((cls, c) => {
    const bands = FLUX_BANDS[cls];
    for (let f = 1; f < spec.frames; f++) { let v = 0; for (const k of bands) { const d = spec.data[f * NBANDS + k] - spec.data[(f - 1) * NBANDS + k]; if (d > 0) v += d; } flux[f * 5 + c] = v; }
  });
  S.flux = (S.flux || []).filter(r => r.a !== a).concat([{ a, b, f0: Math.round(s0 / SR * FPS), frames: spec.frames, data: flux }]);
  S.hits = S.hits.filter(h => h.time < a || h.time >= b).concat(hits).sort((x, y) => x.time - y.time);
  log(`Hits in ${a.toFixed(1)}–${b.toFixed(1)} s: ${countBy(hits)}`);
}
const countBy = hs => Object.entries(hs.reduce((m, h) => ((m[h.cls] = (m[h.cls] || 0) + 1), m), {})).map(([k, v]) => `${k} ${v}`).join(', ');

/** (Re)build patterns for marked bars that have drums. force: also rebuild edited ones. */
function buildPatterns(force) {
  const bars = S.marks.filter(covered).map(i => S.bars[i]);
  if (!bars.length) return;
  assignLanes(S.hits, bars);
  S.swing = detectSwing(S.hits, bars);
  const extra = doubleStrokes(bars);   // quiet second strokes of doubles
  const all = S.hits.concat(extra).sort((x, y) => x.time - y.time);
  for (const b of bars) {
    const old = S.patterns.get(b.i);
    if (old && old.edited && !force) continue;
    S.patterns.set(b.i, { ...barToPattern(all, b, S.swing), edited: false });
    // what was heard in this bar, in steps (for checking against the grid)
    const inBar = all.filter(h => h.time >= b.start - b.step / 4 && h.time < b.end - b.step / 4)
      .map(h => `${((h.time - b.start) / b.step).toFixed(2)} ${h.lane}${h.ghost ? '(ghost)' : ''} ${h.vel}`);
    log(`Bar ${barNo(b.i)} hits [step lane vel]: ` + inBar.join(', '));
  }
  for (const k of [...S.patterns.keys()]) if (!S.marks.includes(k)) S.patterns.delete(k);
  if (force) S.undo = [];
}

// ---------------------------------------------------------------- tabs + grid
function renderTabs() {
  const el = $('tabs'); el.innerHTML = '';
  S.marks.forEach((i, k) => {
    const b = document.createElement('button');
    const has = S.patterns.has(i);
    b.textContent = `${k + 1} · Bar ${barNo(i)}${has ? '' : ' (not extracted)'}`;
    b.className = i === S.tab ? 'on' : '';
    b.onclick = () => { S.tab = i; S.sel = { a: S.bars[i].start, b: S.bars[i].end }; S.playhead = Math.max(0, S.sel.a); ensureVisible(S.sel.a, S.sel.b); renderTabs(); renderGrid(); redraw(); if (playing) startPlay(); };
    el.appendChild(b);
  });
}
function ensureVisible(a, b) {
  const vw = waveW() / S.view.pps;
  if (a < S.view.start || b > S.view.start + vw) { S.view.start = a - (vw - (b - a)) / 2; clampView(); }
}
const SPAN = { t4: 4, t8: 2, t16: 1 };
const glyphSize = v => 0.34 + (0.90 - 0.34) * Math.max(0, VEL.indexOf(v)) / 5;
const GLYPHS = {
  crash: c => `<svg viewBox="0 0 32 32"><path d="M16 3l3 7 7-3-3 7 7 2-7 3 3 7-7-3-3 7-3-7-7 3 3-7-7-3 7-2-3-7 7 3z" fill="none" stroke="${c}" stroke-width="1.6"/><circle cx="16" cy="16" r="2" fill="none" stroke="${c}"/></svg>`,
  openHat: c => `<svg viewBox="0 0 32 32"><path d="M5 6h22L16 14zM5 26h22L16 18z" fill="none" stroke="${c}" stroke-width="1.8"/></svg>`,
  closedHat: c => `<svg viewBox="0 0 32 32"><path d="M3 16L16 7l13 9-13 9z" fill="none" stroke="${c}" stroke-width="1.8"/><path d="M5 16h22" stroke="${c}" stroke-width="3"/></svg>`,
  snare: c => `<svg viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill="none" stroke="${c}" stroke-width="1.8"/><circle cx="16" cy="16" r="6" fill="none" stroke="${c}" stroke-width="1.6"/></svg>`,
  snareAlt: c => `<svg viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill="none" stroke="${c}" stroke-width="1.8"/><path d="M16 9v14M9 16h14" stroke="${c}" stroke-width="1.6"/></svg>`,
  kick: c => `<svg viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill="none" stroke="${c}" stroke-width="1.8"/><circle cx="16" cy="17" r="4" fill="${c}"/><path d="M16 21v7" stroke="${c}" stroke-width="1.6"/></svg>`,
  tom: (c, r) => `<svg viewBox="0 0 32 32"><path d="${octo(16, 16, r)}" fill="none" stroke="${c}" stroke-width="1.8"/></svg>`,
  shot: c => `<svg viewBox="0 0 32 32"><path d="M3 16h3M8 12v8M11 9v14M14 13v6M17 6v20M20 11v10M23 14v4M26 15v2M29 16h1" stroke="${c}" stroke-width="1.6"/></svg>`
};
function octo(cx, cy, r) { let d = ''; for (let k = 0; k < 8; k++) { const a = Math.PI / 8 + k * Math.PI / 4; d += (k ? 'L' : 'M') + (cx + r * Math.cos(a)).toFixed(1) + ' ' + (cy + r * Math.sin(a)).toFixed(1); } return d + 'z'; }
function laneGlyph(l) {
  if (l.role.startsWith('tom')) return GLYPHS.tom(l.color, { tomHigh: 8, tomMid: 10.5, tomFloor: 13 }[l.role]);
  if (l.role.startsWith('oneShot')) return GLYPHS.shot(l.color);
  return GLYPHS[l.role](l.color);
}
function renderGrid() {
  const p = S.tab !== null ? S.patterns.get(S.tab) : null;
  $('empty').classList.toggle('hidden', !!p);
  $('grid').classList.toggle('hidden', !p);
  $('gridInfo').classList.toggle('hidden', !p);
  if (!p) {
    $('empty').innerHTML = !S.mixBuf ? 'Open a song to start.' : !S.marks.length
      ? 'Tap bar numbers on the ruler to mark up to 6 bars (or drag across the waveform and tap Mark selection).<br>Each marked bar becomes one BMY pattern.'
      : 'Tap Extract drums to separate the drums in the marked bars and transcribe them.';
    return;
  }
  const g = $('grid'); g.innerHTML = '';
  const px = { h: 0, w: 0 }, drawers = [];   // notes are sized once all rows are laid out
  const swing = S.swing;
  const shift = s => !swing || swing.type === 'Off' ? 0 : swing.type === '1-8' && s % 4 === 2 ? swing.depth / 100 * 2 / 3 : swing.type === '1-16' && s % 2 === 1 ? swing.depth / 100 / 3 : 0;
  for (const l of LANES) {
    const row = document.createElement('div'); row.className = 'row';
    const color = l.role === 'closedHat' && p.hatUseAlt ? '#d9b326' : l.color;
    row.innerHTML = `<div class="glyph" title="${l.role === 'closedHat' && p.hatUseAlt ? 'Ride (hat lane)' : l.name}">${laneGlyph({ ...l, color })}</div>`;
    row.firstChild.onclick = () => audition(audioCtx(), l.role === 'closedHat' && p.hatUseAlt ? 'ride' : l.role, 'n2');
    const trips = p.cells.filter(c => c.lane === l.role && SPAN[c.ornament]);
    for (let s = 0; s < 16; s++) {
      const cell = document.createElement('div');
      const cov = trips.some(t => s >= t.step && s < t.step + SPAN[t.ornament]);
      cell.className = 'cell' + (s % 4 === 0 ? ' beat' : '') + (cov ? ' cov' : '');
      const c = p.cells.find(x => x.lane === l.role && x.step === s && !SPAN[x.ornament]);
      if (c) cell._note = c;
      cell.onclick = () => editCell(l.role, s);
      row.appendChild(cell);
    }
    // triplet notes drawn across their span
    for (const t of trips) {
      const span = SPAN[t.ornament], bits = t.tripletPattern || '111';
      const vels = [t.velocity, t.note2Velocity || t.velocity, t.note3Velocity || t.velocity];
      for (let k = 0; k < 3; k++) {
        if (bits[k] !== '1') continue;
        const pos = t.step + 0.5 + k * span / 3;      // centre of the note, in steps
        const d = document.createElement('div'); d.className = 'trip';
        const z = glyphSize(vels[k]);
        d.dataset.z = z;
        d.style.cssText = `left:calc(46px + (100% - 46px) * ${(pos / 16).toFixed(4)});background:${color};box-shadow:0 0 0 1px rgba(255,255,255,.55)`;
        row.appendChild(d);
      }
    }
    g.appendChild(row);
    drawers.push(() => {
    const side = Math.max(6, Math.min(px.h, px.w) - 4);
    const dot = (parent, leftPct, z, extra = '') => {
      const n = document.createElement('div'); n.className = 'note';
      const d = Math.round(side * z);
      n.style.cssText = `width:${d}px;height:${d}px;left:${leftPct.toFixed(1)}%;background:${color};${extra}`;
      parent.appendChild(n);
    };
    [...row.children].slice(1).forEach((cell, s) => {
      const c = cell._note; if (!c) return;
      const z = glyphSize(c.velocity), x0 = 50 + shift(s) * 100;
      if (c.ornament === 'double') { dot(cell, x0 - 18, z * 0.9); dot(cell, x0 + 22, z * 0.8); }
      else if (c.ornament === 'flam') { dot(cell, x0 - 26, 0.32, 'opacity:.85'); dot(cell, x0 + 6, z); }
      else dot(cell, x0, z);
      if (c.ornament && !SPAN[c.ornament]) { const t = document.createElement('span'); t.className = 'orn'; t.textContent = c.ornament === 'double' ? 'Db' : c.ornament === 'flam' ? 'Fl' : c.ornament; cell.appendChild(t); }
    });
    // triplet notes, sized in px too
    row.querySelectorAll('.trip').forEach(d => { const zz = +d.dataset.z; const dd = Math.round(side * zz * 0.8); d.style.width = dd + 'px'; d.style.height = dd + 'px'; });
    });
  }
  { const r = g.querySelector('.row .cell').getBoundingClientRect(); px.h = r.height; px.w = r.width; }   // all 12 rows laid out
  drawers.forEach(f => f());
  $('swType').value = swing.type; $('swDepth').value = swing.depth; $('swDepth').disabled = swing.type === 'Off';
  $('hatRide').checked = !!p.hatUseAlt;
  const tripN = p.cells.filter(c => SPAN[c.ornament]).length;
  const dbl = p.cells.filter(c => c.ornament === 'double').length, flm = p.cells.filter(c => c.ornament === 'flam').length;
  $('patNotes').textContent = [tripN ? `${tripN} triplet${tripN > 1 ? 's' : ''}` : '', dbl ? `${dbl} double${dbl > 1 ? 's' : ''}` : '', flm ? `${flm} flam${flm > 1 ? 's' : ''}` : '', ...(p.notes || [])].filter(Boolean).join(' · ');
}
function editCell(lane, step) {
  const p = S.patterns.get(S.tab); if (!p) return;
  S.undo.push({ bar: S.tab, cells: JSON.parse(JSON.stringify(p.cells)), hatUseAlt: p.hatUseAlt });
  const trip = p.cells.find(c => c.lane === lane && SPAN[c.ornament] && step >= c.step && step < c.step + SPAN[c.ornament]);
  const hit = p.cells.find(c => c.lane === lane && c.step === step && !SPAN[c.ornament]);
  if (trip) p.cells.splice(p.cells.indexOf(trip), 1);
  else if (hit) p.cells.splice(p.cells.indexOf(hit), 1);
  else p.cells.push({ lane, step, velocity: 'n2', tripletPattern: '111' });
  p.edited = true;
  updateButtons(); renderGrid(); syncPlayback(); scheduleSave();
}
$('undo').onclick = () => {
  const u = S.undo.pop(); if (!u) return;
  const p = S.patterns.get(u.bar); if (p) { p.cells = u.cells; p.hatUseAlt = u.hatUseAlt; }
  S.tab = u.bar; updateButtons(); renderTabs(); renderGrid(); syncPlayback(); scheduleSave();
};
$('swType').onchange = () => { S.swing = { type: $('swType').value, depth: $('swType').value === 'Off' ? 0 : (+$('swDepth').value || 50) }; renderGrid(); syncPlayback(); scheduleSave(); };
$('swDepth').onchange = () => { S.swing.depth = Math.max(0, Math.min(100, Math.round(+$('swDepth').value || 0))); renderGrid(); syncPlayback(); scheduleSave(); };
$('hatRide').onchange = () => { const p = S.patterns.get(S.tab); if (!p) return; p.hatUseAlt = $('hatRide').checked ? 1 : 0; p.edited = true; renderGrid(); syncPlayback(); scheduleSave(); };

// ---------------------------------------------------------------- playback (3-channel mixer)
const CH = ['mix', 'drums', 'pattern'];
const MIX_DEFAULT = { mix: { on: true, vol: 80 }, drums: { on: false, vol: 80 }, pattern: { on: false, vol: 70 } };
try { S.mixer = JSON.parse(localStorage.getItem('bmyGLMixer')) || MIX_DEFAULT; } catch (e) { S.mixer = MIX_DEFAULT; }
for (const c of CH) S.mixer[c] = Object.assign({}, MIX_DEFAULT[c], S.mixer[c]);
const gainOf = c => (S.mixer[c].on ? Math.pow(S.mixer[c].vol / 100, 2) : 0);
function saveMixer() { try { localStorage.setItem('bmyGLMixer', JSON.stringify(S.mixer)); } catch (e) {} }
function showMixer() {
  for (const c of CH) { $('ch-' + c).setAttribute('aria-pressed', S.mixer[c].on); $('vol-' + c).value = S.mixer[c].vol; }
}
for (const c of CH) {
  $('ch-' + c).onclick = () => { S.mixer[c].on = !S.mixer[c].on; showMixer(); saveMixer(); applyGains(); };
  $('vol-' + c).oninput = () => { S.mixer[c].vol = +$('vol-' + c).value; if (!S.mixer[c].on) { S.mixer[c].on = true; showMixer(); } saveMixer(); applyGains(); };
}
showMixer();

let playing = false, nodes = {}, gains = {}, raf = 0, patA = 0;
let posNow = 0, lastCtxT = 0, loopA = 0, loopB = 0;   // song-time position and loop region
function applyGains() {
  if (!actx) return;
  for (const c of CH) if (gains[c]) gains[c].gain.setTargetAtTime(gainOf(c), actx.currentTime, 0.015);
}
function killNodes() {
  for (const c of CH) if (nodes[c]) { try { nodes[c].stop(); } catch (e) {} nodes[c].disconnect(); }
  nodes = {};
}
/** Bars whose patterns play in [a, b): marked bars with patterns; if none, the open tab's pattern on every bar. */
function patternItems(a, b) {
  const inRange = S.bars.filter(x => x.end > a && x.start < b);
  let items = inRange.filter(x => S.patterns.has(x.i)).map(x => ({ start: x.start, stepSec: x.step, pattern: S.patterns.get(x.i) }));
  const tabP = S.tab !== null && S.patterns.get(S.tab);
  if (!items.length && tabP) items = inRange.map(x => ({ start: x.start, stepSec: x.step, pattern: tabP }));
  return items;
}
/** Keep playing, but follow the current selection, bar lines, swing and pattern. live = mid-drag. */
function syncPlayback(live) {
  if (!playing) return;
  const a = S.sel ? Math.max(0, S.sel.a) : loopA, b = S.sel ? Math.min(S.dur, S.sel.b) : loopB;
  if (b - a < 0.05) return;
  const outside = posNow < a - 0.01 || posNow >= b;
  if (live && !outside && S.loop) {                 // move the loop points of song + drums; pattern catches up on release
    for (const c of ['mix', 'drums']) if (nodes[c]) { nodes[c].loopStart = a; nodes[c].loopEnd = b; }
    loopA = a; loopB = b; return;
  }
  startPlay(outside ? a : posNow);
}
$('loop').onclick = () => { S.loop = !S.loop; $('loop').classList.toggle('on', S.loop); if (playing) startPlay(posNow); };
$('gridPlay').onclick = () => $('play').click();
$('play').onclick = () => { if (!playing && P.clock && !P.rec) stopClock(); return playing ? stopPlay() : startPlay(); };
// playhead on the pattern grid: where playback is inside the bar on screen
function drawGridHead(t) {
  if (t != null && playing) {                       // follow playback: show the pattern of the marked bar being played
    const i = S.marks.find(m => t >= S.bars[m].start && t < S.bars[m].end);
    if (i !== undefined && i !== S.tab && S.patterns.has(i)) { S.tab = i; renderTabs(); renderGrid(); }
  }
  const g = $('grid'); let el = $('gridHead');
  const bar = S.tab != null && S.bars[S.tab], row = g.querySelector('.row');
  const frac = bar && t != null ? (t - bar.start) / (bar.end - bar.start) : -1;
  if (!row || g.classList.contains('hidden') || frac < 0 || frac >= 1) { if (el) el.style.display = 'none'; return; }
  if (!el) { el = document.createElement('div'); el.id = 'gridHead'; g.appendChild(el); }
  const c0 = row.children[1], c1 = row.children[16];
  const x0 = row.offsetLeft + c0.offsetLeft, x1 = row.offsetLeft + c1.offsetLeft + c1.offsetWidth;
  el.style.display = 'block'; el.style.left = (x0 + frac * (x1 - x0)).toFixed(1) + 'px';
  const x = x0 + frac * (x1 - x0);                                         // phone: keep it in view when the grid scrolls sideways
  if (g.scrollWidth > g.clientWidth + 4 && (x > g.scrollLeft + g.clientWidth - 24 || x < g.scrollLeft + 44)) g.scrollLeft = Math.max(0, x - 60);
}
function stopPlay() {
  drawGridHead(null);
  killNodes(); playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶︎ Play'; $('gridPlay').textContent = '▶︎'; $('gridPlay').classList.remove('on');
}
function startPlay(from) {
  if (!S.mixBuf) return;
  const ctx = audioCtx();
  killNodes();
  let a = S.sel ? Math.max(0, S.sel.a) : S.playhead, b = S.sel ? Math.min(S.dur, S.sel.b) : S.dur;
  if (!S.sel) b = Math.min(S.dur, a + 120);
  if (b - a < 0.05) return;
  let pos = from !== undefined && from >= a && from < b ? from : a;
  const t0 = ctx.currentTime + 0.04;
  const make = (c, buf, ls, le, offset) => {
    if (!gains[c]) { gains[c] = ctx.createGain(); gains[c].connect(ctx.destination); }
    gains[c].gain.setValueAtTime(gainOf(c), ctx.currentTime);
    const n = ctx.createBufferSource(); n.buffer = buf; n.connect(gains[c]);
    n.loop = S.loop; n.loopStart = ls; n.loopEnd = le;
    if (S.loop) n.start(t0, offset); else n.start(t0, offset, Math.max(0, le - offset));
    nodes[c] = n;
  };
  make('mix', S.mixBuf, a, b, pos);
  if (S.drumBuf) make('drums', S.drumBuf, a, b, pos);
  const items = patternItems(a, b);
  if (items.length) { patA = a; make('pattern', renderRegion(ctx, a, b, items, S.swing), 0, b - a, pos - a); }
  nodes.mix.onended = () => { if (playing && !S.loop) stopPlay(); };
  posNow = pos; lastCtxT = t0; loopA = a; loopB = b;
  playing = true; $('play').textContent = '■ Stop'; $('gridPlay').textContent = '■'; $('gridPlay').classList.add('on');
  if (!S.drumBuf && S.mixer.drums.on) status('Drums are silent until you extract them.');
  cancelAnimationFrame(raf);
  const tick = () => {
    if (!playing) return;
    const now = ctx.currentTime;
    if (now > lastCtxT) { posNow += now - lastCtxT; lastCtxT = now; }
    if (S.loop && posNow >= loopB) posNow = loopA + ((posNow - loopB) % (loopB - loopA));
    if (!S.loop) posNow = Math.min(posNow, loopB);
    S.playhead = posNow;
    drawRuler(); drawWave($('wMix'), S.mixMono, '#9aa3b5'); drawWave($('wDrums'), S.drumMono, '#ec8a2c', S.extracted);
    drawGridHead(posNow - (ctx.outputLatency || 0));
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- export
// Export: on iPad/iPhone, a bmydrummer:// link opens BMY straight away (needs BMY's Groove Lab link patch);
// File… shares or saves the .bmybank file.
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
function makeBank() {
  const pats = S.marks.filter(i => S.patterns.has(i)).map(i => ({ name: `Bar ${barNo(i)}`, ...S.patterns.get(i) }));
  if (!pats.length) return null;
  const bank = buildBank({ name: $('bankName').value || S.name, patterns: pats, swing: S.swing, bpm: S.grid.bpm });
  return { bank, pats, fname: bankFileName(bank) };
}
const b64url = str => { const bytes = new TextEncoder().encode(str); let bin = ''; for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000)); return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); };
const linkOn = () => { try { return localStorage.getItem('bmyGLLink') === '1'; } catch (e) { return false; } };
function showLinkMode() { const on = isIOS && linkOn(); $('exportFile').classList.toggle('hidden', !on); $('linkMode').checked = linkOn(); $('linkRow').classList.toggle('hidden', !isIOS); }
$('linkMode').onchange = () => { try { localStorage.setItem('bmyGLLink', $('linkMode').checked ? '1' : '0'); } catch (e) {} showLinkMode(); };
showLinkMode();
$('export').onclick = () => {
  if (!isIOS || !linkOn()) return exportFile();
  try {
    const x = makeBank(); if (!x) return;
    const link = `bmydrummer://import?name=${encodeURIComponent(x.fname.replace(/\.bmybank$/, ''))}#${b64url(JSON.stringify(x.bank))}`;
    log(`Sent "${x.fname}" to BMY by link: ${x.pats.length} patterns, ${x.pats.reduce((n, p) => n + p.cells.length, 0)} cells, link ${(link.length / 1024).toFixed(1)} KB`);
    status('Opening BMY Drummer… (if nothing happens, BMY doesn’t have the Groove Lab link update: use File…, or turn the link off in the Log panel)');
    location.href = link;
  } catch (err) { status('Export failed: ' + err.message); log('ERROR ' + errText(err)); }
};
$('exportFile').onclick = () => exportFile();
async function exportFile() {
  try {
    const x = makeBank(); if (!x) return;
    const { bank, pats, fname } = x;
    const json = JSON.stringify(bank, null, 2);
    let file = new File([json], fname, { type: 'application/x-bmybank' });
    if (navigator.canShare && !navigator.canShare({ files: [file] })) file = new File([json], fname, { type: 'application/octet-stream' });
    log(`Exported "${fname}" (${file.type}): ${pats.length} patterns, ${pats.reduce((n, p) => n + p.cells.length, 0)} cells, ${bank.tempo ? bank.tempo.bpm + ' BPM' : 'no tempo'}`);
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      try { await navigator.share({ files: [file], title: fname }); status(`Shared ${fname}.`); return; }
      catch (e) { if (e.name === 'AbortError') return; log('Share failed, downloading instead: ' + e.message); }
    }
    const url = URL.createObjectURL(file), a = document.createElement('a');
    a.href = url; a.download = fname; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    status(`Saved ${fname} to Downloads. Tap it in Files to import it into BMY.`);
  } catch (err) { status('Export failed: ' + err.message); log('ERROR ' + errText(err)); }
}

$('bankName').addEventListener('input', scheduleSave);
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') saveNow(); });
$('saveStem').onclick = async () => {
  if (!S.drumBuf || !S.extracted.length) { status('No drums yet: extract some bars first.'); return; }
  const a = Math.min(...S.extracted.map(r => r[0])), b = Math.max(...S.extracted.map(r => r[1]));
  const s0 = Math.floor(a * SR), s1 = Math.floor(b * SR), ctx = audioCtx(), buf = ctx.createBuffer(2, s1 - s0, SR);
  for (let c = 0; c < 2; c++) buf.getChannelData(c).set(S.drumBuf.getChannelData(c).subarray(s0, s1));
  const fname = `${S.name} drums ${a.toFixed(1)}-${b.toFixed(1)}s.wav`;
  const file = new File([wavBytes(buf)], fname, { type: 'audio/wav' });
  log(`Saved stem ${fname}; grid ${S.grid.bpm} BPM, bar lines at ${S.bars.filter(x => x.end > a && x.start < b).map(x => (x.start - a).toFixed(3)).join(', ')} s into the file`);
  if (navigator.canShare && navigator.canShare({ files: [file] })) { try { await navigator.share({ files: [file], title: fname }); return; } catch (e) { if (e.name === 'AbortError') return; } }
  const url = URL.createObjectURL(file), el = document.createElement('a'); el.href = url; el.download = fname; document.body.appendChild(el); el.click(); el.remove();
};
// ---------------------------------------------------------------- Exam: ADTOF vs BMY on this song's drum stem
// ADTOF is the reference here only (personal testing): where both agree, take it as right; where they
// disagree, Matt listens. Exports a readable report, the raw hits (JSON) and the stem (mono WAV).
const EX_TOL = 0.05, EX5 = ['kick', 'snare', 'tom', 'hihat', 'cymbal'];
function exMatch(a, b) {                              // one-to-one within 50 ms -> [pairs, onlyA, onlyB]
  const used = new Set(), pairs = [], onlyA = [];
  for (const x of a) {
    let best = -1, bd = EX_TOL + 1;
    b.forEach((y, j) => { const d = Math.abs(y.time - x.time); if (!used.has(j) && d <= EX_TOL && d < bd) { best = j; bd = d; } });
    if (best < 0) onlyA.push(x); else { used.add(best); pairs.push([x, b[best]]); }
  }
  return [pairs, onlyA, b.filter((_, j) => !used.has(j))];
}
function exWhere(t) {                                 // "bar 12, beat 2&" from the song's grid
  const i = barAt(t);
  if (i < 0) return { bar: null, label: `${t.toFixed(2)} s` };
  const b = S.bars[i], pos = (t - b.start) / b.step, q = Math.round(pos), beat = Math.floor(q / 4) + 1, sub = ((q % 4) + 4) % 4;
  const off = Math.abs(pos - q) > 0.3 ? '~' : '';
  return { bar: barNo(i), label: `bar ${String(barNo(i)).padStart(3)}  beat ${beat}${['', 'e', '&', 'a'][sub] || ''}${off}` };
}
function exLevel(mono, t) {                           // loudness around the hit, dB below the stem's peak
  const i0 = Math.max(0, Math.floor((t - 0.005) * SR)), i1 = Math.min(mono.length, Math.floor((t + 0.05) * SR));
  let m = 0; for (let i = i0; i < i1; i++) m = Math.max(m, Math.abs(mono[i]));
  return m;
}
async function runExam() {
  if (!S.drumMono || !S.extracted.length) { status('No drums yet: extract some bars first.'); return; }
  if (!(await bmytAvailable())) { status('The BMY transcriber model is not on the server.'); return; }
  busy = true; updateButtons();
  try {
    const o = await getOrt();
    await loadAdtof(o, 'models/adtof.onnx'); await loadBmyt(o);
    const regions = S.extracted.map(r => [...r]).sort((x, y) => x[0] - y[0]);
    const A = [], B = [];
    let peakAll = 0;
    for (const [k, [a, b]] of regions.entries()) {
      status(`Exam: transcribing region ${k + 1} of ${regions.length} with ADTOF and BMY…`, k / regions.length);
      const s0 = Math.floor(a * SR), s1 = Math.min(S.drumMono.length, Math.floor(b * SR)), raw = S.drumMono.subarray(s0, s1);
      let peak = 0; for (let i = 0; i < raw.length; i++) peak = Math.max(peak, Math.abs(raw[i]));
      peakAll = Math.max(peakAll, peak);
      const gain = peak > 0 ? Math.min(31.6, 0.891 / peak) : 1, mono = gain === 1 ? raw : raw.map(v => v * gain);
      const { act, frames } = await activations(o, mono, logSpec(mono));
      hitsFrom(act, frames).forEach(h => A.push({ time: +(h.time + a).toFixed(3), cls: h.cls, s: +h.strength.toFixed(3) }));
      (await bmytHits(o, mono)).forEach(h => B.push({ time: +(h.time + a).toFixed(3), cls: h.cls, model: h.model, p: +h.p.toFixed(3) }));
      await new Promise(r => setTimeout(r));
    }
    const db = t => { const m = exLevel(S.drumMono, t); return m > 0 && peakAll > 0 ? 20 * Math.log10(m / peakAll) : -99; };
    const rows = [], sum = {};
    for (const c of EX5) {
      const [pairs, aOnly, bOnly] = exMatch(A.filter(h => h.cls === c), B.filter(h => h.cls === c));
      sum[c] = { agree: pairs.length, adtofOnly: aOnly.length, bmyOnly: bOnly.length };
      aOnly.forEach(h => rows.push({ t: h.time, drum: c, who: 'ADTOF only', other: B.filter(x => Math.abs(x.time - h.time) <= EX_TOL).map(x => x.model) }));
      bOnly.forEach(h => rows.push({ t: h.time, drum: c, who: 'BMY only', bmy: h.model, other: A.filter(x => Math.abs(x.time - h.time) <= EX_TOL).map(x => x.cls) }));
    }
    rows.sort((x, y) => x.t - y.t);
    // BMY's finer classes on the agreed hits (ADTOF can't check these: for the record)
    const fine = B.reduce((m, h) => ((m[h.model] = (m[h.model] || 0) + 1), m), {});
    const pct = x => (x.agree + x.adtofOnly + x.bmyOnly ? Math.round(100 * x.agree / (x.agree + x.adtofOnly + x.bmyOnly)) : 100);
    const tot = EX5.reduce((t, c) => ({ agree: t.agree + sum[c].agree, adtofOnly: t.adtofOnly + sum[c].adtofOnly, bmyOnly: t.bmyOnly + sum[c].bmyOnly }), { agree: 0, adtofOnly: 0, bmyOnly: 0 });
    const L = [];
    L.push(`BMY GROOVE LAB EXAM: ADTOF vs BMY`, `Song: ${S.name}`, `Groove Lab ${BUILD}; BMY model ${BMYT_V}; ${S.grid.bpm.toFixed(2)} BPM; stem ${regions.map(([a, b]) => `${a.toFixed(1)}–${b.toFixed(1)} s`).join(', ')}`,
      `Match = same drum within ${EX_TOL * 1000} ms. Drums compared as ADTOF's 5 (toms, hats, cymbals each one).`, '',
      `Agreement overall: ${pct(tot)}%  (both ${tot.agree}, ADTOF only ${tot.adtofOnly}, BMY only ${tot.bmyOnly})`, '',
      `drum      both   ADTOF only   BMY only   agree`);
    for (const c of EX5) L.push(`${c.padEnd(8)}${String(sum[c].agree).padStart(6)}${String(sum[c].adtofOnly).padStart(13)}${String(sum[c].bmyOnly).padStart(11)}${String(pct(sum[c])).padStart(7)}%`);
    L.push('', `BMY's own detail: ${Object.entries(fine).map(([k, v]) => `${k} ${v}`).join(', ')}`, '',
      `LISTEN LIST (${rows.length}): each disagreement, with what the other one heard there and how loud the stem is.`,
      `Mark each line A (ADTOF right), B (BMY right) or X (both wrong) if you like, and upload this file with the .json and .wav.`, '');
    let lastBar = null;
    for (const r of rows) {
      const w = exWhere(r.t);
      if (w.bar !== lastBar && lastBar !== null) L.push('');
      lastBar = w.bar;
      const other = r.other.length ? `(other heard: ${[...new Set(r.other)].join('+')})` : '(other heard nothing)';
      L.push(`${w.label}  ${r.t.toFixed(3).padStart(8)} s  ${r.drum.padEnd(7)} ${r.who.padEnd(10)}${r.bmy ? ` [${r.bmy}]` : ''}  ${db(r.t).toFixed(0).padStart(4)} dB  ${other}   ?`);
    }
    const base = `${S.name} exam ${new Date().toISOString().slice(0, 10)}`;
    const json = { song: S.name, build: BUILD, bmyModel: BMYT_V, bpm: S.grid.bpm, bars: S.bars.map(b => +b.start.toFixed(4)), barSec: barSec(),
      firstBar: S.bars.findIndex(b => b.start >= -0.1), regions, tol: EX_TOL, adtof: A, bmy: B, summary: sum, stem: `${base}.wav`, stemStart: regions[0][0] };
    const a0 = regions[0][0], a1 = regions[regions.length - 1][1], s0 = Math.floor(a0 * SR), s1 = Math.min(S.drumMono.length, Math.floor(a1 * SR));
    const buf = audioCtx().createBuffer(1, s1 - s0, SR); buf.getChannelData(0).set(S.drumMono.subarray(s0, s1));
    const files = [new File([L.join('\n')], `${base}.txt`, { type: 'text/plain' }), new File([JSON.stringify(json)], `${base}.json`, { type: 'application/json' }),
      new File([wavBytes(buf)], `${base}.wav`, { type: 'audio/wav' })];
    log(L.slice(0, 13).join('\n'));
    examFiles = { files, base };
    $('examBtn').textContent = 'Save exam (3 files)';
    status(`Exam done: ${pct(tot)}% agreement, ${rows.length} spots to listen to. Tap "Save exam" to share the report, hits and stem.`, 1);
  } catch (err) { status('Exam failed: ' + (err.message || err)); log('ERROR exam ' + errText(err)); }
  finally { busy = false; updateButtons(); }
}
let examFiles = null;
async function saveExam() {                         // a fresh tap: Safari only shares straight after a user gesture
  const { files, base } = examFiles;
  examFiles = null; $('examBtn').textContent = 'Exam';
  if (navigator.canShare && navigator.canShare({ files })) { try { await navigator.share({ files, title: base }); return; } catch (e) { if (e.name === 'AbortError') return; log('share failed: ' + errText(e)); } }
  for (const f of files) { const url = URL.createObjectURL(f), el = document.createElement('a'); el.href = url; el.download = f.name; document.body.appendChild(el); el.click(); el.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000); }
  status(`Saved ${base} (.txt, .json, .wav) to Downloads.`);
}
$('examBtn').onclick = () => (examFiles ? saveExam() : runExam());
$('closeLog').onclick = () => { $('logBox').open = false; };
$('copyLog').onclick = async () => { try { await navigator.clipboard.writeText($('log').textContent); $('copyLog').textContent = 'Copied'; } catch (e) { $('copyLog').textContent = 'Copy failed'; } };

// ---------------------------------------------------------------- Perform: clap / tap / beatbox over a click
const P = {
  bpm: 100, bars: 2, mode: 'lane', lane: 'snare', overdub: true, clickVol: 0.7,
  latency: null, sounds: null, takes: [], key: null, name: null, rec: null, taps: [], stream: null, base: null, clock: null, follow: null, chain: null, gainDb: 0
};
try { const st = JSON.parse(localStorage.getItem('bmyGLPerform') || '{}'); Object.assign(P, { gainDb: st.gainDb || 0, latency: st.latency ?? null, bpm: st.bpm || 100, bars: st.bars || 2, clickVol: st.clickVol ?? 0.7 });
  if (st.sounds) P.sounds = st.sounds.map(x => ({ ...x, centroid: x.centroid && Float32Array.from(x.centroid) })); } catch (e) {}
if (!P.sounds) P.sounds = [{ name: 'Kick', lane: 'kick' }, { name: 'Snare', lane: 'snare' }, { name: 'Hat', lane: 'closedHat' }];
const savePerfPrefs = () => { try { localStorage.setItem('bmyGLPerform', JSON.stringify({ gainDb: P.gainDb || 0, latency: P.latency, bpm: P.bpm, bars: P.bars, clickVol: P.clickVol,
  sounds: P.sounds.map(x => ({ name: x.name, lane: x.lane, centroid: x.centroid ? Array.from(x.centroid) : null, n: x.n || 0 })) })); } catch (e) {} };
const pStatus = m => { $('pStatus').textContent = m; };
const defaultLatency = () => { const c = audioCtx(); return (c.outputLatency || 0) + (c.baseLatency || 0) + 0.015; };
const latency = () => (P.latency ?? defaultLatency());

function showPerf() {
  $('pGain').value = P.gainDb || 0; $('pGainV').textContent = gainLabel(P.gainDb || 0);
  $('pBpm').value = P.bpm; $('pClick').value = Math.round(P.clickVol * 100);
  document.querySelectorAll('#pBars button').forEach(b => b.classList.toggle('on', +b.dataset.n === P.bars));
  document.querySelectorAll('#pMode button').forEach(b => b.classList.toggle('on', b.dataset.m === P.mode));
  $('pLane').classList.toggle('hidden', P.mode !== 'lane'); $('pTeachBox').classList.toggle('hidden', P.mode !== 'beatbox');
  if (!$('pLane').options.length) $('pLane').innerHTML = LANES.map(l => `<option value="${l.role}">${l.name}</option>`).join('');
  $('pLane').value = P.lane;
  $('pSounds').innerHTML = P.sounds.map((x, i) => `<span class="sound ${x.centroid ? 'ok' : ''}"><b>${esc(x.name)}</b>
      <select data-i="${i}" aria-label="${esc(x.name)} lane">${LANES.map(l => `<option value="${l.role}" ${l.role === x.lane ? 'selected' : ''}>${l.name}</option>`).join('')}</select>
      <button data-teach="${i}">${x.centroid ? 'Re-teach' : 'Teach'}</button></span>`).join('');
  $('pSounds').querySelectorAll('select').forEach(el => el.onchange = () => { P.sounds[+el.dataset.i].lane = el.value; savePerfPrefs(); });
  $('pSounds').querySelectorAll('[data-teach]').forEach(el => el.onclick = () => teachSound(+el.dataset.teach));
  $('pLat').textContent = `${Math.round(latency() * 1000)} ms ${P.latency === null ? '(est.)' : ''}`;
  $('pUndo').disabled = !P.takes.length || !!P.rec; $('pClear').disabled = !P.takes.length || !!P.rec;
}
$('perfBtn').onclick = () => {
  const el = $('perf');
  if (!el.classList.contains('hidden')) { if (!P.rec) { stopFollow(); stopClock(); micRelease(); el.classList.add('hidden'); $('perfBtn').setAttribute('aria-expanded', 'false'); } return; }
  closeRecent(); stopPlay(); el.classList.remove('hidden'); $('perfBtn').setAttribute('aria-expanded', 'true'); showPerf();
  try { navigator.permissions.query({ name: 'microphone' }).then(p => { if (p.state === 'granted' && !$('perf').classList.contains('hidden')) micChain().catch(() => {}); }, () => {}); } catch (e) {}
};
$('pBpm').onchange = () => { P.bpm = Math.max(40, Math.min(240, +$('pBpm').value || 100)); savePerfPrefs(); if (P.clock && !P.rec) { stopClock(); startClock(); } };
document.querySelectorAll('#pBars button').forEach(b => b.onclick = () => { if (P.takes.length && +b.dataset.n !== P.bars) { pStatus('Clear the takes first to change the number of bars.'); return; } P.bars = +b.dataset.n; savePerfPrefs(); showPerf(); });
document.querySelectorAll('#pMode button').forEach(b => b.onclick = () => { P.mode = b.dataset.m; showPerf(); });
$('pLane').onchange = () => { P.lane = $('pLane').value; };
$('pClick').oninput = () => { P.clickVol = +$('pClick').value / 100; savePerfPrefs(); if (P.clock) P.clock.gain.gain.value = P.clickVol; };
$('pGain').oninput = () => { P.gainDb = +$('pGain').value; $('pGainV').textContent = gainLabel(P.gainDb); if (P.chain) P.chain.gain.gain.value = Math.pow(10, P.gainDb / 20); savePerfPrefs(); };
$('pLvl').onclick = () => { if (!P.chain) micChain().catch(err => pStatus('Microphone not available: ' + err.message)); };
$('pLatDn').onclick = () => { P.latency = Math.max(0, latency() - 0.005); savePerfPrefs(); showPerf(); };
$('pLatUp').onclick = () => { P.latency = latency() + 0.005; savePerfPrefs(); showPerf(); };

// Tap tempo
$('pTap').onclick = () => {
  const now = performance.now() / 1000;
  P.taps = P.taps.filter(t => now - t < 2.5); P.taps.push(now);
  if (P.taps.length >= 3) { const b = tempoFromOnsets(P.taps); if (b) { P.bpm = b; savePerfPrefs(); $('pBpm').value = b; if (P.clock && !P.rec) { stopClock(); startClock(); } pStatus(`Tap tempo: ${b} BPM (${P.taps.length} taps)`); } }
  else pStatus('Keep tapping…');
};

// mic + recorder shared by Listen, Teach and takes
async function micChain() {
  const ctx = audioCtx();
  if (!P.stream || P.stream.getAudioTracks().every(t => t.readyState === 'ended')) {
    P.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    await ctx.audioWorklet.addModule('js/rec-worklet.js');
    P.chain = null;
  }
  if (!P.chain) {
    const src = ctx.createMediaStreamSource(P.stream), gain = ctx.createGain(), an = ctx.createAnalyser();
    gain.gain.value = Math.pow(10, (P.gainDb || 0) / 20); an.fftSize = 2048;
    src.connect(gain); gain.connect(an);
    const ch = P.chain = { src, gain, an };
    runMeter('pLvl', an, () => P.chain === ch);
  }
  return P.chain;
}
function micRelease() {                     // leaving Perform: give the mic back (unless something still needs it)
  if (P.rec || P.follow) return;
  const ch = P.chain; P.chain = null;
  if (ch) { try { ch.src.disconnect(); ch.gain.disconnect(); } catch (e) {} }
  if (P.stream) { P.stream.getTracks().forEach(t => t.stop()); P.stream = null; }
}
async function micOpen() {
  const ctx = audioCtx(), ch = await micChain();
  const node = new AudioWorkletNode(ctx, 'bmy-recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
  const mute = ctx.createGain(); mute.gain.value = 0;
  ch.gain.connect(node); node.connect(mute); mute.connect(ctx.destination);
  const r = { src: ch.gain, node, mute, chunks: [], frame0: null, frames: 0 };
  node.port.onmessage = e => { if (r.frame0 === null) r.frame0 = e.data.frame; r.chunks.push(e.data.ch[0]); r.frames += e.data.ch[0].length; };
  return r;
}
function micClose(r) {
  try { r.node.port.postMessage('stop'); } catch (e) {}
  try { r.src.disconnect(r.node); } catch (e) {} r.node.disconnect(); r.mute.disconnect();
  const mono = new Float32Array(r.frames); let o = 0; for (const c of r.chunks) { mono.set(c, o); o += c.length; }
  return { mono, frame0: r.frame0 ?? 0 };
}
const sleep = ms => new Promise(res => setTimeout(res, ms));
async function listenFor(secs, label) {
  const r = await micOpen(); const t0 = performance.now();
  while ((performance.now() - t0) / 1000 < secs) { pStatus(`${label} ${Math.ceil(secs - (performance.now() - t0) / 1000)}…`); await sleep(200); }
  return micClose(r);
}
// Listen = follow: keeps listening, finds the tempo and where the beats fall, starts the click on your beat and keeps it in step.
// Holds still while a take records (the take needs a fixed grid).
const F_WIN = 8, F_EVERY = 400;
function followTempo(ts, guess) {
  // circular fit on the 8th-note grid; ±8 % around the guess (or the plain median-interval tempo when there is none)
  let best = null;
  const lo = guess ? guess * 0.92 : 60, hi = guess ? guess * 1.08 : 180;
  for (let bpm = lo; bpm <= hi; bpm += 0.1) {
    const p = 30 / bpm; let re = 0, im = 0;
    for (const t of ts) { const a = 2 * Math.PI * t / p; re += Math.cos(a); im += Math.sin(a); }
    const R = Math.hypot(re, im) / ts.length;
    if (!best || R > best.R) best = { bpm, R };
  }
  return best;
}
function startFollow() {
  return micOpen().then(r => {
    const F = { r, dropped: 0, timer: null, locked: false, lastN: 0 };
    F.timer = setInterval(() => followTick(F), F_EVERY);
    P.follow = F; showListen(); pStatus('Listening… clap, tap or play; the click will join you.');
  });
}
function stopFollow(msg) {
  const F = P.follow; if (!F) return; P.follow = null; clearInterval(F.timer);
  try { micClose(F.r); } catch (e) {}
  showListen(); if (msg) pStatus(msg);
}
function showListen() { const on = !!P.follow; $('pListen').classList.toggle('live', on); $('pListen').setAttribute('aria-pressed', String(on)); $('pListen').textContent = on ? '● Listening' : 'Listen'; }
function followTick(F) {
  const r = F.r, ctx = audioCtx(), sr = ctx.sampleRate;
  // keep the last F_WIN seconds
  while (r.chunks.length > 1 && r.frames - F.dropped - r.chunks[0].length > F_WIN * sr) { F.dropped += r.chunks[0].length; r.chunks.shift(); }
  if (P.rec || r.frame0 === null) return;
  const n = r.frames - F.dropped; if (n < sr * 1.5) return;
  const mono = new Float32Array(n); let o = 0; for (const c of r.chunks) { mono.set(c, o); o += c.length; }
  const lat = latency(), base = (r.frame0 + F.dropped) / sr;
  const on = detectOnsets(mono).filter(x => !looksLikeClick(x.feat)).map(x => base + x.t - lat);   // when you actually played (audio clock)
  const now = ctx.currentTime, recent = on.filter(t => now - t < F_WIN);
  if (recent.length < 4) { if (F.locked && recent.length === 0) pStatus(`Listening… (${P.bpm} BPM)`); return; }
  if (now - recent[recent.length - 1] > 2.5) return;                                    // gone quiet: leave the click alone
  let guess = F.locked ? P.bpm : tempoFromOnsets(recent);
  if (!guess) return;
  const fit = followTempo(recent, guess);
  if (!fit || fit.R < 0.55) { pStatus(`Listening… not steady yet (${recent.length} hits)`); return; }
  const bpm = Math.round((F.locked ? P.bpm + 0.5 * (fit.bpm - P.bpm) : fit.bpm) * 10) / 10;
  const beat = 60 / bpm;
  // which of your hits sit on beats: phase of the quarter grid from hits near it
  let re = 0, im = 0; for (const t of recent) { const a = 2 * Math.PI * t / beat; re += Math.cos(a); im += Math.sin(a); }
  const ph = ((Math.atan2(im, re) / (2 * Math.PI)) * beat + beat) % beat;             // your beats fall at ph + k*beat
  P.bpm = bpm; $('pBpm').value = bpm; savePerfPrefs();
  let C = P.clock;
  if (!C) {
    let t = ph + Math.ceil((now + 0.25 - ph) / beat) * beat;                             // your next beat
    C = startClock(t); F.locked = true;
    pStatus(`Locked on: ${bpm} BPM. The click is with you. Tap Record take when ready.`); return;
  }
  // nudge the running click: tempo, then phase (half the error each time)
  const nextT = C.t0 + C.next * C.beat;
  let err = ((ph - nextT) % beat + beat) % beat; if (err > beat / 2) err -= beat;
  const corr = Math.abs(err) > 0.008 ? 0.5 * err : 0;
  C.beat = beat; C.bar = 4 * beat; C.bpm = bpm; C.t0 = nextT + corr - C.next * beat;
  F.locked = true;
  pStatus(`Following: ${bpm} BPM${Math.abs(err) > 0.015 ? ` · nudging ${Math.round(err * 1000)} ms` : ' · in step'}`);
}
$('pListen').onclick = async () => {
  if (P.follow) { stopFollow(`Stopped listening. Click stays at ${P.bpm} BPM.`); return; }
  try { await startFollow(); } catch (err) { pStatus('Microphone not available: ' + err.message); }
};
async function teachSound(i) {
  try {
    const sd = P.sounds[i];
    const { mono } = await listenFor(5, `Make your ${sd.name} sound 4–6 times, with gaps…`);
    const on = detectOnsets(mono);
    if (on.length < 2) { pStatus(`Only heard ${on.length}. Try again, a little louder.`); return; }
    sd.centroid = centroid(on.map(o => o.feat)); sd.n = on.length; savePerfPrefs(); showPerf();
    // how distinct are the taught sounds?
    const taught = P.sounds.filter(x => x.centroid);
    let worst = ''; for (let a = 0; a < taught.length; a++) for (let b = a + 1; b < taught.length; b++) {
      let c = 0; for (let k = 0; k < taught[a].centroid.length; k++) c += taught[a].centroid[k] * taught[b].centroid[k];
      if (c > 0.85) worst = `${taught[a].name} and ${taught[b].name} sound very alike; make them more different.`;
    }
    pStatus(`Learned ${sd.name} from ${on.length} hits. ${worst}`);
  } catch (err) { pStatus('Microphone not available: ' + err.message); }
}

// Record a take: 1 bar count-in, then loop the bars until Stop. Every pass is layered (folded) into the loop.
// The click: a free-running clock that keeps going between takes (Click button / closing the panel stops it)
function startClock(at) {
  const ctx = audioCtx(), clicks = clickBuffers(ctx), beat = 60 / P.bpm;
  const gain = ctx.createGain(); gain.gain.value = P.clickVol; gain.connect(ctx.destination);
  const C = { t0: at ?? ctx.currentTime + 0.4, beat, bar: 4 * beat, gain, next: 0, sched: [], bpm: P.bpm };
  const schedule = () => {
    while (C.t0 + C.next * C.beat < ctx.currentTime + 0.12) {
      const tt = C.t0 + C.next * C.beat;
      const n = ctx.createBufferSource(); n.buffer = C.next % 4 === 0 ? clicks.accent : clicks.beat; n.connect(gain); n.start(Math.max(ctx.currentTime, tt));
      C.sched.push(tt); if (C.sched.length > 4000) C.sched.splice(0, 2000); C.next++;
    }
  };
  schedule(); C.timer = setInterval(schedule, 25);
  P.clock = C; showClock();
  return C;
}
function stopClock() { if (!P.clock) return; clearInterval(P.clock.timer); try { P.clock.gain.disconnect(); } catch (e) {} P.clock = null; showClock(); }
function showClock() { const on = !!P.clock; $('pClickBtn').classList.toggle('on', on); $('pClickBtn').textContent = on ? '■ Click' : '▶︎ Click'; }
$('pClickBtn').onclick = () => { if (P.rec) return; if (P.clock) { stopFollow(); stopClock(); } else startClock(); };

// Record a take. Click not running: start it with a 1-bar count-in. Click running: start on the next bar line.
$('pRec').onclick = async () => {
  if (P.rec) return stopTake();
  if (P.mode === 'beatbox' && !P.sounds.some(x => x.centroid)) { pStatus('Teach at least one sound first (Teach buttons).'); return; }
  if (P.key && S.key !== P.key) Object.assign(P, { takes: [], key: null, base: null });   // a normal song is open: start a new Perform song
  try {
    const ctx = audioCtx(); stopPlay();
    const r = await micOpen();
    const recStart = ctx.currentTime;
    if (P.clock && Math.abs(P.clock.bpm - P.bpm) > 1e-6) stopClock();   // tempo changed
    let C = P.clock, bar1;
    if (!C) { C = startClock(); bar1 = C.t0 + C.bar; }                 // count-in bar
    else { const k = Math.ceil((recStart + 0.25 - C.t0) / C.bar); bar1 = C.t0 + k * C.bar; }   // next bar line (at least 0.25 s away)
    P.rec = { r, recStart, bar1, C, timer: setInterval(() => {
      const t = ctx.currentTime;
      const msg = t < bar1 ? `Get ready… ${Math.min(4, Math.ceil((bar1 - t) / C.beat))}` : `Recording · pass ${Math.floor((t - bar1) / (C.bar * P.bars)) + 1} · bar ${Math.floor(((t - bar1) % (C.bar * P.bars)) / C.bar) + 1} of ${P.bars}`;
      pStatus(msg);
    }, 100) };
    $('pRec').classList.add('on'); $('pRec').textContent = '■ Stop'; showPerf();
  } catch (err) { P.rec = null; pStatus('Could not start: ' + (err.name === 'NotAllowedError' ? 'microphone access was not allowed.' : err.message)); }
};
async function stopTake() {
  const R = P.rec; P.rec = null; clearInterval(R.timer);
  const C = R.C; R.beat = C.beat; R.bar = C.bar;                              // the click keeps running
  $('pRec').classList.remove('on'); $('pRec').textContent = '● Record take';
  const { mono, frame0 } = micClose(R.r);
  const sr = audioCtx().sampleRate, tOf = i => (frame0 + i) / sr;          // recording sample -> audio clock
  const on = detectOnsets(mono).map(o => ({ ...o, T: tOf(o.t * sr) }));
  // latency: the count-in clicks heard back through the mic (speaker) -> round-trip delay
  const offs = [];
  for (const ct of C.sched.filter(t => t >= R.recStart && t < R.bar1 - 0.05)) { const o = on.find(x => x.T - ct > -0.01 && x.T - ct < 0.3 && looksLikeClick(x.feat)); if (o) offs.push(o.T - ct); }   // clicks before the take starts; only the click itself, not claps
  offs.sort((a, b) => a - b);
  if (offs.length >= 3 && offs[offs.length - 1] - offs[0] < 0.02) { P.latency = offs[Math.floor(offs.length / 2)]; savePerfPrefs(); log(`Perform latency measured from the click: ${(P.latency * 1000).toFixed(0)} ms`); }
  const lat = latency(), step = R.beat / 4, loopSteps = 16 * P.bars, loopSec = R.bar * P.bars;
  const clickTimes = C.sched.filter(t => t >= R.recStart - 0.1).map(t => t + lat);
  const hits = [];
  let clicksIgnored = 0, unsure = 0;
  for (const o of on) {
    const T = o.T - lat;
    if (T < R.bar1 - step / 4) continue;                                            // count-in
    if (clickTimes.some(c => Math.abs(o.T - c) < 0.03) && looksLikeClick(o.feat)) { clicksIgnored++; continue; }
    let pos = (T - R.bar1) / step; pos = ((pos + 0.25) % loopSteps + loopSteps) % loopSteps - 0.25;   // fold passes into the loop
    let lane = P.lane;
    if (P.mode === 'beatbox') { const c = classify(o.feat, P.sounds); if (c.i < 0) continue; if (c.score < 0.5) unsure++; lane = P.sounds[c.i].lane; }
    hits.push({ pos, lane, levelDb: o.levelDb });
  }
  // the take's audio, latency-corrected and folded into the loop (passes summed)
  const loopN = Math.round(loopSec * sr), audio = new Float32Array(loopN);
  const start = Math.round((R.bar1 + lat) * sr) - frame0;
  for (let i = Math.max(0, start); i < mono.length; i++) audio[(i - start) % loopN] += mono[i];
  const lanes = P.mode === 'lane' ? [P.lane] : [...new Set(P.sounds.filter(x => x.centroid).map(x => x.lane))];
  P.takes.push({ id: Date.now(), hits, audio, lanes });
  const passes = Math.max(1, Math.round((mono.length - start) / loopN));
  pStatus(`Layer ${P.takes.length}: ${hits.length} hits over ${passes} pass${passes > 1 ? 'es' : ''}${clicksIgnored ? `, ${clicksIgnored} click(s) ignored` : ''}${unsure ? `, ${unsure} unsure` : ''}. Latency ${Math.round(lat * 1000)} ms.`);
  log(`Perform layer ${P.takes.length}: ${hits.length} hits, lanes ${lanes.join('/')}`);
  await buildPerformSong();
}
$('pUndo').onclick = async () => {
  if (P.rec) return;
  P.takes.pop();
  if (!P.takes.length && !P.base) stopClock();      // nothing left: stop the click too
  await buildPerformSong();
  pStatus(P.takes.length ? `Removed the last layer; ${P.takes.length} left. The click keeps going: tap Record to redo it.` : 'All layers removed; click stopped.');
};
$('pClear').onclick = async () => { if (P.rec) return; stopClock(); P.takes = []; P.key = null; P.base = null; pStatus('Cleared; click stopped. Record a new take to start a new Perform song.'); showPerf(); };

/** Turn the takes into a song: bar lines known, all bars marked, hits in fixed lanes; then the usual grid / mixer / export. */
async function buildPerformSong() {
  const ctx = audioCtx(), sr = ctx.sampleRate, lead = 0.5, bar = 240 / P.bpm, step = bar / 16;
  const loopN = Math.round(bar * P.bars * sr), total = Math.round(lead * sr) + loopN + Math.round(0.5 * sr);
  const buf = ctx.createBuffer(2, total, sr), L = buf.getChannelData(0), R = buf.getChannelData(1);
  const layers = [P.base, ...P.takes.map(t => t.audio)].filter(Boolean);
  for (const a of layers) for (let i = 0; i < loopN && i < a.length; i++) { const j = Math.round(lead * sr) + i; L[j] += a[i]; R[j] += a[i]; }
  let pk = 0; for (let i = 0; i < total; i++) pk = Math.max(pk, Math.abs(L[i]));
  if (pk > 0.98) for (let i = 0; i < total; i++) { L[i] *= 0.98 / pk; R[i] *= 0.98 / pk; }
  if (!P.key) { const d = new Date(); P.key = 'perf-' + d.getTime(); P.name = `Perform ${d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`; }
  stopPlay();
  Object.assign(S, { key: P.key, name: P.name, dur: buf.duration, mixBuf: buf, drumBuf: buf, sel: null, playhead: 0, ovPeaks: null, undo: [], flux: [], locked: true, tab: null });
  S.mixMono = toMono(L, R); S.drumMono = S.mixMono; S.extracted = [[0, buf.duration]];
  S.grid = { bpm: P.bpm, t0: lead }; rebuildBars();
  const first = S.bars.findIndex(b => Math.abs(b.start - lead) < 1e-3);
  S.marks = [...Array(Math.min(6, P.bars)).keys()].map(k => first + k);
  S.hits = P.takes.flatMap(t => t.hits).map(h => ({ time: lead + h.pos * step, cls: 'perform', lane: h.lane, fixedLane: true, levelDb: h.levelDb, strength: 1 })).sort((a, b) => a.time - b.time);
  S.patterns = new Map(); buildPatterns(true);
  S.tab = S.marks[0] ?? null; S.sel = S.marks.length ? { a: S.bars[S.marks[0]].start, b: S.bars[S.marks[S.marks.length - 1]].end } : null;
  S.view.pps = Math.max(8, waveW() / (bar * P.bars + 1)); S.view.start = 0;
  $('songTitle').textContent = S.name; $('bankName').value = S.name; setBpmField();
  S.perf = { bpm: P.bpm, bars: P.bars, takes: P.takes.map(t => ({ id: t.id, lanes: t.lanes, hits: t.hits })) };
  updateButtons(); showLock(); renderTabs(); renderGrid(); redraw(); showPerf();
  try { await putFile({ key: S.key, bytes: wavBytes(buf), type: 'audio/wav' }); } catch (e) {}
  try { await putStems({ key: S.key, regions: [], perfTakes: P.takes.filter(t => t.audio).map(t => ({ id: t.id, audio: t.audio })), perfBase: P.base || null }); } catch (e) {}
  saveNow();
}

// ---------------------------------------------------------------- phone screens (Song / Pattern / Perform)
const phoneMQ = matchMedia('(max-width:700px), (max-height:500px)');
function setTab(t) {
  const app = $('app'), cur = app.dataset.tab || 'song';
  if (cur === 'perform' && t !== 'perform' && P.rec) return;          // finish the take first
  app.dataset.tab = t;
  document.querySelectorAll('#phoneTabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === t)));
  if (phoneMQ.matches) {
    const open = !$('perf').classList.contains('hidden');
    if (t === 'perform' && !open) $('perfBtn').click();
    if (t !== 'perform' && open) $('perfBtn').click();
  }
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
}
const phoneTab = t => { if (phoneMQ.matches) setTab(t); };
document.querySelectorAll('#phoneTabs button').forEach(b => b.onclick = () => setTab(b.dataset.tab));
$('app').dataset.tab = 'song';
if (phoneMQ.matches) $('rec').textContent = '● Rec';
$('status').onclick = () => $('status').classList.toggle('open');
setInterval(() => { const n = S.patterns ? S.patterns.size : 0; const el = $('tabCount'); const v = n ? String(n) : ''; if (el.textContent !== v) el.textContent = v; }, 700);

// ---------------------------------------------------------------- start
const BUILD = '2026-11-27f';
log('Groove Lab loaded, build ' + BUILD + ', separator ' + SEPARATOR_VERSION);
$('buildBtn').textContent = '⟳ ' + BUILD.slice(5);            // e.g. "⟳ 11-27e"
$('buildBtn').title = `Build ${BUILD}: tap to check for a new version and reload`;
$('buildBtn').onclick = async () => {
  if (busy) { status('Busy: try again when this job has finished.'); return; }
  status('Checking for a new version…');
  try { await saveNow(); } catch (e) {}
  let latest = null;
  try {
    const html = await (await fetch('index.html', { cache: 'no-store' })).text();
    latest = (html.match(/app\.js\?b=(\w+)/) || [])[1] || null;
  } catch (e) { status('Offline: can\'t check for a new version right now.'); return; }
  try {                                                       // drop the app shell and transcriber models (re-fetched), keep kits and the separator
    for (const k of await caches.keys()) {
      if (k.startsWith('gl-shell')) await caches.delete(k);
      else if (k.startsWith('gl-libs')) { const c = await caches.open(k); for (const r of await c.keys()) if (r.url.includes('/models/')) await c.delete(r); }
    }
    const reg = navigator.serviceWorker && await navigator.serviceWorker.getRegistration();
    if (reg) await reg.update().catch(() => {});
  } catch (e) { log('refresh: ' + errText(e)); }
  const cur = BUILD.replace(/-/g, '');
  status(latest && latest !== cur ? `Updating to build ${latest}…` : `Build ${BUILD} is the latest. Reloading fresh…`);
  setTimeout(() => location.reload(), 400);
};
// ---------------------------------------------------------------- installable app (Home Screen / Chrome install)
const standalone = matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
if ('serviceWorker' in navigator && !/[?&]nosw\b/.test(location.search) && !/[?&]nodemucs\b/.test(location.search))
  navigator.serviceWorker.register('sw.js').then(r => log('Offline copy ready' + (r.active ? '' : ' (first visit: kept from the next load)'))).catch(e => log('Offline copy not available: ' + e.message));
let installEvt = null;
addEventListener('beforeinstallprompt', e => { e.preventDefault(); installEvt = e; $('installBtn').classList.remove('hidden'); });
if (!standalone && isIOS) $('installBtn').classList.remove('hidden');
if (standalone) log('Running as an installed app');
$('installBtn').onclick = async () => {
  if (installEvt) { installEvt.prompt(); const r = await installEvt.userChoice; if (r.outcome === 'accepted') $('installBtn').classList.add('hidden'); installEvt = null; return; }
  alert('Install Groove Lab on this ' + (/iPhone|iPod/.test(navigator.userAgent) ? 'iPhone' : 'iPad') + ':\n\n1. Tap the Share button (square with an arrow) in Safari.\n2. Tap "Add to Home Screen" (scroll down if you don’t see it).\n3. Tap Add.\n\nThen open Groove Lab from its Home Screen icon. It runs full screen and works offline.\n\nNote: the installed app keeps its own songs, separate from Safari’s.');
};
addEventListener('appinstalled', () => { $('installBtn').classList.add('hidden'); log('Installed'); });

// transcriber: ADTOF (personal use only) or Magenta OaF Drums (candidate for BMY); remembered on this device
let bmyModel = false;
const transcriber = () => { let m = 'adtof'; try { m = localStorage.getItem('bmyGLModel') || 'adtof'; } catch (e) {} return m === 'bmy' && !bmyModel ? 'adtof' : ['adtof', 'oaf', 'bmy'].includes(m) ? m : 'adtof'; };
$('modelSel').value = transcriber();
bmytAvailable().then(ok => {                // BMY's own model appears once models/bmy_drums.onnx is in place
  bmyModel = ok;
  if (ok) { const o = document.createElement('option'); o.value = 'bmy'; o.textContent = 'BMY'; $('modelSel').appendChild(o); $('modelSel').value = transcriber(); }
});
$('modelSel').onchange = () => {
  try { localStorage.setItem('bmyGLModel', $('modelSel').value); } catch (e) {}
  const name = { oaf: 'Magenta', bmy: 'BMY', adtof: 'ADTOF' }[$('modelSel').value];
  log('Transcriber: ' + name);
  if (S.extracted && S.extracted.length && !busy) { status(`Transcribing again with ${name}…`); $('extract').click(); }
  else status(`Transcriber: ${name}. It's used the next time you Extract.`);
};

// preview kit: any of BMY's GeneralUser GS kits; remembered on this device
let kitPick = 0; try { kitPick = +(localStorage.getItem('bmyGLKit') || 0); } catch (e) {}
$('kitSel').innerHTML = KITS.map(([p, n]) => `<option value="${p}">${esc(n)}</option>`).join('');
$('kitSel').value = String(kitPick);
async function useKit(p) {
  $('kitSel').disabled = true;
  try {
    const k = await loadKit(audioCtx(), p); log('Preview kit: ' + k.name);
    try { localStorage.setItem('bmyGLKit', String(p)); } catch (e) {}
    if (playing) syncPlayback(true);
  } catch (e) { log('Preview kit failed: ' + e.message); status('Could not load that kit: ' + e.message); $('kitSel').value = String(kitPick); }
  finally { $('kitSel').disabled = false; }
  kitPick = +$('kitSel').value;
}
$('kitSel').onchange = () => { status('Loading the ' + $('kitSel').selectedOptions[0].textContent + ' kit…'); useKit(+$('kitSel').value).then(() => status(`Preview kit: ${$('kitSel').selectedOptions[0].textContent}. Tap a lane icon to hear a drum.`)); };
useKit(kitPick);
if (lastCrash) { log('LAST RUN STOPPED DURING: ' + lastCrash); status('The last run stopped during: ' + lastCrash + '. Open the Log for details.'); crumbDone(); }
updateButtons(); showLock(); renderGrid(); redraw();
