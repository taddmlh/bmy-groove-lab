// BMY Groove Lab: song -> drum stem (Demucs) -> hits (ADTOF) -> BMY patterns -> .bmybank
import { logSpec, toMono, FPS, NBANDS, BAND_FREQ } from './dsp.js';
import { loadAdtof, activations, hitsFrom, CLASSES } from './adtof.js';
import { detectGrid, barTimes } from './beats.js';
import { LANES, VEL, describeHits, assignLanes, detectSwing, barToPattern } from './transcribe.js';
import { buildBank, bankFileName } from './bank.js';
import { detectOnsets, centroid, classify, tempoFromOnsets, clickBuffers, looksLikeClick } from './perform.js';
import { loadDemucs, separateDrums } from './demucs.js';
import { renderRegion, loadKit, audition } from './synth.js';
import { putSong, getSong, putFile, getFile, hasFile, putStems, getStems, listSongs, deleteSong, askPersist } from './db.js';

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
const crumb = m => { try { localStorage.setItem('bmyGLStage', m); } catch (e) {} };
const crumbDone = () => { try { localStorage.removeItem('bmyGLStage'); } catch (e) {} };
let lastCrash = null; try { lastCrash = localStorage.getItem('bmyGLStage'); } catch (e) {}
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
  const vw = $('wMix').getBoundingClientRect().width / S.view.pps;
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
function handleHit(st) {          // grab the start or end of a bar-aligned selection
  if (!S.grid || S.locked || !selIsBars()) return false;
  const xa = (S.sel.a - S.view.start) * S.view.pps, xb = (S.sel.b - S.view.start) * S.view.pps;
  const n = Math.round((S.sel.b - S.sel.a) / barSec());
  if (Math.abs(st.x - xa) < 22) { Object.assign(st, { mode: 'start', a0: S.sel.a, n }); return true; }
  if (Math.abs(st.x - xb) < 22) { Object.assign(st, { mode: 'end', a0: S.sel.a, n }); return true; }
  return false;
}
function handleMove(st) {
  if (st.mode === 'start') {          // slide every bar line, tempo unchanged
    const a = Math.max(-barSec(), Math.min(S.dur, xToT(st.x)));
    S.grid.t0 = a; S.sel = { a, b: a + st.n * barSec() };
    gridChanged(null, true); S.sel = { a, b: a + st.n * barSec() }; redraw(); syncPlayback(true);
    status(`Bar start ${a.toFixed(3)} s — release to re-transcribe`);
  } else {                            // stretch: start stays, tempo follows the end
    const b = xToT(st.x), len = Math.max(0.2, b - st.a0) / st.n;
    S.grid.bpm = Math.max(30, Math.min(300, Math.round(240 / len * 100) / 100)); S.grid.t0 = st.a0;
    gridChanged(null, true); S.sel = { a: st.a0, b: st.a0 + st.n * barSec() }; redraw(); syncPlayback(true);
    status(`${S.grid.bpm.toFixed(2)} BPM — release to re-transcribe`);
  }
}
function handleUp(st) {
  gridChanged(st.mode === 'start' ? 'Bar lines moved. Drag the right edge to set the tempo.' : `Tempo set to ${S.grid.bpm.toFixed(2)} BPM.`);
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
  down: st => { st.t0 = xToT(st.x); st.mode = 'select'; handleHit(st); },
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
  const vw = $('wMix').getBoundingClientRect().width / S.view.pps;
  S.view.start = st.x / st.r.width * S.dur - vw / 2; clampView(); redraw();
}
function zoom(f) {
  if (!S.dur) return;
  const w = $('wMix').getBoundingClientRect().width, centre = S.view.start + w / S.view.pps / 2;
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
  $('export').disabled = !S.marks.some(i => S.patterns.has(i)) || busy;
  $('undo').disabled = !S.undo.length;
}

// ---------------------------------------------------------------- open song (Files, Recent, Record)
$('file').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  e.target.value = '';
  const bytes = await f.arrayBuffer();
  await openSong({ key: `${f.name}|${f.size}`, name: f.name.replace(/\.[^.]+$/, ''), bytes, type: f.type || 'audio' });
});
/** Decode, then restore saved work for this song or analyse it fresh. */
async function openSong({ key, name, bytes, type, buf }) {
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
    if (saved && saved.state) {
      await restoreState(saved.state);
      status(`Restored your work on "${name}": ${S.grid.bpm.toFixed(2)} BPM, ${S.marks.length} marked bar${S.marks.length === 1 ? '' : 's'}${S.patterns.size ? `, ${S.patterns.size} pattern${S.patterns.size > 1 ? 's' : ''}` : ''}.`, 1);
      log('Restored saved state for ' + key);
    } else {
      status('Finding the tempo and bar lines…', 0.1); await yieldUI();
      crumb('analysing tempo');
      S.mixSpec = await specAsync(S.mixMono, p => status('Finding the tempo and bar lines…', 0.1 + 0.85 * p));
      S.grid = detectGrid(S.mixSpec.data, S.mixSpec.frames); S.mixSpec = null;
      log(`Tempo ${S.grid.bpm} BPM, first bar line at ${S.grid.t0.toFixed(3)} s`);
      rebuildBars();
      S.view.pps = Math.max(8, $('wMix').getBoundingClientRect().width / (barSec() * 12));
      S.view.start = Math.max(-0.2, S.grid.t0 - 0.2);
      status(`${S.grid.bpm} BPM. Tap a bar and play it. If beat 1 is wrong use ◀︎ Beat / Beat ▶︎, or drag the orange grips on the selected bar in the bar-number strip (left = bar start, right = tempo). Two fingers scroll and pinch-zoom. Then tap bar numbers to mark up to 6 bars.`, 1);
    }
    setBpmField();
    crumbDone();
    try { if (!(await hasFile(key))) await putFile({ key, bytes, type }); } catch (e) { log('Could not keep the audio file: ' + e.message); }
    askPersist();
    saveNow();
  } catch (err) {
    crumbDone(); status('Could not open that song: ' + err.message); log('ERROR ' + (err.stack || err));
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
  if (stems && stems.regions && stems.regions.length) {
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
$('recentBtn').onclick = async () => {
  if (!$('recent').classList.contains('hidden')) return closeRecent();
  let list = [];
  try { list = await listSongs(); } catch (e) { status('Song memory is not available in this browser.'); return; }
  const el = $('recent');
  el.innerHTML = `<h3>Recent songs</h3>` + (list.length ? list.map(r => `
    <div class="item" data-key="${esc(r.key)}">
      <div class="meta"><b>${esc(r.name)}</b><small>${r.bpm ? r.bpm.toFixed(1) + ' BPM · ' : ''}${r.marks || 0} marked · ${r.patterns || 0} pattern${r.patterns === 1 ? '' : 's'} · ${ago(r.updated)}</small></div>
      <button data-act="open">Open</button><button data-act="del" title="Forget this song">✕</button>
    </div>`).join('') : '<div class="foot">No songs yet. Songs you open or record are kept here, with your work.</div>')
    + `<div class="foot">Kept in Safari on this iPad. Add the page to your Home Screen so iPadOS keeps it longer.</div>`;
  el.querySelectorAll('.item').forEach(it => {
    const key = it.dataset.key;
    it.querySelector('[data-act=open]').onclick = async () => {
      const f = await getFile(key).catch(() => null), meta = list.find(r => r.key === key);
      if (!f) { status('The audio for that song is no longer stored. Open it from Files again.'); return; }
      await openSong({ key, name: meta.name, bytes: f.bytes, type: f.type });
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

// ---------------------------------------------------------------- Record from the audio input
let rec = null;
async function listInputs() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
    const sel = $('inputSel'), cur = sel.value;
    sel.innerHTML = devs.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || 'Input ' + (i + 1))}</option>`).join('');
    if (cur) sel.value = cur;
    sel.classList.toggle('hidden', devs.length < 2);
  } catch (e) {}
}
$('rec').onclick = async () => {
  if (rec) return stopRecording();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { status('This browser cannot record audio.'); return; }
  stopPlay();
  const ctx = audioCtx();
  try {
    const dev = $('inputSel').value;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: {
      deviceId: dev ? { exact: dev } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 2 } });
    await listInputs();
    await ctx.audioWorklet.addModule('js/rec-worklet.js');
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'bmy-recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain(); mute.gain.value = 0;
    src.connect(node); node.connect(mute); mute.connect(ctx.destination);
    rec = { stream, src, node, mute, chunks: [[], []], frames: 0, t0: performance.now(), peak: 0, label: (stream.getAudioTracks()[0] || {}).label || 'input' };
    node.port.onmessage = e => {
      const ch = e.data.ch; rec.chunks[0].push(ch[0]); rec.chunks[1].push(ch[1] || ch[0]); rec.frames += ch[0].length;
      let pk = 0; for (const v of ch[0]) pk = Math.max(pk, Math.abs(v)); rec.peak = Math.max(pk, rec.peak * 0.9);
      if (rec.frames > SR * 600) stopRecording();
    };
    $('rec').classList.add('on'); $('recMeter').classList.remove('hidden');
    const tick = () => {
      if (!rec) return;
      const s = (performance.now() - rec.t0) / 1000;
      $('rec').textContent = `■ Stop ${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
      $('recMeter').firstChild.style.width = Math.min(100, Math.sqrt(rec.peak) * 100) + '%';
      $('recMeter').firstChild.style.background = rec.peak > 0.95 ? '#e5484d' : '#3fbf6a';
      requestAnimationFrame(tick);
    };
    tick();
    status(`Recording from ${rec.label}. Play the song, then tap Stop (up to 10 minutes).`);
    log('Recording from ' + rec.label);
  } catch (err) {
    rec = null; status('Could not start recording: ' + (err.name === 'NotAllowedError' ? 'microphone access was not allowed (Settings › Safari › Microphone).' : err.message));
    log('ERROR rec ' + (err.stack || err));
  }
};
async function stopRecording() {
  const r = rec; rec = null; if (!r) return;
  try { r.node.port.postMessage('stop'); } catch (e) {}
  r.src.disconnect(); r.node.disconnect(); r.mute.disconnect(); r.stream.getTracks().forEach(t => t.stop());
  $('rec').classList.remove('on'); $('rec').textContent = '● Record'; $('recMeter').classList.add('hidden');
  if (r.frames < SR * 0.5) { status('Recording too short.'); return; }
  const ctx = audioCtx(), buf = ctx.createBuffer(2, r.frames, SR);
  for (let c = 0; c < 2; c++) { const d = buf.getChannelData(c); let o = 0; for (const ch of r.chunks[c]) { d.set(ch, o); o += ch.length; } }
  const when = new Date();
  const name = `Recording ${when.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })} ${when.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`;
  log(`Recorded ${(r.frames / SR).toFixed(1)} s`);
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
      if (!TEST_NO_DEMUCS) await loadDemucs(o, (m, f, c) => { status(m, f * 0.3); if (c) crumb(c); });
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
        const d = TEST_NO_DEMUCS ? { left: L.slice(s0, s1), right: R.slice(s0, s1) } : await separateDrums(o, L.subarray(s0, s1), R.subarray(s0, s1), p => status(`Separating drums… ${Math.round((done + p * (b - a)) / total * 100)}%`, 0.3 + 0.55 * (done + p * (b - a)) / total));
        log(`Separated ${a.toFixed(1)}–${b.toFixed(1)} s in ${((performance.now() - t) / 1000).toFixed(1)} s`);
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
    crumbDone();
    status(`Done. Swing ${S.swing.type === 'Off' ? 'off' : S.swing.type + ' ' + S.swing.depth + '%'}. Check each pattern (turn on Drums and Pattern to hear them together), then Export to BMY.`, 1);
  } catch (err) {
    crumbDone(); status('Extraction failed: ' + err.message); log('ERROR ' + (err.stack || err));
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
    if (pf > 0 && pk > 0.12 * main && pk > 2.8 * floor + 0.25 && !S.hits.some(o => o.cls === h.cls && Math.abs(o.time - t) < 0.3 * b.step))
      out.push({ time: t, cls: h.cls, lane: h.lane, vel: 'sf', levelDb: (h.levelDb || 0) - 10, ghost: true });
  }
  return out;
}
async function transcribeRegion(o, a, b) {
  crumb('transcribing');
  await loadAdtof(o, 'models/adtof.onnx');
  const s0 = Math.floor(a * SR), s1 = Math.min(S.drumMono.length, Math.floor(b * SR));
  // normalise the stem to -1 dBFS peak (max +30 dB) for detection only; playback and velocities are unaffected
  const raw = S.drumMono.subarray(s0, s1);
  let peak = 0; for (let i = 0; i < raw.length; i++) { const v = Math.abs(raw[i]); if (v > peak) peak = v; }
  const gain = peak > 0 ? Math.min(31.6, 0.891 / peak) : 1;
  const mono = gain === 1 ? raw : raw.map(v => v * gain);
  log(`Stem ${a.toFixed(1)}–${b.toFixed(1)} s peaked at ${(20 * Math.log10(peak || 1e-9)).toFixed(1)} dBFS; normalised by ${(20 * Math.log10(gain)).toFixed(1)} dB for transcription`);
  const spec = logSpec(mono);
  const { act, frames } = await activations(o, mono, spec);
  const hits = hitsFrom(act, frames).map(h => ({ ...h, time: h.time + s0 / SR }));
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
  const vw = $('wMix').getBoundingClientRect().width / S.view.pps;
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
$('play').onclick = () => { if (!playing && P.clock && !P.rec) stopClock(); return playing ? stopPlay() : startPlay(); };
function stopPlay() {
  killNodes(); playing = false; cancelAnimationFrame(raf); $('play').textContent = '▶︎ Play';
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
  playing = true; $('play').textContent = '■ Stop';
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
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- export
$('export').onclick = async () => {
  const pats = S.marks.filter(i => S.patterns.has(i)).map(i => ({ name: `Bar ${barNo(i)}`, ...S.patterns.get(i) }));
  if (!pats.length) return;
  try {
    const bank = buildBank({ name: $('bankName').value || S.name, patterns: pats, swing: S.swing, bpm: S.grid.bpm });
    const json = JSON.stringify(bank, null, 2);
    const fname = bankFileName(bank);
    const file = new File([json], fname, { type: 'application/octet-stream' });
    log(`Exported "${fname}": ${pats.length} patterns, ${pats.reduce((n, p) => n + p.cells.length, 0)} cells, ${bank.tempo ? bank.tempo.bpm + ' BPM' : 'no tempo'}`);
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      try { await navigator.share({ files: [file], title: fname }); status(`Shared ${fname}. In the share sheet, choose BMY Drummer (or Save to Files).`); return; }
      catch (e) { if (e.name === 'AbortError') return; log('Share failed, downloading instead: ' + e.message); }
    }
    const url = URL.createObjectURL(file), a = document.createElement('a');
    a.href = url; a.download = fname; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    status(`Saved ${fname} to Downloads. Tap it in Files to import it into BMY.`);
  } catch (err) { status('Export failed: ' + err.message); log('ERROR ' + (err.stack || err)); }
};

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
$('closeLog').onclick = () => { $('logBox').open = false; };
$('copyLog').onclick = async () => { try { await navigator.clipboard.writeText($('log').textContent); $('copyLog').textContent = 'Copied'; } catch (e) { $('copyLog').textContent = 'Copy failed'; } };

// ---------------------------------------------------------------- Perform: clap / tap / beatbox over a click
const P = {
  bpm: 100, bars: 2, mode: 'lane', lane: 'snare', overdub: true, clickVol: 0.7,
  latency: null, sounds: null, takes: [], key: null, name: null, rec: null, taps: [], stream: null, base: null, clock: null
};
try { const st = JSON.parse(localStorage.getItem('bmyGLPerform') || '{}'); Object.assign(P, { latency: st.latency ?? null, bpm: st.bpm || 100, bars: st.bars || 2, clickVol: st.clickVol ?? 0.7 });
  if (st.sounds) P.sounds = st.sounds.map(x => ({ ...x, centroid: x.centroid && Float32Array.from(x.centroid) })); } catch (e) {}
if (!P.sounds) P.sounds = [{ name: 'Kick', lane: 'kick' }, { name: 'Snare', lane: 'snare' }, { name: 'Hat', lane: 'closedHat' }];
const savePerfPrefs = () => { try { localStorage.setItem('bmyGLPerform', JSON.stringify({ latency: P.latency, bpm: P.bpm, bars: P.bars, clickVol: P.clickVol,
  sounds: P.sounds.map(x => ({ name: x.name, lane: x.lane, centroid: x.centroid ? Array.from(x.centroid) : null, n: x.n || 0 })) })); } catch (e) {} };
const pStatus = m => { $('pStatus').textContent = m; };
const defaultLatency = () => { const c = audioCtx(); return (c.outputLatency || 0) + (c.baseLatency || 0) + 0.015; };
const latency = () => (P.latency ?? defaultLatency());

function showPerf() {
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
  if (!el.classList.contains('hidden')) { if (!P.rec) { stopClock(); el.classList.add('hidden'); $('perfBtn').setAttribute('aria-expanded', 'false'); } return; }
  closeRecent(); stopPlay(); el.classList.remove('hidden'); $('perfBtn').setAttribute('aria-expanded', 'true'); showPerf();
};
$('pBpm').onchange = () => { P.bpm = Math.max(40, Math.min(240, +$('pBpm').value || 100)); savePerfPrefs(); if (P.clock && !P.rec) { stopClock(); startClock(); } };
document.querySelectorAll('#pBars button').forEach(b => b.onclick = () => { if (P.takes.length && +b.dataset.n !== P.bars) { pStatus('Clear the takes first to change the number of bars.'); return; } P.bars = +b.dataset.n; savePerfPrefs(); showPerf(); });
document.querySelectorAll('#pMode button').forEach(b => b.onclick = () => { P.mode = b.dataset.m; showPerf(); });
$('pLane').onchange = () => { P.lane = $('pLane').value; };
$('pClick').oninput = () => { P.clickVol = +$('pClick').value / 100; savePerfPrefs(); if (P.clock) P.clock.gain.gain.value = P.clickVol; };
$('pOverdub').onchange = () => { P.overdub = $('pOverdub').checked; };
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
async function micOpen() {
  const ctx = audioCtx();
  if (!P.stream || P.stream.getAudioTracks().every(t => t.readyState === 'ended')) {
    P.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    await ctx.audioWorklet.addModule('js/rec-worklet.js');
  }
  const src = ctx.createMediaStreamSource(P.stream);
  const node = new AudioWorkletNode(ctx, 'bmy-recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
  const mute = ctx.createGain(); mute.gain.value = 0;
  src.connect(node); node.connect(mute); mute.connect(ctx.destination);
  const r = { src, node, mute, chunks: [], frame0: null, frames: 0 };
  node.port.onmessage = e => { if (r.frame0 === null) r.frame0 = e.data.frame; r.chunks.push(e.data.ch[0]); r.frames += e.data.ch[0].length; };
  return r;
}
function micClose(r) {
  try { r.node.port.postMessage('stop'); } catch (e) {}
  r.src.disconnect(); r.node.disconnect(); r.mute.disconnect();
  const mono = new Float32Array(r.frames); let o = 0; for (const c of r.chunks) { mono.set(c, o); o += c.length; }
  return { mono, frame0: r.frame0 ?? 0 };
}
const sleep = ms => new Promise(res => setTimeout(res, ms));
async function listenFor(secs, label) {
  const r = await micOpen(); const t0 = performance.now();
  while ((performance.now() - t0) / 1000 < secs) { pStatus(`${label} ${Math.ceil(secs - (performance.now() - t0) / 1000)}…`); await sleep(200); }
  return micClose(r);
}
$('pListen').onclick = async () => {
  try {
    const { mono } = await listenFor(6, 'Clap or tap quarter notes now…');
    const on = detectOnsets(mono).filter(o => !looksLikeClick(o.feat));
    const b = tempoFromOnsets(on.map(o => o.t));
    if (!b) { pStatus(`Heard ${on.length} hit(s); need at least 3 steady quarter notes. Try again.`); return; }
    P.bpm = b; savePerfPrefs(); $('pBpm').value = b; if (P.clock && !P.rec) { stopClock(); startClock(); } pStatus(`Listened: ${b} BPM from ${on.length} hits.`);
  } catch (err) { pStatus('Microphone not available: ' + err.message); }
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
function startClock() {
  const ctx = audioCtx(), clicks = clickBuffers(ctx), beat = 60 / P.bpm;
  const gain = ctx.createGain(); gain.gain.value = P.clickVol; gain.connect(ctx.destination);
  const C = { t0: ctx.currentTime + 0.4, beat, bar: 4 * beat, gain, next: 0, sched: [], bpm: P.bpm };
  const schedule = () => {
    while (C.t0 + C.next * beat < ctx.currentTime + 0.5) {
      const n = ctx.createBufferSource(); n.buffer = C.next % 4 === 0 ? clicks.accent : clicks.beat; n.connect(gain); n.start(C.t0 + C.next * beat);
      C.sched.push(C.t0 + C.next * beat); if (C.sched.length > 4000) C.sched.splice(0, 2000); C.next++;
    }
  };
  schedule(); C.timer = setInterval(schedule, 50);
  P.clock = C; showClock();
  return C;
}
function stopClock() { if (!P.clock) return; clearInterval(P.clock.timer); try { P.clock.gain.disconnect(); } catch (e) {} P.clock = null; showClock(); }
function showClock() { const on = !!P.clock; $('pClickBtn').classList.toggle('on', on); $('pClickBtn').textContent = on ? '■ Click' : '▶︎ Click'; }
$('pClickBtn').onclick = () => { if (P.rec) return; if (P.clock) stopClock(); else startClock(); };

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
  if (!P.overdub) for (const t of P.takes) t.hits = t.hits.filter(h => !lanes.includes(h.lane));
  P.takes.push({ id: Date.now(), hits, audio, lanes });
  const passes = Math.max(1, Math.round((mono.length - start) / loopN));
  pStatus(`Take ${P.takes.length}: ${hits.length} hits over ${passes} pass${passes > 1 ? 'es' : ''}${clicksIgnored ? `, ${clicksIgnored} click(s) ignored` : ''}${unsure ? `, ${unsure} unsure` : ''}. Latency ${Math.round(lat * 1000)} ms.`);
  log(`Perform take: ${hits.length} hits, lanes ${lanes.join('/')}, ${P.overdub ? 'overdub' : 'replace'}`);
  await buildPerformSong();
}
$('pUndo').onclick = async () => { P.takes.pop(); await buildPerformSong(); pStatus(P.takes.length ? `Undid the last take; ${P.takes.length} left.` : 'All takes undone.'); };
$('pClear').onclick = async () => { P.takes = []; P.key = null; P.base = null; pStatus('Cleared. Record a new take (a new Perform song).'); showPerf(); };

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
  S.view.pps = Math.max(8, $('wMix').getBoundingClientRect().width / (bar * P.bars + 1)); S.view.start = 0;
  $('songTitle').textContent = S.name; $('bankName').value = S.name; setBpmField();
  S.perf = { bpm: P.bpm, bars: P.bars, takes: P.takes.map(t => ({ id: t.id, lanes: t.lanes, hits: t.hits })) };
  updateButtons(); showLock(); renderTabs(); renderGrid(); redraw(); showPerf();
  try { await putFile({ key: S.key, bytes: wavBytes(buf), type: 'audio/wav' }); } catch (e) {}
  saveNow();
}

// ---------------------------------------------------------------- start
log('Groove Lab loaded (v1)');
loadKit(audioCtx()).then(k => log('Preview kit: ' + k.name)).catch(e => log('Preview kit failed, using the simple synth: ' + e.message));
if (lastCrash) { log('LAST RUN STOPPED DURING: ' + lastCrash); status('The last run stopped during: ' + lastCrash + '. Open the Log for details.'); crumbDone(); }
updateButtons(); showLock(); renderGrid(); redraw();
