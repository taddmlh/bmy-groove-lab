// BMY Groove Lab: song -> drum stem (Demucs) -> hits (ADTOF) -> BMY patterns -> .bmybank
import { logSpec, toMono, FPS } from './dsp.js';
import { loadAdtof, activations, hitsFrom } from './adtof.js';
import { detectGrid, barTimes } from './beats.js';
import { LANES, VEL, describeHits, assignLanes, detectSwing, barToPattern } from './transcribe.js';
import { buildBank, bankFileName } from './bank.js';
import { loadDemucs, separateDrums } from './demucs.js';
import { renderRegion } from './synth.js';

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
if (TEST_NO_DEMUCS) window.GL = { S, get loop() { return nodes.mix && [nodes.mix.loopStart, nodes.mix.loopEnd]; }, get nodes() { return Object.keys(nodes); }, get gains() { return Object.fromEntries(Object.entries(gains).map(([k, g]) => [k, +g.gain.value.toFixed(2)])); }, get playing() { return playing; } };

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
function drawOverlays(g, w, h, dpr) {
  if (S.sel) {
    const x0 = tToX(S.sel.a, w, dpr), x1 = tToX(S.sel.b, w, dpr);
    g.fillStyle = 'rgba(255,173,20,.16)'; g.fillRect(x0, 0, x1 - x0, h);
    g.fillStyle = '#ffad14'; g.fillRect(x0, 0, 2 * dpr, h); g.fillRect(x1 - 2 * dpr, 0, 2 * dpr, h);
    if (selIsBars()) {
      const hh = Math.min(h * 0.5, 34 * dpr), y = (h - hh) / 2, hw = 10 * dpr;
      g.fillRect(x0, y, hw, hh); g.fillRect(x1 - hw, y, hw, hh);
      g.fillStyle = '#000';
      for (const x of [x0 + 3 * dpr, x1 - 7 * dpr]) { g.fillRect(x, y + hh * 0.3, dpr, hh * 0.4); g.fillRect(x + 3 * dpr, y + hh * 0.3, dpr, hh * 0.4); }
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
      g.fillStyle = '#26c7ff'; g.beginPath(); g.arc(x0 + 12 * dpr, h / 2, 8 * dpr, 0, 7); g.fill();
      g.fillStyle = '#000'; g.textAlign = 'center'; g.fillText(`${m + 1}`, x0 + 12 * dpr, h / 2 + dpr); g.textAlign = 'left';
    }
    g.fillStyle = '#3a3f4c'; g.fillRect(Math.round(x0), 0, Math.max(1, dpr), h);
    if (barNo(b.i) % every === 0 || m >= 0) {
      g.fillStyle = m >= 0 ? '#e9f8ff' : '#8d929e';
      const label = `${barNo(b.i)}`;
      g.fillText(label, m >= 0 ? x0 + 25 * dpr : x0 + 5 * dpr, h / 2);
    }
  }
  drawOverlays(g, w, h, dpr);
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
function pointerDrag(el, { down, move, up }) {
  let st = null;
  el.addEventListener('pointerdown', e => { el.setPointerCapture(e.pointerId); const r = el.getBoundingClientRect(); st = { x0: e.clientX, x: e.clientX - r.left, moved: false, r }; down && down(st, e); });
  el.addEventListener('pointermove', e => { if (!st) return; const dx = e.clientX - st.x0; if (Math.abs(dx) > 8) st.moved = true; st.dx = dx; st.x = e.clientX - st.r.left; move && move(st, e); });
  const end = e => { if (!st) return; up && up(st, e); st = null; };
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
}
// Ruler: tap = mark/unmark bar; drag = scroll
pointerDrag($('ruler'), {
  down: st => { st.v0 = S.view.start; },
  move: st => { if (!S.grid || !st.moved) return; S.view.start = st.v0 - st.dx / S.view.pps; clampView(); redraw(); },
  up: st => { if (!S.grid || st.moved) return; toggleMark(barAt(xToT(st.x))); }
});
// Waveforms: tap = select that bar (and move playhead); drag = select a range, snapped to bars
for (const id of ['wMix', 'wDrums']) pointerDrag($(id), {
  down: st => {
    st.t0 = xToT(st.x); st.mode = 'select';
    if (S.grid && selIsBars()) {
      const xa = (S.sel.a - S.view.start) * S.view.pps, xb = (S.sel.b - S.view.start) * S.view.pps;
      const n = Math.round((S.sel.b - S.sel.a) / barSec());
      if (Math.abs(st.x - xa) < 22) Object.assign(st, { mode: 'start', a0: S.sel.a, n });
      else if (Math.abs(st.x - xb) < 22) Object.assign(st, { mode: 'end', a0: S.sel.a, n });
    }
  },
  move: st => {
    if (!S.grid || !st.moved) return;
    if (st.mode === 'start') {          // slide every bar line, tempo unchanged
      const a = Math.max(-barSec(), Math.min(S.dur, xToT(st.x)));
      S.grid.t0 = a; S.sel = { a, b: a + st.n * barSec() };
      gridChanged(null, true); S.sel = { a, b: a + st.n * barSec() }; redraw(); syncPlayback(true);
      status(`Bar start ${a.toFixed(3)} s — release to re-transcribe`);
      return;
    }
    if (st.mode === 'end') {            // stretch: start stays, tempo follows the end
      const b = xToT(st.x), len = Math.max(0.2, b - st.a0) / st.n;
      S.grid.bpm = Math.max(30, Math.min(300, Math.round(240 / len * 100) / 100)); S.grid.t0 = st.a0;
      gridChanged(null, true); S.sel = { a: st.a0, b: st.a0 + st.n * barSec() }; redraw(); syncPlayback(true);
      status(`${S.grid.bpm.toFixed(2)} BPM — release to re-transcribe`);
      return;
    }
    const a = Math.min(st.t0, xToT(st.x)), b = Math.max(st.t0, xToT(st.x));
    const ba = barAt(a), bb = barAt(b);
    S.sel = ba >= 0 && bb >= 0 ? { a: S.bars[ba].start, b: S.bars[bb].end } : { a: Math.max(0, a), b: Math.min(S.dur, b) };
    if (!playing) S.playhead = Math.max(0, S.sel.a);
    redraw();
  },
  up: st => {
    if (S.grid && st.moved && (st.mode === 'start' || st.mode === 'end')) {
      gridChanged(st.mode === 'start' ? 'Bar lines moved. Drag the right edge to set the tempo.' : `Tempo set to ${S.grid.bpm.toFixed(2)} BPM.`);
      return;
    }
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
  syncPlayback();
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

// ---------------------------------------------------------------- open song
$('file').addEventListener('change', async e => {
  const f = e.target.files[0]; if (!f) return;
  e.target.value = '';
  stopPlay();
  try {
    busy = true; updateButtons();
    status('Decoding the song…', 0.02); crumb('decoding ' + f.name);
    const ctx = audioCtx();
    const buf = await ctx.decodeAudioData(await f.arrayBuffer());
    Object.assign(S, { name: f.name.replace(/\.[^.]+$/, ''), dur: buf.duration, mixBuf: buf, drumBuf: null, drumMono: null, extracted: [],
      hits: [], patterns: new Map(), marks: [], tab: null, undo: [], sel: null, playhead: 0, ovPeaks: null });
    S.mixMono = toMono(buf.getChannelData(0), buf.numberOfChannels > 1 ? buf.getChannelData(1) : null);
    $('songTitle').textContent = S.name; $('bankName').value = S.name;
    log(`Opened "${f.name}": ${buf.duration.toFixed(1)} s`);
    status('Finding the tempo and bar lines…', 0.1); await yieldUI();
    crumb('analysing tempo');
    S.mixSpec = await specAsync(S.mixMono, p => status('Finding the tempo and bar lines…', 0.1 + 0.85 * p));
    S.grid = detectGrid(S.mixSpec.data, S.mixSpec.frames);
    log(`Tempo ${S.grid.bpm} BPM, first bar line at ${S.grid.t0.toFixed(3)} s`);
    rebuildBars(); setBpmField();
    S.view.pps = Math.max(8, $('wMix').getBoundingClientRect().width / (barSec() * 12));
    S.view.start = Math.max(-0.2, S.grid.t0 - 0.2);
    crumbDone();
    status(`${S.grid.bpm} BPM. Tap a bar and play it. If beat 1 is wrong use ◀︎ Beat / Beat ▶︎, or drag the orange edges of the selected bar (left edge = bar start, right edge = tempo). Then tap bar numbers to mark up to 6 bars.`, 1);
  } catch (err) {
    crumbDone(); status('Could not open that file: ' + err.message); log('ERROR ' + (err.stack || err));
  } finally { busy = false; updateButtons(); renderTabs(); renderGrid(); redraw(); }
});
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
        // transcribe this region
        crumb('transcribing');
        status('Finding hits…', 0.86 + 0.1 * done / total); await yieldUI();
        await loadAdtof(o, 'models/adtof.onnx');
        const mono = S.drumMono.subarray(s0, s1);
        const spec = logSpec(mono);
        const { act, frames } = await activations(o, mono, spec);
        const hits = hitsFrom(act, frames).map(h => ({ ...h, time: h.time + s0 / SR }));
        describeHits(hits, spec.data, spec.frames, Math.round(s0 / SR * FPS));
        S.hits = S.hits.filter(h => h.time < a || h.time >= b).concat(hits).sort((x, y) => x.time - y.time);
        log(`Hits in ${a.toFixed(1)}–${b.toFixed(1)} s: ${countBy(hits)}`);
      }
    }
    buildPatterns(!todo.length);
    S.tab = S.marks.find(i => S.patterns.has(i)) ?? null;
    crumbDone();
    status(`Done. Swing ${S.swing.type === 'Off' ? 'off' : S.swing.type + ' ' + S.swing.depth + '%'}. Check each pattern (turn on Drums and Pattern to hear them together), then Export to BMY.`, 1);
  } catch (err) {
    crumbDone(); status('Extraction failed: ' + err.message); log('ERROR ' + (err.stack || err));
  } finally { busy = false; updateButtons(); renderTabs(); renderGrid(); redraw(); }
};
const countBy = hs => Object.entries(hs.reduce((m, h) => ((m[h.cls] = (m[h.cls] || 0) + 1), m), {})).map(([k, v]) => `${k} ${v}`).join(', ');

/** (Re)build patterns for marked bars that have drums. force: also rebuild edited ones. */
function buildPatterns(force) {
  const bars = S.marks.filter(covered).map(i => S.bars[i]);
  if (!bars.length) return;
  assignLanes(S.hits, bars);
  S.swing = detectSwing(S.hits, bars);
  for (const b of bars) {
    const old = S.patterns.get(b.i);
    if (old && old.edited && !force) continue;
    S.patterns.set(b.i, { ...barToPattern(S.hits, b, S.swing), edited: false });
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
  const swing = S.swing;
  const shift = s => !swing || swing.type === 'Off' ? 0 : swing.type === '1-8' && s % 4 === 2 ? swing.depth / 100 * 2 / 3 : swing.type === '1-16' && s % 2 === 1 ? swing.depth / 100 / 3 : 0;
  for (const l of LANES) {
    const row = document.createElement('div'); row.className = 'row';
    const color = l.role === 'closedHat' && p.hatUseAlt ? '#d9b326' : l.color;
    row.innerHTML = `<div class="glyph" title="${l.role === 'closedHat' && p.hatUseAlt ? 'Ride (hat lane)' : l.name}">${laneGlyph({ ...l, color })}</div>`;
    const trips = p.cells.filter(c => c.lane === l.role && SPAN[c.ornament]);
    for (let s = 0; s < 16; s++) {
      const cell = document.createElement('div');
      const cov = trips.some(t => s >= t.step && s < t.step + SPAN[t.ornament]);
      cell.className = 'cell' + (s % 4 === 0 ? ' beat' : '') + (cov ? ' cov' : '');
      const c = p.cells.find(x => x.lane === l.role && x.step === s && !SPAN[x.ornament]);
      if (c) {
        const n = document.createElement('div'); n.className = 'note';
        const z = glyphSize(c.velocity) * 100;
        n.style.cssText = `height:${z}%;aspect-ratio:1;max-width:${z}%;background:${color};left:${(50 + shift(s) * 100).toFixed(1)}%;top:50%;transform:translate(-50%,-50%)`;
        cell.appendChild(n);
      }
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
        d.style.cssText = `left:calc(46px + (100% - 46px) * ${(pos / 16).toFixed(4)});height:${(z * 70).toFixed(0)}%;aspect-ratio:1;background:${color};box-shadow:0 0 0 1px rgba(255,255,255,.55)`;
        row.appendChild(d);
      }
    }
    g.appendChild(row);
  }
  $('swType').value = swing.type; $('swDepth').value = swing.depth; $('swDepth').disabled = swing.type === 'Off';
  $('hatRide').checked = !!p.hatUseAlt;
  const tripN = p.cells.filter(c => SPAN[c.ornament]).length;
  $('patNotes').textContent = [tripN ? `${tripN} triplet${tripN > 1 ? 's' : ''}` : '', ...(p.notes || [])].filter(Boolean).join(' · ');
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
  updateButtons(); renderGrid(); syncPlayback();
}
$('undo').onclick = () => {
  const u = S.undo.pop(); if (!u) return;
  const p = S.patterns.get(u.bar); if (p) { p.cells = u.cells; p.hatUseAlt = u.hatUseAlt; }
  S.tab = u.bar; updateButtons(); renderTabs(); renderGrid(); syncPlayback();
};
$('swType').onchange = () => { S.swing = { type: $('swType').value, depth: $('swType').value === 'Off' ? 0 : (+$('swDepth').value || 50) }; renderGrid(); syncPlayback(); };
$('swDepth').onchange = () => { S.swing.depth = Math.max(0, Math.min(100, Math.round(+$('swDepth').value || 0))); renderGrid(); syncPlayback(); };
$('hatRide').onchange = () => { const p = S.patterns.get(S.tab); if (!p) return; p.hatUseAlt = $('hatRide').checked ? 1 : 0; p.edited = true; renderGrid(); syncPlayback(); };

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
$('play').onclick = () => (playing ? stopPlay() : startPlay());
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

$('copyLog').onclick = async () => { try { await navigator.clipboard.writeText($('log').textContent); $('copyLog').textContent = 'Copied'; } catch (e) { $('copyLog').textContent = 'Copy failed'; } };

// ---------------------------------------------------------------- start
log('Groove Lab loaded (v1)');
if (lastCrash) { log('LAST RUN STOPPED DURING: ' + lastCrash); status('The last run stopped during: ' + lastCrash + '. Open the Log for details.'); crumbDone(); }
updateButtons(); renderGrid(); redraw();
