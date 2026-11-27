// Playability pass: what a drummer could and probably did play, after the transcriber and before the grid.
// Measured 2026-11-27 on the test rig (132 files) and the EWTRTW / Rosanna exams: about neutral. Pruning quiet
// one-offs helps BMY slightly; recovering repeating near-misses added as many wrong hits as right ones (a repeating
// shaker or bleed passes the repetition test too), so it is off by default. The errors seen on real stems repeat
// every bar and are physically playable, so they need better training data, not this pass.
// Matt, 2026-11-27: "add some smarts based on our physical model". Two parts:
//  1. Repetition: grooves repeat. Every candidate (including weak ones just under the transcriber's threshold) is
//     placed on a 48-tick bar grid (16ths and both triplet grids). A weak candidate is kept when the same drum
//     sounds at that spot in most bars; a quiet borderline hit that happens only once is dropped. Confident hits
//     are never touched, and loud one-offs (fills, accents) stay.
//  2. Physics (BMY's two-hand model): feet (kick, hat pedal) are free; hands are two. A moment that needs three
//     hands (snare, toms, cymbals, hats at once) loses its least confident hand hits. Percussion (shaker...) is
//     usually another player, so it never counts.
// Hits: { time, cls, model?, conf (1 = at threshold), weak?, levelDb? }. Bars: [{ start, end, step }].

const TICKS = 48;                                   // per bar: 16ths = 3 ticks, 8th triplets = 4, 16th triplets = 2
const HANDS = new Set(['snare', 'tom', 'cymbal', 'hihat']);
const key = h => h.model || h.cls;

function tickOf(h, bars) {
  const i = bars.findIndex(b => h.time >= b.start - b.step / 4 && h.time < b.end - b.step / 4);
  if (i < 0) return null;
  const b = bars[i];
  return { bar: i, tick: ((Math.round((h.time - b.start) / b.step * 3) % TICKS) + TICKS) % TICKS };
}

/** -> { hits, added, dropped, impossible, shadows, bars, notes[] } */
export function playability(hits, bars, opt = {}) {
  const minBars = opt.minBars ?? 3, share0 = opt.share ?? 0.6, strong0 = opt.strong ?? 0.34, conf0 = opt.minConf ?? 0;
  const prune = opt.prune ?? true, recover = opt.recover ?? false;   // recovery: off by default (rig, 2026-11-27: no gain)
  const keep = new Set(hits.filter(h => !h.weak));
  const notes = [];
  let added = 0, dropped = 0, impossible = 0, shadows = 0;

  // ---- 1. repetition across bars
  const active = new Set();                                         // bars with any confident hit
  const at = new Map();                                             // drum|tick -> {strong:Set(bar), any:Set(bar)}
  for (const h of hits) {
    const p = tickOf(h, bars); if (!p) continue;
    h._bar = p.bar; h._tick = p.tick;
    if (!h.weak) active.add(p.bar);
    for (const d of [-1, 0, 1]) {                                   // +-1 tick: timing wobble
      const k = key(h) + '|' + ((p.tick + d + TICKS) % TICKS);
      if (!at.has(k)) at.set(k, { strong: new Set(), any: new Set() });
      const e = at.get(k); e.any.add(p.bar); if (!h.weak) e.strong.add(p.bar);
    }
  }
  const nBars = active.size;
  if (nBars >= minBars) {
    // median level per drum, to tell a quiet borderline hit (ghost-like) from a loud one-off
    const lv = {};
    hits.filter(h => !h.weak && h.levelDb !== undefined).forEach(h => (lv[key(h)] = lv[key(h)] || []).push(h.levelDb));
    const med = Object.fromEntries(Object.entries(lv).map(([k, v]) => [k, v.sort((a, b) => a - b)[v.length >> 1]]));
    for (const h of hits) {
      if (h._tick === undefined) continue;
      const e = at.get(key(h) + '|' + h._tick), share = e.any.size / nBars, strongShare = e.strong.size / nBars;
      if (h.weak) {
        // a near-miss the transcriber skipped: keep it when this drum plays here in most bars, and is confidently
        // heard here in some of them (a repeating weak echo of another drum isn't enough), and isn't a second
        // detection of a hit already kept
        const dup = hits.some(x => x !== h && !x.weak && key(x) === key(h) && Math.abs(x.time - h.time) < 0.06);
        if (recover && !dup && h.conf >= conf0 && share >= share0 && strongShare >= strong0) { keep.add(h); added++; h.recovered = true; }
      } else if (prune && h.conf < 1.25 && share <= 1 / nBars + 1e-9 && nBars >= 4
                 && h.levelDb !== undefined && med[key(h)] !== undefined && h.levelDb < med[key(h)] - 8) {
        keep.delete(h); dropped++;                                  // quiet, borderline and only once: likely bleed
      }
    }
  }

  // ---- 1b. shadows: a quiet snare that sounds (almost) only together with the kick is the kick's own boom or the
  // snare wires buzzing in sympathy, not a stroke: real ghost notes fall between and around other hits.
  if (opt.shadow ?? true) {
    const sn = [...keep].filter(h => h.cls === 'snare' && h.levelDb !== undefined);
    const lv = sn.map(h => h.levelDb).sort((a, b) => a - b), top = lv[Math.floor(lv.length * 0.75)] ?? 0;
    const quiet = sn.filter(h => h.levelDb < top - 8);
    const kicks = [...keep].filter(h => h.cls === 'kick');
    const onKick = quiet.filter(h => kicks.some(k => Math.abs(k.time - h.time) <= 0.03));
    if (quiet.length >= 3 && onKick.length >= 0.7 * quiet.length) {
      onKick.forEach(h => { keep.delete(h); shadows++; });
      notes.push(`${onKick.length} quiet snare(s) only ever under the kick: read as kick`);
    }
  }

  // ---- 2. physics: two hands
  const kept = [...keep].sort((a, b) => a.time - b.time);
  for (let i = 0; i < kept.length;) {
    let j = i; while (j + 1 < kept.length && kept[j + 1].time - kept[i].time <= 0.03) j++;
    const group = kept.slice(i, j + 1).filter(h => HANDS.has(h.cls) && !h.pedal);
    // a hat with two other hand drums can be the foot (pedal), so only count it if a third hand is still needed
    const nonHat = group.filter(h => h.cls !== 'hihat');
    if (nonHat.length > 2) {
      nonHat.sort((a, b) => a.conf - b.conf).slice(0, nonHat.length - 2).forEach(h => { keep.delete(h); impossible++; });
    }
    i = j + 1;
  }

  const out = hits.filter(h => keep.has(h));
  out.forEach(h => { delete h._bar; delete h._tick; });
  if (nBars < minBars) notes.push(`repetition needs ${minBars}+ bars (had ${nBars})`);
  return { hits: out, added, dropped, impossible, shadows, bars: nBars, notes };
}
