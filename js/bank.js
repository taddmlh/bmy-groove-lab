// BMY .bmybank writer. Schema 1, matching BMY_PatternBank.swift (BMYPatternBank / BMYPattern / BMYPatternCell).
const uuid = () => (crypto.randomUUID ? crypto.randomUUID() : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
  const r = Math.random() * 16 | 0; return (c === 'x' ? r : (r & 3 | 8)).toString(16);
})).toUpperCase();

const TRIP_OK = new Set(['111', '010', '001', '011', '110', '101']);

function sortKeys(v) {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === 'object') return Object.fromEntries(Object.keys(v).sort().map(k => [k, sortKeys(v[k])]));
  return v;
}

/**
 * patterns: [{ name, cells:[{lane, step, velocity, ornament?, tripletPattern, note2Velocity?, note3Velocity?}],
 *              hatUseAlt }]
 * swing: {type:'Off'|'1-8'|'1-16', depth 0..100}; bpm: number; oneShots: {oneShotA: {gm, name}, ...} (optional).
 */
export function buildBank({ name, patterns, swing, bpm, oneShots }) {
  if (!patterns.length || patterns.length > 6) throw new Error('A bank needs 1 to 6 patterns');
  const sw = swing && swing.type !== 'Off' ? { type: swing.type, depth: Math.max(0, Math.min(100, Math.round(swing.depth))) } : { type: 'Off', depth: 0 };
  const bank = {
    schemaVersion: 1,
    id: 'user.bank.' + uuid(),
    name: (name || '').replace(/[\/:]/g, '-').replace(/^\.+/, '').trim().slice(0, 60) || 'Imported Bank',
    origin: 'user',
    patterns: patterns.map((p, i) => {
      const seen = new Set();
      const cells = [];
      for (const c of p.cells) {
        if (!(c.step >= 0 && c.step <= 15) || seen.has(c.lane + '#' + c.step)) continue;
        seen.add(c.lane + '#' + c.step);
        const o = { lane: c.lane, step: c.step, velocity: c.velocity, tripletPattern: TRIP_OK.has(c.tripletPattern) ? c.tripletPattern : '111' };
        if (c.ornament) o.ornament = c.ornament;
        if (c.ornament && c.note2Velocity) o.note2Velocity = c.note2Velocity;
        if (c.ornament && c.note3Velocity) o.note3Velocity = c.note3Velocity;
        cells.push(o);
      }
      return {
        id: 'user.pattern.' + uuid(),
        name: (p.name || `Bar ${i + 1}`).trim() || `Bar ${i + 1}`,
        cells,
        settings: { swing: sw, values: { hatUseAlt: p.hatUseAlt ? 1 : 0, snareUseAlt: 0 }, tempoMode: 'follow' }
      };
    })
  };
  // which sound each one-shot lane holds (GM note + name), from the transcriber. BMY ignores unknown keys until
  // it reads this (planned: built-in kit sound per one-shot, MIDI-out hint).
  if (oneShots && Object.keys(oneShots).length)
    bank.oneShotSounds = Object.fromEntries(Object.entries(oneShots).map(([lane, x]) => [lane, { gmNote: x.gm, name: x.name }]));
  if (bpm && isFinite(bpm)) bank.tempo = { bpm: Math.min(240, Math.max(40, Math.round(bpm))), rampBars: 0 };
  return sortKeys(bank);
}

export function bankFileName(bank) { return bank.name + '.bmybank'; }
