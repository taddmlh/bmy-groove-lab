// Pack tools/fluid_render.cjs output (fluid_raw.json) into kit/<program>/: one 16-bit mono WAV per lane
// (6 velocity layers back to back) + kit.json. Tails trimmed below -66 dBFS with a 5 ms fade.
const fs = require('fs');
const [, , rawPath, outDir, program, name] = process.argv;
const SR = 44100, VELS = { gh: 28, sf: 46, n1: 66, n2: 82, a1: 102, a2: 116 };
const raw = JSON.parse(fs.readFileSync(rawPath));
const NOTE_FILE = { kick: 36, sidestick: 37, snare: 38, snareAlt: 40, closedHat: 42, tomFloor: 43, openHat: 46, tomMid: 47, tomHigh: 48, crash: 49, ride: 51, oneShotA: 60, oneShotB: 61, oneShotC: 62 };
const thr = Math.pow(10, -66 / 20), fade = Math.round(0.005 * SR);
fs.mkdirSync(outDir, { recursive: true });
const man = { kit: `GeneralUser GS — ${name} (bank 128, program ${program}), rendered with FluidSynth 2.4.6`, sr: SR, velocities: VELS, notes: {} };
const missing = [];
for (const [lane, n] of Object.entries(raw)) {
  const layers = Object.keys(VELS).map(l => Float32Array.from(raw[lane].layers[l]));
  const peak = Math.max(...layers.map(a => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0)));
  if (peak < 1e-4) { missing.push(lane); continue; }                    // this kit has no sound on that note
  const cut = layers.map(a => { let e = a.length - 1; while (e > 0 && Math.abs(a[e]) < thr) e--; const L = Math.min(a.length, e + 64); const o = a.slice(0, L); for (let i = 0; i < Math.min(fade, L); i++) o[L - 1 - i] *= i / fade; return o; });
  const total = cut.reduce((s, a) => s + a.length, 0), b = Buffer.alloc(44 + total * 2);
  b.write('RIFF', 0); b.writeUInt32LE(36 + total * 2, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(SR, 24); b.writeUInt32LE(SR * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34); b.write('data', 36); b.writeUInt32LE(total * 2, 40);
  let o = 0; const lay = [];
  for (const a of cut) { lay.push([o, a.length]); for (const v of a) { b.writeInt16LE(Math.max(-32767, Math.min(32767, Math.round(v * 32767))), 44 + o * 2); o++; } }
  const file = `${NOTE_FILE[lane]}-${lane}.wav`; fs.writeFileSync(`${outDir}/${file}`, b);
  man.notes[lane] = { note: n.note, file, layers: lay, exclusiveClass: lane === 'closedHat' || lane === 'openHat' ? 1 : 0 };
}
fs.writeFileSync(`${outDir}/kit.json`, JSON.stringify(man, null, 1));
console.log(`${name}: ${Object.keys(man.notes).length} lanes${missing.length ? ', missing ' + missing.join(' ') : ''}`);
