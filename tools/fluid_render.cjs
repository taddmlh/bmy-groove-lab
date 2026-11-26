// Render BMY's kit with FluidSynth 2.4.6 (WebAssembly): GeneralUser GS, bank 128 program 0, one layer per BMY velocity.
const fs = require('fs');
const libfluidsynth = require('js-synthesizer/libfluidsynth');
const JSSynth = require('js-synthesizer');
JSSynth.Synthesizer.initializeWithFluidSynthModule(libfluidsynth);
const SR = 44100;
const VELS = { gh: 28, sf: 46, n1: 66, n2: 82, a1: 102, a2: 116 };
const NOTES = { 36: 'kick', 37: 'sidestick', 38: 'snare', 40: 'snareAlt', 42: 'closedHat', 46: 'openHat', 51: 'ride',
  48: 'tomHigh', 47: 'tomMid', 43: 'tomFloor', 49: 'crash', 60: 'oneShotA', 61: 'oneShotB', 62: 'oneShotC' };
const MAXLEN = { crash: 3.0, ride: 2.5, openHat: 1.6, tomFloor: 1.6, tomMid: 1.4, tomHigh: 1.2 };
const [, , sfPath, outDir, program = '0'] = process.argv;
JSSynth.waitForReady().then(async () => {
  const synth = new JSSynth.Synthesizer();
  synth.init(SR);
  const id = await synth.loadSFont(new Uint8Array(fs.readFileSync(sfPath)).buffer);
  synth.setGain(1.0);
  const renderNote = (note, vel, secs) => {
    synth.midiProgramSelect(9, id, 128, +program);
    const n = Math.round(secs * SR), L = new Float32Array(n), R = new Float32Array(n);
    synth.midiNoteOn(9, note, vel);
    const B = 128; let off = false;
    for (let s = 0; s < n; s += B) {
      if (!off && s >= 0.1 * SR) { synth.midiNoteOff(9, note); off = true; }   // short gate, like a sequencer
      const l = new Float32Array(B), r = new Float32Array(B); synth.render([l, r]);
      L.set(l.subarray(0, Math.min(B, n - s)), s); R.set(r.subarray(0, Math.min(B, n - s)), s);
    }
    synth.midiAllSoundsOff(9);
    for (let k = 0; k < 20; k++) synth.render([new Float32Array(B), new Float32Array(B)]);   // flush
    return { L, R };
  };
  const out = {};
  for (const [note, lane] of Object.entries(NOTES)) {
    out[lane] = { note: +note, layers: {} };
    for (const [lab, vel] of Object.entries(VELS)) {
      const { L, R } = renderNote(+note, vel, MAXLEN[lane] || 0.9);
      out[lane].layers[lab] = Array.from(L, (v, i) => 0.5 * (v + R[i]));
    }
  }
  fs.writeFileSync(outDir + '/fluid_raw.json', JSON.stringify(out));
  console.log('rendered', Object.keys(out).length, 'notes with FluidSynth, program', program);
});
