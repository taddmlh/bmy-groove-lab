"""Add percussion samples to Groove Lab's Standard kit (kit/): one WAV per GM note, 6 velocity layers back to back,
listed in kit/kit.json as lanes 'gm<note>'. Used to play One-shot lanes with the sound the transcriber heard.

    python3 tools/render_perc.py /path/GeneralUser-GS.sf2        (needs the fluidsynth command)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pretty_midi
import soundfile as sf

SR = 44100
VELS = {'gh': 28, 'sf': 46, 'n1': 66, 'n2': 82, 'a1': 102, 'a2': 116}
NOTES = {82: 'Shaker', 54: 'Tambourine', 56: 'Cowbell', 63: 'Conga hi', 64: 'Conga lo', 60: 'Bongo hi', 61: 'Bongo lo',
         76: 'Woodblock hi', 77: 'Woodblock lo'}   # PERC_GM in training + the high/low pairs Groove Lab splits by pitch
GAIN = 2.3        # the CLI renders 2.3x quieter than the kit's FluidSynth (WebAssembly) render (snare n2: 0.205 vs 0.471)
KIT = os.path.join(os.path.dirname(__file__), '..', 'kit')
sf2 = sys.argv[1]


def render(note, vel, secs=1.5):
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(0, is_drum=True)
    ins.notes.append(pretty_midi.Note(vel, note, 0.0, 0.1))
    pm.instruments.append(ins)
    with tempfile.TemporaryDirectory() as d:
        mid, wav = os.path.join(d, 'n.mid'), os.path.join(d, 'n.wav')
        pm.write(mid)
        subprocess.run(['fluidsynth', '-ni', '-R', '0', '-C', '0', '-g', '1.0', '-r', str(SR), '-F', wav, sf2, mid], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        y, _ = sf.read(wav, dtype='float32', always_2d=True)
    y = y.mean(axis=1)[:int(secs * SR)] * GAIN
    thr = 10 ** (-66 / 20)
    e = len(y) - 1
    while e > 0 and abs(y[e]) < thr:
        e -= 1
    y = y[:min(len(y), e + 64)].copy()
    f = min(len(y), int(0.005 * SR))
    if f:
        y[-f:] *= np.linspace(1, 0, f)
    return y


man = json.load(open(os.path.join(KIT, 'kit.json')))
for note, name in NOTES.items():
    layers = [render(note, v) for v in VELS.values()]
    data = np.concatenate(layers)
    file = f'{note}-gm{note}.wav'
    sf.write(os.path.join(KIT, file), data, SR, subtype='PCM_16')
    offs, o = [], 0
    for a in layers:
        offs.append([o, len(a)])
        o += len(a)
    man['notes'][f'gm{note}'] = {'note': note, 'name': name, 'file': file, 'layers': offs, 'exclusiveClass': 0}
    print(f'{name} ({note}): {len(data) / SR:.2f} s, peak {np.abs(data).max():.2f}')
json.dump(man, open(os.path.join(KIT, 'kit.json'), 'w'), indent=1)
