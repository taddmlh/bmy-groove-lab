"""BMY transcriber test rig: known grooves -> free kits -> Groove Lab's transcribers -> a text report.

    python3 run.py --kits naked,drs,muldjord,gogodze --kit-dir /path/kits --out /path/rig
                   [--models kits=../../models/bmy_drums.onnx,trial=/tmp/trial1.onnx] [--no-adtof]

Writes, per scenario x kit x condition: <name>.wav (16-bit, for listening / Groove Lab import), <name>.mid and
<name>.txt (every hit: time, drum, velocity); then results/<transcriber>.json and REPORT.txt.
Conditions: 'dry' (the kit as recorded) and 'mixed' (EQ, room, bus compression, saturation, fixed per file).
"""
import argparse
import json
import random
import subprocess
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bmy_transcriber as bt                 # noqa: E402
import sfz_kits                              # noqa: E402
from scenarios import SCENARIOS, TEMPO       # noqa: E402

C9 = bt.CLASSES
TO5 = {'kick': 'kick', 'snare': 'snare', 'tomHigh': 'tom', 'tomMid': 'tom', 'tomFloor': 'tom',
       'hhClosed': 'hihat', 'hhOpen': 'hihat', 'crash': 'cymbal', 'ride': 'cymbal'}
C5 = ['kick', 'snare', 'tom', 'hihat', 'cymbal']
TOL = 0.05                                   # 50 ms, as in the drum-transcription literature


# ----------------------------------------------------------------------------- rendering
def render(hits, kits, name, rnd):
    """Through kit `name`; pieces it lacks from the other kits, in order. -> (audio, hits played)."""
    seconds = max(t for t, _, _ in hits) + 3
    y, played = kits[name].render(hits, seconds, random.Random(1))
    for donor in [k for k in kits if k != name]:
        rest = [h for h in hits if h not in set(played) and h[1] in sfz_kits.ART_OF_PITCH]
        if not rest:
            break
        y2, p2 = kits[donor].render(rest, seconds, random.Random(2))
        if p2:
            r1, r2 = np.sqrt(np.mean(y ** 2)), np.sqrt(np.mean(y2 ** 2)) + 1e-9
            y, played = y + y2 * (r1 / r2 * 0.5 if r1 > 0 else 1), sorted(played + p2)
    return y.astype(np.float32), played


def write_files(base: Path, y, played):
    import pretty_midi
    import soundfile as sf
    sf.write(str(base) + '.wav', bt.normalise(y, -1.0), bt.SR, subtype='PCM_16')
    pm = pretty_midi.PrettyMIDI()
    ins = pretty_midi.Instrument(0, is_drum=True)
    ins.notes = [pretty_midi.Note(int(v), int(p), float(t), float(t) + 0.05) for t, p, v in played]
    pm.instruments.append(ins)
    pm.write(str(base) + '.mid')
    with open(str(base) + '.txt', 'w') as f:
        f.write('time_s\tdrum\tvelocity\tgm_note\n')
        for t, p, v in played:
            f.write(f'{t:.3f}\t{bt.PITCH_CLASS[p]}\t{v}\t{p}\n')


def build(args):
    out = Path(args.out).resolve()
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    kits = {}
    for k in args.kits.split(','):
        kits[k] = sfz_kits.SfzKit(sfz_kits.fetch_kit(k, args.kit_dir), notes=sfz_kits.KITS[k].get('notes'))
    truth = {}
    for sname, fn in SCENARIOS.items():
        hits = [h for h in fn() if h[1] in bt.PITCH_CLASS]
        for k in kits:
            y, played = render(hits, kits, k, None)
            conds = ('dry', 'mixed') + (('shaker',) if args.sf2 else ())
            for cond in conds:
                rs = random.Random(zlib.crc32(f'{sname}{k}'.encode()))
                truth_hits = played
                if cond == 'shaker':                      # + a shaker part locked to the groove: right answer = hi-hat
                    bpm, feel = TEMPO[sname]
                    aux, aux_hits = bt.render_aux(len(y) / bt.SR, bpm, rs, args.sf2, 0.3, feel, ['shaker'])
                    aux = aux[:len(y)]
                    truth_hits = sorted(played + [h for h in aux_hits if h[0] < len(y) / bt.SR - 1])
                    z = y + aux / (np.sqrt(np.mean(aux ** 2)) + 1e-9) * np.sqrt(np.mean(y ** 2)) * 10 ** (-8 / 20)
                    z = bt.augment(bt.normalise(z, -6), rs)
                else:
                    z = y if cond == 'dry' else bt.augment(bt.normalise(y, -6), rs)
                base = out / 'audio' / f'{sname}__{k}__{cond}'
                write_files(base, z, truth_hits)
                truth[str(base) + '.wav'] = [(t, bt.PITCH_CLASS[p], v) for t, p, v in truth_hits]
        print(f'rendered {sname}', flush=True)
    json.dump(truth, open(out / 'truth.json', 'w'))
    return truth


# ----------------------------------------------------------------------------- transcribing
def transcribe(args, wavs):
    res_dir = Path(args.out) / 'results'
    res_dir.mkdir(exist_ok=True)
    repo = str(HERE.parent.parent)
    jobs = [] if args.no_adtof else [('ADTOF', ['adtof'])]
    for spec in args.models.split(','):
        label, path = spec.split('=')
        jobs.append((f'BMY-{label}', ['bmy', '--model', str(Path(path).resolve())]))
    out = {}
    for label, a in jobs:
        f = res_dir / f'{label}.json'
        cmd = ['node', str(HERE / 'transcribe.mjs'), repo, a[0], str(f), *wavs, *a[1:]]
        subprocess.run(cmd, check=True, cwd=args.node_dir)
        out[label] = json.load(open(f))
    return out


# ----------------------------------------------------------------------------- scoring
def match(ref, est):
    """Greedy one-to-one matching within TOL. -> (tp pairs, unmatched ref, unmatched est)."""
    ref, est = sorted(ref), sorted(est)
    used, pairs, miss = set(), [], []
    for r in ref:
        best, bd = None, TOL + 1
        for j, e in enumerate(est):
            if j not in used and abs(e - r) <= TOL and abs(e - r) < bd:
                best, bd = j, abs(e - r)
        if best is None:
            miss.append(r)
        else:
            used.add(best)
            pairs.append((r, est[best]))
    return pairs, miss, [e for j, e in enumerate(est) if j not in used]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(truth, results):
    """-> per transcriber: counts per (scenario, kit, cond, class) and confusion / ghost notes."""
    S = {}
    runs = [(l, r, l == 'ADTOF') for l, r in results.items()]
    runs += [(l + ' (as 5 classes, like ADTOF)', r, True) for l, r in results.items() if l != 'ADTOF']
    for label, res, five in runs:
        classes = C5 if five else C9
        cnt = defaultdict(lambda: np.zeros(3, int))       # key -> tp, fp, fn
        conf = defaultdict(int)                           # (false class, what was really there) -> n
        ghost = defaultdict(lambda: np.zeros(2, int))     # velocity band -> found, total (snare only)
        for wav, ref in truth.items():
            sname, kit, cond = Path(wav).stem.split('__')
            est = res.get(wav, [])
            for c in classes:
                rc = [t for t, d, v in ref if (TO5[d] if five else d) == c]
                ec = [h['time'] for h in est if (h['cls'] if five else h.get('model')) == c]   # BMY's cls is already 5-class
                pairs, miss, extra = match(rc, ec)
                for key in [(sname, kit, cond, c), ('*', kit, cond, c), (sname, '*', '*', c), ('*', '*', cond, c), ('*', '*', '*', c)]:
                    cnt[key] += (len(pairs), len(extra), len(miss))
                for e in extra:                           # what was actually there?
                    near = sorted({(TO5[d] if five else d) for t, d, v in ref if abs(t - e) <= TOL} - {c})
                    conf[(c, '+'.join(near) if near else 'nothing')] += 1
                if c == 'snare':
                    got = {r for r, _ in pairs}
                    for t, d, v in ref:
                        if d == 'snare':
                            band = 'ghost <50' if v < 50 else ('soft 50-89' if v < 90 else 'accent 90+')
                            ghost[band] += (t in got, 1)
        S[label] = dict(cnt=cnt, conf=conf, ghost=ghost, classes=classes)
    return S


def report(S, truth, path):
    L = []
    w = L.append
    kits = sorted({Path(k).stem.split('__')[1] for k in truth})
    conds = [c for c in ('dry', 'mixed', 'shaker') if any(Path(k).stem.endswith('__' + c) for k in truth)]
    scen = list(SCENARIOS)
    w('BMY TRANSCRIBER TEST RIG')
    w(f'{len(truth)} files: {len(scen)} grooves x {len(kits)} kits ({", ".join(kits)}) x conditions. '
      f'Conditions: {", ".join(conds)} (shaker = a shaker part added; its hits count as hi-hat). Hit = within {int(TOL * 1000)} ms. ADTOF has 5 classes (toms, hats, cymbals each one class).')
    w('naked = held out of BMY training (never heard). Others: different performances, but kits BMY trained on.\n')
    for label, s in S.items():
        cnt, classes = s['cnt'], s['classes']
        w('=' * 100)
        w(f'{label}')
        w('=' * 100)
        tot = sum((cnt[('*', '*', '*', c)] for c in classes), np.zeros(3, int))
        p, r, f = prf(*tot)
        w(f'OVERALL  F1 {f:.3f}  precision {p:.3f}  recall {r:.3f}   (tp {tot[0]}, false {tot[1]}, missed {tot[2]})\n')
        w(f'{"drum":10s}' + ''.join(f'{k[:9]:>10s}' for k in kits) + ''.join(f'{c:>8s}' for c in conds) + f'{"all":>8s}   false / missed')
        for c in classes:
            row = f'{c:10s}'
            for k in kits:
                a = cnt[('*', k, 'dry', c)] + cnt[('*', k, 'mixed', c)]
                row += f'{prf(*a)[2]:10.2f}' if a.sum() else f'{"-":>10s}'
            for cond in conds:
                row += f'{prf(*cnt[("*", "*", cond, c)])[2]:8.2f}'
            a = cnt[('*', '*', '*', c)]
            row += f'{prf(*a)[2]:8.2f}   {a[1]:4d} / {a[2]:<4d}'
            w(row)
        w('\nF1 by groove (all kits, both conditions):')
        w(f'{"groove":18s}' + ''.join(f'{c[:7]:>8s}' for c in classes))
        for sn in scen:
            row = f'{sn:18s}'
            for c in classes:
                a = cnt[(sn, '*', '*', c)]
                row += f'{prf(*a)[2]:8.2f}' if a[0] + a[2] else (f'{"x" + str(a[1]):>8s}' if a[1] else f'{"":8s}')
            w(row)
        w('  (x N = N false hits where that drum never plays)')
        w('\nFalse hits: what was really there (top 14):')
        for (c, near), n in sorted(s['conf'].items(), key=lambda kv: -kv[1])[:14]:
            w(f'  {n:5d}  {c:9s} reported where there was {near}')
        w('\nSnare by velocity (found / total):')
        for band in ('ghost <50', 'soft 50-89', 'accent 90+'):
            g = s['ghost'][band]
            if g[1]:
                w(f'  {band:11s} {g[0]:4d} / {g[1]:<4d} ({g[0] / g[1]:.0%})')
        w('')
    Path(path).write_text('\n'.join(L))
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kits', default='naked,drs,muldjord,gogodze')
    ap.add_argument('--kit-dir', default='/content/kits')
    ap.add_argument('--out', default='rig')
    ap.add_argument('--models', default=f'current={HERE.parent.parent / "models/bmy_drums.onnx"}')
    ap.add_argument('--node-dir', default='.', help='folder whose node_modules has onnxruntime-web')
    ap.add_argument('--no-adtof', action='store_true')
    ap.add_argument('--sf2', default='', help='GeneralUser GS .sf2: adds a "shaker" condition (needs fluidsynth)')
    ap.add_argument('--skip-render', action='store_true')
    args = ap.parse_args()
    truth = json.load(open(Path(args.out) / 'truth.json')) if args.skip_render else build(args)
    results = transcribe(args, list(truth))
    print(report(score(truth, results), truth, Path(args.out) / 'REPORT.txt'))
