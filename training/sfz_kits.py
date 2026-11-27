"""Minimal SFZ drum-kit renderer for BMY transcriber training (numpy + soundfile, no audio engine).

Covers what free drum kits use: #define / #include, <control> <global> <master> <group> <region> inheritance,
key / velocity / CC / random / round-robin selection, multi-mic regions (summed to mono), velocity crossfades,
group / off_by choke (hi-hats), volume / amplitude (+ CC), offset, delay and tune. Ignores pan, filters and
envelopes beyond a short choke fade: the output is a mono drum performance, not a polished mix.

Usage:
    kit = SfzKit('/content/kits/gogodze/Programs/Kit.sfz', notes=KITS['gogodze']['notes'])
    y = kit.render(hits, seconds)       # hits: [(time_s, gm_pitch, velocity)]  ->  mono float32 44.1 kHz
"""
from __future__ import annotations

import math
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

SR = 44100
NOTE_NAMES = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}


# ----------------------------------------------------------------------------- free kits (licence-clean)
# repo: github.com/sfzinstruments/<repo>; sfz: program inside it; notes: articulation -> [(note, {cc: value})]
# (first entry that has regions is used). Articulations come from E-GMD pitches via ART_OF_PITCH.
KITS = {
    'gogodze':   dict(repo='karoryfer.gogodze-phu-vol-ii', sfz='Programs/Kit.sfz', licence='CC0-1.0', author='Karoryfer Samples',
                      notes={'tomHigh': [(45, {})], 'tomMid': [(43, {})], 'tomFloor': [(41, {})],
                             'hhClosed': [(42, {}), (46, {4: 0})], 'hhOpen': [(46, {4: 127}), (46, {4: 100})]}),
    'big_rusty': dict(repo='karoryfer.big-rusty-drums', sfz='Programs/01-full.sfz', licence='CC0-1.0', author='Karoryfer Samples',
                      notes={'tomHigh': [(47, {})], 'tomMid': [(45, {})], 'tomFloor': [(43, {}), (41, {})],
                             'hhClosed': [(42, {}), (46, {4: 0})], 'hhOpen': [(46, {4: 127}), (46, {4: 100})],
                             'crash': [(49, {})], 'crash2': [(49, {})]}),
    'swirly':    dict(repo='karoryfer.swirly-drums', sfz='Programs/Full_kit.sfz', licence='CC0-1.0', author='Karoryfer Samples',
                      notes={'tomHigh': [(48, {}), (47, {})], 'tomMid': [(45, {})], 'tomFloor': [(43, {}), (41, {})],
                             'hhClosed': [(42, {}), (46, {4: 0})], 'hhOpen': [(46, {4: 127}), (46, {4: 100})],
                             'crash': [(49, {}), (57, {})], 'crash2': [(57, {}), (49, {})], 'rideBell': [(51, {})]}),
    'unruly':    dict(repo='karoryfer.unruly-drums', sfz='Programs/01-kit-sticks.sfz', licence='CC0-1.0', author='Karoryfer Samples',
                      notes={'tomHigh': [(47, {})], 'tomMid': [(45, {})], 'tomFloor': [(41, {})],
                             'hhClosed': [(42, {}), (46, {4: 0})], 'hhOpen': [(46, {4: 127}), (46, {4: 100})],
                             'crash': [(49, {}), (57, {})], 'crash2': [(57, {}), (49, {})]}),
    'muldjord':  dict(repo='DrumGizmo.MuldjordKit', sfz='DrumGizmo/MuldjordKit/Stereo/DrumGizmo MuldjordKit.sfz', licence='CC-BY-4.0',
                      author='Lars Muldjord / DrumGizmo',
                      notes={'tomHigh': [(48, {}), (47, {})], 'tomMid': [(45, {})], 'tomFloor': [(41, {})], 'hhPedal': [(42, {})],
                             'ride': [(51, {}), (59, {})], 'rideBell': [(53, {}), (55, {})]}),
    'drs':       dict(repo='DrumGizmo.DRSKit', sfz='DrumGizmo/DRSKit/Stereo/DrumGizmo DRSKit.sfz', licence='CC-BY-4.0', author='DrumGizmo team',
                      notes={'tomHigh': [(47, {})], 'tomMid': [(43, {})], 'tomFloor': [(41, {})], 'hhClosed': [(42, {}), (56, {})],
                             'crash': [(49, {}), (54, {})], 'crash2': [(57, {}), (55, {})]}),
    'naked':     dict(repo='WilkinsonAudio.NakedDrums', sfz='Wilkinson Audio/Naked Drums/Stereo/Naked Drums.sfz', licence='CC-BY-4.0',
                      author='Wilkinson Audio',
                      notes={'tomHigh': [(48, {}), (47, {})], 'tomMid': [(45, {})], 'tomFloor': [(43, {}), (41, {})],
                             'hhClosed': [(42, {}), (39, {})], 'hhOpen': [(49, {}), (46, {})],
                             'crash': [(55, {})], 'crash2': [(57, {})]}),
    'sm_drums':  dict(repo='SMDrums', sfz='Programs/SM_Drums_kit.sfz', licence='Public domain', author='Scott McLean, Tod Stillwell, Suleiman Ali',
                      notes={'snare': [(38, {}), (40, {})], 'tomHigh': [(48, {}), (47, {})], 'tomMid': [(45, {})], 'tomFloor': [(43, {})],
                             'hhClosed': [(42, {})], 'hhPedal': [(50, {})], 'hhOpen': [(46, {})],
                             'crash': [(55, {}), (57, {})], 'crash2': [(54, {}), (56, {})], 'ride': [(60, {}), (62, {})],
                             'rideBell': [(61, {}), (63, {})]}),
}
GM = {'kick': [(36, {}), (35, {})], 'snare': [(38, {}), (40, {})], 'sideStick': [(37, {}), (38, {})],
      'tomHigh': [(48, {}), (50, {}), (47, {})], 'tomMid': [(45, {}), (47, {}), (48, {})], 'tomFloor': [(43, {}), (41, {}), (45, {})],
      'hhClosed': [(42, {}), (44, {})], 'hhPedal': [(44, {}), (42, {})], 'hhOpen': [(46, {})],
      'crash': [(49, {}), (57, {})], 'crash2': [(57, {}), (49, {})], 'ride': [(51, {}), (59, {})], 'rideBell': [(53, {}), (51, {})]}
ART_OF_PITCH = {36: 'kick', 35: 'kick', 38: 'snare', 40: 'snare', 39: 'snare', 37: 'sideStick',
                48: 'tomHigh', 50: 'tomHigh', 45: 'tomMid', 47: 'tomMid', 43: 'tomFloor', 58: 'tomFloor', 41: 'tomFloor',
                42: 'hhClosed', 22: 'hhClosed', 44: 'hhPedal', 46: 'hhOpen', 26: 'hhOpen',
                49: 'crash', 55: 'crash', 57: 'crash2', 52: 'crash2', 51: 'ride', 59: 'ride', 53: 'rideBell'}


def note_number(v: str) -> int:
    v = v.strip().lower()
    if re.fullmatch(r'-?\d+', v):
        return int(v)
    m = re.fullmatch(r'([a-g])([#b]?)(-?\d+)', v)
    if not m:
        raise ValueError(v)
    n = NOTE_NAMES[m[1]] + {'#': 1, 'b': -1, '': 0}[m[2]]
    return n + (int(m[3]) + 1) * 12                   # c4 = 60


# ----------------------------------------------------------------------------- parsing
def _strip_comments(t: str) -> str:
    t = re.sub(r'/\*.*?\*/', ' ', t, flags=re.S)
    return re.sub(r'//[^\n]*', '', t)


def preprocess(path: Path, root: Path, defs: dict, depth=0) -> str:
    """Inline #include (relative to the top-level file's folder) and substitute #define variables, in order."""
    if depth > 20:
        raise RecursionError(str(path))
    text = _strip_comments(path.read_text(encoding='utf-8', errors='replace'))
    def one(m):                                         # longest defined name that prefixes the token ($MIC_$ART -> $MIC + _)
        tok = m[0]
        for e in range(len(tok), 1, -1):
            if tok[:e] in defs:
                return defs[tok[:e]] + tok[e:]
        return tok
    sub = lambda s: re.sub(r'\$\w+', one, s)
    out, pos = [], 0
    for m in re.finditer(r'#define\s+(\$\w+)\s+(\S+)|#include\s+"([^"]+)"', text):
        out.append(sub(text[pos:m.start()]))
        if m[1]:
            defs[m[1]] = sub(m[2])
        else:
            inc = (root / sub(m[3]).replace('\\', '/'))
            if inc.exists():
                out.append('\n' + preprocess(inc, root, defs, depth + 1) + '\n')
        pos = m.end()
    out.append(sub(text[pos:]))
    return ''.join(out)


_OPCODE = re.compile(r'([A-Za-z0-9_]+)=(.*?)(?=\s+[A-Za-z0-9_]+=|\s*<|\s*$)', re.S)


def parse(sfz_path: str):
    """-> (regions [dict of opcodes], control dict). Each region carries inherited opcodes and '_path'."""
    sfz = Path(sfz_path)
    root = sfz.parent
    text = preprocess(sfz, root, {})
    regions, control = [], {'cc': {}}
    levels = {'global': {}, 'master': {}, 'group': {}}
    default_path, note_offset = '', 0
    for hm in re.finditer(r'<(\w+)>([^<]*)', text):
        head, body = hm[1].lower(), hm[2]
        ops = {k.lower(): v.strip() for k, v in _OPCODE.findall(body)}
        if head == 'control':
            default_path = ops.get('default_path', default_path).replace('\\', '/')
            note_offset = int(ops.get('note_offset', 0)) + 12 * int(ops.get('octave_offset', 0))
            for k, v in ops.items():
                if k.startswith('set_cc'):
                    control['cc'][int(k[6:])] = float(v)
                elif k.startswith('set_hdcc'):
                    control['cc'][int(k[8:])] = float(v) * 127
        elif head == 'global':
            levels = {'global': ops, 'master': {}, 'group': {}}
        elif head == 'master':
            levels['master'], levels['group'] = ops, {}
        elif head == 'group':
            levels['group'] = ops
        elif head == 'region':
            r = {**levels['global'], **levels['master'], **levels['group'], **ops}
            if 'sample' not in r or r['sample'].startswith('*'):      # *silence, *sine...: generators, not samples
                continue
            r['_path'] = str(root / (default_path + r['sample'].replace('\\', '/')))
            r['_note_offset'] = note_offset
            regions.append(r)
    return regions, control


# ----------------------------------------------------------------------------- region model
class Region:
    __slots__ = ('path', 'lokey', 'hikey', 'center', 'lovel', 'hivel', 'cond', 'lorand', 'hirand', 'seq_len', 'seq_pos',
                 'gain', 'gain_cc', 'veltrack', 'velcurve', 'xf', 'offset', 'offset_cc', 'delay', 'delay_cc', 'tune',
                 'tune_cc', 'keytrack', 'group', 'off_by', 'off_time', 'trigger', 'counter', 'idx')

    def __init__(self, r: dict, idx: int):
        g = lambda k, d=None: r.get(k, d)
        def f(k, d=0.0):
            try:
                return float(r[k]) if k in r and r[k] != '' else d
            except ValueError:                          # undefined $variable or text: ignore, like sfizz
                return d
        self.idx, self.path = idx, r['_path']
        no = r['_note_offset']
        if 'key' in r:
            k = note_number(r['key']); self.lokey = self.hikey = self.center = k
        else:
            self.lokey, self.hikey = note_number(g('lokey', '0')), note_number(g('hikey', '127'))
            self.center = note_number(g('pitch_keycenter', '60'))
        self.lokey += no; self.hikey += no; self.center += no
        self.lovel, self.hivel = int(f('lovel', 0)), int(f('hivel', 127))
        self.cond = []                                  # (cc, lo, hi)
        for k in r:
            m = re.fullmatch(r'(lo|hi)cc(\d+)', k)
            if m and int(m[2]) not in [c for c, _, _ in self.cond]:
                cc = int(m[2])
                self.cond.append((cc, f(f'locc{cc}', 0), f(f'hicc{cc}', 127)))
        self.lorand, self.hirand = f('lorand', 0.0), f('hirand', 1.0)
        self.seq_len, self.seq_pos = int(f('seq_length', 1)), int(f('seq_position', 1))
        self.gain = 10 ** (f('volume', 0) / 20) * f('amplitude', 100) / 100 * 10 ** (f('gain', 0) / 20)
        self.gain_cc = []                               # (cc, kind, depth): kind 'amp' (%), 'vol' (dB)
        for k, v in r.items():
            m = re.fullmatch(r'(amplitude|volume|gain)_(?:on)?cc(\d+)', k)
            if m:
                self.gain_cc.append((int(m[2]), 'amp' if m[1] == 'amplitude' else 'vol', f(k)))
        self.veltrack = f('amp_veltrack', 100) / 100
        pts = sorted((int(k[13:]), f(k)) for k, v in r.items() if re.fullmatch(r'amp_velcurve_\d+', k))
        self.velcurve = pts or None
        self.xf = []                                    # (kind, lo, hi, source): fade in / out over vel or cc
        for k in r:
            m = re.fullmatch(r'xf(in|out)_lo(vel|cc(\d+))', k)
            if m:
                lo = f(k); hi = f(f'xf{m[1]}_hi{m[2]}', lo)
                self.xf.append((m[1], lo, hi, 'vel' if m[2] == 'vel' else int(m[3])))
        self.offset, self.delay = int(f('offset', 0)), f('delay', 0.0)
        self.offset_cc = [(int(re.sub(r'\D', '', k.split('cc')[-1])), f(k)) for k, v in r.items() if re.fullmatch(r'offset_(?:on)?cc\d+', k)]
        self.delay_cc = [(int(re.sub(r'\D', '', k.split('cc')[-1])), f(k)) for k, v in r.items() if re.fullmatch(r'delay_(?:on)?cc\d+', k)]
        self.tune = f('tune', 0) + 100 * f('transpose', 0) + f('pitch', 0)
        self.tune_cc = [(int(re.sub(r'\D', '', k.split('cc')[-1])), f(k)) for k, v in r.items()
                        if re.fullmatch(r'(?:tune|pitch)_(?:on)?cc\d+', k)]
        self.keytrack = f('pitch_keytrack', 100)
        self.group, self.off_by = int(f('group', 0)), int(f('off_by', 0))
        self.off_time = 0.006 if g('off_mode', 'fast') == 'fast' else max(0.006, f('ampeg_release', 0.05))
        self.off_time = float(g('off_time', self.off_time))
        self.trigger = g('trigger', 'attack')
        self.counter = 0

    def matches(self, key, vel, cc):
        if not (self.lokey <= key <= self.hikey and self.lovel <= vel <= self.hivel):
            return False
        return all(lo <= cc.get(c, 0) <= hi for c, lo, hi in self.cond)

    def level(self, vel, cc):
        if self.velcurve:
            xs, ys = zip(*([(0, 0.0)] + self.velcurve + ([(127, 1.0)] if self.velcurve[-1][0] < 127 else [])))
            v = float(np.interp(vel, xs, ys))
        else:
            v = 1 - self.veltrack + self.veltrack * (vel / 127) ** 2
        a = self.gain * v
        for c, kind, d in self.gain_cc:
            x = cc.get(c, 0) / 127
            a *= (d / 100 * x) if kind == 'amp' else 10 ** (d * x / 20)
        for kind, lo, hi, src in self.xf:
            x = vel if src == 'vel' else cc.get(src, 0)
            t = 1.0 if hi <= lo else min(1, max(0, (x - lo) / (hi - lo)))
            a *= math.sqrt(t if kind == 'in' else 1 - t)
        return a


# ----------------------------------------------------------------------------- the kit
class SfzKit:
    def __init__(self, sfz_path: str, notes: dict | None = None, cache_mb: int = 1500):
        raw, control = parse(sfz_path)
        sw = next((r['sw_default'] for r in raw if r.get('sw_default', '').lstrip('-').isdigit()), None)
        sw = None if sw is None else int(sw)

        def keyswitch_ok(r):                            # keep the default keyswitch's articulations only
            if 'sw_down' in r or 'sw_up' in r or 'sw_previous' in r:
                return False                            # needs a held key (cymbal chokes etc.)
            try:
                if 'sw_last' in r:
                    return sw is not None and int(r['sw_last']) == sw
                if 'sw_lolast' in r or 'sw_hilast' in r:
                    return sw is not None and int(r.get('sw_lolast', 0)) <= sw <= int(r.get('sw_hilast', 127))
            except ValueError:
                return False
            return True
        self.regions = [Region(r, i) for i, r in enumerate(r for r in raw if keyswitch_ok(r))]
        self.cc = dict(control['cc'])
        self.own = dict(notes or {})                    # articulations listed for this kit: any of them, chosen per render
        self.notes = {**GM, **self.own}
        self._cache, self._cache_bytes, self._cap = OrderedDict(), 0, cache_mb << 20
        self.name = Path(sfz_path).stem

    # sample cache (mono float32 at 44.1 kHz)
    def sample(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        import soundfile as sf
        y, sr = sf.read(path, dtype='float32', always_2d=True)
        y = y.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            g = math.gcd(SR, sr)
            y = resample_poly(y, SR // g, sr // g).astype(np.float32)
        self._cache[path] = y
        self._cache_bytes += y.nbytes
        while self._cache_bytes > self._cap and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
        return y

    def regions_for(self, key, vel, cc, rnd):
        hits, r_val = [], rnd.random()
        for reg in self.regions:
            if reg.trigger not in ('attack', 'first', 'legato') or not reg.matches(key, vel, cc):
                continue
            reg.counter += 1
            if reg.seq_len > 1 and ((reg.counter - 1) % reg.seq_len) + 1 != reg.seq_pos:
                continue
            if not (reg.lorand <= r_val < reg.hirand or (reg.hirand >= 1 and r_val >= reg.lorand)):
                continue
            hits.append(reg)
        return hits

    def resolve(self, art, rnd=None):
        """Articulation -> (note, cc overrides) that this kit has regions for (kit's own list: random pick)."""
        ok = [(note, cco) for note, cco in self.notes.get(art, [])
              if any(r.matches(note, 100, {**self.cc, **cco}) for r in self.regions)]
        if not ok:
            return None
        return rnd.choice(ok) if (rnd and art in self.own) else ok[0]

    def available(self):
        return {a: self.resolve(a) for a in self.notes}

    def render(self, hits, seconds, rnd=None, class_gain_db=None):
        """hits [(time_s, gm_or_td17_pitch, velocity)] -> (mono float32 at 44.1 kHz, the hits it could play).
        Labels must come from the second value: a kit without (say) cymbals skips those hits."""
        import random as _r
        rnd = rnd or _r.Random(0)
        n = int(seconds * SR)
        out = np.zeros(n + SR * 8, np.float32)
        res = {}
        voices, played = [], []                           # voice: [start, sample, gain, off_by, stop, fade, articulation]
        for t, pitch, vel in sorted(hits):
            art = ART_OF_PITCH.get(pitch)
            if art is None:
                continue
            if art not in res:
                res[art] = self.resolve(art, rnd)
            if res[art] is None:
                continue
            note, cco = res[art]
            cc = {**self.cc, **cco}
            vel = max(1, min(127, int(vel)))
            regs = self.regions_for(note, vel, cc, rnd)
            if not regs:
                continue
            played.append((t, pitch, vel))
            start = int(t * SR)
            groups = {r.group for r in regs if r.group}
            closing = art in ('hhClosed', 'hhPedal')     # closing the hats stops a ringing open hat, whatever the kit's groups say
            for v in voices:                              # choke: new note in group G stops voices with off_by == G
                if v[4] is None and v[0] < start and ((v[3] and v[3] in groups) or (closing and v[6] == 'hhOpen')):
                    v[4] = (start, v[5] if not closing else max(v[5], 0.03))
            g_extra = 10 ** ((class_gain_db or {}).get(art, 0) / 20)
            for r in regs:
                y = self.sample(r.path)
                off = r.offset + int(sum(d * cc.get(c, 0) / 127 for c, d in r.offset_cc))
                y = y[min(len(y), max(0, off)):]
                cents = r.tune + sum(d * cc.get(c, 0) / 127 for c, d in r.tune_cc) + r.keytrack * (note - r.center)
                if abs(cents) > 1:
                    ratio = 2 ** (cents / 1200)
                    idx = np.arange(0, len(y) - 1, ratio)
                    y = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
                d = r.delay + sum(dd * cc.get(c, 0) / 127 for c, dd in r.delay_cc)
                a = r.level(vel, cc) * g_extra
                if a <= 1e-5 or len(y) == 0:
                    continue
                voices.append([start + int(d * SR), y, a, r.off_by, None, r.off_time, art])
        for s, y, a, _, stop, _, _ in voices:
            seg = y * a
            if stop is not None:
                st, ft = stop
                keep = max(0, st - s)
                fade = int(ft * SR)
                seg = seg[:keep + fade].copy()
                if len(seg) > keep:
                    seg[keep:] *= np.linspace(1, 0, len(seg) - keep, dtype=np.float32)
            e = min(len(out), s + len(seg))
            if e > s >= 0:
                out[s:e] += seg[:e - s]
        return out[:n], played


def fetch_kit(name: str, dest: str = '/content/kits') -> str:
    """Shallow-clone a kit from github.com/sfzinstruments (once) and return the path of its .sfz program."""
    import subprocess
    k = KITS[name]
    d = Path(dest) / name
    if not (d / k['sfz']).exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '-q', '--depth', '1', f"https://github.com/sfzinstruments/{k['repo']}", str(d)], check=True)
        subprocess.run(['rm', '-rf', str(d / '.git')], check=True)          # the samples are all we need: halves the disk use
    return str(d / k['sfz'])
