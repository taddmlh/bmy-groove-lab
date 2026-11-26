# Render GeneralUser GS "Standard 1" (bank 128, preset 0) drum notes at BMY's 6 velocity centres.
# Minimal SoundFont 2 voice: zone matching (preset+instrument key/vel ranges), pitch (root/coarse/fine/
# scale/correction), sample offsets + loops, volume envelope, modulation envelope -> filter cutoff,
# 2-pole low-pass with Q, FluidSynth-style attenuation (x0.4) incl. default velocity modulators.
import logging, json, struct, sys, math
logging.disable(logging.WARNING)
import numpy as np, soundfile as sf
from sf2utils.sf2parse import Sf2File

SF2 = '/mnt/user-data/uploads/BMYDrummer/BMYDrummerAU/GeneralUser-GS.sf2'
OUT = sys.argv[1]
SR = 44100
VELS = {'gh': 28, 'sf': 46, 'n1': 66, 'n2': 82, 'a1': 102, 'a2': 116}
NOTES = {36: 'kick', 37: 'sidestick', 38: 'snare', 40: 'snareAlt', 42: 'closedHat', 46: 'openHat', 51: 'ride',
         48: 'tomHigh', 47: 'tomMid', 43: 'tomFloor', 49: 'crash', 60: 'oneShotA', 61: 'oneShotB', 62: 'oneShotC'}
MAXLEN = {'crash': 3.0, 'ride': 2.5, 'openHat': 0.9, 'tomFloor': 1.6, 'tomMid': 1.4, 'tomHigh': 1.2}

raw = open(SF2, 'rb').read()
i = raw.find(b'smpl'); n = struct.unpack('<I', raw[i + 4:i + 8])[0]
SMPL = np.frombuffer(raw[i + 8:i + 8 + n], dtype='<i2').astype(np.float32) / 32768.0
sfile = Sf2File(open(SF2, 'rb'))
kit = [p for p in sfile.presets if getattr(p, 'bank', None) == 128 and p.preset == 0][0]

s16 = lambda v: v - 65536 if v >= 32768 else v
DEF = {8: 13500, 21: -12000, 23: -12000, 25: -12000, 26: -12000, 27: -12000, 28: -12000, 30: -12000,
       33: -12000, 34: -12000, 35: -12000, 36: -12000, 38: -12000, 56: 100, 58: -1}
def gens(bag):
    return {k: s16(v.amount) for k, v in bag.gens.items() if k not in (41, 43, 44, 53)} if hasattr(bag, 'gens') else {}
def inr(rng, x): return rng is None or rng[0] <= x <= rng[1]
def tc(v, d=-12000): return 2.0 ** ((v if v is not None else d) / 1200.0)

def zones(note, vel):
    pglob = {}
    for pb in kit.bags:
        if pb.instrument is None: pglob = gens(pb)
    out = []
    for pb in kit.bags:
        if pb.instrument is None or not inr(pb.key_range, note) or not inr(pb.velocity_range, vel): continue
        pg = dict(pglob); pg.update(gens(pb))
        iglob = {}
        for ib in pb.instrument.bags:
            if ib.sample is None: iglob = gens(ib)
        for ib in pb.instrument.bags:
            if ib.sample is None or not inr(ib.key_range, note) or not inr(ib.velocity_range, vel): continue
            g = dict(iglob); g.update(gens(ib))
            for k, v in pg.items():                       # preset generators add to instrument ones
                if k in (54, 57, 58): continue
                g[k] = g.get(k, DEF.get(k, 0)) + v
            out.append((ib.sample, g))
    return out

def env(g, base, length, linear_db=True):
    """Volume (dB attenuation) or mod (0..1) envelope while the note is held."""
    d, a, h, dc = tc(g.get(base)), tc(g.get(base + 1)), tc(g.get(base + 2)), tc(g.get(base + 3))
    sus = g.get(base + 4, 0)
    t = np.arange(length) / SR
    if linear_db:   # volume: attenuation in dB
        susdb = min(96.0, max(0.0, sus / 10.0))
        att = np.full(length, 96.0)
        atk = (t >= d) & (t < d + a); att[atk] = 20 * np.log10(np.maximum(1e-5, (t[atk] - d) / a)) * -1
        hold = (t >= d + a) & (t < d + a + h); att[hold] = 0
        dec = t >= d + a + h; att[dec] = np.minimum(susdb, 96.0 * (t[dec] - d - a - h) / dc)
        return 10 ** (-np.minimum(att, 96) / 20)
    else:           # mod env: linear 0..1, sustain in 0.1% decrease
        susl = 1 - min(1000, max(0, sus)) / 1000.0
        lv = np.zeros(length)
        atk = (t >= d) & (t < d + a); lv[atk] = (t[atk] - d) / a
        hold = (t >= d + a) & (t < d + a + h); lv[hold] = 1
        dec = t >= d + a + h; lv[dec] = np.maximum(susl, 1 - (t[dec] - d - a - h) / dc * (1 - susl) / max(1e-9, 1 - susl) if susl < 1 else 1)
        return lv

def lowpass(x, fc_cents, q_cb):
    """Time-varying 2-pole low-pass; fc_cents is an array (absolute cents)."""
    y = np.zeros_like(x); z1 = z2 = 0.0
    qlin = 10 ** (max(0, q_cb) / 200.0) * 0.7071
    B = 64
    for s in range(0, len(x), B):
        fc = 8.176 * 2 ** (float(np.mean(fc_cents[s:s + B])) / 1200)
        if fc >= 19000: y[s:s + B] = x[s:s + B]; continue
        w = 2 * math.pi * fc / SR; al = math.sin(w) / (2 * qlin); cw = math.cos(w)
        b0 = (1 - cw) / 2 / (1 + al); b1 = (1 - cw) / (1 + al); b2 = b0; a1 = -2 * cw / (1 + al); a2 = (1 - al) / (1 + al)
        for k in range(s, min(len(x), s + B)):        # direct form II transposed
            v = x[k]; o = b0 * v + z1; z1 = b1 * v - a1 * o + z2; z2 = b2 * v - a2 * o; y[k] = o
    return y

def voice(sample, g, note, vel, maxlen):
    st = sample.start + g.get(0, 0) + 32768 * g.get(4, 0)
    en = sample.end + g.get(1, 0) + 32768 * g.get(12, 0)
    # sf2utils gives loop points relative to the sample's start
    ls = sample.start + sample.start_loop + g.get(2, 0) + 32768 * g.get(45, 0)
    le = sample.start + sample.end_loop + g.get(3, 0) + 32768 * g.get(50, 0)
    looped = g.get(54, 0) in (1, 3)
    root = g.get(58, -1); root = sample.original_pitch if root is None or root < 0 else root
    cents = (note - root) * g.get(56, 100) + g.get(51, 0) * 100 + g.get(52, 0) + sample.pitch_correction
    step = 2 ** (cents / 1200) * sample.sample_rate / SR
    L = int(maxlen * SR)
    pos = np.arange(L) * step
    if looped and le > ls + 4:
        span = le - ls; p = st + pos
        over = p >= le; p[over] = ls + np.mod(p[over] - ls, span)
    else:
        p = st + pos; L = int(min(L, (en - st - 1) / step)); p = p[:L]
    i0 = np.floor(p).astype(int); fr = p - i0
    x = SMPL[i0] * (1 - fr) + SMPL[np.minimum(i0 + 1, len(SMPL) - 1)] * fr
    x = x * env(g, 33, L)
    v = vel / 127.0
    fc = g.get(8, 13500) - 2400 * (1 - v)                       # default velocity -> cutoff modulator
    if g.get(11, 0): fc = fc + g.get(11) * env(g, 25, L, False)
    else: fc = np.full(L, float(fc))
    if np.min(fc) < 13490: x = lowpass(x, np.asarray(fc, dtype=float), g.get(9, 0))
    att_cb = g.get(48, 0) + (-400 * math.log10(max(1, vel) / 127.0))   # + default velocity -> attenuation (concave)
    return x * 10 ** (-(att_cb * 0.4) / 200.0)

manifest = {'kit': 'GeneralUser GS v2 — Standard 1 (bank 128, program 0)', 'sr': SR, 'velocities': VELS, 'notes': {}}
rendered = {}
for note, lane in NOTES.items():
    layers = []
    excl = 0
    for lab, vel in VELS.items():
        zs = zones(note, vel)
        if not zs: print('no zone', note, vel); layers.append(np.zeros(100)); continue
        parts = [voice(s, g, note, vel, MAXLEN.get(lane, 0.9)) for s, g in zs]
        excl = excl or zs[0][1].get(57, 0)
        m = np.zeros(max(len(p) for p in parts))
        for p in parts: m[:len(p)] += p
        if lane == 'openHat':   # the sample's tail rings at ~440 Hz from ~0.8 s; damp it like a foot closing the hat
            t = np.arange(len(m)) / SR
            m *= np.where(t < 0.25, 1.0, np.exp(-(t - 0.25) / 0.16))
        # trim the tail at -60 dB of this layer's peak, 5 ms fade
        pk = np.max(np.abs(m)) + 1e-9; idx = np.where(np.abs(m) > pk * 1e-3)[0]
        end = min(len(m), (idx[-1] if len(idx) else 100) + 220)
        m = m[:end]; f = min(220, len(m)); m[-f:] *= np.linspace(1, 0, f)
        layers.append(m)
    rendered[note] = (lane, layers, excl)

peak = max(np.max(np.abs(l)) for _, ls, _ in rendered.values() for l in ls)
gain = 0.89 / peak
total = 0
for note, (lane, layers, excl) in rendered.items():
    data = np.concatenate([l * gain for l in layers]).astype(np.float32)
    offs, o = [], 0
    for l in layers: offs.append([o, len(l)]); o += len(l)
    fn = f'{note}-{lane}.flac'
    sf.write(f'{OUT}/{fn}', data, SR, subtype='PCM_16')
    manifest['notes'][lane] = {'note': note, 'file': fn, 'layers': offs, 'exclusiveClass': excl}
    import os; total += os.path.getsize(f'{OUT}/{fn}')
    print(f'{note:3} {lane:10} excl {excl} layers(s): ' + ' '.join(f'{len(l)/SR:.2f}' for l in layers) + '  peak dB: ' + ' '.join(f'{20*math.log10(np.max(np.abs(l))*gain+1e-9):.0f}' for l in layers))
json.dump(manifest, open(f'{OUT}/kit.json', 'w'), indent=1)
print('total KB', total // 1024)
