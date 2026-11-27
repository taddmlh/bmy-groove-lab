"""Test-rig grooves with known answers. Each scenario -> hits [(time_s, gm_pitch, velocity)].

The grooves target what goes wrong on real stems: ghost notes, kicks heard as snare, swing / triplet hats,
open-close hats, ride vs crash, tom fills, flams / drags / ruffs, linear patterns, and each drum alone.
Timing and velocity are humanised with a fixed seed, so every run is identical.
"""
import random

K, S, HC, HP, HO, CR, RD, BL, TH, TM, TF = 36, 38, 42, 44, 46, 49, 51, 53, 48, 45, 43


class Groove:
    def __init__(self, bpm, seed, swing=0.0, jitter_ms=6):
        self.beat, self.r, self.swing, self.jit = 60 / bpm, random.Random(seed), swing, jitter_ms / 1000
        self.hits = []

    def at(self, bar, beat, pitch, vel, sub=0.0, beats_per_bar=4):
        """beat: 0-based beat in bar; sub: fraction of a beat (0.5 = 'and'). Swing delays off-beat 8ths."""
        frac = sub
        if self.swing and abs(sub - 0.5) < 1e-6:
            frac = 0.5 + self.swing / 2                  # swing 1/3 -> triplet feel (0.667)
        t = 0.3 + (bar * beats_per_bar + beat + frac) * self.beat + self.r.uniform(-self.jit, self.jit)
        v = max(1, min(127, int(vel + self.r.uniform(-6, 6))))
        self.hits.append((round(max(0.0, t), 4), pitch, v))


def rock8():
    g = Groove(100, 1)
    for b in range(8):
        for i in range(8):
            g.at(b, i // 2, HC, 95 if i % 2 == 0 else 70, (i % 2) / 2)
        g.at(b, 0, K, 110); g.at(b, 2, K, 105); g.at(b, 2, K, 95, 0.5)
        g.at(b, 1, S, 112); g.at(b, 3, S, 115)
    return g.hits


def ghost16():
    g = Groove(92, 2)
    ghost_vels = [18, 25, 32, 40, 48]
    for b in range(8):
        for i in range(16):
            g.at(b, i // 4, HC, 85 if i % 4 == 0 else (70 if i % 2 == 0 else 55), (i % 4) / 4)
        g.at(b, 0, K, 110); g.at(b, 1, K, 100, 0.75); g.at(b, 2, K, 105, 0.5)
        g.at(b, 1, S, 115); g.at(b, 3, S, 118)
        for beat, sub in [(0, 0.75), (1, 0.5), (2, 0.25), (3, 0.75)]:
            g.at(b, beat, S, ghost_vels[(b + beat) % 5], sub)
    return g.hits


def kick_heavy():
    g = Groove(110, 3)
    for b in range(8):
        for i in range(8):
            g.at(b, i // 2, HC, 80, (i % 2) / 2)
        for beat in range(4):
            g.at(b, beat, K, 115)
        g.at(b, 1, K, 90, 0.75); g.at(b, 3, K, 95, 0.25); g.at(b, 3, K, 95, 0.5)
        if b % 2:
            g.at(b, 3, S, 112)
    return g.hits


def shuffle_triplets():
    """EWTRTW-style: triplet hats over a straight backbeat."""
    g = Groove(76, 4)
    for b in range(6):
        for beat in range(4):
            for k in range(3):
                g.at(b, beat, HC, 88 if k == 0 else 62, k / 3)
        g.at(b, 0, K, 112); g.at(b, 2, K, 108); g.at(b, 2, K, 90, 2 / 3)
        g.at(b, 1, S, 115); g.at(b, 3, S, 115)
    return g.hits


def open_close():
    g = Groove(100, 5)
    for b in range(8):
        for i in range(8):
            open_ = (i == 7)
            g.at(b, i // 2, HO if open_ else HC, 95 if i % 2 == 0 else 72, (i % 2) / 2)
        if b >= 4:
            g.at(b, 1, HP, 70); g.at(b, 3, HP, 70)       # foot hats under the stick pattern
        g.at(b, 0, K, 110); g.at(b, 2, K, 105)
        g.at(b, 1, S, 112); g.at(b, 3, S, 115)
    return g.hits


def ride_jazz():
    g = Groove(140, 6, swing=1 / 3)
    for b in range(8):
        for beat in range(4):
            g.at(b, beat, BL if (beat == 0 and b % 2 == 0) else RD, 90 if beat % 2 else 80)
            if beat % 2:
                g.at(b, beat, RD, 65, 0.5)
            g.at(b, beat, K, 38)                         # feathered kick
        g.at(b, 1, HP, 75); g.at(b, 3, HP, 75)
        g.at(b, (b * 3) % 4, S, 45, 0.5)                 # comping
        if b % 4 == 3:
            g.at(b, 3, S, 100, 0.5)
    return g.hits


def crash_accents():
    g = Groove(120, 7)
    for b in range(8):
        for i in range(8):
            if not (i == 0 and b % 2 == 0):
                g.at(b, i // 2, HC, 85 if i % 2 == 0 else 65, (i % 2) / 2)
        if b % 2 == 0:
            g.at(b, 0, CR, 118)
        g.at(b, 0, K, 115); g.at(b, 2, K, 108); g.at(b, 1, S, 112); g.at(b, 3, S, 116)
    return g.hits


def tom_fills():
    g = Groove(100, 8)
    for b in range(8):
        if b % 2 == 0:
            for i in range(8):
                g.at(b, i // 2, HC, 85 if i % 2 == 0 else 65, (i % 2) / 2)
            g.at(b, 0, K, 110); g.at(b, 2, K, 105); g.at(b, 1, S, 112); g.at(b, 3, S, 115)
            if b:
                g.at(b, 0, CR, 115)
        else:
            seq = [S, S, TH, TH, TH, TH, TM, TM, TM, TM, TF, TF, TF, TF, K, TF] if b % 4 == 1 else \
                  [TH, TM, TF, K, TH, TM, TF, K, S, TH, TM, TF, S, S, TF, TF]
            for i, p in enumerate(seq):
                g.at(b, i // 4, p, 100 if i % 4 == 0 else 85, (i % 4) / 4)
    return g.hits


def rudiments():
    """CS End-style: flams, drags, ruffs and short rolls on snare, over kick and hats."""
    g = Groove(90, 9, jitter_ms=3)
    beat = g.beat
    for b in range(6):
        g.at(b, 0, K, 110); g.at(b, 2, K, 105)
        for i in range(4):
            g.at(b, i, HC, 70)
        kind = b % 3
        for main_beat in (1, 3):
            g.at(b, main_beat, S, 112)
            t_main = g.hits[-1][0]
            if kind == 0:                                 # flam: one grace ~30 ms before
                g.hits.append((round(t_main - 0.03, 4), S, 45))
            elif kind == 1:                               # drag: two graces (32nds) before
                g.hits += [(round(t_main - beat / 4, 4), S, 50), (round(t_main - beat / 8, 4), S, 55)]
            else:                                         # ruff: three graces
                g.hits += [(round(t_main - 3 * beat / 8, 4), S, 50), (round(t_main - beat / 4, 4), S, 55),
                           (round(t_main - beat / 8, 4), S, 60)]
        if b == 5:                                        # six-stroke roll into the end
            for k in range(6):
                g.at(b, 3, S, [110, 50, 55, 105, 50, 112][k], 0.5 + k / 12)
    return g.hits


def linear_funk():
    g = Groove(96, 10)
    pat = [K, HC, S, HC, HC, K, HC, S, K, HC, S, HC, HC, K, S, HC]
    for b in range(8):
        for i, p in enumerate(pat):
            v = {K: 108, S: 110 if i in (4, 12) else 70, HC: 75}[p]
            g.at(b, i // 4, p, v, (i % 4) / 4)
    return g.hits


def isolated():
    """Each drum alone, soft to loud: anything detected as another drum is a pure confusion."""
    g = Groove(100, 11, jitter_ms=0)
    order = [K, S, TH, TM, TF, HC, HP, HO, CR, RD, BL]
    for n, p in enumerate(order):
        for k, v in enumerate([30, 60, 90, 120]):
            g.at(n, k, p, v)
    return g.hits


SCENARIOS = {f.__name__: f for f in [rock8, ghost16, kick_heavy, shuffle_triplets, open_close, ride_jazz,
                                     crash_accents, tom_fills, rudiments, linear_funk, isolated]}
