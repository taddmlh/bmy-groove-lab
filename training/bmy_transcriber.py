"""BMY drum transcriber: data preparation, model, training, evaluation and ONNX export.

Licence-clean recipe for a commercially shippable drum transcriber (BMY Drummer is a commercial app):
  * Drums: E-GMD v1.0.0 (Google, CC BY 4.0) audio + aligned MIDI, read straight out of the remote zip
    (no 90 GB download); some clips re-rendered from the same MIDI through BMY's GeneralUser GS kits.
  * Backing instruments: generated here (random chord progressions, bass lines, comping, pads) and rendered
    through GeneralUser GS (GeneralUser GS License v2.0). No third-party songs at all.
  * Realism: per-clip EQ, room reverb, compression, saturation, level; mixed with backing, then run through
    Demucs htdemucs (MIT) -- the same separator Groove Lab uses -- so the model learns real separation artefacts.
  * Model: small CRNN (conv + bidirectional GRU), 9 drum classes, 100 frames/s. Our own design and weights.

Everything is resumable and keeps its results in WORK (point it at Google Drive in Colab).
Used by training/BMY_Transcriber_Colab.ipynb; tested on CPU with stand-ins for Demucs and FluidSynth.
"""
from __future__ import annotations

import io, json, math, os, random, subprocess, tempfile, time, zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- constants
SR = 44100
HOP = 441                     # 10 ms -> 100 frames per second
FPS = SR / HOP
NFFT = 2048
NMELS = 128
FMIN, FMAX = 30.0, 16000.0
CLASSES = ['kick', 'snare', 'tomHigh', 'tomMid', 'tomFloor', 'hhClosed', 'hhOpen', 'crash', 'ride']
# E-GMD (Roland TD-17) note numbers -> class. Unlisted notes (tambourine, cowbell, clave...) are ignored.
PITCH_CLASS = {
    36: 'kick', 35: 'kick',
    38: 'snare', 40: 'snare', 37: 'snare', 39: 'snare',
    48: 'tomHigh', 50: 'tomHigh',
    45: 'tomMid', 47: 'tomMid',
    43: 'tomFloor', 58: 'tomFloor', 41: 'tomFloor',
    42: 'hhClosed', 22: 'hhClosed', 44: 'hhClosed',
    46: 'hhOpen', 26: 'hhOpen',
    49: 'crash', 55: 'crash', 57: 'crash', 52: 'crash',
    51: 'ride', 59: 'ride', 53: 'ride',
}
# TD-17 extras GeneralUser GS doesn't have, mapped to the nearest GM drum when re-rendering through GS kits
TD17_TO_GM = {22: 42, 26: 46, 58: 43, 55: 49, 52: 57, 59: 51}
GS_KITS = [0, 1, 2, 8, 16, 24, 25, 26, 32, 40, 48]          # BMY's GeneralUser GS kits (SFX left out)

EGMD_BASE = 'https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/'
EGMD_ZIP = EGMD_BASE + 'e-gmd-v1.0.0.zip'
EGMD_CSV = EGMD_BASE + 'e-gmd-v1.0.0.csv'
EGMD_ZIP_ZENODO = 'https://zenodo.org/records/4300943/files/e-gmd-v1.0.0.zip?download=1'
GS_SF2_URL = 'https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/HEAD/GeneralUser-GS.sf2'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------- front end
# Log-mel at 100 frames/s. Kept simple so Groove Lab can compute exactly the same thing in JavaScript:
# periodic Hann, 2048-point FFT, centred frames with reflect padding, power spectrum, 128 HTK mel bands
# 30 Hz - 16 kHz with Slaney area normalisation, then log(power + 1e-7). Input is scaled to a -1 dBFS peak first
# (Groove Lab does the same before transcribing).
_MEL = None


def mel_bank():
    global _MEL
    if _MEL is None:
        k = NFFT // 2 + 1
        hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), NMELS + 2))
        freqs = np.arange(k) * SR / NFFT
        W = np.zeros((NMELS, k), np.float64)
        for m in range(NMELS):
            lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
            W[m] = np.maximum(0, np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c))) * (2.0 / (hi - lo))
        _MEL = W.astype(np.float32)
    return _MEL


def normalise(y: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    p = float(np.max(np.abs(y))) if len(y) else 0.0
    if p < 1e-6:
        return y.astype(np.float32)
    g = min(10 ** (30 / 20), (10 ** (peak_db / 20)) / p)      # same cap as Groove Lab (+30 dB)
    return (y * g).astype(np.float32)


def features(y: np.ndarray) -> np.ndarray:
    """y: mono float, 44.1 kHz, already normalised. Returns [frames, 128] float32."""
    y = np.asarray(y, np.float32)
    pad = NFFT // 2
    yp = np.pad(y, pad, mode='reflect') if len(y) > pad else np.pad(y, pad)
    n = 1 + len(y) // HOP
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    idx = np.minimum(idx, len(yp) - 1)
    frames = yp[idx] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = spec.astype(np.float32) @ mel_bank().T
    return np.log(mel + 1e-7).astype(np.float32)


# ----------------------------------------------------------------------------- labels
def midi_hits(midi_bytes: bytes):
    """[(time_s, pitch, velocity)] for all note-ons in a MIDI file."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    out = []
    for inst in pm.instruments:
        for n in inst.notes:
            out.append((float(n.start), int(n.pitch), int(n.velocity)))
    return sorted(out)


def targets(hits, n_frames: int) -> np.ndarray:
    """Onset targets [frames, classes] (uint8, 0..255 = 0..1): 1 at the onset frame, 0.5 either side (target widening)."""
    t = np.zeros((n_frames, len(CLASSES)), np.float32)
    for time_s, pitch, vel in hits:
        c = PITCH_CLASS.get(pitch)
        if c is None:
            continue
        ci = CLASSES.index(c)
        f = int(round(time_s * FPS))
        for d, v in ((0, 1.0), (-1, 0.5), (1, 0.5)):
            if 0 <= f + d < n_frames:
                t[f + d, ci] = max(t[f + d, ci], v)
    return np.round(t * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- E-GMD access
class EGMD:
    """Reads single files out of the remote E-GMD zip with HTTP range requests (pip: remotezip)."""

    def __init__(self, csv_path: str, url: str = EGMD_ZIP):
        import csv
        self.rows = list(csv.DictReader(open(csv_path)))
        self.url = url
        self._z = None

    def _zip(self):
        if self._z is None:
            from remotezip import RemoteZip
            try:
                self._z = RemoteZip(self.url)
            except Exception as e:
                log('main E-GMD host failed (', e, '), trying Zenodo')
                self._z = RemoteZip(EGMD_ZIP_ZENODO)
            names = self._z.namelist()
            self._prefix = names[0].split('/')[0] + '/' if names and not names[0].startswith('drummer') else ''
        return self._z

    def read(self, rel: str) -> bytes:
        z = self._zip()
        return z.read(self._prefix + rel)

    def select(self, split: str, hours: float, seed: int = 0, kits=None):
        """Random clips from one split, spread over kits, up to `hours` of audio."""
        rows = [r for r in self.rows if r['split'] == split and (kits is None or r['kit_name'] in kits)]
        rnd = random.Random(seed)
        rnd.shuffle(rows)
        out, tot = [], 0.0
        for r in rows:
            d = float(r['duration'])
            if d < 2.0:
                continue
            out.append(r)
            tot += d
            if tot >= hours * 3600:
                break
        return out


def read_wav(b: bytes):
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(b), dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- rendering (FluidSynth)
def render_midi(midi_path: str, sf2: str, seconds: float) -> np.ndarray:
    """Render a MIDI file with the fluidsynth command-line player. Mono float32 at 44.1 kHz."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'o.wav')
        subprocess.run(['fluidsynth', '-ni', '-g', '0.6', '-r', str(SR), '-F', out, sf2, midi_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import soundfile as sf
        y, sr = sf.read(out, dtype='float32', always_2d=True)
    y = y.mean(axis=1)
    n = int(seconds * SR)
    return np.pad(y, (0, max(0, n - len(y))))[:n]


RENDER = render_midi          # tests swap this for a stand-in


def drums_through_gs(hits, kit: int, seconds: float, sf2: str) -> np.ndarray:
    """Re-render an E-GMD performance through a GeneralUser GS drum kit."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=kit, is_drum=True)
    for t, p, v in hits:
        p = TD17_TO_GM.get(p, p)
        inst.notes.append(pretty_midi.Note(velocity=max(1, min(127, v)), pitch=p, start=t, end=t + 0.1))
    pm.instruments.append(inst)
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        pm.write(f.name)
        path = f.name
    try:
        return RENDER(path, sf2, seconds)
    finally:
        os.unlink(path)


# ----------------------------------------------------------------------------- generated backing parts
# GM programs that make Demucs work hard: basses, keys, guitars, organ, strings, pads, brass, synths
BASSES = [32, 33, 34, 35, 36, 38, 39]
KEYS = [0, 1, 2, 4, 5, 16, 17, 18]
GUITARS = [24, 25, 26, 27, 28, 29, 30]
PADS = [48, 49, 50, 52, 61, 62, 81, 88, 89, 90, 95]
LEADS = [56, 57, 61, 65, 66, 71, 73, 80, 81, 82]


def backing_midi(seconds: float, bpm: float, rnd: random.Random):
    """A random but musical backing track (no drums): chords, bass line, comping, optional pad and lead."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    beat = 60.0 / bpm
    key = rnd.randrange(12)
    minor = rnd.random() < 0.45
    scale = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    degrees = rnd.choice([[0, 5, 3, 4], [0, 3, 4, 0], [0, 4, 5, 3], [5, 3, 0, 4], [0, 0, 3, 4], [1, 4, 0, 0]])
    swing = rnd.random() < 0.3

    def chord(deg):
        return [scale[(deg + k) % 7] + 12 * ((deg + k) // 7) for k in (0, 2, 4, 6 if rnd.random() < 0.3 else 4)]

    parts = {'bass': rnd.choice(BASSES), 'comp': rnd.choice(KEYS + GUITARS)}
    if rnd.random() < 0.6: parts['pad'] = rnd.choice(PADS)
    if rnd.random() < 0.4: parts['lead'] = rnd.choice(LEADS)
    insts = {k: pretty_midi.Instrument(program=p) for k, p in parts.items()}
    t, bar = 0.0, 0
    while t < seconds:
        notes = chord(degrees[bar % len(degrees)])
        root = 36 + key + notes[0]
        # bass: roots and fifths, 8ths or quarters with some rests
        step = beat / (2 if rnd.random() < 0.6 else 1)
        k = 0
        while k * step < 4 * beat:
            if rnd.random() < 0.85:
                p = root + (7 if rnd.random() < 0.25 else 0) + (12 if rnd.random() < 0.1 else 0)
                st = t + k * step + (0.06 * beat if swing and k % 2 else 0)
                insts['bass'].notes.append(pretty_midi.Note(rnd.randint(70, 115), p, st, st + step * 0.9))
            k += 1
        # comping: rhythmic chord stabs
        pattern = rnd.choice([[0, 1, 2, 3], [0, 1.5, 2, 3.5], [0.5, 1.5, 2.5, 3.5], [0, 2], [0, 0.75, 1.5, 2.5, 3.25]])
        for b in pattern:
            st = t + b * beat
            dur = beat * rnd.choice([0.25, 0.5, 0.9])
            for n in notes[:3 + (rnd.random() < 0.5)]:
                insts['comp'].notes.append(pretty_midi.Note(rnd.randint(55, 100), 60 + key + n, st, st + dur))
        if 'pad' in insts:
            for n in notes[:3]:
                insts['pad'].notes.append(pretty_midi.Note(rnd.randint(40, 80), 60 + key + n, t, t + 4 * beat))
        if 'lead' in insts:
            st = t
            while st < t + 4 * beat:
                d = beat * rnd.choice([0.25, 0.5, 0.5, 1, 1.5])
                if rnd.random() < 0.75:
                    p = 72 + key + scale[rnd.randrange(7)] + (12 if rnd.random() < 0.2 else 0)
                    insts['lead'].notes.append(pretty_midi.Note(rnd.randint(60, 110), p, st, st + d * 0.95))
                st += d
        t += 4 * beat
        bar += 1
    for i in insts.values():
        pm.instruments.append(i)
    return pm


def render_backing(seconds: float, bpm: float, seed: int, sf2: str) -> np.ndarray:
    pm = backing_midi(seconds, bpm, random.Random(seed))
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        pm.write(f.name)
        path = f.name
    try:
        return RENDER(path, sf2, seconds)
    finally:
        os.unlink(path)


# ----------------------------------------------------------------------------- augmentation (drums only)
def _biquad(y, b, a):
    from scipy.signal import lfilter
    return lfilter(b, a, y).astype(np.float32)


def _shelf(y, f0, gain_db, high: bool):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / 2 * np.sqrt(2)
    if high:
        b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    else:
        b = [A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    return _biquad(y, np.array(b) / a[0], np.array(a) / a[0])


def _peak(y, f0, gain_db, q):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    alpha = np.sin(w) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w), 1 - alpha / A]
    return _biquad(y, np.array(b) / a[0], np.array(a) / a[0])


def augment(y: np.ndarray, rnd: random.Random) -> np.ndarray:
    """Make clean module drums sound more like a recorded, mixed kit."""
    from scipy.signal import fftconvolve
    y = y.astype(np.float32)
    if rnd.random() < 0.8:                                   # tonal balance
        y = _shelf(y, rnd.uniform(60, 200), rnd.uniform(-6, 6), high=False)
        y = _shelf(y, rnd.uniform(3000, 9000), rnd.uniform(-8, 6), high=True)
        if rnd.random() < 0.5:
            y = _peak(y, rnd.uniform(200, 4000), rnd.uniform(-6, 6), rnd.uniform(0.5, 2))
    if rnd.random() < 0.6:                                   # room / plate
        rt = rnd.uniform(0.15, 1.2)
        n = int(rt * SR)
        ir = (np.random.default_rng(rnd.randrange(1 << 30)).standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)).astype(np.float32)
        ir = _shelf(ir, rnd.uniform(2000, 8000), -rnd.uniform(0, 12), high=True)
        wet = fftconvolve(y, ir)[:len(y)].astype(np.float32)
        wet *= np.max(np.abs(y)) / (np.max(np.abs(wet)) + 1e-9)
        y = y + rnd.uniform(0.05, 0.4) * wet
    if rnd.random() < 0.5:                                   # bus compression
        thr, ratio = 10 ** (rnd.uniform(-24, -8) / 20) * np.max(np.abs(y)), rnd.uniform(2, 8)
        B = 32                                               # envelope on 32-sample blocks (0.7 ms), then interpolated
        nb = len(y) // B + 1
        a = np.pad(np.abs(y), (0, nb * B - len(y))).reshape(nb, B).max(1)
        att, rel = math.exp(-B / (0.003 * SR)), math.exp(-B / (rnd.uniform(0.05, 0.3) * SR))
        env, gb = 0.0, np.empty(nb, np.float32)
        for i in range(nb):
            x = a[i]
            env = att * env + (1 - att) * x if x > env else rel * env + (1 - rel) * x
            gb[i] = 1.0 if env <= thr else (thr * (env / thr) ** (1 / ratio)) / env
        y = y * np.interp(np.arange(len(y)), np.arange(nb) * B + B / 2, gb).astype(np.float32)
    if rnd.random() < 0.3:                                   # tape / console saturation
        drive = rnd.uniform(1.5, 5)
        y = np.tanh(y / (np.max(np.abs(y)) + 1e-9) * drive) / np.tanh(drive)
    if rnd.random() < 0.15:                                  # lo-fi: band-limit
        y = _shelf(y, rnd.uniform(4000, 8000), -24, high=True)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- Demucs
_DEMUCS = None


def demucs_drums(y: np.ndarray, device: str = 'cuda') -> np.ndarray:
    """Drum stem from Demucs htdemucs (the model Groove Lab uses). Mono in, mono out."""
    global _DEMUCS
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    if _DEMUCS is None:
        _DEMUCS = get_model('htdemucs').to(device).eval()
    m = _DEMUCS
    x = torch.tensor(np.stack([y, y]), dtype=torch.float32)[None].to(device)
    ref = x.mean(1)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(m, (x - mean) / std, device=device, split=True, overlap=0.25, progress=False)[0]
    d = out[m.sources.index('drums')] * std + mean
    return d.mean(0).cpu().numpy().astype(np.float32)


DEMUCS = demucs_drums         # tests swap this for a stand-in


# ----------------------------------------------------------------------------- data preparation
@dataclass
class PrepConfig:
    work: str = 'work'                  # folder for everything (Google Drive in Colab)
    sf2: str = 'GeneralUser-GS.sf2'
    train_hours: float = 40.0
    val_hours: float = 2.0
    test_hours: float = 2.0
    p_gs_kit: float = 0.12              # re-render the performance through a GeneralUser GS kit
    p_clean: float = 0.15               # drums only, no backing, no Demucs (still augmented)
    p_demucs_solo: float = 0.15         # drums only, through Demucs (its artefacts without bleed)
    backing_db: tuple = (-9.0, 6.0)     # backing level relative to drums
    shard_seconds: float = 600.0        # ~10 minutes of features per shard file
    seed: int = 1


def prepare(split: str, cfg: PrepConfig, egmd: EGMD, device: str = 'cuda', limit_s: float | None = None):
    """Turn E-GMD clips into (features, targets) shards in WORK/data/<split>/. Resumes where it stopped."""
    out = Path(cfg.work) / 'data' / split
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / 'done.json'
    done = set(json.load(open(done_path))) if done_path.exists() else set()
    # each recording (performance x kit) is its own clip: keyed by its audio file. E-GMD's 'id' is shared by
    # all 43 kit versions of a performance; folders started before this fix keep using it (legacy).
    legacy = bool(done) and not any(k.endswith('.wav') for k in done)
    key = (lambda r: r['id']) if legacy else (lambda r: r['audio_filename'])
    hours = {'train': cfg.train_hours, 'validation': cfg.val_hours, 'test': cfg.test_hours}[split]
    have = sum(int(np.load(p)['lens'].sum()) for p in out.glob('shard_*.npz')) / FPS     # seconds already prepared
    need = hours * 3600 - have
    rows, tot = [], 0.0                                  # clips not tried yet, until the target hours are covered
    for r in egmd.select(split, 1e9, seed=cfg.seed):
        if tot >= need:
            break
        if key(r) in done:
            continue
        rows.append(r); tot += float(r['duration'])
    log(f'{split}: {len(rows)} clips to prepare ({have / 3600:.1f} h of {hours} h done, {len(done)} clips tried)')
    feats, targs, ids, buf_s = [], [], [], 0.0
    shard = len(list(out.glob('shard_*.npz')))
    t_start = time.time()

    def flush():
        nonlocal feats, targs, ids, buf_s, shard
        if not feats:
            return
        np.savez(out / f'shard_{shard:05d}.npz', x=np.concatenate(feats).astype(np.float16),
                 y=np.concatenate(targs), lens=np.array([len(f) for f in feats]), ids=np.array(ids))
        done.update(ids)
        json.dump(sorted(done), open(done_path, 'w'))
        shard += 1
        feats, targs, ids, buf_s = [], [], [], 0.0

    for k, r in enumerate(rows):
        if limit_s and time.time() - t_start > limit_s:
            log('time limit reached; run again to continue')
            break
        rnd = random.Random(f"{cfg.seed}-{key(r)}")
        try:
            hits = midi_hits(egmd.read(r['midi_filename']))
            dur = float(r['duration'])
            if rnd.random() < cfg.p_gs_kit:
                drums = drums_through_gs(hits, rnd.choice(GS_KITS), dur + 1, cfg.sf2)
            else:
                drums = read_wav(egmd.read(r['audio_filename']))
            drums = augment(normalise(drums, -6), rnd)
            mode = rnd.random()
            if mode < cfg.p_clean:
                mix, sep = drums, False
            elif mode < cfg.p_clean + cfg.p_demucs_solo:
                mix, sep = drums, True
            else:
                bpm = float(r['bpm']) if r.get('bpm') else rnd.uniform(70, 160)
                back = render_backing(len(drums) / SR + 0.1, bpm, rnd.randrange(1 << 30), cfg.sf2)
                back = np.pad(back, (0, max(0, len(drums) - len(back))))[:len(drums)]   # exactly the drums' length
                back = back / (np.sqrt(np.mean(back ** 2)) + 1e-9) * np.sqrt(np.mean(drums ** 2)) * 10 ** (rnd.uniform(*cfg.backing_db) / 20)
                mix, sep = drums + back, True
            stem = DEMUCS(normalise(mix, -1), device) if sep else mix
            x = features(normalise(stem, rnd.uniform(-3, -1)))
            y = targets(hits, len(x))
            feats.append(x); targs.append(y); ids.append(key(r)); buf_s += len(x) / FPS
        except Exception as e:
            log('skipped', key(r), type(e).__name__, str(e)[:120])
            done.add(key(r))
            continue
        if buf_s >= cfg.shard_seconds:
            flush()
        if k % 25 == 0:
            log(f'{split}: {k + 1}/{len(rows)} clips, {(time.time() - t_start) / 60:.1f} min')
    flush()
    log(f'{split}: ready ({len(done)} clips tried)')


# ----------------------------------------------------------------------------- dataset
class Shards:
    """All shards of one split in memory (float16 features), with random fixed-length crops for training."""

    def __init__(self, folder: str):
        xs, ys, starts = [], [], []
        off = 0
        for p in sorted(Path(folder).glob('shard_*.npz')):
            d = np.load(p)
            xs.append(d['x']); ys.append(d['y'])
            for L in d['lens']:
                starts.append((off, int(L)))
                off += int(L)
        if not xs:
            raise FileNotFoundError(f'no shards in {folder}')
        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)
        self.clips = starts
        self.hours = len(self.x) / FPS / 3600

    def crop_batch(self, rnd: np.random.Generator, n: int, T: int):
        X = np.empty((n, T, NMELS), np.float32)
        Y = np.empty((n, T, len(CLASSES)), np.float32)
        for i in range(n):
            s, L = self.clips[rnd.integers(len(self.clips))]
            o = s + (rnd.integers(0, L - T) if L > T else 0)
            x, y = self.x[o:o + T], self.y[o:o + T]
            if len(x) < T:
                x = np.pad(x, ((0, T - len(x)), (0, 0)), constant_values=float(np.log(1e-7)))
                y = np.pad(y, ((0, T - len(y)), (0, 0)))
            X[i], Y[i] = x, y / 255.0
        return X, Y

    def class_rates(self):
        return (self.y >= 255).mean(axis=0)


# ----------------------------------------------------------------------------- model
def make_model(hidden: int = 96):
    import torch.nn as nn
    import torch

    class Block(nn.Module):
        def __init__(s, ci, co, pool):
            super().__init__()
            s.c1 = nn.Conv2d(ci, co, 3, padding=1); s.b1 = nn.BatchNorm2d(co)
            s.c2 = nn.Conv2d(co, co, 3, padding=1); s.b2 = nn.BatchNorm2d(co)
            s.pool, s.drop = pool, nn.Dropout(0.15)

        def forward(s, x):
            x = torch.relu(s.b1(s.c1(x)))
            x = torch.relu(s.b2(s.c2(x)))
            x = nn.functional.max_pool2d(x, (1, s.pool))
            return s.drop(x)

    class CRNN(nn.Module):
        """[B, T, 128] log-mel -> [B, T, 9] onset probabilities."""

        def __init__(s):
            super().__init__()
            s.inorm = nn.BatchNorm1d(NMELS)
            s.b1, s.b2 = Block(1, 32, 3), Block(32, 64, 3)
            s.proj = nn.Linear(64 * (NMELS // 9), 256)
            s.gru = nn.GRU(256, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
            s.out = nn.Linear(2 * hidden, len(CLASSES))

        def forward(s, x, logits: bool = False):
            x = s.inorm(x.transpose(1, 2)).transpose(1, 2)
            x = s.b2(s.b1(x.unsqueeze(1)))                     # [B, C, T, F]
            B, C, T, F = x.shape
            x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
            x = torch.relu(s.proj(x))
            x, _ = s.gru(x)
            z = s.out(x)
            return z if logits else torch.sigmoid(z)

    return CRNN()


# ----------------------------------------------------------------------------- peak picking + scoring
def pick(prob: np.ndarray, thr: float, min_gap: int = 3):
    """Onset frames: local maxima above thr, at least min_gap frames apart."""
    p = prob
    peaks = []
    for f in range(len(p)):
        if p[f] < thr:
            continue
        if f > 0 and p[f - 1] > p[f]:
            continue
        if f + 1 < len(p) and p[f + 1] >= p[f]:
            continue
        if peaks and f - peaks[-1] < min_gap:
            if p[f] > p[peaks[-1]]:
                peaks[-1] = f
            continue
        peaks.append(f)
    return np.array(peaks, int)


def match_f1(ref: np.ndarray, est: np.ndarray, tol: int = 5):
    """Greedy one-to-one onset matching within +-tol frames (5 = 50 ms). Returns tp, fp, fn."""
    used = np.zeros(len(est), bool)
    tp = 0
    for r in ref:
        best, bi = tol + 1, -1
        for i, e in enumerate(est):
            if not used[i] and abs(e - r) <= tol and abs(e - r) < best:
                best, bi = abs(e - r), i
        if bi >= 0:
            used[bi] = True
            tp += 1
    return tp, len(est) - tp, len(ref) - tp


def evaluate(model, data: Shards, device: str, thresholds=None, chunk: int = 3000):
    """Per-class precision / recall / F1 over every clip of a split."""
    import torch
    was_training = model.training
    model.eval()
    C = len(CLASSES)
    thresholds = thresholds or [0.5] * C
    acc = np.zeros((C, 3), int)
    probs_all = []
    with torch.no_grad():
        for s, L in data.clips:
            x = torch.tensor(data.x[s:s + L].astype(np.float32))[None].to(device)
            p = model(x)[0].cpu().numpy()
            probs_all.append((p, data.y[s:s + L]))
            for c in range(C):
                ref = np.where(data.y[s:s + L, c] >= 255)[0]
                est = pick(p[:, c], thresholds[c])
                acc[c] += match_f1(ref, est)
    res = {}
    for c, name in enumerate(CLASSES):
        tp, fp, fn = acc[c]
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F = 2 * P * R / max(1e-9, P + R)
        res[name] = {'precision': round(P, 3), 'recall': round(R, 3), 'f1': round(F, 3), 'n': int(tp + fn)}
    tot = acc.sum(0); P = tot[0] / max(1, tot[0] + tot[1]); R = tot[0] / max(1, tot[0] + tot[2])
    res['all'] = {'precision': round(P, 3), 'recall': round(R, 3), 'f1': round(2 * P * R / max(1e-9, P + R), 3)}
    model.train(was_training)
    return res, probs_all


def tune_thresholds(probs_all, grid=np.arange(0.15, 0.8, 0.05)):
    """Per-class threshold with the best F1 on validation."""
    C = len(CLASSES)
    best = []
    for c in range(C):
        scores = []
        for thr in grid:
            tp = fp = fn = 0
            for p, y in probs_all:
                a, b, d = match_f1(np.where(y[:, c] >= 255)[0], pick(p[:, c], thr))
                tp += a; fp += b; fn += d
            P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
            scores.append(2 * P * R / max(1e-9, P + R))
        best.append(float(grid[int(np.argmax(scores))]))
    return best


# ----------------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    work: str = 'work'
    steps: int = 40000
    batch: int = 32
    crop: int = 400                    # 4 s
    lr: float = 1e-3
    hidden: int = 96
    eval_every: int = 2000
    seed: int = 0
    max_pos_weight: float = 20.0


def train(cfg: TrainConfig, device: str = 'cuda', time_limit_s: float | None = None):
    import torch
    torch.manual_seed(cfg.seed)
    work = Path(cfg.work)
    tr, va = Shards(work / 'data' / 'train'), Shards(work / 'data' / 'validation')
    log(f'train {tr.hours:.1f} h, validation {va.hours:.1f} h')
    rates = tr.class_rates()
    pos_w = torch.tensor(np.clip((1 - rates) / np.maximum(rates, 1e-6), 1, cfg.max_pos_weight) ** 0.5, dtype=torch.float32, device=device)
    model = make_model(cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.05)
    ck = work / 'checkpoint.pt'
    step, best = 0, -1.0
    if ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)   # our own checkpoint
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        step, best = int(st['step']), float(st['best'])
        log(f'resumed at step {step} (best validation F1 {best:.3f})')
    rnd = np.random.default_rng(cfg.seed + step)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    t0, run = time.time(), 0.0
    model.train()
    while step < cfg.steps:
        X, Y = tr.crop_batch(rnd, cfg.batch, cfg.crop)
        if rnd.random() < 0.5:                                   # spectral jitter: small random gain / tilt
            X = (X + rnd.normal(0, 0.5) + np.linspace(-1, 1, NMELS)[None, None, :] * rnd.normal(0, 0.3)).astype(np.float32)
        X, Y = torch.tensor(X, device=device), torch.tensor(Y, device=device)
        with torch.autocast('cuda', enabled=use_amp):
            loss = lossf(model(X, logits=True), Y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % 200 == 0:
            log(f'step {step}/{cfg.steps}  loss {run:.4f}  {(time.time() - t0) / 60:.1f} min')
        if step % cfg.eval_every == 0 or step == cfg.steps:
            res, _ = evaluate(model, va, device)
            f1 = res['all']['f1']
            log('validation F1', f1, ' '.join(f"{k}:{v['f1']}" for k, v in res.items() if k != 'all'))
            if f1 > best:
                best = float(f1)
                torch.save(model.state_dict(), work / 'best.pt')
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'best': best}, ck)
        if time_limit_s and time.time() - t0 > time_limit_s:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'best': best}, ck)
            log('time limit reached; run this step again to continue')
            return
    log('training finished; best validation F1', best)


# ----------------------------------------------------------------------------- export
def export(work: str, device: str = 'cpu', hidden: int = 96, name: str = 'bmy_drums'):
    """best.pt -> <name>.onnx + <name>.json (classes, tuned thresholds, front end), test-set scores."""
    import torch
    work = Path(work)
    model = make_model(hidden)
    model.load_state_dict(torch.load(work / 'best.pt', map_location='cpu', weights_only=False))
    model.eval()
    va = Shards(work / 'data' / 'validation')
    _, probs = evaluate(model, va, 'cpu')
    thr = tune_thresholds(probs)
    meta = {'classes': CLASSES, 'thresholds': thr, 'sr': SR, 'hop': HOP, 'fps': FPS, 'nfft': NFFT, 'n_mels': NMELS,
            'fmin': FMIN, 'fmax': FMAX, 'mel': 'htk, slaney norm', 'log': 'ln(power + 1e-7)', 'input_peak_dbfs': -1,
            'window': 'periodic hann, centred, reflect padding', 'min_gap_frames': 3}
    tst = work / 'data' / 'test'
    if tst.exists() and any(tst.glob('shard_*.npz')):
        meta['test'], _ = evaluate(model, Shards(tst), 'cpu', thr)
        log('test scores', json.dumps(meta['test']))
    model.eval()
    torch.onnx.export(model, (torch.zeros(1, 300, NMELS),), str(work / f'{name}.onnx'), input_names=['mel'], output_names=['onset'],
                      dynamic_axes={'mel': {1: 'T'}, 'onset': {1: 'T'}}, opset_version=17, dynamo=False)
    json.dump(meta, open(work / f'{name}.json', 'w'), indent=1)
    # parity check
    import onnxruntime as ort
    s = ort.InferenceSession(str(work / f'{name}.onnx'))
    x = va.x[:500].astype(np.float32)[None]
    a = s.run(None, {'mel': x})[0]
    with torch.no_grad():
        b = model(torch.tensor(x)).numpy()
    log(f'exported {name}.onnx ({os.path.getsize(work / f"{name}.onnx") // 1024} KB), ONNX vs PyTorch max diff {float(np.abs(a - b).max()):.2e}')
    return meta
