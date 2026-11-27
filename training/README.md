# BMY drum transcriber: training

BMY Drummer is a commercial app, so its Build screen can't use ADTOF (CC BY-NC-SA). This folder trains BMY's own transcriber from licence-clean material only.

**Run it:** open [BMY_Transcriber_Colab.ipynb in Colab](https://colab.research.google.com/github/taddmlh/bmy-groove-lab/blob/main/training/BMY_Transcriber_Colab.ipynb), pick a GPU runtime, **Run all**. Results go to Google Drive › `BMY_Transcriber`. Re-run to continue after a stop.

## Recipe
| Part | Source | Licence |
|---|---|---|
| Drum audio + aligned MIDI | E-GMD v1.0.0: 444 h, 43 Roland TD-17 kit presets (mostly acoustic-style), read file-by-file from the remote zip | CC BY 4.0 (credit Callender, Hawthorne, Engel 2020) |
| Free acoustic kits | the same performances re-played (45 %) through multi-sampled, multi-mic acoustic kits by `sfz_kits.py` (a small numpy SFZ renderer: velocity layers, round robin, mic mix, hi-hat choke), random per-piece levels; pieces a kit lacks come from a second kit. **Naked Drums is held out of training** (validation / test only). | Karoryfer Big Rusty, Swirly, Unruly, Gogodze Phu Vol II: CC0. SM Drums: public domain. DrumGizmo Muldjord + DRS kits (credit Lars Muldjord / DrumGizmo), Wilkinson Audio Naked Drums: **CC BY 4.0 — credit in BMY** |
| Extra kit sounds | the same performances re-played through BMY's GeneralUser GS kits (10 %) | GeneralUser GS License v2.0 |
| Backing instruments | generated here: random chord progressions → bass, comping (keys / guitars), pads, leads, rendered with FluidSynth + GeneralUser GS | ours |
| Realism | per-clip EQ, room reverb, bus compression, saturation, lo-fi band-limit; drums mixed with backing at −9…+6 dB, then **Demucs htdemucs** (as in Groove Lab); 15 % clean drums, 15 % drums-only through Demucs | Demucs: MIT |
| Model | CRNN: 2 conv blocks (32, 64 ch) → 256 → 2-layer BiGRU (96) → 9 sigmoid outputs, ~1 M parameters | ours |

No commercial songs are used for training. Matt's Groove Lab stems are used **only for testing** against ADTOF.

**Front end** (Groove Lab will compute the same in JavaScript): input scaled to a −1 dBFS peak; periodic Hann, 2048-point FFT, hop 441 (100 frames/s), centred frames with reflect padding; power → 128 HTK mel bands 30 Hz–16 kHz (Slaney area norm) → ln(power + 1e-7).

**Classes (14):** kick (36), snare (38, 40, 37, 39), tomHigh (48, 50), tomMid (45, 47), tomFloor (43, 58), hhClosed (42, 22, 44), hhOpen (46, 26), crash (49, 55, 57, 52), ride (51, 59, 53); percussion for BMY's one-shot lanes: shaker (69 cabasa, 70 maracas, 82 shaker, 73/74 guiro), tambourine (54), bell (56 cowbell, 67/68 agogo, 80/81 triangle), handDrum (60–66 bongos, congas, timbales), wood (75 claves, 76/77 woodblocks). Percussion comes from generated parts (35 % of clips, GeneralUser GS sounds, locked to the tempo).

**Targets:** onset frame = 1, neighbours = 0.5. Loss: BCE with per-class positive weights (√ of imbalance, capped). Training: random 4 s crops, AdamW + one-cycle, mixed precision, small random gain/tilt on the features.

**Output:** `bmy_drums.onnx` (input `mel` [1, T, 128] → `onset` [1, T, 9]) and `bmy_drums.json` (classes, per-class thresholds tuned on validation, front-end settings, test scores). Peak picking: local maxima above the class threshold, ≥ 3 frames apart.

## Presets
| | data | steps | time on a T4 |
|---|---|---|---|
| trial | 10 h train, 1 h validation, 1 h test (TD-17 + GeneralUser only) | 15 000 | ~2 h total |
| kits | as trial, with the free acoustic kits | 15 000 | ~2 h total |
| full | 40 h train, 2 h validation, 2 h test | 40 000 | ~6 h total, over 2 sessions |

## Testing
`bmy_transcriber.py` is tested on CPU with a fake E-GMD (grooves rendered from BMY's kit samples), stand-ins for FluidSynth and Demucs: data preparation and resume, training and resume, evaluation, threshold tuning, ONNX export with a PyTorch-vs-ONNX parity check. Remote E-GMD access, FluidSynth and Demucs run for real only in Colab (the notebook's step 3 checks them on one clip first).

`sfz_kits.py` is checked on Gogodze and Muldjord: every rendered hit lands on its MIDI time, closed / pedal hats choke a ringing open hat, and only hits a kit actually played are labelled. First result: the trial model reads Gogodze's (real, miked) hats as snare (31 snares for 12), the same fault seen on Matt's stems, which is why the kits were added.
