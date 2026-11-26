# BMY Groove Lab

Personal tool: pull the drums out of a song on iPad and turn up to 6 bars into a BMY Drummer `.bmybank`.
Runs entirely in Safari (iPadOS 26+, WebGPU). Nothing is uploaded.

**Use it:** https://taddmlh.github.io/bmy-groove-lab/  ·  diagnostics: `speedtest.html`

## Pipeline
1. **Tempo + bar lines** from the full mix (`js/beats.js`), correctable (BPM, ÷2/×2, nudge, "Bar 1 here").
2. **Drum separation** of the marked bars only: htdemucs split into 21 half-precision parts on WebGPU (`js/demucs.js`).
3. **Transcription**: ADTOF Frame_RNN (5 classes) exported to ONNX (`models/adtof.onnx`, parity 7e-7 vs PyTorch); JS front end matches librosa output to 4e-7 (`js/dsp.js`, `js/adtof.js`).
4. **BMY mapping** (`js/transcribe.js`): toms split by pitch, open/closed hat and crash/ride by decay, velocity from drum-stem loudness → `gh sf n1 n2 a1 a2`, bank-wide swing (1/8, 1/16) and per-beat triplets (`t4`, `t8` with `tripletPattern`, `note2/3Velocity`).
5. **Export** (`js/bank.js`): schema 1, as `BMY_PatternBank.swift` validates it.

Test hook: `?nodemucs` treats the song as an already-separated drum track.

## Credits
- Demucs v4 (Meta, MIT); split ONNX export [monteslu/htdemucs-web-onnx](https://huggingface.co/monteslu/htdemucs-web-onnx) (MIT); pre/post-processing [demucs-web](https://github.com/timcsy/demucs-web) (MIT).
- ADTOF (Zehren, Alunno, Bientinesi), [MZehren/ADTOF](https://github.com/MZehren/ADTOF), licensed CC BY-NC-SA 4.0; PyTorch port [xavriley/ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch). `models/adtof.onnx` is a format conversion of those weights and carries the same non-commercial licence: fine for this personal tool, not for shipping inside a paid app.
- Preview kit: GeneralUser GS by S. Christian Collins (GeneralUser GS License v2.0), rendered by `tools/render_kit.py`; see `kit/README.md`.
- ONNX Runtime Web (MIT).
