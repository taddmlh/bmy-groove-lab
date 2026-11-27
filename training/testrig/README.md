# Transcriber test rig

Known grooves → free kits → Groove Lab's own transcriber code → a text report with the right answers.

- `scenarios.py`: 11 grooves aimed at real failure cases (ghost notes, kick-heavy, triplet hats over a straight backbeat, open/close hats, jazz ride + bell, crash accents, tom fills, flams/drags/ruffs, linear funk, each drum alone).
- `run.py`: renders every groove through each kit (`sfz_kits.py`), dry and mixed (EQ, room, compression). Writes `.wav`, `.mid` and `.txt` (every hit) per file, runs the transcribers and writes `REPORT.txt`.
- `transcribe.mjs`: runs `js/adtof.js` / `js/bmyt.js` in Node (onnxruntime-web), with the same −1 dBFS normalisation as the app.

```
npm install onnxruntime-web@1.31.1          # once, in a folder of your choice (--node-dir)
python3 run.py --kits naked,drs,muldjord,gogodze --kit-dir /path/to/kits --out rig \
               --models kits=../../models/bmy_drums.onnx,trial=/path/trial.onnx --node-dir <folder>
```

Report: F1 per drum per kit and per groove; false hits named by what was really there ("snare reported where there was kick"); snare found by velocity band (ghosts < 50). BMY is also scored collapsed to ADTOF's 5 classes, so the two compare directly.

Not covered yet: Demucs stems (separation artefacts, bleed from bass and guitars). That is where real songs differ, so it is the next condition to add (needs Demucs: Colab or the Mac).
