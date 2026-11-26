# Preview kit

BMY Drummer's built-in kit: **GeneralUser GS** by S. Christian Collins (https://www.schristiancollins.com/generaluser.php),
drum programs as BMY offers them (0 Standard 1 in `kit/`; 1 Standard 2, 2 Standard 3, 8 Room, 16 Power, 24 Electronic, 25 808/909, 26 Dance, 32 Jazz, 40 Brush, 48 Orchestral, 56 SFX in `kit/<program>/`), rendered per BMY velocity label (gh 28, sf 46, n1 66, n2 82, a1 102, a2 116)
with FluidSynth 2.4.6 (`tools/fluid_render.cjs`, js-synthesizer) from BMY's bundled `GeneralUser-GS.sf2`, packed with `tools/pack_kit.cjs`. Lanes a kit has no sound for (SFX: kick, side stick, snare) fall back to Standard 1. `tools/render_kit.py` is an older home-made renderer kept for reference.

GeneralUser GS License v2.0: free to use, including in software projects, with credit; do not sell the bank itself
or claim it as your own work.
