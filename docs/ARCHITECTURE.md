# VideoToAudio Architecture

This project turns camera or image input into music through three execution
paths:

- `ambient`: continuous camera-driven local synthesis, no API calls.
- `llm`: image/camera snapshot to OpenAI description, then ElevenLabs music.
- `arrangement`: camera snapshot to OpenAI-selected MIDI, then local MIDI
  arrangement/playback through FluidSynth and SoundFonts.

The deployment goal for Raspberry Pi should be one stable command that selects
one mode, with shared camera, feature extraction, config, and output folders.

## Current Shape

```text
                         +--------------------------+
                         | OpenCV image features    |
                         | brightness, hue, motion  |
                         +------------+-------------+
                                      |
+------------------+       +----------+----------+       +-------------------+
| main.py          |       | synth.py            |       | mac.py            |
| image -> LLM     |       | Pi camera +         |       | Mac webcam MIDI   |
| -> ElevenLabs    |       | FluidSynth browser  |       | test + arranger   |
+--------+---------+       +----------+----------+       +---------+---------+
         |                            |                            |
         v                            v                            v
+------------------+       +---------------------+       +-------------------+
| module/          |       | module/             |       | module/           |
| image_describer  |       | image_describer     |       | midi_selector     |
| eleven_labs      |       | midi_selector       |       | midi_arranger     |
+------------------+       +---------------------+       | soundfont_picker  |
                                                          +-------------------+

+------------------+
| ambient.py       |
| webcam -> vibe   |
| -> local synth   |
+--------+---------+
         |
         v
+------------------------------+
| module/ambient_model         |
| module/sounddevice_engine    |
| module/pyo_ambient_engine    |
+------------------------------+
```

The reusable package is already doing most of the right work. The problem is
that the top-level entrypoints still mix orchestration, camera handling, browser
serving, audio playback, and mode-specific logic.

## Target Shape For Pi

Create one launcher that chooses the mode:

```bash
python3 app.py ambient
python3 app.py llm --camera-index 0
python3 app.py arrangement
```

For browser-controlled camera use:

```bash
python3 app.py ambient --camera-index 0 --host 0.0.0.0 --port 8091
python3 app.py arrangement --camera-index 0 --host 0.0.0.0 --port 8090
python3 app.py llm --camera-index 0 --output-dir generated_music
```

The code should be split by responsibility:

```text
app.py
  Parses mode and shared flags.
  Dispatches to one mode runner.

module/config.py
  Loads .env and CLI values.
  Resolves output dirs, MIDI dirs, SoundFont paths, API keys.

module/camera.py
  Owns camera capture.
  Hides PiCamera2 vs OpenCV webcam differences.
  Returns BGR OpenCV frames.

module/features.py
  Owns visual feature extraction.
  Reuses current image_describer.extract_features logic.

module/modes/ambient.py
  Starts ambient browser app and local synth engine.

module/modes/llm.py
  Runs image/camera snapshot -> OpenAI -> ElevenLabs.

module/modes/arrangement.py
  Runs camera snapshot -> OpenAI MIDI choice -> arrangement -> playback.

module/web.py
  Optional shared HTTP helpers for status JSON and MJPEG streaming.
```

## Mode Responsibilities

### ambient

Best for live installation use when you want no network/API dependency.

```text
camera frame
  -> extract_features
  -> vibe_from_features
  -> smooth_vibe
  -> ambient_state_for
  -> SoundDeviceAmbientEngine or PyoAmbientEngine
```

Use this for deployment first because it has the fewest external dependencies:

- camera
- OpenCV
- NumPy
- sounddevice/PortAudio

No OpenAI, ElevenLabs, MIDI library, or SoundFont is required.

### llm

Best for generating a finished audio file from an image or snapshot.

```text
camera snapshot or image
  -> extract_features
  -> extract_image_parts
  -> OpenAI image description
  -> build_elevenlabs_prompt
  -> ElevenLabs Music API
  -> generated_music/
```

Use this when you have:

- stable network access
- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`
- enough patience for generation latency

### arrangement

Best for local playback with a controllable music library.

```text
camera snapshot
  -> extract_features
  -> OpenAI image description
  -> choose chord MIDI
  -> choose drum MIDI
  -> create_full_arrangement
  -> choose SoundFont
  -> FluidSynth playback
```

Use this when you want local audio output but still want the LLM to choose the
musical direction. It needs:

- `OPENAI_API_KEY`
- chord MIDI library
- drum MIDI library
- FluidSynth executable
- `.sf2` SoundFont

## Recommended Refactor Order

1. Add `app.py` as the only command you run on the Pi.
2. Keep existing `ambient.py`, `main.py`, `mac.py`, and `synth.py` working while
   `app.py` delegates to them.
3. Extract shared config/env loading into `module/config.py`.
4. Extract camera handling into `module/camera.py`.
5. Move mode-specific browser apps into `module/modes/`.
6. Retire or simplify the older top-level scripts once the launcher is stable.

This order avoids a big rewrite. The first version of `app.py` can be a thin
router around the current scripts, then you can pull shared pieces out safely.

## First Pi Deployment Target

Start with:

```bash
python3 app.py ambient --host 0.0.0.0 --port 8091 --camera-index 0
```

Then add arrangement:

```bash
python3 app.py arrangement \
  --host 0.0.0.0 \
  --port 8090 \
  --camera-index 0 \
  --midi-dir midi/free_chord_progressions \
  --drum-dir midi/dmp_drum_patterns \
  --soundfont soundfonts/GeneralUser_GS.sf2
```

Add LLM/ElevenLabs last:

```bash
python3 app.py llm --camera-index 0 --output-dir generated_music
```

For offline testing without camera hardware:

```bash
python3 app.py llm --image path/to/image.jpg --output-dir generated_music
```

For a hackathon/demo, the most reliable hierarchy is:

1. `ambient`: always available when the camera/audio stack works.
2. `arrangement`: good local playback with one OpenAI call per selection.
3. `llm`: highest latency and most API-dependent, but best finished output.

## Minimal Launcher Design

The first `app.py` should not duplicate logic. It should parse the selected mode
and call existing `main()` functions with adjusted `sys.argv`.

```text
app.py ambient      -> ambient.main()
app.py llm          -> module.modes.llm.main()
app.py arrangement  -> mac.main() initially, then a Pi-specific arrangement mode
```

Later, replace the delegation with direct calls into `module/modes/*` once the
behavior is stable on the Pi.

## Configuration

Use `.env` for secrets and machine-specific paths:

```text
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
MIDI_DIR=midi/free_chord_progressions
DRUM_DIR=midi/dmp_drum_patterns
SOUNDFONT_PATH=soundfonts/GeneralUser_GS.sf2
GENERATED_MUSIC_DIR=generated_music
```

Avoid hard-coding Pi-specific paths inside mode logic. The launcher/config layer
should resolve defaults, and the mode should receive explicit values.

## What To Improve Next

- `synth.py` currently duplicates feature extraction that also exists in
  `module/image_describer.py`; consolidate on one feature extractor.
- `mac.py` is useful, but its name is now too platform-specific for arrangement
  generation. The arrangement flow should move into `module/modes/arrangement.py`.
- `ambient.py` is the cleanest mode boundary today and is a good model for the
  other modes.
- Add a smoke-test command for each mode that does not require camera hardware.
- Add explicit dependency files per deployment profile:
  - `requirements-pi-ambient.txt`
  - `requirements-pi-arrangement.txt`
  - `requirements-pi-llm.txt`
