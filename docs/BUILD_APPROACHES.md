# Software Build Approaches

This document explains the different ways we approached building VideoToAudio.
It is written so sections can be lifted directly into presentation slides.

## Project Goal

VideoToAudio explores how a camera image can become music. The core idea is:

```text
camera / image input
  -> visual analysis
  -> musical interpretation
  -> audio output
```

We tried several approaches because each one gives a different balance of
speed, reliability, creativity, and deployment complexity.

## Approach 1: Direct Camera-To-Synth Mapping

The first approach was to map visual features directly to music parameters.

```text
camera frame
  -> OpenCV feature extraction
  -> musical parameter mapping
  -> FluidSynth playback
```

The software extracts features such as:

- brightness
- contrast
- saturation
- dominant hue
- edge density
- contour count
- motion between frames

Those features are mapped to musical controls:

- brightness changes note velocity and energy
- hue chooses the musical key
- saturation chooses the instrument
- edge density changes rhythm speed
- contour count changes the number of voices
- motion enables more active rhythms and drums

This approach is fast and works locally. It does not need an internet
connection or external AI APIs during playback.

### Strengths

- Very low latency.
- Works continuously with live camera input.
- Runs locally once audio and camera dependencies are installed.
- Good for real-time interaction and installation-style demos.

### Limitations

- The music can feel mechanical because the mapping is rule-based.
- It does not understand image meaning, only low-level visual features.
- More complex musical structure is hard to create with simple mappings.

## Approach 2: LLM Image Description To Generated Music

The second approach used AI models to interpret the image semantically.

```text
camera snapshot / image file
  -> OpenCV crops and features
  -> OpenAI image description
  -> ElevenLabs music prompt
  -> generated audio file
```

Instead of only reading pixels, this approach asks a vision model to describe
the image. The description is then combined with OpenCV features to create a
music prompt for ElevenLabs Music.

Example interpretation:

```text
visual scene: dark indoor room, warm light, person in foreground
musical direction: intimate, minimal, soft piano and muted strings
```

This creates more expressive and finished-sounding results because the system
can respond to the meaning of the scene, not just brightness or color.

### Strengths

- Produces polished generated music.
- Understands subjects, setting, mood, and context.
- Good for creating a finished audio output from a single image.
- Easier to explain to users because the system can show the description and
  prompt it generated.

### Limitations

- Requires network access.
- Requires OpenAI and ElevenLabs API keys.
- Higher latency because audio generation takes time.
- Less suitable for continuous live interaction.

## Approach 3: LLM-Guided MIDI Selection

The third approach kept playback local but used an LLM to choose from existing
MIDI files.

```text
camera snapshot
  -> image description
  -> local MIDI manifest
  -> LLM chooses matching MIDI
  -> FluidSynth playback
```

Here, the system has a library of MIDI files with metadata such as style, mood,
energy, and tags. The LLM compares the image description against the available
MIDI options and chooses the best match.

This is a middle ground between fully local synthesis and fully generated music.
The intelligence comes from the LLM, but the actual playback remains local.

### Strengths

- More musically coherent than raw feature-to-note mapping.
- Local playback is fast after the MIDI file is selected.
- The music library can be curated.
- Lower generation cost than creating full audio with ElevenLabs.

### Limitations

- Still requires an OpenAI API call for selection.
- Output quality depends heavily on the MIDI library.
- The system can only choose from what already exists.

## Approach 4: Vibe-Driven Arrangement Generation

The fourth approach generates a new MIDI arrangement from selected musical
building blocks.

```text
camera snapshot
  -> image description and visual features
  -> choose chord progression MIDI
  -> choose drum pattern MIDI
  -> analyze harmony
  -> generate arrangement sections
  -> FluidSynth playback
```

This approach combines:

- a chord progression library
- a drum pattern library
- OpenCV-derived vibe values
- harmonic analysis with `music21`
- MIDI rendering with `mido`
- SoundFont playback with FluidSynth

The arranger creates sections such as intro, verse, chorus, bridge, and outro.
It can add chord, bass, pad, lead, and drum layers depending on the image vibe.

For example:

- bright and colorful scenes get more energetic arrangements
- dark scenes get softer palettes and slower-feeling textures
- complex scenes get more layers and movement

### Strengths

- Produces more structured music than simple MIDI playback.
- Still uses local audio playback.
- Gives control over arrangement length, tempo, instruments, and SoundFonts.
- Good balance between AI decision-making and deterministic local generation.

### Limitations

- More complex than selecting one MIDI file.
- Requires curated MIDI libraries.
- Requires FluidSynth, SoundFonts, `mido`, and `music21`.
- Needs careful testing on Raspberry Pi performance.

## Approach 5: API-Free Ambient Synthesis

The fifth approach was designed for reliable Raspberry Pi deployment.

```text
live camera frame
  -> OpenCV features
  -> smoothed vibe profile
  -> modal key and chord state
  -> local real-time synth engine
```

This mode does not use OpenAI, ElevenLabs, MIDI files, or SoundFonts. It creates
continuous ambient audio locally using either:

- `sounddevice` + NumPy
- optional `pyo`

The system smooths the camera-derived values over time so the music changes
gradually rather than jumping every frame.

The generated sound uses:

- drones
- pads
- sparse arpeggios
- texture/noise layers
- modal chord progressions

### Strengths

- Most reliable for live deployment.
- No API keys or internet connection needed.
- Continuous real-time response.
- Lower operational complexity than LLM or MIDI workflows.

### Limitations

- More ambient and textural than song-like.
- Does not understand the semantic meaning of the image.
- Depends on stable local audio output through PortAudio.

## Why We Moved To A Unified Launcher

At first, each experiment had its own entrypoint:

```text
synth.py     -> Pi camera FluidSynth experiment
main.py      -> image to ElevenLabs music
mac.py       -> Mac webcam MIDI and arrangement testing
ambient.py   -> API-free ambient synth
```

That worked during experimentation, but it became harder to deploy because the
user had to know which script to run and which dependencies each script needed.

The improved architecture introduces one launcher:

```bash
python3 app.py ambient
python3 app.py llm
python3 app.py arrangement
```

This makes deployment simpler because the user chooses a mode instead of
choosing a script.

## Current Mode Selector

The software now has three main modes:

### `ambient`

Use when the priority is reliability and live interaction.

```bash
python3 app.py ambient --host 0.0.0.0 --port 8091 --camera-index 0
```

### `llm`

Use when the priority is a finished AI-generated audio file from the current
camera image.

```bash
python3 app.py llm --camera-index 0 --output-dir generated_music
```

For testing without a camera:

```bash
python3 app.py llm --image path/to/image.jpg --output-dir generated_music
```

### `arrangement`

Use when the priority is local MIDI playback with AI-guided musical selection
and structured arrangement generation.

```bash
python3 app.py arrangement \
  --host 0.0.0.0 \
  --port 8090 \
  --camera-index 0 \
  --soundfont soundfonts/GeneralUser_GS.sf2
```

## Comparison Summary

| Approach | Uses AI? | Internet Needed? | Output Type | Best For |
| --- | --- | --- | --- | --- |
| Direct camera-to-synth | No | No | live synth | low-latency interaction |
| LLM + ElevenLabs | Yes | Yes | generated audio file | polished final music |
| LLM MIDI selection | Yes | Yes for selection | local MIDI playback | curated music matching |
| Arrangement generation | Partly | Yes for selection | generated MIDI arrangement | structured local music |
| Ambient synthesis | No | No | live ambient audio | reliable Pi deployment |

## Final Architecture Direction

The strongest architecture is a hybrid system:

```text
shared camera capture
  -> shared feature extraction
  -> selected mode
      -> ambient
      -> llm
      -> arrangement
```

This keeps the input pipeline consistent while allowing different musical
outputs depending on the demo goal.

For Raspberry Pi deployment, the recommended order is:

1. Get `ambient` working first because it is the most reliable.
2. Add `arrangement` once local FluidSynth playback is stable.
3. Add `llm` once API keys and network access are confirmed.

## Slide-Friendly Takeaway

We did not build one fixed pipeline immediately. We explored multiple
image-to-music strategies:

- direct pixel-to-music mapping for speed
- LLM description for semantic understanding
- MIDI selection for curated local playback
- arrangement generation for musical structure
- ambient synthesis for reliable live deployment

The final design keeps these as selectable modes behind one launcher, making
the project easier to deploy, demo, and extend.
