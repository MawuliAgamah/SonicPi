Playing around with camera-driven image/audio experiments.

## Install

Install Python dependencies:

```bash
pip install -r requirements.txt
```

On Raspberry Pi OS, install the system audio/camera dependencies too:

```bash
sudo apt update
sudo apt install fluidsynth fluid-soundfont-gm python3-opencv python3-picamera2
```

## Image description module

`module/image_describer.py` can:

- load an image with OpenCV
- extract visual features similar to `synth.py`
- crop useful image parts, including the full image, center crop, quadrants, and contour-based regions
- send the image parts to the OpenAI Responses API for a concise description

Install the image dependencies if your environment does not already have them:

```bash
pip install numpy opencv-python
```

Set your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

Run the module from the repo root:

```bash
python3 -m module.image_describer path/to/image.jpg --save-parts extracted_parts
```

Use it from Python:

```python
from module import describe_image

result = describe_image("path/to/image.jpg")
print(result.description)
```

## Root entrypoint

Use `main.py` when you want one place to orchestrate the full image-to-music flow:

```bash
export OPENAI_API_KEY="your_openai_key"
export ELEVENLABS_API_KEY="your_elevenlabs_key"

python3 main.py path/to/image.jpg --save-parts --music-model music_v1
```

That entrypoint imports from `module/`, then runs:

- OpenCV image loading and feature extraction
- OpenAI image description
- ElevenLabs music prompt creation
- ElevenLabs Music generation
- output saving to `generated_music/`

You can choose the ElevenLabs model with `--music-model` or `ELEVENLABS_MUSIC_MODEL`:

```bash
python3 main.py path/to/image.jpg --music-model music_v1 --output-format mp3_44100_128
```

## Camera to ElevenLabs Music

ElevenLabs Music API access requires a paid ElevenLabs account.

In browser mode, `synth.py` now shows a `Generate music from camera` button. When pressed, it:

- snapshots the current camera frame
- asks OpenAI for an image description
- turns that description and the OpenCV features into a music prompt
- sends the prompt to ElevenLabs Music
- saves the returned audio in `generated_music/`
- makes the generated audio playable in the browser

Set both API keys before running:

```bash
export OPENAI_API_KEY="your_openai_key"
export ELEVENLABS_API_KEY="your_elevenlabs_key"
```

Run the camera UI:

```bash
python3 synth.py --browser
```

Optional settings:

```bash
export ELEVENLABS_MUSIC_MODEL="music_v1"
export ELEVENLABS_MUSIC_OUTPUT_FORMAT="mp3_44100_128"
export GENERATED_MUSIC_DIR="generated_music"
```

## Camera-selected MIDI playback

You can also keep a folder of premade `.mid` files and let OpenAI choose which one FluidSynth should play based on the current camera image.

### Test on macOS first

Use `mac.py` to test the same camera-to-MIDI idea with your Mac webcam before moving to the Pi:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_key"
python3 mac.py --midi-dir midi/free_chord_progressions
```

Then open the printed local URL, usually:

```text
http://127.0.0.1:8090/
```

Browser buttons:

- `Snapshot and choose MIDI`: snapshots the webcam frame, chooses a MIDI file from the manifest, and plays/opens it
- `Play chord + drums`: chooses a chord progression and drum pattern from separate libraries, merges them into a longer arrangement, and plays it
- `Replay last MIDI`: plays/opens the last chosen MIDI again
- `Stop MIDI`: stops local FluidSynth playback

The arranger saves generated files in `generated_arrangements/`. Useful options:

```bash
python3 mac.py \
  --midi-dir midi/free_chord_progressions \
  --drum-dir midi/dmp_drum_patterns \
  --target-seconds 60 \
  --tempo-bpm 110 \
  --chord-program 0
```

Use `--chord-program -1` to preserve the selected chord MIDI's original instruments.

By default, `mac.py` opens the MIDI file with macOS if local FluidSynth is not configured. To audition through FluidSynth on Mac, install the executable:

```bash
brew install fluid-synth
```

Then provide a `.sf2` SoundFont. `mac.py` auto-detects common Homebrew locations, or you can set it explicitly:

```bash
python3 mac.py --midi-dir midi/free_chord_progressions --soundfont /path/to/FluidR3_GM.sf2
```

You can also put the path in `.env`:

```bash
SOUNDFONT_PATH=/path/to/FluidR3_GM.sf2
```

### Import Free-Chord-Progressions

The repo includes an importer for BenLeon2001's Free-Chord-Progressions packs. It downloads the GitHub archive, extracts all `.mid` / `.midi` files from the ZIP packs, and builds `midi_manifest.json` automatically:

```bash
python3 scripts/import_free_chord_progressions.py --output-dir midi/free_chord_progressions
```

Then run the camera app against that folder:

```bash
export OPENAI_API_KEY="your_openai_key"
python3 synth.py --browser --midi-dir midi/free_chord_progressions
```

### Import dmp_midi drum patterns

The repo also includes an importer for `gvellut/dmp_midi/input`. Those source patterns are JSON drum-machine patterns. The importer converts each pattern to a General MIDI drum `.mid` file and writes a role-aware manifest:

```bash
python3 scripts/import_dmp_midi_drums.py --output-dir midi/dmp_drum_patterns
```

Generated drum MIDI uses General MIDI percussion on channel 10:

```text
BassDrum     -> 36
SnareDrum    -> 38
RimShot      -> 37
Clap         -> 39
ClosedHiHat  -> 42
OpenHiHat    -> 46
LowTom       -> 45
MediumTom    -> 47
HighTom      -> 50
Cymbal       -> 49
Cowbell      -> 56
Tambourine   -> 54
```

These drum files work with the same General MIDI SoundFont you already use for FluidSynth, such as `soundfonts/GeneralUser_GS.sf2`.

To test only drum patterns on Mac:

```bash
python3 mac.py --midi-dir midi/dmp_drum_patterns --soundfont soundfonts/GeneralUser_GS.sf2
```

Create a MIDI folder on the Pi:

```bash
mkdir -p ~/midi
```

Put `.mid` or `.midi` files in that folder, then run:

```bash
export OPENAI_API_KEY="your_openai_key"
python3 synth.py --browser --midi-dir ~/midi
```

In the browser UI, press `Play matching MIDI`. The app will:

- snapshot the camera frame
- describe it with OpenAI
- show OpenAI the local MIDI filenames
- pick the best matching MIDI file
- play that file through the existing FluidSynth engine

The camera still influences playback controls while the MIDI starts: brightness affects volume, motion/edge density affect expression, and saturation affects chorus/reverb.

### MIDI manifest

For better choices, add metadata in `~/midi/midi_manifest.json`. Start from the repo example:

```bash
cp midi_manifest.example.json ~/midi/midi_manifest.json
```

Each track entry can include:

```json
{
  "file": "moonlight-sonata.mid",
  "title": "Moonlight Sonata",
  "composer": "Ludwig van Beethoven",
  "style": "classical piano",
  "mood": "dark, reflective, intimate",
  "energy": "low",
  "tags": ["night", "melancholy", "solo piano", "slow"],
  "notes": "Good for dim scenes, calm portraits, shadows, or quiet indoor images."
}
```

`file` must match either the filename or the relative path inside the MIDI folder. Files missing from the manifest still work; the selector will fall back to the filename.

You can also pass a manifest explicitly:

```bash
python3 synth.py --browser --midi-dir ~/midi --midi-manifest ~/midi/midi_manifest.json
```
