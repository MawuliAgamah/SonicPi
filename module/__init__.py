"""Reusable image extraction, description, and music generation helpers."""

from .image_describer import (
    ImageDescription,
    ImagePart,
    describe_image,
    extract_features,
    extract_image_parts,
    load_image,
    save_parts,
)
from .eleven_labs import (
    ElevenLabsMusicResult,
    build_elevenlabs_prompt,
    generate_elevenlabs_music,
    save_elevenlabs_music_result,
)
from .midi_selector import (
    MidiCandidate,
    MidiSelection,
    choose_midi_for_image,
    choose_midi_from_description,
    list_midi_files,
    load_midi_manifest,
)
from .midi_arranger import (
    MidiArrangementResult,
    create_chord_drum_arrangement,
    default_arrangement_path,
)

__all__ = [
    "ElevenLabsMusicResult",
    "ImageDescription",
    "ImagePart",
    "MidiArrangementResult",
    "MidiCandidate",
    "MidiSelection",
    "build_elevenlabs_prompt",
    "choose_midi_for_image",
    "choose_midi_from_description",
    "create_chord_drum_arrangement",
    "default_arrangement_path",
    "describe_image",
    "extract_features",
    "extract_image_parts",
    "generate_elevenlabs_music",
    "list_midi_files",
    "load_midi_manifest",
    "load_image",
    "save_elevenlabs_music_result",
    "save_parts",
]
