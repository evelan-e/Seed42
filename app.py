from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import streamlit as st


SAMPLE_RATE = 22_050
ESSENTIA_SAMPLE_RATE = 16_000
ESSENTIA_MODEL_DIR = Path("essentia_models")
ESSENTIA_MODEL_BASE = "https://essentia.upf.edu/models"
MOOD_NAMES = ("happy", "sad", "relaxed", "aggressive", "party")
STOPWORDS = set(
    "a an the and or but if then than that this in on into to for with "
    "without about oh yeah uh ah hey".split()
)
FALLBACK_PROMPT = "dark club space, sub-bass pulse, neon haze, slow camera drift"
MAX_PROMPT_WORDS = 40

_essentia_models: dict[str, Any] | None = None


def _download_model(subdir: str, filename: str) -> Path:
    destination = ESSENTIA_MODEL_DIR / filename
    if destination.exists():
        return destination

    ESSENTIA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    url = f"{ESSENTIA_MODEL_BASE}/{subdir}/{filename}"
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


@st.cache_resource
def load_essentia_models() -> dict[str, Any]:
    global _essentia_models
    if _essentia_models is not None:
        return _essentia_models

    import collections
    import collections.abc

    for name, value in {
        "float": float,
        "int": int,
        "complex": complex,
        "object": object,
        "bool": bool,
        "str": str,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    for name, value in {
        "NINF": -np.inf,
        "PINF": np.inf,
        "Inf": np.inf,
        "float_": np.float64,
        "complex_": np.complex128,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)
    for name in (
        "MutableSequence",
        "Iterable",
        "Callable",
        "MutableMapping",
        "Mapping",
        "Sequence",
    ):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))

    try:
        from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    except ImportError as error:
        raise RuntimeError(
            "Essentia is not installed. Install the optional model dependencies "
            "described in README.md, or turn off genre/mood classification."
        ) from error

    model_files: dict[str, tuple[str, str]] = {
        "embeddings": ("feature-extractors/discogs-effnet", "discogs-effnet-bs64-1.pb"),
        "genre": (
            "classification-heads/genre_discogs400",
            "genre_discogs400-discogs-effnet-1.pb",
        ),
        "genre_meta": (
            "classification-heads/genre_discogs400",
            "genre_discogs400-discogs-effnet-1.json",
        ),
    }
    for mood in MOOD_NAMES:
        for extension in (".pb", ".json"):
            model_files[f"{mood}{extension}"] = (
                f"classification-heads/mood_{mood}",
                f"mood_{mood}-discogs-effnet-1{extension}",
            )
    paths = {
        key: _download_model(subdir, filename)
        for key, (subdir, filename) in model_files.items()
    }

    with paths["genre_meta"].open(encoding="utf-8") as metadata_file:
        genre_classes = json.load(metadata_file)["classes"]
    embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename=str(paths["embeddings"]),
        output="PartitionedCall:1",
    )
    genre_head = TensorflowPredict2D(
        graphFilename=str(paths["genre"]),
        input="serving_default_model_Placeholder",
        output="PartitionedCall",
    )
    mood_heads = {}
    for mood in MOOD_NAMES:
        with paths[f"{mood}.json"].open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        schema = metadata.get("schema", {})
        input_name = schema.get("inputs", [{}])[0].get("name", "model/Placeholder")
        output_name = next(
            (
                output["name"]
                for output in schema.get("outputs", [])
                if output.get("output_purpose") == "predictions"
            ),
            "model/Softmax",
        )
        mood_head = TensorflowPredict2D(
            graphFilename=str(paths[f"{mood}.pb"]),
            input=input_name,
            output=output_name,
        )
        mood_heads[mood] = (mood_head, metadata["classes"].index(mood))

    _essentia_models = {
        "embedding": embedding_model,
        "genre": genre_head,
        "genre_classes": genre_classes,
        "moods": mood_heads,
    }
    return _essentia_models


@st.cache_resource
def load_whisper_model() -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "Faster-Whisper is not installed. Install the optional model dependencies "
            "described in README.md, or turn off lyric theme extraction."
        ) from error

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel("tiny", device=device, compute_type=compute_type)


def _mood_label(embeddings: np.ndarray, mood_heads: dict[str, Any]) -> str:
    scores = {}
    for mood, (head, index) in mood_heads.items():
        scores[mood] = float(np.mean(head(embeddings), axis=0)[index])
    selected = sorted(
        ((score, mood) for mood, score in scores.items() if score >= 0.5),
        reverse=True,
    )[:2]
    return ", ".join(mood for _, mood in selected) if selected else "neutral"


def _lyric_themes(
    audio: np.ndarray,
    whisper_model: Any,
    count: int = 4,
) -> str:
    segments, _ = whisper_model.transcribe(audio, beam_size=1, vad_filter=True)
    tokens = []
    for segment in segments:
        if getattr(segment, "no_speech_prob", 0.0) <= 0.6:
            words = re.sub(r"[^a-z\s]", " ", segment.text.lower()).split()
            tokens.extend(word for word in words if len(word) >= 4 and word not in STOPWORDS)
    themes = [word for word, _ in Counter(tokens).most_common(count)]
    return ", ".join(themes) if themes else "instrumental"


def dominant_frequency_band(audio: np.ndarray, sample_rate: int) -> str:
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)

    def band_energy(low: int, high: int) -> float:
        mask = (frequencies > low) & (frequencies <= high)
        return float(np.mean(spectrum[mask, :])) if np.any(mask) else 0.0

    bands = {
        "Low-End / Sub-Bass Focus (20–120 Hz)": band_energy(20, 120),
        "Mid-Range / Vocal & Synth Focus (120 Hz–2 kHz)": band_energy(120, 2000),
        "High-End / Percussion & Hi-Hat Focus (2–8 kHz)": band_energy(2000, 8000),
    }
    return max(bands, key=bands.get)


def energy_descriptor(rms: float) -> str:
    if rms < 0.15:
        phase = "Low Energy / Breakdown Phase"
    elif rms < 0.40:
        phase = "Moderate Energy / Steady Groove Phase"
    elif rms < 0.65:
        phase = "High Energy / Build-up Phase"
    else:
        phase = "Peak Energy / Drop Phase"
    return f"{rms:.3f} ({phase})"


def _llm_input(features: dict[str, Any]) -> str:
    genre = features["primary_genre"].split("---")[-1]
    energy_phase = energy_descriptor(features["energy"]).split("(")[1].rstrip(")")
    band_name = features["dominant_frequency_band"].split("(")[0].strip()
    return (
        f"A {features['mood']} mood, {energy_phase} {features['bpm']} BPM "
        f"{genre} atmosphere, featuring {band_name} with {features['lyric_themes']} themes."
    )


def _trim_prompt(text: str, max_words: int = MAX_PROMPT_WORDS) -> str:
    cleaned = text.replace("\n", ", ").replace("**", "").replace('"', "").replace("\u2011", "-")
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
        if "," in cleaned:
            cleaned = cleaned.rsplit(",", 1)[0]
    return cleaned.strip(" ,.")


def generate_visual_prompt(
    features: dict[str, Any],
    api_key: str | None,
) -> tuple[str, str]:
    if not api_key:
        bpm = features["bpm"]
        tempo = (
            "slow drifting pulse"
            if bpm < 75
            else "laid-back rhythmic pulse"
            if bpm < 100
            else "steady rhythmic pulse"
            if bpm < 125
            else "driving rhythmic pulse"
            if bpm < 155
            else "rapid rhythmic pulse"
        )
        centroid = features["spectral_centroid_hz"]
        tone = (
            "deep shadowed tones"
            if centroid < 250
            else "warm amber tones"
            if centroid < 700
            else "clear crystalline tones"
            if centroid < 1800
            else "bright vivid tones"
            if centroid < 4000
            else "sparkling electric tones"
        )
        intensity = (
            "restrained dynamics"
            if features["energy"] < 0.15
            else "balanced dynamics"
            if features["energy"] < 0.40
            else "powerful dynamics"
            if features["energy"] < 0.65
            else "explosive dynamics"
        )
        mood = features["mood"]
        if mood == "unavailable":
            mood = "atmospheric" if centroid < 1800 else "electric"
        band = features["dominant_frequency_band"].split("(")[0].strip()
        band = band.replace("Low-End / Sub-Bass Focus", "sub-bass")
        band = band.replace("Mid-Range / Vocal & Synth Focus", "voice and synth")
        band = band.replace("High-End / Percussion & Hi-Hat Focus", "percussion and hi-hat")
        prompt = (
            f"abstract {mood} dreamscape, {band} textures, {tempo}, {tone}, "
            f"{intensity}, cinematic lighting, layered atmosphere, drifting particles, "
            "slow camera movement"
        )
        return _trim_prompt(prompt), "Local prompt (no Groq API key)"

    from openai import OpenAI

    system_prompt = (
        "You write prompts for a real-time SDXL-Turbo image stream. "
        "Turn the audio cue into ONE line of 8 to 12 short visual phrases separated by commas, "
        f"{MAX_PROMPT_WORDS} words maximum.\n"
        "Order: main scene first, then surface materials, lighting colours, atmosphere "
        "(fog, particles, air movement), glowing elements, and camera motion last.\n"
        "RULES: no full sentences, no markdown, no quotes, no numbers or BPM, "
        "no genre names, no words or text to appear in the image."
    )
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Audio cue: {_llm_input(features)}"},
        ],
        temperature=0.7,
        max_tokens=400,
        extra_body={"reasoning_effort": "low"},
    )
    generated = (response.choices[0].message.content or "").strip()
    return _trim_prompt(generated or FALLBACK_PROMPT), "Groq LLM"


def _analyze_window(
    audio: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    use_essentia: bool,
    whisper_model: Any | None,
) -> dict[str, Any]:
    energy = float(np.sqrt(np.mean(np.square(audio))))
    tempo = librosa.feature.tempo(y=audio, sr=sample_rate, start_bpm=120)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)

    primary_genre = "unavailable"
    secondary_genre = "N/A"
    mood = "unavailable"
    if use_essentia:
        models = load_essentia_models()
        audio_16k = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=ESSENTIA_SAMPLE_RATE,
        )
        audio_16k = np.ascontiguousarray(audio_16k, dtype=np.float32)
        embeddings = models["embedding"](audio_16k)
        probabilities = np.mean(models["genre"](embeddings), axis=0)
        genre_indexes = np.argsort(probabilities)[::-1][:2]
        genres = [models["genre_classes"][index] for index in genre_indexes]
        primary_genre = genres[0]
        secondary_genre = genres[1] if len(genres) > 1 else "N/A"
        mood = _mood_label(embeddings, models["moods"])

    themes = (
        _lyric_themes(
            librosa.resample(audio, orig_sr=sample_rate, target_sr=ESSENTIA_SAMPLE_RATE),
            whisper_model,
        )
        if whisper_model is not None
        else "not extracted"
    )
    band = dominant_frequency_band(audio, sample_rate)
    spectral_centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
    )
    features = {
        "timestamp_sec": round(start_seconds, 1),
        "bpm": bpm,
        "energy": round(energy, 4),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "primary_genre": primary_genre,
        "secondary_genre": secondary_genre,
        "mood": mood,
        "lyric_themes": themes,
        "dominant_frequency_band": band,
    }
    features["llm_input"] = _llm_input(features)
    return features


def analyze_audio(
    audio_bytes: bytes,
    filename: str,
    window_seconds: int,
    use_essentia: bool,
    extract_lyrics: bool,
) -> list[dict[str, Any]]:
    suffix = Path(filename).suffix or ".audio"
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_path = temporary_file.name
        audio, sample_rate = librosa.load(temporary_path, sr=SAMPLE_RATE, mono=True)
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)

    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty or could not be decoded.")

    whisper_model = load_whisper_model() if extract_lyrics else None
    window_samples = window_seconds * sample_rate
    results = []
    for start in range(0, len(audio), window_samples):
        segment = audio[start : start + window_samples]
        results.append(
            _analyze_window(
                segment,
                sample_rate,
                start / sample_rate,
                use_essentia,
                whisper_model,
            )
        )
    return results


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")


st.set_page_config(page_title="Audio Analyser", page_icon="🎵", layout="wide")
st.title("Audio to Visual Prompts")
st.caption("Upload a song to generate a visual prompt for each analysis window.")

with st.sidebar:
    st.header("Analysis options")
    window_seconds = st.selectbox(
        "Analysis window",
        (30, 60),
        index=1,
        format_func=lambda value: f"{value} seconds",
    )
    use_essentia = st.checkbox("Classify genre and mood (Essentia models)", value=False)
    extract_lyrics = st.checkbox("Extract lyric themes (Faster-Whisper)", value=False)
    groq_api_key = st.text_input(
        "Groq API key (optional)",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Used only for visual-prompt generation. Leave blank to use the local prompt template.",
    )

uploaded_audio = st.file_uploader(
    "Choose an audio file",
    type=("mp3", "wav", "m4a", "flac", "ogg", "aac"),
)

if uploaded_audio is not None:
    audio_bytes = uploaded_audio.getvalue()
    audio_id = hashlib.sha256(audio_bytes).digest()
    st.audio(audio_bytes, format=uploaded_audio.type)
    if st.button("Generate prompts", type="primary"):
        try:
            with st.spinner("Analysing audio and generating prompts..."):
                analysis = analyze_audio(
                    audio_bytes,
                    uploaded_audio.name,
                    window_seconds,
                    use_essentia,
                    extract_lyrics,
                )
                prompts = []
                for row in analysis:
                    prompt, source = generate_visual_prompt(row, groq_api_key.strip() or None)
                    prompts.append(
                        {
                            "timestamp_sec": row["timestamp_sec"],
                            "prompt": prompt,
                            "source": source,
                        }
                    )
            st.session_state["visual_prompts"] = prompts
            st.session_state["analysis_audio_id"] = audio_id
        except Exception as error:
            st.error(f"Pipeline failed: {error}")
            st.exception(error)

prompts = (
    st.session_state.get("visual_prompts")
    if uploaded_audio is not None
    and st.session_state.get("analysis_audio_id") == audio_id
    else None
)
if prompts:
    st.subheader("Generated prompts")
    for item in prompts:
        st.markdown(f"**{item['timestamp_sec']:.1f}s**")
        st.write(item["prompt"])
    st.download_button(
        "Download prompts JSON",
        data=_json_bytes(prompts),
        file_name="visual-prompts.json",
        mime="application/json",
    )
elif uploaded_audio is None:
    st.info("Upload an audio file to get started.")
