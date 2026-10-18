"""Feature extraction for one window of audio.

Tempo (Madmom, falling back to Librosa), energy and its phase, the loudest
frequency band, genre and mood (Essentia) and lyric keywords (Whisper), plus
the one-sentence description the LLM turns into an image prompt. Each result
carries short machine keys (phase, band) next to the readable labels, and
per-stage timings.
"""
import io
import time
from collections import Counter

import librosa
import numpy as np
import soundfile as sf

from . import config
from .models import MODELS, STOPWORDS

PHASES = [  # (upper RMS bound, key, label)
    (0.15, "breakdown", "Low Energy / Breakdown Phase"),
    (0.40, "groove", "Moderate Energy / Steady Groove Phase"),
    (0.65, "build", "High Energy / Build-up Phase"),
    (float("inf"), "drop", "Peak Energy / Drop Phase"),
]

BANDS = [  # (key, label, lo Hz, hi Hz)
    ("sub", "Low-End / Sub-Bass Focus (60 Hz - 120 Hz)", 20, 120),
    ("mid", "Mid-Range / Vocal & Synth Focus (120 Hz - 2 kHz)", 120, 2000),
    ("high", "High-End / Percussion & Hi-Hat Focus (2 kHz - 8 kHz)", 2000, 8000),
]


# Jamendo tags that name a use or an occasion rather than how the music feels, or
# that would steer an image prompt somewhere unwanted. Never picked or scored.
JAMENDO_SKIP = {"advertising", "background", "children", "christmas", "commercial",
                "corporate", "documentary", "film", "game", "holiday", "motivational",
                "movie", "sexy", "sport", "trailer", "travel"}
# Jamendo scores each of its 56 tags independently and they run well below 0.5,
# so a much lower bar. A starting guess: tune it on real tracks.
JAMENDO_MIN = 0.05


def get_mood(embeddings, threshold=0.5, top_k=2):
    """Up to `top_k` moods above the bar, strongest first, or 'neutral'; plus every score.

    MOOD_MODEL=jamendo: one head scores all its tags at once, bar JAMENDO_MIN.
    MOOD_MODEL=heads: five yes/no heads, one per mood, bar `threshold`.
    """
    scores = {}
    if MODELS.mood_tagger:
        head, classes = MODELS.mood_tagger
        threshold = JAMENDO_MIN
        try:
            probs = np.mean(head(embeddings), axis=0)
            scores = {c: float(p) for c, p in zip(classes, probs) if c not in JAMENDO_SKIP}
        except Exception:
            pass
    for m, (head, idx) in MODELS.mood_heads.items():
        try:
            scores[m] = float(np.mean(head(embeddings), axis=0)[idx])
        except Exception:
            pass
    picked = sorted(((s, m) for m, s in scores.items() if s >= threshold), reverse=True)[:top_k]
    return (", ".join(m for _, m in picked) if picked else "neutral"), scores


def extract_fast_features(y_slice, sr=config.SR, n_themes=3):
    """Runs Librosa, Madmom, Essentia and Whisper on one audio slice, entirely in RAM."""
    timings = {}

    # --- 1. Librosa: energy and fallback tempo ---------------------------------
    t = time.perf_counter()
    rms = float(np.sqrt(np.mean(y_slice ** 2)))
    librosa_bpm = round(float(np.atleast_1d(
        librosa.feature.tempo(y=y_slice, sr=sr, start_bpm=120))[0]), 1)
    timings["librosa"] = time.perf_counter() - t

    # --- 2. Madmom: neural beat tracking --------------------------------------
    madmom_bpm, beat_count = 0.0, 0
    if MODELS.madmom_ok:
        t = time.perf_counter()
        try:
            wav_io = io.BytesIO()                        # in-memory WAV, no disk writes
            sf.write(wav_io, y_slice, sr, format="WAV")
            wav_io.seek(0)
            beat_times = np.asarray(
                MODELS.DBNBeatTrackingProcessor(fps=100)(MODELS.RNNBeatProcessor()(wav_io)), dtype=float)
            if len(beat_times) >= 2:
                madmom_bpm = round(60.0 / float(np.median(np.diff(beat_times))), 1)
                beat_count = len(beat_times)
        except Exception:
            madmom_bpm = 0.0
        timings["madmom"] = time.perf_counter() - t

    if madmom_bpm > 0:
        bpm, bpm_source = madmom_bpm, "Madmom neural"
    else:
        bpm, bpm_source = librosa_bpm, "Librosa DSP"

    # --- 3. Essentia: genre and mood ------------------------------------------
    y16 = None
    mood_scores = {}
    if MODELS.essentia_ok or MODELS.whisper_ok:
        y16 = librosa.resample(y_slice, orig_sr=sr, target_sr=config.ESSENTIA_SR) if sr != config.ESSENTIA_SR else y_slice
        y16 = np.ascontiguousarray(y16, dtype=np.float32)

    if MODELS.essentia_ok:
        t = time.perf_counter()
        embeddings = MODELS.embedding_model(y16)
        genre_probs = np.mean(MODELS.genre_head(embeddings), axis=0)
        top_idx = np.argsort(genre_probs)[::-1][:2]
        top_genres = [MODELS.genre_classes[i] for i in top_idx]
        genre_scores = [round(float(genre_probs[i]), 3) for i in top_idx]
        mood_str, mood_scores = get_mood(embeddings) if MODELS.mood_ok else ("unavailable", {})
        timings["essentia"] = time.perf_counter() - t
    else:
        top_genres, genre_scores, mood_str = ["unavailable"], [], "unavailable"

    primary_genre = top_genres[0]
    secondary_genre = top_genres[1] if len(top_genres) > 1 else "N/A"
    genres_str = ", ".join(top_genres)

    # --- 4. Whisper: lyric keywords -------------------------------------------
    # Only word counts are kept; the transcript itself is never stored or returned.
    if MODELS.whisper_ok:
        t = time.perf_counter()
        segments, _ = MODELS.whisper.transcribe(y16, beam_size=1, vad_filter=True)
        tokens = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0.0) <= 0.6:
                words = "".join(c.lower() if c.isalpha() or c.isspace() else " " for c in seg.text).split()
                tokens.extend(w for w in words if len(w) >= 4 and w not in STOPWORDS)
        top_themes = [w for w, _ in Counter(tokens).most_common(n_themes)]
        themes_str = ", ".join(top_themes) if top_themes else "instrumental"
        timings["whisper"] = time.perf_counter() - t
    else:
        themes_str = "unavailable"

    return {
        "bpm": bpm,
        "bpm_source": bpm_source,
        "madmom_bpm": madmom_bpm,
        "librosa_bpm": librosa_bpm,
        "beat_count": beat_count,
        "energy": round(rms, 4),
        "top_2_genres": genres_str,
        "genre_scores": genre_scores,
        "primary_genre": primary_genre,
        "secondary_genre": secondary_genre,
        "mood": mood_str,
        "mood_scores": {k: round(v, 3) for k, v in mood_scores.items()},
        "lyric_themes": themes_str,
        "timings": {k: round(v * 1000) for k, v in timings.items()},
    }


def get_dominant_frequency_band(y_slice, sr):
    stft = np.abs(librosa.stft(y_slice, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band_energy(lo, hi):
        mask = (freqs > lo) & (freqs <= hi)
        return float(np.mean(stft[mask, :])) if np.any(mask) else 0.0

    energies = {key: band_energy(lo, hi) for key, _, lo, hi in BANDS}
    key = max(energies, key=energies.get)
    label = next(lbl for k, lbl, _, _ in BANDS if k == key)
    return key, label


def get_energy_phase(rms):
    for bound, key, label in PHASES:
        if rms < bound:
            return key, label


def get_energy_descriptor(rms):
    return f"{rms:.3f} ({get_energy_phase(rms)[1]})"


def build_llm_input(features, energy_desc, dominant_band):
    """One sentence describing the window, used as the LLM's input.

    When a model is missing (lite mode, a failed download) its part is left
    out instead of sending the LLM the word "unavailable".
    """
    energy_phase = energy_desc.split("(")[1].rstrip(")")
    band_name = dominant_band.split("(")[0].strip()
    mood = features["mood"]
    genre = features["primary_genre"]
    themes = features["lyric_themes"]

    if "unavailable" not in (mood, genre, themes):
        genre_name = genre.split("---")[-1]    # "Electronic---House" -> "House"
        return (f"A {mood} mood, {energy_phase} {features['bpm']} BPM {genre_name} atmosphere, "
                f"featuring {band_name} with {themes} themes.")

    parts = [f"A {mood} mood," if mood != "unavailable" else "A",
             f"{energy_phase} {features['bpm']} BPM"]
    if genre != "unavailable":
        parts.append(genre.split("---")[-1])
    parts.append(f"atmosphere, featuring {band_name}")
    sentence = " ".join(parts)
    if themes != "unavailable":
        sentence += f" with {themes} themes"
    return sentence + "."


def analyse_window(y_slice, sr=config.SR):
    """Everything the test page shows for one window, except the LLM and API steps."""
    t0 = time.perf_counter()
    features = extract_fast_features(y_slice, sr=sr, n_themes=config.N_THEMES)
    phase_key, phase_label = get_energy_phase(features["energy"])
    t = time.perf_counter()
    band_key, band_label = get_dominant_frequency_band(y_slice, sr)
    features["timings"]["bands"] = round((time.perf_counter() - t) * 1000)
    energy_desc = get_energy_descriptor(features["energy"])
    features.update({
        "phase": phase_key,
        "phase_label": phase_label,
        "band": band_key,
        "band_label": band_label,
        "llm_input": build_llm_input(features, energy_desc, band_label),
        "analysis_ms": round((time.perf_counter() - t0) * 1000),
    })
    return features


def energy_envelope(y, sr=config.SR, max_points=1200):
    """Raw RMS over the whole track for the page's energy strip.

    Same RMS scale as the window energy, so the phase thresholds line up.
    """
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if len(rms) > max_points:
        block = int(np.ceil(len(rms) / max_points))
        pad = (-len(rms)) % block
        rms = np.pad(rms, (0, pad)).reshape(-1, block).mean(axis=1)
        dt = block * hop / sr
    else:
        dt = hop / sr
    return {"values": [round(float(v), 4) for v in rms], "dt": dt}
