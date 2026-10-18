"""Loads every model once and keeps it in memory.

Each component loads independently and reports its own status, so a failed
download disables one feature instead of the whole pipeline: genre and mood
become "unavailable", or tempo falls back to Librosa.
"""
import collections
import collections.abc
import json
import logging
import os
import threading
import time
import urllib.request
import warnings

import numpy as np

from . import config

log = logging.getLogger("pipeline.models")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# --- NumPy / collections shims (must run before importing madmom) ---------
for _name, _py in {"float": float, "int": int, "complex": complex,
                   "object": object, "bool": bool, "str": str}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _py)
for _name, _val in {"NINF": -np.inf, "PINF": np.inf, "Inf": np.inf,
                    "float_": np.float64, "complex_": np.complex128}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _val)
for _abc in ("MutableSequence", "Iterable", "Callable", "MutableMapping", "Mapping", "Sequence"):
    if not hasattr(collections, _abc):
        setattr(collections, _abc, getattr(collections.abc, _abc))

MODEL_BASE = "https://essentia.upf.edu/models"
MOOD_NAMES = ["happy", "sad", "relaxed", "aggressive", "party"]   # MOOD_MODEL=heads
JAMENDO = "mtg_jamendo_moodtheme-discogs-effnet-1"                 # MOOD_MODEL=jamendo
STOPWORDS = set("a an the and or but if then than that this in on into to for with "
                "without about oh yeah uh ah hey".split())


class Models:
    """Holds the loaded models plus a status line per component for /api/health."""

    def __init__(self):
        self.status = {
            "madmom": {"state": "pending", "detail": ""},
            "essentia": {"state": "pending", "detail": ""},
            "mood": {"state": "pending", "detail": ""},
            "whisper": {"state": "pending", "detail": ""},
        }
        self.ready = threading.Event()
        self.device = "cpu"
        # Filled by load()
        self.RNNBeatProcessor = None
        self.DBNBeatTrackingProcessor = None
        self.embedding_model = None
        self.genre_head = None
        self.genre_classes = []
        self.mood_heads = {}        # MOOD_MODEL=heads: {mood: (head, index of "yes")}
        self.mood_tagger = None     # MOOD_MODEL=jamendo: (head, tag names)
        self.whisper = None

    # -- helpers -----------------------------------------------------------
    def _set(self, key, state, detail=""):
        self.status[key] = {"state": state, "detail": detail}
        log.info("%-8s %s %s", key, state, detail)

    @property
    def madmom_ok(self):
        return self.status["madmom"]["state"] == "ready"

    @property
    def essentia_ok(self):
        return self.status["essentia"]["state"] == "ready"

    @property
    def mood_ok(self):
        return self.status["mood"]["state"] == "ready"

    @property
    def whisper_ok(self):
        return self.status["whisper"]["state"] == "ready"

    def load_in_background(self):
        threading.Thread(target=self.load, name="model-loader", daemon=True).start()

    # -- loading -----------------------------------------------------------
    def load(self):
        t0 = time.time()
        if config.LITE:
            for k in self.status:
                self._set(k, "skipped", "lite mode")
            self.ready.set()
            return

        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                self.device = "cuda"
        except Exception:
            pass

        self._load_madmom()
        self._load_essentia()
        self._load_whisper()
        log.info("models loaded in %.1fs on %s", time.time() - t0, self.device)
        self.ready.set()

    def _load_madmom(self):
        self._set("madmom", "loading")
        try:
            from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
            self.RNNBeatProcessor = RNNBeatProcessor
            self.DBNBeatTrackingProcessor = DBNBeatTrackingProcessor
            self._set("madmom", "ready")
        except Exception as e:
            self._set("madmom", "failed", f"falling back to Librosa tempo: {e}")

    def _download_essentia(self):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        files = {
            "embeddings": ("feature-extractors/discogs-effnet", "discogs-effnet-bs64-1.pb"),
            "genre": ("classification-heads/genre_discogs400", "genre_discogs400-discogs-effnet-1.pb"),
            "genre_meta": ("classification-heads/genre_discogs400", "genre_discogs400-discogs-effnet-1.json"),
        }
        for ext in (".pb", ".json"):
            if config.MOOD_MODEL == "jamendo":
                files[f"jamendo{ext}"] = ("classification-heads/mtg_jamendo_moodtheme", f"{JAMENDO}{ext}")
            else:
                for m in MOOD_NAMES:
                    files[f"mood_{m}{ext}"] = (f"classification-heads/mood_{m}", f"mood_{m}-discogs-effnet-1{ext}")
        for subdir, fname in files.values():
            dest = os.path.join(config.MODEL_DIR, fname)
            if not os.path.exists(dest):
                log.info("downloading %s", fname)
                tmp = dest + ".part"
                urllib.request.urlretrieve(f"{MODEL_BASE}/{subdir}/{fname}", tmp)
                os.replace(tmp, dest)   # no half-written model files left behind

    def _load_essentia(self):
        self._set("essentia", "loading", "downloading models on first run")
        self._set("mood", "loading")
        try:
            from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
            self._download_essentia()
            d = config.MODEL_DIR
            self.embedding_model = TensorflowPredictEffnetDiscogs(
                graphFilename=os.path.join(d, "discogs-effnet-bs64-1.pb"),
                output="PartitionedCall:1",
            )
            with open(os.path.join(d, "genre_discogs400-discogs-effnet-1.json")) as f:
                self.genre_classes = json.load(f)["classes"]
            self.genre_head = TensorflowPredict2D(
                graphFilename=os.path.join(d, "genre_discogs400-discogs-effnet-1.pb"),
                input="serving_default_model_Placeholder",
                output="PartitionedCall",
            )
            self._set("essentia", "ready", f"{len(self.genre_classes)} genres")
        except Exception as e:
            self._set("essentia", "failed", f"genre and mood unavailable: {e}")
            self._set("mood", "failed", "needs the Essentia embedding model")
            return

        # Mood loads separately so a mood failure never disables genre
        if config.MOOD_MODEL == "jamendo":
            try:
                self.mood_tagger = self._mood_head(TensorflowPredict2D, JAMENDO)
                self._set("mood", "ready", f"Jamendo, {len(self.mood_tagger[1])} mood and theme tags")
            except Exception as e:
                self._set("mood", "failed", f"Jamendo mood/theme: {e}")
            return
        skipped = []
        for m in MOOD_NAMES:
            try:
                head, classes = self._mood_head(TensorflowPredict2D, f"mood_{m}-discogs-effnet-1")
                self.mood_heads[m] = (head, classes.index(m))
            except Exception as e:
                skipped.append(f"{m} ({e})")
        if self.mood_heads:
            self._set("mood", "ready", ", ".join(self.mood_heads) +
                      (f"; skipped {', '.join(skipped)}" if skipped else ""))
        else:
            self._set("mood", "failed", "; ".join(skipped))

    @staticmethod
    def _mood_head(TensorflowPredict2D, name):
        """Loads a mood head and returns (head, class names)."""
        d = config.MODEL_DIR
        with open(os.path.join(d, f"{name}.json")) as f:
            meta = json.load(f)
        # Tensor names differ between Essentia releases; read them from the metadata
        schema = meta.get("schema", {})
        inp = schema.get("inputs", [{}])[0].get("name", "model/Placeholder")
        out = next((o["name"] for o in schema.get("outputs", [])
                    if o.get("output_purpose") == "predictions"), "model/Softmax")
        head = TensorflowPredict2D(graphFilename=os.path.join(d, f"{name}.pb"), input=inp, output=out)
        return head, meta["classes"]

    def _load_whisper(self):
        self._set("whisper", "loading", f"{config.WHISPER_SIZE}, downloads on first run")
        try:
            from faster_whisper import WhisperModel
            compute = "float16" if self.device == "cuda" else "int8"
            self.whisper = WhisperModel(config.WHISPER_SIZE, device=self.device, compute_type=compute)
            self._set("whisper", "ready", f"{config.WHISPER_SIZE} on {self.device}")
        except Exception as e:
            self._set("whisper", "failed", f"lyric themes unavailable: {e}")


MODELS = Models()
