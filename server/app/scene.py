"""Scenes as gradual, revisable states instead of a new prompt every window.

The first audible window sets the scene. After that each window is compared
with the features the scene was written for, and the scene is:

  hold       little change: the same prompt is re-sent (sending nothing would
             leave the stream without a prompt and the image goes black)
  evolve     moderate change: the LLM revises the current prompt, keeping its
             main scene and changing a few phrases
  blend      large change, or a gap of near silence: the LLM writes a halfway
             prompt between the current scene and the new audio, and the
             change is marked pending
  new scene  the next window confirms the change: a fresh scene is written.
             After a gap (a likely track change) a moderate change is enough.
  revert     the next window doesn't confirm it: back to the scene from before
             the blend

Time also moves the scene on, so steady music doesn't freeze it: a scene that
hasn't changed for SCENE_DRIFT_SEC evolves anyway ("drift"), and one that has
lasted SCENE_MAX_SEC blends and is then replaced by a scene with different
subjects ("timed"). Times are audio seconds, so an uploaded file behaves like
live input.

A new uploaded file is a known track boundary: its first window blends from
the previous scene and its second window always starts a new scene.

The prompt can also be set by hand (POST /api/scene). Unlocked, the pipeline
keeps it and evolves from it when the music changes. Locked, every window
re-sends it unchanged.

There is one tracker for the server, like the one stream in control.py.
"""
import logging
import math
import threading
import time

import librosa
import numpy as np

from . import config
from .prompt import blend_visual_prompt, evolve_visual_prompt, generate_visual_prompt

log = logging.getLogger("pipeline.scene")

PHASE_ORDER = ["breakdown", "groove", "build", "drop"]
WEIGHTS = {"bpm": 0.25, "genre": 0.25, "mood": 0.2, "energy": 0.2, "band": 0.1}
REF_KEYS = ("bpm", "energy", "phase", "band", "primary_genre", "secondary_genre", "mood_scores")


# ---------------------------------------------------------------- change score
def _bpm_change(a, b):
    if a <= 0 or b <= 0:
        return None
    d = abs(math.log2(b / a))
    d = min(d, abs(d - 1))              # half or double time counts as the same tempo
    return min(1.0, (2 ** d - 1) / 0.10)  # a 10% tempo change scores 1


def _genre_change(a, b):
    ga, gb = a.get("primary_genre"), b.get("primary_genre")
    if not ga or not gb or "unavailable" in (ga, gb):
        return None
    if ga == gb:
        return 0.0
    if gb == a.get("secondary_genre") or ga == b.get("secondary_genre"):
        return 0.4                      # top two swapped places
    if ga.split("---")[0] == gb.split("---")[0]:
        return 0.6                      # same parent genre
    return 1.0


def _mood_change(a, b):
    ma, mb = a.get("mood_scores") or {}, b.get("mood_scores") or {}
    keys = ma.keys() & mb.keys()
    if not keys:
        return None
    return min(1.0, sum(abs(ma[k] - mb[k]) for k in keys) / len(keys) / 0.25)


def _energy_change(a, b):
    ea, eb = a["energy"], b["energy"]
    rel = abs(eb - ea) / max(ea, eb, 0.05)
    jump = abs(PHASE_ORDER.index(a["phase"]) - PHASE_ORDER.index(b["phase"]))
    return max(min(1.0, rel / 0.6), 1.0 if jump >= 2 else 0.0)


def change_score(ref, rec):
    """0 = same music as the scene was written for, 1 = different in every way.

    Parts that a missing model can't provide (lite mode) are left out and the
    remaining weights are scaled up.
    """
    parts = {
        "bpm": _bpm_change(ref["bpm"], rec["bpm"]),
        "genre": _genre_change(ref, rec),
        "mood": _mood_change(ref, rec),
        "energy": _energy_change(ref, rec),
        "band": 0.0 if ref["band"] == rec["band"] else 1.0,
    }
    parts = {k: round(v, 2) for k, v in parts.items() if v is not None}
    total = sum(WEIGHTS[k] for k in parts)
    score = sum(WEIGHTS[k] * v for k, v in parts.items()) / total
    return round(score, 3), parts


def find_gap(y, sr=config.SR):
    """(start, end) in seconds of the longest near-silent stretch, if it lasts
    at least SCENE_GAP_SEC. Near silent means under a tenth of the window's
    loud level, so it works for quiet microphone input too."""
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    if not len(rms):
        return None
    loud = float(np.percentile(rms, 90))
    if loud < config.SCENE_SILENCE_RMS:
        return None                     # the whole window is quiet; handled as silence
    quiet = np.append(rms < 0.1 * loud, False)
    best, start = (0, 0), None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if (best[1] - best[0]) * hop / sr < config.SCENE_GAP_SEC:
        return None
    return best[0] * hop / sr, best[1] * hop / sr


# ---------------------------------------------------------------- tracker
LLM_STEPS = {"initial", "evolve", "blend", "new_scene", "new_track"}


class SceneTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.track = 0
        self.scene_id = 0
        self._clear()

    def _clear(self):
        self.prompt = None
        self.ref = None             # features the current scene was written for
        self.pending = None         # {"before": prompt, "track": bool, "forced": bool} while a change is unconfirmed
        self.locked = False
        self.manual = False
        self.decision = None
        self.updated = None
        self.last_gap_end = -1.0
        self.scene_t = None         # audio time the current scene started (None: from the next window)
        self.changed_t = None       # audio time the prompt last changed

    def describe(self):
        return {"prompt": self.prompt, "scene": self.scene_id, "track": self.track,
                "locked": self.locked, "manual": self.manual, "pending": self.pending is not None,
                "decision": self.decision, "updated": self.updated,
                "thresholds": {"evolve": config.SCENE_EVOLVE_AT, "change": config.SCENE_CHANGE_AT}}

    def current_prompt(self):
        with self.lock:
            return self.prompt

    def start_track(self, source):
        """A known track boundary: a new upload, or the start of a live session."""
        with self.lock:
            self.track += 1
            self.ref = None
            self.last_gap_end = -1.0
            self.pending = None
            self.scene_t = self.changed_t = None     # audio time restarts with each track
            if self.prompt and not self.locked:
                # First window blends from the previous scene, second one replaces it
                self.pending = {"before": self.prompt, "track": True, "forced": True, "boundary": True}
            log.info("track %d: %s", self.track, source)

    def set_manual(self, prompt=None, locked=None):
        with self.lock:
            if prompt is not None:
                if prompt != self.prompt:
                    self.scene_id += 1
                self.prompt = prompt
                self.manual = True
                self.pending = None
                self.ref = None             # the next window becomes its reference
                self.scene_t = self.changed_t = None
                self.decision = "manual"
                self.updated = time.time()
            if locked is not None:
                if self.locked and not locked:
                    self.ref = None         # resume from the music playing now, not from before the lock
                self.locked = bool(locked) and self.prompt is not None
            return self.describe()

    def step(self, rec, gap, use_llm):
        """Decides what this window sends, fills in the prompt fields of `rec`
        and returns the prompt (None = send nothing)."""
        with self.lock:
            if self.prompt and self.scene_t is None:
                self.scene_t = self.changed_t = rec["t_end"]
            score, parts = change_score(self.ref, rec) if self.ref else (None, {})
            gap_at = None
            if gap and rec["t_start"] + gap[0] >= self.last_gap_end:   # an overlapping window sees the same gap again
                self.last_gap_end = rec["t_start"] + gap[1]
                gap_at = round(rec["t_start"] + gap[0], 2)

            decision, note = self._decide(rec, score, gap_at)
            if decision in LLM_STEPS and not use_llm:
                decision, note = ("hold", "Prompt generation is off, so the scene is held.") if self.prompt \
                    else ("wait", "Prompt generation is off and there's no scene yet.")
                self.pending = None

            llm = {"prompt": None, "source": None, "ms": 0, "error": None}
            if decision == "initial":
                llm = generate_visual_prompt(rec["llm_input"])
            elif decision in ("new_scene", "new_track"):
                llm = generate_visual_prompt(rec["llm_input"], previous=self.prompt)
            elif decision == "evolve":
                llm = evolve_visual_prompt(self.prompt, rec["llm_input"])
            elif decision == "blend":
                llm = blend_visual_prompt(self.prompt, rec["llm_input"])

            if decision in LLM_STEPS and llm["source"] != "llm" and self.prompt:
                # The LLM failed: keep what's on screen rather than jump to the fallback prompt
                note = f"{llm['error']} Holding the current scene instead."
                decision, llm = "hold", {**llm, "prompt": None, "source": None}
                self.pending = None
            self._commit(decision, rec, llm["prompt"])

            source = {"hold": "held", "revert": "held", "locked": "manual",
                      "wait": "off" if not use_llm else "waiting"}.get(decision, llm["source"])
            rec.update({"prompt": self.prompt if decision != "wait" else None, "prompt_source": source,
                        "llm_ms": llm["ms"], "llm_error": llm["error"]})
            rec["scene"] = {"decision": decision, "note": note, "score": score, "parts": parts,
                            "gap_at": gap_at, **self.describe()}
            return rec["prompt"]

    def _decide(self, rec, score, gap_at):
        evolve_at, change_at = config.SCENE_EVOLVE_AT, config.SCENE_CHANGE_AT
        if self.locked:
            return "locked", "Your prompt is locked and re-sent every window."
        if rec["energy"] < config.SCENE_SILENCE_RMS:
            if not self.prompt:
                return "wait", "Near silence. Waiting for audio before choosing the first scene."
            if not self.pending:
                self.pending = {"before": self.prompt, "track": True, "forced": False}
            return "hold", "Near silence, so a track may be changing. Holding the scene."
        if not self.prompt:
            return "initial", "First audible window, so it sets the scene."

        p = self.pending
        if p:
            self.pending = None
            if p.get("boundary"):
                if p.get("blended"):
                    return "new_track", "New track. The scene that blended in is now replaced with one written for it."
                p["blended"] = True
                self.pending = p
                return "blend", "New track. Blending from the previous track's scene before replacing it."
            threshold = evolve_at if p["track"] else change_at
            if p["forced"] or score is None or score >= threshold:
                if p["track"]:
                    return "new_track", f"Change {score} after a gap confirms a new track (threshold {threshold})."
                return "new_scene", f"Change {score} held for a second window, so the scene is replaced."
            if self.prompt != p["before"]:
                self.prompt = p["before"]
                return "revert", f"Change fell to {score}, under {threshold}. Back to the scene from before the blend."
            return "hold", f"Change {score} after the gap is under {threshold}. Same track, scene held."

        if score is None:
            # A prompt set by hand has no features yet; this window becomes its reference
            self.ref = {k: rec[k] for k in REF_KEYS}
            return "hold", "This window becomes the reference for your prompt. Later windows are compared with it."
        if gap_at is not None:
            self.pending = {"before": self.prompt, "track": True, "forced": False}
            return "blend", f"Gap of near silence at {gap_at:.1f} s. Blending toward the new audio until the next window confirms."
        if score >= change_at:
            self.pending = {"before": self.prompt, "track": False, "forced": False}
            return "blend", f"Change {score} is over {change_at}. Blending; the next window confirms or reverts."
        t = rec["t_end"]
        if config.SCENE_MAX_SEC > 0 and t - self.scene_t >= config.SCENE_MAX_SEC:
            self.pending = {"before": self.prompt, "track": False, "forced": True}
            return "blend", (f"The scene has run {t - self.scene_t:.0f} s (limit {config.SCENE_MAX_SEC:.0f} s). "
                             "Blending; the next window replaces it with new subjects.")
        if score >= evolve_at:
            return "evolve", f"Change {score} is between {evolve_at} and {change_at}, so the scene is adjusted."
        if config.SCENE_DRIFT_SEC > 0 and t - self.changed_t >= config.SCENE_DRIFT_SEC:
            return "evolve", (f"Change {score} is small, but the prompt hasn't changed for "
                              f"{t - self.changed_t:.0f} s, so the scene drifts a little.")
        return "hold", f"Change {score} is under {evolve_at}. Same prompt re-sent."

    def _commit(self, decision, rec, new_prompt):
        if new_prompt:
            self.prompt = new_prompt
            self.manual = False
            self.updated = time.time()
        if decision in ("initial", "new_scene", "new_track"):
            self.scene_id += 1
            self.scene_t = rec["t_end"]
        if new_prompt or decision == "revert":
            self.changed_t = rec["t_end"]
        if decision in ("initial", "new_scene", "new_track", "evolve") or (decision == "blend" and self.ref is None):
            self.ref = {k: rec[k] for k in REF_KEYS}
        self.decision = decision


SCENE = SceneTracker()
