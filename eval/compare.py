"""Baseline vs refined pipeline on a folder of tracks.

    docker compose run --rm eval                       # tracks in eval/tracks
    docker compose run --rm eval eval/tracks/tagged    # another folder of tracks

Baseline, the simplest end-to-end pipeline:
  features  Librosa only: tempo, energy phase, loudest band
  genre     always the most common label in labels.csv (majority class)
  prompt    a fixed template filled in from phase and band, no LLM
  scenes    none; the prompt is rebuilt every window

Refined, the pipeline the server runs:
  features  Madmom tempo, Essentia genre and mood, Whisper lyric themes
  prompt    written by the LLM (needs GROQ_API_KEY)
  scenes    scene tracking: hold, adjust, blend, replace (scene.py)

Both run on the same windows of the same tracks in one pass. Nothing is sent
to the Control API. Results go to eval/results/<time>/: windows.csv (one row
per window), summary.md (the comparison tables) and summary.json.

summary.md has the same parts as the notebook's evaluation: a short summary,
1. run time (per window, per stage, per track), 2. what changes on screen
(prompt and scene changes and why), 3. accuracy (needs labels).

Ground truth is optional, in <tracks>/labels.csv:

    file,genre,bpm,boundaries
    song1.wav,rock,120,
    mix.mp3,,,95.5;201

genre and bpm score feature accuracy. boundaries (seconds, ;-separated) are
where one song ends and the next begins in a file of joined songs, and score
whether scene changes happen where the music changes.
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# The app package is at server/app in the repo and at /srv/app in the container
for base in (ROOT / "server", ROOT):
    if (base / "app").is_dir():
        sys.path.insert(0, str(base))
        break

import librosa  # noqa: E402

from app import config, prompt  # noqa: E402
from app.features import analyse_window  # noqa: E402
from app.jobs import window_starts  # noqa: E402
from app.models import MODELS  # noqa: E402
from app.scene import SceneTracker, find_gap  # noqa: E402

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

# Baseline prompt: one phrase per energy phase, one per loudest band
PHASE_PHRASES = {
    "breakdown": "dim empty space, soft blue haze, slow drift",
    "groove": "night city street, warm amber light, steady pan",
    "build": "tunnel of rising light, magenta glow, camera pushing forward",
    "drop": "neon arena, white strobes, fast camera shake",
}
BAND_PHRASES = {
    "sub": "deep pulsing shapes",
    "mid": "glowing synth ribbons",
    "high": "glittering sparks",
}

SCENE_STARTS = {"new_scene", "new_track"}
STAGES = ["librosa", "bands", "madmom", "essentia", "whisper"]   # analysis steps timed in analyse_window()
BASE_STAGES = ("librosa", "bands")                                 # all the baseline needs
STAGE_NAMES = {"bands": "Frequency bands", "llm": "LLM (prompt writing)"}
TEMPO_JUMP = 0.25   # a 25%+ tempo change between windows counts as a jump


def baseline_prompt(rec):
    return f"{PHASE_PHRASES[rec['phase']]}, {BAND_PHRASES[rec['band']]}"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def genre_match(label, predicted):
    """'metal' matches 'Rock---Heavy Metal'; 'hiphop' matches 'Hip Hop---Trap'."""
    return bool(label) and norm(label) in norm(predicted)


def tempo_hits(est, truth):
    """(acc1, acc2): within 4% of the true tempo; acc2 also accepts x2, x3, /2, /3."""
    if not est or not truth:
        return None, None
    acc1 = abs(est - truth) <= 0.04 * truth
    acc2 = any(abs(est - truth * f) <= 0.04 * truth * f for f in (1, 2, 3, 1 / 2, 1 / 3))
    return acc1, acc2


def tempo_jump(bpm, prev_bpm):
    return bool(bpm) and bool(prev_bpm) and abs(bpm / prev_bpm - 1) >= TEMPO_JUMP


def read_labels(path):
    if not path or not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = {}
        for r in csv.DictReader(f):
            bounds = [float(b) for b in (r.get("boundaries") or "").split(";") if b.strip()]
            rows[r["file"].strip()] = {
                "genre": (r.get("genre") or "").strip(),
                "bpm": float(r["bpm"]) if (r.get("bpm") or "").strip() else None,
                "boundaries": bounds,
            }
        return rows


def scene_cause(decision, prev_note):
    """Why a scene started: the track changed, the music changed, or the SCENE_MAX_SEC timer ran out."""
    if decision == "new_track":
        return "track"
    if decision == "new_scene":
        return "timer" if prev_note.startswith("The scene has run") else "music"
    return None


def run_track(path, label, args):
    """Every window of one track through both pipelines. Returns the window rows."""
    y, sr = librosa.load(path, sr=config.SR, mono=True)
    window, step = int(args.window * sr), int(args.step * sr)
    scenes = SceneTracker()          # a fresh scene tracker for every track
    rows, prev_base, prev_ref, prev_note = [], None, None, ""
    prev_base_bpm = prev_ref_bpm = None

    for i, start in enumerate(window_starts(len(y), window, step)):
        end = min(start + window, len(y))
        rec = analyse_window(y[start:end], sr=sr)
        rec.update({"index": i, "t_start": round(start / sr, 2), "t_end": round(end / sr, 2)})
        base = baseline_prompt(rec)
        scenes.step(rec, find_gap(y[start:end]), use_llm=not args.no_llm)
        ref = rec["prompt"]
        decision = rec["scene"]["decision"]
        t = rec["timings"]
        llm_ms = rec.get("llm_ms") or 0

        row = {
            "track": path.name, "window": i, "t_start": rec["t_start"], "t_end": rec["t_end"],
            "energy": rec["energy"], "phase": rec["phase"], "band": rec["band"],
            "true_bpm": label.get("bpm"), "base_bpm": rec["librosa_bpm"], "ref_bpm": rec["bpm"],
            "ref_bpm_source": rec["bpm_source"],
            "true_genre": label.get("genre", ""), "ref_genre": rec["primary_genre"],
            "ref_mood": rec["mood"], "ref_themes": rec["lyric_themes"],
            "music_description": rec["llm_input"],
            "base_prompt": base, "ref_prompt": ref or "", "ref_prompt_source": rec["prompt_source"],
            "ref_decision": decision, "ref_scene_cause": scene_cause(decision, prev_note),
            "ref_note": rec["scene"]["note"],
            "base_changed": prev_base is not None and base != prev_base,
            "ref_changed": prev_ref is not None and ref is not None and ref != prev_ref,
            "base_scene_start": prev_base is not None and base != prev_base,
            "ref_scene_start": decision in SCENE_STARTS,
            "base_tempo_jump": tempo_jump(rec["librosa_bpm"], prev_base_bpm),
            "ref_tempo_jump": tempo_jump(rec["bpm"], prev_ref_bpm),
            **{f"{k}_ms": t.get(k, 0) for k in STAGES},
            "llm_ms": llm_ms,
            "llm_error": rec.get("llm_error") or "",
            "base_ms": sum(t.get(k, 0) for k in BASE_STAGES),
            "ref_ms": rec["analysis_ms"] + llm_ms,
        }
        rows.append(row)
        prev_base, prev_ref = base, (ref if ref is not None else prev_ref)
        prev_base_bpm, prev_ref_bpm, prev_note = rec["librosa_bpm"], rec["bpm"], rec["scene"]["note"]
        print(f"  window {i + 1}: {decision:<9} {rec['bpm']:>6} BPM  {rec['primary_genre']}")
    return rows, len(y) / sr


def boundary_scores(rows, boundaries, starts_key, window, step):
    """(boundaries caught, total, scene changes not near a boundary).

    A boundary counts as caught if a scene starts in a window ending between
    the boundary and one window plus one step after it, because scene tracking
    blends first and confirms on the next window.
    """
    starts = [r["t_end"] for r in rows if r[starts_key]]
    near = lambda t, b: b <= t <= b + window + step  # noqa: E731
    caught = sum(any(near(t, b) for t in starts) for b in boundaries)
    spurious = sum(not any(near(t, b) for b in boundaries) for t in starts)
    return caught, len(boundaries), spurious


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def pct(xs):
    xs = [x for x in xs if x is not None]
    return 100 * sum(xs) / len(xs) if xs else None


def fmt(v, unit="", digits=1):
    return "n/a" if v is None else f"{v:.{digits}f}{unit}"


def mmss(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def short(name, n=28):
    name = os.path.splitext(name)[0]
    return name if len(name) <= n else name[:n - 1] + "…"


def summarise(rows, durations, labels, args):
    audible = [r for r in rows if r["energy"] >= config.SCENE_SILENCE_RMS]
    minutes = sum(durations.values()) / 60
    step_ms = args.step * 1000
    s = {"audio_sec": sum(durations.values()), "step_ms": step_ms}

    # Run time
    for side in ("base", "ref"):
        ms = [r[f"{side}_ms"] for r in rows]
        s[f"{side}_total_sec"] = sum(ms) / 1000
        s[f"{side}_ms"] = mean(ms)
        s[f"{side}_ms_median"] = statistics.median(ms)
        s[f"{side}_ms_max"] = max(ms)
        s[f"{side}_on_time"] = pct(m <= step_ms for m in ms)

    # Where the time goes, per stage of the refined pipeline
    all_ms = sum(r["ref_ms"] for r in rows) or 1
    s["stages"] = []
    for k in STAGES + ["llm"]:
        col = [r[f"{k}_ms"] for r in rows]
        used = [m for m in col if m > 0]
        if used:
            s["stages"].append({"stage": STAGE_NAMES.get(k, k.capitalize()), "median_ms": statistics.median(used),
                                "max_ms": max(col), "total_sec": sum(col) / 1000,
                                "share": 100 * sum(col) / all_ms, "baseline_uses": k in BASE_STAGES})

    # Per track
    s["tracks"] = []
    for track, sec in durations.items():
        g = [r for r in rows if r["track"] == track]
        s["tracks"].append({"track": track, "length_sec": sec, "windows": len(g),
                            "base_sec": sum(r["base_ms"] for r in g) / 1000,
                            "ref_sec": sum(r["ref_ms"] for r in g) / 1000,
                            "ref_max_ms": max(r["ref_ms"] for r in g),
                            "ref_on_time": pct(r["ref_ms"] <= step_ms for r in g)})

    # What changes on screen
    causes = Counter(r["ref_scene_cause"] for r in rows if r["ref_scene_cause"])
    s["scene_causes"] = {k: causes.get(k, 0) for k in ("track", "music", "timer")}
    s["base_prompts"] = len({r["base_prompt"] for r in rows})
    s["ref_prompts"] = len({r["ref_prompt"] for r in rows if r["ref_prompt"]})
    s["base_held"] = pct(not r["base_changed"] for r in rows)
    s["ref_held"] = pct(r["ref_prompt_source"] == "held" for r in rows)
    s["llm_errors"] = [r["llm_error"] for r in rows if r["llm_error"]]
    for side in ("base", "ref"):
        s[f"{side}_changes"] = sum(r[f"{side}_changed"] for r in rows)
        s[f"{side}_scenes"] = sum(r[f"{side}_scene_start"] for r in rows)
        s[f"{side}_changes_per_min"] = s[f"{side}_changes"] / minutes if minutes else None
        s[f"{side}_scenes_per_min"] = s[f"{side}_scenes"] / minutes if minutes else None
        s[f"{side}_tempo_jumps"] = sum(r[f"{side}_tempo_jump"] for r in rows)

    # Tempo
    tempo = [r for r in audible if r["true_bpm"]]
    for side in ("base", "ref"):
        hits = [tempo_hits(r[f"{side}_bpm"], r["true_bpm"]) for r in tempo]
        s[f"{side}_tempo_acc1"] = pct(h[0] for h in hits)
        s[f"{side}_tempo_acc2"] = pct(h[1] for h in hits)
    s["tempo_windows"] = len(tempo)

    # Genre: the baseline always answers the most common label
    genre = [r for r in audible if r["true_genre"]]
    track_genres = [lab["genre"] for lab in labels.values() if lab["genre"]]
    majority = Counter(norm(g) for g in track_genres).most_common(1)[0][0] if track_genres else None
    s["majority_genre"] = majority
    s["base_genre_acc"] = pct(norm(r["true_genre"]) == majority for r in genre) if majority else None
    s["ref_genre_acc"] = pct(genre_match(r["true_genre"], r["ref_genre"]) for r in genre) \
        if any(r["ref_genre"] != "unavailable" for r in genre) else None
    s["genre_windows"] = len(genre)

    # Song changes in files of joined songs
    for side in ("base", "ref"):
        caught = total = spurious = 0
        for track, lab in labels.items():
            if lab["boundaries"]:
                c, t, sp = boundary_scores([r for r in rows if r["track"] == track], lab["boundaries"],
                                           f"{side}_scene_start", args.window, args.step)
                caught, total, spurious = caught + c, total + t, spurious + sp
        s[f"{side}_boundary_recall"] = 100 * caught / total if total else None
        s[f"{side}_boundary_spurious"] = spurious if total else None
        s["boundaries"] = total

    s["llm_prompts"] = sum(r["ref_prompt_source"] == "llm" for r in rows)
    return s


def table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *["| " + " | ".join(str(x) for x in r) + " |" for r in body]]


def write_markdown(path, s, meta):
    step = meta["step"]
    minutes = s["audio_sec"] / 60
    # Scene tracking only changes scenes when the LLM writes the new prompt
    no_llm = "needs the LLM; scenes are held without it" if not s["llm_prompts"] else ""
    c = s["scene_causes"]

    # In short
    on_time = s["ref_on_time"]
    keeps_up = ("keeps up with live audio" if on_time == 100 else
                "just about keeps up with live audio" if on_time >= 90 else "can't keep up with live audio")
    slowest = max(s["stages"], key=lambda x: x["total_sec"])["stage"] if s["stages"] else "n/a"
    summary = [
        f"- **Speed:** the refined pipeline needed **{mmss(s['ref_total_sec'])}** to process "
        f"**{mmss(s['audio_sec'])}** of audio (baseline: {s['base_total_sec']:.1f} s). {on_time:.0f}% of its "
        f"windows finished inside the {step:g} s step, so it {keeps_up}. Slowest step: **{slowest}**.",
        f"- **Variety:** it asked for **{s['ref_prompts']}** different pictures; "
        f"the baseline only ever has {s['base_prompts']}.",
        f"- **Scene changes:** {s['ref_scenes']} in total: {c['track']} at track changes, "
        f"{c['music']} because the music changed, {c['timer']} because a scene hit the "
        f"{config.SCENE_MAX_SEC:g} s limit." + (f" ({no_llm})" if no_llm else ""),
        f"- **Tempo stability:** the tempo jumped {s['ref_tempo_jumps']} times "
        f"(baseline {s['base_tempo_jumps']}). Jumps usually mean the detector halved or doubled the beat.",
    ]
    if s["llm_errors"]:
        summary.append(f"- ⚠️ **{len(s['llm_errors'])} LLM calls failed** (scene held instead): `{s['llm_errors'][0][:100]}`")

    # 1. Run time
    run_rows = [
        ("Audio processed", mmss(s["audio_sec"]), mmss(s["audio_sec"]),
         f"{meta['tracks']} tracks, {meta['windows']} windows"),
        ("Total processing time",
         mmss(s["base_total_sec"]) if s["base_total_sec"] >= 60 else f"{s['base_total_sec']:.1f} s",
         mmss(s["ref_total_sec"]), "all windows added up (analysis + LLM)"),
        ("Time per window, typical", fmt(s["base_ms_median"], " ms", 0), fmt(s["ref_ms_median"], " ms", 0), "median"),
        ("Time per window, average", fmt(s["base_ms"], " ms", 0), fmt(s["ref_ms"], " ms", 0), ""),
        ("Time per window, slowest", fmt(s["base_ms_max"], " ms", 0), fmt(s["ref_ms_max"], " ms", 0), ""),
        (f"Windows finished within {step:g} s", fmt(s["base_on_time"], "%", 0), fmt(on_time, "%", 0),
         f"a new window arrives every {step:g} s; under 100% means it falls behind live audio"),
        ("Wall-clock time of the whole run", "", mmss(meta["wall_sec"]),
         "includes loading models and audio, both pipelines"),
    ]
    stage_rows = [(x["stage"], f"{x['median_ms']:.0f} ms", f"{x['max_ms']:.0f} ms", f"{x['total_sec']:.1f} s",
                   f"{x['share']:.0f}%", "yes" if x["baseline_uses"] else "no") for x in s["stages"]]
    track_rows = [(short(x["track"]), mmss(x["length_sec"]), x["windows"], f"{x['base_sec']:.1f} s",
                   mmss(x["ref_sec"]), f"{x['ref_max_ms']:.0f} ms", fmt(x["ref_on_time"], "%", 0))
                  for x in s["tracks"]]

    # 2. What changes on screen
    per_min = lambda n: f"{n / minutes:.2f}" if minutes else "n/a"  # noqa: E731
    behaviour_rows = [
        ("Different prompts used", s["base_prompts"], s["ref_prompts"], "how many distinct pictures were asked for"),
        ("Prompt changes per minute", per_min(s["base_changes"]), per_min(s["ref_changes"]),
         no_llm or "any change to the text sent, big or small"),
        ("Scene changes per minute", per_min(s["base_scenes"]), per_min(s["ref_scenes"]),
         no_llm or "a completely new picture"),
        ("… because the track changed", "–", c["track"], "baseline can't tell"),
        ("… because the music changed", s["base_scenes"], c["music"], "baseline: phase or band flipped"),
        ("… because of the timer", "–", c["timer"], f"scene older than SCENE_MAX_SEC ({config.SCENE_MAX_SEC:g} s)"),
        ("Windows that kept the same prompt", fmt(s["base_held"], "%", 0), fmt(s["ref_held"], "%", 0),
         "higher = calmer screen"),
        ("Tempo jumps", s["base_tempo_jumps"], s["ref_tempo_jumps"],
         f"tempo moved {TEMPO_JUMP:.0%}+ from the last window; usually a half/double-time error, lower is better"),
    ]
    if s["llm_errors"]:
        behaviour_rows.append(("LLM failures (scene held instead)", "–", len(s["llm_errors"]), s["llm_errors"][0][:80]))

    # 3. Accuracy
    acc_rows = [
        ("Tempo accuracy, within 4%", fmt(s["base_tempo_acc1"], "%"), fmt(s["ref_tempo_acc1"], "%"),
         f"{s['tempo_windows']} labelled windows" if s["tempo_windows"] else "needs bpm labels"),
        ("Tempo accuracy, allowing x2 / /2", fmt(s["base_tempo_acc2"], "%"), fmt(s["ref_tempo_acc2"], "%"), ""),
        ("Genre accuracy", fmt(s["base_genre_acc"], "%"), fmt(s["ref_genre_acc"], "%"),
         f"{s['genre_windows']} labelled windows; baseline always answers '{s['majority_genre']}'"
         if s["majority_genre"] else "needs genre labels"),
        ("Track changes caught", fmt(s["base_boundary_recall"], "%"), fmt(s["ref_boundary_recall"], "%"),
         no_llm or (f"{s['boundaries']} labelled boundaries" if s["boundaries"] else "needs boundary labels")),
        ("Scene changes away from a track change", fmt(s["base_boundary_spurious"], "", 0),
         fmt(s["ref_boundary_spurious"], "", 0), no_llm or "lower is better"),
    ]

    lines = [
        "# Baseline vs refined pipeline", "",
        f"Run {meta['time']}, {meta['tracks']} tracks, {meta['windows']} windows, "
        f"window {meta['window']:g} s, step {step:g} s.", "",
        f"Mode: {meta['mode']}. Models: {meta['models']}. "
        f"LLM: {meta['llm']} ({s['llm_prompts']} prompts written by it).", "",
        "**In short**", "", *summary, "",
        "## 1. Run time", "",
        *table(["", "Baseline", "Refined", "Notes"], run_rows), "",
        "**Where the time goes (refined pipeline, per window)**", "",
        *table(["Step", "Typical", "Slowest", "Total", "Share", "Baseline uses it"], stage_rows), "",
        "**Per track**", "",
        *table(["Track", "Length", "Windows", "Baseline", "Refined", "Slowest window", f"Within {step:g} s"],
               track_rows), "",
        "## 2. What changes on screen", "",
        *table(["", "Baseline", "Refined", "What it means"], behaviour_rows), "",
        "## 3. Accuracy", "",
        *table(["Metric", "Baseline", "Refined", "Notes"], acc_rows), "",
        "Baseline: Librosa tempo, energy and band; majority-class genre; template prompt; no scene tracking.",
        "Refined: Madmom, Essentia and Whisper features; LLM prompt; scene tracking.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracks", nargs="?", default=str(HERE / "tracks"), help="folder of audio files")
    ap.add_argument("--labels", help="ground truth CSV (default: labels.csv in the tracks folder)")
    ap.add_argument("--window", type=float, default=config.DEFAULT_WINDOW_SEC, help="window length, seconds")
    ap.add_argument("--step", type=float, default=config.DEFAULT_STEP_SEC, help="step between windows, seconds")
    ap.add_argument("--no-llm", action="store_true", help="refined pipeline without the LLM")
    ap.add_argument("--out", default=str(HERE / "results"), help="where to write results")
    args = ap.parse_args()
    t0 = time.perf_counter()

    tracks_dir = Path(args.tracks)
    files = sorted(p for p in tracks_dir.glob("*") if p.suffix.lower() in AUDIO)
    if not files:
        sys.exit(f"No audio files in {tracks_dir}. Put .wav, .mp3 or .flac files there.")
    labels = read_labels(Path(args.labels) if args.labels else tracks_dir / "labels.csv")
    unknown = set(labels) - {p.name for p in files}
    if unknown:
        print(f"labels.csv lists files that aren't in {tracks_dir}: {', '.join(sorted(unknown))}")
    if not prompt.llm_enabled() or args.no_llm:
        print("Warning: no LLM, so the refined pipeline holds one scene per track and its "
              "scene results mean little. Set GROQ_API_KEY in .env for a fair comparison.")

    print(f"Loading models ({config.PIPELINE_MODE} mode)...")
    MODELS.load()
    models = ", ".join(f"{k} {v['state']}" for k, v in MODELS.status.items())
    print(models)

    rows, durations = [], {}
    for n, path in enumerate(files, 1):
        print(f"[{n}/{len(files)}] {path.name}")
        try:
            track_rows, durations[path.name] = run_track(path, labels.get(path.name, {}), args)
            rows += track_rows
        except Exception as e:
            print(f"  skipped: {e}")
    if not rows:
        sys.exit("No track could be analysed.")

    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "windows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    s = summarise(rows, durations, {k: v for k, v in labels.items() if k in durations}, args)
    meta = {"time": time.strftime("%Y-%m-%d %H:%M"), "tracks": len(durations), "windows": len(rows),
            "window": args.window, "step": args.step, "mode": config.PIPELINE_MODE, "models": models,
            "llm": "off" if args.no_llm else (config.LLM_MODEL if prompt.llm_enabled() else "no key, fallback prompt"),
            "wall_sec": time.perf_counter() - t0}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "metrics": s}, indent=2), encoding="utf-8")
    print("\n" + write_markdown(out / "summary.md", s, meta))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
