"""Baseline vs refined pipeline on a folder of tracks.

    docker compose run --rm eval                       # tracks in eval/tracks
    docker compose run --rm eval eval/tracks --judge   # also score prompts with the LLM

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

JUDGE_PROMPT = (
    "You rate prompts for a real-time image stream that visualises music. "
    "Given a description of the music and a visual prompt, rate how well the "
    "prompt fits the music on a scale of 1 (no fit) to 5 (excellent fit), "
    "considering mood, energy, tempo and genre. Reply with the digit only."
)


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


def judge(music, visual):
    """1-5 from the LLM, or None if it isn't available or answers oddly."""
    if prompt._client is None or not visual:
        return None
    try:
        reply = prompt._client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": f"Music: {music}\nPrompt: {visual}"}],
            temperature=0,
            max_tokens=400,
            extra_body={"reasoning_effort": "low"},
        )
        m = re.search(r"[1-5]", reply.choices[0].message.content or "")
        return int(m.group()) if m else None
    except Exception as e:
        print(f"    judge failed: {e}")
        return None


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


def run_track(path, label, args):
    """Every window of one track through both pipelines. Returns the window rows."""
    y, sr = librosa.load(path, sr=config.SR, mono=True)
    window, step = int(args.window * sr), int(args.step * sr)
    scenes = SceneTracker()          # a fresh scene tracker for every track
    rows, prev_base, prev_ref = [], None, None

    for i, start in enumerate(window_starts(len(y), window, step)):
        end = min(start + window, len(y))
        rec = analyse_window(y[start:end], sr=sr)
        rec.update({"index": i, "t_start": round(start / sr, 2), "t_end": round(end / sr, 2)})
        base = baseline_prompt(rec)
        scenes.step(rec, find_gap(y[start:end]), use_llm=not args.no_llm)
        ref = rec["prompt"]
        decision = rec["scene"]["decision"]
        t = rec["timings"]

        row = {
            "track": path.name, "window": i, "t_start": rec["t_start"], "t_end": rec["t_end"],
            "energy": rec["energy"], "phase": rec["phase"], "band": rec["band"],
            "true_bpm": label.get("bpm"), "base_bpm": rec["librosa_bpm"], "ref_bpm": rec["bpm"],
            "ref_bpm_source": rec["bpm_source"],
            "true_genre": label.get("genre", ""), "ref_genre": rec["primary_genre"],
            "ref_mood": rec["mood"], "ref_themes": rec["lyric_themes"],
            "music_description": rec["llm_input"],
            "base_prompt": base, "ref_prompt": ref or "", "ref_prompt_source": rec["prompt_source"],
            "ref_decision": decision,
            "base_changed": prev_base is not None and base != prev_base,
            "ref_changed": prev_ref is not None and ref is not None and ref != prev_ref,
            "base_scene_start": prev_base is not None and base != prev_base,
            "ref_scene_start": decision in SCENE_STARTS,
            "base_ms": t.get("librosa", 0) + t.get("bands", 0),
            "ref_ms": rec["analysis_ms"] + rec["llm_ms"],
            "base_judge": None, "ref_judge": None,
        }
        if args.judge:
            row["base_judge"] = judge(rec["llm_input"], base)
            row["ref_judge"] = judge(rec["llm_input"], ref)
        rows.append(row)
        prev_base, prev_ref = base, (ref if ref is not None else prev_ref)
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


def summarise(rows, durations, labels, args):
    audible = [r for r in rows if r["energy"] >= config.SCENE_SILENCE_RMS]
    minutes = sum(durations.values()) / 60
    s = {}

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
    s["ref_genre_acc"] = pct(genre_match(r["true_genre"], r["ref_genre"]) for r in genre)         if any(r["ref_genre"] != "unavailable" for r in genre) else None
    s["genre_windows"] = len(genre)

    # Prompts and scenes
    for side in ("base", "ref"):
        s[f"{side}_changes_per_min"] = sum(r[f"{side}_changed"] for r in rows) / minutes if minutes else None
        s[f"{side}_scenes_per_min"] = sum(r[f"{side}_scene_start"] for r in rows) / minutes if minutes else None
        caught = total = spurious = 0
        for track, lab in labels.items():
            if lab["boundaries"]:
                c, t, sp = boundary_scores([r for r in rows if r["track"] == track], lab["boundaries"],
                                           f"{side}_scene_start", args.window, args.step)
                caught, total, spurious = caught + c, total + t, spurious + sp
        s[f"{side}_boundary_recall"] = 100 * caught / total if total else None
        s[f"{side}_boundary_spurious"] = spurious if total else None
        s["boundaries"] = total
        s[f"{side}_judge"] = mean(r[f"{side}_judge"] for r in rows)
        s[f"{side}_ms"] = mean(r[f"{side}_ms"] for r in rows)

    s["llm_prompts"] = sum(r["ref_prompt_source"] == "llm" for r in rows)
    return s


def write_markdown(path, s, meta):
    # Scene tracking only changes scenes when the LLM writes the new prompt
    no_llm = "needs the LLM; scenes are held without it" if not s["llm_prompts"] else ""
    rows = [
        ("Tempo accuracy, within 4%", fmt(s["base_tempo_acc1"], "%"), fmt(s["ref_tempo_acc1"], "%"),
         f"{s['tempo_windows']} labelled windows"),
        ("Tempo accuracy, allowing x2 / /2", fmt(s["base_tempo_acc2"], "%"), fmt(s["ref_tempo_acc2"], "%"), ""),
        ("Genre accuracy", fmt(s["base_genre_acc"], "%"), fmt(s["ref_genre_acc"], "%"),
         f"{s['genre_windows']} labelled windows; baseline always answers '{s['majority_genre']}'"
         if s["majority_genre"] else "needs genre labels"),
        ("Track changes caught", fmt(s["base_boundary_recall"], "%"), fmt(s["ref_boundary_recall"], "%"),
         no_llm or (f"{s['boundaries']} labelled boundaries" if s["boundaries"] else "needs boundary labels")),
        ("Scene changes away from a track change", fmt(s["base_boundary_spurious"], "", 0),
         fmt(s["ref_boundary_spurious"], "", 0), no_llm or "lower is better"),
        ("Scene changes per minute", fmt(s["base_scenes_per_min"], "", 2), fmt(s["ref_scenes_per_min"], "", 2), no_llm),
        ("Prompt changes per minute", fmt(s["base_changes_per_min"], "", 2), fmt(s["ref_changes_per_min"], "", 2),
         no_llm or "includes scene adjustments"),
        ("Prompt fit, LLM judge (1-5)", fmt(s["base_judge"], "", 2), fmt(s["ref_judge"], "", 2),
         "needs --judge and GROQ_API_KEY" if s["base_judge"] is None else ""),
        ("Time per window", fmt(s["base_ms"], " ms", 0), fmt(s["ref_ms"], " ms", 0), "analysis + LLM"),
    ]
    lines = [
        "# Baseline vs refined pipeline", "",
        f"Run {meta['time']}, {meta['tracks']} tracks, {meta['windows']} windows, "
        f"window {meta['window']} s, step {meta['step']} s.", "",
        f"Mode: {meta['mode']}. Models: {meta['models']}. "
        f"LLM: {meta['llm']} ({s['llm_prompts']} prompts written by it).", "",
        "| Metric | Baseline | Refined | Notes |", "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows], "",
        "Baseline: Librosa tempo, energy and band; majority-class genre; template prompt; no scene tracking.",
        "Refined: Madmom, Essentia and Whisper features; LLM prompt; scene tracking.", "",
        "The judge sees the refined pipeline's description of the music, and the refined prompt was "
        "written from that same description, so treat the judge score as a guide, not proof.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tracks", nargs="?", default=str(HERE / "tracks"), help="folder of audio files")
    ap.add_argument("--labels", help="ground truth CSV (default: labels.csv in the tracks folder)")
    ap.add_argument("--window", type=float, default=config.DEFAULT_WINDOW_SEC, help="window length, seconds")
    ap.add_argument("--step", type=float, default=config.DEFAULT_STEP_SEC, help="step between windows, seconds")
    ap.add_argument("--judge", action="store_true", help="score every prompt with the LLM (2 calls per window)")
    ap.add_argument("--no-llm", action="store_true", help="refined pipeline without the LLM")
    ap.add_argument("--out", default=str(HERE / "results"), help="where to write results")
    args = ap.parse_args()

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
    if args.judge and not prompt.llm_enabled():
        print("--judge needs GROQ_API_KEY; prompts won't be scored.")

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
            "llm": "off" if args.no_llm else (config.LLM_MODEL if prompt.llm_enabled() else "no key, fallback prompt")}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "metrics": s}, indent=2), encoding="utf-8")
    print("\n" + write_markdown(out / "summary.md", s, meta))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
