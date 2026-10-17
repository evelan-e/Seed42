# seed42 pipeline test

A local app for testing the audio → prompt pipeline from `AudioAnalyser_clean.ipynb`
outside Colab. Drop in a track or listen through the mic and see, for every window:
tempo, energy phase, loudest band, genre, mood, lyric themes, the sentence sent to
the LLM, the prompt it wrote, and what the Control API answered.

It runs against the notebook's mock Control API by default, so nothing touches a
real stream.

## Run it

Pick one of three ways.

**Docker (Linux, Windows, Intel Macs)**

```bash
cp .env.example .env          # add your GROQ_API_KEY
docker compose up --build
```

Open http://localhost:8000. The first build takes several minutes (Madmom compiles),
and the first start downloads the Essentia and Whisper models into a volume, so later
starts are fast.

On Apple Silicon this works but runs under x86 emulation, because essentia-tensorflow
has no Linux arm64 build. Use the next option instead.

**Without Docker (recommended on Apple Silicon Macs)**

Needs Python 3.12, the Xcode command line tools (`xcode-select --install`) and
ffmpeg (`brew install ffmpeg`).

```bash
cp .env.example .env          # add your GROQ_API_KEY
scripts/run-local.sh
```

This creates `server/.venv`, installs the pinned packages once, and starts the mock
and the server together. Ctrl-C stops both. If `python3.12` isn't on your PATH:
`PYTHON=/path/to/python3.12 scripts/run-local.sh`.

**Lite mode (any machine, under a minute)**

```bash
scripts/run-local.sh --lite
```

Librosa features, the LLM and the Control API only. Genre, mood and themes show as
unavailable. Use it for work on the page or the Control API path.

## Using the page

- **Choose track** or drop a file anywhere on the page. Windows appear as they finish.
- **Listen live** streams the mic to the server. Speech processing (echo cancellation,
  noise suppression, auto gain) is switched off so music reaches the pipeline unchanged.
- **Window** and **Step** set the analysis window and how far it moves. 20/10 is the
  default; the notebook's one-minute engine is 60/60.
- **Generate prompt** and **Send to stream** let you switch off the LLM or Control API
  step for a run.
- The energy strip shows the track's loudness against the phase thresholds
  (0.15, 0.40, 0.65). The lane under it has one segment per window, shaded by phase.
  Click a segment to jump to its row and seek the audio. A red mark means the Control
  API call failed; a red dash in live mode means a window was skipped.
- `j` / `k` move between windows.
- The pills next to the run show which models loaded, whether the LLM is on, and the
  current stream. Click them for details. **New stream** ends the stream so the next
  window creates a fresh one.

## Scenes

The prompt is a scene that lasts, not a new image every window
(`server/app/scene.py`). The first audible window sets it. Each later window gets a
change score against the music the scene was written for (tempo, genre, mood,
energy, loudest band), and the scene is:

- **held** under `SCENE_EVOLVE_AT` (0.25): the same prompt is re-sent, so the stream
  is never left without one
- **adjusted** from 0.25: the LLM revises the current prompt, changing a few phrases
- **blended** from `SCENE_CHANGE_AT` (0.5), or after a gap of near silence: a halfway
  prompt, then **replaced** if the next window confirms the change, or **reverted** if
  it doesn't

Each uploaded file is a new track: its first window blends from the previous scene,
its second starts a new one. In live mode a silence gap (`SCENE_GAP_SEC`) or a
sustained large change marks a new track. Each row says what the scene did and why,
and ticks above the lane mark where scenes start.

**Current scene** shows the prompt on the stream. Edit it, or press **Edit** on any
row to start from that prompt, then **Send to stream** (Ctrl+Enter). Unlocked, the
pipeline keeps your prompt and adjusts from it as the music changes. **Lock scene**
re-sends it every window until you unlock.

## Testing failure handling

The mock is deliberately strict. Change its flags in `docker-compose.yml`, or run it
yourself with `python mock/mock-control-api.py --help`:

- `--fail-rate 0.2` returns 502 on a fifth of calls
- `--unstable` ends streams after 45 s and makes DELETE fail
- `--not-ready 409` uses the other legal not-ready code
- `--async-create` answers 202 on create and requires polling

The mock also prints a warning if a prompt update carries a creation-time field, which
on the real API would reload the pipeline and stop the show for about 30 seconds.

## Live mode timing

If a window is still being analysed when the next one is due, the next one is skipped
and marked on the page. Each row shows how long analysis took; when that's longer than
the step, the row says so. On CPU, expect a few seconds per 20 s window with every
model loaded, so a 5 s step will skip windows. That measurement is what you need
before settling window and step for real live use.

## Before pointing at the real Control API

The notebook and the mock send prompts with `PATCH /api/streams`. The Control API
PDF (20 Aug 2026) documents `POST /api/prompts` with `{stream_id, body}`. Confirm
which one the real API serves and update `send_prompt()` in `server/app/control.py`,
then set `CONTROL_API_URL`, `CONTROL_API_EMAIL` and `CONTROL_API_PASSWORD` in `.env`.

## Layout

```
mock/mock-control-api.py     the notebook's mock, plus --host for Docker
server/app/
  config.py                  settings from .env (replaces the Colab config cell and secret)
  models.py                  loads Madmom, Essentia and Whisper once, with per-model status
  features.py                extract_fast_features and helpers, unchanged from the notebook
  prompt.py                  Groq call and trim_prompt
  control.py                 Seed42Client, reporting status codes instead of raising
  jobs.py                    the per-window pipeline and the background worker for files
  scene.py                   scene state: hold, adjust, blend, replace; track changes
  live.py                    the /api/live WebSocket
  main.py                    FastAPI routes
  static/                    the test page
server/requirements*.txt     pinned and tested together on Python 3.12
scripts/run-local.sh         no-Docker launcher
legacy/streamlit/            the earlier Streamlit analyser, kept for reference
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Upload a track (multipart `file`, plus `window_sec`, `step_sec`, `use_llm`, `send_to_stream`). Returns `{id}`. |
| `GET /api/analyze/{id}?since=N` | Status and results; `since` returns only windows after the first N. |
| `DELETE /api/analyze/{id}` | Stop a job. |
| `WS /api/live` | Send `{"type":"start", ...}`, then Float32 mono PCM at 22050 Hz. |
| `GET /api/health` | Model, LLM and Control API status. |
| `POST /api/stream/reset` | End the current stream. |
| `GET /api/scene` | The current scene. |
| `POST /api/scene` | `{prompt?, locked?, send}`: set the prompt by hand, lock or unlock it. |

These follow Part 1.2 of the implementation plan, so `deepAnalysis.js` can call the
same endpoints later.

## Differences from the notebook

The analysis code is the notebook's. The changes are:

- The Groq key comes from `GROQ_API_KEY` instead of a Colab secret.
- `torch` is gone; it was only used to detect a GPU, which `ctranslate2` does.
- If a model is missing, `build_llm_input` leaves its part out instead of sending the
  word "unavailable" to the LLM. With every model loaded the sentence is identical.
- The Whisper transcript is never stored or returned, only the most frequent words.
- Each result carries per-stage timings for the page.
