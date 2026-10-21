# seed42 audio → prompt pipeline

seed42 turns music into visuals: a real-time image stream (SDXL-Turbo) draws whatever
prompt it is given. This code listens to the music and keeps writing that prompt, so
the picture follows what is playing.

## How it works

Audio comes in from an uploaded track or live from a microphone, desk or DJ software.
It is cut into overlapping windows (by default 10 seconds long, one every 5 seconds),
and each window goes through the same five steps. Every file below is in `server/app/`.

```
 uploaded track ─┐
  (jobs.py)      ├─▶ 1 Analyse ─▶ 2 Describe ─▶ 3 Decide scene ─▶ 4 Write prompt ─▶ 5 Send
 live audio ─────┘   features.py   features.py   scene.py          prompt.py         control.py
  (live.py)          models.py
```

| Step | What happens | File |
|---|---|---|
| **In** | A track is decoded and cut into windows by a background worker; live audio arrives over a WebSocket and is kept as a rolling window | `jobs.py`, `live.py` |
| **1 Analyse** | Tempo (Madmom, or Librosa as a fallback), energy and its phase (breakdown, groove, build, drop), the loudest frequency band, genre and mood (Essentia), lyric keywords (Whisper). The models load once at startup. | `features.py`, `models.py` |
| **2 Describe** | The results become one sentence, for example *"A party, happy mood, Steady Groove Phase 125 BPM House atmosphere, featuring Low-End with instrumental themes."* | `features.py` |
| **3 Decide scene** | The window is compared with the music the current prompt was written for. The scene is held, adjusted, blended or replaced, so the picture changes when the music does, not every window. See [Scenes](#scenes). | `scene.py` |
| **4 Write prompt** | When the scene changes, an LLM on Groq writes or revises a short visual prompt from the sentence. | `prompt.py` |
| **5 Send** | The prompt goes to the seed42 Control API, which drives the image stream. By default this is the mock in `mock/`, so nothing touches a real stream. | `control.py` |
| **Out** | Each window's record (features, prompt, scene decision, Control API answer) goes back to the test page at http://localhost:8000 | `main.py`, `static/` |

`run_window()` in `jobs.py` chains steps 1 to 5 for one window; file uploads and live
mode both call it. All settings come from `.env` through `config.py`.

## Layout

In the order audio flows through it:

```
server/app/
  config.py         settings, read from .env (see .env.example)
  main.py           the web server (FastAPI): API routes, and serves the test page
  jobs.py           IN: uploaded tracks, and run_window(), the five steps for one window
  live.py           IN: live audio over the /api/live WebSocket
  models.py         1: loads Madmom, Essentia and Whisper once, with per-model status
  features.py       1-2: per-window features and the sentence sent to the LLM
  scene.py          3: scene decisions: hold, adjust, blend, replace; track changes
  prompt.py         4: the Groq LLM call and prompt clean-up
  control.py        5: Control API client, reporting status codes instead of raising
  static/           OUT: the test page (index.html, app.js, styles.css)
server/Dockerfile   the pipeline image
server/requirements*.txt   pinned and tested together on Python 3.12
mock/mock-control-api.py   mock of the seed42 Control API
eval/compare.py     baseline vs refined comparison
eval/tracks/        your test audio and labels.csv (not committed)
eval/results/       comparison output, one folder per run (not committed)
scripts/run-local.sh       launcher without Docker (macOS, Linux)
docker-compose.yml  runs the pipeline and the mock; the eval service runs the comparison
legacy/streamlit/   the earlier Streamlit analyser, kept for reference
```

## Run it

There are three ways to run it. Pick by computer:

| Computer | Use |
|---|---|
| Windows | [Docker](#option-1-docker) |
| Linux, Intel Mac | [Docker](#option-1-docker), or [without Docker](#option-2-without-docker-macos-and-linux) |
| Apple Silicon Mac (M1 and later) | [Without Docker](#option-2-without-docker-macos-and-linux). Docker works but runs slowly under x86 emulation, because essentia-tensorflow has no Linux arm64 build. |
| Any, for page or Control API work only | [Lite mode](#option-3-lite-mode) |

### Before you start: a Groq API key

The LLM that writes the prompts runs on Groq. Without a key the app still runs, but
every window uses one fixed fallback prompt.

1. Sign in at https://console.groq.com and open **API Keys**.
2. Create a key and copy it. It is shown only once.

### Option 1: Docker

**1. Install the tools** (once)

- **Docker Desktop** from https://www.docker.com/products/docker-desktop/. On Windows,
  accept the WSL 2 option during install and restart if asked. Linux can use Docker
  Engine with the Compose plugin instead.
- **Git** from https://git-scm.com, to download the code. Or download the repository
  as a ZIP from GitHub and unzip it.
- Several GB of free disk space for the image and the models.

**2. Get the code**

```bash
git clone https://github.com/evelan-e/Seed42.git
cd Seed42
```

If you already have the folder, open a terminal in it instead. On Windows use
PowerShell, or the terminal in VS Code (**Terminal → New Terminal**).

**3. Create the settings file**

```bash
cp .env.example .env
```

On Windows PowerShell: `Copy-Item .env.example .env`. Open `.env` in any text editor,
paste your key after `GROQ_API_KEY=`, and save. Leave the other settings as they are
for now; `.env.example` explains each one.

**4. Start Docker Desktop** and wait until it shows that the engine is running.

**5. Build and start**

```bash
docker compose up --build
```

The first build takes several minutes while Madmom compiles. The first start then
downloads the Essentia and Whisper models into a Docker volume, so later starts are
fast. It is ready when the log shows `Uvicorn running on http://0.0.0.0:8000`.

**6. Open `http://localhost:8000`

The pills at the top right show what's working. **Models** counts up while they load,
then shows **ready**. **LLM** shows the model name (**fallback prompt** means no key).
**Stream** shows **not created yet** until the first prompt is sent to the mock. Click a
pill for details.

**7. Stop it** with Ctrl-C in the terminal, or `docker compose down` from another one.

**Everyday commands**

| To | Run |
|---|---|
| Start again | `docker compose up` |
| Start after changing code or pulling updates | `docker compose up --build` |
| Run in the background | `docker compose up -d` |
| Follow the server log | `docker compose logs -f pipeline` |
| Stop | `docker compose down` |
| Delete everything, including the downloaded models | `docker compose down --rmi local --volumes` |

### Option 2: Without Docker (macOS and Linux)

Not for Windows: essentia-tensorflow has no Windows build.

**1. Install the tools** (once)

- macOS: the Xcode command line tools (`xcode-select --install`), then Python 3.12 and
  ffmpeg, for example with Homebrew: `brew install python@3.12 ffmpeg`
- Linux (Debian/Ubuntu): `sudo apt install python3.12 python3.12-venv build-essential ffmpeg git`

The C compiler is needed because Madmom compiles on install. ffmpeg decodes mp3 and m4a.

**2. Get the code and create `.env`**, as in Docker steps 2 and 3.

**3. Start**

```bash
scripts/run-local.sh
```

The first run creates `server/.venv` and installs the pinned packages (about 2 GB, a
while). Later runs start straight away. It starts the mock and the server together;
open http://localhost:8000 when it prints the address. Ctrl-C stops both.

If `python3.12` isn't on your PATH: `PYTHON=/path/to/python3.12 scripts/run-local.sh`.

### Option 3: Lite mode

Librosa features, the LLM and the Control API only; genre, mood and themes show as
unavailable. It starts in seconds, for work on the page or the Control API path.

- Without Docker: `scripts/run-local.sh --lite` (installs in under a minute)
- With Docker: set `PIPELINE_MODE=lite` in `.env`, then `docker compose up`. Set it back
  to `full` for the whole pipeline.

### If something goes wrong

| Problem | Fix |
|---|---|
| `docker` is not recognised | Start Docker Desktop, wait for the engine, then open a new terminal |
| `port is already allocated` | Something else uses port 8000 or 8787, often an older copy of this app. `docker ps` lists running containers; stop the old one with `docker compose down` in its folder, or `docker rm -f <name>` |
| The page looks like an old version | The browser kept the old page. Press Ctrl+F5, or open it in a private window |
| **LLM** pill shows fallback prompt | `GROQ_API_KEY` is missing from `.env`. Add it, then restart |
| **Models** pill shows only some ready, for example 2 of 4 | Usually the model download failed. Check the internet connection and the log (`docker compose logs pipeline`), then restart |
| Live mode says audio input was blocked | Allow microphone access for the page in the browser's site settings. Open the page at `http://localhost:8000`, not by IP address |

## Using the page

- **Choose track** or drop a file anywhere on the page. Windows appear as they finish.
- **Listen live** streams the input chosen in the dropdown next to it to the server
  (see [Live input from a desk or DJ software](#live-input-from-a-desk-or-dj-software)).
  Speech processing (echo cancellation, noise suppression, auto gain) is switched off so
  music reaches the pipeline unchanged.
- **Window** and **Step** set the analysis window and how far it moves. 10/5 is the
  default; 60/60 analyses one minute at a time.
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

## Live input from a desk or DJ software

Live mode takes any audio input the computer has, not just a microphone:

| Source | How to connect it |
|---|---|
| DJ mixer or desk | Its USB audio output, or a line out into an audio interface (Focusrite or similar). It appears as an input device. |
| DJ software on the same computer | A virtual audio cable: VB-Audio Cable or Voicemeeter on Windows, BlackHole or Loopback on macOS. Set the DJ software's master output to the cable; the cable appears as an input. |
| Room sound | The built-in or a USB microphone. Fine for testing; crowd noise affects the analysis. |

Choose the input in the dropdown next to **Listen live**. Device names appear once the
browser has been allowed to use audio input; clicking the dropdown asks for that. The
choice is saved in that browser on that computer, so after a reload or reboot the same
input is used. If the saved input isn't plugged in, the dropdown says
"(not connected)" and live mode won't start until it's back or another input is chosen,
so a set never quietly falls back to the laptop microphone. The run line shows which
input is being used.

While listening, the level meter shows the input's loudness, and after 10 seconds
says whether it suits the analysis:

| Message | Meaning | Fix |
|---|---|---|
| Level OK | Loud enough for the energy phases to work | |
| Quiet | Average level under 0.05 RMS. The phase thresholds are set for finished tracks at full volume, so a quiet input reads as a breakdown all the time. | Raise the gain on the desk or interface, or the DJ software's master |
| Clipping | Samples at full scale in the last 2 seconds; the signal is distorted and reads as a drop | Lower the gain |
| No signal | Near silence for 10 seconds | Check the input chosen, the cable, and that the source is playing |

A virtual cable carrying the DJ software's master is usually at the right level
already. A physical line in depends on how the gain is set, so check the meter at
soundcheck.

Things to know:

- Browsers usually capture only the first two channels of a multi-channel interface.
  Route the music to inputs 1 and 2, or use a virtual cable.
- Browsers only allow audio input on `http://localhost` or HTTPS. Open the page on the
  computer running the server, or serve it over HTTPS.
- Capture runs in the browser tab, so keep it open for the whole set.

### Timing

If a window is still being analysed when the next one is due, the next one is skipped
and marked on the page. Each row shows how long analysis took; when that's longer than
the step, the row says so. On CPU, expect a few seconds per 10 s window with every
model loaded, so a 5 s step will skip windows. That measurement is what you need
before settling window and step for real live use.

## Scenes

The prompt is a scene that lasts, not a new image every window
(`server/app/scene.py`). The first audible window sets it. Each later window gets a
change score against the music the scene was written for (tempo, genre, mood,
energy, loudest band), and the scene is:

- **held** under `SCENE_EVOLVE_AT` (0.2): the same prompt is re-sent, so the stream
  is never left without one
- **adjusted** from 0.2: the LLM revises the current prompt, changing a few phrases
- **blended** from `SCENE_CHANGE_AT` (0.4), or after a gap of near silence: a halfway
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

## Baseline vs refined comparison

`eval/compare.py` runs a folder of tracks through two pipelines and writes a comparison
table, for showing that the refined pipeline beats the simplest one.

| | Baseline | Refined |
|---|---|---|
| Features | Librosa: tempo, energy phase, loudest band | + Madmom tempo, Essentia genre and mood, Whisper lyric themes |
| Genre | Always the most common label (majority class) | Essentia |
| Prompt | Fixed template from phase and band | Written by the LLM |
| Scenes | None; rebuilt every window | Scene tracking |

Both pipelines run on the same windows of the same tracks in one pass, and nothing is
sent to the Control API.

### Running it

1. Start Docker, open a terminal in the project folder, and check `GROQ_API_KEY` is set
   in `.env`. Without the LLM, scene tracking holds one scene per track, so the scene
   rows mean little (the script warns you).
2. Put audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aiff`) in `eval/tracks/`.
3. Optionally add ground truth in `eval/tracks/labels.csv` (see below). Without it the
   accuracy rows show `n/a`.
4. Run one of:

   ```bash
   docker compose run --rm eval
   docker compose run --rm eval --judge
   ```

   The first run builds the image and loads the models; after that it prints one line
   per window. When it finishes, the table is printed and saved.

| Option | What it does |
|---|---|
| `--judge` | Also scores every prompt from both pipelines with the LLM, 1 to 5 (two LLM calls per window) |
| `--window 20 --step 10` | Window length and the gap between window starts, in seconds |
| `--no-llm` | Runs the refined pipeline without the LLM |
| `--labels path.csv` | A labels file somewhere other than `eval/tracks/labels.csv` |
| `--out folder` | Where to write results, instead of `eval/results/` |
| `eval/other-folder` | A different folder of tracks, given before the options |

Audio files and results are ignored by git, so they are never committed.

### Ground truth: labels.csv

One row per file. Every column except `file` can be left empty.

```csv
file,genre,bpm,boundaries
blues.00001.wav,blues,98,
rock.00012.wav,rock,128,
two-songs-joined.mp3,,,95.5
```

| Column | Meaning | Scores |
|---|---|---|
| `file` | File name in `eval/tracks/`, exactly as it is on disk | |
| `genre` | The track's genre in plain words: `rock`, `house`, `hiphop`, `metal` | Genre accuracy |
| `bpm` | The track's tempo | Tempo accuracy |
| `boundaries` | For a file of songs joined together, the second(s) where one song ends and the next begins, separated by `;` | Track changes caught |

A genre counts as correct when the label appears anywhere in the predicted genre, so
`metal` matches `Rock---Heavy Metal` and `house` matches `Electronic---Tech House`.
Labelled datasets such as GTZAN (10 genres, 100 tracks each) work well. Use several
genres; with tracks of one genre the genre score says little.

### Results

Each run writes a folder `eval/results/<date>-<time>/` with three files:

- `summary.md`: the comparison table, for a report
- `windows.csv`: every window from both pipelines, for checking details and making charts
- `summary.json`: the same numbers as `summary.md`, for scripts

#### summary.md

The header gives the run: number of tracks and windows, window and step, the mode, which
models loaded, and how many prompts the LLM wrote. If a model shows `failed` or
`skipped`, the refined column is missing that feature.

| Metric | What it measures | Better |
|---|---|---|
| Tempo accuracy, within 4% | Share of labelled windows whose tempo is within 4% of `bpm`. Baseline uses Librosa, refined uses Madmom. | Higher |
| Tempo accuracy, allowing x2 / /2 | The same, but also accepting double, triple, half or a third of the tempo. Tempo trackers often lock onto half or double time, which still looks in time on screen. | Higher |
| Genre accuracy | Share of labelled windows with the right genre. The baseline always answers the most common label in `labels.csv`, the textbook simplest classifier; refined must beat that to be worth having. | Higher |
| Track changes caught | Share of `boundaries` with a scene change soon after: in a window ending between the boundary and one window plus one step later, because scene tracking blends first and confirms on the next window. | Higher |
| Scene changes away from a track change | Scene changes in files with `boundaries` that aren't near one: false alarms. | Lower |
| Scene changes per minute | How often a new scene starts. For the baseline, any change of its template prompt; for refined, a new scene or new track, not the first scene of each track. Within a single song this should be low. | Lower within a song |
| Prompt changes per minute | How often the prompt text changes at all, including refined's small adjustments (evolve and blend). Higher here with few scene changes means the picture follows the music without jumping. | Neither: read with the row above |
| Prompt fit, LLM judge (1-5) | Average score the LLM gives each prompt for how well it fits the music. Only with `--judge`. | Higher |
| Time per window | Average processing time per window. Baseline counts the Librosa steps; refined counts every model plus the LLM call. In live mode it has to stay under the step (10 s by default). | Lower |

`n/a` means the run had nothing to score that row with, usually missing labels.

Two cautions when reporting results:

- The judge sees the refined pipeline's description of the music, and the refined
  prompt was written from that same description, so it favours the refined prompt.
  Treat it as a guide, or back it up with people rating prompts without knowing which
  pipeline wrote them.
- Results from a few tracks are anecdotes. Use enough tracks, in enough genres, for the
  numbers to mean something.

#### windows.csv

One row per window. Columns starting `base_` are the baseline, `ref_` the refined
pipeline, `true_` the ground truth. Open it in Excel, or in VS Code with the Rainbow CSV
extension, to see it as columns.

| Column | Meaning |
|---|---|
| `track`, `window` | File name and window number, from 0 |
| `t_start`, `t_end` | Where the window starts and ends in the track, in seconds |
| `energy`, `phase`, `band` | Loudness (RMS), its energy phase (breakdown, groove, build, drop) and the loudest frequency band (sub, mid, high). The same in both pipelines. |
| `true_bpm`, `base_bpm`, `ref_bpm` | Tempo: from `labels.csv`, from Librosa, from the refined pipeline |
| `ref_bpm_source` | `Madmom neural`, or `Librosa DSP` when Madmom found too few beats |
| `true_genre`, `ref_genre` | Genre from `labels.csv`, and Essentia's top genre as `Parent---Style`. There is no baseline column: the baseline doesn't analyse genre. |
| `ref_mood`, `ref_themes` | Essentia's moods and Whisper's most frequent lyric words (`instrumental` if none). Refined only. |
| `music_description` | The sentence describing the window that the LLM writes its prompt from |
| `base_prompt`, `ref_prompt` | The prompt each pipeline would send for this window |
| `ref_prompt_source` | `llm` (written this window), `held` (the previous prompt re-sent), `fallback` (the LLM failed), `waiting` (silence before the first scene) or `off` (no LLM) |
| `ref_decision` | What scene tracking did: `initial` (first scene), `hold`, `evolve` (adjusted), `blend` (halfway to new music), `new_scene`, `new_track`, `revert` (back to the scene before a blend) or `wait` (silence, no scene yet) |
| `base_changed`, `ref_changed` | Whether the prompt differs from the previous window's |
| `base_scene_start`, `ref_scene_start` | Whether a new scene started in this window |
| `base_ms`, `ref_ms` | Processing time for this window, in milliseconds |
| `base_judge`, `ref_judge` | LLM fit score 1 to 5, with `--judge` |

For a report, a few rows of `base_prompt` next to `ref_prompt`, and the `base_bpm` and
`ref_bpm` columns charted over time, show the difference well.

## Connecting to the real Control API

`server/app/control.py` follows seed42's `control-api.md`, which the mock in `mock/`
implements: log in at `/api/auth`, create a stream, `PATCH /api/streams` with
`{id, params}`. Don't edit the mock; it is seed42's. Switching to the real API is
only `.env`:

- **Testing live with your own account:** set `CONTROL_API_URL=https://www.seed42.app`
  and your seed42 `CONTROL_API_EMAIL` and `CONTROL_API_PASSWORD`. The app creates a
  stream of its own, as with the mock.
- **A real performance:** seed42's own client creates the stream and gives you its
  id. Also set `CONTROL_API_STREAM_ID` to that id. The app then sends to that stream
  only, and never creates or ends one, so **New stream** is disabled. If that stream
  ends, prompts fail with a message saying so until you set the new id.

Leave `CONTROL_API_URL` unset to use the mock, under Docker or not.

### Testing failure handling with the mock

The mock is deliberately strict. Change its flags in `docker-compose.yml`, or run it
yourself with `python mock/mock-control-api.py --help`:

- `--unstable` ends streams after 45 s and makes DELETE fail
- `--not-ready 409` uses the other legal not-ready code
- `--async-create` answers 202 on create and requires polling
- `--ready-after 20` keeps a new stream not ready for 20 s instead of 4

The mock also prints a warning if a prompt update carries a creation-time field, which
on the real API would reload the pipeline and stop the show for about 30 seconds.

seed42's guide (`control-api.md`, section 6) asks for one run against each of
`--not-ready 404`, `--not-ready 409`, `--async-create` and `--unstable` before asking
for live access.

#### Running a test

1. **Set the flags** on the mock's `command` line in `docker-compose.yml`. A flag and
   its value are two separate items:

   ```yaml
   command: ["python", "-u", "mock-control-api.py", "--host", "0.0.0.0", "--fail-rate", "0.2"]
   ```

   `"--fail-rate 0.2"` as one item makes the mock exit with *unrecognized arguments*,
   and the app then reports *Failed to resolve 'mock'*. Keep `"--host", "0.0.0.0"`, or
   the app can't reach the mock.
2. **Restart:** `docker compose up -d`. Use this rather than `docker compose up -d mock`,
   which recreates the mock but leaves the app stopped (the browser shows
   `ERR_CONNECTION_REFUSED`). Until you restart, the old mock and its flags keep running.
3. **Watch the mock:** `docker compose logs -f mock`. This stops by itself whenever the
   mock is restarted, so run it again after step 2. Ctrl-C stops watching, not the mock.
4. **Upload a track**, or listen live, and check the results below.
5. **Put the line back** when done (no flags after `"0.0.0.0"`) and `docker compose up -d`.

After a restart the mock has forgotten the app's stream, so the first call is answered
`404` (stream not found) and the app creates a new stream. That's expected.

The pipeline log (`docker compose logs -f pipeline`) shows the page talking to the app,
so it stays `200 OK` throughout. The Control API results are on the page (each window's
row, red marks on the lane, the Stream pill) and in the mock's log.

#### What each test should show

| Flags | Mock log | Page |
|---|---|---|
| `--not-ready 404 --ready-after 20` | `POST 201`, then `PATCH 404` three times per window, then `200` | `404 from stream`, *Still not ready after 3 tries*, then `200` |
| `--not-ready 409 --ready-after 20` | The same with `409` | `409 from stream`, then `200` |
| `--async-create` | `POST 202`, `GET ?id=…` every 2 s until ready, then `PATCH 200` | `200`, with no retries |
| `--unstable` (let it run over a minute) | `PATCH 200` until 45 s, `PATCH 502`, then `POST 201` for a new stream and `200` again | One `502 from stream`, *The stream has ended*, then `200` |
| `--fail-rate 0.2` | About one call in five `502`, then a new stream | `502 from stream` on some windows |

Without `--ready-after 20` a stream is ready after 4 s, so the third try succeeds and
the page shows `200, 3 tries` instead of the failure. That tests recovery; the longer
wait tests giving up.

A 404 that says *not found* (the stream doesn't exist) is not retried; only *not ready*
is. With an uploaded file the windows run faster than real time, so several windows can
each make their three tries while one stream is starting.

With `--fail-rate`, the mock answers 502 before reading the request body, then reads
that leftover body as a new request and logs `code 400, message Bad request version`.
Those lines come from the mock, not from the app, and can be ignored.

## API

The test page (`server/app/static/app.js`) uses these endpoints, and any other front end
can call them the same way. Routes are in `server/app/main.py`, the live protocol in
`server/app/live.py`. All responses are JSON.

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Server status: pipeline mode, which models loaded, whether the LLM is on, the Control API stream, the current scene |
| `POST /api/analyze` | Upload a track to analyse. Returns a job id |
| `GET /api/analyze/{id}` | A job's progress and results so far |
| `DELETE /api/analyze/{id}` | Stop a job |
| `WS /api/live` | Stream live audio and get results as each window finishes |
| `GET /api/scene` | The prompt currently on the stream, and the scene state |
| `POST /api/scene` | Set the prompt by hand, or lock or unlock it |
| `POST /api/stream/reset` | End the current Control API stream; the next window creates a new one |

### Analysing a file

1. `POST /api/analyze` as multipart form data:

   | Field | Default | Meaning |
   |---|---|---|
   | `file` | required | The audio file, up to `MAX_UPLOAD_MB` (200 MB) |
   | `window_sec` | 10 | Window length in seconds, at least 2 |
   | `step_sec` | 5 | Seconds between window starts, at least 1 |
   | `use_llm` | true | Write prompts with the LLM |
   | `send_to_stream` | true | Send each prompt to the Control API |

   Returns `{"id": "..."}`. Jobs run one at a time, in order.

2. Poll `GET /api/analyze/{id}?since=N` every second or so. `status` goes `queued` →
   `decoding` → `waiting-for-models` → `running` → `done` (or `failed`, `cancelled`).
   `results` holds one record per finished window: features, prompt, scene decision and
   Control API outcome. Pass `since` = the number of windows you already have to get
   only the new ones. The server keeps the 20 most recent jobs.

### Live audio

Open a WebSocket to `/api/live`, then:

1. Send a text message: `{"type": "start", "window_sec": 10, "step_sec": 5, "use_llm": true, "send_to_stream": true}`.
2. Send the audio as binary messages: mono Float32 samples at 22050 Hz, in chunks of
   about half a second.
3. Receive text messages:
   - `{"type": "started"}`: ready for audio
   - `{"type": "window", ...}`: one finished window, the same record as for files
   - `{"type": "skipped", ...}`: the previous window was still being analysed, so this
     one was skipped; the current prompt was re-sent
   - `{"type": "error", "message": ...}`

### Setting the prompt by hand

`POST /api/scene` with JSON:

| Field | Meaning |
|---|---|
| `prompt` | A new prompt. Leave it out to change only the lock |
| `locked` | `true` re-sends the prompt every window unchanged; `false` lets the pipeline adjust it as the music changes |
| `send` | Send the prompt to the stream now (default `true`) |

## Design notes

- Each model loads on its own. If one fails, only its feature is missing: tempo falls
  back to Librosa, or genre and mood show as unavailable.
- If a model is missing, `build_llm_input` leaves its part out instead of sending the
  word "unavailable" to the LLM.
- A GPU is used automatically when `ctranslate2` finds one.
- The Whisper transcript is never stored or returned, only the most frequent words.
- Each result carries per-stage timings for the page.
