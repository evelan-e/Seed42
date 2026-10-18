/* seed42 pipeline test page.
   File mode: POST /api/analyze, then poll GET /api/analyze/{id}?since=N.
   Live mode: mic -> AudioWorklet -> 22050 Hz mono Float32 -> WS /api/live. */
(() => {
  "use strict";

  const SR = 22050;
  const $ = (id) => document.getElementById(id);
  const el = {
    file: $("fileInput"), fileLabel: document.querySelector('label[for="fileInput"]'),
    live: $("liveBtn"), input: $("inputSelect"),
    level: $("level"), meterFill: $("meterFill"), meterPeak: $("meterPeak"), levelText: $("levelText"),
    windowSec: $("windowSec"), stepSec: $("stepSec"),
    useLlm: $("useLlm"), sendStream: $("sendStream"),
    pillModels: $("pillModels"), pillLlm: $("pillLlm"), pillStream: $("pillStream"),
    healthBody: $("healthBody"),
    run: $("runLine"), cancel: $("cancelBtn"), resetStream: $("resetStreamBtn"),
    strip: $("strip"), canvas: $("stripCanvas"), tip: $("stripTip"), player: $("player"),
    notice: $("notice"), windows: $("windows"), empty: $("empty"),
    tpl: $("rowTpl"), drop: $("dropOverlay"),
    sceneBox: $("scene"), sceneText: $("sceneText"), sceneSend: $("sceneSend"), sceneLock: $("sceneLock"),
    sceneRevert: $("sceneRevert"), sceneStatus: $("sceneStatus"), sceneResult: $("sceneResult"),
  };

  const PHASE_LABEL = { breakdown: "Breakdown", groove: "Steady groove", build: "Build-up", drop: "Drop" };
  const PHASE_COLOR = {};
  const BAND = {
    sub: ["Sub-bass", "60 to 120 Hz"],
    mid: ["Mid-range", "120 Hz to 2 kHz, vocals and synths"],
    high: ["High end", "2 to 8 kHz, hats and percussion"],
  };
  const THRESHOLDS = [0.15, 0.40, 0.65];
  const SCENE_LABEL = {
    initial: "First scene", hold: "Held", evolve: "Adjusted", blend: "Blending",
    new_scene: "New scene", new_track: "New track", revert: "Change didn't last",
    locked: "Locked", wait: "Waiting", manual: "Your prompt",
  };
  const SCENE_STARTS = new Set(["initial", "new_scene", "new_track"]);
  const PART_LABEL = { bpm: "tempo", genre: "genre", mood: "mood", energy: "energy", band: "band" };

  const S = {
    mode: null,             // "file" | "live"
    jobId: null, since: 0, pollTimer: null, jobStatus: null,
    filename: "", duration: 0, total: 0,
    windowSec: 20, stepSec: 10,
    envelope: null,         // {values, dt}
    records: [],            // window records and skipped markers
    selected: null,         // index into records
    objectUrl: null,
    live: null,             // {ws, ac, stream, node, ...}
    liveElapsed: 0,
    hover: null,
    scene: null,            // the server's current scene
    sceneDirty: false,      // the editor holds text the user hasn't sent
  };

  // ---------------------------------------------------------------- helpers
  const fmt = (t) => {
    t = Math.max(0, t || 0);
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };
  const secs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
  const node = (tag, text, cls) => {
    const n = document.createElement(tag);
    if (text != null) n.textContent = text;
    if (cls) n.className = cls;
    return n;
  };
  function notice(msg, kind = "error") {
    if (!msg) { el.notice.hidden = true; return; }
    el.notice.textContent = msg;
    el.notice.className = "notice" + (kind === "info" ? " info" : "");
    el.notice.hidden = false;
  }
  function readOpts() {
    const w = Math.max(2, Number(el.windowSec.value) || 20);
    const s = Math.max(1, Number(el.stepSec.value) || 10);
    el.windowSec.value = w; el.stepSec.value = s;
    return { window_sec: w, step_sec: s, use_llm: el.useLlm.checked, send_to_stream: el.sendStream.checked };
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  for (const k of Object.keys(PHASE_LABEL)) PHASE_COLOR[k] = cssVar(`--ph-${k}`);
  function phaseOf(v) { return v < 0.15 ? "breakdown" : v < 0.40 ? "groove" : v < 0.65 ? "build" : "drop"; }

  // ---------------------------------------------------------------- health
  let healthTimer = null;
  async function refreshHealth() {
    let h;
    try {
      const r = await fetch("/api/health");
      h = await r.json();
    } catch {
      setPill(el.pillModels, "err", "server unreachable");
      healthTimer = setTimeout(refreshHealth, 3000);
      return;
    }
    const comps = Object.values(h.models);
    const ready = comps.filter((c) => c.state === "ready").length;
    const failed = comps.filter((c) => c.state === "failed").length;
    if (h.mode === "lite") setPill(el.pillModels, "warn", "lite mode");
    else if (!h.models_ready) setPill(el.pillModels, "warn", `loading, ${ready} of ${comps.length}`);
    else if (failed) setPill(el.pillModels, "warn", `${ready} of ${comps.length} ready`);
    else setPill(el.pillModels, "ok", "ready");

    if (h.llm.enabled) setPill(el.pillLlm, "ok", h.llm.model.split("/").pop());
    else setPill(el.pillLlm, "warn", "fallback prompt");

    const c = h.control_api;
    if (c.last_error) setPill(el.pillStream, "err", "error");
    else if (c.stream_id) setPill(el.pillStream, "ok", c.stream_id.slice(0, 12));
    else setPill(el.pillStream, "", "not created yet");
    // A stream seed42 gave us is theirs to end, not ours
    el.resetStream.disabled = c.given;
    el.resetStream.title = c.given
      ? "Disabled: this stream was given by seed42 (CONTROL_API_STREAM_ID)"
      : "End the current stream; the next window creates a new one";

    renderHealthBody(h);
    renderScene(h.scene);
    if (!S.defaultsApplied && h.defaults) {
      el.windowSec.value = h.defaults.window_sec; el.stepSec.value = h.defaults.step_sec;
      S.defaultsApplied = true;
    }
    healthTimer = setTimeout(refreshHealth, h.models_ready ? 8000 : 2000);
  }
  function setPill(p, kind, text) {
    p.className = "pill" + (kind ? " " + kind : "");
    p.querySelector("span").textContent = text;
  }
  function renderHealthBody(h) {
    const dl = node("dl");
    const add = (k, v, cls) => { dl.append(node("dt", k)); dl.append(node("dd", v, cls)); };
    add("Mode", h.mode === "lite" ? "Lite: Librosa only, no genre, mood or themes" : `Full, on ${h.device}`);
    const names = { madmom: "Madmom", essentia: "Essentia", mood: "Mood", whisper: "Whisper" };
    for (const [k, v] of Object.entries(h.models)) {
      add(names[k] || k, v.state + (v.detail ? `: ${v.detail}` : ""), "state-" + v.state);
    }
    add("LLM", h.llm.enabled ? h.llm.model : "No GROQ_API_KEY. Every window uses the fallback prompt.");
    add("Control API", h.control_api.url);
    add("Stream", h.control_api.given ? `${h.control_api.stream_id}, given by seed42`
      : h.control_api.stream_id || "Created on the first prompt sent");
    if (h.control_api.last_error) add("Last error", h.control_api.last_error, "state-failed");
    el.healthBody.replaceChildren(dl);
  }

  // ---------------------------------------------------------------- scene editor
  function renderScene(sc) {
    if (!sc) return;
    S.scene = sc;
    if (!S.sceneDirty) el.sceneText.value = sc.prompt || "";
    el.sceneLock.checked = sc.locked;
    el.sceneBox.classList.toggle("is-locked", sc.locked);
    if (!sc.prompt) {
      el.sceneStatus.textContent = "No scene yet. The first audible window sets one, or type your own.";
      return;
    }
    const what = sc.locked ? "Your prompt, locked" : sc.manual ? "Your prompt" : SCENE_LABEL[sc.decision] || "Set";
    el.sceneStatus.replaceChildren(node("strong", `Scene ${sc.scene}`),
      `${sc.track ? `, track ${sc.track}` : ""}. ${what}${sc.pending ? ", change pending" : ""}.`);
  }
  function setDirty(d) {
    S.sceneDirty = d;
    el.sceneBox.classList.toggle("is-dirty", d);
    el.sceneRevert.hidden = !d;
  }
  function sceneResult(text, kind) {
    el.sceneResult.textContent = text;
    el.sceneResult.className = "scene-result" + (kind ? " " + kind : "");
  }
  function editPrompt(text) {
    el.sceneText.value = text;
    setDirty(text !== (S.scene && S.scene.prompt));
    sceneResult("");
    el.sceneBox.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    el.sceneText.focus();
  }
  async function postScene(body) {
    sceneResult("Sending.");
    let r, j;
    try {
      r = await fetch("/api/scene", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      j = await r.json();
    } catch {
      return sceneResult("Couldn't reach the server. Check that it's still running.", "err");
    }
    if (!r.ok) return sceneResult(typeof j.detail === "string" ? j.detail : `Failed with ${r.status}.`, "err");
    if (body.prompt != null) setDirty(false);
    renderScene(j.scene);
    const a = j.api;
    if (a && a.ok) sceneResult(`Sent, ${a.code} in ${secs(a.ms)}.`, "ok");
    else if (a && a.superseded) sceneResult("Saved, but a newer prompt was sent before it.", "err");
    else if (a) sceneResult(`Saved, but the stream didn't take it. ${a.error || a.code}`, "err");
    else sceneResult(j.scene.locked ? "Locked. Every window re-sends this prompt." : "Unlocked. The pipeline adjusts from here.", "ok");
    clearTimeout(healthTimer); refreshHealth();
  }
  function sendScene() {
    const prompt = el.sceneText.value.trim();
    if (!prompt) return sceneResult("Type a prompt first.", "err");
    postScene({ prompt, locked: el.sceneLock.checked, send: true });
  }

  // ---------------------------------------------------------------- reset view
  function resetView() {
    S.records = []; S.selected = null; S.since = 0; S.total = 0; S.envelope = null;
    S.duration = 0; S.jobStatus = null; S.liveElapsed = 0;
    el.windows.replaceChildren(el.empty);
    setEmpty("idle");
    notice(null);
    drawStrip();
  }

  const EMPTY = {
    idle: ["Choose a track or listen live. Each window's analysis, the prompt it produces, and the Control API's answer appear here.",
           "You can also drop an audio file anywhere on this page."],
    file: ["Analysing the first window.", "On CPU with every model loaded, expect several seconds per window."],
    live: ["Listening. The first window is analysed once a full window of audio has arrived.",
           "Play music near the microphone. Speech-tuned processing is switched off so the music reaches the pipeline unchanged."],
  };
  function setEmpty(kind) {
    const [a, b] = EMPTY[kind];
    el.empty.children[0].textContent = a;
    el.empty.children[1].textContent = b;
    el.empty.hidden = false;
  }

  // ---------------------------------------------------------------- file mode
  async function startFile(file) {
    if (!file) return;
    if (S.live) await stopLive();
    await cancelJob();
    resetView();
    const opts = readOpts();
    S.mode = "file"; S.filename = file.name; S.windowSec = opts.window_sec; S.stepSec = opts.step_sec;

    if (S.objectUrl) URL.revokeObjectURL(S.objectUrl);
    S.objectUrl = URL.createObjectURL(file);
    el.player.src = S.objectUrl;
    el.player.hidden = false;

    el.run.innerHTML = "";
    el.run.append(node("strong", file.name), " uploading.");
    setEmpty("file");

    const fd = new FormData();
    fd.append("file", file);
    for (const [k, v] of Object.entries(opts)) fd.append(k, String(v));
    let r;
    try {
      r = await fetch("/api/analyze", { method: "POST", body: fd });
    } catch {
      return notice("Couldn't reach the server. Check that it's still running.");
    }
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      return notice(body.detail || `Upload failed with ${r.status}.`);
    }
    S.jobId = (await r.json()).id;
    el.cancel.hidden = false;
    poll();
  }

  async function poll() {
    if (!S.jobId) return;
    const id = S.jobId;
    let j;
    try {
      const r = await fetch(`/api/analyze/${id}?since=${S.since}`);
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.status);
      j = await r.json();
    } catch (e) {
      notice(`Lost track of the job: ${e.message}`);
      return stopPolling();
    }
    if (id !== S.jobId) return;

    if (j.envelope) { S.envelope = j.envelope; }
    if (j.duration) S.duration = j.duration;
    S.total = j.total_windows || 0;
    S.jobStatus = j.status;
    for (const rec of j.results) addRecord(rec);
    S.since += j.results.length;
    renderRunLine(j);
    drawStrip();

    if (["done", "failed", "cancelled"].includes(j.status)) {
      if (j.status === "failed") notice(j.error);
      return stopPolling();
    }
    S.pollTimer = setTimeout(poll, 1000);
  }
  function stopPolling() {
    clearTimeout(S.pollTimer); S.pollTimer = null;
    el.cancel.hidden = true;
  }
  async function cancelJob() {
    if (S.jobId && S.pollTimer) {
      fetch(`/api/analyze/${S.jobId}`, { method: "DELETE" }).catch(() => {});
    }
    stopPolling();
    S.jobId = null;
  }

  function renderRunLine(j) {
    el.run.innerHTML = "";
    el.run.append(node("strong", j.filename), ` ${fmt(j.duration)}. `);
    const done = j.done_windows, total = j.total_windows;
    const text = {
      queued: "Queued behind another track.",
      decoding: "Decoding audio.",
      "waiting-for-models": "Waiting for the models to finish loading.",
      running: `${done} of ${total} windows analysed.`,
      done: `All ${total} windows analysed.`,
      failed: "Analysis failed.",
      cancelled: `Stopped after ${done} of ${total} windows.`,
    }[j.status] || j.status;
    el.run.append(text);
  }

  // ---------------------------------------------------------------- audio input
  // Any input the system has: a microphone, a desk soundcard or audio interface,
  // or a virtual cable carrying DJ software's output. The choice is kept in this
  // browser so a gig setup doesn't fall back to the laptop mic after a reload.
  const INPUT_KEY = "seed42.input";       // {id, label}
  const MISSING = "missing";
  let inputsNamed = false;
  const store = {
    get() { try { return JSON.parse(localStorage.getItem(INPUT_KEY)); } catch { return null; } },
    set(v) { try { v ? localStorage.setItem(INPUT_KEY, JSON.stringify(v)) : localStorage.removeItem(INPUT_KEY); } catch {} },
  };

  async function listInputs() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      el.input.hidden = true;
      return;
    }
    let inputs = [];
    try {
      inputs = (await navigator.mediaDevices.enumerateDevices())
        .filter((d) => d.kind === "audioinput" && d.deviceId !== "default" && d.deviceId !== "communications");
    } catch {}
    // Browsers hide device names until the page has been allowed to use audio input
    const named = inputsNamed = inputs.some((d) => d.label);
    const saved = store.get();
    el.input.innerHTML = "";
    el.input.append(new Option("Default input", ""));
    inputs.forEach((d, i) => el.input.append(new Option(d.label || `Input ${i + 1}`, d.deviceId)));
    if (!named) {
      const o = new Option("Allow audio access to see input names", "", false, false);
      o.disabled = true;
      el.input.append(o);
    }

    el.input.classList.remove("is-missing");
    if (!saved) { el.input.value = ""; return; }
    const match = inputs.find((d) => d.deviceId === saved.id) || inputs.find((d) => d.label && d.label === saved.label);
    if (match) {
      el.input.value = match.deviceId;
      if (match.deviceId !== saved.id) store.set({ id: match.deviceId, label: match.label });
    } else if (named) {
      // The saved input is unplugged: say so instead of quietly using another one
      const o = new Option(`${saved.label} (not connected)`, MISSING);
      el.input.prepend(o);
      el.input.value = MISSING;
      el.input.classList.add("is-missing");
    } else {
      el.input.value = "";
    }
  }

  // Asking for audio once unlocks the device names; the stream is closed straight away
  async function unlockInputNames() {
    if (inputsNamed || !navigator.mediaDevices) return;
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      s.getTracks().forEach((t) => t.stop());
      await listInputs();
    } catch {}
  }

  el.input.addEventListener("pointerdown", unlockInputNames);
  el.input.addEventListener("keydown", unlockInputNames);
  el.input.addEventListener("change", () => {
    const o = el.input.selectedOptions[0];
    if (el.input.value === MISSING) return;
    store.set(el.input.value ? { id: el.input.value, label: o.text } : null);
    el.input.classList.remove("is-missing");
    el.input.querySelector(`option[value="${MISSING}"]`)?.remove();
  });
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", () => { if (!S.live) listInputs(); });
  }

  // ---------------------------------------------------------------- input level
  // The energy phases use fixed loudness thresholds tuned on finished tracks, so a
  // desk output with the gain too low reads as a breakdown all night, and a clipping
  // one as a drop. The meter warns about both before the set starts.
  const LEVEL = {
    silence: 0.01,      // the server's SCENE_SILENCE_RMS
    quiet: 0.05,        // average under this for LEVEL.span: raise the gain
    clip: 0.99,         // a sample this close to full scale
    span: 20,           // half-second chunks averaged: 10 s
  };

  function updateLevel(L, chunk) {
    let s = 0;
    for (let i = 0; i < chunk.length; i++) s += chunk[i] * chunk[i];
    const rms = Math.sqrt(s / chunk.length);
    L.levels.push({ rms, peak: L.peak });
    if (L.levels.length > LEVEL.span) L.levels.shift();
    L.peak = 0;

    // dBFS from -60 to 0 across the meter
    const pos = (v) => `${Math.max(0, Math.min(100, ((20 * Math.log10(Math.max(v, 1e-6))) + 60) / 60 * 100))}%`;
    el.meterFill.style.width = pos(rms);
    el.meterPeak.style.left = `calc(${pos(L.levels.at(-1).peak)} - 2px)`;

    const recent = L.levels.slice(-4);   // the last 2 s
    const avg = Math.sqrt(L.levels.reduce((a, l) => a + l.rms * l.rms, 0) / L.levels.length);
    let state, text;
    if (recent.some((l) => l.peak >= LEVEL.clip)) {
      [state, text] = ["err", "Clipping: lower the input gain"];
    } else if (L.levels.length < LEVEL.span) {
      [state, text] = ["", "Checking level"];
    } else if (avg < LEVEL.silence) {
      [state, text] = ["warn", "No signal: check the input and cable"];
    } else if (avg < LEVEL.quiet) {
      [state, text] = ["warn", "Quiet: raise the gain, or energy reads as breakdown"];
    } else {
      [state, text] = ["ok", "Level OK"];
    }
    el.level.className = `level ${state}`.trim();
    if (el.levelText.textContent !== text) el.levelText.textContent = text;
  }

  // ---------------------------------------------------------------- live mode
  const WORKLET = `
    class Tap extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0];
        if (ch && ch.length && ch[0].length) {
          const n = ch[0].length, out = new Float32Array(n);
          for (let c = 0; c < ch.length; c++) { const d = ch[c]; for (let i = 0; i < n; i++) out[i] += d[i] / ch.length; }
          this.port.postMessage(out, [out.buffer]);
        }
        return true;
      }
    }
    registerProcessor("tap", Tap);`;

  async function startLive() {
    if (el.input.value === MISSING) {
      return notice(`The saved input, ${store.get().label}, isn't connected. Plug it in, or choose another input.`);
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return notice("This browser blocks audio input here. Open the page at http://localhost:8000, or over HTTPS.");
    }
    await cancelJob();
    resetView();
    const opts = readOpts();
    S.mode = "live"; S.windowSec = opts.window_sec; S.stepSec = opts.step_sec;
    S.envelope = { values: [], dt: 1024 / SR };
    el.player.hidden = true; el.player.removeAttribute("src");

    let stream;
    const deviceId = el.input.value;
    try {
      // Processing off: echo cancellation and noise suppression are built for speech and damage music
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false, noiseSuppression: false, autoGainControl: false,
          ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
        },
      });
    } catch (e) {
      S.mode = null;
      if (e.name === "OverconstrainedError" || e.name === "NotFoundError") {
        listInputs();
        return notice("That input isn't available any more. Check it's plugged in, then choose it again.");
      }
      return notice("Audio input was blocked. Allow it for this page in the browser's site settings, then try again.");
    }
    listInputs();   // device names are visible now that access is allowed
    const inputName = (stream.getAudioTracks()[0] || {}).label || "the default input";

    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const blobUrl = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await ac.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);
    const src = ac.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(ac, "tap");
    const mute = ac.createGain(); mute.gain.value = 0;
    src.connect(tap); tap.connect(mute); mute.connect(ac.destination);

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/live`);
    ws.binaryType = "arraybuffer";

    const L = S.live = { ws, ac, stream, tap, ratio: ac.sampleRate / SR, pending: new Float32Array(0),
                         out: [], outLen: 0, levelCarry: new Float32Array(0), started: false,
                         inputName, levels: [], peak: 0 };

    ws.onopen = () => ws.send(JSON.stringify({ type: "start", ...opts }));
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "started") {
        L.started = true;
        if (!m.models_ready) notice("Models are still loading. Windows are skipped until they're ready.", "info");
      } else if (m.type === "window") {
        notice(null);
        addRecord(m);
        drawStrip();
      } else if (m.type === "skipped") {
        addRecord({ skipped: true, reason: m.reason, resent: m.resent, t_start: m.t_start, t_end: m.t_end, index: `s${m.t_end}` });
      } else if (m.type === "error") {
        notice(m.message);
      }
    };
    ws.onclose = () => { if (S.live === L) { stopLive(); notice("The live connection closed. Check the server log."); } };

    tap.port.onmessage = (ev) => onLiveSamples(L, ev.data);
    setEmpty("live");
    el.live.setAttribute("aria-pressed", "true");
    el.live.textContent = "Stop listening";
    el.fileLabel.classList.add("is-disabled");
    el.file.disabled = true;
    el.input.disabled = true;
    el.level.className = "level";
    el.levelText.textContent = "Checking level";
    el.meterFill.style.width = "0";
    el.level.hidden = false;
    renderLiveLine();
  }

  function onLiveSamples(L, block) {
    // Peak before resampling, so clipping isn't smoothed away
    for (let i = 0; i < block.length; i++) { const a = Math.abs(block[i]); if (a > L.peak) L.peak = a; }
    // Append and resample to 22050 Hz (linear; fine for these features)
    const merged = new Float32Array(L.pending.length + block.length);
    merged.set(L.pending); merged.set(block, L.pending.length);
    const n = Math.floor((merged.length - 1) / L.ratio);
    if (n <= 0) { L.pending = merged; return; }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * L.ratio, i0 = Math.floor(p), f = p - i0;
      out[i] = merged[i0] * (1 - f) + merged[i0 + 1] * f;
    }
    L.pending = merged.slice(Math.floor(n * L.ratio));
    L.out.push(out); L.outLen += out.length;

    // Send ~0.5 s at a time
    if (L.outLen >= SR / 2) {
      const chunk = new Float32Array(L.outLen);
      let o = 0; for (const a of L.out) { chunk.set(a, o); o += a.length; }
      L.out = []; L.outLen = 0;
      if (L.started && L.ws.readyState === 1) L.ws.send(chunk.buffer);
      pushLevels(L, chunk);
      updateLevel(L, chunk);
      S.liveElapsed += chunk.length / SR;
      renderLiveLine();
      drawStrip();
    }
  }

  function pushLevels(L, chunk) {
    const hop = 1024;
    const all = new Float32Array(L.levelCarry.length + chunk.length);
    all.set(L.levelCarry); all.set(chunk, L.levelCarry.length);
    let i = 0;
    for (; i + hop <= all.length; i += hop) {
      let s = 0; for (let j = i; j < i + hop; j++) s += all[j] * all[j];
      S.envelope.values.push(Math.sqrt(s / hop));
    }
    L.levelCarry = all.slice(i);
  }

  function renderLiveLine() {
    el.run.innerHTML = "";
    const n = S.records.filter((r) => !r.skipped).length;
    const wait = Math.max(0, S.windowSec - S.liveElapsed);
    el.run.append(node("strong", "Listening"), ` to ${S.live ? S.live.inputName : "input"} for ${fmt(S.liveElapsed)}. `);
    el.run.append(wait > 0 ? `First window in ${Math.ceil(wait)} s.` : `${n} window${n === 1 ? "" : "s"} analysed.`);
  }

  async function stopLive() {
    const L = S.live;
    if (!L) return;
    S.live = null;
    try { L.ws.close(); } catch {}
    try { L.stream.getTracks().forEach((t) => t.stop()); } catch {}
    try { await L.ac.close(); } catch {}
    el.live.setAttribute("aria-pressed", "false");
    el.live.textContent = "Listen live";
    el.fileLabel.classList.remove("is-disabled");
    el.file.disabled = false;
    el.input.disabled = false;
    el.level.hidden = true;
    listInputs();
    if (S.mode === "live") {
      const n = S.records.filter((r) => !r.skipped).length;
      el.run.innerHTML = "";
      el.run.append(node("strong", "Stopped"), ` after ${fmt(S.liveElapsed)}. ${n} window${n === 1 ? "" : "s"} analysed.`);
    }
  }

  // ---------------------------------------------------------------- rows
  function addRecord(rec) {
    S.records.push(rec);
    if (rec.scene) renderScene(rec.scene);
    el.empty.hidden = true;
    const row = rec.skipped ? skippedRow(rec) : windowRow(rec);
    rec._row = row;
    row.dataset.i = S.records.length - 1;
    if (S.mode === "live") el.windows.insertBefore(row, el.windows.firstChild);
    else el.windows.append(row);
    requestAnimationFrame(() => row.classList.add("is-new"));
  }

  function skippedRow(rec) {
    const a = node("article", null, "win is-skipped");
    const side = node("div", null, "win-side");
    side.append(node("span", `${fmt(rec.t_start)} to ${fmt(rec.t_end)}`, "win-time"));
    const main = node("div", `Skipped: ${rec.reason}.${rec.resent ? " The current scene was re-sent so the stream keeps a prompt." : ""}`, "win-main");
    a.append(side, main);
    return a;
  }

  function windowRow(rec) {
    const row = el.tpl.content.firstElementChild.cloneNode(true);
    const q = (s) => row.querySelector(s);

    const time = q(".win-time");
    time.textContent = `${fmt(rec.t_start)} to ${fmt(rec.t_end)}`;
    time.addEventListener("click", (e) => { e.preventDefault(); select(S.records.indexOf(rec), true); });

    const ph = q(".win-phase");
    ph.querySelector("i").className = "ph-" + rec.phase;
    ph.querySelector("span").textContent = PHASE_LABEL[rec.phase];

    // Control API outcome
    const api = q(".win-api"), a = rec.api || {};
    let apiText, apiKind;
    if (a.skipped) { apiText = "Not sent"; apiKind = ""; }
    else if (a.superseded) { apiText = "Replaced by a newer prompt"; apiKind = ""; }
    else if (a.ok) {
      apiText = `${a.code} in ${secs(a.ms)}` + (a.attempts > 1 ? `, ${a.attempts} tries` : "");
      apiKind = a.attempts > 1 ? "warn" : "ok";
    } else { apiText = a.code ? `${a.code} from stream` : "No response"; apiKind = "err"; }
    api.className = "win-api " + apiKind;
    api.querySelector("span").textContent = apiText;

    // Scene decision
    const sc = rec.scene || {};
    const sl = q(".win-scene");
    sl.classList.toggle("is-change", SCENE_STARTS.has(sc.decision));
    sl.querySelector("span").textContent = (SCENE_LABEL[sc.decision] || "No scene") +
      (sc.score != null ? `, change ${sc.score.toFixed(2)}` : "");

    // Prompt
    const p = q(".win-prompt"), meta = q(".win-meta");
    const src = rec.prompt_source;
    if (src === "off" || src === "waiting") {
      p.textContent = src === "off" ? "Prompt generation is off for this run." : "Nothing sent yet. Waiting for audio before choosing the first scene.";
      p.classList.add("is-off");
    } else {
      p.textContent = rec.prompt;
      if (src === "held") {
        p.classList.add("is-held");
        meta.append("Same prompt re-sent.");
      } else if (src === "manual") {
        p.classList.add("is-held");
        meta.append("Your locked prompt, re-sent.");
      } else if (src !== "llm") {
        p.classList.add("is-fallback");
        const why = rec.llm_error && rec.llm_error.startsWith("GROQ_API_KEY") ? " because GROQ_API_KEY isn't set" : "";
        meta.append(node("span", "Fallback prompt", "flag"), `${why}, ${rec.prompt_words} words.`);
      } else {
        meta.append(`${rec.prompt_words} words, written in ${secs(rec.llm_ms)}.`);
      }
      const edit = node("button", "Edit", "btn btn-quiet btn-edit");
      edit.type = "button";
      edit.title = "Copy this prompt into the scene editor";
      edit.addEventListener("click", () => editPrompt(rec.prompt));
      meta.append(edit);
    }
    q(".win-note").textContent = sc.note || "";

    const inp = q(".win-input");
    inp.append("From ", node("q", rec.llm_input));

    // Facts
    const facts = q(".facts");
    const fact = (label, value, small, na) => {
      const d = node("div");
      d.append(node("dt", label));
      const dd = node("dd", value, na ? "na" : null);
      if (small) dd.append(node("small", small));
      d.append(dd);
      facts.append(d);
    };
    fact("Tempo", `${Number(rec.bpm).toFixed(1)} BPM`,
      rec.bpm_source === "Madmom neural" ? `Madmom, ${rec.beat_count} beats. Librosa said ${rec.librosa_bpm}.` : "Librosa estimate");
    fact("Energy", rec.energy.toFixed(3), PHASE_LABEL[rec.phase]);
    if (sc.score != null) {
      const top = Object.entries(sc.parts || {}).filter(([, v]) => v > 0).sort((x, y) => y[1] - x[1]).slice(0, 3)
        .map(([k, v]) => `${PART_LABEL[k] || k} ${v.toFixed(2)}`).join(", ");
      const th = sc.thresholds || {};
      fact("Scene change", sc.score.toFixed(2), `${top || "nothing changed"}. Adjust at ${th.evolve}, replace at ${th.change}.`);
    }
    fact("Loudest band", BAND[rec.band][0], BAND[rec.band][1]);
    if (rec.primary_genre === "unavailable") fact("Genre", "Not available", "Essentia not loaded", true);
    else {
      const [parent, name] = splitGenre(rec.primary_genre);
      const second = rec.secondary_genre && rec.secondary_genre !== "N/A" ? splitGenre(rec.secondary_genre)[1] : null;
      const sc = rec.genre_scores || [];
      fact("Genre", name, [parent, second && `then ${second}`, sc[0] != null && `p ${sc[0]}`].filter(Boolean).join(", "));
    }
    if (rec.mood === "unavailable") fact("Mood", "Not available", "Mood model not loaded", true);
    else {
      const top = Object.entries(rec.mood_scores || {}).sort((x, y) => y[1] - x[1]).slice(0, 3)
        .map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ");
      fact("Mood", rec.mood, top);
    }
    if (rec.lyric_themes === "unavailable") fact("Lyric themes", "Not available", "Whisper not loaded", true);
    else fact("Lyric themes", rec.lyric_themes === "instrumental" ? "Instrumental" : rec.lyric_themes,
      rec.lyric_themes === "instrumental" ? "No vocals detected" : "Most frequent words, not the transcript");
    const t = rec.timings || {};
    const parts = [["Madmom", t.madmom], ["Essentia", t.essentia], ["Whisper", t.whisper], ["LLM", rec.llm_ms], ["Stream", a.ms]]
      .filter(([, v]) => v).map(([k, v]) => `${k} ${secs(v)}`).join(", ");
    fact("Time taken", secs(rec.analysis_ms + (rec.llm_ms || 0) + (a.ms || 0)), parts);

    // Problems
    const errs = q(".win-errors");
    if (rec.llm_error && !rec.llm_error.startsWith("GROQ_API_KEY") && src !== "held") errs.append(node("li", rec.llm_error, "warn"));
    if (a.error) errs.append(node("li", a.error));
    if (S.mode === "live" && rec.analysis_ms > S.stepSec * 1000) {
      errs.append(node("li", `Analysis took ${secs(rec.analysis_ms)}, longer than the ${S.stepSec} s step, so later windows get skipped.`, "warn"));
    }
    return row;
  }

  function splitGenre(g) {
    const [parent, name] = g.split("---");
    return name ? [parent, name] : ["", parent];
  }

  function select(i, seek) {
    if (i == null || i < 0) return;
    S.selected = i;
    const rec = S.records[i];
    if (rec._row) rec._row.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    if (seek && S.mode === "file" && !el.player.hidden) {
      el.player.currentTime = rec.t_start;
    }
    drawStrip();
  }

  // ---------------------------------------------------------------- strip
  function timeSpan() {
    if (S.mode === "live") return Math.max(60, S.liveElapsed * 1.05);
    return S.duration || 0;
  }

  // The lane shows each window's own step so overlapping windows tile without overlap
  function laneSegment(rec, i) {
    if (S.mode === "live") {
      const first = i === 0 || S.records.slice(0, i).every((r) => r.skipped);
      return [first ? rec.t_start : Math.max(rec.t_start, rec.t_end - S.stepSec), rec.t_end];
    }
    const isLast = rec.index === S.total - 1;
    return [rec.t_start, isLast ? rec.t_end : Math.min(rec.t_end, rec.t_start + S.stepSec)];
  }

  function drawStrip() {
    const c = el.canvas, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const span = timeSpan();
    const laneH = 16, gap = 6, top = 10;
    const envH = H - laneH - gap - top;
    const env = S.envelope;
    const vals = env ? env.values : [];
    let vmax = 0.2;
    for (const v of vals) if (v > vmax) vmax = v;
    vmax *= 1.12;
    const y = (v) => top + envH - (v / vmax) * envH;
    const x = (t) => (span ? (t / span) * W : 0);

    // Phase thresholds
    g.font = "11px Poppins, system-ui, sans-serif";
    g.textBaseline = "bottom";
    for (const th of THRESHOLDS) {
      if (th > vmax) continue;
      const yy = Math.round(y(th)) + 0.5;
      g.strokeStyle = "#26262a"; g.setLineDash([3, 4]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, yy); g.lineTo(W, yy); g.stroke();
      g.setLineDash([]);
      g.fillStyle = "#6e6e73"; g.fillText(th.toFixed(2), 6, yy - 2);
    }

    // Selected window band
    if (S.selected != null && S.records[S.selected] && !S.records[S.selected].skipped) {
      const r = S.records[S.selected];
      g.fillStyle = "rgba(244,244,245,0.06)";
      g.fillRect(x(r.t_start), top - 10, Math.max(1, x(r.t_end) - x(r.t_start)), envH + 10);
    }

    // Envelope, each column tinted by its phase
    if (vals.length && span) {
      const colW = Math.max(1, x(env.dt));
      for (let i = 0; i < vals.length; i++) {
        const v = vals[i], xx = x(i * env.dt);
        if (xx > W) break;
        g.fillStyle = PHASE_COLOR[phaseOf(v)];
        g.globalAlpha = 0.85;
        g.fillRect(xx, y(v), colW + 0.5, top + envH - y(v));
      }
      g.globalAlpha = 1;
    }

    // Window lane: pending (file mode), done, skipped
    const laneY = H - laneH;
    if (S.mode === "file" && S.total && span) {
      for (let i = S.records.length; i < S.total; i++) {
        const t0 = i * S.stepSec;
        const t1 = i === S.total - 1 ? Math.min(span, t0 + S.windowSec) : t0 + S.stepSec;
        g.strokeStyle = "#2c2c2e"; g.lineWidth = 1;
        g.strokeRect(x(t0) + 1.5, laneY + 0.5, Math.max(1, x(t1) - x(t0) - 2), laneH - 1);
      }
    }
    S.records.forEach((r, i) => {
      if (!span) return;
      if (r.skipped) {
        g.fillStyle = "rgba(239,68,68,0.35)";
        g.fillRect(x(r.t_end - S.stepSec) + 1, laneY + laneH / 2 - 1, Math.max(1, x(S.stepSec) - 2), 2);
        return;
      }
      const [t0, t1] = laneSegment(r, i);
      g.fillStyle = PHASE_COLOR[r.phase];
      g.fillRect(x(t0) + 1, laneY, Math.max(1, x(t1) - x(t0) - 2), laneH);
      if (r.api && r.api.ok === false && !r.api.superseded) {
        g.fillStyle = cssVar("--err");
        g.fillRect(x(t0) + 1, laneY, 3, laneH);
      }
      // Tick above the lane where a scene starts
      if (r.scene && SCENE_STARTS.has(r.scene.decision)) {
        g.fillStyle = "#ffffff";
        g.fillRect(x(t0) + 1, laneY - 6, 2, 5);
      }
      if (i === S.selected) {
        g.strokeStyle = "#ffffff"; g.lineWidth = 1.5;
        g.strokeRect(x(t0) + 1.75, laneY + 0.75, Math.max(1, x(t1) - x(t0) - 3.5), laneH - 1.5);
      }
    });

    // Playhead
    if (S.mode === "file" && !el.player.hidden && span) {
      const px = Math.round(x(el.player.currentTime)) + 0.5;
      g.strokeStyle = "#ffffff"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(px, 0); g.lineTo(px, H); g.stroke();
    }
    // Hover line
    if (S.hover != null) {
      g.strokeStyle = "rgba(255,255,255,0.35)"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(S.hover + 0.5, 0); g.lineTo(S.hover + 0.5, H); g.stroke();
    }
  }

  function recordAtTime(t) {
    let hit = -1;
    S.records.forEach((r, i) => {
      if (r.skipped) return;
      const [t0, t1] = laneSegment(r, i);
      if (t >= t0 && t < t1) hit = i;
    });
    return hit;
  }

  el.canvas.addEventListener("mousemove", (e) => {
    const span = timeSpan();
    if (!span) return;
    const rect = el.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left, t = (px / rect.width) * span;
    S.hover = px;
    const env = S.envelope;
    const v = env && env.values[Math.floor(t / env.dt)];
    const i = recordAtTime(t);
    let text = fmt(t);
    if (v != null) text += `, energy ${v.toFixed(2)}`;
    if (i >= 0) text += `, window ${S.records.filter((r, k) => k <= i && !r.skipped).length}`;
    el.tip.textContent = text;
    el.tip.style.left = Math.min(rect.width - 70, Math.max(70, px)) + "px";
    el.tip.hidden = false;
    drawStrip();
  });
  el.canvas.addEventListener("mouseleave", () => { S.hover = null; el.tip.hidden = true; drawStrip(); });
  el.canvas.addEventListener("click", (e) => {
    const span = timeSpan();
    if (!span) return;
    const rect = el.canvas.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * span;
    const i = recordAtTime(t);
    if (i >= 0) select(i, true);
    else if (S.mode === "file" && !el.player.hidden) el.player.currentTime = t;
  });

  // Playback follows along: mark the current window's row
  let raf = null;
  function tick() {
    drawStrip();
    const i = recordAtTime(el.player.currentTime);
    S.records.forEach((r, k) => r._row && r._row.classList.toggle("is-current", k === i));
    raf = el.player.paused ? null : requestAnimationFrame(tick);
  }
  el.player.addEventListener("play", () => { if (!raf) raf = requestAnimationFrame(tick); });
  el.player.addEventListener("seeked", tick);
  el.player.addEventListener("pause", tick);

  new ResizeObserver(() => drawStrip()).observe(el.strip);
  const bar = document.querySelector(".bar"), stripWrap = document.querySelector(".strip-wrap");
  new ResizeObserver(() => {
    const root = document.documentElement.style;
    root.setProperty("--bar-h", bar.offsetHeight + "px");
    root.setProperty("--strip-h", stripWrap.offsetHeight + "px");
  }).observe(document.body);

  // ---------------------------------------------------------------- wiring
  el.file.addEventListener("change", () => { startFile(el.file.files[0]); el.file.value = ""; });
  el.live.addEventListener("click", () => (S.live ? stopLive() : startLive()));
  el.cancel.addEventListener("click", cancelJob);
  el.sceneSend.addEventListener("click", sendScene);
  el.sceneText.addEventListener("input", () => setDirty(el.sceneText.value.trim() !== ((S.scene && S.scene.prompt) || "")));
  el.sceneText.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendScene(); }
  });
  el.sceneRevert.addEventListener("click", () => { setDirty(false); renderScene(S.scene); sceneResult(""); });
  el.sceneLock.addEventListener("change", () => {
    const locked = el.sceneLock.checked, text = el.sceneText.value.trim();
    if (locked && !text) {
      el.sceneLock.checked = false;
      return sceneResult("Type a prompt first, then lock it.", "err");
    }
    // Locking unsent edits sends them too, so what's locked is what's on screen
    if (S.sceneDirty && text) postScene({ prompt: text, locked, send: true });
    else postScene({ locked, send: false });
  });
  el.resetStream.addEventListener("click", async () => {
    await fetch("/api/stream/reset", { method: "POST" }).catch(() => {});
    clearTimeout(healthTimer); refreshHealth();
  });

  let dragDepth = 0;
  addEventListener("dragenter", (e) => { if (S.live) return; e.preventDefault(); dragDepth++; el.drop.hidden = false; });
  addEventListener("dragover", (e) => e.preventDefault());
  addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; el.drop.hidden = true; } });
  addEventListener("drop", (e) => {
    e.preventDefault(); dragDepth = 0; el.drop.hidden = true;
    const f = e.dataTransfer.files[0];
    if (f && !S.live) startFile(f);
  });
  addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    const real = S.records.map((r, i) => (r.skipped ? -1 : i)).filter((i) => i >= 0);
    if (!real.length) return;
    const pos = real.indexOf(S.selected);
    if (e.key === "j") { e.preventDefault(); select(real[Math.min(real.length - 1, pos + 1)], true); }
    if (e.key === "k") { e.preventDefault(); select(real[Math.max(0, pos - 1)], true); }
  });

  refreshHealth();
  listInputs();
  drawStrip();
})();
