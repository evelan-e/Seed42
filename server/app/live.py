"""Live input over a WebSocket: microphone audio in, one result per window out.

The page sends mono Float32 PCM at 22050 Hz in ~0.5 s chunks. The server keeps
a rolling buffer one window long, and every `step` seconds runs that window
through the same pipeline as uploaded files.

If the previous window is still being analysed when the next one is due, the
new one is skipped and the page is told; the current scene's prompt is
re-sent anyway so the stream always has one. On CPU a 20 s window takes several
seconds, so a short step will skip windows -- that's the measurement you want
before tuning window and step for real live use.

Protocol
  page -> server  text   {"type":"start", window_sec, step_sec, use_llm, send_to_stream}
  page -> server  binary Float32 samples
  server -> page  {"type":"started"} | {"type":"window", ...record}
                  {"type":"skipped", t_end, reason, resent} | {"type":"error", message}
"""
import asyncio
import json
import logging

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from . import config
from .control import LINK
from .jobs import run_window
from .models import MODELS
from .scene import SCENE

log = logging.getLogger("pipeline.live")


async def live_session(ws: WebSocket):
    await ws.accept()
    opts = None
    buf = np.zeros(0, dtype=np.float32)
    received = 0            # total samples since start
    next_due = None         # sample count at which the next window is due
    busy = False
    index = 0
    tasks = set()

    async def send(msg):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            pass

    async def analyse(y, i, t_start, t_end):
        nonlocal busy
        try:
            rec = await asyncio.to_thread(run_window, y, i, t_start, t_end,
                                          opts["use_llm"], opts["send_to_stream"])
            rec["live"] = True
            await send({"type": "window", **rec})
        except Exception as e:
            log.exception("live window failed")
            await send({"type": "error", "message": f"Window {i + 1} failed: {e}"})
        finally:
            busy = False

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    window_sec = max(2.0, float(data.get("window_sec", config.DEFAULT_WINDOW_SEC)))
                    step_sec = max(1.0, float(data.get("step_sec", config.DEFAULT_STEP_SEC)))
                    opts = {"window": int(window_sec * config.SR), "step": int(step_sec * config.SR),
                            "use_llm": bool(data.get("use_llm", True)),
                            "send_to_stream": bool(data.get("send_to_stream", True))}
                    next_due = opts["window"]
                    SCENE.start_track("live input")
                    await send({"type": "started", "sample_rate": config.SR,
                                "models_ready": MODELS.ready.is_set()})
                continue

            chunk = msg.get("bytes")
            if not chunk or opts is None:
                continue
            samples = np.frombuffer(chunk, dtype=np.float32)
            received += len(samples)
            buf = np.concatenate([buf, samples])[-opts["window"]:]

            if received >= next_due:
                t_end = received / config.SR
                t_start = max(0.0, t_end - len(buf) / config.SR)
                next_due += opts["step"]
                if busy or not MODELS.ready.is_set():
                    reason = "previous window still running" if busy else "models still loading"
                    # Re-send the current scene so the stream isn't left without a prompt
                    prompt = SCENE.current_prompt() if opts["send_to_stream"] else None
                    if prompt:
                        keep = asyncio.create_task(asyncio.to_thread(LINK.send, prompt))
                        tasks.add(keep)
                        keep.add_done_callback(tasks.discard)
                    await send({"type": "skipped", "t_start": round(t_start, 2),
                                "t_end": round(t_end, 2), "reason": reason, "resent": bool(prompt)})
                    continue
                busy = True
                task = asyncio.create_task(analyse(buf.copy(), index, t_start, t_end))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                index += 1
    except WebSocketDisconnect:
        pass
    finally:
        log.info("live session closed after %.1fs", received / config.SR)
