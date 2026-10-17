"""Local test server for the seed42 audio -> prompt pipeline.

    uvicorn app.main:app --port 8000      (from server/)
    open http://localhost:8000

Endpoints match Part 1.2 of the implementation plan, under /api:
  POST   /api/analyze          upload a track, returns {id}
  GET    /api/analyze/{id}     status + results; ?since=N returns only new windows
  DELETE /api/analyze/{id}     stop a job
  WS     /api/live             live mic input
  GET    /api/health           model, LLM and Control API status
  POST   /api/stream/reset     end the current stream so the next window makes a new one
  GET    /api/scene            the current scene
  POST   /api/scene            set the prompt by hand and/or lock it: {prompt?, locked?, send}
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, jobs
from .control import LINK
from .live import live_session
from .models import MODELS
from .prompt import llm_enabled
from .scene import SCENE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    MODELS.load_in_background()   # server answers immediately; the page shows load progress
    jobs.start_worker()
    yield
    LINK.reset()


app = FastAPI(title="seed42 pipeline test", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {
        "mode": config.PIPELINE_MODE,
        "device": MODELS.device,
        "models_ready": MODELS.ready.is_set(),
        "models": MODELS.status,
        "llm": {"enabled": llm_enabled(), "model": config.LLM_MODEL if llm_enabled() else None},
        "control_api": LINK.describe(),
        "scene": SCENE.describe(),
        "defaults": {"window_sec": config.DEFAULT_WINDOW_SEC, "step_sec": config.DEFAULT_STEP_SEC},
    }


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...),
            window_sec: float = Form(config.DEFAULT_WINDOW_SEC),
            step_sec: float = Form(config.DEFAULT_STEP_SEC),
            use_llm: bool = Form(True),
            send_to_stream: bool = Form(True)):
    if window_sec < 2 or step_sec < 1:
        raise HTTPException(400, "Window must be at least 2 s and step at least 1 s.")
    try:
        path = jobs.save_upload(file.file, file.filename)
    except ValueError as e:
        raise HTTPException(413, str(e))
    job = jobs.submit(jobs.Job(file.filename or "upload", path, window_sec, step_sec,
                               use_llm, send_to_stream))
    return {"id": job.id}


@app.get("/api/analyze/{job_id}")
def analyze_status(job_id: str, since: int = 0):
    job = jobs.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No job with that id. The server may have restarted.")
    return job.view(since)


@app.delete("/api/analyze/{job_id}")
def analyze_cancel(job_id: str):
    job = jobs.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "No job with that id.")
    job.cancelled = True
    return {"ok": True}


@app.post("/api/stream/reset")
def stream_reset():
    LINK.reset()
    return {"ok": True}


class SceneEdit(BaseModel):
    prompt: str | None = None      # a new prompt typed on the page; omit to change only the lock
    locked: bool | None = None
    send: bool = True              # send the prompt to the stream now


@app.get("/api/scene")
def scene_state():
    return SCENE.describe()


@app.post("/api/scene")
def scene_edit(edit: SceneEdit):
    prompt = " ".join(edit.prompt.split()) if edit.prompt is not None else None
    if prompt == "":
        raise HTTPException(400, "The prompt is empty.")
    if edit.locked and prompt is None and SCENE.current_prompt() is None:
        raise HTTPException(400, "There's no scene to lock yet. Type a prompt first.")
    state = SCENE.set_manual(prompt, edit.locked)
    api = LINK.send(prompt) if prompt and edit.send else None
    return {"scene": state, "api": api}


@app.websocket("/api/live")
async def live(ws: WebSocket):
    await live_session(ws)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
