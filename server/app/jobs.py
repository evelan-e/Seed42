"""Window pipeline plus the background worker for uploaded files.

run_window() is the notebook's one-minute engine for a single window:
features -> input sentence -> scene decision (scene.py) -> Control API.
The scene decides whether the LLM is called at all; a held scene re-sends
the same prompt.
File jobs and live sessions both use it, behind one lock, because the models
are shared and not safe to call from two threads at once.
"""
import logging
import os
import queue
import tempfile
import threading
import time
import uuid

import librosa

from . import config
from .control import LINK
from .features import analyse_window, energy_envelope
from .models import MODELS
from .scene import SCENE, find_gap

log = logging.getLogger("pipeline.jobs")

ANALYSIS_LOCK = threading.Lock()


def run_window(y_slice, index, t_start, t_end, use_llm=True, send_to_stream=True):
    """One window through the whole pipeline. Returns the record the page renders."""
    with ANALYSIS_LOCK:
        rec = analyse_window(y_slice, sr=config.SR)
    gap = find_gap(y_slice)

    rec.update({"index": index, "t_start": round(t_start, 2), "t_end": round(t_end, 2)})

    # The scene decides whether this window holds, adjusts, blends or replaces the prompt
    SCENE.step(rec, gap, use_llm)
    rec["prompt_words"] = len(rec["prompt"].split()) if rec["prompt"] else 0

    if send_to_stream and rec["prompt"]:
        rec["api"] = LINK.send(rec["prompt"])
    else:
        rec["api"] = {"ok": None, "code": None, "attempts": 0, "ms": 0,
                      "stream_id": None, "error": None, "skipped": True}
    return rec


class Job:
    def __init__(self, filename, path, window_sec, step_sec, use_llm, send_to_stream):
        self.id = uuid.uuid4().hex[:12]
        self.filename = filename
        self.path = path
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.use_llm = use_llm
        self.send_to_stream = send_to_stream
        self.status = "queued"          # queued -> decoding -> waiting-for-models -> running -> done | failed | cancelled
        self.error = None
        self.duration = None
        self.envelope = None
        self.total_windows = None
        self.results = []
        self.cancelled = False
        self.created = time.time()

    def view(self, since=0):
        return {
            "id": self.id, "filename": self.filename, "status": self.status, "error": self.error,
            "duration": self.duration, "window_sec": self.window_sec, "step_sec": self.step_sec,
            "use_llm": self.use_llm, "send_to_stream": self.send_to_stream,
            "total_windows": self.total_windows, "done_windows": len(self.results),
            "envelope": self.envelope if since == 0 else None,
            "results": self.results[since:],
        }


JOBS = {}
_queue = queue.Queue()


def submit(job):
    JOBS[job.id] = job
    _queue.put(job.id)
    # Keep memory bounded: forget all but the 20 newest jobs
    for old in sorted(JOBS.values(), key=lambda j: j.created)[:-20]:
        JOBS.pop(old.id, None)
    return job


def window_starts(n_samples, window, step):
    """Window start offsets. A track shorter than one window is analysed whole."""
    if n_samples <= window:
        return [0]
    return list(range(0, n_samples - window + 1, step))


def _process(job):
    try:
        job.status = "decoding"
        y, sr = librosa.load(job.path, sr=config.SR, mono=True)
        job.duration = round(len(y) / sr, 2)
        job.envelope = energy_envelope(y, sr)
        window, step = int(job.window_sec * sr), int(job.step_sec * sr)
        starts = window_starts(len(y), window, step)
        job.total_windows = len(starts)

        if not MODELS.ready.is_set():
            job.status = "waiting-for-models"
            MODELS.ready.wait()

        job.status = "running"
        SCENE.start_track(job.filename)
        for i, start in enumerate(starts):
            if job.cancelled:
                job.status = "cancelled"
                return
            end = min(start + window, len(y))
            rec = run_window(y[start:end], i, start / sr, end / sr,
                             use_llm=job.use_llm, send_to_stream=job.send_to_stream)
            job.results.append(rec)
        job.status = "done"
    except Exception as e:
        log.exception("job %s failed", job.id)
        job.status = "failed"
        job.error = f"Couldn't analyse {job.filename}: {e}"
    finally:
        try:
            os.unlink(job.path)
        except OSError:
            pass


def _worker():
    while True:
        job_id = _queue.get()
        job = JOBS.get(job_id)
        if job and not job.cancelled:
            _process(job)
        elif job:
            job.status = "cancelled"


def start_worker():
    threading.Thread(target=_worker, name="job-worker", daemon=True).start()


def save_upload(fileobj, filename):
    suffix = os.path.splitext(filename or "")[1] or ".audio"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="seed42-")
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with os.fdopen(fd, "wb") as out:
        while chunk := fileobj.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                os.unlink(path)
                raise ValueError(f"File is larger than {config.MAX_UPLOAD_MB} MB (MAX_UPLOAD_MB).")
            out.write(chunk)
    return path
