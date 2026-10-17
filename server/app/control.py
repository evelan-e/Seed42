"""seed42 Control API client: log in, create a stream, send prompts.

Every call reports what happened -- status code, attempts, latency -- instead
of raising, because showing that on the test page is the point.

Note: this matches the mock (PATCH /api/streams). The Control API PDF describes
POST /api/prompts with {stream_id, body}. When you point this at the real API,
check which one it serves and change send_prompt() to match.
"""
import logging
import threading
import time

import requests

from . import config

log = logging.getLogger("pipeline.control")


class Seed42Client:
    def __init__(self, base_url, timeout_seconds=12):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.token = None

    def login(self, email, password):
        r = requests.post(f"{self.base_url}/api/auth",
                          json={"action": "login", "email": email, "password": password},
                          timeout=self.timeout)
        r.raise_for_status()
        self.token = r.json()["token"]
        return self.token

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def create_stream(self, initial_prompt):
        r = requests.post(f"{self.base_url}/api/streams", headers=self._headers(),
                          json={"params": {"model_id": "stabilityai/sdxl-turbo",
                                           "width": 1024, "height": 576, "prompt": initial_prompt}},
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()["id"], r.status_code == 202

    def get_status(self, stream_id):
        r = requests.get(f"{self.base_url}/api/streams", headers=self._headers(),
                         params={"id": stream_id}, timeout=self.timeout)
        if r.status_code == 404:
            return "not_found"
        r.raise_for_status()
        return r.json().get("status")

    def wait_until_ready(self, stream_id, poll_every=2, max_wait=30):
        waited = 0
        while waited < max_wait:
            if self.get_status(stream_id) == "ready":
                return True
            time.sleep(poll_every)
            waited += poll_every
        return False

    def send_prompt(self, stream_id, prompt, max_retries=3, retry_wait=2):
        """Returns {code, attempts, ms, body}. 404/409 mean not ready and are retried."""
        t0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            # Only the prompt -- a hot field. Never send creation-time fields here.
            r = requests.patch(f"{self.base_url}/api/streams", headers=self._headers(),
                               json={"id": stream_id, "params": {"prompt": prompt}},
                               timeout=self.timeout)
            if r.status_code in (404, 409) and attempt <= max_retries:
                time.sleep(retry_wait)
                continue
            try:
                body = r.json()
            except ValueError:
                body = {"raw": r.text[:200]}
            return {"code": r.status_code, "attempts": attempt,
                    "ms": round((time.perf_counter() - t0) * 1000), "body": body}

    def end_stream(self, stream_id):
        try:
            r = requests.delete(f"{self.base_url}/api/streams", headers=self._headers(),
                                json={"stream_id": stream_id}, timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False


HINTS = {
    401: "Not authenticated. Check CONTROL_API_EMAIL / CONTROL_API_PASSWORD, or the mock's --token.",
    404: "Still not ready after retries, or the stream no longer exists. A new stream is created on the next window.",
    409: "The stream never left 'starting' within the retry budget.",
    502: "Upstream error. With the mock, this is --fail-rate or --unstable.",
}


class StreamLink:
    """One stream for the whole server session, created on first use."""

    def __init__(self):
        self.client = Seed42Client(config.CONTROL_API_URL)
        self.stream_id = None
        self.lock = threading.Lock()
        self.last_error = None

    def describe(self):
        return {"url": config.CONTROL_API_URL, "stream_id": self.stream_id, "last_error": self.last_error}

    def _ensure_stream(self):
        if self.stream_id:
            return
        self.client.login(config.CONTROL_API_EMAIL, config.CONTROL_API_PASSWORD)
        sid, needs_poll = self.client.create_stream(config.CONTROL_API_INITIAL_PROMPT)
        if needs_poll and not self.client.wait_until_ready(sid):
            raise RuntimeError(f"stream {sid} was not ready after 30s")
        self.stream_id = sid
        log.info("stream ready: %s", sid)

    def send(self, prompt):
        """Returns {ok, code, attempts, ms, stream_id, error} and never raises."""
        with self.lock:
            t0 = time.perf_counter()
            try:
                self._ensure_stream()
            except requests.ConnectionError:
                self.last_error = f"Can't reach the Control API at {config.CONTROL_API_URL}. Is the mock running?"
                return {"ok": False, "code": None, "attempts": 0, "ms": 0,
                        "stream_id": None, "error": self.last_error}
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                self.last_error = f"Couldn't create a stream: {code}. {HINTS.get(code, '')}".strip()
                return {"ok": False, "code": code, "attempts": 0,
                        "ms": round((time.perf_counter() - t0) * 1000),
                        "stream_id": None, "error": self.last_error}
            except Exception as e:
                self.last_error = f"Couldn't create a stream: {e}"
                return {"ok": False, "code": None, "attempts": 0, "ms": 0,
                        "stream_id": None, "error": self.last_error}

            sid = self.stream_id
            try:
                res = self.client.send_prompt(sid, prompt)
            except requests.RequestException as e:
                self.last_error = f"Request failed: {e}"
                return {"ok": False, "code": None, "attempts": 0, "ms": 0,
                        "stream_id": sid, "error": self.last_error}

            ok = res["code"] == 200
            error = None if ok else (HINTS.get(res["code"]) or f"Unexpected {res['code']}: {res['body']}")
            stream_gone = res["code"] in (401, 404) or (
                res["code"] == 502 and (res["body"] or {}).get("error") == "stream ended")
            if stream_gone:
                self.stream_id = None   # start over on the next window
            self.last_error = error
            return {"ok": ok, "code": res["code"], "attempts": res["attempts"],
                    "ms": res["ms"], "stream_id": sid, "error": error}

    def reset(self):
        with self.lock:
            if self.stream_id:
                self.client.end_stream(self.stream_id)
            self.stream_id = None
            self.last_error = None


LINK = StreamLink()
