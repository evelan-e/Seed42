"""seed42 Control API client: log in, create a stream, send prompts.

Every call reports what happened -- status code, attempts, latency -- instead
of raising, because showing that on the test page is the point.

Follows the client's control-api.md: PATCH /api/streams with {id, params}, and
the request discipline in its section 4 (one request in flight, last write
wins, 12 s timeout, retry only "not ready").
"""
import logging
import threading
import time

import requests

from . import config

log = logging.getLogger("pipeline.control")


def is_not_ready(code, body):
    """The one retryable answer: a 409, or a 404 whose body says not ready.
    Any other 404 means the stream doesn't exist, and retrying won't help."""
    return code == 409 or (code == 404 and "not ready" in str(body).lower())


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

    def send_prompt(self, stream_id, prompt, max_attempts=3, retry_wait=2):
        """Returns {code, attempts, ms, body}. Only "not ready" is retried."""
        t0 = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            # Only the prompt -- a hot field. Never send creation-time fields here.
            r = requests.patch(f"{self.base_url}/api/streams", headers=self._headers(),
                               json={"id": stream_id, "params": {"prompt": prompt}},
                               timeout=self.timeout)
            try:
                body = r.json()
            except ValueError:
                body = {"raw": r.text[:200]}
            if is_not_ready(r.status_code, body) and attempt < max_attempts:
                time.sleep(retry_wait)
                continue
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
    400: "Bad request. Retrying won't help; check what is being sent.",
    401: "Not logged in, usually an expired token. The next prompt logs in again. "
         "If it keeps happening, check CONTROL_API_EMAIL / CONTROL_API_PASSWORD.",
    404: "Still not ready after 3 tries.",
    409: "Still not ready after 3 tries.",
}
NEXT_OWN = "A new stream is created on the next window."
NEXT_GIVEN = ("This app doesn't replace a stream seed42 gave it. "
              "Ask seed42 for the new id and set CONTROL_API_STREAM_ID.")


class StreamLink:
    """One stream for the whole server session.

    With CONTROL_API_STREAM_ID set, that stream is used as given: in a real
    performance seed42's own client creates it, so this never creates or ends
    one. Without it, a stream is created on first use, for development.

    One request is in flight at a time. A prompt that arrives while one is
    being sent waits; if a newer one arrives too, the waiting one is dropped
    (last write wins) and reported as superseded.
    """

    def __init__(self):
        self.client = Seed42Client(config.CONTROL_API_URL)
        self.given_id = config.CONTROL_API_STREAM_ID or None
        self.stream_id = self.given_id
        self.cond = threading.Condition()
        self.busy = False
        self.latest = 0         # ticket of the newest send() call
        self.last_error = None

    def describe(self):
        return {"url": config.CONTROL_API_URL, "stream_id": self.stream_id,
                "given": bool(self.given_id), "last_error": self.last_error}

    def _ensure_stream(self):
        if self.client.token is None:
            self.client.login(config.CONTROL_API_EMAIL, config.CONTROL_API_PASSWORD)
        if self.stream_id:
            return
        sid, needs_poll = self.client.create_stream(config.CONTROL_API_INITIAL_PROMPT)
        if needs_poll and not self.client.wait_until_ready(sid):
            raise RuntimeError(f"stream {sid} was not ready after 30s")
        self.stream_id = sid
        log.info("stream ready: %s", sid)

    def send(self, prompt):
        """Returns {ok, code, attempts, ms, stream_id, error} and never raises."""
        with self.cond:
            self.latest += 1
            ticket = self.latest
            while self.busy:
                self.cond.wait()
            if ticket != self.latest:
                return {"ok": False, "superseded": True, "code": None, "attempts": 0, "ms": 0,
                        "stream_id": self.stream_id,
                        "error": "Replaced by a newer prompt before it was sent."}
            self.busy = True
        try:
            return self._send(prompt)
        finally:
            with self.cond:
                self.busy = False
                self.cond.notify_all()

    def _send(self, prompt):
        t0 = time.perf_counter()
        try:
            self._ensure_stream()
        except requests.ConnectionError:
            self.last_error = f"Can't reach the Control API at {config.CONTROL_API_URL}. Is the mock running?"
            return {"ok": False, "code": None, "attempts": 0, "ms": 0,
                    "stream_id": self.stream_id, "error": self.last_error}
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 401:
                self.client.token = None
            self.last_error = f"Couldn't log in or create a stream: {code}. {HINTS.get(code, '')}".strip()
            return {"ok": False, "code": code, "attempts": 0,
                    "ms": round((time.perf_counter() - t0) * 1000),
                    "stream_id": self.stream_id, "error": self.last_error}
        except Exception as e:
            self.last_error = f"Couldn't log in or create a stream: {e}"
            return {"ok": False, "code": None, "attempts": 0, "ms": 0,
                    "stream_id": self.stream_id, "error": self.last_error}

        sid = self.stream_id
        try:
            res = self.client.send_prompt(sid, prompt)
        except requests.RequestException as e:
            self.last_error = f"Request failed: {e}"
            return {"ok": False, "code": None, "attempts": 0, "ms": 0,
                    "stream_id": sid, "error": self.last_error}

        code, body = res["code"], res["body"]
        ok = code == 200
        log.info("sent %s -> %s x%s (%d words) %r | %s", sid, code, res["attempts"],
                 len(prompt.split()), prompt, str(body)[:160])
        error = None
        if code == 401:
            self.client.token = None    # log in again next time, same stream
            error = HINTS[401]
        elif code == 502 or (code == 404 and not is_not_ready(code, body)):
            # The guide: a 502 means the stream is gone and you need a new one
            what = "The stream has ended" if code == 502 else "The stream doesn't exist"
            if self.given_id:
                error = f"{what} ({code}). {NEXT_GIVEN}"
            else:
                self.stream_id = None
                error = f"{what} ({code}). {NEXT_OWN}"
        elif not ok:
            error = HINTS.get(code) or f"Unexpected {code}: {body}"
        self.last_error = error
        return {"ok": ok, "code": code, "attempts": res["attempts"],
                "ms": res["ms"], "stream_id": sid, "error": error}

    def reset(self):
        """End our own stream so the next prompt creates a new one. A stream
        seed42 gave us is left alone: ending it would stop their show."""
        with self.cond:
            while self.busy:
                self.cond.wait()
            if self.stream_id and not self.given_id:
                self.client.end_stream(self.stream_id)
                self.stream_id = None
            self.last_error = None


LINK = StreamLink()
