#!/usr/bin/env python3
"""Mock of the seed42 Control API (/api/streams), for integration development.

Live streams are not a good build target: they cost real time to start, and an
unreliable one is indistinguishable from a bug in your own client. This
reproduces the contract in control-api.md closely enough to build the
whole request lifecycle against, on a laptop, for free.

    python mock-control-api.py
    # http://127.0.0.1:8787   token=test-token

Deliberately unfriendly where the real API is unfriendly, because a mock that
only serves the happy path teaches a client to crash in production:

  * a missing or wrong token is a real 401, not a warning
  * create is followed by a window of not-ready, so the retry loop is exercised
    rather than skipped
  * --not-ready 409 answers the other legal not-ready code, so a client that
    only handles 404 fails here rather than later
  * a PATCH carrying a creation-time field is answered 200, exactly as the real
    API would, and warns on stdout. That is the failure this file most exists
    to catch: it is invisible in the response, and in production it reloads the
    pipeline and stops the show for ~30 seconds.

Standard library only, on purpose: nobody should have to install anything to
test against this.
"""
import argparse
import json
import random
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Fields fixed when the stream is created. Sending one in a PATCH forces a
# pipeline reload on the real API. Mirrors control-api.md section 3.
COLD_FIELDS = {
    "model_id", "width", "height", "acceleration",
    "lora_dict", "use_lcm_lora", "filters",
}
COLD_CONTROLNET_FIELDS = {"model_id", "preprocessor", "preprocessor_params"}

STREAMS = {}
LOCK = threading.Lock()
CFG = None


def now():
    return time.monotonic()


class Stream:
    """One mock stream. ready_at is when it stops answering not-ready;
    dies_at is set only under --unstable."""

    def __init__(self, params):
        self.id = "str_" + secrets.token_hex(8)
        self.params = dict(params or {})
        self.created = now()
        self.ready_at = self.created + CFG.ready_after
        self.dies_at = self.created + 45 if CFG.unstable else None
        self.patches = 0

    @property
    def state(self):
        if self.dies_at and now() >= self.dies_at:
            return "dead"
        return "ready" if now() >= self.ready_at else "starting"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------
    def log_message(self, fmt, *args):
        print("  %s" % (fmt % args))

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return None

    def _authed(self):
        """The real endpoint gates every verb but OPTIONS. Same here: a client
        that never sees a 401 in development meets its first one live."""
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and header[7:] == CFG.token:
            return True
        self._send(401, {"error": "Not authenticated"})
        return False

    def _maybe_fail(self):
        if CFG.fail_rate and random.random() < CFG.fail_rate:
            self._send(502, {"error": "Upstream error",
                             "details": "injected by --fail-rate"})
            return True
        return False

    def _login(self):
        body = self._body()
        if body is None:
            return self._send(400, {"error": "Invalid JSON"})
        if body.get("action") != "login":
            return self._send(400, {"error": "Unknown action"})
        if not body.get("email") or not body.get("password"):
            return self._send(400, {"error": "email and password required"})
        print("  login %s -> token" % body["email"])
        return self._send(200, {"ok": True, "token": CFG.token})

    def _not_ready(self):
        """The one retryable case. Both codes are legal; a client must treat
        them alike, so the mock can be told to answer either."""
        if CFG.not_ready == 409:
            return self._send(409, {"error": "Stream is not running", "status": "starting"})
        return self._send(404, {"error": "stream not ready"})

    # -- verbs ------------------------------------------------------------
    def do_OPTIONS(self):
        self._send(200, {})

    def do_GET(self):
        if not self._authed():
            return
        # Only meaningful after a 202. A client that polls unconditionally
        # should discover that here rather than in production.
        if not CFG.async_create:
            return self._send(404, {"error": "No status endpoint for this stream"})
        sid = (parse_qs(urlparse(self.path).query).get("id") or [None])[0]
        if not sid:
            return self._send(400, {"error": "id is required"})
        with LOCK:
            s = STREAMS.get(sid)
        if not s:
            return self._send(404, {"error": "Stream not found"})
        if s.state == "dead":
            return self._send(502, {"id": sid, "status": "failed",
                                    "error": "stream ended before it was ready"})
        if s.state == "ready":
            return self._send(200, {"id": sid, "status": "ready"})
        return self._send(200, {"id": sid, "status": "starting"})

    def do_POST(self):
        # /api/auth is the one call that carries no token, because it is how
        # you get one. Routed by path so a client written against the real
        # API -- login first, then streams -- runs here unchanged. Any
        # non-empty email and password are accepted; the answer is the
        # mock's own token.
        if urlparse(self.path).path.rstrip("/").endswith("/api/auth"):
            return self._login()
        if not self._authed():
            return
        if self._maybe_fail():
            return
        body = self._body()
        if body is None:
            return self._send(400, {"error": "Invalid JSON"})
        s = Stream(body.get("params"))
        with LOCK:
            STREAMS[s.id] = s
        print("  created %s (ready in %.0fs%s)"
              % (s.id, CFG.ready_after, ", ends at 45s" if s.dies_at else ""))
        if CFG.async_create:
            return self._send(202, {"id": s.id, "status": "starting"})
        return self._send(201, {
            "id": s.id,
            "stream_key": "stk_" + secrets.token_hex(8),
            "output_stream_url": "rtmp://mock.local/live/" + secrets.token_hex(4),
            "params": s.params,
        })

    def do_PATCH(self):
        if not self._authed():
            return
        if self._maybe_fail():
            return
        body = self._body()
        if body is None:
            return self._send(400, {"error": "Invalid JSON"})
        sid = body.get("id")
        if not sid:
            return self._send(400, {"error": "Stream id is required for PATCH"})
        with LOCK:
            s = STREAMS.get(sid)
        if not s:
            return self._send(404, {"error": "Stream not found"})
        if s.state == "dead":
            return self._send(502, {"error": "stream ended"})
        if s.state == "starting":
            return self._not_ready()

        params = body.get("params") or {}
        self._warn_cold(params)
        s.patches += 1
        s.params.update(params)
        print("  patch #%d on %s: %s" % (s.patches, sid, _summarise(params)))
        return self._send(200, {"ok": True, "id": sid})

    def do_DELETE(self):
        if not self._authed():
            return
        body = self._body()
        if body is None:
            return self._send(400, {"error": "Invalid JSON"})
        sid = (body or {}).get("stream_id") or (body or {}).get("id")
        if not sid:
            return self._send(400, {"error": "Stream id is required for DELETE"})
        with LOCK:
            existed = STREAMS.pop(sid, None) is not None
        if CFG.unstable:
            # DELETE is allowed to fail. Clients must tolerate it, not retry it.
            return self._send(502, {"error": "Upstream error"})
        print("  deleted %s%s" % (sid, "" if existed else " (unknown)"))
        return self._send(200, {"ok": True, "id": sid})

    # -- the check the real API will not do for you -----------------------
    def _warn_cold(self, params):
        hits = sorted(COLD_FIELDS.intersection(params))
        for cn in params.get("controlnets") or []:
            if isinstance(cn, dict):
                hits += ["controlnets[].%s" % f
                         for f in sorted(COLD_CONTROLNET_FIELDS.intersection(cn))]
        if hits:
            print("\n  * COLD FIELD IN PATCH: %s" % ", ".join(hits))
            print("      Production would reload the pipeline (~30s) and the")
            print("      performance would stop. Send only hot fields.")
            print("      See control-api.md section 3.\n")


def _summarise(params):
    bits = []
    if "prompt" in params:
        p = str(params["prompt"])
        bits.append("prompt=%r" % (p[:40] + "..." if len(p) > 40 else p))
    for k in ("t_index_list", "guidance_scale", "delta", "seed", "num_inference_steps"):
        if k in params:
            bits.append("%s=%s" % (k, params[k]))
    cns = params.get("controlnets")
    if isinstance(cns, list):
        bits.append("cn=[%s]" % ",".join(
            str(c.get("conditioning_scale")) for c in cns if isinstance(c, dict)))
    return " ".join(bits) or "(empty)"


def main():
    global CFG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; use 0.0.0.0 inside Docker")
    ap.add_argument("--token", default="test-token")
    ap.add_argument("--not-ready", type=int, choices=(404, 409), default=404,
                    dest="not_ready",
                    help="which not-ready code to answer (both are legal; your "
                         "client must handle either)")
    ap.add_argument("--async-create", action="store_true", dest="async_create",
                    help="answer 202 on create and require the GET poll")
    ap.add_argument("--ready-after", type=float, default=4.0,
                    help="seconds a new stream answers not-ready (default 4)")
    ap.add_argument("--always-ready", action="store_true",
                    help="skip the not-ready window entirely")
    ap.add_argument("--unstable", action="store_true",
                    help="streams end ~45s after creation and DELETE fails")
    ap.add_argument("--fail-rate", type=float, default=0.0,
                    help="probability of a 502 on create/patch, e.g. 0.2")
    CFG = ap.parse_args()
    if CFG.always_ready:
        CFG.ready_after = 0.0

    # Line-buffer stdout. Python block-buffers when it is not a terminal, so
    # piping this to a file or running it from an IDE swallowed every warning
    # until the process exited -- including the cold-field warning, which is
    # the main reason to run this at all.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover - Python < 3.7
        pass

    srv = ThreadingHTTPServer((CFG.host, CFG.port), Handler)
    print("seed42 Control API mock")
    print("  http://127.0.0.1:%d   token=%s" % (CFG.port, CFG.token))
    print("  Authorization: Bearer %s" % CFG.token)
    print("  not-ready=%d%s%s"
          % (CFG.not_ready,
             "  async-create" if CFG.async_create else "",
             "  unstable" if CFG.unstable else ""))
    print("  contract: control-api.md      ctrl-c to stop\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()