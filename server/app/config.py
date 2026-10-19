"""All settings, read once at startup.

Each value comes from an environment variable of the same name, or from the
.env file at the project root (see .env.example), and falls back to the
default shown here. Docker Compose passes .env to the container.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")   # project root
    load_dotenv()                                                # or the current folder
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass


# --- Pipeline mode ---------------------------------------------------------
# full = Librosa + Madmom + Essentia + Whisper
# lite = Librosa only; genre, mood and lyric themes are skipped. Starts in
#        seconds, for working on the page or the Control API path.
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "full").strip().lower()
LITE = PIPELINE_MODE == "lite"

# --- Audio and models ------------------------------------------------------
SR = 22050              # every track is resampled to this rate for analysis
ESSENTIA_SR = 16000     # rate the Essentia and Whisper models expect

MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).resolve().parents[1] / "essentia_models"))
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "tiny")     # tiny, base, small, ...
N_THEMES = int(os.getenv("N_THEMES", "4"))           # lyric keywords kept per window
# --- LLM: writes the image prompt (Groq, through the OpenAI SDK) ------------
# Without a key every window uses the fallback prompt in prompt.py.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_PROMPT_WORDS = int(os.getenv("MAX_PROMPT_WORDS", "40"))

# --- seed42 Control API: receives the prompts --------------------------------
# Defaults point at the mock in mock/. For the real API set the URL
# (https://seed42.app) and your own seed42 email and password.
CONTROL_API_URL = os.getenv("CONTROL_API_URL", "http://127.0.0.1:8787")
CONTROL_API_EMAIL = os.getenv("CONTROL_API_EMAIL", "test@example.com")
CONTROL_API_PASSWORD = os.getenv("CONTROL_API_PASSWORD", "anything")
CONTROL_API_INITIAL_PROMPT = os.getenv("CONTROL_API_INITIAL_PROMPT", "abstract shapes, vivid, neon")
# Empty: create our own stream (development). Set: send to this stream, the one
# seed42 creates for a performance, and never create or end a stream.
CONTROL_API_STREAM_ID = os.getenv("CONTROL_API_STREAM_ID", "").strip()

# --- Scenes (scene.py) -------------------------------------------------------
# Each window gets a change score from 0 (same music) to 1 (different in every way).
SCENE_EVOLVE_AT = float(os.getenv("SCENE_EVOLVE_AT", "0.25"))      # at or above: adjust the scene
SCENE_CHANGE_AT = float(os.getenv("SCENE_CHANGE_AT", "0.5"))       # at or above: blend, then replace if it lasts
SCENE_GAP_SEC = float(os.getenv("SCENE_GAP_SEC", "1.5"))           # near silence this long suggests a track change
SCENE_SILENCE_RMS = float(os.getenv("SCENE_SILENCE_RMS", "0.01"))  # window energy under this counts as silence

# --- Test page ---------------------------------------------------------------
DEFAULT_WINDOW_SEC = float(os.getenv("DEFAULT_WINDOW_SEC", "20"))  # what the window and step boxes start at
DEFAULT_STEP_SEC = float(os.getenv("DEFAULT_STEP_SEC", "10"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
