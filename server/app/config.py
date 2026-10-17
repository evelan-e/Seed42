"""Everything you may need to change, read from the environment (or .env).

This replaces the notebook's configuration cell and the Colab secret lookup.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # .env at the project root, one level above server/
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv()
except ImportError:  # dotenv is optional
    pass


def _bool(name, default):
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# full = Librosa + Madmom + Essentia + Whisper (the notebook pipeline)
# lite = Librosa only; genre, mood and themes are skipped. For working on the
#        test page or the Control API path without the heavy models.
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "full").strip().lower()
LITE = PIPELINE_MODE == "lite"

SR = 22050                      # analysis sample rate, as in the notebook
ESSENTIA_SR = 16000

MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).resolve().parents[1] / "essentia_models"))
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "tiny")
N_THEMES = int(os.getenv("N_THEMES", "4"))

# LLM (Groq through the OpenAI SDK)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_PROMPT_WORDS = int(os.getenv("MAX_PROMPT_WORDS", "40"))

# seed42 Control API (the mock by default)
CONTROL_API_URL = os.getenv("CONTROL_API_URL", "http://127.0.0.1:8787")
CONTROL_API_EMAIL = os.getenv("CONTROL_API_EMAIL", "test@example.com")
CONTROL_API_PASSWORD = os.getenv("CONTROL_API_PASSWORD", "anything")
CONTROL_API_INITIAL_PROMPT = os.getenv("CONTROL_API_INITIAL_PROMPT", "abstract shapes, vivid, neon")

# Scenes (scene.py). Change scores run from 0 (same music) to 1 (different in every way).
SCENE_EVOLVE_AT = float(os.getenv("SCENE_EVOLVE_AT", "0.25"))    # at or above: adjust the scene
SCENE_CHANGE_AT = float(os.getenv("SCENE_CHANGE_AT", "0.5"))     # at or above: blend, then replace if it lasts
SCENE_GAP_SEC = float(os.getenv("SCENE_GAP_SEC", "1.5"))         # near silence this long suggests a track change
SCENE_SILENCE_RMS = float(os.getenv("SCENE_SILENCE_RMS", "0.01"))  # window energy under this counts as silence

# Defaults the test page starts with
DEFAULT_WINDOW_SEC = float(os.getenv("DEFAULT_WINDOW_SEC", "20"))
DEFAULT_STEP_SEC = float(os.getenv("DEFAULT_STEP_SEC", "10"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
