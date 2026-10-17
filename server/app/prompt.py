"""Audio description -> short SDXL-Turbo prompt (notebook section 5.1).

The Colab secret is replaced by GROQ_API_KEY. Without a key the pipeline still
runs and uses FALLBACK_PROMPT, so the Control API path can be tested anyway.
"""
import logging
import time

from . import config

log = logging.getLogger("pipeline.llm")

FALLBACK_PROMPT = "dark club space, sub-bass pulse, neon haze, slow camera drift"

SYSTEM_PROMPT = (
    "You write prompts for a real-time SDXL-Turbo image stream. "
    "Turn the audio cue into ONE line of 8 to 12 short visual phrases separated by commas, "
    f"{config.MAX_PROMPT_WORDS} words maximum.\n"
    "Order: main scene first, then surface materials, lighting colours, atmosphere "
    "(fog, particles, air movement), glowing elements, and camera motion last.\n"
    "RULES: no full sentences, no markdown, no quotes, no numbers or BPM, "
    "no genre names, no words or text to appear in the image."
)

_client = None
if config.GROQ_API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.GROQ_API_KEY)
        log.info("Groq client ready (model: %s)", config.LLM_MODEL)
    except Exception as e:
        log.warning("Groq client failed to start: %s", e)


def llm_enabled():
    return _client is not None


def trim_prompt(text, max_words=config.MAX_PROMPT_WORDS):
    """Cleans formatting and cuts to `max_words`, ending on a complete phrase."""
    text = (text.replace("\n", ", ").replace("**", "")
                .replace('"', "").replace("\u2011", "-"))
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        if "," in text:
            text = text.rsplit(",", 1)[0]
    return text.strip(" ,.")


EVOLVE_PROMPT = SYSTEM_PROMPT + (
    "\nYou are revising the scene already on screen, not replacing it. Keep its main scene "
    "and most of its phrases. Change at most three phrases, preferring lighting colours, "
    "atmosphere, glowing elements and camera motion, so it fits the new audio cue. "
    "Return the whole revised line."
)

BLEND_PROMPT = SYSTEM_PROMPT + (
    "\nThe music is changing. Write a transitional line halfway between the current scene "
    "and the new audio cue: keep the setting of the current scene, and bring in the "
    "lighting, atmosphere and motion the new cue suggests, so the image can morph toward "
    "a new scene."
)


def generate_visual_prompt(llm_input):
    """A fresh scene. Returns {prompt, source, ms, error}; source is 'llm', 'fallback' or 'empty-reply'."""
    return _complete(SYSTEM_PROMPT, f"Audio cue: {llm_input}", 0.7)


def evolve_visual_prompt(current, llm_input):
    """The current scene, adjusted a little for the new audio."""
    return _complete(EVOLVE_PROMPT, f"Current scene: {current}\nNew audio cue: {llm_input}", 0.4)


def blend_visual_prompt(current, llm_input):
    """Halfway between the current scene and the new audio."""
    return _complete(BLEND_PROMPT, f"Current scene: {current}\nNew audio cue: {llm_input}", 0.6)


def _complete(system, user, temperature):
    if _client is None:
        return {"prompt": FALLBACK_PROMPT, "source": "fallback", "ms": 0,
                "error": "GROQ_API_KEY is not set, so the fallback prompt was used."}
    t = time.perf_counter()
    try:
        reply = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=400,                           # gpt-oss counts its reasoning in this budget
            extra_body={"reasoning_effort": "low"},   # keeps reasoning short so the answer isn't cut off
        )
        text = (reply.choices[0].message.content or "").strip()
        ms = round((time.perf_counter() - t) * 1000)
        if not text:
            return {"prompt": FALLBACK_PROMPT, "source": "empty-reply", "ms": ms,
                    "error": "The LLM returned an empty reply, so the fallback prompt was used."}
        return {"prompt": trim_prompt(text), "source": "llm", "ms": ms, "error": None}
    except Exception as e:
        return {"prompt": FALLBACK_PROMPT, "source": "fallback",
                "ms": round((time.perf_counter() - t) * 1000),
                "error": f"LLM call failed: {e}"}
