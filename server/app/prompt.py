"""Audio description -> short SDXL-Turbo prompt, written by an LLM on Groq.

Needs GROQ_API_KEY. Without a key the pipeline still runs and uses
FALLBACK_PROMPT, so the Control API path can be tested anyway.
"""
import logging
import time

from . import config

log = logging.getLogger("pipeline.llm")

FALLBACK_PROMPT = "glowing jellyfish drifting in dark water, neon haze, slow camera drift"

SYSTEM_PROMPT = (
    "You write prompts for a real-time SDXL-Turbo image stream. "
    "Turn the audio cue into ONE line of 8 to 12 short visual phrases separated by commas, "
    f"{config.MAX_PROMPT_WORDS} words maximum.\n"
    "Order: one to three imaginative subjects first, then their setting, surface materials, "
    "lighting colours, atmosphere (fog, particles, air movement), and camera motion last.\n"
    "The subjects are surprising things that carry the mood: creatures, animals, alien "
    "lifeforms, plants or coral, surreal objects or machines, landscape features or cosmic bodies. "
    "Mix kinds when there is more than one, and show how they relate (orbiting, growing from, "
    "chasing, carried by). Translate the music into their look and movement; never picture "
    "the music itself.\n"
    "RULES: no full sentences, no markdown, no quotes, no numbers or BPM, "
    "no genre names, no words or text to appear in the image, "
    "no people, crowds, dancers, figures, faces or bodies, "
    "no clubs, lounges, dance floors, stages, speakers, subwoofers or audio visualizers."
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


def generate_visual_prompt(llm_input, previous=None):
    """A fresh scene. Returns {prompt, source, ms, error}; source is 'llm', 'fallback' or 'empty-reply'.
    With `previous`, the new scene is asked to use different subjects and setting."""
    user = f"Audio cue: {llm_input}"
    if previous:
        user += (f"\nPrevious scene: {previous}\n"
                 "Write a new scene with different subjects and a different setting.")
    return _complete(SYSTEM_PROMPT, user, 0.7)


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
            max_tokens=1000,                          # gpt-oss counts its reasoning in this budget
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
