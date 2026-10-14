# Local Audio Analyser

A local Streamlit app for uploading a song, analysing it in 30- or 60-second
windows, and generating a short visual prompt for each window. It does not
connect to or send prompts to the seed42 Control API.

## Run on Windows

1. Install Python 3.10 or newer.
2. Open PowerShell in this folder and create an environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

Streamlit opens the local app in your browser. Upload a song and choose
**Generate prompts**. The app analyses the song and displays one visual prompt
per window, with a JSON download.

## Optional models and LLM

- **Genre and mood:** select *Classify genre and mood* in the sidebar. The
  first run downloads Essentia's Discogs and mood models into
  `essentia_models/`. Essentia's TensorFlow package may require a supported
  Python/platform combination; if it cannot be installed on your Windows
  setup, run the app in WSL or leave this option off.
- **Lyric themes:** select *Extract lyric themes*. This needs Faster-Whisper
  and downloads its `tiny` model on first use.
- To install both optional model packages, try
  `python -m pip install -r requirements-models.txt`. Essentia may not install
  on every Windows/Python combination; if so, use WSL with a supported Python
  version or leave genre/mood classification off.
- **Groq prompt generation:** enter a Groq API key in the sidebar or set
  `GROQ_API_KEY` before launching. The key is not saved by the app. Without a
  key, the app creates sound-aware local prompts using each window's tempo,
  energy, spectral brightness, and dominant frequency band.

MP3 and some other formats may need FFmpeg available on `PATH`. WAV files can
be used to test the app without installing FFmpeg.
