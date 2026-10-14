# Local Audio Analyser

jcc heree see me??

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

- **Lyric themes:** select *Extract lyric themes*. This needs Faster-Whisper
  and downloads its `tiny` model on first use. Install it with
  `python -m pip install -r requirements-models.txt`.
- **Groq prompt generation:** enter a Groq API key in the sidebar or set
  `GROQ_API_KEY` before launching. The key is not saved by the app. Without a
  key, the app creates sound-aware local prompts using each window's tempo,
  energy, spectral brightness, and dominant frequency band.

MP3 and some other formats may need FFmpeg available on `PATH`. WAV files can
be used to test the app without installing FFmpeg.