# Gujarati Audio Factory

End-to-end local app: upload a PDF/TXT or paste textbook content, enter a question, and receive a source-grounded Gujarati two-host WAV audio plus the generated transcript.

It uses Gemini's document processing API for PDFs, a Gemini text model for the conversational script, and Gemini multi-speaker TTS for the audio. This is the code-driven equivalent of the NotebookLM workflow; it does not automate NotebookLM's private UI.

## Setup

1. Install Python 3.11+.
2. In this folder, create `.env` by copying `.env.example`.
3. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey), then place it in `.env` as `GEMINI_API_KEY`.
4. Run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

5. Open `http://127.0.0.1:8000`.

## Use

1. Upload a textbook PDF or TXT, or paste the textbook answer/chapter content.
2. Enter the exact question/topic to explain.
3. Choose **Short**, **Default**, or **Full chapter / Maximum**. Maximum covers all useful supplied-source material without artificially shortening it; it does not pad or invent content.
4. Click **Generate audio**, listen on the results page, and download the shareable `audio_overview.mp3` or `transcript.txt`.

## Notes

- Gemini TTS models are preview models; keep model names configurable in `.env` because Google may rename them.
- Gemini files are stored by the API for 48 hours, per the official Files API documentation.
- The app intentionally does not store the API key in source code. Keep `.env` private.
- The app creates both MP3 and WAV automatically. Gemini capacity errors are retried automatically before a job is marked failed.

## Architecture

`PDF/TXT/pasted content -> Gemini document understanding -> Gujarati two-host transcript -> Gemini multi-speaker TTS -> WAV + transcript`
