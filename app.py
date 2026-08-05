"""PDF/text -> Gujarati two-host teaching audio.

Run: uvicorn app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import re
import shutil
import time
import uuid
import wave
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
import lameenc

load_dotenv()
ROOT = Path(__file__).parent
JOBS = ROOT / "jobs"
JOBS.mkdir(exist_ok=True)
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.6-flash")
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
HOST_A_VOICE = os.getenv("HOST_A_VOICE", "Aoede")
HOST_B_VOICE = os.getenv("HOST_B_VOICE", "Leda")

app = FastAPI(title="Gujarati Audio Factory")
app.mount("/jobs", StaticFiles(directory=JOBS), name="jobs")

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Gujarati Audio Factory</title>
<style>body{font-family:system-ui;background:#101827;color:#e8eefb;max-width:850px;margin:40px auto;padding:0 20px}textarea,input,select,button{width:100%;box-sizing:border-box;margin:8px 0;padding:12px;border-radius:8px;border:1px solid #40506b;background:#162236;color:white}button{background:#5b7cfa;border:0;font-weight:700;cursor:pointer}small{color:#aab8d0}.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}</style></head><body>
<h1>Gujarati Audio Factory</h1><p>Upload a textbook PDF or paste content. The app creates a source-grounded, natural two-host Gujarati teaching audio and its transcript.</p>
<form action=/generate method=post enctype=multipart/form-data><label>PDF or TXT source (optional)</label><input type=file name=file accept=".pdf,.txt,.md">
<label>Or paste textbook content (optional)</label><textarea name=content rows=8 placeholder="Paste the question and answer, or chapter content here"></textarea>
<label>Question or topic to explain</label><input name=question required placeholder="e.g., પ્રશ્ન 9: સંતાનમાં બે વિકલ્પો શા માટે હોય છે?"><div class=row><label>Length<select name=length><option value=default>Default (about 6-9 min)</option><option value=short>Short (about 3-4 min)</option><option value=maximum>Full chapter / Maximum (source-aware)</option></select></label><label>Output name<input name=title value="gujarati_audio_overview"></label></div>
<button type=submit>Generate audio</button></form><p><small>Requires GEMINI_API_KEY in .env. Generation can take several minutes. Do not close the request page.</small></p></body></html>"""

def client() -> genai.Client:
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "replace_me":
        raise HTTPException(400, "Set GEMINI_API_KEY in .env before generating audio.")
    return genai.Client()

def wait_for_file(c: genai.Client, f):
    for _ in range(120):
        info = c.files.get(name=f.name)
        state = str(getattr(info, "state", ""))
        if "PROCESSING" not in state:
            if "FAILED" in state:
                raise RuntimeError("Gemini could not process this document.")
            return info
        time.sleep(1)
    raise RuntimeError("Document processing timed out.")

def with_retry(action, label: str, attempts: int = 4):
    """Absorb temporary Gemini 429/503 capacity errors during longer jobs."""
    last_error = None
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            transient = any(token in str(exc) for token in ("429", "500", "502", "503", "504", "UNAVAILABLE"))
            if not transient or attempt == attempts - 1:
                break
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{label} failed after retrying: {last_error}") from last_error

def source_part(c: genai.Client, path: Path | None, pasted: str):
    if pasted.strip():
        return {"type": "text", "text": pasted.strip()}
    if not path:
        raise HTTPException(400, "Upload a PDF/TXT file or paste source content.")
    if path.suffix.lower() == ".pdf":
        f = wait_for_file(c, c.files.upload(file=str(path)))
        return {"type": "document", "uri": f.uri, "mime_type": f.mime_type}
    return {"type": "text", "text": path.read_text(encoding="utf-8", errors="replace")}

def script_prompt(question: str, length: str) -> str:
    targets = {
        "short": "450 to 600",
        "default": "1,000 to 1,250",
        "maximum": "the longest complete, non-repetitive explanation the source genuinely needs (for a full chapter, normally 3,500 to 5,000; for a short source, stay proportional and do not pad)",
    }
    target = targets.get(length, targets["default"])
    coverage = "Cover every relevant section and example in the supplied source; do not artificially shorten it, but do not pad, repeat, or invent material." if length == "maximum" else "Focus tightly on the stated topic and its source-grounded examples."
    return f"""You are an expert Gujarati school science educator. Based only on the supplied source, write a {target}-word Gujarati podcast transcript answering this topic: {question}

Output ONLY the spoken transcript. Use exactly two speakers named Host A and Host B, with every turn formatted `Host A:` or `Host B:`.

It must sound like a soft, warm, unhurried conversation between two friends—not a classroom reading and not mechanical speaker-by-speaker alternation. Host A is a calm, affectionate explainer. Host B is a genuinely curious learner who listens, responds to the immediately previous point, occasionally thinks aloud, and asks only useful follow-up questions. Vary the turn lengths naturally: most are one to three sentences; a few can be longer. Use reactions such as હા, એક મિનિટ, અરે વાહ, સાચી વાત, and exactly only when emotionally natural. Never force a reaction into every turn. Avoid repeated greetings, generic filler, slogans, or sentences such as “ચાલો મિત્રો”.

Begin with one relatable daily-life observation. Unpack difficult terms one at a time, use only source-grounded examples, and end with a concise exam-ready recap. {coverage} Do not invent facts, references, medical claims, or extra questions. Keep Gujarati script throughout except familiar terms such as DNA when helpful."""

def split_script(script: str, maximum: int = 9000) -> list[str]:
    # Keep complete speaking turns together so the two-host rhythm survives TTS chunking.
    blocks = [b.strip() for b in re.split(r"(?=Host [AB]:)", script) if b.strip()]
    chunks, current = [], ""
    for block in blocks:
        if len(current) + len(block) + 2 > maximum and current:
            chunks.append(current)
            current = block
        else:
            current = (current + "\n\n" + block).strip()
    if current:
        chunks.append(current)
    return chunks

def tts(c: genai.Client, script: str, wav_path: Path) -> None:
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(speaker="Host A", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=HOST_A_VOICE))),
                    types.SpeakerVoiceConfig(speaker="Host B", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=HOST_B_VOICE))),
                ]
            )
        ),
    )
    pcm = bytearray()
    for index, chunk in enumerate(split_script(script), start=1):
        direction = "Read this as a very soft, sweet, warm Gujarati conversation between two close friends. Speak noticeably slower than a classroom explanation: relaxed, clear, and never rushed. Host A has a gentle, reassuring smile in the voice; Host B sounds calm, curious, and comfortable. Leave a brief natural pause after an important idea or a genuine reaction, but never add long silence. Use subtle emotion and smooth sentence endings. Never sound like an announcement, a textbook, a debate, or a fast AI narration. Do not announce speaker labels.\n\n"
        response = with_retry(
            lambda: c.models.generate_content(model=TTS_MODEL, contents=direction + chunk, config=config),
            f"Audio generation for part {index}",
        )
        try:
            pcm.extend(response.candidates[0].content.parts[0].inline_data.data)
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(f"TTS returned no audio for chunk {index}.") from exc
    with wave.open(str(wav_path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000); out.writeframes(pcm)

def encode_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Create a shareable MP3 alongside the original WAV without FFmpeg."""
    with wave.open(str(wav_path), "rb") as source:
        if source.getsampwidth() != 2:
            raise RuntimeError("Expected 16-bit PCM audio for MP3 encoding.")
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(128)
        encoder.set_in_sample_rate(source.getframerate())
        encoder.set_channels(source.getnchannels())
        encoder.set_quality(2)
        payload = encoder.encode(source.readframes(source.getnframes())) + encoder.flush()
    mp3_path.write_bytes(payload)

@app.get("/", response_class=HTMLResponse)
def home(): return PAGE

@app.post("/generate", response_class=HTMLResponse)
async def generate(file: UploadFile | None = File(None), content: str = Form(""), question: str = Form(...), length: str = Form("default"), title: str = Form("audio_overview")):
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", title).strip("_") or "audio_overview"
    job = JOBS / f"{safe}_{uuid.uuid4().hex[:8]}"; job.mkdir()
    local = None
    if file and file.filename:
        local = job / Path(file.filename).name
        with local.open("wb") as out: shutil.copyfileobj(file.file, out)
    try:
        c = client(); source = source_part(c, local, content)
        result = with_retry(
            lambda: c.interactions.create(model=TEXT_MODEL, input=[source, {"type": "text", "text": script_prompt(question, length)}]),
            "Script generation",
        )
        script = result.output_text.strip()
        if "Host A:" not in script or "Host B:" not in script: raise RuntimeError("The script model did not return a valid two-host transcript. Try again.")
        transcript = job / "transcript.txt"; transcript.write_text(script, encoding="utf-8")
        audio = job / "audio_overview.wav"; tts(c, script, audio)
        encode_mp3(audio, job / "audio_overview.mp3")
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    base = f"/jobs/{job.name}"
    return HTMLResponse(f"""<!doctype html><html><head><meta charset=utf-8><title>Audio ready</title>
    <style>body{{font-family:system-ui;background:#101827;color:#e8eefb;max-width:760px;margin:40px auto;padding:0 20px}}a{{display:inline-block;padding:12px 16px;background:#5b7cfa;color:white;text-decoration:none;border-radius:8px;font-weight:700;margin:8px 8px 8px 0}}audio{{width:100%;margin:16px 0}}</style></head>
    <body><h2>Your Gujarati Audio Overview is ready</h2><audio controls src='{base}/audio_overview.mp3'></audio>
    <p><a href='{base}/audio_overview.mp3' download>Download audio (MP3)</a><a href='{base}/transcript.txt' download>Download transcript</a><a href='{base}/audio_overview.wav' download>WAV backup</a></p>
    <p><a href='/'>Create another audio</a></p></body></html>""")
