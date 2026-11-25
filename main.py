import subprocess
import uuid
from pathlib import Path

from anthropic import Anthropic
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI

app = FastAPI()
WORK = Path("/tmp/work")
WORK.mkdir(parents=True, exist_ok=True)

claude = Anthropic()          # ANTHROPIC_API_KEY env variable se key leta hai
openai_client = OpenAI()      # OPENAI_API_KEY env variable se key leta hai


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/process")
async def process(file: UploadFile = File(...)):
    job = uuid.uuid4().hex
    raw = WORK / f"{job}_raw.mp4"
    vertical = WORK / f"{job}_vertical.mp4"
    srt = WORK / f"{job}.srt"
    final = WORK / f"{job}_final.mp4"

    raw.write_bytes(await file.read())

    # 1. Video ko 9:16 (Reels format) mein badlo
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
               "pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(vertical),
    ], check=True, capture_output=True)

    # 2. Bolne wali awaaz se subtitles (SRT) banao
    with open(vertical, "rb") as audio:
        srt_text = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio, response_format="srt",
        )
    srt.write_text(srt_text, encoding="utf-8")

    # 3. Subtitles video par laga do
    subprocess.run([
        "ffmpeg", "-y", "-i", str(vertical),
        "-vf", f"subtitles={srt}",
        "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(final),
    ], check=True, capture_output=True)

    # 4. AI se caption aur hashtags banwao
    msg = claude.messages.create(
        model="claude-haiku-5-5",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": (
                f"Ye video ke subtitles hain (SRT format):\n\n{srt_text}\n\n"
                "Inke basis par ek catchy hook, 2-3 line ka caption, "
                "aur 15 relevant hashtags banao. Hinglish mein likho."
            ),
        }],
    )

    return {"caption": msg.content[0].text, "video_file": final.name}


@app.get("/videos/{name}")
def get_video(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid name")
    path = WORK / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="video/mp4")
