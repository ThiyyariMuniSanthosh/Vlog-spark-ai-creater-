import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import google.generativeai as genai

app = FastAPI(title="VlogSpark AI Backend", version="2.0.0")


class ContentRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=300)
    platform: str = Field(default="YouTube Shorts")
    language: str = Field(default="English")
    content_type: str = Field(default="Educational")
    tone: str = Field(default="Casual & Conversational")
    duration: str = Field(default="60 sec")
    audience: str = Field(default="Students & Beginners")


@app.post("/api/generate-content")
async def generate_content(req: ContentRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are VlogSpark AI, a warm, human-sounding video content director.
Turn the following title/topic into a natural, conversational, ready-to-record production package.
Never sound robotic or use generic AI clichés. Use short spoken-aloud sentences and natural regional expressions if an Indian language is selected.

Title: {req.title}
Platform: {req.platform}
Language: {req.language}
Content Type: {req.content_type}
Tone: {req.tone}
Duration: {req.duration}
Target Audience: {req.audience}

Return strict JSON with keys:
hook, intro, talking_points (list), example_story, full_script, voiceover,
speaking_guide (list of {{line, how, pause, emphasis, expression, speed}}),
scenes (list of {{num, time, shot, camera, line, on_screen, broll, visual_prompt}}),
captions (list of {{label, text}}), hashtags, music_mood, srt_preview.
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc
