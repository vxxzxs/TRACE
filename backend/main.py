from fastapi import FastAPI, UploadFile, Form, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pypdf import PdfReader
import os
import json
import io
import requests

from services.job_analyzer import analyze_job
from services.resume_analyzer import analyze_resume as analyze_resume_structured

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def compare_resume_to_jd(jd_text, resume_text):
    prompt = f"""You are a recruiting evaluation engine. Compare this resume against this job description.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

Return ONLY valid JSON in this exact format, no other text:
{{
  "overall_score": <number 0-100>,
  "matches": [
    {{"requirement": "<JD requirement>", "evidence": "<what resume shows>", "strength": "strong/moderate/weak/missing"}}
  ],
  "summary": "<one sentence explanation>"
}}"""

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    data = response.json()
    raw = data["choices"][0]["message"]["content"]
    raw = raw.strip().strip("```json").strip("```").strip()
    return json.loads(raw)


@app.get("/")
def home():
    return {"message": "TRACE backend is running"}


@app.post("/analyze-job")
async def analyze_job_endpoint(job_description: str = Form(...)):
    result = analyze_job(job_description)
    return result


@app.post("/analyze-resume")
async def analyze_resume_endpoint(resume: UploadFile = File(...)):
    file_bytes = await resume.read()
    if resume.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
    else:
        resume_text = file_bytes.decode("utf-8", errors="ignore")

    result = analyze_resume_structured(resume_text)
    return result


@app.post("/analyze")
async def analyze(resume: UploadFile = File(...), jd_text: str = Form(...)):
    file_bytes = await resume.read()

    if resume.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file_bytes)
    else:
        resume_text = file_bytes.decode("utf-8", errors="ignore")

    result = compare_resume_to_jd(jd_text, resume_text)
    return result
