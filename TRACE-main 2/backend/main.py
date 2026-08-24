"""TRACE FastAPI backend.

Run from backend/:
    uvicorn main:app --reload
"""
from __future__ import annotations

import io
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from services.job_analyzer import analyze_job
from services.matcher import match_candidate, rank_candidates
from services.resume_analyzer import analyze_resume

load_dotenv()

app = FastAPI(
    title="TRACE — Transparent Recruitment Matching",
    version="1.0.0",
    description="Requirement-aware, explainable candidate matching for recruiters.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # acceptable for local hackathon demo
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobRequest(BaseModel):
    job_description: str = Field(min_length=1)

class ResumeRequest(BaseModel):
    resume_text: str = Field(min_length=1)
    candidate_id: Optional[str] = None
    name: Optional[str] = None

class CandidateInput(BaseModel):
    candidate_id: Optional[str] = None
    name: str
    text: Optional[str] = None
    resume_text: Optional[str] = None
    profile: Optional[dict] = None

class MatchRequest(BaseModel):
    profile: dict
    candidates: List[dict]

def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages).strip()

def _extract_uploaded_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        return _extract_pdf_text(data)
    return data.decode("utf-8", errors="ignore").strip()

def _public_profile(profile: dict) -> dict:
    """Return the shape expected by the Streamlit UI while preserving rich data."""
    return profile

@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "TRACE",
        "message": "Transparent Requirement-to-Candidate Evaluation API",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze-job")
def analyze_job_endpoint(request: JobRequest):
    try:
        return _public_profile(analyze_job(request.job_description))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Job analysis failed: {exc}")

@app.post("/analyze-resume")
def analyze_resume_endpoint(request: ResumeRequest):
    try:
        candidate_id = request.candidate_id or "candidate_001"
        return analyze_resume(request.resume_text, candidate_id, request.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {exc}")

@app.post("/analyze-resume-file")
async def analyze_resume_file(
    resume: UploadFile = File(...),
    candidate_id: Optional[str] = Form(None),
):
    try:
        data = await resume.read()
        text = _extract_uploaded_text(resume.filename or "", data)
        if not text:
            raise ValueError("Could not extract any text from the uploaded resume.")
        return analyze_resume(
            text,
            candidate_id or (os.path.splitext(resume.filename or "candidate")[0]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume file analysis failed: {exc}")

@app.post("/match")
def match_endpoint(request: MatchRequest):
    try:
        results = []
        for i, candidate in enumerate(request.candidates, 1):
            # Accept either a previously analyzed profile or raw resume text.
            if candidate.get("profile"):
                profile = dict(candidate["profile"])
            else:
                text = candidate.get("text") or candidate.get("resume_text")
                if not text:
                    raise ValueError(f"Candidate {i} has no resume text or profile.")
                profile = analyze_resume(
                    text,
                    candidate.get("candidate_id") or f"candidate_{i:03d}",
                    candidate.get("name"),
                )
            results.append(match_candidate(profile, request.profile))
        return {"rankings": rank_candidates(results)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Matching failed: {exc}")

@app.post("/analyze")
async def legacy_analyze(resume: UploadFile = File(...), jd_text: str = Form(...)):
    """Compatibility endpoint for simple single-resume demos."""
    try:
        text = _extract_uploaded_text(resume.filename or "", await resume.read())
        profile = analyze_job(jd_text)
        candidate = analyze_resume(text, os.path.splitext(resume.filename or "candidate")[0])
        return match_candidate(candidate, profile)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
