"""Pydantic schemas kept for reuse by future frontend/API clients."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class JobRequest(BaseModel):
    job_description: str = Field(min_length=1)

class ResumeRequest(BaseModel):
    resume_text: str = Field(min_length=1)
    candidate_id: Optional[str] = None
    name: Optional[str] = None

class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    skills: List[str] = []
    experience_years: float = 0
    education: str = "Not specified"
    location: str = "Not specified"
    evidence: Dict[str, Any] = {}
