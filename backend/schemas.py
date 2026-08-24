from pydantic import BaseModel
from typing import List


class SkillWeight(BaseModel):
    name: str
    weight: float


class ExperienceRequirement(BaseModel):
    minimum_years: float
    weight: float


class IdealProfile(BaseModel):
    role: str
    must_have: List[SkillWeight]
    preferred: List[SkillWeight]
    experience: ExperienceRequirement
    avoid: List[str]


class CandidateProfile(BaseModel):
    name: str
    skills: List[str]
    experience_years: float
    education: str
    location: str
    evidence: dict
