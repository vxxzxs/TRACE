"""Resume analyzer with optional Mistral enhancement and deterministic fallback."""
import json, os, re, requests
from dotenv import load_dotenv
from .matcher import parse_resume

from dotenv import load_dotenv
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

def _clean_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)

def _llm_analyze(resume_text: str):
    prompt = f"""Extract a structured candidate profile from this resume.
Return ONLY JSON:
{{
"name": "...", "skills": [], "experience_years": 0,
"education": "...", "location": "...", "evidence": {{}},
"projects": []
}}
Only use evidence present in the resume. Keep evidence concrete and concise.

RESUME:
{resume_text}"""
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        json={"model": MISTRAL_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
        timeout=45,
    )
    r.raise_for_status()
    return _clean_json(r.json()["choices"][0]["message"]["content"])

def analyze_resume(resume_text: str, candidate_id: str = "candidate_001", name: str | None = None) -> dict:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume is empty.")
    if MISTRAL_API_KEY:
        try:
            result = _llm_analyze(resume_text)
            result.setdefault("candidate_id", candidate_id)
            result.setdefault("name", name or "Unknown Candidate")
            result.setdefault("skills", [])
            result.setdefault("experience_years", 0)
            result.setdefault("education", "Not specified")
            result.setdefault("location", "Not specified")
            result.setdefault("evidence", {})
            result.setdefault("projects", [])
            result["raw_text"] = resume_text
            result["source"] = "mistral"
            return result
        except Exception:
            pass
    result = parse_resume(resume_text, candidate_id, name)
    result["source"] = "deterministic"
    return result
