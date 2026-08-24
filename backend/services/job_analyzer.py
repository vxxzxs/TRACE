"""JD analyzer with optional Mistral enhancement and deterministic fallback."""
import json, os, re, requests
from dotenv import load_dotenv
from .matcher import analyze_job_description

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

def _clean_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)

def _llm_analyze(jd_text: str):
    prompt = f"""Extract an Ideal Recruit Profile from this job description.
Return ONLY JSON with keys:
role, company, must_have, preferred, nice_to_have, experience, education, location, work_mode, avoid.
Each skill item must contain requirement, category, importance, mandatory, expected_level, keywords, type.
Experience must contain minimum_years and weight. Education must contain required and weight.
Do not invent requirements not supported by the JD.

JOB DESCRIPTION:
{jd_text}"""
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        json={"model": MISTRAL_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1},
        timeout=45,
    )
    r.raise_for_status()
    return _clean_json(r.json()["choices"][0]["message"]["content"])

def analyze_job(job_description: str) -> dict:
    if not job_description or not job_description.strip():
        raise ValueError("Job description is empty.")
    if MISTRAL_API_KEY:
        try:
            result = _llm_analyze(job_description)
            # Ensure required containers exist even if the LLM omitted optional fields.
            for key in ("must_have", "preferred", "nice_to_have", "avoid"):
                result.setdefault(key, [])
            result.setdefault("role", "Not specified")
            result.setdefault("experience", {"minimum_years": 0, "weight": 0.10})
            result.setdefault("education", {"required": None, "weight": 0.05})
            result.setdefault("location", None)
            result.setdefault("work_mode", None)
            result["source"] = "mistral"
            return result
        except Exception:
            pass
    result = analyze_job_description(job_description)
    result["source"] = "deterministic"
    return result
