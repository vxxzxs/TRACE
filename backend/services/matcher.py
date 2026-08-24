"""
matcher.py — Core recruitment intelligence.

Contains the deterministic (no-LLM-required) implementation of the pipeline:

    JD text -> analyze_job_description() -> Ideal Recruit Profile
    Resume text -> parse_resume() -> Candidate Profile
    (Candidate Profile, Ideal Profile) -> match_candidate() -> scored result
    [scored results] -> rank_candidates() -> ranked list

This is intentionally heuristic / keyword+synonym based rather than a real
NLP pipeline, per the hackathon-prototype scope. Everything here is designed
to be swapped out later for an LLM-backed version without changing the
shapes that main.py depends on.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration (kept here, not hardcoded into logic, so it's easy to tune)
# ---------------------------------------------------------------------------

# Category weights used to combine category scores into an overall score.
# These are renormalized at score time if a category has no requirements.
CATEGORY_WEIGHTS: Dict[str, float] = {
    "must_have": 0.50,
    "preferred": 0.30,
    "experience": 0.10,
    "education": 0.05,
    "nice_to_have": 0.05,
}

# Match strength scale, per the spec.
STRENGTH_NO_EVIDENCE = 0.0
STRENGTH_PARTIAL = 0.5
STRENGTH_STRONG = 0.75
STRENGTH_EXCELLENT = 1.0

# Recommendation bands: (minimum score, label), checked highest-first.
# Thresholds are inclusive lower bounds so float scores like 89.2 land
# correctly in "Strong Match" rather than falling through integer gaps.
RECOMMENDATION_BANDS: List[Tuple[float, str]] = [
    (90, "Excellent Match"),
    (75, "Strong Match"),
    (60, "Potential Match"),
    (40, "Weak Match"),
    (0, "Poor Match"),
]

# A small taxonomy of common skills -> synonyms/related keywords.
# This drives both JD requirement extraction and resume skill detection.
# Easy to extend; not meant to be exhaustive.
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Python": ["python"],
    "Machine Learning": ["machine learning", "ml ", " ml,", "ml.", "ml)"],
    "Deep Learning": ["deep learning", "neural network", "neural networks"],
    "TensorFlow": ["tensorflow", "tf.keras", "keras"],
    "PyTorch": ["pytorch", "torch"],
    "NLP": ["nlp", "natural language processing", "text classification",
            "named entity recognition", "sentiment analysis"],
    "LLM": ["llm", "large language model", "gpt", "transformer", "rag",
            "retrieval augmented generation", "prompt engineering"],
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "REST APIs": ["rest api", "rest apis", "restful", "api development",
                  "backend services", "backend api"],
    "Docker": ["docker", "containeriz"],
    "Kubernetes": ["kubernetes", "k8s"],
    "MLOps": ["mlops", "ml ops", "ci/cd for ml", "model deployment"],
    "SQL": ["sql", "postgres", "postgresql", "mysql", "sqlite"],
    "NoSQL": ["nosql", "mongodb", "dynamodb"],
    "Java": ["java "],
    "JavaScript": ["javascript", " js,", " js.", "node.js", "nodejs"],
    "React": ["react", "reactjs", "react.js"],
    "Git": ["git ", "github", "gitlab", "version control"],
    "Data Analysis": ["data analysis", "pandas", "numpy", "data wrangling"],
    "Computer Vision": ["computer vision", "opencv", "image classification",
                         "object detection"],
    "Communication": ["communication skills", "stakeholder", "presented",
                       "cross-functional"],
    "Leadership": ["led a team", "team lead", "managed a team", "mentored"],
}

# Phrases that mark a JD section as mandatory / must-have.
MANDATORY_SECTION_MARKERS = [
    "must have", "must-have", "required", "requirements", "minimum qualifications",
    "minimum requirements", "you must have", "required skills",
]

# Phrases that mark a JD section as preferred (strongly desired, not mandatory).
PREFERRED_SECTION_MARKERS = [
    "preferred", "strongly preferred", "desired", "we'd love", "ideally",
]

# Phrases that mark a JD section as nice-to-have.
NICE_TO_HAVE_SECTION_MARKERS = [
    "nice to have", "nice-to-have", "bonus", "good to have", "a plus",
    "pluses", "extra credit",
]

# Default "avoid / risk factor" boilerplate, per the spec's example profile.
DEFAULT_AVOID_FACTORS = [
    "No evidence of required technical skills",
    "Experience significantly below minimum requirement",
]

DEGREE_PATTERNS = [
    (r"\bph\.?d\b", "PhD", 4),
    (r"\bmaster'?s?\b|\bm\.?tech\b|\bm\.?sc\b|\bmba\b", "Master's degree", 3),
    (r"\bbachelor'?s?\b|\bb\.?tech\b|\bb\.?sc\b|\bb\.?e\b\.?", "Bachelor's degree", 2),
    (r"\bdiploma\b", "Diploma", 1),
]

YEARS_EXPERIENCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _find_sentence_containing(text: str, needle: str) -> Optional[str]:
    """Return the sentence (rough split) in `text` that contains `needle`, if any."""
    if not needle:
        return None
    lowered_text = text.lower()
    idx = lowered_text.find(needle.lower())
    if idx == -1:
        return None
    # Expand outward to sentence boundaries.
    start = max(text.rfind(".", 0, idx), text.rfind("\n", 0, idx))
    end_candidates = [p for p in (text.find(".", idx), text.find("\n", idx)) if p != -1]
    end = min(end_candidates) if end_candidates else len(text)
    sentence = text[start + 1: end + 1].strip()
    return sentence if sentence else text[max(0, idx - 40): idx + 60].strip()


def _keywords_present(text_lower: str, keywords: List[str]) -> List[str]:
    return [kw for kw in keywords if kw in text_lower]


def _extract_years_experience(text: str) -> float:
    matches = YEARS_EXPERIENCE_RE.findall(text)
    if not matches:
        return 0.0
    return max(float(m) for m in matches)


def _extract_required_years(text: str) -> Optional[float]:
    matches = YEARS_EXPERIENCE_RE.findall(text)
    if not matches:
        return None
    return min(float(m) for m in matches)  # "2+ years" -> minimum bar is 2


def _extract_degree(text: str) -> Tuple[Optional[str], int]:
    """Return (degree label, rank) for the highest-ranked degree mentioned."""
    text_lower = text.lower()
    best_label, best_rank = None, 0
    for pattern, label, rank in DEGREE_PATTERNS:
        if re.search(pattern, text_lower) and rank > best_rank:
            best_label, best_rank = label, rank
    return best_label, best_rank


def _guess_job_title(jd_text: str) -> str:
    lines = [ln.strip() for ln in jd_text.strip().splitlines() if ln.strip()]
    if not lines:
        return "Not specified"
    first_line = lines[0]
    # Common patterns: "Job Title: X", "Role: X", or just the title itself.
    for prefix in ("job title:", "role:", "position:", "title:"):
        if first_line.lower().startswith(prefix):
            return first_line[len(prefix):].strip()
    # If the first line is short, assume it IS the title.
    if len(first_line.split()) <= 8:
        return first_line
    return "Not specified"


def _split_sections(jd_text: str) -> Dict[str, str]:
    """
    Split JD text into rough sections keyed by 'mandatory' / 'preferred' /
    'nice_to_have' / 'general', based on header-like marker phrases.
    Falls back to putting everything in 'general' if no markers are found.
    """
    text_lower = jd_text.lower()
    all_markers = (
        [(m, "mandatory") for m in MANDATORY_SECTION_MARKERS]
        + [(m, "preferred") for m in PREFERRED_SECTION_MARKERS]
        + [(m, "nice_to_have") for m in NICE_TO_HAVE_SECTION_MARKERS]
    )
    hits = []
    for marker, section in all_markers:
        for m in re.finditer(re.escape(marker), text_lower):
            hits.append((m.start(), section))
    if not hits:
        return {"general": jd_text}

    hits.sort(key=lambda h: h[0])
    sections: Dict[str, str] = {}
    for i, (start, section) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(jd_text)
        chunk = jd_text[start:end]
        sections[section] = sections.get(section, "") + "\n" + chunk
    return sections


# ---------------------------------------------------------------------------
# Step 1: JD -> Ideal Recruit Profile
# ---------------------------------------------------------------------------

def analyze_job_description(jd_text: str) -> Dict[str, Any]:
    """
    Deterministic JD analysis. Extracts requirements by:
      1. Splitting the JD into mandatory / preferred / nice-to-have sections
         using header phrases (falls back to a single 'general' section).
      2. Scanning each section against the skill taxonomy for recognizable
         requirements.
      3. Separately extracting an experience-years requirement and an
         education requirement, wherever in the text they appear.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("job_description text is empty")

    sections = _split_sections(jd_text)
    job_title = _guess_job_title(jd_text)

    must_have: List[Dict[str, Any]] = []
    preferred: List[Dict[str, Any]] = []
    nice_to_have: List[Dict[str, Any]] = []

    def bucket_for(section_name: str) -> List[Dict[str, Any]]:
        if section_name == "mandatory":
            return must_have
        if section_name == "preferred":
            return preferred
        if section_name == "nice_to_have":
            return nice_to_have
        return must_have  # 'general' text defaults to must_have (safer default)

    seen_requirements = set()
    for section_name, section_text in sections.items():
        section_lower = section_text.lower()
        target_list = bucket_for(section_name)
        mandatory_flag = section_name in ("mandatory", "general")
        for requirement, keywords in SKILL_TAXONOMY.items():
            if requirement in seen_requirements:
                continue
            found_kws = _keywords_present(section_lower, keywords)
            if found_kws:
                importance = {
                    "mandatory": 1.0,
                    "general": 0.8,
                    "preferred": 0.7,
                    "nice_to_have": 0.3,
                }[section_name]
                expected_level = {
                    "mandatory": "strong",
                    "general": "strong",
                    "preferred": "working",
                    "nice_to_have": "basic",
                }[section_name]
                target_list.append({
                    "requirement": requirement,
                    "category": section_name if section_name != "general" else "must_have",
                    "importance": importance,
                    "mandatory": mandatory_flag,
                    "expected_level": expected_level,
                    "evidence_needed": f"Demonstrated use of {requirement} in experience or projects",
                    "keywords": keywords,
                    "type": "skill",
                })
                seen_requirements.add(requirement)

    # Experience-years requirement (search whole JD, not just sections).
    required_years = _extract_required_years(jd_text)
    if required_years is not None:
        must_have.append({
            "requirement": f"{required_years:g}+ years experience",
            "category": "must_have",
            "importance": 1.0,
            "mandatory": True,
            "expected_level": "strong",
            "evidence_needed": "Total relevant work experience meeting or exceeding the minimum",
            "keywords": ["years", "experience"],
            "type": "experience",
            "required_years": required_years,
        })

    # Education requirement.
    degree_label, degree_rank = _extract_degree(jd_text)
    if degree_label:
        must_have.append({
            "requirement": degree_label,
            "category": "must_have",
            "importance": 0.6,
            "mandatory": True,
            "expected_level": "strong",
            "evidence_needed": "Relevant completed degree",
            "keywords": [degree_label.lower()],
            "type": "education",
            "required_degree_rank": degree_rank,
        })

    if not must_have and not preferred and not nice_to_have:
        raise ValueError(
            "Could not extract any recognizable requirements from the job description"
        )

    return {
        "job_title": job_title,
        "ideal_profile": {
            "must_have": must_have,
            "preferred": preferred,
            "nice_to_have": nice_to_have,
            "avoid": list(DEFAULT_AVOID_FACTORS),
        },
    }


# ---------------------------------------------------------------------------
# Step 2: Resume -> Candidate Profile
# ---------------------------------------------------------------------------

def parse_resume(resume_text: str, candidate_id: str, name: str) -> Dict[str, Any]:
    """
    Lightweight resume parsing: detects skills via the shared taxonomy,
    estimates years of experience, and extracts an education level.
    Not intended to be a full resume parser — good enough to drive matching.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume text is empty")

    text_lower = resume_text.lower()
    detected_skills = [
        requirement
        for requirement, keywords in SKILL_TAXONOMY.items()
        if _keywords_present(text_lower, keywords)
    ]

    years_experience = _extract_years_experience(resume_text)
    degree_label, degree_rank = _extract_degree(resume_text)

    # Very lightweight project/certification extraction: lines mentioning
    # the words "project" or "certified"/"certification".
    projects = [
        ln.strip(" -•\t")
        for ln in resume_text.splitlines()
        if "project" in ln.lower() and len(ln.strip()) > 5
    ][:5]
    certifications = [
        ln.strip(" -•\t")
        for ln in resume_text.splitlines()
        if ("certified" in ln.lower() or "certification" in ln.lower()) and len(ln.strip()) > 5
    ][:5]

    return {
        "candidate_id": candidate_id,
        "name": name,
        "skills": detected_skills,
        "years_experience": years_experience,
        "degree": degree_label,
        "degree_rank": degree_rank,
        "projects": projects,
        "certifications": certifications,
        "raw_text": resume_text,
    }


# ---------------------------------------------------------------------------
# Step 3: (Candidate Profile, Ideal Profile) -> scored match
# ---------------------------------------------------------------------------

def _match_skill_requirement(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    text_lower = candidate["raw_text"].lower()
    requirement_name = req["requirement"]
    keywords = req["keywords"]

    found_kws = _keywords_present(text_lower, keywords)
    in_detected_skills = requirement_name in candidate["skills"]

    if in_detected_skills and len(found_kws) >= 1:
        strength = STRENGTH_EXCELLENT
        status = "matched"
    elif found_kws:
        # Something matched but weakly (e.g. only a loosely related synonym).
        strength = STRENGTH_STRONG if len(found_kws) > 1 else STRENGTH_PARTIAL
        status = "matched" if strength >= STRENGTH_STRONG else "partial"
    else:
        strength = STRENGTH_NO_EVIDENCE
        status = "missing"

    evidence = None
    if found_kws:
        evidence = _find_sentence_containing(candidate["raw_text"], found_kws[0])

    return {
        "requirement": requirement_name,
        "status": status,
        "strength": strength,
        "evidence": evidence,
    }


def _match_experience_requirement(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    required_years = req.get("required_years", 0.0)
    candidate_years = candidate["years_experience"]

    if candidate_years >= required_years:
        strength, status = STRENGTH_EXCELLENT, "matched"
    elif candidate_years >= required_years - 1:
        strength, status = STRENGTH_PARTIAL, "partial"
    else:
        strength, status = STRENGTH_NO_EVIDENCE, "missing"

    evidence = None
    if candidate_years > 0:
        evidence = f"Resume indicates approximately {candidate_years:g} years of relevant experience."

    return {
        "requirement": req["requirement"],
        "status": status,
        "strength": strength,
        "evidence": evidence,
    }


def _match_education_requirement(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    required_rank = req.get("required_degree_rank", 0)
    candidate_rank = candidate.get("degree_rank", 0)

    if candidate_rank >= required_rank and candidate_rank > 0:
        strength, status = STRENGTH_EXCELLENT, "matched"
    elif candidate_rank > 0:
        strength, status = STRENGTH_PARTIAL, "partial"
    else:
        strength, status = STRENGTH_NO_EVIDENCE, "missing"

    evidence = f"Resume indicates: {candidate['degree']}" if candidate.get("degree") else None

    return {
        "requirement": req["requirement"],
        "status": status,
        "strength": strength,
        "evidence": evidence,
    }


def _match_requirement(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    req_type = req.get("type", "skill")
    if req_type == "experience":
        return _match_experience_requirement(req, candidate)
    if req_type == "education":
        return _match_education_requirement(req, candidate)
    return _match_skill_requirement(req, candidate)


def _category_score(matched_results: List[Dict[str, Any]]) -> Optional[float]:
    """Average strength * 100 for a list of match results, or None if empty."""
    if not matched_results:
        return None
    avg_strength = sum(r["strength"] for r in matched_results) / len(matched_results)
    return round(avg_strength * 100, 1)


def _recommendation_for_score(score: float) -> str:
    for threshold, label in RECOMMENDATION_BANDS:
        if score >= threshold:
            return label
    return "Poor Match"


def match_candidate(candidate: Dict[str, Any], ideal_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score one candidate against the Ideal Recruit Profile.
    Returns a fully-populated result dict matching the API response shape.
    """
    must_have = ideal_profile.get("must_have", [])
    preferred = ideal_profile.get("preferred", [])
    nice_to_have = ideal_profile.get("nice_to_have", [])

    # Separate out experience/education requirements (wherever they live)
    # so we can report them as their own category, per the spec's example
    # category_scores shape.
    def split_by_type(reqs: List[Dict[str, Any]]):
        skills = [r for r in reqs if r.get("type", "skill") == "skill"]
        experience = [r for r in reqs if r.get("type") == "experience"]
        education = [r for r in reqs if r.get("type") == "education"]
        return skills, experience, education

    must_skills, must_experience, must_education = split_by_type(must_have)
    pref_skills, pref_experience, pref_education = split_by_type(preferred)
    nice_skills, _, _ = split_by_type(nice_to_have)

    all_experience_reqs = must_experience + pref_experience
    all_education_reqs = must_education + pref_education

    must_results = [_match_requirement(r, candidate) for r in must_skills]
    pref_results = [_match_requirement(r, candidate) for r in pref_skills]
    nice_results = [_match_requirement(r, candidate) for r in nice_skills]
    experience_results = [_match_requirement(r, candidate) for r in all_experience_reqs]
    education_results = [_match_requirement(r, candidate) for r in all_education_reqs]

    category_scores_raw = {
        "must_have": _category_score(must_results),
        "preferred": _category_score(pref_results),
        "experience": _category_score(experience_results),
        "education": _category_score(education_results),
        "nice_to_have": _category_score(nice_results),
    }

    # Renormalize weights across categories that actually have requirements.
    present_categories = {k: v for k, v in category_scores_raw.items() if v is not None}
    if not present_categories:
        raise ValueError("Ideal profile has no requirements to match against")

    weight_sum = sum(CATEGORY_WEIGHTS[k] for k in present_categories)
    overall_score = sum(
        present_categories[k] * (CATEGORY_WEIGHTS[k] / weight_sum) for k in present_categories
    )
    overall_score = round(overall_score, 1)

    # Combine all requirement-level results for the flat matched/partial/missing lists.
    all_requirement_defs = must_skills + pref_skills + nice_skills + all_experience_reqs + all_education_reqs
    all_results = must_results + pref_results + nice_results + experience_results + education_results

    matched_requirements = [r for r in all_results if r["status"] == "matched"]
    partial_matches = [r for r in all_results if r["status"] == "partial"]
    missing_requirements = []
    for req_def, result in zip(all_requirement_defs, all_results):
        if result["status"] == "missing":
            missing_requirements.append({
                "requirement": result["requirement"],
                "importance": req_def.get("category", "must_have"),
            })

    # Mandatory failure check: any mandatory requirement with status "missing".
    critical_mandatory_failure = any(
        req_def.get("mandatory") and result["status"] == "missing"
        for req_def, result in zip(all_requirement_defs, all_results)
    )

    concerns: List[str] = []
    if critical_mandatory_failure:
        failed_names = [
            result["requirement"]
            for req_def, result in zip(all_requirement_defs, all_results)
            if req_def.get("mandatory") and result["status"] == "missing"
        ]
        concerns.append(f"Fails mandatory requirement(s): {', '.join(failed_names)}")
    if category_scores_raw["experience"] is not None and category_scores_raw["experience"] < 75:
        concerns.append("Experience level is below the target for this role")
    if category_scores_raw["nice_to_have"] is not None and category_scores_raw["nice_to_have"] < 30:
        concerns.append("Limited evidence of nice-to-have skills")
    if not concerns:
        concerns.append("No significant concerns identified")

    recommendation = _recommendation_for_score(overall_score)

    top_matches = ", ".join(r["requirement"] for r in matched_requirements[:3]) or "no strong matches"
    top_gap = missing_requirements[0]["requirement"] if missing_requirements else None
    summary = f"Overall score {overall_score}/100 ({recommendation}). Strong evidence for {top_matches}."
    if critical_mandatory_failure:
        summary += " However, this candidate fails a mandatory requirement."
    elif top_gap:
        summary += f" Main gap: {top_gap}."

    return {
        "candidate_id": candidate["candidate_id"],
        "name": candidate["name"],
        "overall_score": overall_score,
        "recommendation": recommendation,
        "critical_mandatory_failure": critical_mandatory_failure,
        "category_scores": category_scores_raw,
        "matched_requirements": matched_requirements,
        "partial_matches": partial_matches,
        "missing_requirements": missing_requirements,
        "concerns": concerns,
        "summary": summary,
    }


def rank_candidates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort candidate results by overall_score descending. Candidates with a
    critical mandatory failure are sorted after all candidates without one,
    even if their raw score is higher — mirrors the spec's "don't let a
    keyword-heavy but mandatory-failing candidate rank #1" principle.
    """
    ranked = sorted(
        results,
        key=lambda r: (r["critical_mandatory_failure"], -r["overall_score"]),
    )
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return ranked