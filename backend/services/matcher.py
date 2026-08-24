"""
TRACE matching engine.

Pipeline:
JD text -> Ideal Recruit Profile -> Candidate Profile -> requirement matching
-> weighted score -> ranked candidates.

The matcher is deterministic so the hackathon demo keeps working without an
LLM/API key. LLM extraction can be layered on top by the analyzer services.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

CATEGORY_WEIGHTS = {
    "must_have": 0.50,
    "preferred": 0.25,
    "experience": 0.10,
    "education": 0.05,
    "location": 0.05,
    "nice_to_have": 0.05,
}

RECOMMENDATION_BANDS = [
    (90, "Excellent Match"),
    (75, "Strong Match"),
    (60, "Potential Match"),
    (40, "Weak Match"),
    (0, "Poor Match"),
]

# Canonical skill -> phrases/synonyms. Keep this deliberately transparent.
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "Python": ["python"],
    "Machine Learning": ["machine learning", "ml workflow", "ml model", "ml models"],
    "Deep Learning": ["deep learning", "neural network", "neural networks"],
    "Data Structures & Algorithms": ["data structures", "algorithms", "data structure", "dsa"],
    "NumPy": ["numpy"],
    "pandas": ["pandas"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow", "tf.keras"],
    "PyTorch": ["pytorch", "torch"],
    "NLP": ["nlp", "natural language processing", "text classification", "sentiment analysis", "named entity recognition"],
    "LLM": ["llm", "large language model", "gpt", "transformer", "rag", "retrieval augmented generation", "prompt engineering"],
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "REST APIs": ["rest api", "rest apis", "restful api", "api development", "backend api", "model-serving api"],
    "Docker": ["docker", "containerization", "containerized"],
    "Kubernetes": ["kubernetes", "k8s"],
    "MLOps": ["mlops", "ml ops", "model deployment", "model serving", "deployment"],
    "SQL": ["sql", "postgres", "postgresql", "mysql", "sqlite"],
    "NoSQL": ["nosql", "mongodb", "dynamodb"],
    "Java": ["java"],
    "JavaScript": ["javascript", "node.js", "nodejs"],
    "React": ["react", "reactjs", "react.js"],
    "Git": ["git", "github", "gitlab", "version control"],
    "Jupyter": ["jupyter", "jupyter notebook", "notebook"],
    "Data Analysis": ["data analysis", "data analytics", "data wrangling", "exploratory data analysis", "eda"],
    "Data Visualization": ["data visualization", "visualization", "matplotlib", "seaborn", "power bi", "tableau"],
    "Computer Vision": ["computer vision", "opencv", "image classification", "object detection"],
    "Communication": ["communication skills", "technical communication", "stakeholder", "presented", "presentation", "cross-functional"],
    "Leadership": ["led a team", "team lead", "managed a team", "mentored"],
    "Problem Solving": ["problem solving", "problem-solving", "troubleshooting", "analytical thinking", "analytical skills", "critical thinking"],
    "Documentation": ["documentation", "documented", "experiment logs"],
}

DEGREE_PATTERNS = [
    (r"\bph\.?\s*d\b", "PhD", 4),
    (r"\b(master'?s|m\.?tech|m\.?sc|mba)\b", "Master's degree", 3),
    (r"\b(bachelor'?s|b\.?tech|b\.?sc|b\.?e)\b", "Bachelor's degree", 2),
    (r"\bdiploma\b", "Diploma", 1),
]
YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs|year)\b", re.I)

MANDATORY_MARKERS = ["must-have", "must have", "required skills", "requirements", "minimum qualifications", "minimum requirements"]
PREFERRED_MARKERS = ["preferred skills", "preferred", "strongly preferred", "desired", "advantage"]
NICE_MARKERS = ["nice to have", "nice-to-have", "bonus", "good to have", "a plus"]
PROJECT_MARKERS = ["project", "internship", "research", "work experience", "experience", "hackathon", "competition"]

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()

def _contains(text: str, phrase: str) -> bool:
    t = _norm(text)
    p = _norm(phrase)
    if not p:
        return False
    # Word-boundary matching for short/simple tokens prevents "java" matching
    # "javascript", while phrases remain literal substring matches.
    if re.fullmatch(r"[a-z0-9+#.-]+", p) and len(p) <= 8:
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t))
    return p in t

def _find_evidence(text: str, phrase: str) -> Optional[str]:
    if not phrase:
        return None
    low = text.lower()
    idx = low.find(phrase.lower())
    if idx < 0:
        return None
    start_candidates = [text.rfind(".", 0, idx), text.rfind("\n", 0, idx)]
    start = max(start_candidates)
    ends = [p for p in (text.find(".", idx), text.find("\n", idx)) if p >= 0]
    end = min(ends) if ends else len(text)
    out = text[start + 1:end + 1].strip(" \t\r\n-•")
    return out[:300] if out else text[max(0, idx-60):idx+240].strip()

def _extract_years(text: str) -> float:
    vals = [float(x) for x in YEARS_RE.findall(text)]
    return max(vals) if vals else 0.0

def _extract_min_years(text: str) -> Optional[float]:
    # Handle ranges such as "0 to 1 year" / "0-1 years" before generic
    # expressions. For an entry-level range, the lower bound is the minimum.
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", text, re.I)
    if m:
        return float(m.group(1))
    vals = [float(x) for x in YEARS_RE.findall(text)]
    return min(vals) if vals else None

def _extract_degree(text: str) -> Tuple[Optional[str], int]:
    best = (None, 0)
    for pattern, label, rank in DEGREE_PATTERNS:
        if re.search(pattern, text, re.I) and rank > best[1]:
            best = (label, rank)
    return best

def _job_title(text: str) -> str:
    for line in [x.strip() for x in text.splitlines() if x.strip()][:8]:
        m = re.match(r"(?:job title|role|position|title)\s*:?\s*(.+)", line, re.I)
        if m:
            return m.group(1).strip()
        if len(line.split()) <= 8 and line.lower() not in {"job description"}:
            return line
    return "Not specified"

def _section_chunks(text: str) -> Dict[str, str]:
    """Split a JD on its actual section-heading lines."""
    heading_map = {
        "must-have skills": "mandatory",
        "preferred skills": "preferred",
        "nice to have": "nice_to_have",
        "nice-to-have": "nice_to_have",
        "minimum experience": "experience_info",
        "education requirements": "education_info",
        "certifications": "certification_info",
        "location and work mode": "location_info",
        "responsibilities": "responsibilities_info",
        "constraints and deal breakers": "constraints_info",
        "evidence standard": "evidence_info",
        "selection focus": "selection_info",
    }
    sections = {"general": []}
    current = "general"
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        matched = None
        for heading, key in heading_map.items():
            if low.startswith(heading):
                matched = key
                break
        if matched:
            current = matched
            sections.setdefault(current, [])
            remainder = line[len(next(h for h in heading_map if low.startswith(h))):].strip(" :-")
            if remainder:
                sections[current].append(remainder)
        elif line:
            sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items() if v}

def analyze_job_description(jd_text: str) -> Dict[str, Any]:
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description is empty.")
    chunks = _section_chunks(jd_text)
    title = _job_title(jd_text)
    buckets = {"must_have": [], "preferred": [], "nice_to_have": []}
    seen = set()

    for sec, content in chunks.items():
        target = {"mandatory": "must_have", "preferred": "preferred", "nice_to_have": "nice_to_have"}.get(sec)
        if target is None:
            continue
        for skill, phrases in SKILL_TAXONOMY.items():
            if skill in seen:
                continue
            found = [p for p in phrases if _contains(content, p)]
            if found:
                importance = {"must_have": 1.0, "preferred": 0.7, "nice_to_have": 0.35}[target]
                buckets[target].append({
                    "requirement": skill,
                    "category": target,
                    "importance": importance,
                    "mandatory": target == "must_have",
                    "expected_level": "strong" if target == "must_have" else "working",
                    "keywords": phrases,
                    "type": "skill",
                })
                seen.add(skill)

    min_years = _extract_min_years(jd_text)
    if min_years is not None:
        buckets["must_have"].append({
            "requirement": f"{min_years:g}+ years relevant experience",
            "category": "must_have", "importance": 1.0, "mandatory": True,
            "expected_level": "entry-level", "keywords": ["years", "experience"],
            "type": "experience", "required_years": min_years,
        })

    degree, degree_rank = _extract_degree(jd_text)
    if degree:
        education_text = chunks.get("education_info", "")
        field_phrases = ["artificial intelligence", "machine learning", "computer science",
                         "information technology", "data science", "electronics"]
        relevant_fields = [x for x in field_phrases if x in education_text.lower()]
        buckets["must_have"].append({
            "requirement": degree, "category": "must_have", "importance": 0.7,
            "mandatory": True, "expected_level": "relevant field",
            "keywords": [degree.lower()], "type": "education",
            "required_degree_rank": degree_rank,
            "relevant_fields": relevant_fields,
        })

    # Explicit practical-evidence requirement.
    if re.search(r"(at least one|one or more).{0,100}(completed )?(project|internship|research|technical work)", jd_text, re.I):
        buckets["must_have"].append({
            "requirement": "Practical project / technical evidence",
            "category": "must_have", "importance": 1.0, "mandatory": True,
            "expected_level": "demonstrated",
            "keywords": ["project", "internship", "research", "technical work"],
            "type": "project",
        })

    location = None
    mode = None
    low = jd_text.lower()
    m = re.search(r"location\s*:\s*([^\n.]+)", jd_text, re.I)
    if m: location = m.group(1).strip()
    if "hybrid" in low: mode = "Hybrid"
    elif "remote" in low: mode = "Remote"
    elif "on-site" in low or "onsite" in low: mode = "On-site"

    return {
        "role": title,
        "company": _extract_company(jd_text),
        "must_have": buckets["must_have"],
        "preferred": buckets["preferred"],
        "nice_to_have": buckets["nice_to_have"],
        "experience": {"minimum_years": min_years or 0, "weight": 0.10},
        "education": {"required": degree, "weight": 0.05},
        "location": location,
        "work_mode": mode,
        "avoid": [
            "No evidence of required technical skills",
            "No evidence of practical technical work",
        ],
    }

def _extract_company(text: str) -> Optional[str]:
    m = re.search(r"Company Name\s*:?\s*([^\n]+)", text, re.I)
    return m.group(1).strip() if m else None

def parse_resume(resume_text: str, candidate_id: str, name: Optional[str] = None) -> Dict[str, Any]:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume is empty.")
    skills = []
    evidence = {}
    for skill, phrases in SKILL_TAXONOMY.items():
        found = [p for p in phrases if _contains(resume_text, p)]
        if found:
            skills.append(skill)
            evidence[skill] = _find_evidence(resume_text, found[0])
    degree, degree_rank = _extract_degree(resume_text)
    years = _extract_years(resume_text)
    location = _extract_location(resume_text)
    candidate_name = name or _extract_name(resume_text) or "Unknown Candidate"
    project_evidence = [line.strip(" -•\t") for line in resume_text.splitlines()
                        if any(k in line.lower() for k in PROJECT_MARKERS) and len(line.strip()) > 8][:8]
    return {
        "candidate_id": candidate_id,
        "name": candidate_name,
        "skills": skills,
        "experience_years": years,
        "education": degree or "Not specified",
        "degree_rank": degree_rank,
        "location": location or "Not specified",
        "evidence": evidence,
        "projects": project_evidence,
        "raw_text": resume_text,
    }

def _extract_name(text: str) -> Optional[str]:
    for line in [x.strip() for x in text.splitlines() if x.strip()][:8]:
        low = line.lower()
        if any(x in low for x in ["resume", "curriculum", "@", "phone", "email", "linkedin", "job description"]):
            continue
        if 1 <= len(line.split()) <= 5 and re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", line):
            return line
    return None

def _extract_location(text: str) -> Optional[str]:
    # Prefer an explicit LOCATION & WORK MODE section over incidental mentions.
    m = re.search(r"location\s*(?:&|and)\s*work mode\s*\n\s*([^\n.]+)", text, re.I)
    if m:
        section = m.group(1)
        for city in ["Navi Mumbai", "Mumbai", "Thane", "Pune", "Delhi", "Bengaluru", "Bangalore", "Hyderabad", "Chennai"]:
            if re.search(r"\b" + re.escape(city) + r"\b", section, re.I):
                return city
    m = re.search(r"(?:based in|located in|location)\s*:?\s*([^\n.,]+)", text, re.I)
    if m:
        value = m.group(1).strip()
        for city in ["Navi Mumbai", "Mumbai", "Thane", "Pune", "Delhi", "Bengaluru", "Bangalore", "Hyderabad", "Chennai"]:
            if re.search(r"\b" + re.escape(city) + r"\b", value, re.I):
                return city
    cities = ["Navi Mumbai", "Mumbai", "Thane", "Pune", "Delhi", "Bengaluru", "Bangalore", "Hyderabad", "Chennai"]
    for city in cities:
        if re.search(r"\b" + re.escape(city) + r"\b", text, re.I):
            return city
    return None

def _match_skill(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    found = [p for p in req.get("keywords", []) if _contains(candidate["raw_text"], p)]
    canonical = req["requirement"]
    if canonical in candidate.get("skills", []):
        strength, status = 1.0, "matched"
    elif found:
        strength, status = 0.75, "matched"
    else:
        strength, status = 0.0, "missing"
    evidence = candidate.get("evidence", {}).get(canonical) if canonical in candidate.get("evidence", {}) else None
    if not evidence and found:
        evidence = _find_evidence(candidate["raw_text"], found[0])
    return {"requirement": canonical, "status": status, "strength": strength, "evidence": evidence}

def _match_experience(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    required = float(req.get("required_years", 0))
    actual = float(candidate.get("experience_years", 0))
    if actual >= required:
        strength, status = 1.0, "matched"
    elif required == 0:
        strength, status = 1.0, "matched"
    elif actual >= required * 0.5:
        strength, status = 0.5, "partial"
    else:
        strength, status = 0.0, "missing"
    return {
        "requirement": req["requirement"], "status": status, "strength": strength,
        "evidence": f"Resume indicates approximately {actual:g} years of experience."
        if actual else "No clear experience duration found."
    }

def _match_education(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    required = req.get("required_degree_rank", 0)
    actual = candidate.get("degree_rank", 0)
    raw = candidate.get("raw_text", "").lower()
    relevant_fields = req.get("relevant_fields", [])
    field_match = not relevant_fields or any(x in raw for x in relevant_fields)

    if actual >= required and actual > 0 and field_match:
        strength, status = 1.0, "matched"
    elif actual > 0:
        strength, status = 0.5, "partial"
    else:
        strength, status = 0.0, "missing"

    evidence = candidate.get("education") if actual else "No relevant degree evidence found."
    if actual and relevant_fields and not field_match:
        evidence = f"{candidate.get('education', 'Degree found')} — degree level is present, but the resume does not show a target field ({', '.join(relevant_fields)})."

    return {
        "requirement": req["requirement"], "status": status, "strength": strength,
        "evidence": evidence
    }

def _match_project(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    text = candidate.get("raw_text", "")
    found = any(_contains(text, x) for x in req.get("keywords", []))
    projects = candidate.get("projects", [])
    if projects or found:
        evidence = projects[0] if projects else _find_evidence(text, "project")
        return {"requirement": req["requirement"], "status": "matched", "strength": 1.0, "evidence": evidence}
    return {"requirement": req["requirement"], "status": "missing", "strength": 0.0, "evidence": "No concrete project or equivalent technical evidence found."}

def _match_requirement(req: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    typ = req.get("type", "skill")
    if typ == "experience": return _match_experience(req, candidate)
    if typ == "education": return _match_education(req, candidate)
    if typ == "project": return _match_project(req, candidate)
    return _match_skill(req, candidate)

def _score(results: List[Dict[str, Any]]) -> Optional[float]:
    if not results: return None
    return round(sum(x["strength"] for x in results) / len(results) * 100, 1)

def _recommendation(score: float, critical: bool) -> str:
    # A candidate with a critical mandatory failure should never be presented
    # as an ordinary "Strong/Weak" fit.
    if critical:
        return "Potential Match" if score >= 75 else "Poor Match"
    for threshold, label in RECOMMENDATION_BANDS:
        if score >= threshold: return label
    return "Poor Match"

def match_candidate(candidate: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    groups = {
        "must_have": profile.get("must_have", []),
        "preferred": profile.get("preferred", []),
        "nice_to_have": profile.get("nice_to_have", []),
    }
    # Keep typed requirements in their own scoring categories so they are not
    # double-counted inside the skill-category score.
    results_by_group = {
        k: [_match_requirement(r, candidate) for r in v if r.get("type", "skill") == "skill"]
        for k, v in groups.items()
    }
    exp_reqs = [r for v in groups.values() for r in v if r.get("type") == "experience"]
    edu_reqs = [r for v in groups.values() for r in v if r.get("type") == "education"]
    project_reqs = [r for v in groups.values() for r in v if r.get("type") == "project"]
    exp_results = [_match_requirement(r, candidate) for r in exp_reqs]
    edu_results = [_match_requirement(r, candidate) for r in edu_reqs]

    # Location is a soft/critical eligibility signal, not a keyword score.
    location_result = _location_match(profile, candidate)

    scores = {
        "must_have": _score(results_by_group["must_have"]),
        "preferred": _score(results_by_group["preferred"]),
        "experience": _score(exp_results),
        "education": _score(edu_results),
        "location": location_result["score"],
        "nice_to_have": _score(results_by_group["nice_to_have"]),
    }
    present = {k: v for k, v in scores.items() if v is not None}
    weights = {k: CATEGORY_WEIGHTS[k] for k in present}
    total_w = sum(weights.values()) or 1
    overall = round(sum(present[k] * weights[k] / total_w for k in present), 1)

    all_defs = []
    all_results = []
    for key in ("must_have", "preferred", "nice_to_have"):
        skill_defs = [r for r in groups[key] if r.get("type", "skill") == "skill"]
        all_defs.extend(skill_defs)
        all_results.extend(results_by_group[key])
    typed_defs = exp_reqs + edu_reqs + project_reqs
    typed_results = exp_results + edu_results + [_match_requirement(r, candidate) for r in project_reqs]
    all_defs.extend(typed_defs)
    all_results.extend(typed_results)

    matched = [r for r in all_results if r["status"] == "matched"]
    partial = [r for r in all_results if r["status"] == "partial"]
    missing = [{"requirement": r["requirement"], "importance": d.get("category", "must_have")}
               for d, r in zip(all_defs, all_results) if r["status"] == "missing"]

    critical = any(d.get("mandatory") and r["status"] == "missing" for d, r in zip(all_defs, all_results))
    # Location can be a deal-breaker only when the JD explicitly requires a city
    # and the candidate's text clearly says they cannot/won't attend. A different
    # nearby city is treated as a partial fit.
    if location_result["critical_failure"]:
        critical = True

    concerns = []
    if critical:
        failed = [r["requirement"] for d, r in zip(all_defs, all_results)
                  if d.get("mandatory") and r["status"] == "missing"]
        if location_result["critical_failure"]: failed.append("Mumbai hybrid attendance")
        concerns.append("Fails mandatory requirement(s): " + ", ".join(failed))
    if location_result["status"] == "partial":
        concerns.append("Location/work-mode fit needs confirmation.")
    if scores["experience"] is not None and scores["experience"] < 75:
        concerns.append("Experience evidence is below the target.")
    if not concerns:
        concerns.append("No significant concerns identified.")

    summary = f"{_recommendation(overall, critical)} at {overall}/100. "
    top = ", ".join(x["requirement"] for x in matched[:3])
    if top: summary += f"Strong evidence for {top}. "
    if missing: summary += f"Main gap: {missing[0]['requirement']}."
    else: summary += "No major requirement gaps identified."

    return {
        "candidate_id": candidate["candidate_id"],
        "name": candidate["name"],
        "overall_score": overall,
        "recommendation": _recommendation(overall, critical),
        "critical_mandatory_failure": critical,
        "category_scores": scores,
        "matched_requirements": matched,
        "partial_matches": partial,
        "missing_requirements": missing,
        "location_fit": location_result,
        "concerns": concerns,
        "summary": summary,
    }

def _location_match(profile: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    required = (profile.get("location") or "").lower()
    cand = (candidate.get("location") or "").lower()
    if not required:
        return {"status": "not_applicable", "score": None, "evidence": None, "critical_failure": False}
    # Mumbai metro locations are considered potentially commutable for the demo.
    if "mumbai" in required:
        if "mumbai" in cand and "navi" not in cand:
            return {"status": "matched", "score": 100.0, "evidence": candidate_location_evidence(candidate), "critical_failure": False}
        if any(x in cand for x in ["thane", "navi mumbai"]):
            return {"status": "partial", "score": 60.0, "evidence": candidate_location_evidence(candidate), "critical_failure": False}
        if cand in {"not specified", ""}:
            return {"status": "unknown", "score": 40.0, "evidence": "Candidate location not clearly specified.", "critical_failure": False}
        return {"status": "weak", "score": 20.0, "evidence": candidate_location_evidence(candidate), "critical_failure": False}
    return {"status": "unknown", "score": 50.0, "evidence": candidate_location_evidence(candidate), "critical_failure": False}

def candidate_location_evidence(candidate: Dict[str, Any]) -> str:
    return f"Candidate location: {candidate.get('location', 'Not specified')}."

def rank_candidates(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(results, key=lambda r: (r.get("critical_mandatory_failure", False), -r.get("overall_score", 0)))
    for i, result in enumerate(ranked, 1):
        result["rank"] = i
    return ranked
