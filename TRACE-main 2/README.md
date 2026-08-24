# TRACE
### Transparent Requirement-to-Candidate Evaluation

TRACE is a hackathon prototype for explainable AI-assisted recruitment.

Instead of treating recruitment as keyword matching, TRACE converts a Job Description into an **Ideal Recruit Profile**, structures each candidate resume, evaluates requirements individually, and produces a weighted candidate ranking with evidence and gaps.

## Product flow

```text
Job Description
      ↓
Ideal Recruit Profile
  ├─ Must Have
  ├─ Preferred
  ├─ Nice to Have
  ├─ Experience
  ├─ Education
  └─ Location / Work Mode
      ↓
Resume Parsing
      ↓
Candidate Profile
      ↓
Requirement-by-Requirement Matching
      ↓
Weighted Score + Mandatory Checks + Evidence
      ↓
Ranked Candidates
      ↓
Recruiter Explanation
```

## Repository

```text
TRACE-main/
├── backend/
│   ├── main.py
│   ├── schemas.py
│   ├── requirements.txt
│   ├── .env.example
│   └── services/
│       ├── __init__.py
│       ├── matcher.py
│       ├── job_analyzer.py
│       └── resume_analyzer.py
├── Frontend/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── jd_junior_ai_ml_engineer-6.txt
│   └── resume_*.pdf
└── README.md
```

## Architecture

### Backend
- **FastAPI**: API layer
- **matcher.py**: deterministic recruitment intelligence and scoring
- **job_analyzer.py**: JD extraction; uses Mistral when configured, otherwise deterministic fallback
- **resume_analyzer.py**: resume extraction; uses Mistral when configured, otherwise deterministic fallback
- **pypdf**: PDF text extraction
- **Pydantic**: request validation

### Frontend
- **Streamlit**
- Uploads PDF/TXT resumes
- Calls FastAPI
- Displays Ideal Recruit Profile, ranking, scores, evidence and concerns

## Important prototype behavior

The matcher is intentionally deterministic. This means the demo still works without an LLM API key.

If `MISTRAL_API_KEY` is present, the JD/resume analyzers attempt Mistral extraction first. If that call fails, the system automatically falls back to deterministic extraction.

The score is explainable rather than an opaque model output.

Default scoring weights:

- Must Have: 50%
- Preferred: 25%
- Experience: 10%
- Education: 5%
- Location: 5%
- Nice to Have: 5%

Weights are automatically renormalized when a category is not present.

Mandatory requirements can prevent a candidate from ranking above candidates who satisfy all critical requirements.

## Run locally

### 1. Create a virtual environment

From the `TRACE-main` directory:

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Optional: configure Mistral

Copy:

```text
backend/.env.example
```

to:

```text
backend/.env
```

and set:

```text
MISTRAL_API_KEY=your_key_here
```

You can also set:

```text
MISTRAL_MODEL=mistral-large-latest
```

**The application does not require an API key for the deterministic demo.**

Never commit a real API key or `.env` file.

### 4. Start FastAPI

Open Terminal 1:

```bash
cd backend
uvicorn main:app --reload
```

Backend:

```text
http://localhost:8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

### 5. Install frontend dependencies

Open Terminal 2 from `TRACE-main`:

```bash
pip install -r Frontend/requirements.txt
```

### 6. Start Streamlit

```bash
streamlit run Frontend/app.py
```

Open the URL Streamlit prints, normally:

```text
http://localhost:8501
```

## Demo workflow

1. Open the Streamlit frontend.
2. Paste the supplied JD from `data/jd_junior_ai_ml_engineer-6.txt`.
3. Click **Generate Ideal Recruit Profile**.
4. Review Must Have / Preferred / Experience / Education / Location.
5. Upload the four supplied resume PDFs.
6. Click **Analyze & Rank Candidates**.
7. Open the ranking.
8. Go to **Evidence**.
9. Select the top candidate.
10. Show matched requirements, evidence, missing requirements, location fit and concerns.

The supplied data is intentionally calibrated into different candidate-quality tiers so the ranking demonstrates the system's decision-making.

## API

### Health

```http
GET /
GET /health
```

### Analyze JD

```http
POST /analyze-job
Content-Type: application/json

{
  "job_description": "..."
}
```

### Analyze resume text

```http
POST /analyze-resume
Content-Type: application/json

{
  "resume_text": "...",
  "candidate_id": "candidate_001",
  "name": "Candidate A"
}
```

### Analyze resume file

```http
POST /analyze-resume-file
multipart/form-data
resume=<PDF or TXT>
```

### Match candidates

```http
POST /match
Content-Type: application/json

{
  "profile": { "...": "..." },
  "candidates": [
    {
      "profile": { "...": "..." },
      "name": "Candidate A"
    }
  ]
}
```

## What the score means

Each requirement receives a match strength:

```text
1.00  Direct/strong evidence
0.75  Relevant evidence
0.50  Partial evidence
0.00  Missing
```

The final score is the weighted combination of requirement categories.

The system also separately reports:

- matched requirements
- partial matches
- missing requirements
- mandatory failures
- location fit
- concerns
- evidence
- recruiter-facing summary

## Hackathon positioning

TRACE's key distinction is:

> **It does not simply ask whether a resume contains the right keywords. It evaluates how well the candidate satisfies the actual requirements of the job, and shows the evidence behind that decision.**

The current prototype is intentionally optimized for a reliable end-to-end demonstration rather than production-scale deployment.
