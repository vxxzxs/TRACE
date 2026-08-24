import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def analyze_resume(resume_text: str) -> dict:
    prompt = f"""You are a recruiting analysis engine. Read this resume and extract a structured Candidate Profile.

RESUME:
{resume_text}

Return ONLY valid JSON in this exact format, no other text, no markdown:
{{
  "name": "<candidate name, or 'Unknown Candidate' if not found>",
  "skills": ["<skill1>", "<skill2>"],
  "experience_years": <number, total years of relevant experience>,
  "education": "<highest degree and field>",
  "location": "<city, or 'Not specified' if not found>",
  "evidence": {{
    "<skill name>": "<one sentence quoting/describing how this skill shows up in the resume>"
  }}
}}

List all skills you can find. In "evidence", include an entry for each skill in your skills list, explaining specifically how/where it appears in the resume (project, job, course, etc.) — not just that it's listed."""

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
