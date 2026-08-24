import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def analyze_job(job_description: str) -> dict:
    prompt = f"""You are a recruiting analysis engine. Read this job description and extract a structured Ideal Recruit Profile.

JOB DESCRIPTION:
{job_description}

Return ONLY valid JSON in this exact format, no other text, no markdown:
{{
  "role": "<job title>",
  "must_have": [
    {{"name": "<skill name>", "weight": <number 0.0 to 1.0>}}
  ],
  "preferred": [
    {{"name": "<skill name>", "weight": <number 0.0 to 1.0>}}
  ],
  "experience": {{
    "minimum_years": <number>,
    "weight": <number 0.0 to 1.0>
  }},
  "avoid": []
}}

Weight means how important that skill is to the role (1.0 = critical, 0.5 = somewhat important). List 3-6 must_have skills and 2-4 preferred skills based on what's actually in the job description."""

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
