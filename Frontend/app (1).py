"""
AI Recruiter Copilot — Streamlit frontend
Talks to a FastAPI backend exposing:
  POST /analyze-job     -> Ideal Recruit Profile
  POST /analyze-resume  -> Candidate Profile
  POST /match           -> Rankings + evidence

1 page, 4 sections (Screens 1-4):
  1. Job Input
  2. Ideal Recruit Profile
  3. Candidate Ranking
  4. Candidate Evidence
"""

import streamlit as st
import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="AI Recruiter Copilot", page_icon="🧭", layout="wide")

DEFAULTS = {
    "backend_url": "http://localhost:8000",
    "job_description": "",
    "profile": None,          # dict returned by /analyze-job
    "candidates_raw": [],     # list of {"name": str, "text": str}
    "rankings": None,         # dict returned by /match  -> {"rankings": [...]}
    "selected_candidate": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.sidebar.header("⚙️ Backend")
st.session_state.backend_url = st.sidebar.text_input(
    "FastAPI base URL", value=st.session_state.backend_url
)
st.sidebar.caption("Expects POST /analyze-job, /analyze-resume, /match")


def api_post(path: str, payload: dict, timeout: int = 60):
    url = st.session_state.backend_url.rstrip("/") + path
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to backend at {url}. Is FastAPI running?"
    except requests.exceptions.Timeout:
        return None, f"Request to {path} timed out."
    except requests.exceptions.HTTPError as e:
        return None, f"Backend error on {path}: {e} — {resp.text[:300]}"
    except Exception as e:
        return None, f"Unexpected error calling {path}: {e}"


def match_icon(match: str) -> str:
    return {"strong": "✅ Strong", "partial": "🔶 Partial", "missing": "❌ Missing"}.get(
        (match or "").lower(), f"❔ {match}"
    )


def medal(i: int) -> str:
    return {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, f"#{i + 1}")


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
st.title("🧭 AI Recruiter Copilot")
st.caption("Job description → ideal recruit profile → candidate ranking → evidence.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Job Input",
    "2️⃣ Ideal Recruit Profile",
    "3️⃣ Candidate Ranking",
    "4️⃣ Candidate Evidence",
])

# --------------------------------------------------------------------------
# Screen 1 — Job Input
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Job Description")
    st.session_state.job_description = st.text_area(
        "Paste the job description",
        value=st.session_state.job_description,
        height=260,
        placeholder="e.g. We are looking for a Machine Learning Engineer with 2+ years experience in Python...",
    )

    if st.button("✨ Generate Profile", type="primary", disabled=not st.session_state.job_description):
        with st.spinner("Analyzing job description..."):
            data, err = api_post("/analyze-job", {"job_description": st.session_state.job_description})
        if err:
            st.error(err)
        else:
            st.session_state.profile = data
            st.session_state.rankings = None  # invalidate downstream results
            st.success("Ideal Recruit Profile generated — see Section 2.")

# --------------------------------------------------------------------------
# Screen 2 — Ideal Recruit Profile
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Ideal Recruit Profile")
    profile = st.session_state.profile

    if not profile:
        st.info("Generate a profile in Section 1 first.")
    else:
        st.markdown(f"### {profile.get('role', 'Role')}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**MUST HAVE**")
            for item in profile.get("must_have", []):
                st.progress(
                    float(item.get("weight", 0)),
                    text=f"{item.get('name')} — {round(item.get('weight', 0) * 100)}%",
                )
        with c2:
            st.markdown("**PREFERRED**")
            for item in profile.get("preferred", []):
                st.progress(
                    float(item.get("weight", 0)),
                    text=f"{item.get('name')} — {round(item.get('weight', 0) * 100)}%",
                )

        exp = profile.get("experience", {})
        if exp:
            st.markdown("**Experience**")
            st.write(
                f"Minimum {exp.get('minimum_years', '—')} years "
                f"(weight {round(exp.get('weight', 0) * 100)}%)"
            )

        avoid = profile.get("avoid", [])
        if avoid:
            st.markdown("**Avoid / red flags**")
            st.write(", ".join(avoid))

        with st.expander("Raw profile JSON"):
            st.json(profile)

# --------------------------------------------------------------------------
# Screen 3 — Candidate Ranking
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Candidates")

    if not st.session_state.profile:
        st.info("Generate the Ideal Recruit Profile in Section 2 first.")
    else:
        uploaded = st.file_uploader(
            "Upload resume .txt files (one per candidate)",
            type=["txt"],
            accept_multiple_files=True,
        )
        if uploaded:
            existing_names = {c["name"] for c in st.session_state.candidates_raw}
            for f in uploaded:
                name = f.name.rsplit(".", 1)[0]
                if name not in existing_names:
                    st.session_state.candidates_raw.append(
                        {"name": name, "text": f.read().decode("utf-8", errors="ignore")}
                    )
            st.success(f"Loaded {len(uploaded)} resume file(s).")

        with st.expander("➕ Add a candidate manually"):
            with st.form("add_candidate", clear_on_submit=True):
                cname = st.text_input("Candidate name")
                ctext = st.text_area("Resume text", height=150)
                if st.form_submit_button("Add candidate") and cname and ctext:
                    st.session_state.candidates_raw.append({"name": cname, "text": ctext})
                    st.success(f"Added {cname}.")

        if st.session_state.candidates_raw:
            st.markdown(f"**{len(st.session_state.candidates_raw)} candidate(s) staged:** "
                        + ", ".join(c["name"] for c in st.session_state.candidates_raw))
            if st.button("🗑️ Clear candidates"):
                st.session_state.candidates_raw = []
                st.session_state.rankings = None
                st.rerun()

        st.divider()

        if st.button(
            "🔍 Analyze Candidates",
            type="primary",
            disabled=not st.session_state.candidates_raw,
        ):
            candidate_profiles = []
            progress = st.progress(0.0, text="Analyzing resumes...")
            total = len(st.session_state.candidates_raw)
            failed = []

            for i, cand in enumerate(st.session_state.candidates_raw):
                data, err = api_post("/analyze-resume", {"resume_text": cand["text"]})
                if err:
                    failed.append((cand["name"], err))
                else:
                    if "name" not in data or not data["name"]:
                        data["name"] = cand["name"]
                    candidate_profiles.append(data)
                progress.progress((i + 1) / total, text=f"Analyzed {cand['name']}")

            for name, err in failed:
                st.warning(f"Skipped {name}: {err}")

            if candidate_profiles:
                with st.spinner("Running matching engine..."):
                    result, err = api_post(
                        "/match",
                        {"profile": st.session_state.profile, "candidates": candidate_profiles},
                    )
                if err:
                    st.error(err)
                else:
                    st.session_state.rankings = result
                    st.success("Ranking complete — see results below and Section 4 for evidence.")

        rankings = (st.session_state.rankings or {}).get("rankings")
        if rankings:
            st.markdown("### Ranked candidates")
            for i, r in enumerate(sorted(rankings, key=lambda x: x.get("score", 0), reverse=True)):
                st.markdown(f"{medal(i)}  **{r['candidate']}** — {r['score']}%")
                st.progress(min(max(r.get("score", 0) / 100, 0.0), 1.0))

# --------------------------------------------------------------------------
# Screen 4 — Candidate Evidence
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Candidate Evidence")

    rankings = (st.session_state.rankings or {}).get("rankings")
    if not rankings:
        st.info("Rank candidates in Section 3 first.")
    else:
        ordered = sorted(rankings, key=lambda x: x.get("score", 0), reverse=True)
        names = [r["candidate"] for r in ordered]
        choice = st.selectbox("Select a candidate", names)
        st.session_state.selected_candidate = choice

        chosen = next(r for r in ordered if r["candidate"] == choice)

        st.markdown(f"## {choice} — {chosen.get('score', 0)}%")
        st.progress(min(max(chosen.get("score", 0) / 100, 0.0), 1.0))

        analysis = chosen.get("analysis", [])
        if analysis:
            st.markdown("### Requirement match")
            for item in analysis:
                with st.container(border=True):
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"**{item.get('requirement', '—')}**")
                        st.write(match_icon(item.get("match")))
                    with c2:
                        st.markdown("**Evidence**")
                        st.write(item.get("evidence") or "—")
        else:
            st.write("No requirement-level analysis returned for this candidate.")

        with st.expander("Raw match JSON"):
            st.json(chosen)
