"""TRACE Streamlit demo frontend."""
from __future__ import annotations
import os
import requests
import streamlit as st

st.set_page_config(page_title="TRACE — AI Recruiter", page_icon="◈", layout="wide")

if "backend_url" not in st.session_state: st.session_state.backend_url = "http://localhost:8000"
if "job_description" not in st.session_state: st.session_state.job_description = ""
if "profile" not in st.session_state: st.session_state.profile = None
if "candidates_raw" not in st.session_state: st.session_state.candidates_raw = []
if "rankings" not in st.session_state: st.session_state.rankings = None

st.sidebar.title("TRACE")
st.sidebar.caption("Transparent Requirement-to-Candidate Evaluation")
st.session_state.backend_url = st.sidebar.text_input("FastAPI URL", st.session_state.backend_url)
st.sidebar.caption("Backend must be running before analysis.")

def api_json(path, payload, timeout=90):
    url = st.session_state.backend_url.rstrip("/") + path
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to {url}. Start the FastAPI backend first."
    except requests.exceptions.Timeout:
        return None, f"{path} timed out. Check the backend/API key."
    except requests.exceptions.HTTPError:
        return None, f"{path} failed ({r.status_code}): {r.text[:500]}"
    except Exception as e:
        return None, str(e)

def api_file(path, file_obj, timeout=90):
    url = st.session_state.backend_url.rstrip("/") + path
    try:
        file_obj.seek(0)
        r = requests.post(url, files={"resume": (file_obj.name, file_obj.getvalue(), file_obj.type)}, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to {url}. Start the FastAPI backend first."
    except requests.exceptions.Timeout:
        return None, f"{path} timed out."
    except requests.exceptions.HTTPError:
        return None, f"{path} failed ({r.status_code}): {r.text[:500]}"
    except Exception as e:
        return None, str(e)

def item_name(item):
    return item.get("requirement") or item.get("name") or "Requirement"

def score_bar(value):
    return max(0.0, min(float(value or 0) / 100, 1.0))

st.title("TRACE")
st.caption("From job description → Ideal Recruit Profile → evidence-backed candidate ranking.")

tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Job", "02 · Ideal Profile", "03 · Ranking", "04 · Evidence"
])

with tab1:
    st.subheader("Job Description")
    st.session_state.job_description = st.text_area(
        "Paste the complete JD",
        st.session_state.job_description,
        height=320,
        placeholder="Paste a job description containing requirements, preferred skills, experience, education, location and constraints.",
    )
    if st.button("Generate Ideal Recruit Profile", type="primary", disabled=not st.session_state.job_description.strip()):
        with st.spinner("Extracting requirements..."):
            data, err = api_json("/analyze-job", {"job_description": st.session_state.job_description})
        if err:
            st.error(err)
        else:
            st.session_state.profile = data
            st.session_state.rankings = None
            st.success("Profile generated.")
    if st.session_state.profile:
        p = st.session_state.profile
        st.caption(f"Role: {p.get('role','Not specified')} · Company: {p.get('company') or 'Not specified'} · Source: {p.get('source','unknown')}")

with tab2:
    p = st.session_state.profile
    st.subheader("Ideal Recruit Profile")
    if not p:
        st.info("Generate the profile in Job first.")
    else:
        st.markdown(f"### {p.get('role', 'Role')}")
        if p.get("location") or p.get("work_mode"):
            st.info(f"Work fit: {p.get('location') or 'Location not specified'} · {p.get('work_mode') or 'Mode not specified'}")
        cols = st.columns(3)
        for col, title, key in zip(cols, ["MUST HAVE", "PREFERRED", "NICE TO HAVE"], ["must_have","preferred","nice_to_have"]):
            with col:
                st.markdown(f"**{title}**")
                items = p.get(key, [])
                if not items: st.caption("None extracted")
                for x in items:
                    st.write(f"• {item_name(x)}")
                    if isinstance(x, dict) and "importance" in x:
                        st.progress(score_bar(float(x["importance"])), text=f"Importance {float(x['importance']):.0%}")
        e = p.get("experience") or {}
        edu = p.get("education") or {}
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Minimum experience", f"{e.get('minimum_years', 0):g} years")
        with c2:
            st.metric("Education", edu.get("required") or "Flexible")
        if p.get("avoid"):
            st.markdown("**Risk / deal-breaker signals**")
            for x in p["avoid"]: st.write(f"• {x}")
        with st.expander("Raw profile"):
            st.json(p)

with tab3:
    p = st.session_state.profile
    st.subheader("Candidate Ranking")
    if not p:
        st.info("Generate an Ideal Recruit Profile first.")
    else:
        uploaded = st.file_uploader(
            "Upload candidate resumes",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="resume_uploader",
        )
        if uploaded:
            known = {x["name"] for x in st.session_state.candidates_raw}
            for f in uploaded:
                if f.name not in known:
                    st.session_state.candidates_raw.append({"name": f.name, "file": f})
            st.success(f"{len(uploaded)} file(s) staged.")
        if st.session_state.candidates_raw:
            st.write("**Staged:** " + ", ".join(x["name"] for x in st.session_state.candidates_raw))
            if st.button("Clear staged resumes"):
                st.session_state.candidates_raw = []
                st.session_state.rankings = None
                st.rerun()
        if st.button("Analyze & Rank Candidates", type="primary", disabled=not st.session_state.candidates_raw):
            profiles = []
            failures = []
            progress = st.progress(0.0)
            for i, cand in enumerate(st.session_state.candidates_raw):
                data, err = api_file("/analyze-resume-file", cand["file"])
                if err: failures.append(f"{cand['name']}: {err}")
                else: profiles.append(data)
                progress.progress((i+1)/len(st.session_state.candidates_raw), text=f"Analyzing {cand['name']}")
            for x in failures: st.warning(x)
            if profiles:
                result, err = api_json("/match", {"profile": p, "candidates": [{"profile": x, "name": x.get("name","Unknown")} for x in profiles]})
                if err: st.error(err)
                else:
                    st.session_state.rankings = result
                    st.success("Ranking complete.")
        rankings = (st.session_state.rankings or {}).get("rankings", [])
        if rankings:
            for r in rankings:
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### #{r.get('rank')} · {r.get('name')}")
                    st.progress(score_bar(r.get("overall_score")), text=f"{r.get('overall_score')}% · {r.get('recommendation')}")
                with right:
                    if r.get("critical_mandatory_failure"):
                        st.error("Critical gap")
                    else:
                        st.success("Eligible")
                st.caption(r.get("summary",""))

with tab4:
    rankings = (st.session_state.rankings or {}).get("rankings", [])
    st.subheader("Why this candidate scored this way")
    if not rankings:
        st.info("Run candidate analysis first.")
    else:
        choice = st.selectbox("Candidate", [r.get("name") for r in rankings])
        chosen = next(r for r in rankings if r.get("name") == choice)
        st.markdown(f"## {chosen['name']} · {chosen['overall_score']}%")
        st.progress(score_bar(chosen["overall_score"]), text=chosen["recommendation"])
        if chosen.get("category_scores"):
            cols = st.columns(6)
            for col, (k,v) in zip(cols, chosen["category_scores"].items()):
                with col:
                    if v is not None: st.metric(k.replace("_"," ").title(), f"{v:.0f}%")
        st.markdown("### Requirement evidence")
        for section, title in [
            ("matched_requirements", "Matched"),
            ("partial_matches", "Partial"),
            ("missing_requirements", "Missing"),
        ]:
            items = chosen.get(section, [])
            if items:
                st.markdown(f"**{title}**")
                for x in items:
                    if section == "missing_requirements":
                        st.write(f"❌ {x.get('requirement')} · {x.get('importance')}")
                    else:
                        icon = "✓" if x.get("status") == "matched" else "△"
                        st.write(f"{icon} **{x.get('requirement')}** — {x.get('strength',0):.0%}")
                        if x.get("evidence"): st.caption(x["evidence"])
        if chosen.get("location_fit"):
            st.markdown("**Location fit**")
            st.write(chosen["location_fit"])
        if chosen.get("concerns"):
            st.markdown("**Concerns**")
            for x in chosen["concerns"]: st.write(f"• {x}")
        st.markdown("**Decision explanation**")
        st.write(chosen.get("summary",""))
