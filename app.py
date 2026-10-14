"""
SkillProof AI — Evidence-Based Career Readiness Engine
Streamlit Application using google-genai SDK, GitHub REST API, pypdf, pandas, and plotly.
"""

import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pypdf import PdfReader
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Attempt importing google-genai SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="SkillProof AI — Career Readiness Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 4px;
    }
    .strong-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .weak-badge {
        background-color: #fef9c3;
        color: #a16207;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .none-badge {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> str:
    """Extract all readable text from an uploaded PDF resume."""
    try:
        reader = PdfReader(pdf_file)
        pages_text = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text).strip()
    except Exception as e:
        st.error(f"Failed to parse PDF resume: {str(e)}")
        return ""


def clean_llm_json_response(raw_text: str) -> Any:
    """Safely sanitize and decode JSON response from LLM output."""
    cleaned = raw_text.strip()
    # Strip markdown backticks
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        # Fallback regex search for json block
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Unable to parse response as JSON. Raw preview: {raw_text[:200]}")


def fetch_github_profile_and_repos(
    username: str,
    token: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[str]]:
    """Fetch user profile metadata and public repositories via GitHub REST API."""
    clean_username = username.strip().lstrip("@")
    if not clean_username:
        return None, None, "GitHub username cannot be empty."

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SkillProof-AI"
    }
    if token and token.strip():
        headers["Authorization"] = f"token {token.strip()}"

    # 1. Fetch user profile
    user_url = f"https://api.github.com/users/{clean_username}"
    try:
        user_res = requests.get(user_url, headers=headers, timeout=12)
        if user_res.status_code == 404:
            return None, None, f"GitHub account '{clean_username}' was not found. Please double-check the handle."
        elif user_res.status_code == 403:
            return None, None, "GitHub API rate limit exceeded. Please provide an optional GITHUB_TOKEN in the sidebar."
        elif user_res.status_code != 200:
            return None, None, f"GitHub API error ({user_res.status_code}): {user_res.text}"
        user_profile = user_res.json()
    except Exception as e:
        return None, None, f"Network error while connecting to GitHub: {str(e)}"

    # 2. Fetch public repositories
    repos_url = f"https://api.github.com/users/{clean_username}/repos"
    params = {"per_page": 100, "sort": "updated", "type": "all"}
    try:
        repos_res = requests.get(repos_url, headers=headers, params=params, timeout=15)
        if repos_res.status_code != 200:
            return user_profile, None, f"GitHub API repos error ({repos_res.status_code}): {repos_res.text}"
        repos_data = repos_res.json()
        return user_profile, repos_data, None
    except Exception as e:
        return user_profile, None, f"Error fetching repositories: {str(e)}"


def extract_claimed_skills_gemini(
    client: genai.Client,
    resume_text: str
) -> List[Dict[str, str]]:
    """Use Gemini 2.5 Flash to extract claimed technical skills as structured JSON."""
    prompt = f"""
You are an expert technical recruiter and resume analysis engine.
Extract all explicitly claimed technical skills from the candidate's resume text below.

Include:
- Programming Languages (e.g. Python, TypeScript, C++, Rust, Go, SQL)
- Frameworks & Libraries (e.g. React, Next.js, Django, FastAPI, PyTorch, Node.js)
- Databases & Storage (e.g. PostgreSQL, Redis, MongoDB, Snowflake)
- Cloud & DevOps (e.g. Docker, Kubernetes, AWS, GCP, GitHub Actions, Terraform)
- Developer Tools & Specialized Architectures (e.g. Git, GraphQL, Kafka, REST APIs, Microservices)

Resume Content:
\"\"\"
{resume_text}
\"\"\"

Output Requirement:
Respond with ONLY a valid JSON array of objects. Do not include markdown explanation.
Each object must contain:
- "skill": Standard canonical name of the skill (e.g., "Python", "Docker", "PostgreSQL")
- "category": One of ["Languages", "Frameworks & Libraries", "Databases", "Cloud & DevOps", "Tools & Architecture"]
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    raw_output = response.text or "[]"
    parsed = clean_llm_json_response(raw_output)
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict) and "skills" in parsed:
        return parsed["skills"]
    return []


def verify_skills_with_gemini(
    client: genai.Client,
    claimed_skills: List[Dict[str, str]],
    repos: List[Dict[str, Any]],
    github_username: str
) -> List[Dict[str, Any]]:
    """Cross-examine claimed skills against public GitHub activity using Gemini 2.5 Flash."""
    # Summarize repos to give compact, high-signal context
    condensed_repos = []
    for r in repos:
        condensed_repos.append({
            "name": r.get("name"),
            "description": r.get("description") or "No description",
            "language": r.get("language") or "Not specified",
            "stars": r.get("stargazers_count", 0),
            "fork": r.get("fork", False),
            "topics": r.get("topics", [])
        })

    prompt = f"""
You are an automated evidence-based skill verification system.
Your job is to compare a candidate's claimed technical skills from their resume against their public GitHub repositories.

Candidate GitHub Handle: {github_username}

Claimed Skills:
{json.dumps(claimed_skills, indent=2)}

Public GitHub Repositories ({len(condensed_repos)} repos indexed):
{json.dumps(condensed_repos, indent=2)}

Verification Rules:
Categorize each skill into EXACTLY ONE of the following three categories:
1. "Strong Public Evidence":
   - The candidate has original public repositories where this skill is the primary language, framework, or explicitly documented core dependency.
   - Significant implementation or starred repository evidence is present.
2. "Weak Public Evidence":
   - Minor, incidental evidence found (e.g., a fork with minimal original edits, a basic single-file configuration, toy starter repo, or only mentioned in a topic/readme without core implementation).
3. "No Public Evidence":
   - Zero traces found across public repository languages, descriptions, names, or topics.

Output Format:
Return ONLY a valid JSON array of objects. Each object must have:
- "skill": name of the skill
- "category": category of the skill
- "status": exactly "Strong Public Evidence", "Weak Public Evidence", or "No Public Evidence"
- "reason": crisp 1-sentence factual explanation citing specific repos or lack of evidence
- "evidence_repos": list of matching repository names (empty if none)
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    raw_output = response.text or "[]"
    parsed = clean_llm_json_response(raw_output)
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict) and "results" in parsed:
        return parsed["results"]
    return []


def calculate_public_proof_index(verification_results: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    """
    Calculate Public Proof Index (0-100%):
    - Strong Public Evidence: 1.0 weight
    - Weak Public Evidence: 0.4 weight
    - No Public Evidence: 0.0 weight
    """
    total = len(verification_results)
    if total == 0:
        return 0.0, 0, 0, 0

    strong_cnt = sum(1 for r in verification_results if r.get("status") == "Strong Public Evidence")
    weak_cnt = sum(1 for r in verification_results if r.get("status") == "Weak Public Evidence")
    none_cnt = sum(1 for r in verification_results if r.get("status") == "No Public Evidence")

    score = ((strong_cnt * 1.0) + (weak_cnt * 0.4)) / total * 100.0
    return round(score, 1), strong_cnt, weak_cnt, none_cnt


def generate_bridge_plan(
    client: genai.Client,
    unverified_skills: List[Dict[str, Any]],
    candidate_handle: str
) -> str:
    """Generate a personalized 2-Week Weekend GitHub Bridge Plan for unverified/weak skills."""
    skills_summary = [f"{s.get('skill')} ({s.get('category')} - Status: {s.get('status')})" for s in unverified_skills]

    prompt = f"""
You are a Principal Software Engineering Career Coach.
A developer with the GitHub handle '{candidate_handle}' has claimed the following technical skills on their resume, but they currently lack strong public evidence on GitHub:

Unverified/Weak Skills:
{json.dumps(skills_summary, indent=2)}

Create a high-impact, personalized "2-Week Weekend GitHub Bridge Plan" to help them build verifiable, production-grade public repository evidence over two upcoming weekends.

Structure the plan clearly with Markdown:
### 🎯 Executive Objective
Brief strategy on turning these skill gaps into standout portfolio highlights.

### 🗓️ Weekend 1: Architecture & Core Implementation (Saturday & Sunday)
- **Project Concept & Repository Blueprint**: Recommended repository name, tech stack combining multiple unverified skills into 1 cohesive production-grade project.
- **Saturday Milestones**: Data models, core logic, API endpoints/services.
- **Sunday Milestones**: Containerization (Docker), configuration, and initial functional commit.
- **Verifiable Proof Artifacts**: Specific files (e.g. `docker-compose.yml`, benchmark tests, schemas) that prove real competence.

### 🗓️ Weekend 2: Production Polish, Testing & Public Showcase (Saturday & Sunday)
- **Saturday Milestones**: Automated unit/integration tests with GitHub Actions CI workflow (`.github/workflows/ci.yml`), logging, and error handling.
- **Sunday Milestones**: Live deployment (Vercel, Render, or Railway), dynamic badge integration, interactive API docs / Swagger.
- **README & Demonstration Engineering**: High-conversion README structure with architectural diagrams, GIF walkthrough, and performance metrics.

### 🚀 Recruiter Checklist & Portfolio Hygiene
3 immediate tactical tips for their GitHub profile (Pinning repos, README profile badge, commit history etiquette).
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4
        )
    )
    return response.text or "Unable to generate plan at this time."


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg",
        width=48
    )
    st.title("SkillProof AI")
    st.markdown("**Evidence-Based Career Readiness Engine**")
    st.divider()

    st.subheader("🔑 Credentials Setup")
    env_gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=env_gemini_key,
        type="password",
        help="Required. Fallback loaded from .env if present. Powered by Gemini 2.5 Flash."
    )

    env_github_token = os.getenv("GITHUB_TOKEN", "")
    github_token = st.text_input(
        "GitHub Token (Optional)",
        value=env_github_token,
        type="password",
        help="Optional personal access token to prevent GitHub API rate limits (60 req/hr unauthenticated)."
    )

    st.divider()
    st.markdown("### ℹ️ How It Works")
    st.markdown(
        """
        1. **Upload Resume**: We parse your PDF via `pypdf`.
        2. **Skill Extraction**: Gemini 2.5 Flash extracts claimed tech skills into structured JSON.
        3. **GitHub Audit**: REST API queries public repositories, languages, and star activity.
        4. **Verification**: Gemini classifies skills as **Strong**, **Weak**, or **No Public Evidence**.
        5. **Bridge Plan**: Generates an actionable 2-week weekend roadmap for closing proof gaps.
        """
    )
    st.divider()
    st.caption("SkillProof AI • Built with Gemini 2.5 Flash & Streamlit")


# ---------------------------------------------------------
# Main Page Header & Ethical Disclaimer Banner
# ---------------------------------------------------------
st.markdown("<div class='main-title'>🛡️ SkillProof AI — Evidence-Based Career Readiness Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Verify resume technical claims against demonstrable public GitHub engineering artifacts.</div>", unsafe_allow_html=True)

# Ethical Disclaimer Banner
st.warning(
    "⚖️ **Ethical Disclaimer**: SkillProof AI indexes public GitHub repositories. Private work and offline projects are excluded; "
    "unindexed skills reflect a public evidence gap, not a lack of student ability."
)

st.write("")


# ---------------------------------------------------------
# Main Input Form
# ---------------------------------------------------------
col_resume, col_github = st.columns([1.2, 1], gap="medium")

with col_resume:
    st.markdown("#### 📄 1. Candidate Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume in PDF format",
        type=["pdf"],
        help="Upload candidate PDF resume. Text will be securely extracted for skill identification."
    )

with col_github:
    st.markdown("#### 🐙 2. Public GitHub Profile")
    github_username = st.text_input(
        "GitHub Username",
        placeholder="e.g. torvalds or your-github-handle",
        help="Enter the exact GitHub handle of the candidate."
    )

analyze_btn = st.button("🚀 Analyze & Verify Skills", type="primary", use_container_width=True)


# ---------------------------------------------------------
# Execution & Analysis Workflow
# ---------------------------------------------------------
if analyze_btn:
    # Validation checks
    if not uploaded_file:
        st.error("Please upload a candidate PDF resume before proceeding.")
        st.stop()

    if not github_username.strip():
        st.error("Please enter a valid GitHub username.")
        st.stop()

    if not gemini_api_key.strip():
        st.error("Please provide a Gemini API Key in the sidebar or in your .env file.")
        st.stop()

    if not GENAI_AVAILABLE:
        st.error("The `google-genai` SDK is not installed. Please run `pip install google-genai`.")
        st.stop()

    # Initialize Gemini client
    try:
        client = genai.Client(api_key=gemini_api_key.strip())
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {str(e)}")
        st.stop()

    with st.status("🔍 Running SkillProof Verification Engine...", expanded=True) as status_box:
        # Step 1: Extract Resume Text
        st.write("📄 **Step 1/4**: Extracting text from PDF resume using `pypdf`...")
        resume_text = extract_text_from_pdf(uploaded_file)
        if not resume_text:
            st.error("Could not extract any readable text from the uploaded PDF. Please check the file.")
            status_box.update(label="Analysis aborted", state="error")
            st.stop()

        # Step 2: Extract Claimed Skills via Gemini
        st.write("🤖 **Step 2/4**: Identifying technical skills with **Gemini 2.5 Flash**...")
        try:
            claimed_skills = extract_claimed_skills_gemini(client, resume_text)
        except Exception as e:
            st.error(f"Gemini skill extraction failed: {str(e)}")
            status_box.update(label="Skill extraction failed", state="error")
            st.stop()

        if not claimed_skills:
            st.warning("No technical skills could be extracted from the resume.")
            status_box.update(label="No skills found", state="complete")
            st.stop()

        # Step 3: Query GitHub REST API
        st.write(f"🐙 **Step 3/4**: Querying GitHub REST API for `{github_username.strip()}` repositories...")
        profile, repos, err = fetch_github_profile_and_repos(github_username, github_token)
        if err:
            st.error(err)
            status_box.update(label="GitHub API Query Failed", state="error")
            st.stop()

        st.write(f"Found **{len(repos)}** public repositories for `{github_username.strip()}`.")

        # Step 4: Verify Skills with Gemini
        st.write("⚖️ **Step 4/4**: Cross-examining claimed skills against repository evidence...")
        try:
            verification_results = verify_skills_with_gemini(client, claimed_skills, repos, github_username.strip())
        except Exception as e:
            st.error(f"Verification engine failed: {str(e)}")
            status_box.update(label="Verification failed", state="error")
            st.stop()

        # Step 5: Generate 2-Week Weekend Bridge Plan for unverified skills
        unverified_skills = [
            s for s in verification_results
            if s.get("status") in ["Weak Public Evidence", "No Public Evidence"]
        ]

        bridge_plan = ""
        if unverified_skills:
            st.write("🛠️ Synthesizing personalized **2-Week Weekend Bridge Plan**...")
            try:
                bridge_plan = generate_bridge_plan(client, unverified_skills, github_username.strip())
            except Exception as e:
                bridge_plan = f"Roadmap generation error: {str(e)}"
        else:
            bridge_plan = "🎉 **Outstanding!** All claimed technical skills have verified strong public GitHub evidence. No bridge plan needed."

        status_box.update(label="Verification complete!", state="complete", expanded=False)

    # Persist in session state
    st.session_state["verified_results"] = verification_results
    st.session_state["bridge_plan"] = bridge_plan
    st.session_state["profile"] = profile
    st.session_state["repo_count"] = len(repos)
    st.session_state["target_username"] = github_username.strip()


# ---------------------------------------------------------
# Results Presentation
# ---------------------------------------------------------
if "verified_results" in st.session_state:
    results = st.session_state["verified_results"]
    plan = st.session_state["bridge_plan"]
    profile = st.session_state.get("profile") or {}
    repo_count = st.session_state.get("repo_count", 0)
    target_user = st.session_state.get("target_username", "")

    ppi_score, strong_c, weak_c, none_c = calculate_public_proof_index(results)
    total_skills = len(results)

    st.write("---")
    st.subheader(f"📊 Public Verification Audit for @{target_user}")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val' style='color: #2563eb;'>{ppi_score}%</div>
                <div class='metric-lbl'>Public Proof Index</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val' style='color: #0f172a;'>{total_skills}</div>
                <div class='metric-lbl'>Claimed Skills</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val' style='color: #16a34a;'>{strong_c}</div>
                <div class='metric-lbl'>Strong Evidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val' style='color: #d97706;'>{weak_c}</div>
                <div class='metric-lbl'>Weak Evidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m5:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-val' style='color: #dc2626;'>{none_c}</div>
                <div class='metric-lbl'>Evidence Gap</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # Visualizations Section
    col_chart1, col_chart2 = st.columns([1.2, 1], gap="medium")

    with col_chart1:
        st.markdown("##### 📈 Public Evidence Distribution")
        chart_df = pd.DataFrame([
            {"Evidence Level": "Strong Public Evidence", "Count": strong_c, "Color": "#22c55e"},
            {"Evidence Level": "Weak Public Evidence", "Count": weak_c, "Color": "#eab308"},
            {"Evidence Level": "No Public Evidence", "Count": none_c, "Color": "#ef4444"}
        ])

        fig = px.bar(
            chart_df,
            x="Count",
            y="Evidence Level",
            orientation="h",
            color="Evidence Level",
            color_discrete_map={
                "Strong Public Evidence": "#22c55e",
                "Weak Public Evidence": "#eab308",
                "No Public Evidence": "#ef4444"
            },
            text="Count"
        )
        fig.update_layout(
            showlegend=False,
            height=280,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis_title="Number of Skills",
            yaxis_title=None,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.markdown("##### 🎯 Public Proof Gauge")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ppi_score,
            number={"suffix": "%", "font": {"size": 36, "color": "#0f172a"}},
            title={"text": "Portfolio Proof Readiness", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#2563eb"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 75], "color": "#fef9c3"},
                    {"range": [75, 100], "color": "#dcfce7"}
                ],
                "threshold": {
                    "line": {"color": "#16a34a", "width": 4},
                    "thickness": 0.75,
                    "value": ppi_score
                }
            }
        ))
        gauge_fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    # Detailed Skills Table
    st.markdown("##### 📋 Verified Technical Claims Breakdown")

    # Format into DataFrame
    table_rows = []
    for r in results:
        status_val = r.get("status", "No Public Evidence")
        repos_matched = ", ".join(r.get("evidence_repos", [])) or "None"
        table_rows.append({
            "Skill": r.get("skill", ""),
            "Category": r.get("category", "General"),
            "Evidence Level": status_val,
            "Audit Findings": r.get("reason", ""),
            "Supporting Repositories": repos_matched
        })
    df_results = pd.DataFrame(table_rows)

    # Render with Streamlit Column Config
    st.dataframe(
        df_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Skill": st.column_config.TextColumn("Skill", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Evidence Level": st.column_config.TextColumn(
                "Evidence Status",
                width="medium"
            ),
            "Audit Findings": st.column_config.TextColumn("Verification Reason", width="large"),
            "Supporting Repositories": st.column_config.TextColumn("Matching Repos", width="medium"),
        }
    )

    # Downloadable CSV & JSON export
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Export Verification Audit (CSV)",
            data=df_results.to_csv(index=False),
            file_name=f"skillproof_audit_{target_user}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            "📥 Export Full Audit (JSON)",
            data=json.dumps(results, indent=2),
            file_name=f"skillproof_audit_{target_user}.json",
            mime="application/json",
            use_container_width=True
        )

    st.write("")

    # Expandable 2-Week Weekend Bridge Plan
    with st.expander("📅 Personalized 2-Week Weekend GitHub Bridge Plan", expanded=True):
        st.markdown(plan)

    # Footer note
    st.markdown(
        """
        <div style='text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 30px;'>
            SkillProof AI • Evidence-Based Career Readiness Engine • Evaluates Public Open-Source Traces Only
        </div>
        """,
        unsafe_allow_html=True
    )
