import os
import json
import requests
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(
    page_title="SkillProof AI — Evidence-Based Readiness Engine",
    page_icon="🛡️",
    layout="wide"
)

# Force clear white headings for dark theme
st.markdown("""
    <style>
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ SkillProof AI")
st.markdown("### *Evidence-Based Career Readiness & Skill Verification Engine*")

st.warning(
    "🔒 **Ethics & Privacy Notice:** SkillProof AI indexes public GitHub repositories. "
    "Private work and offline projects are excluded. Unindexed skills reflect a **Public Evidence Gap**, not a lack of student ability."
)

st.sidebar.header("🔑 Configuration & Credentials")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
github_token_input = st.sidebar.text_input("GitHub Token (Optional)", type="password", value=os.getenv("GITHUB_TOKEN", ""))

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📄 Step 1: Upload Student Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])

with col2:
    st.markdown("#### 🐙 Step 2: Enter GitHub Profile")
    raw_github_input = st.text_input("GitHub Username or Profile URL", placeholder="e.g., YendetiYashashwini")

def fetch_github_data(username_or_url, token=None):
    if not username_or_url:
        return [], None
    clean_username = username_or_url.strip().rstrip("/").split("/")[-1]
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = f"https://api.github.com/users/{clean_username}/repos?per_page=100&sort=updated"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return [], f"GitHub user '{clean_username}' not found or API rate limited."
        
        repos = response.json()
        repo_summary = []
        for repo in repos:
            repo_summary.append({
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo.get("stargazers_count", 0)
            })
        return repo_summary, None
    except Exception as e:
        return [], str(e)

if st.button("🚀 Analyze & Verify Skills", type="primary"):
    if not api_key_input:
        st.error("❌ Please enter a valid Gemini API Key in the sidebar.")
    elif not uploaded_file or not raw_github_input:
        st.error("❌ Please upload a resume PDF and enter your GitHub username or URL.")
    else:
        with st.spinner("🔄 Parsing resume, fetching GitHub data, and computing skill evidence..."):
            try:
                reader = PdfReader(uploaded_file)
                resume_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"

                client = genai.Client(api_key=api_key_input)

                extraction_prompt = f"""
                Extract a JSON list of technical skills (languages, frameworks, databases, tools) mentioned in this resume.
                Return ONLY a JSON array of strings, e.g. ["Python", "Machine Learning", "SQL"].
                
                Resume Text:
                {resume_text[:4000]}
                """
                
                # Standard stable flash model
                response_ext = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=extraction_prompt,
                )
                
                clean_json_str = response_ext.text.strip().replace("```json", "").replace("```", "").strip()
                claimed_skills = json.loads(clean_json_str)

                gh_repos, gh_error = fetch_github_data(raw_github_input, github_token_input)
                if gh_error:
                    st.warning(f"⚠️ {gh_error}")

                comparison_prompt = f"""
                Compare these claimed resume skills against public GitHub repos.
                Claimed Skills: {json.dumps(claimed_skills)}
                GitHub Repos: {json.dumps(gh_repos)}

                For EACH skill, mark status as: "Strong Public Evidence", "Weak Public Evidence", or "No Public Evidence".
                Provide a short 1-line reason for each.
                Provide an overall proof_score percentage (0-100).
                Provide a 2-week bridge_plan (markdown) to build public proof for unverified skills.

                Return STRICT JSON format:
                {{
                  "evaluations": [
                    {{"skill": "SkillName", "status": "Strong Public Evidence", "reason": "Found in repository language/code"}}
                  ],
                  "proof_score": 85,
                  "bridge_plan": "### Week 1..."
                }}
                """

                response_eval = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=comparison_prompt,
                )

                clean_eval_str = response_eval.text.strip().replace("```json", "").replace("```", "").strip()
                result_data = json.loads(clean_eval_str)

                st.success("✅ Analysis Successfully Completed!")

                m1, m2, m3 = st.columns(3)
                m1.metric("Claimed Skills", len(claimed_skills))
                m2.metric("Public Proof Score", f"{result_data.get('proof_score', 0)}%")
                m3.metric("GitHub Repos Indexed", len(gh_repos))

                st.markdown("### 🔍 Skill Verification Matrix")
                eval_df = pd.DataFrame(result_data.get("evaluations", []))
                st.dataframe(eval_df, use_container_width=True)

                st.markdown("### 🚀 Personalized 2-Week Bridge Plan")
                st.markdown(result_data.get("bridge_plan", "Plan pending."))

            except Exception as e:
                st.error(f"❌ Execution Error: {str(e)}")
