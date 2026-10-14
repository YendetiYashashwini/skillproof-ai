import os
import streamlit as st
import requests
import json
import pandas as pd

plotly_available = True
try:
    import plotly.express as px
except ImportError:
    plotly_available = False

from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(
    page_title="SkillProof AI — Evidence-Based Readiness Engine",
    page_icon="🛡️",
    layout="wide"
)

# Dark mode contrast styling
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("🔑 Configuration & Setup")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
github_token_input = st.sidebar.text_input("GitHub Token (Optional)", type="password", value=os.getenv("GITHUB_TOKEN", ""))

st.title("🛡️ SkillProof AI")
st.markdown("### *Evidence-Based Career Readiness & Skill Verification Engine*")

st.warning(
    "🔒 **Ethics & Privacy Notice:** SkillProof AI indexes public GitHub repositories. "
    "Private work and offline projects are excluded. Unindexed skills reflect a **Public Evidence Gap**, not a lack of student ability."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📄 Step 1: Upload Student Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"])

with col2:
    st.markdown("#### 🐙 Step 2: Enter GitHub Profile")
    raw_github_input = st.text_input("GitHub Username or Profile URL", placeholder="e.g., YendetiYashashwini")

def fetch_github_data(username_or_url, token=None):
    clean_username = username_or_url.strip().rstrip("/").split("/")[-1]
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = f"https://api.github.com/users/{clean_username}/repos?per_page=100&sort=updated"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None, f"GitHub API Error: Status code {response.status_code} for user '{clean_username}'"
        
        repos = response.json()
        repo_summary = []
        for repo in repos:
            repo_summary.append({
                "name": repo.get("name"),
                "description": repo.get("description"),
                "language": repo.get("language"),
                "stars": repo.get("stargazers_count", 0),
                "fork": repo.get("fork", False)
            })
        return repo_summary, None
    except Exception as e:
        return None, str(e)

if st.button("🚀 Analyze & Verify Skills", type="primary"):
    if not api_key_input:
        st.error("❌ Please provide a valid Gemini API Key in the sidebar.")
    elif not uploaded_file or not raw_github_input:
        st.error("❌ Please upload a resume PDF and enter your GitHub username or URL.")
    else:
        with st.spinner("🔄 Extracting resume data, querying GitHub, and running Gemini reasoning..."):
            try:
                reader = PdfReader(uploaded_file)
                resume_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"

                client = genai.Client(api_key=api_key_input)

                extraction_prompt = f"""
                Analyze the following resume text and extract a JSON list of technical skills 
                (Languages, Frameworks, Databases, Tools, Cloud Platforms) mentioned.
                Return ONLY valid JSON format like: ["Python", "React", "Docker", "PostgreSQL"]
                
                Resume Text:
                {resume_text[:4000]}
                """
                
                extraction_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=extraction_prompt,
                )
                
                clean_json_text = extraction_response.text.strip().replace("```json", "").replace("```", "").strip()
                claimed_skills = json.loads(clean_json_text)

                gh_repos, gh_error = fetch_github_data(raw_github_input, github_token_input)
                if gh_error:
                    st.error(f"⚠️ GitHub Warning: {gh_error}. Proceeding with resume analysis only.")
                    gh_repos = []

                comparison_prompt = f"""
                You are an expert technical recruiter and auditor.
                Compare the candidate's claimed resume skills against their public GitHub repositories summary.
                
                Claimed Skills: {json.dumps(claimed_skills)}
                GitHub Repositories Summary: {json.dumps(gh_repos)}
                
                For EACH claimed skill, categorize it into one of these exact three statuses:
                1. "Strong Public Evidence" (Found active code repositories, primary language match, or dependency reference)
                2. "Weak Public Evidence" (Mentioned in description/README, but minimal direct implementation depth)
                3. "No Public Evidence" (No public repositories or files found reflecting this technology)
                
                Also provide a concise 1-line reason for each.
                Additionally, generate an overall Public Proof Score percentage (0-100) and a practical 2-week weekend bridge plan to convert unverified skills into public code.
                
                Return response strictly in valid JSON format with keys:
                - "evaluations": list of objects with fields: "skill", "status", "reason"
                - "proof_score": integer (0 to 100)
                - "bridge_plan": string (markdown format recommendations for 2 weeks)
                """

                comparison_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=comparison_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )

                result_data = json.loads(comparison_response.text)

                st.success("✅ Analysis Complete!")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Claimed Skills", len(claimed_skills))
                m2.metric("Public Proof Score", f"{result_data.get('proof_score', 0)}%")
                m3.metric("GitHub Repositories Indexed", len(gh_repos))

                st.markdown("### 🔍 Skill Verification Breakdown")
                eval_list = result_data.get("evaluations", [])
                
                display_df = pd.DataFrame(eval_list)
                st.dataframe(display_df, use_container_width=True)

                if plotly_available and not display_df.empty:
                    st.markdown("### 📊 Status Distribution Chart")
                    status_counts = display_df["status"].value_counts().reset_index()
                    status_counts.columns = ["Status", "Count"]
                    fig = px.bar(status_counts, x="Status", y="Count", color="Status", 
                                 color_discrete_map={
                                     "Strong Public Evidence": "green",
                                     "Weak Public Evidence": "orange",
                                     "No Public Evidence": "gray"
                                 })
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🚀 Personalized 2-Week GitHub Bridge Plan")
                st.markdown(result_data.get("bridge_plan", "No plan generated."))

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
