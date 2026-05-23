%%writefile app.py

import streamlit as st
import PyPDF2
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Resume Comparator",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📄 AI Resume Comparator")

st.write("Upload resumes and compare candidates using AI")


# ==========================================
# JOB DESCRIPTION INPUT
# ==========================================

job_description = st.text_area(
    "Enter Job Description",
    height=200
)


# ==========================================
# FILE UPLOADER
# ==========================================

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


# ==========================================
# PDF TEXT EXTRACTION FUNCTION
# ==========================================

def extract_text(pdf_file):

    text = ""

    reader = PyPDF2.PdfReader(pdf_file)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# ==========================================
# LOAD AI MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

    return model


model = load_model()


# ==========================================
# MAIN LOGIC
# ==========================================

if uploaded_files and job_description:

    st.success("Resumes Uploaded Successfully")

    results = []

    # Convert job description into AI embedding
    jd_embedding = model.encode(job_description)

    # ======================================
    # PROCESS EACH RESUME
    # ======================================

    for uploaded_file in uploaded_files:

        # Extract text
        resume_text = extract_text(uploaded_file)

        # Resume embedding
        resume_embedding = model.encode(resume_text)

        # Similarity score
        similarity = cosine_similarity(
            [jd_embedding],
            [resume_embedding]
        )[0][0]

        # ==================================
        # SIMPLE SKILL DETECTION
        # ==================================

        skills_database = [
            "Python",
            "Java",
            "Machine Learning",
            "Deep Learning",
            "SQL",
            "TensorFlow",
            "React",
            "Node.js",
            "Power BI",
            "Spring Boot",
            "NLP",
            "HTML",
            "CSS",
            "JavaScript"
        ]

        detected_skills = []

        for skill in skills_database:

            if skill.lower() in resume_text.lower():
                detected_skills.append(skill)

        # ==================================
        # STORE RESULT
        # ==================================

        results.append({

            "Resume Name": uploaded_file.name,

            "Match Score (%)": round(
                similarity * 100,
                2
            ),

            "Detected Skills":
            ", ".join(detected_skills)

        })


    # ======================================
    # CREATE DATAFRAME
    # ======================================

    results_df = pd.DataFrame(results)

    # ======================================
    # SORT RESULTS
    # ======================================

    results_df = results_df.sort_values(
        by="Match Score (%)",
        ascending=False
    )

    # ======================================
    # SHOW RESULTS
    # ======================================

    st.header("🏆 Resume Ranking Results")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # ======================================
    # BEST CANDIDATE
    # ======================================

    best_candidate = results_df.iloc[0]

    st.success(
        f"Best Candidate: "
        f"{best_candidate['Resume Name']} "
        f"with score "
        f"{best_candidate['Match Score (%)']}%"
    )

    # ======================================
    # DOWNLOAD BUTTON
    # ======================================

    csv = results_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Results CSV",
        data=csv,
        file_name="resume_results.csv",
        mime="text/csv"
    )


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("About Project")

st.sidebar.info(
    '''
    AI Resume Comparator

    Features:
    - Resume Ranking
    - AI Matching
    - Skill Detection
    - Job Description Matching
    - CSV Export

    Technologies:
    - Python
    - Streamlit
    - Hugging Face
    - NLP
    '''
)