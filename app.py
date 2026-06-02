import streamlit as st
import pdfplumber
import pickle
import re
import numpy as np

from skills import JOB_ROLES

# =====================================================
# LOAD MODEL FILES
# =====================================================

model = pickle.load(open("model.pkl", "rb"))

tfidf = pickle.load(open("tfidf.pkl", "rb"))

common_skills = pickle.load(
    open("common_skills.pkl", "rb")
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)

# =====================================================
# TITLE
# =====================================================

st.title("📄 AI Resume Screening System")

st.markdown(
    """
    Upload your resume and select a job role.
    
    The AI model will:
    - Extract skills
    - Compare with industry requirements
    - Predict ATS match score
    - Show matched & missing skills
    - Give hiring recommendation
    """
)

st.divider()

# =====================================================
# JOB ROLE SELECTION
# =====================================================

job_role = st.selectbox(
    "Select Job Role",
    list(JOB_ROLES.keys())
)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# =====================================================
# EXTRACT PDF TEXT
# =====================================================

def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += " " + page_text.lower()

    return text

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9+# ]',
        ' ',
        text
    )

    return text

# =====================================================
# EXTRACT SKILLS
# =====================================================

def extract_skills(text):

    extracted = []

    for skill in common_skills:

        if skill.lower() in text:

            extracted.append(skill)

    return list(set(extracted))

# =====================================================
# MAIN LOGIC
# =====================================================

if uploaded_file is not None:

    # -------------------------------------------------
    # Extract Resume Text
    # -------------------------------------------------

    resume_text = extract_text(uploaded_file)

    resume_text = clean_text(resume_text)

    # -------------------------------------------------
    # Extract Skills
    # -------------------------------------------------

    resume_skills = extract_skills(resume_text)

    # -------------------------------------------------
    # Required Skills
    # -------------------------------------------------

    required_skills = JOB_ROLES[job_role]

    # -------------------------------------------------
    # ML Prediction
    # -------------------------------------------------

    combined_text = (
        " ".join(resume_skills)
        + " "
        + job_role.lower()
    )

    vector = tfidf.transform([combined_text])

    predicted_score = model.predict(vector)[0]

    # Convert to 1-10 scale
    predicted_score = predicted_score * 10

    predicted_score = max(1, min(10, predicted_score))

    predicted_score = round(predicted_score, 1)

    # -------------------------------------------------
    # Matched Skills
    # -------------------------------------------------

    matched_skills = list(
        set(resume_skills) &
        set(required_skills)
    )

    # -------------------------------------------------
    # Missing Skills
    # -------------------------------------------------

    missing_skills = list(
        set(required_skills) -
        set(resume_skills)
    )

    # -------------------------------------------------
    # Final Decision
    # -------------------------------------------------

    if predicted_score >= 8:

        decision = "SHORTLISTED"

    elif predicted_score >= 5:

        decision = "MAYBE"

    else:

        decision = "REJECTED"

    # =================================================
    # RESULTS SECTION
    # =================================================

    st.divider()

    st.subheader("📊 Resume Screening Result")

    # -------------------------------------------------
    # Score
    # -------------------------------------------------

    st.metric(
        label="ATS Match Score",
        value=f"{predicted_score}/10"
    )

    st.progress(predicted_score / 10)

    # -------------------------------------------------
    # Matched Skills
    # -------------------------------------------------

    st.subheader("✅ Matched Skills")

    if matched_skills:

        for skill in matched_skills:

            st.success(skill)

    else:

        st.warning("No matched skills found")

    # -------------------------------------------------
    # Missing Skills
    # -------------------------------------------------

    st.subheader("❌ Missing Skills")

    if missing_skills:

        for skill in missing_skills:

            st.error(skill)

    else:

        st.success("No missing skills")

    # -------------------------------------------------
    # Extracted Resume Skills
    # -------------------------------------------------

    st.subheader("🛠 Extracted Resume Skills")

    if resume_skills:

        st.write(", ".join(resume_skills))

    else:

        st.warning("No skills extracted")

    # -------------------------------------------------
    # Final Decision
    # -------------------------------------------------

    st.subheader("📌 Final Decision")

    if decision == "SHORTLISTED":

        st.success("SHORTLISTED")

    elif decision == "MAYBE":

        st.warning("MAYBE")

    else:

        st.error("REJECTED")

    # -------------------------------------------------
    # Recommendations
    # -------------------------------------------------

    st.subheader("📚 Recommended Skills to Learn")

    if missing_skills:

        for skill in missing_skills:

            st.info(skill)

    else:

        st.success(
            "Your resume matches most required skills."
        )