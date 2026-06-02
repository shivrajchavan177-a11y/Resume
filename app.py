import streamlit as st
import pdfplumber
import pickle
import re
import numpy as np

from skills import JOB_ROLES

# =====================================================
# LOAD MODEL
# =====================================================

model = pickle.load(open("model.pkl", "rb"))

mlb = pickle.load(open("mlb.pkl", "rb"))

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Resume Screening System")

st.write("Upload Resume and Select Job Role")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

# =====================================================
# JOB ROLE
# =====================================================

job_role = st.selectbox(
    "Select Job Role",
    list(JOB_ROLES.keys())
)

# =====================================================
# EXTRACT TEXT
# =====================================================

def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text.lower()

    return text

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    text = re.sub(r'[^a-zA-Z0-9+# ]', ' ', text)

    return text.lower()

# =====================================================
# EXTRACT SKILLS
# =====================================================

def extract_skills(text):

    extracted_skills = []

    all_skills = mlb.classes_

    for skill in all_skills:

        if skill.lower() in text:
            extracted_skills.append(skill)

    return list(set(extracted_skills))

# =====================================================
# MAIN LOGIC
# =====================================================

if uploaded_file is not None:

    # Extract resume text
    resume_text = extract_text(uploaded_file)

    resume_text = clean_text(resume_text)

    # Extract skills
    resume_skills = extract_skills(resume_text)

    # Required skills
    required_skills = JOB_ROLES[job_role]

    # Encode skills
    encoded_skills = mlb.transform([resume_skills])

    # Predict score
    predicted_score = model.predict(encoded_skills)[0]

    # Limit score between 1-10
    predicted_score = max(1, min(10, predicted_score))

    predicted_score = round(predicted_score, 1)

    # Matched skills
    matched_skills = list(
        set(resume_skills) &
        set(required_skills)
    )

    # Missing skills
    missing_skills = list(
        set(required_skills) -
        set(resume_skills)
    )

    # Final decision
    if predicted_score >= 8:

        decision = "SHORTLISTED"

    elif predicted_score >= 7:

        decision = "MAYBE"

    else:

        decision = "REJECTED"

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.divider()

    st.subheader("📊 Resume Screening Result")

    st.metric(
        label="Match Score",
        value=f"{predicted_score}/10"
    )

    st.progress(predicted_score / 10)

    # Matched Skills
    st.success("Matched Skills")

    if matched_skills:

        for skill in matched_skills:
            st.write(f"✔ {skill}")

    else:
        st.write("No matched skills found")

    # Missing Skills
    st.error("Missing Skills")

    if missing_skills:

        for skill in missing_skills:
            st.write(f"✘ {skill}")

    else:
        st.write("No missing skills")

    # Resume skills
    st.info("Extracted Resume Skills")

    st.write(resume_skills)

    # Final decision
    st.subheader("Final Decision")

    if decision == "SHORTLISTED":

        st.success(decision)

    elif decision == "MAYBE":

        st.warning(decision)

    else:

        st.error(decision)
