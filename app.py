import streamlit as st
import pdfplumber
import pickle
import re

from sklearn.metrics.pairwise import cosine_similarity

from skills import JOB_ROLES

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Smart Resume Screening System",
    layout="centered"
)

# =====================================================
# LOAD FILES
# =====================================================

model = pickle.load(open("model.pkl", "rb"))

tfidf = pickle.load(open("tfidf.pkl", "rb"))

common_skills = pickle.load(
    open("common_skills.pkl", "rb")
)

# =====================================================
# TITLE
# =====================================================

st.title("Smart Resume Screening System")

st.divider()

# =====================================================
# JOB ROLE
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
# PDF TEXT EXTRACTION
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
# SKILL EXTRACTION
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
    # TF-IDF INPUT
    # -------------------------------------------------

    combined_text = (
        " ".join(resume_skills)
        + " "
        + job_role.lower()
    )

    vector = tfidf.transform([combined_text])

    # =================================================
    # ML SCORE
    # =================================================

    ml_score = model.predict(vector)[0] * 10

    ml_score = max(1, min(10, ml_score))

    # =================================================
    # SKILL MATCH SCORE
    # =================================================

    match_percent = (
        len(matched_skills) /
        len(required_skills)
    )

    base_score = match_percent * 10

    # =================================================
    # COSINE SIMILARITY SCORE
    # =================================================

    resume_text_for_similarity = (
        " ".join(resume_skills)
    )

    role_text = (
        " ".join(required_skills)
    )

    resume_vector = tfidf.transform(
        [resume_text_for_similarity]
    )

    role_vector = tfidf.transform(
        [role_text]
    )

    similarity = cosine_similarity(
        resume_vector,
        role_vector
    )[0][0]

    similarity_score = similarity * 10

    # =================================================
    # FINAL HYBRID SCORE
    # =================================================

    predicted_score = (
        base_score * 0.85 +
        similarity_score * 0.10 +
        ml_score * 0.05
    )

    # -------------------------------------------------
    # Prevent Unrealistic High Scores
    # -------------------------------------------------

    if len(matched_skills) <= 1:

        predicted_score = min(predicted_score, 4)

    elif len(matched_skills) <= 2:

        predicted_score = min(predicted_score, 6)

    predicted_score = max(1, min(10, predicted_score))

    predicted_score = round(predicted_score, 1)

    # =================================================
    # FINAL DECISION
    # =================================================

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

    st.subheader("Resume Screening Result")

    # -------------------------------------------------
    # ATS SCORE
    # -------------------------------------------------

    st.metric(
        label="ATS Match Score",
        value=f"{predicted_score}/10"
    )

    st.progress(float(predicted_score) / 10)

    # =================================================
    # MATCHED SKILLS
    # =================================================

    st.subheader("Matched Skills")

    if matched_skills:

        for skill in matched_skills:

            st.success(skill)

    else:

        st.warning("No matched skills found")

    # =================================================
    # MISSING SKILLS
    # =================================================

    st.subheader("Missing Skills")

    if missing_skills:

        for skill in missing_skills:

            st.error(skill)

    else:

        st.success("No missing skills")

    # =================================================
    # FINAL DECISION
    # =================================================

    st.subheader("Final Decision")

    if decision == "SHORTLISTED":

        st.success("SHORTLISTED")

    elif decision == "MAYBE":

        st.warning("MAYBE")

    else:

        st.error("REJECTED")
