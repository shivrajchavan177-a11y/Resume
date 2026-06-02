import streamlit as st
import pdfplumber
import pickle
import re

from skills import JOB_ROLES

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered"
)

# =====================================================
# LOAD MODEL FILES
# =====================================================

model = pickle.load(open("model.pkl", "rb"))

tfidf = pickle.load(open("tfidf.pkl", "rb"))

common_skills = pickle.load(
    open("common_skills.pkl", "rb")
)

# =====================================================
# TITLE
# =====================================================

st.title("📄 AI Resume Screening System")

st.markdown("""
This AI-powered ATS system:

✅ Extracts skills from resume  
✅ Compares with industry job roles  
✅ Predicts ATS score using ML  
✅ Shows matched & missing skills  
✅ Gives hiring recommendation  
""")

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
    # Create Combined Text
    # -------------------------------------------------

    combined_text = (
        " ".join(resume_skills)
        + " "
        + job_role.lower()
    )

    # -------------------------------------------------
    # TF-IDF Transformation
    # -------------------------------------------------

    vector = tfidf.transform([combined_text])

    # -------------------------------------------------
    # Predict Score
    # -------------------------------------------------

    predicted_score = model.predict(vector)[0]

    # Convert score to 1-10 scale
    predicted_score = predicted_score * 10

    predicted_score = max(1, min(10, predicted_score))

    predicted_score = float(round(predicted_score, 1))

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
    # ATS SCORE
    # -------------------------------------------------

    st.metric(
        label="ATS Match Score",
        value=f"{predicted_score}/10"
    )

    st.progress(float(predicted_score) / 10)

    # -------------------------------------------------
    # MATCHED SKILLS
    # -------------------------------------------------

    st.subheader("✅ Matched Skills")

    if matched_skills:

        for skill in matched_skills:

            st.success(skill)

    else:

        st.warning("No matched skills found")

    # -------------------------------------------------
    # MISSING SKILLS
    # -------------------------------------------------

    st.subheader("❌ Missing Skills")

    if missing_skills:

        for skill in missing_skills:

            st.error(skill)

    else:

        st.success("No missing skills")

    # -------------------------------------------------
    # EXTRACTED RESUME SKILLS
    # -------------------------------------------------

    st.subheader("🛠 Extracted Resume Skills")

    if resume_skills:

        st.write(", ".join(resume_skills))

    else:

        st.warning("No skills extracted")

    # -------------------------------------------------
    # FINAL DECISION
    # -------------------------------------------------

    st.subheader("📌 Final Decision")

    if decision == "SHORTLISTED":

        st.success("SHORTLISTED")

    elif decision == "MAYBE":

        st.warning("MAYBE")

    else:

        st.error("REJECTED")

    # -------------------------------------------------
    # RECOMMENDED SKILLS
    # -------------------------------------------------

    st.subheader("📚 Recommended Skills to Learn")

    if missing_skills:

        for skill in missing_skills:

            st.info(skill)

    else:

        st.success(
            "Your resume matches most required skills."
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Resume Screening System using NLP, TF-IDF and XGBoost"
)
