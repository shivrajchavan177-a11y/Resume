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
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.skill-box {
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    font-weight: bold;
}

.matched {
    background-color: #1e5631;
    color: white;
}

.missing {
    background-color: #7a1f1f;
    color: white;
}

.score-box {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    background-color: #262730;
    color: white;
    margin-bottom: 20px;
}

.decision-box {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

.shortlisted {
    background-color: #1e5631;
}

.maybe {
    background-color: #8a6d1d;
}

.rejected {
    background-color: #7a1f1f;
}

</style>
""", unsafe_allow_html=True)

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

st.markdown("""
<h1 style='text-align: center;'>
Smart Resume Screening System
</h1>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Upload Resume")

job_role = st.sidebar.selectbox(
    "Select Job Role",
    list(JOB_ROLES.keys())
)

uploaded_file = st.sidebar.file_uploader(
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

    # Prevent Unrealistic Scores

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
        decision_class = "shortlisted"

    elif predicted_score >= 5:

        decision = "MAYBE"
        decision_class = "maybe"

    else:

        decision = "REJECTED"
        decision_class = "rejected"

    # =================================================
    # TOP METRICS
    # =================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class='score-box'>
            <h2>{predicted_score}/10</h2>
            <p>ATS Match Score</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class='score-box'>
            <h2>{len(matched_skills)}</h2>
            <p>Matched Skills</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class='score-box'>
            <h2>{len(missing_skills)}</h2>
            <p>Missing Skills</p>
        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # PROGRESS BAR
    # =================================================

    st.progress(float(predicted_score) / 10)

    st.divider()

    # =================================================
    # SKILLS SECTION
    # =================================================

    left, right = st.columns(2)

    # -------------------------------------------------
    # MATCHED SKILLS
    # -------------------------------------------------

    with left:

        st.subheader("Matched Skills")

        if matched_skills:

            for skill in matched_skills:

                st.markdown(
                    f"""
                    <div class='skill-box matched'>
                    {skill}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.warning("No matched skills found")

    # -------------------------------------------------
    # MISSING SKILLS
    # -------------------------------------------------

    with right:

        st.subheader("Missing Skills")

        if missing_skills:

            for skill in missing_skills:

                st.markdown(
                    f"""
                    <div class='skill-box missing'>
                    {skill}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.success("No missing skills")

    st.divider()

    # =================================================
    # FINAL DECISION
    # =================================================

    st.markdown(f"""
    <div class='decision-box {decision_class}'>
        {decision}
    </div>
    """, unsafe_allow_html=True)

else:

    st.info("Upload a resume PDF to start screening.")
