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
    background-color: #F5F7FA;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #1E293B;
}

section[data-testid="stSidebar"] {
    background-color: #E2E8F0;
}

.metric-card {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

.metric-card h2 {
    color: #2563EB;
    margin: 0;
}

.metric-card p {
    color: #475569;
    margin-top: 5px;
}

.skill-box {
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-weight: 600;
    color: white;
}

.matched {
    background-color: #22C55E;
}

.missing {
    background-color: #EF4444;
}

.decision-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.shortlisted {
    background-color: #16A34A;
}

.maybe {
    background-color: #F59E0B;
}

.rejected {
    background-color: #DC2626;
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
<h1 style='text-align:center;'>
Smart Resume Screening System
</h1>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Resume Screening")

job_role = st.sidebar.selectbox(
    "Select Job Role",
    list(JOB_ROLES.keys())
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Multiple Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
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

if uploaded_files:

    required_skills = JOB_ROLES[job_role]

    results = []

    for uploaded_file in uploaded_files:

        # -------------------------------------------------
        # Resume Name
        # -------------------------------------------------

        candidate_name = uploaded_file.name

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
        # COSINE SIMILARITY
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
        # FINAL SCORE
        # =================================================

        predicted_score = (
            base_score * 0.85 +
            similarity_score * 0.10 +
            ml_score * 0.05
        )

        # Prevent Unrealistic High Scores

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
        # STORE RESULTS
        # =================================================

        results.append({
            "name": candidate_name,
            "score": predicted_score,
            "decision": decision,
            "decision_class": decision_class,
            "matched": matched_skills,
            "missing": missing_skills
        })

    # =====================================================
    # SORT RESULTS
    # =====================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.subheader("Resume Ranking")

    rank = 1

    for result in results:

        st.markdown(f"""
        <div class='metric-card'>
            <h2>#{rank} - {result['name']}</h2>
            <h2>{result['score']}/10</h2>
            <p>ATS Match Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(result['score']) / 10)

        # -------------------------------------------------
        # Skills Columns
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Matched Skills")

            if result["matched"]:

                for skill in result["matched"]:

                    st.markdown(
                        f"""
                        <div class='skill-box matched'>
                        {skill}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.warning("No matched skills")

        with col2:

            st.subheader("Missing Skills")

            if result["missing"]:

                for skill in result["missing"]:

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

        # -------------------------------------------------
        # Final Decision
        # -------------------------------------------------

        st.markdown(f"""
        <div class='decision-box {result['decision_class']}'>
            {result['decision']}
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        rank += 1

else:

    st.info(
        "Upload one or more resume PDFs to start screening."
    )
