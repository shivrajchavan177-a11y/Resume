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
    background-color: #F4F7FB;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

h1,h2,h3 {
    color: #1E293B;
}

section[data-testid="stSidebar"] {
    background-color: #E2E8F0;
}

.resume-card {
    background-color: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.score-text {
    font-size: 32px;
    font-weight: bold;
    color: #2563EB;
}

.skill-tag-green {
    display: inline-block;
    background-color: #DCFCE7;
    color: #166534;
    padding: 6px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
    font-weight: 600;
}

.skill-tag-red {
    display: inline-block;
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 6px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
    font-weight: 600;
}

.shortlisted {
    background-color: #16A34A;
    color: white;
    padding: 8px 15px;
    border-radius: 25px;
    display: inline-block;
    font-weight: bold;
}

.maybe {
    background-color: #F59E0B;
    color: white;
    padding: 8px 15px;
    border-radius: 25px;
    display: inline-block;
    font-weight: bold;
}

.rejected {
    background-color: #DC2626;
    color: white;
    padding: 8px 15px;
    border-radius: 25px;
    display: inline-block;
    font-weight: bold;
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
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
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
                text += " " + page_text.lower()

    return text

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    text = re.sub(
        r'[^a-zA-Z0-9+# ]',
        ' ',
        text.lower()
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

if uploaded_files:

    required_skills = JOB_ROLES[job_role]

    results = []

    for uploaded_file in uploaded_files:

        # -------------------------------------------------
        # EXTRACT TEXT
        # -------------------------------------------------

        resume_text = extract_text(uploaded_file)

        resume_text = clean_text(resume_text)

        # -------------------------------------------------
        # EXTRACT SKILLS
        # -------------------------------------------------

        resume_skills = extract_skills(resume_text)

        # -------------------------------------------------
        # MATCHED / MISSING
        # -------------------------------------------------

        matched_skills = list(
            set(resume_skills) &
            set(required_skills)
        )

        missing_skills = list(
            set(required_skills) -
            set(resume_skills)
        )

        # -------------------------------------------------
        # MODEL INPUT
        # -------------------------------------------------

        combined_text = (
            " ".join(resume_skills)
            + " "
            + job_role.lower()
        )

        vector = tfidf.transform([combined_text])

        # -------------------------------------------------
        # ML SCORE
        # -------------------------------------------------

        ml_score = model.predict(vector)[0] * 10

        ml_score = max(1, min(10, ml_score))

        # -------------------------------------------------
        # SKILL SCORE
        # -------------------------------------------------

        match_percent = (
            len(matched_skills) /
            len(required_skills)
        )

        base_score = match_percent * 10

        # -------------------------------------------------
        # SIMILARITY
        # -------------------------------------------------

        resume_vector = tfidf.transform(
            [" ".join(resume_skills)]
        )

        role_vector = tfidf.transform(
            [" ".join(required_skills)]
        )

        similarity = cosine_similarity(
            resume_vector,
            role_vector
        )[0][0]

        similarity_score = similarity * 10

        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        predicted_score = (
            base_score * 0.85 +
            similarity_score * 0.10 +
            ml_score * 0.05
        )

        if len(matched_skills) <= 1:

            predicted_score = min(predicted_score, 4)

        elif len(matched_skills) <= 2:

            predicted_score = min(predicted_score, 6)

        predicted_score = max(1, min(10, predicted_score))

        predicted_score = round(predicted_score, 1)

        # -------------------------------------------------
        # FINAL DECISION
        # -------------------------------------------------

        if predicted_score >= 8:

            decision = "SHORTLISTED"
            decision_class = "shortlisted"

        elif predicted_score >= 5:

            decision = "MAYBE"
            decision_class = "maybe"

        else:

            decision = "REJECTED"
            decision_class = "rejected"

        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        results.append({
            "name": uploaded_file.name,
            "score": predicted_score,
            "decision": decision,
            "decision_class": decision_class,
            "matched": matched_skills[:5],
            "missing": missing_skills[:5]
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
    # TOP STATS
    # =====================================================

    top1, top2 = st.columns(2)

    with top1:
        st.metric(
            "Total Resumes",
            len(results)
        )

    with top2:
        shortlisted_count = len([
            r for r in results
            if r["decision"] == "SHORTLISTED"
        ])

        st.metric(
            "Shortlisted",
            shortlisted_count
        )

    st.divider()

    # =====================================================
    # SHOW 2 RESUMES PER ROW
    # =====================================================

    for i in range(0, len(results), 2):

        cols = st.columns(2)

        for j in range(2):

            if i + j < len(results):

                result = results[i + j]

                with cols[j]:

                    st.markdown(
                        f"""
                        <div class='resume-card'>

                        <h3>
                        {result['name']}
                        </h3>

                        <div class='score-text'>
                        {result['score']}/10
                        </div>

                        <br>

                        <div class='{result['decision_class']}'>
                        {result['decision']}
                        </div>

                        <br><br>
                        """,
                        unsafe_allow_html=True
                    )

                    # =====================================
                    # MATCHED + MISSING SIDE BY SIDE
                    # =====================================

                    left, right = st.columns(2)

                    with left:

                        st.markdown("#### Matched")

                        for skill in result["matched"]:

                            st.markdown(
                                f"""
                                <span class='skill-tag-green'>
                                {skill}
                                </span>
                                """,
                                unsafe_allow_html=True
                            )

                    with right:

                        st.markdown("#### Missing")

                        for skill in result["missing"]:

                            st.markdown(
                                f"""
                                <span class='skill-tag-red'>
                                {skill}
                                </span>
                                """,
                                unsafe_allow_html=True
                            )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

else:

    st.info(
        "Upload multiple resume PDFs to start screening."
    )
