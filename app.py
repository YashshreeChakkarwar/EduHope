import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import date, datetime

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="EduHope | No Student Left Behind",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "eduhope_model.joblib"
DB_PATH = "eduhope_support.db"

# Reference values fill the model features not shown in the short form.
# The existing trained model is reused; it is NOT retrained here.
REFERENCE_VALUES = {
    "marital status": 1,
    "application mode": 8,
    "application order": 1,
    "course": 10,
    "daytime/evening attendance": 1,
    "previous qualification": 1,
    "nacionality": 1,
    "mother's qualification": 13,
    "father's qualification": 14,
    "mother's occupation": 6,
    "father's occupation": 8,
    "displaced": 1,
    "educational special needs": 0,
    "debtor": 0,
    "tuition fees up to date": 1,
    "gender": 0,
    "scholarship holder": 0,
    "age at enrollment": 20,
    "international": 0,
    "curricular units 1st sem (credited)": 0,
    "curricular units 1st sem (enrolled)": 6,
    "curricular units 1st sem (evaluations)": 8,
    "curricular units 1st sem (approved)": 5,
    "curricular units 1st sem (grade)": 12.28571429,
    "curricular units 1st sem (without evaluations)": 0,
    "curricular units 2nd sem (credited)": 0,
    "curricular units 2nd sem (enrolled)": 6,
    "curricular units 2nd sem (evaluations)": 8,
    "curricular units 2nd sem (approved)": 5,
    "curricular units 2nd sem (grade)": 12.2,
    "curricular units 2nd sem (without evaluations)": 0,
    "unemployment rate": 11.1,
    "inflation rate": 1.4,
    "gdp": 0.32,
}


def normalize_feature(name):
    return str(name).replace("\ufeff", "").replace("ï»¿", "").strip().lower()


# --------------------------------------------------
# DESIGN + INPUT TEXT VISIBILITY FIX
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp { background: #F8F6FF; color: #281B42; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #241443, #56308F); }
[data-testid="stSidebar"] * { color: white; }
.hero {
    padding: 26px; border-radius: 22px; color: white;
    background: linear-gradient(120deg, #452477, #8757D9);
    margin-bottom: 24px;
}
.hero h1 { color: white; font-size: 32px; }
.hero p { color: #F0E7FF; }
.metric-card {
    background: white; border: 1px solid #E8DFFA;
    border-radius: 16px; padding: 18px; min-height: 110px;
}
.metric-label { color: #766B8B; font-size: 13px; }
.metric-value { color: #6137A5; font-size: 26px; font-weight: 800; margin-top: 8px; }
.support-card {
    background: white; color: #281B42; padding: 18px;
    border-radius: 15px; border: 1px solid #E8DFFA; margin-bottom: 12px;
}
.support-card h3, .support-card p, .support-card b { color: #281B42 !important; }
div.stButton > button, div.stFormSubmitButton > button {
    background: #7044BF; color: white; border: 0;
    border-radius: 11px; min-height: 42px; font-weight: 600;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background: #543091; color: white; border: 0;
}
/* Questions, field labels and normal text on the light page must be dark */
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] .stMarkdown,
[data-testid="stMain"] .stMarkdown p,
[data-testid="stMain"] .stMarkdown h1,
[data-testid="stMain"] .stMarkdown h2,
[data-testid="stMain"] .stMarkdown h3 {
    color: #281B42 !important;
}

/* Keep typed text white inside the existing dark input boxes */
.stApp input, .stApp textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    color: #FFFFFF !important;
}
.stApp input, .stApp textarea {
    caret-color: #FFFFFF !important;
}
.stApp input::placeholder, .stApp textarea::placeholder {
    color: #D8CDEB !important; opacity: 1 !important;
}
[data-baseweb="select"] * { color: #FFFFFF !important; }
[data-testid="stDateInput"] input { color: #FFFFFF !important; }
[data-testid="stNumberInput"] input { color: #FFFFFF !important; }
[data-testid="stTextInput"] input { color: #FFFFFF !important; }
[data-testid="stTextArea"] textarea { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD EXISTING TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle and "feature_names" in bundle:
        return bundle["model"], list(bundle["feature_names"])
    raise ValueError(
        "The model file must contain a dictionary with 'model' and 'feature_names'."
    )


try:
    model, feature_names = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.info(
        "Ensure eduhope_model.joblib is in the app folder/repository and "
        "contains the existing trained model and feature_names."
    )
    st.stop()


# --------------------------------------------------
# SHORT-FORM MODEL INPUTS (9 INPUTS, NOT 35)
# --------------------------------------------------
def build_student_features(
    age, fees, scholarship,
    sem1_enrolled, sem1_approved, sem1_grade,
    sem2_enrolled, sem2_approved, sem2_grade
):
    reference_lookup = {
        normalize_feature(name): value for name, value in REFERENCE_VALUES.items()
    }
    missing = [
        name for name in feature_names
        if normalize_feature(name) not in reference_lookup
    ]
    if missing:
        raise ValueError(
            "Reference values are missing for these model features: "
            + ", ".join(missing)
        )

    values = {
        name: reference_lookup[normalize_feature(name)]
        for name in feature_names
    }
    entered = {
        "Age at enrollment": age,
        "Tuition fees up to date": fees,
        "Scholarship holder": scholarship,
        "Curricular units 1st sem (enrolled)": sem1_enrolled,
        "Curricular units 1st sem (approved)": sem1_approved,
        "Curricular units 1st sem (grade)": sem1_grade,
        "Curricular units 2nd sem (enrolled)": sem2_enrolled,
        "Curricular units 2nd sem (approved)": sem2_approved,
        "Curricular units 2nd sem (grade)": sem2_grade,
    }
    entered_lookup = {normalize_feature(k): v for k, v in entered.items()}
    for feature in feature_names:
        key = normalize_feature(feature)
        if key in entered_lookup:
            values[feature] = entered_lookup[key]
    return values


def predict_student(values):
    input_df = pd.DataFrame(
        [[values[feature] for feature in feature_names]],
        columns=feature_names,
    )
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
    return str(prediction), probabilities


# --------------------------------------------------
# DATABASE: SUPPORT PLANS
# --------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS support_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_ref TEXT NOT NULL,
                concern TEXT NOT NULL,
                action_plan TEXT NOT NULL,
                mentor TEXT NOT NULL,
                follow_up_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Needs follow-up',
                created_at TEXT NOT NULL
            )
        """)


def save_plan(student_ref, concern, action, mentor, follow_up):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO support_plans
            (student_ref, concern, action_plan, mentor, follow_up_date, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            student_ref.strip(), concern, action.strip(), mentor.strip(),
            follow_up.isoformat(), "Needs follow-up", date.today().isoformat()
        ))


def get_plans():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("""
            SELECT id, student_ref, concern, action_plan, mentor,
                   follow_up_date, status, created_at
            FROM support_plans
            ORDER BY follow_up_date ASC, id DESC
        """, conn)


def update_status(plan_id, status):
    allowed = ["Needs follow-up", "Support in progress", "Follow-up completed"]
    if status not in allowed:
        raise ValueError("Invalid status")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE support_plans SET status = ? WHERE id = ?", (status, int(plan_id)))


init_db()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for key, default in {
    "latest_prediction": None,
    "latest_probabilities": None,
    "latest_student_ref": "",
    "passport_data": {},
    "strength_data": {},
    "roadmap_data": {},
    "simulator_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 🎓 EduHope")
    st.caption("NO STUDENT LEFT BEHIND")
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Risk Report",
            "Comeback Pathway",
            "Support Tracker",
            "Future-Me Simulator",
            "Student Success Passport",
            "Academic Strength Finder",
            "Roadmap",
        ],
    )
    st.divider()
    st.caption("Every prediction is an opportunity to offer support.")


# --------------------------------------------------
# DASHBOARD (EXISTING)
# --------------------------------------------------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Welcome to EduHope 💜</h1>
        <p>Every student deserves a chance, support, and a path forward.</p>
        <p><b>No Student Left Behind.</b></p>
    </div>
    """, unsafe_allow_html=True)

    plans = get_plans()
    total = len(plans)
    needs = int((plans["status"] == "Needs follow-up").sum()) if total else 0
    active = int((plans["status"] == "Support in progress").sum()) if total else 0
    completed = int((plans["status"] == "Follow-up completed").sum()) if total else 0

    cols = st.columns(4)
    for col, (label, value) in zip(cols, [
        ("Support plans", total), ("Needs follow-up", needs),
        ("Support in progress", active), ("Follow-up completed", completed)
    ]):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("## How EduHope helps")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="support-card"><h3>🔎 Notice</h3>
        Explore predicted student outcomes with your existing ML model.</div>
        <div class="support-card"><h3>💜 Support</h3>
        Record a practical action plan with a mentor and follow-up date.</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="support-card"><h3>🌱 Follow through</h3>
        Keep track of plans that still need attention.</div>
        <div class="support-card"><h3>🤝 Remember every student</h3>
        Track support activity instead of leaving a prediction as the final result.</div>
        """, unsafe_allow_html=True)
    st.info("Dashboard totals reflect support plans entered into this app, not the total number of students at a college.")


# --------------------------------------------------
# RISK REPORT (EXISTING, NINE INPUTS ONLY)
# --------------------------------------------------
elif page == "Risk Report":
    st.markdown("""
    <div class="hero"><h1>Student Risk Report 💜</h1>
    <p>A few student details can help us explore where support may be useful.</p></div>
    """, unsafe_allow_html=True)
    st.info(
        "Only nine student details are requested. The remaining model features use fixed reference values. "
        "The result is an estimate, not a certainty."
    )

    with st.form("student_form"):
        student_ref = st.text_input("Student reference", placeholder="Example: STU-001 (use a fictional ID for demos)")
        st.markdown("### Student details")
        age = st.number_input("Age at enrollment", min_value=15, max_value=100, value=20, step=1)
        fees_option = st.selectbox("Are tuition fees up to date?", ["Yes", "No"])
        fees = 1 if fees_option == "Yes" else 0
        scholarship_option = st.selectbox("Does the student hold a scholarship?", ["No", "Yes"])
        scholarship = 1 if scholarship_option == "Yes" else 0

        st.markdown("### First-semester performance")
        sem1_enrolled = st.number_input("Units enrolled in semester 1", min_value=0, max_value=30, value=6, step=1)
        sem1_approved = st.number_input("Units approved in semester 1", min_value=0, max_value=30, value=5, step=1)
        sem1_grade = st.number_input("Average grade in semester 1 (0–20)", min_value=0.0, max_value=20.0, value=12.3, step=0.1)

        st.markdown("### Second-semester performance")
        sem2_enrolled = st.number_input("Units enrolled in semester 2", min_value=0, max_value=30, value=6, step=1)
        sem2_approved = st.number_input("Units approved in semester 2", min_value=0, max_value=30, value=5, step=1)
        sem2_grade = st.number_input("Average grade in semester 2 (0–20)", min_value=0.0, max_value=20.0, value=12.2, step=0.1)
        submitted = st.form_submit_button("Predict Student Outcome", use_container_width=True)

    if submitted:
        if not student_ref.strip():
            st.error("Please enter a student reference.")
        elif sem1_approved > sem1_enrolled:
            st.error("Semester 1 approved units cannot exceed enrolled units.")
        elif sem2_approved > sem2_enrolled:
            st.error("Semester 2 approved units cannot exceed enrolled units.")
        else:
            try:
                vals = build_student_features(
                    age, fees, scholarship, sem1_enrolled, sem1_approved, sem1_grade,
                    sem2_enrolled, sem2_approved, sem2_grade
                )
                pred, probs = predict_student(vals)
                st.session_state.latest_prediction = pred
                st.session_state.latest_probabilities = probs
                st.session_state.latest_student_ref = student_ref.strip()
                st.success("Prediction generated.")
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    prediction = st.session_state.latest_prediction
    probabilities = st.session_state.latest_probabilities
    if prediction is not None:
        st.markdown("## Model result")
        if prediction == "Graduate":
            st.success("Predicted outcome: Graduate 🎓")
            st.write("Continue encouraging the student's progress and offer support whenever needed.")
        elif prediction == "Dropout":
            st.warning("Predicted outcome: Dropout")
            st.write("Consider a supportive conversation to understand the student's academic, financial, or personal challenges.")
        else:
            st.info(f"Predicted outcome: {prediction} 📚")
            st.write("Consider regular check-ins to help the student continue progressing.")

        if probabilities is not None:
            st.markdown("### Estimated class probabilities")
            for label, probability in sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True):
                st.write(f"**{label}: {probability * 100:.1f}%**")
                st.progress(float(probability))
        st.caption(
            "Unentered features use fixed reference values. This estimate may differ from a prediction made using all actual student details. "
            "Do not use this estimate alone to make decisions about a student."
        )
        st.markdown("### What happens next?")
        st.write("If support may be useful, visit Comeback Pathway to record an action plan and follow-up date.")


# --------------------------------------------------
# COMEBACK PATHWAY (EXISTING)
# --------------------------------------------------
elif page == "Comeback Pathway":
    st.markdown("""
    <div class="hero"><h1>💜 Comeback Pathway</h1>
    <p>A prediction should never be the end of a student's story.</p></div>
    """, unsafe_allow_html=True)
    st.write("Record an agreed action, the person coordinating support, and a date to check in.")
    with st.form("support_form"):
        student_ref = st.text_input("Student reference", value=st.session_state.latest_student_ref, placeholder="Example: STU-001")
        concern = st.selectbox("Area where support may help", [
            "Academic progress", "Difficulty with coursework", "Financial support information",
            "Attendance or engagement check-in", "Personal wellbeing check-in", "Other"
        ])
        action = st.text_area(
            "Agreed support action",
            placeholder="Example: Arrange a meeting with the academic mentor and agree on a weekly study plan."
        )
        mentor = st.text_input("Mentor or staff reference", placeholder="Example: MENTOR-01")
        follow_up = st.date_input("Follow-up date", value=date.today(), min_value=date.today())
        save_clicked = st.form_submit_button("Create Support Plan", use_container_width=True)

    if save_clicked:
        if not student_ref.strip() or not action.strip() or not mentor.strip():
            st.error("Please enter the student reference, action, and mentor.")
        else:
            save_plan(student_ref, concern, action, mentor, follow_up)
            st.success("Support plan created! Track its progress on the Support Tracker page.")
    st.markdown("### Our promise")
    st.markdown("""
    <div class="support-card"><b>Notice → Listen → Support → Follow up</b><br><br>
    EduHope encourages constructive support. A model prediction should never be used on its own to punish, exclude, or deny a student an opportunity.</div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# SUPPORT TRACKER (EXISTING)
# --------------------------------------------------
elif page == "Support Tracker":
    st.markdown("""
    <div class="hero"><h1>🤝 No Student Left Behind Tracker</h1>
    <p>Every support plan deserves a follow-up.</p></div>
    """, unsafe_allow_html=True)
    plans = get_plans()
    if plans.empty:
        st.info("No support plans have been created yet. Go to Comeback Pathway to create one.")
    else:
        for _, plan in plans.iterrows():
            with st.container(border=True):
                st.markdown(f"### {plan['student_ref']} — {plan['concern']}")
                st.write(f"**Action:** {plan['action_plan']}")
                st.write(f"**Mentor:** {plan['mentor']}")
                st.write(f"**Follow-up date:** {plan['follow_up_date']}")
                st.write(f"**Current status:** {plan['status']}")
                status_key = f"status_{int(plan['id'])}"
                new_status = st.selectbox(
                    "Update status",
                    ["Needs follow-up", "Support in progress", "Follow-up completed"],
                    index=["Needs follow-up", "Support in progress", "Follow-up completed"].index(plan["status"]),
                    key=status_key
                )
                if st.button("Save status", key=f"save_{int(plan['id'])}"):
                    update_status(int(plan["id"]), new_status)
                    st.success("Status updated.")
                    st.rerun()


# --------------------------------------------------
# FUTURE-ME SIMULATOR (NEW)
# --------------------------------------------------
elif page == "Future-Me Simulator":
    st.markdown("""
    <div class="hero"><h1>🔮 Future-Me Simulator</h1>
    <p>Explore how a possible improvement in semester performance changes the model's estimate.</p></div>
    """, unsafe_allow_html=True)
    st.write("Enter a baseline using the same nine details as the Risk Report. Then try a future scenario.")
    with st.form("future_simulator_form"):
        sim_ref = st.text_input("Student reference", value=st.session_state.latest_student_ref, placeholder="Example: STU-001")
        age = st.number_input("Age at enrollment", 15, 100, 20, key="sim_age")
        fees_label = st.selectbox("Are tuition fees up to date?", ["Yes", "No"], key="sim_fees")
        scholarship_label = st.selectbox("Does the student hold a scholarship?", ["No", "Yes"], key="sim_scholarship")
        st.markdown("### Current semester performance")
        s1e = st.number_input("Units enrolled in semester 1", 0, 30, 6, key="sim_s1e")
        s1a = st.number_input("Units approved in semester 1", 0, 30, 4, key="sim_s1a")
        s1g = st.number_input("Average grade in semester 1 (0–20)", 0.0, 20.0, 10.0, step=0.1, key="sim_s1g")
        s2e = st.number_input("Units enrolled in semester 2", 0, 30, 6, key="sim_s2e")
        s2a = st.number_input("Units approved in semester 2", 0, 30, 4, key="sim_s2a")
        s2g = st.number_input("Average grade in semester 2 (0–20)", 0.0, 20.0, 10.0, step=0.1, key="sim_s2g")
        st.markdown("### Future scenario")
        future_s1a = st.number_input("Possible approved units in semester 1", 0, 30, 5, key="future_s1a")
        future_s1g = st.number_input("Possible semester 1 grade", 0.0, 20.0, 13.0, step=0.1, key="future_s1g")
        future_s2a = st.number_input("Possible approved units in semester 2", 0, 30, 5, key="future_s2a")
        future_s2g = st.number_input("Possible semester 2 grade", 0.0, 20.0, 13.0, step=0.1, key="future_s2g")
        run_sim = st.form_submit_button("Compare My Scenarios", use_container_width=True)

    if run_sim:
        if not sim_ref.strip():
            st.error("Please enter a student reference.")
        elif s1a > s1e or s2a > s2e or future_s1a > s1e or future_s2a > s2e:
            st.error("Approved units cannot exceed enrolled units.")
        else:
            try:
                fee_val = 1 if fees_label == "Yes" else 0
                scholarship_val = 1 if scholarship_label == "Yes" else 0
                current_values = build_student_features(age, fee_val, scholarship_val, s1e, s1a, s1g, s2e, s2a, s2g)
                future_values = build_student_features(age, fee_val, scholarship_val, s1e, future_s1a, future_s1g, s2e, future_s2a, future_s2g)
                current_pred, current_prob = predict_student(current_values)
                future_pred, future_prob = predict_student(future_values)
                st.session_state.simulator_result = {
                    "ref": sim_ref.strip(), "current": current_pred, "future": future_pred,
                    "current_prob": current_prob, "future_prob": future_prob
                }
            except Exception as error:
                st.error(f"Simulation failed: {error}")

    result = st.session_state.simulator_result
    if result:
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="support-card"><h3>📍 Current scenario</h3></div>', unsafe_allow_html=True)
            st.metric("Model estimate", result["current"])
            if result["current_prob"] is not None:
                st.write("Class probabilities")
                for label, p in zip(model.classes_, result["current_prob"]):
                    st.write(f"{label}: {p*100:.1f}%")
        with right:
            st.markdown('<div class="support-card"><h3>🌱 Future scenario</h3></div>', unsafe_allow_html=True)
            st.metric("Model estimate", result["future"], delta="Scenario comparison only" if result["future"] != result["current"] else "Same predicted class")
            if result["future_prob"] is not None:
                st.write("Class probabilities")
                for label, p in zip(model.classes_, result["future_prob"]):
                    st.write(f"{label}: {p*100:.1f}%")
        st.info("This is a what-if simulation, not a promise or a guarantee. It changes only the values entered for the scenario and keeps other features at reference values.")


# --------------------------------------------------
# STUDENT SUCCESS PASSPORT (NEW)
# --------------------------------------------------
elif page == "Student Success Passport":
    st.markdown("""
    <div class="hero"><h1>🪪 Student Success Passport</h1>
    <p>A positive snapshot of goals, strengths, milestones, and support.</p></div>
    """, unsafe_allow_html=True)
    with st.form("passport_form"):
        ref = st.text_input("Student reference", value=st.session_state.latest_student_ref, placeholder="Example: STU-001")
        display_name = st.text_input("Student name or nickname", placeholder="Use a nickname for a demo")
        goal = st.text_input("My current goal", placeholder="Example: Improve semester grades")
        skills = st.multiselect("Skills or strengths", [
            "Problem solving", "Communication", "Teamwork", "Creativity",
            "Programming", "Consistency", "Leadership", "Asking for help"
        ])
        milestone = st.text_area("Recent achievement or progress", placeholder="Example: Completed two assignments on time.")
        support = st.text_area("Support that helps me", placeholder="Example: Weekly mentor check-in and a study timetable.")
        passport_submit = st.form_submit_button("Save Success Passport", use_container_width=True)
    if passport_submit:
        if not ref.strip():
            st.error("Please enter a student reference.")
        else:
            st.session_state.passport_data[ref.strip()] = {
                "name": display_name.strip() or "Student",
                "goal": goal.strip() or "Goal not added yet",
                "skills": skills,
                "milestone": milestone.strip() or "No milestone added yet",
                "support": support.strip() or "No support preference added yet",
                "updated": date.today().isoformat()
            }
            st.success("Success Passport saved for this session.")
    ref_to_show = st.text_input("View passport for student reference", value=st.session_state.latest_student_ref, key="passport_view_ref")
    passport = st.session_state.passport_data.get(ref_to_show.strip())
    if passport:
        st.markdown(f"## 🌟 {passport['name']}'s Success Passport")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="support-card"><b>Current goal</b><p>{passport["goal"]}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="support-card"><b>Recent progress</b><p>{passport["milestone"]}</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="support-card"><b>Strengths</b><p>{", ".join(passport["skills"]) if passport["skills"] else "Add strengths to your passport."}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="support-card"><b>Helpful support</b><p>{passport["support"]}</p></div>', unsafe_allow_html=True)
        st.caption(f"Last updated: {passport['updated']}")
    else:
        st.info("Save a passport above to view the student's success snapshot. This version keeps passport entries in the current app session.")


# --------------------------------------------------
# ACADEMIC STRENGTH FINDER (NEW)
# --------------------------------------------------
elif page == "Academic Strength Finder":
    st.markdown("""
    <div class="hero"><h1>💡 Academic Strength Finder</h1>
    <p>Spot areas of confidence and areas where extra support may help.</p></div>
    """, unsafe_allow_html=True)
    st.write("Enter scores from 0–100. These scores are used for this strengths summary and do not retrain or alter the ML model.")
    with st.form("strength_form"):
        ref = st.text_input("Student reference", value=st.session_state.latest_student_ref, placeholder="Example: STU-001")
        subjects = ["Programming", "Mathematics", "Computer Networks", "Database Systems", "Communication", "Problem Solving"]
        scores = {}
        cols = st.columns(2)
        for i, subject in enumerate(subjects):
            with cols[i % 2]:
                scores[subject] = st.number_input(f"{subject} score (0–100)", min_value=0, max_value=100, value=70, key=f"strength_{subject}")
        strength_submit = st.form_submit_button("Find My Strengths", use_container_width=True)
    if strength_submit:
        if not ref.strip():
            st.error("Please enter a student reference.")
        else:
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            st.session_state.strength_data[ref.strip()] = {"scores": scores.copy(), "ordered": ordered}
            st.success("Academic strengths summary created.")
    view_ref = st.text_input("View strengths for student reference", value=st.session_state.latest_student_ref, key="strength_view_ref")
    data = st.session_state.strength_data.get(view_ref.strip())
    if data:
        ordered = data["ordered"]
        st.markdown("### Your academic snapshot")
        st.metric("Average score", f"{sum(data['scores'].values()) / len(data['scores']):.1f}/100")
        st.markdown("#### Strongest areas")
        for subject, score in ordered[:2]:
            st.write(f"**{subject} — {score}/100**")
            st.progress(score / 100)
        st.markdown("#### Areas to build")
        for subject, score in sorted(ordered, key=lambda item: item[1])[:2]:
            st.write(f"**{subject} — {score}/100**")
            st.progress(score / 100)
        st.write("Suggested next step: choose one area to build this week and set a small, measurable practice goal.")
    else:
        st.info("Submit scores above to generate the academic strengths summary.")


# --------------------------------------------------
# ROADMAP (NEW)
# --------------------------------------------------
elif page == "Roadmap":
    st.markdown("""
    <div class="hero"><h1>🗺️ Your Roadmap</h1>
    <p>Turn one goal into small steps you can actually complete.</p></div>
    """, unsafe_allow_html=True)
    with st.form("roadmap_form"):
        ref = st.text_input("Student reference", value=st.session_state.latest_student_ref, placeholder="Example: STU-001")
        goal = st.selectbox("What would you like to work on?", [
            "Improve semester grades", "Pass a difficult subject", "Build a consistent study routine",
            "Improve attendance", "Prepare for placements", "Build confidence in programming"
        ])
        timeframe = st.selectbox("Plan duration", ["1 week", "2 weeks", "1 month", "6 weeks"])
        hours = st.slider("Realistic study hours per week", min_value=1, max_value=30, value=6)
        obstacle = st.selectbox("What may get in the way?", [
            "Time management", "Difficult concepts", "Low motivation", "Distractions",
            "Financial or personal pressure", "Not sure yet"
        ])
        roadmap_submit = st.form_submit_button("Build My Roadmap", use_container_width=True)
    if roadmap_submit:
        if not ref.strip():
            st.error("Please enter a student reference.")
        else:
            st.session_state.roadmap_data[ref.strip()] = {
                "goal": goal, "timeframe": timeframe, "hours": hours, "obstacle": obstacle
            }
            st.success("Your roadmap is ready.")
    roadmap_ref = st.text_input("View roadmap for student reference", value=st.session_state.latest_student_ref, key="roadmap_view_ref")
    roadmap = st.session_state.roadmap_data.get(roadmap_ref.strip())
    if roadmap:
        st.markdown(f"## 🎯 Goal: {roadmap['goal']}")
        st.write(f"**Duration:** {roadmap['timeframe']}  |  **Study time:** {roadmap['hours']} hours/week")
        steps = [
            ("Step 1 — Set up", "List the topics or tasks to complete. Pick the two most important ones."),
            ("Step 2 — Practise", f"Schedule short practice sessions across the week, using about {roadmap['hours']} hours in total."),
            ("Step 3 — Check progress", "At the end of the week, attempt a quiz, solve practice questions, or explain the topic aloud."),
            ("Step 4 — Ask for support", f"If {roadmap['obstacle'].lower()} slows you down, contact a mentor, teacher, or study partner early."),
            ("Step 5 — Reflect and reset", "Mark what worked, celebrate one improvement, and adjust next week's plan.")
        ]
        for title, description in steps:
            st.markdown(f'<div class="support-card"><b>{title}</b><p>{description}</p></div>', unsafe_allow_html=True)
        st.checkbox("I reviewed my roadmap and will take the first step", key=f"roadmap_done_{roadmap_ref}")
    else:
        st.info("Complete the form above to create a roadmap for your selected goal.")


st.sidebar.caption("EduHope is a support tool. Predictions and summaries are not guarantees.")
