
import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import date

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

# Reference values calculated from the original CSV.
# These fill features that are not shown in the short form.
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
    return (
        str(name)
        .replace("\ufeff", "")
        .replace("ï»¿", "")
        .strip()
        .lower()
    )


# --------------------------------------------------
# PURPLE DESIGN
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F8F6FF;
    color: #281B42;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #241443, #56308F);
}
[data-testid="stSidebar"] * {
    color: white;
}
.hero {
    padding: 26px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(120deg, #452477, #8757D9);
    margin-bottom: 24px;
}
.hero h1 { color: white; font-size: 32px; }
.hero p { color: #F0E7FF; }
.metric-card {
    background: white;
    border: 1px solid #E8DFFA;
    border-radius: 16px;
    padding: 18px;
    min-height: 110px;
}
.metric-label { color: #766B8B; font-size: 13px; }
.metric-value {
    color: #6137A5;
    font-size: 26px;
    font-weight: 800;
    margin-top: 8px;
}
.support-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #E8DFFA;
    margin-bottom: 12px;
}
div.stButton > button, div.stFormSubmitButton > button {
    background: #7044BF;
    color: white;
    border: 0;
    border-radius: 11px;
    min-height: 42px;
    font-weight: 600;
}
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background: #543091;
    color: white;
    border: 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EXISTING MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], list(bundle["feature_names"])


try:
    model, feature_names = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.info(
        "Check that eduhope_model.joblib is accessible and contains "
        "'model' and 'feature_names'. If your deployment downloads it "
        "from Google Drive, restore that download logic here."
    )
    st.stop()


# --------------------------------------------------
# BUILD SHORT-FORM INPUTS
# --------------------------------------------------
def build_student_features(
    age, fees, scholarship,
    sem1_enrolled, sem1_approved, sem1_grade,
    sem2_enrolled, sem2_approved, sem2_grade
):
    reference_lookup = {
        normalize_feature(name): value
        for name, value in REFERENCE_VALUES.items()
    }

    # Fail clearly if a model feature has no reference value.
    missing = [
        name for name in feature_names
        if normalize_feature(name) not in reference_lookup
    ]

    if missing:
        raise ValueError(
            "Reference values are missing for model features: "
            + ", ".join(missing)
        )

    # Preserve the exact feature names and order used by the model.
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

    entered_lookup = {
        normalize_feature(name): value
        for name, value in entered.items()
    }

    for feature in feature_names:
        normalized = normalize_feature(feature)
        if normalized in entered_lookup:
            values[feature] = entered_lookup[normalized]

    return values


def predict_student(values):
    input_df = pd.DataFrame(
        [[values[feature] for feature in feature_names]],
        columns=feature_names,
    )

    prediction = model.predict(input_df)[0]

    probabilities = (
        model.predict_proba(input_df)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    return prediction, probabilities


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
            INSERT INTO support_plans (
                student_ref, concern, action_plan, mentor,
                follow_up_date, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            student_ref.strip(),
            concern,
            action,
            mentor.strip(),
            follow_up.isoformat(),
            "Needs follow-up",
            date.today().isoformat(),
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
    allowed = [
        "Needs follow-up",
        "Support in progress",
        "Follow-up completed",
    ]

    if status not in allowed:
        raise ValueError("Invalid status")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE support_plans SET status = ? WHERE id = ?",
            (status, int(plan_id)),
        )


init_db()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "latest_probabilities" not in st.session_state:
    st.session_state.latest_probabilities = None

if "latest_student_ref" not in st.session_state:
    st.session_state.latest_student_ref = ""


# --------------------------------------------------
# SIDEBAR
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
        ],
    )

    st.divider()
    st.caption("Every prediction is an opportunity to offer support.")


# --------------------------------------------------
# DASHBOARD
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

    needs = int(
        (plans["status"] == "Needs follow-up").sum()
    ) if total else 0

    active = int(
        (plans["status"] == "Support in progress").sum()
    ) if total else 0

    completed = int(
        (plans["status"] == "Follow-up completed").sum()
    ) if total else 0

    cols = st.columns(4)

    cards = [
        ("Support plans", total),
        ("Needs follow-up", needs),
        ("Support in progress", active),
        ("Follow-up completed", completed),
    ]

    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("## How EduHope helps")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="support-card">
        <h3>🔎 Notice</h3>
        Explore predicted student outcomes with your existing ML model.
        </div>
        <div class="support-card">
        <h3>💜 Support</h3>
        Record a practical action plan with a mentor and follow-up date.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="support-card">
        <h3>🌱 Follow through</h3>
        Keep track of plans that still need attention.
        </div>
        <div class="support-card">
        <h3>🤝 Remember every student</h3>
        Track support activity instead of leaving a prediction as the final result.
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Dashboard totals reflect support plans entered into this app, "
        "not the total number of students at a college."
    )


# --------------------------------------------------
# RISK REPORT: ONLY NINE MODEL INPUTS
# --------------------------------------------------
elif page == "Risk Report":
    st.markdown("""
    <div class="hero">
        <h1>Student Risk Report 💜</h1>
        <p>A few student details can help us explore where support may be useful.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Only nine student details are requested. The remaining model "
        "features use reference values from the original dataset. "
        "The result is an estimate, not a certainty."
    )

    with st.form("student_form"):
        student_ref = st.text_input(
            "Student reference",
            placeholder="Example: STU-001 (use a fictional ID for demos)",
        )

        st.markdown("### Student details")

        age = st.number_input(
            "Age at enrollment",
            min_value=15,
            max_value=100,
            value=20,
            step=1,
        )

        fees_option = st.selectbox(
            "Are tuition fees up to date?",
            ["Yes", "No"],
        )
        fees = 1 if fees_option == "Yes" else 0

        scholarship_option = st.selectbox(
            "Does the student hold a scholarship?",
            ["No", "Yes"],
        )
        scholarship = 1 if scholarship_option == "Yes" else 0

        st.markdown("### First-semester performance")

        sem1_enrolled = st.number_input(
            "Units enrolled in semester 1",
            min_value=0,
            max_value=30,
            value=6,
            step=1,
        )

        sem1_approved = st.number_input(
            "Units approved in semester 1",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
        )

        sem1_grade = st.number_input(
            "Average grade in semester 1 (0–20)",
            min_value=0.0,
            max_value=20.0,
            value=12.3,
            step=0.1,
        )

        st.markdown("### Second-semester performance")

        sem2_enrolled = st.number_input(
            "Units enrolled in semester 2",
            min_value=0,
            max_value=30,
            value=6,
            step=1,
        )

        sem2_approved = st.number_input(
            "Units approved in semester 2",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
        )

        sem2_grade = st.number_input(
            "Average grade in semester 2 (0–20)",
            min_value=0.0,
            max_value=20.0,
            value=12.2,
            step=0.1,
        )

        submitted = st.form_submit_button(
            "Predict Student Outcome",
            use_container_width=True,
        )

    if submitted:
        if not student_ref.strip():
            st.error("Please enter a student reference.")

        elif sem1_approved > sem1_enrolled:
            st.error(
                "Semester 1 approved units cannot exceed enrolled units."
            )

        elif sem2_approved > sem2_enrolled:
            st.error(
                "Semester 2 approved units cannot exceed enrolled units."
            )

        else:
            try:
                values = build_student_features(
                    age,
                    fees,
                    scholarship,
                    sem1_enrolled,
                    sem1_approved,
                    sem1_grade,
                    sem2_enrolled,
                    sem2_approved,
                    sem2_grade,
                )

                prediction, probabilities = predict_student(values)

                st.session_state.latest_prediction = str(prediction)
                st.session_state.latest_probabilities = probabilities
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
            st.write(
                "Continue encouraging the student's progress and "
                "offer support whenever needed."
            )

        elif prediction == "Dropout":
            st.warning("Predicted outcome: Dropout")
            st.write(
                "Consider a supportive conversation to understand "
                "the student's academic, financial, or personal challenges."
            )

        else:
            st.info(f"Predicted outcome: {prediction} 📚")
            st.write(
                "Consider regular check-ins to help the student "
                "continue progressing."
            )

        if probabilities is not None:
            st.markdown("### Estimated class probabilities")

            for label, probability in sorted(
                zip(model.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            ):
                st.write(f"**{label}: {probability * 100:.1f}%**")
                st.progress(float(probability))

        st.caption(
            "Unentered features use fixed dataset reference values. "
            "The prediction may be less accurate when a student's "
            "actual circumstances differ from those values. Do not "
            "use this estimate alone to make decisions about a student."
        )

        st.markdown("### What happens next?")
        st.write(
            "If support may be useful, visit Comeback Pathway to record "
            "an action plan and follow-up date."
        )


# --------------------------------------------------
# COMEBACK PATHWAY
# --------------------------------------------------
elif page == "Comeback Pathway":
    st.markdown("""
    <div class="hero">
        <h1>💜 Comeback Pathway</h1>
        <p>A prediction should never be the end of a student's story.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Record an agreed action, the person coordinating support, "
        "and a date to check in."
    )

    default_ref = st.session_state.latest_student_ref

    with st.form("support_form"):
        student_ref = st.text_input(
            "Student reference",
            value=default_ref,
            placeholder="Example: STU-001",
        )

        concern = st.selectbox(
            "Area where support may help",
            [
                "Academic progress",
                "Difficulty with coursework",
                "Financial support information",
                "Attendance or engagement check-in",
                "Personal wellbeing check-in",
                "Other",
            ],
        )

        action = st.text_area(
            "Agreed support action",
            placeholder=(
                "Example: Arrange a meeting with the academic mentor "
                "and agree on a weekly study plan."
            ),
        )

        mentor = st.text_input(
            "Mentor or staff reference",
            placeholder="Example: MENTOR-01",
        )

        follow_up = st.date_input(
            "Follow-up date",
            value=date.today(),
            min_value=date.today(),
        )

        save_clicked = st.form_submit_button(
            "Create Support Plan",
            use_container_width=True,
        )

    if save_clicked:
        if not student_ref.strip() or not action.strip() or not mentor.strip():
            st.error("Please enter the student reference, action, and mentor.")
        else:
            save_plan(student_ref, concern, action, mentor, follow_up)
            st.success(
                "Support plan created! Track its progress on the Support Tracker page."
            )

    st.markdown("### Our promise")

    st.markdown("""
    <div class="support-card">
    <b>Notice → Listen → Support → Follow up</b><br><br>
    EduHope encourages constructive support. A model prediction should
    never be used on its own to punish, exclude, or deny a student an opportunity.
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# SUPPORT TRACKER
# --------------------------------------------------
elif page == "Support Tracker":
    st.markdown("""
    <div class="hero">
        <h1>🤝 No Student Left Behind Tracker</h1>
        <p>Every support plan deserves a follow-up.</p>
    </div>
    """, unsafe_allow_html=True)

    plans = get_plans()

    if plans.empty:
        st.info(
            "No support plans have been created yet. "
            "Visit Comeback Pathway to create your first plan."
        )

    else:
        needs = int((plans["status"] == "Needs follow-up").sum())
        active = int((plans["status"] == "Support in progress").sum())
        completed = int((plans["status"] == "Follow-up completed").sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Needs follow-up", needs)
        c2.metric("Support in progress", active)
        c3.metric("Follow-up completed", completed)

        st.markdown("## Support plans")

        for _, plan in plans.iterrows():
            plan_id = int(plan["id"])
            follow_date = str(plan["follow_up_date"])

            is_overdue = (
                follow_date < date.today().isoformat()
                and plan["status"] != "Follow-up completed"
            )

            with st.expander(
                f"Plan #{plan_id} · {plan['student_ref']} · {plan['status']}"
            ):
                st.write(f"**Concern:** {plan['concern']}")
                st.write(f"**Agreed action:** {plan['action_plan']}")
                st.write(f"**Mentor:** {plan['mentor']}")
                st.write(f"**Follow-up date:** {follow_date}")

                if is_overdue:
                    st.warning("Follow-up date has passed. Please review this plan.")

                status_options = [
                    "Needs follow-up",
                    "Support in progress",
                    "Follow-up completed",
                ]

                selected_status = st.selectbox(
                    "Update status",
                    status_options,
                    index=status_options.index(plan["status"]),
                    key=f"status_{plan_id}",
                )

                if st.button("Save status", key=f"save_{plan_id}"):
                    update_status(plan_id, selected_status)
                    st.success("Status updated.")
                    st.rerun()

        st.markdown("## All support records")
        st.dataframe(plans, use_container_width=True, hide_index=True)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "EduHope · No Student Left Behind · "
    "Predictions support human judgment; they do not replace it."
)
