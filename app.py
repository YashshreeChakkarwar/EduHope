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
# LOAD YOUR EXISTING RANDOM FOREST MODEL
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
        "Ensure eduhope_model.joblib is present in the app environment "
        "and contains 'model' and 'feature_names'."
    )
    st.stop()

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
# PREDICTION FUNCTION
# Uses all existing features in their original order.
# --------------------------------------------------
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
    needs = int((plans["status"] == "Needs follow-up").sum()) if total else 0
    active = int((plans["status"] == "Support in progress").sum()) if total else 0
    completed = int((plans["status"] == "Follow-up completed").sum()) if total else 0

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
# RISK REPORT
# --------------------------------------------------
elif page == "Risk Report":
    st.markdown("""
    <div class="hero">
        <h1>Student Risk Report</h1>
        <p>Explore a model estimate and consider whether support may help.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "All original model features are retained to preserve compatibility. "
        "Use the same feature values and units as the training dataset."
    )

    with st.form("student_form"):
        student_ref = st.text_input(
            "Student reference",
            placeholder="Example: STU-001 (use a fictional ID for demos)",
        )
        values = {}

        for feature in feature_names:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=0.0,
                key=f"feature_{feature}",
            )

        submitted = st.form_submit_button(
            "Predict Outcome",
            use_container_width=True,
        )

    if submitted:
        try:
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
        elif prediction == "Dropout":
            st.warning("Predicted outcome: Dropout")
            st.write(
                "Consider whether a supportive academic check-in may help. "
                "This is not a certainty."
            )
        else:
            st.info(f"Predicted outcome: {prediction} 📚")

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
            "Model estimates are not guarantees and do not explain why "
            "a student may need support."
        )

        st.markdown("### What happens next?")
        st.write(
            "If support may be useful, create a plan in Comeback Pathway. "
            "Students can receive support regardless of the predicted outcome."
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
                "Support plan created! You can track its progress "
                "on the Support Tracker page."
            )

    st.markdown("### Our promise")
    st.markdown("""
    <div class="support-card">
    <b>Notice → Listen → Support → Follow up</b><br><br>
    EduHope is designed to encourage constructive support. A model
    prediction should never be used on its own to punish, exclude,
    or deny a student an opportunity.
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
        total = len(plans)
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

                current_status = plan["status"]
                selected_status = st.selectbox(
                    "Update status",
                    status_options,
                    index=status_options.index(current_status),
                    key=f"status_{plan_id}",
                )

                if st.button(
                    "Save status",
                    key=f"save_{plan_id}",
                ):
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
