import os
import streamlit as st
import pandas as pd
import joblib
import gdown

st.set_page_config(page_title="EduHope", page_icon="🎓")

MODEL_PATH = "eduhope_model.joblib"
DRIVE_FILE_ID = "1V9ptd87Nlp5ojM9rVC4prcBvMWMT0L2a"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        result = gdown.download(url, MODEL_PATH, quiet=False)
        if result is None:
            raise RuntimeError("Model download failed. Check Google Drive sharing.")

    return joblib.load(MODEL_PATH)

try:
    bundle = load_model()
    model = bundle["model"]
    feature_names = bundle["feature_names"]
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

st.title("🎓 EduHope")
st.subheader("No Student Left Behind")
st.write(
    "Predict student outcomes using machine learning "
    "and identify where additional support may help."
)

st.info("Enter the numeric values for the dataset features below.")

with st.form("student_form"):
    values = {}
    columns = st.columns(3)

    for i, feature in enumerate(feature_names):
        with columns[i % 3]:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=0.0
            )

    submitted = st.form_submit_button("Predict Outcome")

if submitted:
    input_df = pd.DataFrame(
        [[values[f] for f in feature_names]],
        columns=feature_names
    )

    prediction = model.predict(input_df)[0]

    st.subheader("Predicted Student Outcome")

    if prediction == "Graduate":
        st.success("Predicted outcome: Graduate 🎓")
    elif prediction == "Dropout":
        st.warning("Predicted outcome: Dropout")
        st.write("Consider whether additional academic or personal support may help.")
    else:
        st.info("Predicted outcome: Enrolled 📚")

    st.caption(
        "This is a prediction, not a certainty or a judgment about the student."
    )
