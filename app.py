import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EduHope", page_icon="🎓")

bundle = joblib.load("eduhope_model.joblib")
model = bundle["model"]
feature_names = bundle["feature_names"]

st.title("🎓 EduHope")
st.subheader("No Student Left Behind")
st.write("Predict student outcomes and identify where additional support may help.")

st.info("Enter the numeric values for the student's dataset features below.")

with st.form("student_form"):
    values = {}

    for feature in feature_names:
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

    st.subheader("Prediction")

    if prediction == "Graduate":
        st.success("Predicted outcome: Graduate 🎓")
    elif prediction == "Dropout":
        st.warning("Predicted outcome: Dropout")
        st.write("Consider whether additional academic or personal support may help.")
    else:
        st.info("Predicted outcome: Enrolled 📚")

    st.caption("This is a prediction, not a certainty or a judgment about the student.")