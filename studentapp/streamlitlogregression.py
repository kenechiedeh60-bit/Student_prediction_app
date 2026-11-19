import joblib
import pandas as pd
import streamlit as st

model = joblib.load("pass_model.pkl")

st.title("Will the students pass?")
st.write("Move the slider, then click Predict.")

hours_studied = st.slider("Hours Studied", 0.0, 10.0, 5.0, 0.5)
attendance_pct = st.slider("Attendance (%)", 40, 100, 80)
previous_score = st.slider("Previous Score", 0, 100, 60)

if st.button("Predict"):
    new_student = pd.DataFrame(
        {
            "hours_studied":[hours_studied], 
            "attendance_pct": [attendance_pct],
            "previous_score": [previous_score],
        }
    )

    prediction = model.predict(new_student)[0]
    chance = model.predict_proba(new_student)[0][1]

    if prediction == 1:
        st.success(f"This student will probably PASS. Chance of passing: {chance:.0%}")
    else:
        st.error(f"This student will probably FAIL. Chance of passing: {chance:.0%}")
