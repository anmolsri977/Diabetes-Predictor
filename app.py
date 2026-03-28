import streamlit as st
import numpy as np
import pickle
import pandas as pd

# load model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.set_page_config(page_title="Diabetes Predictor", layout="centered")
with st.sidebar:
    st.title("🩺 Diabetes Predictor")
    st.markdown("### About")
    st.write("This app predicts the risk of diabetes based on medical inputs.")
    
    st.markdown("### Instructions")
    st.write("Fill all the input fields and click Predict.")

st.title("🩺 Diabetes Risk Predictor")

st.markdown("### Enter patient details below")


# inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    bp = st.slider("Blood Pressure", 0, 150, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 50.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 1, 120, 25)



if st.button("Predict"):
   # input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": bp,
    "SkinThickness": skin,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
    }])

    st.markdown("### 📋 Input Summary")

    st.write(input_data)

    
    prediction = model.predict(input_data)
    
    st.markdown("## 🔍 Prediction Result")
    st.markdown("---")

    probability = model.predict_proba(input_data)[0][1]
    st.write(f"### Risk Score: {probability*100:.2f}%")
    st.progress(int(probability * 100))

    if probability > 0.7:
        st.error(f"🔴 Very High Risk ({probability*100:.2f}%)")
        st.write("⚠ Immediate medical consultation is strongly advised.")

    elif probability > 0.4:
        st.warning(f"🟡 Moderate Risk ({probability*100:.2f}%)")
        st.write("⚠ You should consider lifestyle changes and consult a doctor.")

    else:
        st.success(f"🟢 Low Risk ({probability*100:.2f}%)")
        st.write("✅ Risk is low, but maintain a healthy lifestyle.")

import matplotlib.pyplot as plt

st.markdown("### 📊 Feature Importance")

features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DPF","Age"]
importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance)
st.pyplot(fig)



st.info("This is a machine learning prediction tool. Please consult a doctor for medical advice.")

st.markdown("---")
st.caption("Built by Anmol Kumar Srivastava | Machine Learning Project")
