import streamlit as st
import numpy as np
import pickle
import pandas as pd

# load model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Risk Predictor")
st.markdown("### Enter patient details below")


# inputs
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 50.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)



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
    

    probability = model.predict_proba(input_data)[0][1]
    st.write(f"### Risk Score: {probability*100:.2f}%")
    st.progress(int(probability * 100))

    if probability > 0.7:
        st.markdown("🔴 **Very High Risk Zone**")
    elif probability > 0.4:
        st.markdown("🟡 **Moderate Risk Zone**")
    else:
        st.markdown("🟢 **Low Risk Zone**")

import matplotlib.pyplot as plt

st.markdown("### 📊 Feature Importance")

features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DPF","Age"]
importance = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(features, importance)
st.pyplot(fig)



st.info("This is a machine learning prediction tool. Please consult a doctor for medical advice.")
