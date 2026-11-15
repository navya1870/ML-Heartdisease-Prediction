import streamlit as st
import pandas as pd
import joblib

model = joblib.load("LogisticRegression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

print("Expected columns:", expected_columns)
print("Model classes:", model.classes_)

st.title("Heart stroke prediction by navya")
st.markdown("Provide the following details")

age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
resting_bp = st.number_input("Resting Blood Pressure(mm Hg)",80,300,120)
cholesterol = st.number_input("Cholesterol(mg/dL)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120mg/dL",[0,1])
resting_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr = st.slider("Max Heart Rate",60,250,120)
exercise_angina = st.selectbox("Exercise-Induced Angina",["Y","N"])
oldpeak = st.slider("OldPeak(ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])

if st.button("Predict"):
    raw_input = {
    "Age": age,
    "Sex_F": 1 if sex == "F" else 0,
    "Sex_M": 1 if sex == "M" else 0,
    "ChestPainType_ASY": 1 if chest_pain == "ASY" else 0,
    "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG_LVH": 1 if resting_ecg == "LVH" else 0,
    "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
    "MaxHR": max_hr,
    "ExerciseAngina_N": 1 if exercise_angina == "N" else 0,
    "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
    "Oldpeak": oldpeak,
    "ST_Slope_Down": 1 if st_slope == "Down" else 0,
    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }
    input_df = pd.DataFrame([raw_input])
    input_df = input_df[expected_columns]
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("High risk of Heart disease")
    else:
        st.success("Low risk of Heart disease")



#ye jo hmara dataframe the sbse pehle csv me se uski sari column lekr to raw input bna diya
#but baad me jo ye label encoding kri thi uski vjh se bhut sari columns bn gyi thi to idhr kya kr rhe 
#user se input lege vhi jo uper itne sare bnaya h slider vgrh
#fir jo bhi chij select krli to baakio me 0 ho jaega jo select kiya usme 1  ye ek hi column ki 
#baat hori h mtlb ek type ki chij ki"""


#scaled_input = scaler.transform(input_df)

#ye scaling krte agr knn ya jisme scalingki jrurat pdti h vesa model use krte to
#we are using logistic regression so no scaling required
